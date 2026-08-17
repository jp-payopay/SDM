from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

DataMode = Literal["presence_only", "presence_absence"]
BackgroundMethod = Literal["random", "ratio", "disk", "sre"]
SplitMethod = Literal["random", "kfold", "spatial_block"]
BlockShape = Literal["square", "hexagon"]
EnsembleMethod = Literal["mean", "weighted_auc", "weighted_tss"]

ALGORITHMS = ("lr", "gam", "rf", "gbm", "xgb", "svm", "mlp", "maxent", "enfa")


@dataclass
class OccurrenceConfig:
    path: str = ""
    layer_name: str = ""
    x_field: str = "x"
    y_field: str = "y"
    presence_field: str = ""
    crs: str = "EPSG:4326"


@dataclass
class RasterConfig:
    paths: list[str] = field(default_factory=list)
    projection_paths: list[str] = field(default_factory=list)


@dataclass
class CleaningConfig:
    auto_clean: bool = True
    thin_to_raster_resolution: bool = True


@dataclass
class BackgroundConfig:
    # How many points to draw, for every method except "ratio".
    count: int = 10_000
    method: BackgroundMethod = "random"
    # "ratio": points are placed at random, exactly as the "random" method
    # does, but the count scales with the data instead of being fixed. At 4
    # per presence, 50 records give 200 pseudo-absences and 300 give 1,200,
    # which keeps the balance between the two classes steady no matter how
    # many records a species happens to have.
    ratio: float = 4.0
    # "disk": keep only locations whose nearest presence is at least
    # min_distance and at most max_distance away. Both are always real-world
    # meters, regardless of the predictor rasters' CRS, and are converted to
    # the raster's native units (latitude-corrected for geographic CRSs) in
    # stages.collect_labeled_points_and_extract. max_distance = 0 means no
    # upper limit.
    min_distance: float = 0.0
    max_distance: float = 50_000.0
    # "sre": fraction trimmed off each end of the presences' values per
    # predictor before the environmental envelope is drawn, so one outlying
    # record cannot stretch it over the whole study area.
    sre_quantile: float = 0.025

    def resolve_count(self, n_presence: int) -> int:
        """How many background points to draw for this many presences.

        Resolved here rather than at the point of use so the preview page and
        the pipeline can never disagree about it.

        A fractional multiplier rounds half up, not to even: 1.5 per presence
        over 35 records gives 53, which is the number a user doing the
        multiplication in their head will expect. Both values are known to be
        non-negative (validate() rejects a ratio of zero or less), so adding a
        half and truncating is enough.
        """
        if self.method == "ratio":
            return max(1, int(self.ratio * n_presence + 0.5))
        return int(self.count)


@dataclass
class VIFConfig:
    cutoff: float = 10.0
    # False skips stepwise elimination entirely and keeps every predictor —
    # for users who want to handle multicollinearity themselves (or not at
    # all), rather than an implicit "raise the cutoff very high" workaround.
    enabled: bool = True


@dataclass
class SplitConfig:
    method: SplitMethod = "spatial_block"
    k: int = 5
    test_size: float = 0.25
    auto_block_size: bool = True
    # Always real-world meters (ignored when auto_block_size is True).
    # Converted to the raster's native CRS units in stages.make_folds.
    block_size: float = 0.0
    # Tessellation for spatial_block CV. Both interpret block_size as the
    # same ground area per block — see core/split/spatial_block.py.
    block_shape: BlockShape = "square"


@dataclass
class ModelingConfig:
    algorithms: list[str] = field(default_factory=lambda: list(ALGORITHMS))
    replicates: int = 5
    # Per-algorithm hyperparameter overrides the user set by hand in the
    # "Show / edit model configuration" dialog, keyed by algorithm code
    # (e.g. {"rf": {"n_estimators": 800}}). Only values the user actually
    # changed from the defaults are stored here; everything else falls back
    # to each model's built-in default at build time.
    hyperparameters: dict[str, dict] = field(default_factory=dict)


@dataclass
class EnsembleConfig:
    method: EnsembleMethod = "weighted_tss"


@dataclass
class OutputConfig:
    directory: str = ""
    write_html_report: bool = True
    save_models: bool = True


def _background_from_dict(sub: dict | None) -> BackgroundConfig:
    """Build a BackgroundConfig, accepting run_config.json files written
    before the buffered background became a min/max distance disk.

    Back then the method was called "buffered" and had a single
    `buffer_distance`, which was the outer radius with no inner one — exactly
    what max_distance means now, so an old config reruns unchanged.
    """
    if not sub:
        return BackgroundConfig()
    data = dict(sub)
    legacy_distance = data.pop("buffer_distance", None)
    if legacy_distance is not None and "max_distance" not in data:
        data["max_distance"] = float(legacy_distance)
    if data.get("method") == "buffered":
        data["method"] = "disk"
    # Scaling the count with the presence count was briefly a separate
    # `count_mode` flag alongside the method; it is now a method of its own.
    if data.pop("count_mode", None) == "ratio":
        data["method"] = "ratio"
    return BackgroundConfig(**data)


@dataclass
class SDMConfig:
    data_mode: DataMode = "presence_only"
    occurrence: OccurrenceConfig = field(default_factory=OccurrenceConfig)
    rasters: RasterConfig = field(default_factory=RasterConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    vif: VIFConfig = field(default_factory=VIFConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    random_seed: int = 42
    version: str = "1.0.1"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict) -> "SDMConfig":
        def build(dc_cls, sub):
            return dc_cls(**sub) if sub else dc_cls()

        return cls(
            data_mode=data.get("data_mode", "presence_only"),
            occurrence=build(OccurrenceConfig, data.get("occurrence")),
            rasters=build(RasterConfig, data.get("rasters")),
            cleaning=build(CleaningConfig, data.get("cleaning")),
            background=_background_from_dict(data.get("background")),
            vif=build(VIFConfig, data.get("vif")),
            split=build(SplitConfig, data.get("split")),
            modeling=build(ModelingConfig, data.get("modeling")),
            ensemble=build(EnsembleConfig, data.get("ensemble")),
            output=build(OutputConfig, data.get("output")),
            random_seed=data.get("random_seed", 42),
            version=data.get("version", "1.0.1"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SDMConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.occurrence.path:
            errors.append("Occurrence file path is required.")
        if not self.rasters.paths:
            errors.append("At least one predictor raster is required.")
        if self.data_mode == "presence_absence" and not self.occurrence.presence_field:
            errors.append("Presence field is required for presence/absence mode.")
        if self.background.method == "ratio":
            if self.background.ratio <= 0:
                errors.append("Pseudo-absences per presence must be greater than zero.")
        elif self.background.count < 100:
            errors.append("Background count should be at least 100.")
        if self.background.method == "disk":
            if self.background.min_distance < 0 or self.background.max_distance < 0:
                errors.append("Disk background distances cannot be negative.")
            elif (
                self.background.max_distance > 0
                and self.background.min_distance >= self.background.max_distance
            ):
                errors.append(
                    "Minimum distance must be smaller than maximum distance for the "
                    "disk background (set maximum to 0 for no upper limit)."
                )
        if self.background.method == "sre" and not (0.0 <= self.background.sre_quantile < 0.5):
            errors.append("SRE quantile must be at least 0 and below 0.5.")
        if self.vif.cutoff <= 1.0:
            errors.append("VIF cutoff must be greater than 1.")
        if self.split.k < 2:
            errors.append("k must be at least 2 for k-fold or spatial-block CV.")
        if self.split.method == "random" and not (0.0 < self.split.test_size < 1.0):
            errors.append("test_size must be between 0 and 1 (exclusive) for random hold-out.")
        if (
            self.split.method == "spatial_block"
            and not self.split.auto_block_size
            and self.split.block_size <= 0
        ):
            errors.append(
                "Block size must be positive when auto block size is disabled "
                "(a non-positive value would otherwise silently fall back to "
                "auto-sizing instead of the value you intended)."
            )
        if self.split.block_shape not in ("square", "hexagon"):
            errors.append(f"Unknown block_shape: {self.split.block_shape!r}")
        if self.modeling.replicates < 1:
            errors.append("Replicates must be >= 1.")
        unknown = set(self.modeling.algorithms) - set(ALGORITHMS)
        if unknown:
            errors.append(f"Unknown algorithms: {sorted(unknown)}")
        if not self.modeling.algorithms:
            errors.append("Select at least one algorithm.")
        if self.data_mode == "presence_absence" and "enfa" in self.modeling.algorithms:
            errors.append(
                "ENFA is a presence-only method (it models presences against the "
                "full sample as the 'available environment') and isn't meaningful "
                "in presence/absence mode."
            )
        if not self.output.directory:
            errors.append("Output directory is required.")
        return errors
