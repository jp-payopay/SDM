from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cleaning.coord_clean import CleaningReport
from .cleaning.thinning import ThinningReport
from .io.occurrences import OccurrenceData
from .io.rasters import RasterStack
from .predictors.vif import VIFReport


@dataclass
class PipelineSession:
    """Accumulates artifacts as wizard pages (or a plain Pipeline.run() call)
    work through the pipeline stages, so later stages can reuse earlier work
    instead of recomputing it.
    """

    stack: RasterStack | None = None
    occ_raw: OccurrenceData | None = None
    occ: OccurrenceData | None = None
    cleaning_report: CleaningReport | None = None
    thinning_report: ThinningReport | None = None
    px: np.ndarray | None = None
    py: np.ndarray | None = None
    presence_flag: np.ndarray | None = None
    X_full: np.ndarray | None = None
    feature_names: list[str] | None = None
    X_kept: np.ndarray | None = None
    kept_names: list[str] | None = None
    kept_idx: list[int] | None = None
    vif_report: VIFReport | None = None
    proj_stack: RasterStack | None = None
    stage_hashes: dict[str, str] = field(default_factory=dict)

    def invalidate(self, *fields: str) -> None:
        """Reset the named session fields to unset. Does not touch
        stage_hashes, which is keyed by stage name (e.g. "occurrence"), not
        by these session field names — see SDMWizard.invalidate_from, which
        clears the corresponding stage_hashes entries itself."""
        defaults = PipelineSession()
        for name in fields:
            setattr(self, name, getattr(defaults, name))
