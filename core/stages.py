from __future__ import annotations

import numpy as np

from .background.disk import sample_disk
from .background.random import sample_random
from .background.sre import sample_sre
from .cleaning.coord_clean import CleaningReport, auto_clean
from .cleaning.thinning import ThinningReport, thin_to_pixel
from .config import SDMConfig
from .io.occurrences import OccurrenceData, reproject_occurrences
from .io.rasters import RasterStack, extract_values
from .predictors.vif import VIFReport, stepwise_vif
from .split.kfold import kfold
from .split.random_split import random_train_test
from .split.spatial_block import SpatialBlockPlan, spatial_block_folds
from .units import distance_to_crs_units

Fold = tuple[np.ndarray, np.ndarray]


def stage_clean(
    cfg: SDMConfig,
    occ: OccurrenceData,
    stack: RasterStack,
) -> tuple[OccurrenceData, CleaningReport | None, ThinningReport | None]:
    """Reproject occurrences into the predictor stack's CRS (a no-op if
    already matching), then apply auto coordinate cleaning and/or thinning
    per cfg.cleaning.

    Reprojection runs unconditionally, not gated by cfg.cleaning.auto_clean —
    every downstream consumer (extent/nodata checks here, and raw pixel
    extraction in collect_labeled_points_and_extract) assumes occ.x/occ.y are
    already in the raster's own CRS units.
    """
    occ = reproject_occurrences(occ, stack.crs)
    cleaning_rep: CleaningReport | None = None
    thinning_rep: ThinningReport | None = None
    if cfg.cleaning.auto_clean:
        occ, cleaning_rep = auto_clean(occ, stack)
    if cfg.cleaning.thin_to_raster_resolution:
        occ, thinning_rep = thin_to_pixel(occ, stack)
    return occ, cleaning_rep, thinning_rep


def collect_labeled_points_and_extract(
    cfg: SDMConfig,
    occ: OccurrenceData,
    stack: RasterStack,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build the labeled (presence + background) point set and extract
    predictor values at each point, dropping rows with any non-finite value.

    Returns (px, py, presence_flag, X_full, feature_names).
    """
    if cfg.data_mode == "presence_absence":
        px, py, presence_flag = occ.x, occ.y, occ.presence.astype(np.uint8)
    else:
        # presence-only: user's points are all presence; generate background
        pres_mask = (
            occ.presence.astype(bool)
            if occ.presence.max() > 0
            else np.ones(len(occ.x), dtype=bool)
        )
        pres_x = occ.x[pres_mask]
        pres_y = occ.y[pres_mask]
        n_bg = cfg.background.resolve_count(len(pres_x))
        # "ratio" places points exactly as "random" does; the two differ only
        # in how n_bg above was arrived at.
        if cfg.background.method in ("random", "ratio"):
            bx, by = sample_random(stack, n_bg, rng=rng)
        elif cfg.background.method == "sre":
            bx, by = sample_sre(
                stack, pres_x, pres_y, n_bg,
                quantile=cfg.background.sre_quantile, rng=rng,
            )
        else:
            # The disk's distances are always real-world meters; convert to
            # the stack's native CRS units here (pass-through for projected
            # CRSs, latitude-corrected for geographic ones) so the ring is
            # applied correctly regardless of the raster's CRS. A max of 0
            # means "no upper limit" and must stay 0 through the conversion.
            lat = float(pres_y.mean())
            min_crs_units = distance_to_crs_units(
                cfg.background.min_distance, stack.crs, lat
            )
            max_crs_units = (
                distance_to_crs_units(cfg.background.max_distance, stack.crs, lat)
                if cfg.background.max_distance > 0
                else 0.0
            )
            bx, by = sample_disk(
                stack, pres_x, pres_y, n_bg, min_crs_units, max_crs_units, rng=rng
            )
        px = np.concatenate([pres_x, bx])
        py = np.concatenate([pres_y, by])
        presence_flag = np.concatenate(
            [np.ones(len(pres_x), dtype=np.uint8), np.zeros(len(bx), dtype=np.uint8)]
        )

    feature_names = list(stack.names)
    all_x_full = extract_values(stack, px, py)
    finite = np.all(np.isfinite(all_x_full), axis=1)
    px, py, presence_flag = px[finite], py[finite], presence_flag[finite]
    X_full = all_x_full[finite]
    return px, py, presence_flag, X_full, feature_names


def stage_vif(
    cfg: SDMConfig,
    X_full: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str], list[int], VIFReport]:
    """Run stepwise VIF predictor selection, or keep every predictor
    untouched if cfg.vif.enabled is False."""
    if not cfg.vif.enabled:
        kept_idx = list(range(len(feature_names)))
        return (
            X_full,
            list(feature_names),
            kept_idx,
            VIFReport(cutoff=cfg.vif.cutoff, retained=list(feature_names), skipped=True),
        )
    X_kept, kept_names, vif_report = stepwise_vif(
        X_full, feature_names, cutoff=cfg.vif.cutoff
    )
    kept_idx = [feature_names.index(n) for n in kept_names]
    return X_kept, kept_names, kept_idx, vif_report


def make_folds(
    cfg: SDMConfig,
    X: np.ndarray,
    y: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    stack: RasterStack,
    rng: np.random.Generator,
) -> tuple[list[Fold], SpatialBlockPlan | None, np.ndarray | None]:
    """Generate CV folds per cfg.split. Returns (folds, plan, fold_id); plan
    and fold_id are None for non-spatial split methods.
    """
    method = cfg.split.method
    if method == "random":
        return random_train_test(len(y), cfg.split.test_size, rng, y=y), None, None
    if method == "kfold":
        return kfold(len(y), cfg.split.k, rng, y=y), None, None
    if method == "spatial_block":
        if cfg.split.auto_block_size:
            block_size = 0.0
        else:
            # cfg.split.block_size is always real-world meters; convert to
            # the stack's native CRS units (see collect_labeled_points_and_extract).
            block_size = distance_to_crs_units(
                cfg.split.block_size, stack.crs, float(py.mean())
            )
        folds, plan, fold_id = spatial_block_folds(
            px, py, stack, k=cfg.split.k, block_size=block_size,
            block_shape=cfg.split.block_shape, rng=rng,
        )
        return folds, plan, fold_id
    raise ValueError(f"Unknown split method: {method}")


def validate_matching_bands(a: RasterStack, b: RasterStack) -> None:
    """Verify the projection stack (b) has exactly the training stack's (a)
    predictors, in exactly the same order.

    Order matters: kept_idx is computed against the training stack's band
    order and then reused as-is to index into the projection stack's bands
    (predict_raster, mess, mop) — a length-only check would silently let a
    same-set-different-order projection stack through, feeding each model
    the wrong predictor's values at every pixel with no error.
    """
    if a.names == b.names:
        return
    if len(a.names) != len(b.names):
        raise ValueError(
            f"Projection raster stack has {len(b.names)} bands but training stack has {len(a.names)}."
        )
    if sorted(a.names) == sorted(b.names):
        raise ValueError(
            "Projection rasters are the same predictors as training but in a different "
            f"order.\n  Training order:   {a.names}\n  Projection order: {b.names}\n"
            "Reorder the projection raster selection to match the training predictor order."
        )
    missing = [n for n in a.names if n not in b.names]
    extra = [n for n in b.names if n not in a.names]
    raise ValueError(
        "Projection rasters don't match the training predictors.\n"
        f"  Training:   {a.names}\n  Projection: {b.names}\n"
        f"  Missing from projection: {missing or 'none'}\n"
        f"  Not in training:         {extra or 'none'}"
    )
