from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..io.rasters import RasterStack, extract_values


@dataclass
class SpatialBlockPlan:
    block_size: float
    n_blocks_x: int
    n_blocks_y: int
    source: str  # "auto_variogram" | "auto_fallback" | "user"
    # Which fold each occupied block was assigned to, keyed by block id
    # (row * n_blocks_x + col). Only blocks that actually contain points
    # appear here. Used to draw the block partitioning as colored polygons.
    fold_of_block: dict[int, int] = field(default_factory=dict)


def spatial_block_folds(
    x: np.ndarray,
    y: np.ndarray,
    stack: RasterStack,
    k: int = 5,
    block_size: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], SpatialBlockPlan, np.ndarray]:
    """Spatial block CV. If block_size <= 0, estimate it via a variogram range.

    Blocks are assigned to folds in a shuffled round-robin so each fold contains
    a spatially distributed set of blocks.
    """
    if rng is None:
        rng = np.random.default_rng()
    if block_size <= 0:
        block_size, source = _auto_block_size(stack, rng)
    else:
        source = "user"

    minx, miny, maxx, maxy = stack.bounds
    n_bx = max(1, int(np.ceil((maxx - minx) / block_size)))
    n_by = max(1, int(np.ceil((maxy - miny) / block_size)))
    col = np.clip(((x - minx) / block_size).astype(int), 0, n_bx - 1)
    row = np.clip(((y - miny) / block_size).astype(int), 0, n_by - 1)
    block_id = row * n_bx + col

    unique_blocks = np.unique(block_id)
    rng.shuffle(unique_blocks)
    fold_of_block: dict[int, int] = {
        int(b): i % k for i, b in enumerate(unique_blocks)
    }
    fold_id = np.asarray([fold_of_block[int(b)] for b in block_id])

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        test = np.where(fold_id == i)[0]
        train = np.where(fold_id != i)[0]
        if len(test) == 0 or len(train) == 0:
            continue
        folds.append((train, test))
    plan = SpatialBlockPlan(
        block_size=float(block_size),
        n_blocks_x=int(n_bx),
        n_blocks_y=int(n_by),
        source=source,
        fold_of_block={int(b): int(f) for b, f in fold_of_block.items()},
    )
    return folds, plan, fold_id


def _auto_block_size(stack: RasterStack, rng: np.random.Generator) -> tuple[float, str]:
    """Estimate spatial autocorrelation range via a simple empirical variogram
    on the first predictor. Falls back to extent/20 if the variogram is degenerate.
    """
    minx, miny, maxx, maxy = stack.bounds
    extent = min(maxx - minx, maxy - miny)
    fallback = float(extent / 20.0)
    n_samples = 2000
    xs = rng.uniform(minx, maxx, size=n_samples)
    ys = rng.uniform(miny, maxy, size=n_samples)
    try:
        vals = extract_values(stack, xs, ys)[:, 0]
    except Exception:
        return fallback, "auto_fallback"
    ok = np.isfinite(vals)
    xs, ys, vals = xs[ok], ys[ok], vals[ok]
    if len(vals) < 200:
        return fallback, "auto_fallback"

    n_pairs = 20_000
    idx_i = rng.integers(0, len(vals), size=n_pairs)
    idx_j = rng.integers(0, len(vals), size=n_pairs)
    keep = idx_i != idx_j
    idx_i, idx_j = idx_i[keep], idx_j[keep]
    d = np.hypot(xs[idx_i] - xs[idx_j], ys[idx_i] - ys[idx_j])
    gamma = 0.5 * (vals[idx_i] - vals[idx_j]) ** 2

    n_bins = 20
    max_d = np.percentile(d, 90)
    if max_d <= 0:
        return fallback, "auto_fallback"
    edges = np.linspace(0, max_d, n_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    which = np.digitize(d, edges) - 1
    mean_g = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = which == b
        if m.sum() > 20:
            mean_g[b] = float(gamma[m].mean())
    finite = np.isfinite(mean_g)
    if finite.sum() < 5:
        return fallback, "auto_fallback"
    sill = float(np.nanpercentile(mean_g, 95))
    if sill <= 0:
        return fallback, "auto_fallback"
    threshold = 0.9 * sill
    hit = np.where(mean_g >= threshold)[0]
    if len(hit) == 0:
        return fallback, "auto_fallback"
    est_range = float(centers[hit[0]])
    est_range = max(est_range, fallback / 4.0)
    est_range = min(est_range, extent / 3.0)
    return est_range, "auto_variogram"
