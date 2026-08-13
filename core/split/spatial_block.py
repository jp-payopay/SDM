from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..io.rasters import RasterStack, extract_values

_SQRT3 = math.sqrt(3.0)

BLOCK_SHAPES = ("square", "hexagon")


@dataclass
class SpatialBlockPlan:
    block_size: float
    n_blocks_x: int
    n_blocks_y: int
    source: str  # "auto_variogram" | "auto_fallback" | "user"
    shape: str = "square"  # "square" | "hexagon"
    # Which fold each occupied block was assigned to, keyed by block id.
    fold_of_block: dict[int, int] = field(default_factory=dict)
    # block id -> (center_x, center_y), in the same CRS units as x/y. Used to
    # draw the block partitioning as polygons (see block_polygon_corners) —
    # the *only* geometry a UI layer needs; it never re-derives block extents
    # itself, so what's drawn always matches what was actually partitioned on.
    block_centers: dict[int, tuple[float, float]] = field(default_factory=dict)


def spatial_block_folds(
    x: np.ndarray,
    y: np.ndarray,
    stack: RasterStack,
    k: int = 5,
    block_size: float = 0.0,
    block_shape: str = "square",
    rng: np.random.Generator | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], SpatialBlockPlan, np.ndarray]:
    """Spatial block CV. If block_size <= 0, estimate it via a variogram range.

    block_shape selects the tessellation: "square" (a regular row/column
    grid) or "hexagon" (a regular hexagonal tiling — every cell has 6
    equidistant neighbors, unlike a square's mix of orthogonal/diagonal
    neighbors at different distances, which is why some spatial-CV tools
    default to it). Both interpret block_size as the ground area of one
    block (a square's edge length, squared; a hexagon sized to cover that
    same area) — so auto block sizing (the variogram-range estimate) applies
    unchanged regardless of which shape is picked.

    Blocks are assigned to folds in a shuffled round-robin so each fold gets
    a near-equal share of blocks, spatially scattered across the extent
    (this is the "random" block-to-fold assignment used by e.g. R's blockCV
    and ENMeval — deliberately not spatially clustered into a handful of
    mega-regions, since the point is holding out spatially-independent test
    blocks, not partitioning the map into k contiguous zones).
    """
    if block_shape not in BLOCK_SHAPES:
        raise ValueError(f"block_shape must be one of {BLOCK_SHAPES}, got {block_shape!r}")
    if rng is None:
        rng = np.random.default_rng()
    if block_size <= 0:
        block_size, source = _auto_block_size(stack, rng)
    else:
        source = "user"

    minx, miny, maxx, maxy = stack.bounds
    if block_shape == "hexagon":
        block_id, centers = _assign_hex_blocks(x, y, block_size)
        n_bx, n_by = _hex_grid_dims(minx, miny, maxx, maxy, block_size)
    else:
        block_id, centers, n_bx, n_by = _assign_square_blocks(x, y, minx, miny, maxx, maxy, block_size)

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
    if not folds:
        raise ValueError(
            f"Spatial-block CV produced zero usable folds: block_size={block_size:g} "
            f"partitions the predictor extent into only {n_bx}x{n_by} block(s) "
            f"({len(unique_blocks)} occupied), too coarse to spread across k={k} "
            "folds with both a non-empty train and test set in each. Use a smaller "
            "block size, or enable auto block size."
        )
    plan = SpatialBlockPlan(
        block_size=float(block_size),
        n_blocks_x=int(n_bx),
        n_blocks_y=int(n_by),
        source=source,
        shape=block_shape,
        fold_of_block={int(b): int(f) for b, f in fold_of_block.items()},
        block_centers={int(b): centers[int(b)] for b in unique_blocks},
    )
    return folds, plan, fold_id


def block_polygon_corners(shape: str, cx: float, cy: float, block_size: float) -> list[tuple[float, float]]:
    """Vertices (in order) of the polygon representing one block, for
    drawing. Uses the exact same size math spatial_block_folds partitioned
    on, so a drawn block always matches the real assignment — never a
    separately-eyeballed approximation."""
    if shape == "hexagon":
        return _hex_corners(cx, cy, _hex_circumradius(block_size))
    half = block_size / 2.0
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


def _assign_square_blocks(
    x: np.ndarray, y: np.ndarray, minx: float, miny: float, maxx: float, maxy: float, block_size: float,
) -> tuple[np.ndarray, dict[int, tuple[float, float]], int, int]:
    n_bx = max(1, int(np.ceil((maxx - minx) / block_size)))
    n_by = max(1, int(np.ceil((maxy - miny) / block_size)))
    col = np.clip(((x - minx) / block_size).astype(int), 0, n_bx - 1)
    row = np.clip(((y - miny) / block_size).astype(int), 0, n_by - 1)
    block_id = row * n_bx + col
    centers: dict[int, tuple[float, float]] = {}
    for b in np.unique(block_id):
        r, c = divmod(int(b), n_bx)
        centers[int(b)] = (minx + (c + 0.5) * block_size, miny + (r + 0.5) * block_size)
    return block_id, centers, n_bx, n_by


# ----- hexagon tessellation -----
#
# Flat-top regular hexagons (pointy corners left/right, flat edges top/
# bottom) on the standard "axial coordinates" grid — see
# https://www.redblobgames.com/grids/hexagons/ for the reference derivation
# of the pixel<->axial formulas and the cube-coordinate rounding below.
# `size` is always the circumradius (center-to-corner distance); public
# callers instead speak in `block_size`, converted via _hex_circumradius so
# a hexagon covers the same ground area as a block_size x block_size square
# — the same "how big is one block" quantity block_size means for the square
# shape, so auto block sizing (the variogram-range estimate) produces
# comparably-sized blocks regardless of which shape is picked.
_HEX_AREA_PER_CIRCUMRADIUS_SQ = 3.0 * _SQRT3 / 2.0  # area = this * size**2


def _hex_circumradius(block_size: float) -> float:
    return block_size / math.sqrt(_HEX_AREA_PER_CIRCUMRADIUS_SQ)


def _hex_pixel_to_axial(px: np.ndarray, py: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    q = (2.0 / 3.0 * px) / size
    r = (-1.0 / 3.0 * px + _SQRT3 / 3.0 * py) / size
    return q, r


def _hex_axial_to_pixel(q: float, r: float, size: float) -> tuple[float, float]:
    x = size * (1.5 * q)
    y = size * (_SQRT3 / 2.0 * q + _SQRT3 * r)
    return x, y


def _hex_axial_round(qf: np.ndarray, rf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-hex rounding via cube coordinates — rounding q and r
    independently misassigns points near cell boundaries; this doesn't."""
    xf, zf = qf, rf
    yf = -xf - zf
    rx, ry, rz = np.round(xf), np.round(yf), np.round(zf)
    dx, dy, dz = np.abs(rx - xf), np.abs(ry - yf), np.abs(rz - zf)
    cond1 = (dx > dy) & (dx > dz)
    cond2 = (~cond1) & (dy > dz)
    cond3 = (~cond1) & (~cond2)
    rx_final = np.where(cond1, -ry - rz, rx)
    ry_final = np.where(cond2, -rx - rz, ry)
    rz_final = np.where(cond3, -rx - ry, rz)
    return rx_final.astype(np.int64), rz_final.astype(np.int64)


def _hex_corners(cx: float, cy: float, size: float) -> list[tuple[float, float]]:
    return [
        (cx + size * math.cos(math.radians(60 * i)), cy + size * math.sin(math.radians(60 * i)))
        for i in range(6)
    ]


def _assign_hex_blocks(
    x: np.ndarray, y: np.ndarray, block_size: float,
) -> tuple[np.ndarray, dict[int, tuple[float, float]]]:
    """Assign each point to a hexagonal cell. block_id is an arbitrary stable
    integer per unique occupied cell (hex tessellation has no rectangular
    row/col indexing the way a square grid does)."""
    size = _hex_circumradius(block_size)
    qf, rf = _hex_pixel_to_axial(x, y, size)
    q, r = _hex_axial_round(qf, rf)
    keys = list(zip(q.tolist(), r.tolist()))
    unique_keys = sorted(set(keys))
    key_to_id = {key: i for i, key in enumerate(unique_keys)}
    block_id = np.asarray([key_to_id[key] for key in keys], dtype=np.int64)
    centers = {
        bid: _hex_axial_to_pixel(key[0], key[1], size) for key, bid in key_to_id.items()
    }
    return block_id, centers


def _hex_grid_dims(minx: float, miny: float, maxx: float, maxy: float, block_size: float) -> tuple[int, int]:
    """Approximate column/row counts across the extent, for the summary
    display only (hex cells have no exact rectangular grid dimensions)."""
    size = _hex_circumradius(block_size)
    col_spacing = 1.5 * size
    row_spacing = _SQRT3 * size
    n_bx = max(1, int(np.ceil((maxx - minx) / col_spacing))) if col_spacing > 0 else 1
    n_by = max(1, int(np.ceil((maxy - miny) / row_spacing))) if row_spacing > 0 else 1
    return n_bx, n_by


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
