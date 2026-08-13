from __future__ import annotations

import numpy as np
import pytest

from sdm_plugin.core.io.rasters import load_stack
from sdm_plugin.core.split.kfold import kfold
from sdm_plugin.core.split.random_split import random_train_test
from sdm_plugin.core.split.spatial_block import (
    _assign_hex_blocks,
    _hex_axial_to_pixel,
    _hex_circumradius,
    block_polygon_corners,
    spatial_block_folds,
)


def test_random_train_test_shapes():
    folds = random_train_test(100, test_size=0.3, rng=np.random.default_rng(0))
    assert len(folds) == 1
    train, test = folds[0]
    assert len(test) == 30
    assert len(train) == 70
    assert set(train.tolist()).isdisjoint(set(test.tolist()))


def test_kfold_shapes_and_coverage():
    n = 97
    k = 5
    folds = kfold(n, k, rng=np.random.default_rng(0))
    assert len(folds) == k
    seen_test = set()
    for train, test in folds:
        assert set(train.tolist()).isdisjoint(set(test.tolist()))
        seen_test |= set(test.tolist())
    assert seen_test == set(range(n))


def test_random_train_test_stratified_preserves_class_ratio():
    rng = np.random.default_rng(0)
    y = np.array([1] * 20 + [0] * 80)  # 20% presence, 80% background
    train, test = random_train_test(len(y), test_size=0.25, rng=rng, y=y)[0]
    assert set(train.tolist()).isdisjoint(set(test.tolist()))
    assert len(test) == 5 + 20  # round(20*0.25) presence + round(80*0.25) background
    assert (y[test] == 1).sum() == 5
    assert (y[test] == 0).sum() == 20


def test_kfold_stratified_preserves_class_ratio_each_fold():
    rng = np.random.default_rng(0)
    y = np.array([1] * 20 + [0] * 80)
    folds = kfold(len(y), k=5, rng=rng, y=y)
    assert len(folds) == 5
    seen_test = set()
    for train, test in folds:
        assert set(train.tolist()).isdisjoint(set(test.tolist()))
        # each fold's test set should have ~1/5 of each class (4 presence, 16 background)
        assert (y[test] == 1).sum() == 4
        assert (y[test] == 0).sum() == 16
        seen_test |= set(test.tolist())
    assert seen_test == set(range(len(y)))


def test_spatial_block_folds_returns_fold_id(tiny_stack):
    stack = load_stack(tiny_stack)
    rng = np.random.default_rng(0)
    n = 300
    minx, miny, maxx, maxy = stack.bounds
    x = rng.uniform(minx, maxx, size=n)
    y = rng.uniform(miny, maxy, size=n)

    folds, plan, fold_id = spatial_block_folds(x, y, stack, k=4, rng=rng)

    assert len(fold_id) == n
    assert fold_id.min() >= 0
    assert fold_id.max() < plan.n_blocks_x * plan.n_blocks_y
    for train, test in folds:
        assert len(train) > 0
        assert len(test) > 0
        assert set(train.tolist()).isdisjoint(set(test.tolist()))


def test_spatial_block_folds_user_block_size(tiny_stack):
    stack = load_stack(tiny_stack)
    rng = np.random.default_rng(1)
    n = 200
    minx, miny, maxx, maxy = stack.bounds
    x = rng.uniform(minx, maxx, size=n)
    y = rng.uniform(miny, maxy, size=n)

    folds, plan, fold_id = spatial_block_folds(x, y, stack, k=3, block_size=5.0, rng=rng)
    assert plan.source == "user"
    assert plan.block_size == 5.0
    assert len(fold_id) == n


def test_spatial_block_folds_oversized_block_size_raises_clearly(tiny_stack):
    """Regression test: a manual block_size larger than the raster extent
    used to silently produce zero folds, which surfaced later as an opaque
    'need at least one array to concatenate' error deep in the pipeline.
    It must now fail immediately with a clear, actionable message.
    """
    stack = load_stack(tiny_stack)
    rng = np.random.default_rng(2)
    n = 100
    minx, miny, maxx, maxy = stack.bounds
    x = rng.uniform(minx, maxx, size=n)
    y = rng.uniform(miny, maxy, size=n)

    with pytest.raises(ValueError, match="zero usable folds"):
        spatial_block_folds(x, y, stack, k=4, block_size=1000.0, rng=rng)


def test_spatial_block_folds_invalid_shape_raises(tiny_stack):
    stack = load_stack(tiny_stack)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 40, size=50)
    y = rng.uniform(0, 40, size=50)
    with pytest.raises(ValueError, match="block_shape"):
        spatial_block_folds(x, y, stack, k=3, block_shape="triangle", rng=rng)


def test_spatial_block_folds_hexagon_shape(tiny_stack):
    """Same end-to-end contract as the square-grid test above, but for the
    hexagon tessellation: valid folds, disjoint train/test, and the plan
    correctly reports the shape and per-block centers used."""
    stack = load_stack(tiny_stack)
    rng = np.random.default_rng(0)
    n = 300
    minx, miny, maxx, maxy = stack.bounds
    x = rng.uniform(minx, maxx, size=n)
    y = rng.uniform(miny, maxy, size=n)

    folds, plan, fold_id = spatial_block_folds(x, y, stack, k=4, block_shape="hexagon", rng=rng)

    assert plan.shape == "hexagon"
    assert len(fold_id) == n
    assert fold_id.min() >= 0
    assert fold_id.max() < 4
    assert plan.block_centers  # at least one occupied hex cell
    for train, test in folds:
        assert len(train) > 0
        assert len(test) > 0
        assert set(train.tolist()).isdisjoint(set(test.tolist()))


def test_block_polygon_corners_hexagon_is_regular():
    """The drawn hexagon must actually be a regular hexagon (all 6 corners
    equidistant from the center, all 6 sides equal length) — otherwise the
    real partitioning and what gets drawn for it could silently diverge."""
    cx, cy, block_size = 12.5, -4.0, 30.0
    corners = block_polygon_corners("hexagon", cx, cy, block_size)
    assert len(corners) == 6

    radii = [np.hypot(px - cx, py - cy) for px, py in corners]
    assert np.allclose(radii, radii[0], rtol=1e-9)

    sides = [
        np.hypot(corners[i][0] - corners[(i + 1) % 6][0], corners[i][1] - corners[(i + 1) % 6][1])
        for i in range(6)
    ]
    assert np.allclose(sides, sides[0], rtol=1e-9)


def test_block_polygon_corners_hexagon_matches_block_size_area():
    """block_size means "same ground area as a block_size x block_size
    square" for the hexagon shape too — verify that calibration directly via
    the shoelace formula on the drawn polygon."""
    block_size = 40.0
    corners = block_polygon_corners("hexagon", 0.0, 0.0, block_size)
    xs = np.array([c[0] for c in corners])
    ys = np.array([c[1] for c in corners])
    area = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
    assert area == pytest.approx(block_size ** 2, rel=1e-6)


def test_hex_axial_round_trip_at_known_centers():
    """A point placed exactly at a hex cell's own center must be assigned to
    that same cell — the nearest-hex (cube-coordinate) rounding must not
    push it into a neighboring cell due to floating-point noise."""
    block_size = 25.0
    size = _hex_circumradius(block_size)
    qr_pairs = [(0, 0), (1, 0), (-1, 0), (0, 1), (2, -3), (5, 5)]
    xs, ys, expected = [], [], []
    for q, r in qr_pairs:
        cx, cy = _hex_axial_to_pixel(q, r, size)
        xs.append(cx)
        ys.append(cy)
        expected.append((q, r))
    block_id, centers = _assign_hex_blocks(np.array(xs), np.array(ys), block_size)
    # Every input center must land in its own distinct block...
    assert len(set(block_id.tolist())) == len(qr_pairs)
    # ...and round-trip back to the exact pixel it started from.
    for bid, (cx, cy) in zip(block_id, zip(xs, ys)):
        assert centers[int(bid)] == pytest.approx((cx, cy), abs=1e-9)


def test_assign_hex_blocks_tight_cluster_shares_one_block():
    """Points within a small fraction of the cell size of each other should
    (almost always) land in the same hex cell, not scatter across many."""
    rng = np.random.default_rng(3)
    block_size = 50.0
    cx, cy = 100.0, 100.0
    n = 200
    # Jitter well inside a single cell's extent (circumradius ~28.9 here).
    x = cx + rng.uniform(-3.0, 3.0, size=n)
    y = cy + rng.uniform(-3.0, 3.0, size=n)
    block_id, _centers = _assign_hex_blocks(x, y, block_size)
    assert len(set(block_id.tolist())) == 1
