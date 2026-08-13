from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from sdm_plugin.core.io.rasters import load_stack
from sdm_plugin.core.prediction.mess_mop import mess, mop


def _constant_band_stack(tmp_path, name_to_value: dict[str, float]):
    """A tiny raster stack where every band is a distinct constant value, so
    which band got sampled is directly readable off the output."""
    transform = from_origin(0, 5, 1, 1)
    paths = []
    for name, value in name_to_value.items():
        p = tmp_path / f"{name}.tif"
        with rasterio.open(
            p, "w", driver="GTiff", height=5, width=5, count=1,
            dtype="float32", crs="EPSG:32633", transform=transform,
        ) as dst:
            dst.write(np.full((5, 5), value, dtype="float32"), 1)
        paths.append(str(p))
    return load_stack(paths)


def test_mess_uses_kept_feature_idx_not_positional(tmp_path):
    """Regression test: when VIF drops a non-trailing predictor (e.g. keeps
    a/c/e out of a/b/c/d/e), kept_feature_idx=[0, 2, 4] must select those
    same bands out of the (unreduced) projection stack — not bands
    [0, 1, 2] (a/b/c), which is what a stale `range(len(kept_idx))` would
    silently produce.
    """
    stack = _constant_band_stack(
        tmp_path, {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    )
    kept_idx = [0, 2, 4]  # positions of a, c, e in the full stack

    rng = np.random.default_rng(0)
    # Training ranges tightly bracket a/c/e's projected constant values but
    # exclude b/d's — if the wrong bands were sampled, MESS would report
    # heavy novelty (large negative values) instead of in-range similarity.
    X_kept = np.stack(
        [
            rng.uniform(0.9, 1.1, 200),
            rng.uniform(2.9, 3.1, 200),
            rng.uniform(4.9, 5.1, 200),
        ],
        axis=1,
    )

    mess_arr = mess(stack, X_kept, kept_feature_idx=kept_idx)
    assert np.all(np.isfinite(mess_arr))
    assert np.all(mess_arr > 0), "expected in-range similarity when a/c/e are sampled correctly"

    mop_arr = mop(stack, X_kept, kept_feature_idx=kept_idx)
    assert np.all(np.isfinite(mop_arr))
    assert np.all(mop_arr < 1.0), "expected small dissimilarity when a/c/e are sampled correctly"
