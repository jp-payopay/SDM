from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from sdm_plugin.core.io.rasters import iter_windows, load_stack


def test_iter_windows_uses_each_bands_own_nodata_only(tmp_path):
    """Regression test: a band with no nodata of its own must not have
    another band's nodata value applied to it. Otherwise a genuine data
    value that happens to equal a different predictor's nodata sentinel
    (e.g. -9999, a very common convention) gets silently punched out to NaN.
    """
    transform = from_origin(0, 5, 1, 1)

    a_path = tmp_path / "A.tif"
    a = np.full((5, 5), 1.0, dtype=np.float32)
    a[0, 0] = -9999.0  # A's own nodata sentinel
    with rasterio.open(
        a_path, "w", driver="GTiff", height=5, width=5, count=1,
        dtype="float32", crs="EPSG:32633", transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(a, 1)

    b_path = tmp_path / "B.tif"
    b = np.full((5, 5), 2.0, dtype=np.float32)
    b[1, 1] = -9999.0  # a genuine value for B, which defines no nodata at all
    with rasterio.open(
        b_path, "w", driver="GTiff", height=5, width=5, count=1,
        dtype="float32", crs="EPSG:32633", transform=transform, nodata=None,
    ) as dst:
        dst.write(b, 1)

    stack = load_stack([str(a_path), str(b_path)])
    _row_off, _h, arr = next(iter_windows(stack, block_rows=256))

    assert np.isnan(arr[0, 0, 0]), "A's own nodata cell must still be masked"
    assert arr[1, 1, 1] == -9999.0, "B's genuine -9999 value must survive (B has no nodata set)"
    assert np.isfinite(arr[1]).all()
