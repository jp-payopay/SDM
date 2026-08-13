import numpy as np
import rasterio
from rasterio.transform import from_origin

from sdm_plugin.core.evaluation.threshold import apply_threshold, apply_threshold_masked
from sdm_plugin.core.io.outputs import write_raster
from sdm_plugin.core.io.rasters import RasterStack


def _stack(h, w):
    return RasterStack(
        names=["b"], paths=["unused"], crs="EPSG:4326",
        transform=from_origin(0.0, float(h), 1.0, 1.0),
        width=w, height=h, nodata=-9999.0, shape=(h, w),
    )


def test_apply_threshold_masked_marks_nan_as_nodata():
    r = np.array([[0.1, np.nan], [0.9, 0.4]])
    out = apply_threshold_masked(r, 0.5, nodata=255)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0 and out[1, 0] == 1 and out[1, 1] == 0
    assert out[0, 1] == 255  # masked cell is nodata, not 0
    # The plain version fills masked cells with 0 (kept for plotting/in-memory).
    assert apply_threshold(r, 0.5)[0, 1] == 0


def test_binary_geotiff_is_clipped_to_mask_like_continuous(tmp_path):
    cont = np.array(
        [[0.2, np.nan, 0.8], [np.nan, 0.6, 0.1], [0.9, 0.3, np.nan]], dtype=np.float32
    )
    st = _stack(*cont.shape)
    write_raster(tmp_path / "cont.tif", cont, st)
    write_raster(
        tmp_path / "bin.tif",
        apply_threshold_masked(cont, 0.5, nodata=255),
        st, dtype="uint8", nodata=255,
    )

    with rasterio.open(tmp_path / "cont.tif") as c:
        cont_masked = c.read_masks(1) == 0
    with rasterio.open(tmp_path / "bin.tif") as b:
        assert b.nodata == 255
        bin_masked = b.read_masks(1) == 0
        bin_data = b.read(1)

    expected = ~np.isfinite(cont)
    # Binary nodata footprint matches the continuous raster's, i.e. the data mask.
    assert np.array_equal(bin_masked, expected)
    assert np.array_equal(bin_masked, cont_masked)
    # Masked cells carry nodata (255), not a spurious 0 = "unsuitable".
    assert (bin_data[expected] == 255).all()
    # Valid cells are a real 0/1 classification.
    assert set(np.unique(bin_data[~expected]).tolist()) <= {0, 1}
