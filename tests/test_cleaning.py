import numpy as np

from sdm_plugin.core.cleaning.coord_clean import auto_clean
from sdm_plugin.core.cleaning.thinning import thin_to_pixel
from sdm_plugin.core.io.occurrences import OccurrenceData
from sdm_plugin.core.io.rasters import load_stack


def test_auto_clean_removes_bad_points(tiny_stack):
    stack = load_stack(tiny_stack)
    x = np.array([5.0, 5.0, 0.0, np.nan, 100.0, 20.0])
    y = np.array([5.0, 5.0, 0.0, 10.0, 100.0, 20.0])
    presence = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
    occ = OccurrenceData(x=x, y=y, presence=presence, crs="EPSG:32633")
    cleaned, report = auto_clean(occ, stack)
    assert report.n_input == 6
    assert "nan_coords" in report.dropped
    assert "zero_zero" in report.dropped
    assert "duplicate" in report.dropped
    assert "out_of_extent" in report.dropped
    assert 20.0 in cleaned.x.tolist()


def test_thin_to_pixel_reduces_to_unique_cells(tiny_stack):
    stack = load_stack(tiny_stack)
    x = np.array([5.2, 5.7, 5.4, 20.0])  # first three fall in same pixel
    y = np.array([5.3, 5.9, 5.1, 20.0])
    presence = np.array([1, 1, 1, 1], dtype=np.uint8)
    occ = OccurrenceData(x=x, y=y, presence=presence, crs="EPSG:32633")
    thinned, rep = thin_to_pixel(occ, stack)
    assert rep.n_output == 2
