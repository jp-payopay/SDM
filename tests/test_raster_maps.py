from __future__ import annotations

import numpy as np

from sdm_plugin.core.viz.raster_maps import plot_raster_map


def test_plot_raster_map_continuous(tmp_path):
    arr = np.random.default_rng(0).random((10, 10))
    arr[0, 0] = np.nan
    out = plot_raster_map(arr, tmp_path / "cont.png", title="test")
    assert out.exists()


def test_plot_raster_map_categorical_with_mask(tmp_path):
    arr = np.zeros((5, 5), dtype=np.uint8)
    arr[2:, 2:] = 1
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, :] = True  # simulate off-extent row
    out = plot_raster_map(arr, tmp_path / "bin.png", title="binary", categorical=True, mask=mask)
    assert out.exists()


def test_plot_raster_map_creates_parent_dir(tmp_path):
    arr = np.ones((3, 3))
    out = plot_raster_map(arr, tmp_path / "nested" / "map.png")
    assert out.exists()


def test_plot_raster_map_categorical_respects_cmap(tmp_path):
    """The two class colors come from the requested cmap's low/high end, not
    a fixed hardcoded pair — different cmaps should paint different pixels."""
    import matplotlib.image as mpimg

    arr = np.zeros((6, 6), dtype=np.uint8)
    arr[3:, 3:] = 1
    greens = plot_raster_map(arr, tmp_path / "greens.png", categorical=True, cmap="Greens")
    purples = plot_raster_map(arr, tmp_path / "purples.png", categorical=True, cmap="Purples")
    img_greens = mpimg.imread(greens)
    img_purples = mpimg.imread(purples)
    assert not np.array_equal(img_greens, img_purples)
