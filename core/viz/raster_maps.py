from __future__ import annotations

from pathlib import Path

import numpy as np

from . import PUBLICATION_DPI


def plot_raster_map(
    array: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "",
    cmap: str = "viridis",
    categorical: bool = False,
    mask: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Static PNG preview of a raster array, for embedding in the HTML
    report (not a replacement for opening the real GeoTIFF in GIS software).

    `categorical=True` renders a 2-class (0/1) map with a discrete
    unsuitable/suitable legend instead of a continuous colorbar — for binary
    suitability maps. Their nodata isn't recoverable from the array alone
    (`apply_threshold` zeroes non-finite cells rather than preserving them),
    so callers should pass `mask` (e.g. non-finite cells of the source
    continuous raster) to blank those areas out correctly. The two class
    colors are the low/high end of `cmap` (so e.g. cmap="Greens" gives a
    light/dark green pair), not a fixed hardcoded pair.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array, dtype=np.float64)
    invalid = ~np.isfinite(data)
    if mask is not None:
        invalid = invalid | mask
    masked = np.ma.array(data, mask=invalid)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    if categorical:
        base = plt.get_cmap(cmap)
        cm = ListedColormap([base(0.15), base(0.85)])
        cm.set_bad(color="white")
        im = ax.imshow(masked, cmap=cm, vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
        cbar.ax.set_yticklabels(["unsuitable", "suitable"])
    else:
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad(color="white")
        im = ax.imshow(masked, cmap=cm, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=PUBLICATION_DPI)
    plt.close(fig)
    return out_path
