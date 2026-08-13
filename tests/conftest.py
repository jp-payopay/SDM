from __future__ import annotations

import os
import sys
from pathlib import Path

# Render figures at a low resolution during tests so the pipeline test does not
# spend most of its time writing 1200 dpi (6000 px) PNGs. Production defaults to
# 1200 dpi; this must be set before core.viz is imported (it reads the value at
# import time). See core/viz/__init__.py::PUBLICATION_DPI.
os.environ.setdefault("SDM_FIGURE_DPI", "100")

import importlib.util

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

# Tests import via `from sdm_plugin.core... import ...`, but the checkout
# directory itself may not be named "sdm_plugin" (a fresh clone, a CI
# checkout path, a second local copy, etc). Rather than rely on
# sys.path + the real directory name happening to match, register this
# checkout's __init__.py under the literal package name "sdm_plugin" in
# sys.modules regardless of what the directory is actually called.
PLUGIN_ROOT = Path(__file__).parent.parent
PACKAGE_NAME = "sdm_plugin"
if PACKAGE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PLUGIN_ROOT / "__init__.py",
        submodule_search_locations=[str(PLUGIN_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)


@pytest.fixture
def tiny_stack(tmp_path):
    """Build a 3-band synthetic raster stack (40 rows x 40 cols)."""
    height, width = 40, 40
    transform = from_origin(0.0, 40.0, 1.0, 1.0)  # top-left at (0, 40), 1x1 cells
    paths = []
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    bands = [
        (xx + yy) / (height + width),  # smooth gradient
        np.sin(xx / 5) + np.cos(yy / 5),  # oscillatory
        rng.normal(size=(height, width)).astype(np.float32),  # noise
    ]
    for i, arr in enumerate(bands):
        p = tmp_path / f"band_{i}.tif"
        with rasterio.open(
            p, "w",
            driver="GTiff", height=height, width=width, count=1,
            dtype="float32", crs="EPSG:32633", transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(arr.astype(np.float32), 1)
        paths.append(str(p))
    return paths


@pytest.fixture
def po_csv(tmp_path):
    """Presence-only CSV inside the tiny_stack extent."""
    import pandas as pd

    rng = np.random.default_rng(1)
    x = rng.uniform(1.0, 39.0, size=50)
    y = rng.uniform(1.0, 39.0, size=50)
    df = pd.DataFrame({"x": x, "y": y})
    p = tmp_path / "occ_po.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def pa_csv(tmp_path):
    import pandas as pd

    rng = np.random.default_rng(2)
    x = rng.uniform(1.0, 39.0, size=100)
    y = rng.uniform(1.0, 39.0, size=100)
    presence = ((x + y) > 40).astype(int)
    df = pd.DataFrame({"x": x, "y": y, "presence": presence})
    p = tmp_path / "occ_pa.csv"
    df.to_csv(p, index=False)
    return str(p)
