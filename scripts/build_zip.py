"""Build a clean sdm_plugin.zip suitable for the QGIS plugin repository.

Packages only what the plugin needs at runtime (source, metadata, icon,
license, README) and skips test artifacts, caches, and dev-only assets that
otherwise balloon the zip to hundreds of MB.

Usage:
    python scripts/build_zip.py [output_zip_path]
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR_NAME = "sdm_plugin"

# Top-level entries to include, relative to PLUGIN_ROOT.
INCLUDE = [
    "__init__.py",
    "plugin.py",
    "metadata.txt",
    "icon.png",
    "LICENSE",
    "README.md",
    "core",
    "ui",
    "deps",
]

EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".claude", ".git"}
EXCLUDE_FILE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_FILE_NAMES = {".DS_Store", "settings.local.json"}


def iter_files(path: Path):
    if path.is_file():
        yield path
        return
    for child in sorted(path.rglob("*")):
        if child.is_dir():
            continue
        if any(part in EXCLUDE_DIR_NAMES for part in child.relative_to(PLUGIN_ROOT).parts):
            continue
        if child.suffix in EXCLUDE_FILE_SUFFIXES or child.name in EXCLUDE_FILE_NAMES:
            continue
        yield child


def build(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in INCLUDE:
            src = PLUGIN_ROOT / entry
            if not src.exists():
                raise FileNotFoundError(f"expected plugin entry not found: {src}")
            for file in iter_files(src):
                arcname = Path(PLUGIN_DIR_NAME) / file.relative_to(PLUGIN_ROOT)
                zf.write(file, arcname)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else PLUGIN_ROOT / "dist" / "sdm_plugin.zip"
    build(out)
