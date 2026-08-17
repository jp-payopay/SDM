from __future__ import annotations

import numpy as np
from rasterio.crs import CRS


def is_geographic_crs(crs: str) -> bool:
    """True if `crs` (e.g. "EPSG:4326") uses angular (degree) units rather
    than a projected linear unit (e.g. metres)."""
    if not crs:
        raise ValueError(
            "Predictor rasters have no embedded CRS, so distances can't be "
            "converted to real-world units. Reproject the predictor rasters "
            "to a CRS with defined units before using a disk background or a "
            "manual (non-auto) spatial block size."
        )
    return CRS.from_string(crs).is_geographic


def meters_per_degree_latitude(latitude_deg: float) -> float:
    """WGS84-ellipsoid meters-per-degree-of-latitude at a given latitude
    (Snyder 1987 / NOAA approximation). Latitude degrees are nearly (but not
    exactly) constant across the globe, unlike longitude degrees, which
    shrink strongly toward the poles — this latitude-based scale is used as
    the single isotropic conversion factor for buffer/block-size distances,
    which are themselves already applied isotropically (the same value in
    both x and y) by the disk-background and spatial-block code.
    """
    lat = np.radians(latitude_deg)
    return (
        111132.92
        - 559.82 * np.cos(2 * lat)
        + 1.175 * np.cos(4 * lat)
        - 0.0023 * np.cos(6 * lat)
    )


def distance_to_crs_units(distance_m: float, crs: str, latitude_deg: float) -> float:
    """Convert a real-world distance in meters to the linear units of `crs`.
    Projected CRSs (already metres) pass through unchanged; geographic CRSs
    (degrees) get an ellipsoid-corrected conversion at `latitude_deg`."""
    if not is_geographic_crs(crs):
        return distance_m
    return distance_m / meters_per_degree_latitude(latitude_deg)


def crs_units_to_meters(value: float, crs: str, latitude_deg: float) -> float:
    """Inverse of distance_to_crs_units — used to display a value already in
    a raster's native CRS units (e.g. an auto-computed spatial block size)
    back in real-world meters."""
    if not is_geographic_crs(crs):
        return value
    return value * meters_per_degree_latitude(latitude_deg)


def format_meters(meters: float) -> str:
    """Human-friendly distance string, switching to km above 1000 m."""
    if meters >= 1000:
        return f"{meters / 1000:.2f} km"
    return f"{meters:.0f} m"
