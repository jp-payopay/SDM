from __future__ import annotations

from sdm_plugin.core.units import (
    crs_units_to_meters,
    distance_to_crs_units,
    format_meters,
    is_geographic_crs,
    meters_per_degree_latitude,
)


def test_is_geographic_crs():
    assert is_geographic_crs("EPSG:4326") is True
    assert is_geographic_crs("EPSG:32633") is False


def test_meters_per_degree_latitude_near_equator():
    # ~110.57 km per degree of latitude at the equator, per WGS84 (smallest
    # meridional radius of curvature there; it grows toward the poles).
    assert abs(meters_per_degree_latitude(0.0) - 110_574) < 50


def test_meters_per_degree_latitude_varies_with_latitude():
    # Meters-per-degree-of-latitude increases slightly toward the poles
    # (the ellipsoid flattens), so this should not be a constant function.
    assert meters_per_degree_latitude(60.0) > meters_per_degree_latitude(0.0)


def test_distance_to_crs_units_projected_passthrough():
    assert distance_to_crs_units(50_000.0, "EPSG:32633", 14.0) == 50_000.0
    assert crs_units_to_meters(50_000.0, "EPSG:32633", 14.0) == 50_000.0


def test_distance_to_crs_units_geographic_roundtrip():
    meters = 50_000.0
    crs = "EPSG:4326"
    lat = 14.0
    degrees = distance_to_crs_units(meters, crs, lat)
    assert 0 < degrees < 1  # 50 km is a fraction of a degree
    back = crs_units_to_meters(degrees, crs, lat)
    assert abs(back - meters) < 1e-6


def test_format_meters():
    assert format_meters(500) == "500 m"
    assert format_meters(45_250) == "45.25 km"
    assert format_meters(1_000) == "1.00 km"
