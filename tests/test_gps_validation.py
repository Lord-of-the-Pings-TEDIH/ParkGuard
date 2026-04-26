"""Tests for app.pipeline.gps_validation.

Pure-math unit tests with no FastAPI / database / ffprobe dependencies, so
they run fast and are stable across CI environments.
"""

from __future__ import annotations

import math

import pytest

from app.pipeline.gps_validation import (
    CROSS_SOURCE_TOLERANCE_M,
    GpsSource,
    GpsValidationError,
    GpsValidationStatus,
    ResolvedGpsPose,
    ValidatedCoordinates,
    cross_validate_sources,
    haversine_distance_m,
    is_inside_romania,
    is_null_island,
    validate_coordinates,
    validate_heading,
    validate_latitude,
    validate_longitude,
)


# -- validate_latitude / longitude -------------------------------------------


@pytest.mark.parametrize("value", [-90.0, -45.5, 0.0, 46.77, 90.0])
def test_validate_latitude_accepts_valid(value: float) -> None:
    assert validate_latitude(value) == value


@pytest.mark.parametrize("value", [-90.0001, 90.1, math.nan, math.inf, -math.inf])
def test_validate_latitude_rejects_invalid(value: float) -> None:
    with pytest.raises(GpsValidationError):
        validate_latitude(value)


@pytest.mark.parametrize("value", [-180.0, -23.5, 0.0, 23.5895, 180.0])
def test_validate_longitude_accepts_valid(value: float) -> None:
    assert validate_longitude(value) == value


@pytest.mark.parametrize("value", [-180.001, 180.001, math.nan, math.inf])
def test_validate_longitude_rejects_invalid(value: float) -> None:
    with pytest.raises(GpsValidationError):
        validate_longitude(value)


# -- validate_heading --------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0, 0.0),
        (90.0, 90.0),
        (359.9, 359.9),
        (360.0, 0.0),       # full revolution wraps to zero
        (-90.0, 270.0),     # west of north
        (720.0, 0.0),       # two revolutions
    ],
)
def test_validate_heading_normalises(value: float, expected: float) -> None:
    assert validate_heading(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_validate_heading_rejects_non_finite(value: float) -> None:
    with pytest.raises(GpsValidationError):
        validate_heading(value)


# -- is_null_island / is_inside_romania --------------------------------------


@pytest.mark.parametrize(
    "lat, lon, expected",
    [
        (0.0, 0.0, True),
        (0.00005, -0.00005, True),     # within tolerance
        (0.001, 0.0, False),           # outside default tolerance (1e-4)
        (45.0, 25.0, False),
    ],
)
def test_is_null_island(lat: float, lon: float, expected: bool) -> None:
    assert is_null_island(lat, lon) is expected


@pytest.mark.parametrize(
    "lat, lon, expected",
    [
        (46.77, 23.59, True),          # Cluj-Napoca
        (44.43, 26.10, True),          # Bucharest
        (43.6, 28.0, True),             # Constanța region (in box)
        (48.4, 22.0, True),             # extreme north-west of box
        (50.0, 25.0, False),            # too far north (Poland-ish)
        (40.0, 25.0, False),            # too far south
        (45.0, 19.0, False),            # too far west
        (45.0, 32.0, False),            # too far east
    ],
)
def test_is_inside_romania(lat: float, lon: float, expected: bool) -> None:
    assert is_inside_romania(lat, lon) is expected


# -- validate_coordinates ----------------------------------------------------


def test_validate_coordinates_accepts_romanian_point() -> None:
    coords = validate_coordinates(46.77, 23.59)
    assert isinstance(coords, ValidatedCoordinates)
    assert coords.latitude == pytest.approx(46.77)
    assert coords.longitude == pytest.approx(23.59)


def test_validate_coordinates_rejects_null_island() -> None:
    with pytest.raises(GpsValidationError, match="Null Island"):
        validate_coordinates(0.0, 0.0)


def test_validate_coordinates_rejects_out_of_country_by_default() -> None:
    with pytest.raises(GpsValidationError, match="Romania"):
        validate_coordinates(-33.8688, 151.2093)  # Sydney


def test_validate_coordinates_allows_out_of_country_when_disabled() -> None:
    coords = validate_coordinates(-33.8688, 151.2093, require_country=False)
    assert coords.latitude == pytest.approx(-33.8688)
    assert coords.longitude == pytest.approx(151.2093)


def test_validate_coordinates_rejects_nan() -> None:
    with pytest.raises(GpsValidationError):
        validate_coordinates(math.nan, 23.59)


def test_validate_coordinates_rejects_out_of_range() -> None:
    with pytest.raises(GpsValidationError, match="WGS84"):
        validate_coordinates(95.0, 23.59)


# -- haversine + cross-validation -------------------------------------------


def test_haversine_distance_zero_for_identical_points() -> None:
    assert haversine_distance_m((46.77, 23.59), (46.77, 23.59)) == pytest.approx(0.0, abs=1e-6)


def test_haversine_distance_matches_known_pair() -> None:
    # Bucharest <-> Cluj-Napoca, ~325 km great-circle.
    distance = haversine_distance_m((44.4268, 26.1025), (46.7700, 23.5895))
    assert 320_000 < distance < 330_000


def test_cross_validate_sources_within_tolerance() -> None:
    primary = ValidatedCoordinates(latitude=46.7700, longitude=23.5895)
    secondary = ValidatedCoordinates(latitude=46.77005, longitude=23.5896)
    agree, distance = cross_validate_sources(primary, secondary)
    assert agree is True
    assert distance < CROSS_SOURCE_TOLERANCE_M


def test_cross_validate_sources_outside_tolerance() -> None:
    primary = ValidatedCoordinates(latitude=46.7700, longitude=23.5895)
    # ~140 m east of primary — well outside the 50 m default tolerance.
    secondary = ValidatedCoordinates(latitude=46.7700, longitude=23.5913)
    agree, distance = cross_validate_sources(primary, secondary)
    assert agree is False
    assert distance > CROSS_SOURCE_TOLERANCE_M


def test_cross_validate_sources_custom_tolerance() -> None:
    primary = ValidatedCoordinates(latitude=46.7700, longitude=23.5895)
    secondary = ValidatedCoordinates(latitude=46.7700, longitude=23.5913)
    agree, _ = cross_validate_sources(primary, secondary, tolerance_m=200.0)
    assert agree is True


# -- ResolvedGpsPose smoke test ---------------------------------------------


def test_resolved_pose_dataclass_defaults() -> None:
    pose = ResolvedGpsPose(
        latitude=46.77,
        longitude=23.59,
        heading_deg=90.0,
        source=GpsSource.EXPLICIT,
    )
    assert pose.status is GpsValidationStatus.OK
    assert pose.warnings == []
    assert pose.accuracy_m is None
