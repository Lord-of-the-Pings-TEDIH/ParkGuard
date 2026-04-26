"""Tests for app.pipeline.geolocation.PlateGeolocationCalculator.

These tests verify the pinhole-projection geometry without depending on
PaddleOCR / YOLO / a database — they are pure-math unit tests.
"""

from __future__ import annotations

import math

import pytest
from geopy.distance import geodesic

from app.pipeline.geolocation import (
    CameraIntrinsics,
    PlateGeolocationCalculator,
    VehiclePose,
)


def _intrinsics(
    *,
    pitch_deg: float = 0.0,
    height_m: float = 1.5,
    focal_px: float = 1000.0,
    width: int = 1920,
    height: int = 1080,
) -> CameraIntrinsics:
    return CameraIntrinsics(
        height_m=height_m,
        focal_length_px=focal_px,
        pitch_deg=pitch_deg,
        image_width_px=width,
        image_height_px=height,
    )


def _bbox_centred_at(cx: float, cy: float, w: float = 40.0, h: float = 20.0) -> tuple[float, float, float, float]:
    return (cx - w / 2.0, cy - h / 2.0, w, h)


def test_horizon_centre_pixel_clamps_to_max_distance() -> None:
    """A plate at the horizon (depression → 0) is clamped to ``max_distance_m``.

    The previous implementation raised ValueError, which made portrait
    handheld videos lose GPS for nearly every detection.  The contract now
    is: the projection always returns *some* coordinate; far / above-horizon
    plates land at ``max_distance_m`` ahead of the camera.
    """
    max_distance_m = 50.0
    calc = PlateGeolocationCalculator(
        _intrinsics(pitch_deg=0.0), max_distance_m=max_distance_m
    )
    pose = VehiclePose(latitude=45.0, longitude=25.0, heading_deg=0.0)
    bbox = _bbox_centred_at(960, 540)

    target_lat, target_lon = calc.project(pose, bbox)
    distance = geodesic((pose.latitude, pose.longitude), (target_lat, target_lon)).meters
    assert distance == pytest.approx(max_distance_m, rel=1e-3)


def test_above_horizon_pixel_clamps_to_max_distance() -> None:
    """A plate above the principal point (negative dv) also clamps cleanly."""
    calc = PlateGeolocationCalculator(
        _intrinsics(pitch_deg=5.0), max_distance_m=50.0
    )
    pose = VehiclePose(latitude=46.7700, longitude=23.5895, heading_deg=90.0)
    # 200 pixels above the principal point: with pitch=5° and focal=1000,
    # dv=-200 → alpha_v=-11.3° → depression=-6.3° (above horizon).
    bbox = _bbox_centred_at(960, 540 - 200)
    target_lat, target_lon = calc.project(pose, bbox)
    distance = geodesic((pose.latitude, pose.longitude), (target_lat, target_lon)).meters
    assert distance == pytest.approx(50.0, rel=1e-3)


def test_camera_below_plate_height_rejected() -> None:
    """The geometry inverts if camera height ≤ plate height — must reject early."""
    with pytest.raises(ValueError):
        PlateGeolocationCalculator(_intrinsics(height_m=0.5), plate_height_m=0.6)


def test_zero_focal_length_rejected() -> None:
    with pytest.raises(ValueError):
        PlateGeolocationCalculator(_intrinsics(focal_px=0.0))


def test_centre_pixel_with_known_pitch_yields_expected_distance() -> None:
    """A plate dead-centre (only pitch contributes) lands ΔH/tan(pitch) metres ahead.

    Heading 90° (East) means the projected GPS sits east of the police car at
    the predicted distance — geopy's geodesic gives us a ground-truth check.
    """
    pitch_deg = 10.0
    intrinsics = _intrinsics(pitch_deg=pitch_deg)
    calc = PlateGeolocationCalculator(intrinsics, plate_height_m=0.6)

    pose = VehiclePose(latitude=46.7700, longitude=23.5895, heading_deg=90.0)
    bbox = _bbox_centred_at(intrinsics.image_width_px / 2, intrinsics.image_height_px / 2)

    expected_distance = (intrinsics.height_m - 0.6) / math.tan(math.radians(pitch_deg))
    expected_point = geodesic(meters=expected_distance).destination(
        (pose.latitude, pose.longitude), bearing=90.0
    )

    target_lat, target_lon = calc.project(pose, bbox)

    # Tolerance: ~1 cm in degrees at this latitude.  The pinhole math is
    # deterministic so we can be tight.
    assert target_lat == pytest.approx(expected_point.latitude, abs=1e-7)
    assert target_lon == pytest.approx(expected_point.longitude, abs=1e-7)


def test_horizontal_pixel_offset_rotates_azimuth() -> None:
    """A plate offset to the right of the principal point increases the bearing.

    With heading 0° (North), shifting the bbox right means the projected GPS
    point should sit east-of-north — i.e. final latitude increases less than
    a pure-north projection, and longitude increases.
    """
    intrinsics = _intrinsics(pitch_deg=10.0)
    calc = PlateGeolocationCalculator(intrinsics)
    pose = VehiclePose(latitude=46.7700, longitude=23.5895, heading_deg=0.0)

    centre_bbox = _bbox_centred_at(intrinsics.image_width_px / 2, intrinsics.image_height_px / 2)
    right_bbox = _bbox_centred_at(intrinsics.image_width_px / 2 + 200, intrinsics.image_height_px / 2)

    centre_lat, centre_lon = calc.project(pose, centre_bbox)
    right_lat, right_lon = calc.project(pose, right_bbox)

    assert right_lon > centre_lon  # rotated eastward
    assert right_lat < centre_lat  # less northward component than centre


def test_lower_pixel_means_closer_distance() -> None:
    """Lower in the frame ⇒ steeper depression ⇒ shorter ground distance.

    We compare the great-circle distance from the police car for two bboxes
    sharing the same horizontal centre but differing vertically.
    """
    intrinsics = _intrinsics(pitch_deg=5.0)
    calc = PlateGeolocationCalculator(intrinsics)
    pose = VehiclePose(latitude=46.7700, longitude=23.5895, heading_deg=0.0)

    higher_bbox = _bbox_centred_at(intrinsics.image_width_px / 2, intrinsics.image_height_px / 2 + 50)
    lower_bbox = _bbox_centred_at(intrinsics.image_width_px / 2, intrinsics.image_height_px / 2 + 200)

    higher = calc.project(pose, higher_bbox)
    lower = calc.project(pose, lower_bbox)

    d_higher = geodesic((pose.latitude, pose.longitude), higher).meters
    d_lower = geodesic((pose.latitude, pose.longitude), lower).meters

    assert d_lower < d_higher


def test_negative_or_zero_bbox_dimensions_rejected() -> None:
    calc = PlateGeolocationCalculator(_intrinsics(pitch_deg=10.0))
    pose = VehiclePose(latitude=0.0, longitude=0.0, heading_deg=0.0)
    with pytest.raises(ValueError):
        calc.project(pose, (100.0, 100.0, 0.0, 20.0))
    with pytest.raises(ValueError):
        calc.project(pose, (100.0, 100.0, 40.0, -1.0))
