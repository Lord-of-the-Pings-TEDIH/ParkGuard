"""Project a YOLO bounding box into real-world GPS coordinates.

Mobile-LPR scenario: a camera mounted on a moving police car detects a parked
plate.  Given the car's GPS pose and the camera's intrinsics, we estimate
where on the ground the plate sits, so the parking-validation service can
look up the closest registered spot.

The model is a flat-ground pinhole camera:
  - the world is a single horizontal plane,
  - the camera has zero roll,
  - the principal point sits at the image centre.

These assumptions are accurate enough for distances < ~15 m, which matches
the operational range of a side-mounted ALPR camera in a parking lane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from geopy.distance import geodesic


# A Romanian plate sits roughly 0.5–0.7 m above the ground; 0.6 m is the
# accepted average for the ground-projection geometry.
DEFAULT_PLATE_HEIGHT_M: float = 0.6

# Far-distance clamp for the ground projection.  When a detected plate sits
# at or above the geometric horizon (typical of distant cars in handheld
# portrait video), the ray-to-ground intersection is mathematically at
# infinity.  Rather than refusing to project, we cap the distance at this
# value so every detected plate still gets *some* GPS coordinate.  The
# trade-off is reduced accuracy for far cars, which is acceptable since the
# downstream parking-spot lookup uses a small radius and will simply return
# NO_SPOT_FOUND for these.
DEFAULT_MAX_DISTANCE_M: float = 50.0


@dataclass(frozen=True)
class CameraIntrinsics:
    """Static, calibration-time properties of the police-car camera."""

    height_m: float                # Z_camera, height above ground in metres
    focal_length_px: float         # f, focal length expressed in pixels
    pitch_deg: float               # downward tilt from horizontal (positive = nose-down)
    image_width_px: int
    image_height_px: int


@dataclass(frozen=True)
class VehiclePose:
    """Instantaneous GPS pose of the police car."""

    latitude: float
    longitude: float
    heading_deg: float             # 0 = North, 90 = East (true bearing, degrees)


class PlateGeolocationCalculator:
    """Convert a plate bounding box in pixels to a GPS coordinate on the ground."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        plate_height_m: float = DEFAULT_PLATE_HEIGHT_M,
        max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
    ) -> None:
        if intrinsics.focal_length_px <= 0:
            raise ValueError("focal_length_px must be > 0")
        if intrinsics.height_m <= plate_height_m:
            # The geometry collapses (or inverts) if the camera is at or below
            # the plate's height — the depression ray would never hit the ground.
            raise ValueError(
                "camera height must exceed the plate height "
                f"({intrinsics.height_m} <= {plate_height_m})"
            )
        if max_distance_m <= 0:
            raise ValueError("max_distance_m must be > 0")

        self.intrinsics = intrinsics
        self.plate_height_m = plate_height_m
        self.max_distance_m = max_distance_m
        self._delta_height_m: float = intrinsics.height_m - plate_height_m
        # Minimum depression angle for which we still treat the projection as
        # finite.  Below this the plate is "near or above the horizon" and we
        # clamp distance to ``max_distance_m`` instead of failing.
        self._min_depression_rad: float = math.atan(
            self._delta_height_m / max_distance_m
        )

    def project(
        self,
        pose: VehiclePose,
        bbox_xywh: Tuple[float, float, float, float],
    ) -> Tuple[float, float]:
        """Return ``(latitude, longitude)`` for the plate centre on the ground.

        ``bbox_xywh`` is the YOLO output in pixel space:
        ``(x_top_left, y_top_left, width, height)``.
        """
        distance_m, azimuth_deg = self._compute_distance_and_bearing(pose, bbox_xywh)

        target = geodesic(meters=distance_m).destination(
            (pose.latitude, pose.longitude),
            bearing=azimuth_deg,
        )
        return (target.latitude, target.longitude)

    def _compute_distance_and_bearing(
        self,
        pose: VehiclePose,
        bbox_xywh: Tuple[float, float, float, float],
    ) -> Tuple[float, float]:
        x, y, w, h = bbox_xywh
        if w <= 0 or h <= 0:
            raise ValueError(f"bbox width/height must be positive, got {(w, h)}")

        cx_pixel = x + w / 2.0
        cy_pixel = y + h / 2.0

        principal_x = self.intrinsics.image_width_px / 2.0
        principal_y = self.intrinsics.image_height_px / 2.0
        f = self.intrinsics.focal_length_px

        # Pixel offsets from the optical axis. Positive dv means the plate is
        # below the principal point in the image, i.e. closer to the ground.
        du = cx_pixel - principal_x
        dv = cy_pixel - principal_y

        # Pinhole angles. atan2 with the focal length keeps the formula
        # numerically stable for off-centre detections.
        alpha_h_rad = math.atan2(du, f)
        alpha_v_rad = math.atan2(dv, f)

        # Total angle below the horizontal: camera tilt + per-pixel tilt.
        # Plates near or above the geometric horizon (typical for distant
        # cars in a handheld portrait shot) are clamped to a configurable
        # maximum distance so every plate still produces a GPS estimate.
        depression_rad = math.radians(self.intrinsics.pitch_deg) + alpha_v_rad
        if depression_rad < self._min_depression_rad:
            depression_rad = self._min_depression_rad

        distance_m = self._delta_height_m / math.tan(depression_rad)

        # True-north azimuth: vehicle heading + horizontal pixel offset.
        azimuth_deg = (pose.heading_deg + math.degrees(alpha_h_rad)) % 360.0

        return distance_m, azimuth_deg
