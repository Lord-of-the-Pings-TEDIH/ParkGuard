import os
import re
import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, computed_field

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")


class PlateOut(BaseModel):
    id: uuid.UUID
    normalized_text: str
    county_code: str | None = None
    county_name: str | None = None
    plate_type: str | None = None
    first_seen_at: datetime
    last_seen_at: datetime
    seen_count: int

    model_config = ConfigDict(from_attributes=True)


class DetectionOut(BaseModel):
    id: uuid.UUID
    frame_id: uuid.UUID

    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    detection_confidence: float

    ocr_raw_text: str | None = None
    ocr_normalized_text: str | None = None
    is_valid_ro_plate: bool

    crop_image_path: str | None = None

    ticket_status: str | None = None
    ticket_expires_at: datetime | None = None

    target_latitude: float | None = None
    target_longitude: float | None = None
    spot_match_status: Literal["MATCH", "WRONG_PLATE", "NO_SPOT_FOUND"] | None = None
    target_distance_m: float | None = None
    matched_spot_id: int | None = None

    created_at: datetime
    plate_id: uuid.UUID | None = None

    model_config = ConfigDict(from_attributes=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def crop_image_url(self) -> str | None:
        """Return a fully-rooted URL for the crop image.

        Handles three possible storage formats for crop_image_path:
          - None            → None
          - "abc.jpg"       → "/api/crops/abc.jpg"
          - "crops/abc.jpg" → "/api/crops/abc.jpg"
          - "sid/abc.jpg"   → "/api/crops/sid/abc.jpg"
        """
        if self.crop_image_path is None:
            return None
        raw_path = self.crop_image_path.strip().replace("\\", "/")
        if not raw_path:
            return None
        if raw_path.startswith("crops/"):
            raw_path = raw_path[len("crops/") :]

        normalized = os.path.normpath(raw_path).replace("\\", "/")
        if normalized.startswith("../") or normalized == ".." or normalized.startswith("/"):
            normalized = os.path.basename(raw_path)

        return f"/api/crops/{normalized}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def voting_tag(self) -> Literal["final", "not_final"]:
        """Return voting stage tag for this detection.

        * ``not_final``: track is still active in voting (no ticket lookup yet)
        * ``final``: track closed/disappeared and ticket status was resolved
        """
        return "final" if self.ticket_status is not None else "not_final"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def plate_annotation(self) -> str:
        """Stable plate annotation key used for frontend deduplication."""
        if self.ocr_normalized_text:
            normalized = self.ocr_normalized_text.strip().upper()
            if normalized and not normalized.startswith("INVALID:"):
                compact = _NON_ALNUM_RE.sub("", normalized)
                if compact:
                    return compact

        if self.ocr_raw_text:
            raw = self.ocr_raw_text.strip().upper()
            compact_raw = _NON_ALNUM_RE.sub("", raw)
            if compact_raw:
                return compact_raw

        return str(self.id)


class SessionOut(BaseModel):
    id: uuid.UUID
    source_filename: str
    status: str
    fps_target: float | None = None
    frames_processed: int
    total_frames: int | None = None
    error_message: str | None = None
    created_at: datetime
    ended_at: datetime | None = None

    gps_latitude: float | None = None
    gps_longitude: float | None = None
    gps_heading_deg: float | None = None
    gps_source: str | None = None
    gps_accuracy_m: float | None = None
    gps_validation_status: str | None = None
    # Newline-separated list of soft-validation warnings.  Stored as TEXT
    # (rather than JSONB) so the SQLite test harness doesn't need a shim,
    # and split into ``gps_warnings`` below for the frontend.
    gps_validation_warnings: str | None = None

    zone_id: int | None = None
    identified_lot_id: int | None = None

    model_config = ConfigDict(from_attributes=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def gps_warnings(self) -> list[str]:
        """Split the persisted newline-separated warning bundle into a list."""
        raw = self.gps_validation_warnings
        if not raw:
            return []
        return [line for line in raw.splitlines() if line.strip()]
