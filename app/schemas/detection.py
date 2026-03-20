import os
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, computed_field


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

    created_at: datetime
    plate_id: uuid.UUID | None = None

    model_config = ConfigDict(from_attributes=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def crop_image_url(self) -> str | None:
        """Return a fully-rooted URL for the crop image (e.g. /crops/abc.jpg).

        Handles three possible storage formats for crop_image_path:
          - None            → None
          - "abc.jpg"       → "/crops/abc.jpg"
          - "crops/abc.jpg" → "/crops/abc.jpg"
        """
        if self.crop_image_path is None:
            return None
        filename = os.path.basename(self.crop_image_path)
        return f"/crops/{filename}"


class SessionOut(BaseModel):
    id: uuid.UUID
    source_filename: str
    status: str
    fps_target: float | None = None
    frames_processed: int
    error_message: str | None = None
    created_at: datetime
    ended_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)
