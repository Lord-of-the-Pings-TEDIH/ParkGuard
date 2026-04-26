"""Detection-pipeline ORM models: Session, Frame, Detection, Plate."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Session(Base):
    """A single video-processing session (one source file / stream)."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="pending"
    )
    fps_target: Mapped[float | None] = mapped_column(Float, nullable=True)
    frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    total_frames: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Mobile-LPR pose: GPS position and heading of the police car at capture time.
    # All three must be set together to enable spot validation; otherwise the
    # geolocation step is skipped and the session behaves like a static camera.
    gps_latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    gps_longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    gps_heading_deg: Mapped[float | None] = mapped_column(Float, nullable=True)

    # GPS provenance + quality so the UI can show where the pose came from
    # ("explicit"/"video_metadata"/"config_default") and any soft-validation
    # warnings raised while resolving it.  ``gps_accuracy_m`` is the
    # horizontal-accuracy estimate in metres (when available from the video
    # container — iOS 11+).  All fields nullable for static-camera sessions.
    gps_source: Mapped[str | None] = mapped_column(String(30), nullable=True)
    gps_accuracy_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    gps_validation_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    gps_validation_warnings: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    frames: Mapped[list["Frame"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Frame(Base):
    """A single frame extracted from a session's video source."""

    __tablename__ = "frames"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False, index=True
    )
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    pts_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    session: Mapped["Session"] = relationship(back_populates="frames")
    detections: Mapped[list["Detection"]] = relationship(
        back_populates="frame", cascade="all, delete-orphan"
    )


class Detection(Base):
    """A single licence-plate detection within a frame."""

    __tablename__ = "detections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    frame_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("frames.id"), nullable=False, index=True
    )

    # Bounding box
    bbox_x: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_w: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_h: Mapped[float] = mapped_column(Float, nullable=False)

    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # OCR results
    ocr_raw_text: Mapped[str | None] = mapped_column(String(50), nullable=True)
    ocr_normalized_text: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )
    is_valid_ro_plate: Mapped[bool] = mapped_column(
        Boolean, default=False
    )

    crop_image_path: Mapped[str | None] = mapped_column(
        String(500), nullable=True
    )

    # Ticketing
    ticket_status: Mapped[str | None] = mapped_column(
        String(30), nullable=True
    )
    ticket_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Mobile-LPR projection result.  Populated when the parent Session has GPS
    # pose set; otherwise the values stay NULL and the frontend hides the spot
    # match badge.
    target_latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    spot_match_status: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )
    target_distance_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    matched_spot_id: Mapped[int | None] = mapped_column(
        ForeignKey("parking_spots.id"), nullable=True, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    frame: Mapped["Frame"] = relationship(back_populates="detections")
    plate: Mapped["Plate | None"] = relationship(back_populates="detections")

    # FK to plates (by normalized text)
    plate_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("plates.id"), nullable=True, index=True
    )


class Plate(Base):
    """A unique Romanian licence plate observed across sessions."""

    __tablename__ = "plates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    normalized_text: Mapped[str] = mapped_column(
        String(20), unique=True, nullable=False
    )
    county_code: Mapped[str | None] = mapped_column(String(5), nullable=True)
    county_name: Mapped[str | None] = mapped_column(String(50), nullable=True)
    plate_type: Mapped[str | None] = mapped_column(String(30), nullable=True)

    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    seen_count: Mapped[int] = mapped_column(Integer, default=1)

    # Relationships
    detections: Mapped[list["Detection"]] = relationship(back_populates="plate")
