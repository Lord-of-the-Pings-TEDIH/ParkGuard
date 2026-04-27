from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class SpotOccupancyEvent(Base):
    """One patrol detection of a foreign plate at a reserved spot."""

    __tablename__ = "spot_occupancy_events"
    __table_args__ = (
        # Used by the session-deduplication check in record_occupancy_event.
        Index("ix_occ_event_spot_plate_session", "spot_id", "plate_text", "session_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    spot_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("parking_spots.id", ondelete="CASCADE"), index=True
    )
    plate_text: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    detection_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("detections.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # 1.5 for night / weekend, 1.0 otherwise — stored so score replay is cheap.
    time_weight: Mapped[float] = mapped_column(
        Float, nullable=False, default=1.0, server_default="1.0"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class SpotOccupancyRecord(Base):
    """Aggregated suspicion record per (spot, foreign-plate) pair."""

    __tablename__ = "spot_occupancy_records"
    __table_args__ = (UniqueConstraint("spot_id", "plate_text"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    spot_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("parking_spots.id", ondelete="CASCADE"), index=True
    )
    plate_text: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    event_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    # Distinct calendar days with at least one event in the last 30 days.
    distinct_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    accumulated_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, server_default="0"
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    is_flagged: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
