from datetime import datetime

from sqlalchemy import JSON, String, Boolean, ForeignKey, DateTime, Float, Index, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class ParkingSpot(Base):
    __tablename__ = "parking_spots"

    id: Mapped[int] = mapped_column(primary_key=True)
    spot_label: Mapped[str] = mapped_column(String(20), nullable=False)
    parking_lot_id: Mapped[int] = mapped_column(
        ForeignKey("parking_lots.id"), nullable=False
    )
    # Sequential position within the lot.  Used for the X / X+1 / X+2
    # neighbouring-spot analysis from Darius's spec.  NULL means unordered.
    spot_sequence: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_occupied: Mapped[bool] = mapped_column(Boolean, default=False)

    # Mobile-LPR geolocation: GPS centre of the spot + the plate that may park here.
    # Nullable so existing rows / public spots without a reservation stay valid.
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    assigned_plate: Mapped[str | None] = mapped_column(
        String(20), nullable=True, index=True
    )
    # Additional plates permitted at this spot (e.g. a spouse's car).
    # Stored as a JSON array of compact plate strings.
    allowed_plates: Mapped[list | None] = mapped_column(JSON, nullable=True)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    parking_lot: Mapped["ParkingLot"] = relationship(back_populates="spots")

    __table_args__ = (
        # Composite index used by the bounding-box prefilter in
        # app/services/parking_validation.py.
        Index("ix_parking_spots_lat_lon", "latitude", "longitude"),
    )
