from datetime import datetime

from sqlalchemy import String, Boolean, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class ParkingSpot(Base):
    __tablename__ = "parking_spots"

    id: Mapped[int] = mapped_column(primary_key=True)
    spot_label: Mapped[str] = mapped_column(String(20), nullable=False)
    parking_lot_id: Mapped[int] = mapped_column(
        ForeignKey("parking_lots.id"), nullable=False
    )
    is_occupied: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    parking_lot: Mapped["ParkingLot"] = relationship(back_populates="spots")  
    detections: Mapped[list["Detection"]] = relationship(back_populates="parking_spot")  
