from datetime import datetime

from sqlalchemy import Float, String, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id"), nullable=False
    )
    parking_spot_id: Mapped[int | None] = mapped_column(
        ForeignKey("parking_spots.id"), nullable=True
    )
    vehicle_id: Mapped[int | None] = mapped_column(
        ForeignKey("vehicles.id"), nullable=True
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    image_path: Mapped[str | None] = mapped_column(String(500))
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    camera: Mapped["Camera"] = relationship(back_populates="detections")  
    parking_spot: Mapped["ParkingSpot | None"] = relationship(back_populates="detections")  
    vehicle: Mapped["Vehicle | None"] = relationship(back_populates="detections")  
    alerts: Mapped[list["Alert"]] = relationship(back_populates="detection")  
