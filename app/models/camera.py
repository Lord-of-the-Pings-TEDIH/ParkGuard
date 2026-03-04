from datetime import datetime

from sqlalchemy import String, Boolean, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    stream_url: Mapped[str] = mapped_column(String(500), nullable=False)
    parking_lot_id: Mapped[int] = mapped_column(
        ForeignKey("parking_lots.id"), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    parking_lot: Mapped["ParkingLot"] = relationship(back_populates="cameras")  
    detections: Mapped[list["Detection"]] = relationship(back_populates="camera") 
