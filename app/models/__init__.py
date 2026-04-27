"""ParkGuard ORM models – import every model so Base.metadata knows all tables."""

from app.models.base import Base
from app.models.role import Role
from app.models.user import User
from app.models.parking_lot import ParkingLot
from app.models.camera import Camera
from app.models.parking_spot import ParkingSpot
from app.models.vehicle import Vehicle
from app.models.detection import Detection, Frame, Plate, Session
from app.models.alert import Alert
from app.models.parking import (
    ParkingZone,
    ParkingTicket,
    ParkingSubscription,
    TicketCheck,
)
from app.models.occupancy import SpotOccupancyEvent, SpotOccupancyRecord

__all__ = [
    "Base",
    "Role",
    "User",
    "ParkingLot",
    "Camera",
    "ParkingSpot",
    "Vehicle",
    "Session",
    "Frame",
    "Detection",
    "Plate",
    "Alert",
    "ParkingZone",
    "ParkingTicket",
    "ParkingSubscription",
    "TicketCheck",
    "SpotOccupancyEvent",
    "SpotOccupancyRecord",
]
