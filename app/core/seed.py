import logging
from datetime import datetime, timedelta, date, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.parking import ParkingZone, ParkingTicket, ParkingSubscription
from app.models.parking_lot import ParkingLot
from app.models.parking_spot import ParkingSpot

logger = logging.getLogger(__name__)


# Mobile-LPR demo coordinates: a small block in central Cluj-Napoca.  Spots
# are spaced ~6 m apart so a 3 m search radius cleanly disambiguates them.
# Real deployments will overwrite these via the /api/parking/spots endpoint
# or a CSV import; this is just enough data to demo the feature end-to-end.
_DEMO_PARKING_LOT_NAME = "Cluj-Napoca Centru — Demo"
_DEMO_SPOTS: list[tuple[str, float, float, str | None]] = [
    # (label, latitude, longitude, assigned_plate)
    ("A-01", 46.770100, 23.589500, "B11AAA"),
    ("A-02", 46.770155, 23.589500, "B22BBB"),
    ("A-03", 46.770210, 23.589500, "CJ05XYZ"),
    ("A-04", 46.770265, 23.589500, None),  # unreserved spot
]

async def seed_db(session: AsyncSession) -> None:
    """Seed the database with initial parking data if it is empty."""
    # Check if we already have parking zones/tickets
    result = await session.execute(select(ParkingZone).limit(1))
    if result.scalar_one_or_none() is not None:
        logger.info("Database already seeded. Skipping seed.")
        return

    logger.info("Seeding database with initial parking test data...")
    
    # Create the single zone
    zone = ParkingZone(
        name="Downtown Zone",
        code="DT-01",
        grace_period_min=15,
        is_active=True
    )
    session.add(zone)
    await session.flush()
    
    now = datetime.now(timezone.utc)
    today = date.today()
    
    # 2 Active tickets
    t1 = ParkingTicket(
        plate_text="B11AAA",
        zone_id=zone.id,
        valid_from=now - timedelta(hours=1),
        valid_until=now + timedelta(hours=1),
        ticket_ref="TKT-001",
        payment_method="SMS"
    )
    t2 = ParkingTicket(
        plate_text="B22BBB",
        zone_id=zone.id,
        valid_from=now - timedelta(minutes=30),
        valid_until=now + timedelta(hours=2),
        ticket_ref="TKT-002",
        payment_method="CARD"
    )
    
    # 1 Grace period ticket 
    t3 = ParkingTicket(
        plate_text="B33CCC",
        zone_id=zone.id,
        valid_from=now - timedelta(hours=2),
        valid_until=now - timedelta(minutes=5),
        ticket_ref="TKT-003",
        payment_method="CASH"
    )
    
    # 5 Expired tickets
    t4 = ParkingTicket(
        plate_text="B44DDD",
        zone_id=zone.id,
        valid_from=now - timedelta(days=1),
        valid_until=now - timedelta(hours=2),
        ticket_ref="TKT-004",
        payment_method="SMS"
    )
    t5 = ParkingTicket(
        plate_text="B55EEE",
        zone_id=zone.id,
        valid_from=now - timedelta(hours=5),
        valid_until=now - timedelta(hours=2),
        ticket_ref="TKT-005",
        payment_method="CARD"
    )
    t6 = ParkingTicket(
        plate_text="B66FFF",
        zone_id=zone.id,
        valid_from=now - timedelta(days=3),
        valid_until=now - timedelta(days=2),
        ticket_ref="TKT-006",
        payment_method="APP"
    )
    t7 = ParkingTicket(
        plate_text="B99III",
        zone_id=zone.id,
        valid_from=now - timedelta(days=5),
        valid_until=now - timedelta(days=4),
        ticket_ref="TKT-007",
        payment_method="SMS"
    )
    t8 = ParkingTicket(
        plate_text="B00JJJ",
        zone_id=zone.id,
        valid_from=now - timedelta(days=10),
        valid_until=now - timedelta(days=9),
        ticket_ref="TKT-008",
        payment_method="APP"
    )
    
    session.add_all([t1, t2, t3, t4, t5, t6, t7, t8])
    
    # 2 Active Subscriptions
    s1 = ParkingSubscription(
        plate_text="B77GGG",
        zone_id=zone.id,
        subscription_type="MONTHLY",
        start_date=today - timedelta(days=5),
        end_date=today + timedelta(days=25),
        holder_name="John Doe",
        ref_number="SUB-001"
    )
    s2 = ParkingSubscription(
        plate_text="B88HHH",
        zone_id=zone.id,
        subscription_type="YEARLY",
        start_date=today - timedelta(days=100),
        end_date=today + timedelta(days=265),
        holder_name="Jane Smith",
        ref_number="SUB-002"
    )
    
    session.add_all([s1, s2])

    # Mobile-LPR demo data: a small parking lot with GPS-tagged spots and a
    # mix of reserved/unreserved entries.  Idempotent on its own — safe to
    # re-run because the outer guard already returned if the zone exists.
    lot = ParkingLot(
        name=_DEMO_PARKING_LOT_NAME,
        location="Cluj-Napoca, RO",
        total_spots=len(_DEMO_SPOTS),
    )
    session.add(lot)
    await session.flush()

    for label, lat, lon, plate in _DEMO_SPOTS:
        session.add(
            ParkingSpot(
                spot_label=label,
                parking_lot_id=lot.id,
                latitude=lat,
                longitude=lon,
                assigned_plate=plate,
            )
        )

    await session.commit()
    logger.info(
        "Database successfully seeded: 10 plate fixtures + %d demo parking spots.",
        len(_DEMO_SPOTS),
    )
