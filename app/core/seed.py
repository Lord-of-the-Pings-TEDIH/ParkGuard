import logging
from datetime import datetime, timedelta, date, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.parking import ParkingZone, ParkingTicket, ParkingSubscription
from app.models.parking_lot import ParkingLot
from app.models.parking_spot import ParkingSpot

logger = logging.getLogger(__name__)


# Residential parking demo: 10 reserved spots placed at the GPS coordinates
# where the bundled video1.MOV was actually filmed (Sălaj county, lat≈47.258,
# lon≈23.254).  Each spot has an ``assigned_plate`` for a fictional resident
# that is *not* one of the plates detected in video1, so every car visible
# in that video naturally trips the WRONG_PLATE branch of the validator and
# feeds the suspicious-occupancy scoring engine.
#
# Spot positions are spread along latitude in ~3 m steps (delta 0.00003°)
# so each plate's projected coordinate lands closest to a single owner.
# Spaced tighter than the 40 m search radius on purpose: it is realistic for
# a residential block, and the haversine refinement in
# validate_parking_at_location picks the nearest spot deterministically.
_DEMO_PARKING_LOT_NAME = "Bloc Rezidențial — Strada Demo"
_DEMO_PARKING_LOT_LOCATION = "Zalău, jud. Sălaj"
_DEMO_SPOTS: list[tuple[str, float, float, str | None]] = [
    # (label, latitude, longitude, assigned_plate)
    ("A-01", 47.258250, 23.254340, "MM 88 OWN"),
    ("A-02", 47.258280, 23.254340, "AB 12 ION"),
    ("A-03", 47.258310, 23.254340, "BV 22 POP"),
    ("A-04", 47.258340, 23.254340, "SB 33 STN"),
    ("A-05", 47.258370, 23.254340, "B 99 GRG"),
    ("A-06", 47.258400, 23.254340, "TM 11 RDU"),
    ("A-07", 47.258430, 23.254340, "CT 44 ELN"),
    ("A-08", 47.258460, 23.254340, "AR 55 VLA"),
    ("A-09", 47.258490, 23.254340, "DJ 66 ANA"),
    ("A-10", 47.258520, 23.254340, "GL 77 MIH"),
]

async def _seed_residential_lot(session: AsyncSession) -> None:
    """Idempotent residential-lot seeding — runs every startup.

    Decoupled from the zone/ticket guard above so the lot can be re-created
    after a partial wipe (DELETE FROM parking_spots/parking_lots) without
    needing to drop the whole schema and reprocess uploaded videos.  Uses
    the lot name as the identity key — if a row with the same name exists
    we touch the spot list to match ``_DEMO_SPOTS``.
    """
    existing_lot = (
        await session.execute(
            select(ParkingLot).where(ParkingLot.name == _DEMO_PARKING_LOT_NAME)
        )
    ).scalar_one_or_none()

    if existing_lot is None:
        lot = ParkingLot(
            name=_DEMO_PARKING_LOT_NAME,
            location=_DEMO_PARKING_LOT_LOCATION,
            total_spots=len(_DEMO_SPOTS),
        )
        session.add(lot)
        await session.flush()
    else:
        lot = existing_lot

    existing_spots = {
        s.spot_label: s
        for s in (
            await session.execute(
                select(ParkingSpot).where(ParkingSpot.parking_lot_id == lot.id)
            )
        ).scalars().all()
    }

    for label, lat, lon, plate in _DEMO_SPOTS:
        spot = existing_spots.get(label)
        if spot is None:
            session.add(
                ParkingSpot(
                    spot_label=label,
                    parking_lot_id=lot.id,
                    latitude=lat,
                    longitude=lon,
                    assigned_plate=plate,
                )
            )
        else:
            # Refresh in place — lets us tweak _DEMO_SPOTS coords/owners
            # without a manual migration.
            spot.latitude = lat
            spot.longitude = lon
            spot.assigned_plate = plate

    await session.commit()


async def seed_db(session: AsyncSession) -> None:
    """Seed the database with initial parking data if it is empty."""
    # Residential lot is idempotent and runs every startup.  See the docstring
    # of _seed_residential_lot for why it isn't gated on the zone check below.
    await _seed_residential_lot(session)

    # Check if we already have parking zones/tickets
    result = await session.execute(select(ParkingZone).limit(1))
    if result.scalar_one_or_none() is not None:
        logger.info("Database already seeded (zones present). Skipping ticket seed.")
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

    await session.commit()
    logger.info(
        "Database successfully seeded: 10 plate fixtures + %d demo parking spots.",
        len(_DEMO_SPOTS),
    )
