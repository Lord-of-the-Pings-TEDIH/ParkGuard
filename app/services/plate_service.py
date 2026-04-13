"""Atomic upsert for plates (Postgres + SQLite)."""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.detection import Plate


async def upsert_plate(
    normalized_text: str,
    county_code: str | None,
    county_name: str | None,
    plate_type: str | None,
    db: AsyncSession,
) -> Plate:
    """Insert a plate or bump ``seen_count`` / ``last_seen_at`` atomically.

    Single ``INSERT ... ON CONFLICT (normalized_text) DO UPDATE`` round-trip.
    Picks the Postgres or SQLite dialect module based on the bound engine so
    the same call works in both prod and tests.
    """
    dialect_name = db.bind.dialect.name if db.bind is not None else "postgresql"
    insert = sqlite.insert if dialect_name == "sqlite" else postgresql.insert

    stmt = insert(Plate).values(
        normalized_text=normalized_text,
        county_code=county_code,
        county_name=county_name,
        plate_type=plate_type,
        first_seen_at=func.now(),
        last_seen_at=func.now(),
        seen_count=1,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[Plate.normalized_text],
        set_={
            "last_seen_at": func.now(),
            "seen_count": Plate.seen_count + 1,
        },
    ).returning(Plate)

    result = await db.execute(stmt)
    await db.flush()
    return result.scalar_one()
