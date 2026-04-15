"""End-to-end worker test against seeded parking data.

A 10s synthetic video is processed by ``process_video`` with stubbed
detector + OCR. The OCR alternates between two known plates:

* ``B11AAA`` — has an *active* ticket (from seed)
* ``B22BBB`` — unknown plate, no coverage

We rely on the real DB writes, the real ``lookup_ticket`` and the real
crop-saving code path.
"""

from __future__ import annotations

import asyncio
import itertools
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncGenerator

import cv2
import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import BigInteger, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import JSONB, UUID

@compiles(JSONB, "sqlite")
def _compile_jsonb(type_, compiler, **kw):
    return "TEXT"

@compiles(UUID, "sqlite")
def _compile_uuid(type_, compiler, **kw):
    return "CHAR(32)"

@compiles(BigInteger, "sqlite")
def _compile_bigint(type_, compiler, **kw):
    return "INTEGER"

from app.models.base import Base
from app.models.detection import Detection, Session as SessionRow
from app.models.parking import ParkingTicket, ParkingZone, TicketCheck
from app.pipeline import worker

UTC = timezone.utc
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)


@pytest_asyncio.fixture(scope="module", autouse=True)
async def _setup_database():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    async with TestingSessionLocal() as session:
        yield session


def _db_factory():
    return TestingSessionLocal()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_video(path: Path, seconds: int, fps: int = 25) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), "VideoWriter failed (codec missing?)"
    try:
        for i in range(seconds * fps):
            frame = np.full((h, w, 3), 30, dtype=np.uint8)
            x = (i * 4) % (w - 60)
            frame[100:140, x : x + 60] = 230  # bright "plate-like" rectangle
            writer.write(frame)
    finally:
        writer.release()


class _StubDetector:
    """Returns a single bbox per frame so we get one detection per frame."""

    def __init__(self, *_a, **_kw):
        pass

    def detect(self, frame, frame_index: int = 0):
        h, w = frame.shape[:2]
        return [{"bbox": (40, 100, 60, 40), "confidence": 0.9}]

    @staticmethod
    def crop_plate(frame, bbox, padding: int = 4):
        x, y, w, h = bbox
        return frame[y : y + h, x : x + w]


def _make_alternating_reader():
    """Stub PlateReader: alternate between B11AAA (active) and B22BBB (unknown)."""
    plates = itertools.cycle(["B11AAA", "B22BBB"])

    class _StubReader:
        def __init__(self, *_a, **_kw):
            pass

        def read_plate(self, crop):
            return next(plates), 0.95

    return _StubReader


async def _seed_parking(db: AsyncSession) -> None:
    zone = ParkingZone(
        name="Downtown", code="DT-01", grace_period_min=15, is_active=True
    )
    db.add(zone)
    await db.flush()

    now = datetime.now(UTC)
    db.add(
        ParkingTicket(
            plate_text="B11AAA",
            zone_id=zone.id,
            valid_from=now - timedelta(hours=1),
            valid_until=now + timedelta(hours=1),
            ticket_ref="TKT-ACTIVE",
            payment_method="SMS",
        )
    )
    await db.commit()


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_worker_end_to_end_with_seeded_plates(
    db: AsyncSession, tmp_path, monkeypatch
):
    await _seed_parking(db)

    video = tmp_path / "ten.mp4"
    _make_video(video, seconds=10, fps=25)

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    monkeypatch.setattr("app.pipeline.worker.settings.CROPS_DIR", str(crops_dir))
    monkeypatch.setattr("app.pipeline.worker.PlateDetector", _StubDetector)
    monkeypatch.setattr("app.pipeline.worker.PlateReader", _make_alternating_reader())

    session_id = uuid.uuid4()
    db.add(SessionRow(id=session_id, source_filename="ten.mp4", status="pending"))
    await db.commit()

    worker.PROCESSING_SEMAPHORE = asyncio.Semaphore(1)

    await worker.process_video(
        session_id=session_id,
        video_path=str(video),
        fps_target=5,
        db_factory=_db_factory,
    )

    async with TestingSessionLocal() as fresh:
        # 1. Session marked done
        row = await fresh.get(SessionRow, session_id)
        assert row is not None
        assert row.status == "done", f"got {row.status!r} ({row.error_message!r})"
        assert row.frames_processed > 0

        # 2. Detections table populated
        det_count = await fresh.scalar(select(func.count()).select_from(Detection))
        assert det_count and det_count > 0

        # 3. At least one detection with ticket_status='active'
        active_count = await fresh.scalar(
            select(func.count())
            .select_from(Detection)
            .where(Detection.ticket_status == "active")
        )
        assert active_count and active_count >= 1

        # 4. ticket_checks audit rows exist
        check_count = await fresh.scalar(select(func.count()).select_from(TicketCheck))
        assert check_count and check_count > 0

        # 5. Crop image files written
        crop_files = list(crops_dir.glob("*.jpg"))
        assert len(crop_files) > 0
