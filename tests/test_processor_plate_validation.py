from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import BigInteger, select
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles


@compiles(JSONB, "sqlite")
def compile_jsonb(type_, compiler, **kw):
    return "TEXT"


@compiles(UUID, "sqlite")
def compile_uuid(type_, compiler, **kw):
    return "CHAR(32)"


@compiles(BigInteger, "sqlite")
def compile_bigint(type_, compiler, **kw):
    return "INTEGER"


from app.models.base import Base
from app.models.detection import Detection, Frame, Plate, Session
import app.pipeline.processor as processor

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)


@pytest_asyncio.fixture(scope="module", autouse=True)
async def setup_database():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncGenerator[AsyncSession, None]:
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()


class FakeDetector:
    def __init__(self, model_path: str, conf_threshold: float) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[dict]:
        _ = frame, frame_index
        return [{"bbox": (1, 1, 4, 2), "confidence": 0.93}]

    @staticmethod
    def crop_plate(frame: np.ndarray, bbox: tuple[int, int, int, int], padding: int = 0) -> np.ndarray:
        x, y, w, h = bbox
        _ = padding
        return frame[y : y + h, x : x + w]


def _single_frame(*_args, **_kwargs) -> list[tuple[int, int, np.ndarray]]:
    frame = np.zeros((24, 48, 3), dtype=np.uint8)
    frame[:, ::2] = 255
    return [(0, 0, frame)]


async def _create_session_and_video(db: AsyncSession, uploads_dir: Path) -> Session:
    session_obj = Session(source_filename="sample.mp4")
    db.add(session_obj)
    await db.flush()

    video_path = uploads_dir / f"{session_obj.id}_{session_obj.source_filename}"
    video_path.write_bytes(b"fake-video")
    return session_obj


def _configure_processor(
    monkeypatch: pytest.MonkeyPatch,
    uploads_dir: Path,
    crops_dir: Path,
    lookup_mock: AsyncMock,
) -> None:
    monkeypatch.setattr(processor.settings, "UPLOAD_DIR", str(uploads_dir), raising=False)
    monkeypatch.setattr(processor.settings, "CROPS_DIR", str(crops_dir), raising=False)
    monkeypatch.setattr(processor.settings, "OCR_MIN_SHARPNESS", 0.0, raising=False)
    monkeypatch.setattr(processor.settings, "OCR_ANGLES", "0", raising=False)
    monkeypatch.setattr(processor.settings, "OCR_MIN_CONF", 0.1, raising=False)
    monkeypatch.setattr(processor.settings, "MIN_TRACK_VOTES", 1, raising=False)
    monkeypatch.setattr(processor, "PlateDetector", FakeDetector)
    monkeypatch.setattr(processor, "extract_frames", _single_frame)
    monkeypatch.setattr(processor, "lookup_ticket", lookup_mock)


@pytest.mark.asyncio
async def test_invalid_plate_does_not_trigger_lookup(
    db: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads_dir = tmp_path / "uploads"
    crops_dir = tmp_path / "crops"
    uploads_dir.mkdir()
    crops_dir.mkdir()

    lookup_mock = AsyncMock(return_value=("unknown", None, None, None))
    _configure_processor(monkeypatch, uploads_dir, crops_dir, lookup_mock)

    initial_plate_count = len((await db.execute(select(Plate.id))).scalars().all())
    session_obj = await _create_session_and_video(db, uploads_dir)

    await processor.process_session(
        session_obj.id,
        db,
        ocr=lambda _crop: "GARBAGE",
        zone_id=1,
    )

    detection = (
        await db.execute(
            select(Detection)
            .join(Frame, Detection.frame_id == Frame.id)
            .where(Frame.session_id == session_obj.id)
        )
    ).scalars().one()

    assert detection.ocr_raw_text == "GARBAGE"
    assert detection.ocr_normalized_text == "INVALID: județ invalid"
    assert detection.is_valid_ro_plate is False
    assert detection.plate_id is None
    assert detection.ticket_status is None
    assert lookup_mock.await_count == 0

    final_plate_count = len((await db.execute(select(Plate.id))).scalars().all())
    assert final_plate_count == initial_plate_count


@pytest.mark.asyncio
async def test_valid_normalized_plate_triggers_lookup(
    db: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads_dir = tmp_path / "uploads"
    crops_dir = tmp_path / "crops"
    uploads_dir.mkdir()
    crops_dir.mkdir()

    lookup_mock = AsyncMock(return_value=("active", None, 123, None))
    _configure_processor(monkeypatch, uploads_dir, crops_dir, lookup_mock)

    initial_plate_count = len((await db.execute(select(Plate.id))).scalars().all())
    session_obj = await _create_session_and_video(db, uploads_dir)

    await processor.process_session(
        session_obj.id,
        db,
        ocr=lambda _crop: "CJ05XY0",
        zone_id=1,
    )

    detection = (
        await db.execute(
            select(Detection)
            .join(Frame, Detection.frame_id == Frame.id)
            .where(Frame.session_id == session_obj.id)
        )
    ).scalars().one()

    lookup_mock.assert_awaited_once()
    assert lookup_mock.await_args.kwargs["plate_text"] == "CJ05XYO"
    assert lookup_mock.await_args.kwargs["detection_id"] == detection.id

    assert detection.ocr_raw_text == "CJ05XY0"
    assert detection.ocr_normalized_text == "CJ 05 XYO"
    assert detection.is_valid_ro_plate is True
    assert detection.ticket_status == "active"
    assert detection.plate_id is not None

    final_plate_count = len((await db.execute(select(Plate.id))).scalars().all())
    assert final_plate_count == initial_plate_count + 1
