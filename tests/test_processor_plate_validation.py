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
from app.schemas.detection import DetectionOut

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


class FakeDetectorNoHits(FakeDetector):
    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[dict]:
        _ = frame, frame_index
        return []


def _single_frame(*_args, **_kwargs) -> list[tuple[int, int, np.ndarray]]:
    frame = np.zeros((24, 48, 3), dtype=np.uint8)
    frame[:, ::2] = 255
    return [(0, 0, frame)]


def _two_frames(*_args, **_kwargs) -> list[tuple[int, int, np.ndarray]]:
    frame = np.zeros((24, 48, 3), dtype=np.uint8)
    frame[:, ::2] = 255
    return [(0, 0, frame), (1, 40, frame.copy())]


def _three_frames(*_args, **_kwargs) -> list[tuple[int, int, np.ndarray]]:
    frame = np.zeros((24, 48, 3), dtype=np.uint8)
    frame[:, ::2] = 255
    return [(0, 0, frame), (1, 40, frame.copy()), (2, 80, frame.copy())]


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
    *,
    min_track_votes: int = 1,
    extract_frames_fn=_single_frame,
    detector_cls=FakeDetector,
) -> None:
    monkeypatch.setattr(processor.settings, "UPLOAD_DIR", str(uploads_dir), raising=False)
    monkeypatch.setattr(processor.settings, "CROPS_DIR", str(crops_dir), raising=False)
    monkeypatch.setattr(processor.settings, "OCR_MIN_SHARPNESS", 0.0, raising=False)
    monkeypatch.setattr(processor.settings, "OCR_ANGLES", "0", raising=False)
    monkeypatch.setattr(processor.settings, "OCR_MIN_CONF", 0.1, raising=False)
    monkeypatch.setattr(processor.settings, "MIN_TRACK_VOTES", min_track_votes, raising=False)
    monkeypatch.setattr(processor, "PlateDetector", detector_cls)
    monkeypatch.setattr(processor, "extract_frames", extract_frames_fn)
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
    detection_out = DetectionOut.model_validate(detection)
    assert detection_out.voting_tag == "not_final"
    assert detection_out.plate_annotation == "GARBAGE"
    assert lookup_mock.await_count == 0

    final_plate_count = len((await db.execute(select(Plate.id))).scalars().all())
    assert final_plate_count == initial_plate_count


@pytest.mark.asyncio
async def test_frames_without_detections_are_skipped_for_db_persistence(
    db: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads_dir = tmp_path / "uploads"
    crops_dir = tmp_path / "crops"
    uploads_dir.mkdir()
    crops_dir.mkdir()

    lookup_mock = AsyncMock(return_value=("unknown", None, None, None))
    _configure_processor(
        monkeypatch,
        uploads_dir,
        crops_dir,
        lookup_mock,
        detector_cls=FakeDetectorNoHits,
    )

    session_obj = await _create_session_and_video(db, uploads_dir)

    await processor.process_session(
        session_obj.id,
        db,
        ocr=lambda _crop: "CJ05XY0",
        zone_id=1,
    )

    await db.refresh(session_obj)
    frame_rows = (
        await db.execute(select(Frame).where(Frame.session_id == session_obj.id))
    ).scalars().all()
    detection_rows = (
        await db.execute(
            select(Detection)
            .join(Frame, Detection.frame_id == Frame.id)
            .where(Frame.session_id == session_obj.id)
        )
    ).scalars().all()

    assert session_obj.frames_processed == 1
    assert frame_rows == []
    assert detection_rows == []
    assert lookup_mock.await_count == 0


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
    detection_out = DetectionOut.model_validate(detection)
    assert detection_out.voting_tag == "final"
    assert detection_out.plate_annotation == "CJ05XYO"

    final_plate_count = len((await db.execute(select(Plate.id))).scalars().all())
    assert final_plate_count == initial_plate_count + 1


@pytest.mark.asyncio
async def test_sparse_track_finalizes_with_best_candidate_when_session_ends(
    db: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads_dir = tmp_path / "uploads"
    crops_dir = tmp_path / "crops"
    uploads_dir.mkdir()
    crops_dir.mkdir()

    lookup_mock = AsyncMock(return_value=("active", None, 123, None))
    _configure_processor(
        monkeypatch,
        uploads_dir,
        crops_dir,
        lookup_mock,
        min_track_votes=2,
    )

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

    assert detection.ocr_raw_text == "CJ05XY0"
    assert detection.ocr_normalized_text == "CJ 05 XYO"
    assert detection.is_valid_ro_plate is True
    assert detection.ticket_status == "active"
    assert detection.plate_id is not None
    assert lookup_mock.await_count == 1

    detection_out = DetectionOut.model_validate(detection)
    assert detection_out.voting_tag == "final"
    assert detection_out.plate_annotation == "CJ05XYO"


@pytest.mark.asyncio
async def test_invalid_normalized_candidate_is_ignored_in_voting(
    db: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads_dir = tmp_path / "uploads"
    crops_dir = tmp_path / "crops"
    uploads_dir.mkdir()
    crops_dir.mkdir()

    lookup_mock = AsyncMock(return_value=("active", None, 123, None))
    _configure_processor(
        monkeypatch,
        uploads_dir,
        crops_dir,
        lookup_mock,
        min_track_votes=2,
        extract_frames_fn=_two_frames,
    )

    session_obj = await _create_session_and_video(db, uploads_dir)

    ocr_outputs = iter(["CJ05XY0", "GARBAGE"])

    await processor.process_session(
        session_obj.id,
        db,
        ocr=lambda _crop: next(ocr_outputs),
        zone_id=1,
    )

    detections = (
        await db.execute(
            select(Detection)
            .join(Frame, Detection.frame_id == Frame.id)
            .where(Frame.session_id == session_obj.id)
            .order_by(Frame.frame_index.asc())
        )
    ).scalars().all()

    assert len(detections) == 2
    assert detections[0].ocr_normalized_text == "CJ 05 XYO"
    assert detections[1].ocr_normalized_text == "INVALID: județ invalid"

    # Second sample is invalid and must not increase vote count to final.
    # Track finalization fallback should still use the best valid sample.
    assert detections[0].ticket_status == "active"
    assert detections[1].ticket_status is None
    assert detections[0].plate_id is not None
    assert detections[1].plate_id is None
    assert lookup_mock.await_count == 1

    assert DetectionOut.model_validate(detections[0]).voting_tag == "final"
    assert DetectionOut.model_validate(detections[1]).voting_tag == "not_final"
    assert DetectionOut.model_validate(detections[0]).plate_annotation == "CJ05XYO"
    assert DetectionOut.model_validate(detections[1]).plate_annotation == "GARBAGE"


@pytest.mark.asyncio
async def test_sparse_track_uses_highest_confidence_plate_for_fallback(
    db: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads_dir = tmp_path / "uploads"
    crops_dir = tmp_path / "crops"
    uploads_dir.mkdir()
    crops_dir.mkdir()

    lookup_mock = AsyncMock(return_value=("active", None, 123, None))
    _configure_processor(
        monkeypatch,
        uploads_dir,
        crops_dir,
        lookup_mock,
        min_track_votes=3,
        extract_frames_fn=_two_frames,
    )

    session_obj = await _create_session_and_video(db, uploads_dir)
    ocr_outputs = iter([("CJ05XY0", 0.61), ("CJ05XYP", 0.95)])

    await processor.process_session(
        session_obj.id,
        db,
        ocr=lambda _crop: next(ocr_outputs),
        zone_id=1,
    )

    detections = (
        await db.execute(
            select(Detection)
            .join(Frame, Detection.frame_id == Frame.id)
            .where(Frame.session_id == session_obj.id)
            .order_by(Frame.frame_index.asc())
        )
    ).scalars().all()

    assert len(detections) == 2
    assert detections[0].ocr_normalized_text == "CJ 05 XYO"
    assert detections[1].ocr_normalized_text == "CJ 05 XYP"

    lookup_mock.assert_awaited_once()
    assert lookup_mock.await_args.kwargs["plate_text"] == "CJ05XYP"
    assert lookup_mock.await_args.kwargs["detection_id"] == detections[1].id

    assert detections[0].ticket_status is None
    assert detections[0].plate_id is None
    assert detections[1].ticket_status == "active"
    assert detections[1].plate_id is not None


@pytest.mark.asyncio
async def test_track_stays_not_final_while_visible_and_finalizes_after_disappear(
    db: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploads_dir = tmp_path / "uploads"
    crops_dir = tmp_path / "crops"
    uploads_dir.mkdir()
    crops_dir.mkdir()

    lookup_mock = AsyncMock(return_value=("active", None, 123, None))
    _configure_processor(
        monkeypatch,
        uploads_dir,
        crops_dir,
        lookup_mock,
        min_track_votes=1,
        extract_frames_fn=_three_frames,
    )

    session_obj = await _create_session_and_video(db, uploads_dir)

    await processor.process_session(
        session_obj.id,
        db,
        ocr=lambda _crop: "CJ05XY0",
        zone_id=1,
    )

    detections = (
        await db.execute(
            select(Detection)
            .join(Frame, Detection.frame_id == Frame.id)
            .where(Frame.session_id == session_obj.id)
            .order_by(Frame.frame_index.asc())
        )
    ).scalars().all()

    assert len(detections) == 3
    assert [d.ticket_status for d in detections] == [None, None, "active"]
    assert DetectionOut.model_validate(detections[0]).voting_tag == "not_final"
    assert DetectionOut.model_validate(detections[1]).voting_tag == "not_final"
    assert DetectionOut.model_validate(detections[2]).voting_tag == "final"
    lookup_mock.assert_awaited_once()
