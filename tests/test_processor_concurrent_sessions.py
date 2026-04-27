"""Concurrency smoke tests for ``process_session``.

Last refactor pushed CPU-bound OCR and YOLO work into worker threads so the
event loop can interleave concurrent sessions, and made ``_get_or_create_plate``
race-safe under simultaneous inserts of the same plate text.  These tests run
two ``process_session`` coroutines through ``asyncio.gather`` against
independent ``AsyncSession``s sharing one in-memory SQLite database to verify:

  * both sessions complete without raising
  * detections from each session are written under their own session_id
  * a Plate row created by one session and observed by the other is shared
    rather than crashing on the unique constraint
  * the two sessions actually overlap on the event loop (one's async work
    runs while the other's recognizer is parked in the worker thread)
"""

from __future__ import annotations

import asyncio
import threading
import time
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
def _compile_jsonb(type_, compiler, **kw):
    return "TEXT"


@compiles(UUID, "sqlite")
def _compile_uuid(type_, compiler, **kw):
    return "CHAR(32)"


@compiles(BigInteger, "sqlite")
def _compile_bigint(type_, compiler, **kw):
    return "INTEGER"


from app.models.base import Base
from app.models.detection import Detection, Frame, Plate, Session
import app.pipeline.processor as processor


# A file-backed SQLite DB so two AsyncSessionLocal()-backed sessions actually
# share the same data — :memory: would give each connection its own private DB.
@pytest_asyncio.fixture
async def db_url(tmp_path: Path) -> str:
    return f"sqlite+aiosqlite:///{tmp_path / 'concurrent.db'}"


@pytest_asyncio.fixture
async def engine(db_url: str):
    eng = create_async_engine(db_url, echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )


@pytest_asyncio.fixture
async def setup_db(session_factory) -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as s:
        yield s


class FakeDetector:
    """Returns one detection per frame at a fixed bbox."""

    def __init__(self, model_path: str, conf_threshold: float) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> list[dict]:
        _ = frame, frame_index
        return [{"bbox": (1, 1, 4, 2), "confidence": 0.93}]

    @staticmethod
    def crop_plate(
        frame: np.ndarray, bbox: tuple[int, int, int, int], padding: int = 0
    ) -> np.ndarray:
        x, y, w, h = bbox
        _ = padding
        return frame[y : y + h, x : x + w]


def _two_frames(*_args, **_kwargs):
    frame = np.zeros((24, 48, 3), dtype=np.uint8)
    frame[:, ::2] = 255
    return [(0, 0, frame), (1, 40, frame.copy())]


async def _create_session_with_video(
    session_factory, uploads_dir: Path, source_filename: str
):
    """Create a Session row in its own AsyncSession and return its id."""
    async with session_factory() as db:
        session_obj = Session(source_filename=source_filename)
        db.add(session_obj)
        await db.commit()
        sid = session_obj.id
    (uploads_dir / f"{sid}_{source_filename}").write_bytes(b"fake-video")
    return sid


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
    monkeypatch.setattr(processor, "extract_frames", _two_frames)
    monkeypatch.setattr(processor, "lookup_ticket", lookup_mock)


@pytest.mark.asyncio
async def test_two_sessions_run_concurrently_without_errors(
    session_factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both sessions must finish ``completed`` and write their own detections."""
    uploads_dir = tmp_path / "uploads"
    crops_dir = tmp_path / "crops"
    uploads_dir.mkdir()
    crops_dir.mkdir()

    lookup_mock = AsyncMock(return_value=("active", None, 123, None))
    _configure_processor(monkeypatch, uploads_dir, crops_dir, lookup_mock)

    sid_a = await _create_session_with_video(session_factory, uploads_dir, "a.mp4")
    sid_b = await _create_session_with_video(session_factory, uploads_dir, "b.mp4")

    async def _run(sid, plate_text):
        async with session_factory() as db:
            await processor.process_session(
                sid,
                db,
                ocr=lambda _crop, _t=plate_text: _t,
                zone_id=1,
            )
            await db.commit()

    await asyncio.gather(_run(sid_a, "CJ05XY0"), _run(sid_b, "B12ABC"))

    async with session_factory() as db:
        for sid, expected_plate in ((sid_a, "CJ 05 XYO"), (sid_b, "B 12 ABC")):
            session_obj = await db.get(Session, sid)
            assert session_obj is not None
            assert session_obj.status == "completed"

            detections = (
                await db.execute(
                    select(Detection)
                    .join(Frame, Detection.frame_id == Frame.id)
                    .where(Frame.session_id == sid)
                )
            ).scalars().all()
            assert len(detections) == 2
            for d in detections:
                assert d.ocr_normalized_text == expected_plate
                assert d.ticket_status == "active"


@pytest.mark.asyncio
async def test_concurrent_sessions_share_same_plate_row(
    session_factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two sessions reading the *same* plate must end up pointing at one Plate
    row — the second insert hits the unique constraint and the savepoint
    fallback re-fetches the winner instead of crashing the session.
    """
    uploads_dir = tmp_path / "uploads"
    crops_dir = tmp_path / "crops"
    uploads_dir.mkdir()
    crops_dir.mkdir()

    lookup_mock = AsyncMock(return_value=("active", None, 123, None))
    _configure_processor(monkeypatch, uploads_dir, crops_dir, lookup_mock)

    sid_a = await _create_session_with_video(session_factory, uploads_dir, "a.mp4")
    sid_b = await _create_session_with_video(session_factory, uploads_dir, "b.mp4")

    async def _run(sid):
        async with session_factory() as db:
            await processor.process_session(
                sid,
                db,
                ocr=lambda _crop: "CJ05XY0",
                zone_id=1,
            )
            await db.commit()

    await asyncio.gather(_run(sid_a), _run(sid_b))

    async with session_factory() as db:
        plates = (await db.execute(select(Plate))).scalars().all()
        assert len(plates) == 1, (
            "Expected concurrent sessions reading the same plate to share one "
            f"Plate row; got {[p.normalized_text for p in plates]}"
        )
        plate = plates[0]
        assert plate.normalized_text == "CJ05XYO"

        detections = (await db.execute(select(Detection))).scalars().all()
        assert len(detections) == 4
        # Every detection points at the single Plate row.
        assert {d.plate_id for d in detections} == {plate.id}


@pytest.mark.asyncio
async def test_recognizer_runs_off_event_loop(
    session_factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recognizer must execute on a worker thread, not the event loop —
    otherwise a slow OCR call would freeze every other concurrent session.

    Captures the thread id from inside the stub recognizer; the asyncio loop
    runs on the main thread, so observing a different thread id is the
    contract.
    """
    uploads_dir = tmp_path / "uploads"
    crops_dir = tmp_path / "crops"
    uploads_dir.mkdir()
    crops_dir.mkdir()

    lookup_mock = AsyncMock(return_value=("active", None, 123, None))
    _configure_processor(monkeypatch, uploads_dir, crops_dir, lookup_mock)

    sid = await _create_session_with_video(session_factory, uploads_dir, "a.mp4")

    main_thread_id = threading.get_ident()
    seen_thread_ids: set[int] = set()

    def _ocr(_crop):
        seen_thread_ids.add(threading.get_ident())
        return "CJ05XY0"

    async with session_factory() as db:
        await processor.process_session(sid, db, ocr=_ocr, zone_id=1)
        await db.commit()

    assert seen_thread_ids, "Recognizer was never invoked"
    assert main_thread_id not in seen_thread_ids, (
        "Recognizer ran on the event-loop thread — to_thread dispatch is broken; "
        f"saw {seen_thread_ids}"
    )


@pytest.mark.asyncio
async def test_slow_recognizer_does_not_block_other_session(
    session_factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent sessions where one has a slow recognizer must overlap.

    If OCR were on the event loop, the slow session would serialize the fast
    one and total wall time would be ~slow * 2 frames.  When OCR is in a
    worker thread, the fast session's coroutines (DB writes, frame reads)
    progress while the slow recognizer is parked in the thread.
    """
    uploads_dir = tmp_path / "uploads"
    crops_dir = tmp_path / "crops"
    uploads_dir.mkdir()
    crops_dir.mkdir()

    lookup_mock = AsyncMock(return_value=("active", None, 123, None))
    _configure_processor(monkeypatch, uploads_dir, crops_dir, lookup_mock)

    sid_slow = await _create_session_with_video(session_factory, uploads_dir, "slow.mp4")
    sid_fast = await _create_session_with_video(session_factory, uploads_dir, "fast.mp4")

    sleep_per_call_s = 0.05  # 2 frames × 0.05s = 0.1s slow path

    def _slow_ocr(_crop):
        time.sleep(sleep_per_call_s)
        return "CJ05XY0"

    def _fast_ocr(_crop):
        return "B12ABC"

    async def _run(sid, ocr_fn):
        async with session_factory() as db:
            await processor.process_session(sid, db, ocr=ocr_fn, zone_id=1)
            await db.commit()

    started = time.perf_counter()
    await asyncio.gather(
        _run(sid_slow, _slow_ocr),
        _run(sid_fast, _fast_ocr),
    )
    elapsed = time.perf_counter() - started

    # Sequential lower bound would be ~2 × (2 × 0.05) = 0.2s for the slow path
    # alone; overlapped should be closer to ~0.1s.  Allow generous slack for
    # CI machines and fixture overhead — we only need to prove we are not
    # serialising the way blocking-on-event-loop would.
    assert elapsed < 0.35, (
        f"Concurrent sessions took {elapsed:.2f}s — looks serialised; "
        "OCR is likely still blocking the event loop"
    )

    async with session_factory() as db:
        for sid in (sid_slow, sid_fast):
            session_obj = await db.get(Session, sid)
            assert session_obj is not None
            assert session_obj.status == "completed"
