"""Integration test: process a 5-second synthetic video end-to-end."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import AsyncGenerator

import cv2
import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import BigInteger
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
from app.models import parking  # noqa: F401
from app.models.detection import Session as SessionRow
from app.pipeline import worker

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


def _make_video(path: Path, seconds: int = 5, fps: int = 25) -> None:
    """Write a small synthetic mp4: black frames with a moving white box."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), "VideoWriter failed to open — codec missing?"
    try:
        total = seconds * fps
        for i in range(total):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            x = (i * 4) % (w - 40)
            frame[100:140, x:x + 40] = 255
            writer.write(frame)
    finally:
        writer.release()


class _StubDetector:
    """Replaces PlateDetector — returns no detections, no model load."""

    def __init__(self, *_args, **_kwargs):
        pass

    def detect(self, frame, frame_index: int = 0):
        return []

    @staticmethod
    def crop_plate(frame, bbox, padding: int = 4):
        return frame


class _StubReader:
    def __init__(self, *_args, **_kwargs):
        pass

    def read_plate(self, crop):
        return "", 0.0


@pytest.mark.asyncio
async def test_process_5s_video_marks_done(db: AsyncSession, tmp_path, monkeypatch):
    video_path = tmp_path / "tiny.mp4"
    _make_video(video_path, seconds=5, fps=25)

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    monkeypatch.setattr("app.pipeline.worker.settings.CROPS_DIR", str(crops_dir))
    monkeypatch.setattr("app.pipeline.worker.PlateDetector", _StubDetector)
    monkeypatch.setattr("app.pipeline.worker.PlateReader", _StubReader)

    session_id = uuid.uuid4()
    db.add(SessionRow(id=session_id, source_filename="tiny.mp4", status="pending"))
    await db.commit()

    worker.PROCESSING_SEMAPHORE = asyncio.Semaphore(1)

    await worker.process_video(
        session_id=session_id,
        video_path=str(video_path),
        fps_target=5,
        db_factory=_db_factory,
    )

    async with TestingSessionLocal() as fresh:
        row = await fresh.get(SessionRow, session_id)
        assert row is not None, "Session row vanished"
        assert row.status == "done", f"expected 'done', got {row.status!r} ({row.error_message!r})"
        assert row.frames_processed > 0
        assert row.total_frames and row.total_frames > 0
