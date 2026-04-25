"""Tests for app.pipeline.video_metadata.extract_video_gps."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from app.pipeline import video_metadata
from app.pipeline.video_metadata import extract_video_gps, parse_iso6709


# ---------------------------- parse_iso6709 ---------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("+47.2584+023.2537+202.369/", (47.2584, 23.2537)),
        ("+46.7701+023.5895/", (46.7701, 23.5895)),
        ("-33.8688+151.2093/", (-33.8688, 151.2093)),
        ("+00.0000-000.0000/", (0.0, 0.0)),
        ("+47.2584+023.2537", (47.2584, 23.2537)),  # missing trailing slash
    ],
)
def test_parse_iso6709_valid(raw: str, expected: tuple[float, float]) -> None:
    parsed = parse_iso6709(raw)
    assert parsed is not None
    lat, lon = parsed
    assert lat == pytest.approx(expected[0])
    assert lon == pytest.approx(expected[1])


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not-a-coord",
        "47.2584,23.2537",         # decimal-comma form, not ISO 6709
        "+91.0000+023.2537/",      # latitude > 90
        "+47.2584+181.0000/",      # longitude > 180
    ],
)
def test_parse_iso6709_invalid(raw: str) -> None:
    assert parse_iso6709(raw) is None


# ---------------------------- extract_video_gps -----------------------------


def _ffprobe_response(tags: dict) -> str:
    return json.dumps({"format": {"tags": tags}})


def test_extract_video_gps_returns_none_when_ffprobe_missing(tmp_path) -> None:
    fake_video = tmp_path / "x.mov"
    fake_video.write_bytes(b"")
    with patch.object(video_metadata.shutil, "which", return_value=None):
        assert extract_video_gps(fake_video) is None


def test_extract_video_gps_parses_apple_iso6709(tmp_path) -> None:
    fake_video = tmp_path / "x.mov"
    fake_video.write_bytes(b"")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=_ffprobe_response(
            {"com.apple.quicktime.location.ISO6709": "+47.2584+023.2537+202.369/"}
        ),
        stderr="",
    )
    with patch.object(video_metadata.shutil, "which", return_value="/usr/bin/ffprobe"), \
         patch.object(video_metadata.subprocess, "run", return_value=completed):
        result = extract_video_gps(fake_video)
    assert result is not None
    lat, lon = result
    assert lat == pytest.approx(47.2584)
    assert lon == pytest.approx(23.2537)


def test_extract_video_gps_uses_legacy_xyz_atom(tmp_path) -> None:
    fake_video = tmp_path / "x.mp4"
    fake_video.write_bytes(b"")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=_ffprobe_response({"©xyz": "-33.8688+151.2093/"}),
        stderr="",
    )
    with patch.object(video_metadata.shutil, "which", return_value="/usr/bin/ffprobe"), \
         patch.object(video_metadata.subprocess, "run", return_value=completed):
        result = extract_video_gps(fake_video)
    assert result == pytest.approx((-33.8688, 151.2093))


def test_extract_video_gps_returns_none_when_no_location_tag(tmp_path) -> None:
    fake_video = tmp_path / "x.mp4"
    fake_video.write_bytes(b"")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=_ffprobe_response({"com.apple.quicktime.make": "Apple"}),
        stderr="",
    )
    with patch.object(video_metadata.shutil, "which", return_value="/usr/bin/ffprobe"), \
         patch.object(video_metadata.subprocess, "run", return_value=completed):
        assert extract_video_gps(fake_video) is None


def test_extract_video_gps_handles_ffprobe_failure(tmp_path) -> None:
    fake_video = tmp_path / "x.mp4"
    fake_video.write_bytes(b"")
    completed = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="boom"
    )
    with patch.object(video_metadata.shutil, "which", return_value="/usr/bin/ffprobe"), \
         patch.object(video_metadata.subprocess, "run", return_value=completed):
        assert extract_video_gps(fake_video) is None


def test_extract_video_gps_handles_subprocess_timeout(tmp_path) -> None:
    fake_video = tmp_path / "x.mp4"
    fake_video.write_bytes(b"")
    with patch.object(video_metadata.shutil, "which", return_value="/usr/bin/ffprobe"), \
         patch.object(
             video_metadata.subprocess,
             "run",
             side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=10),
         ):
        assert extract_video_gps(fake_video) is None
