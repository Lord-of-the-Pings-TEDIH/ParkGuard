"""Tests for app.pipeline.video_metadata."""

from __future__ import annotations

import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from PIL.TiffImagePlugin import IFDRational


def _R(numerator: float, denominator: float = 1) -> IFDRational:
    """Shorthand for the rational type Pillow expects in EXIF tags."""
    return IFDRational(numerator, denominator)

from app.pipeline import video_metadata
from app.pipeline.video_metadata import (
    MediaGpsMetadata,
    extract_media_gps_metadata,
    extract_video_gps,
    extract_video_gps_metadata,
    parse_iso6709,
)


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


# ---------------------- extract_video_gps_metadata --------------------------


def test_extract_metadata_includes_accuracy_and_recorded_at(tmp_path) -> None:
    fake_video = tmp_path / "x.mov"
    fake_video.write_bytes(b"")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=_ffprobe_response(
            {
                "com.apple.quicktime.location.ISO6709": "+46.7701+023.5895/",
                "com.apple.quicktime.location.accuracy.horizontal": "4.7",
                "com.apple.quicktime.creationdate": "2026-04-26T10:30:00+00:00",
            }
        ),
        stderr="",
    )
    with patch.object(video_metadata.shutil, "which", return_value="/usr/bin/ffprobe"), \
         patch.object(video_metadata.subprocess, "run", return_value=completed):
        meta = extract_video_gps_metadata(fake_video)
    assert meta is not None
    assert meta.latitude == pytest.approx(46.7701)
    assert meta.longitude == pytest.approx(23.5895)
    assert meta.accuracy_m == pytest.approx(4.7)
    assert meta.recorded_at == datetime(2026, 4, 26, 10, 30, tzinfo=timezone.utc)


def test_extract_metadata_drops_negative_accuracy(tmp_path) -> None:
    """iOS reports ``-1`` for "no fix" — must surface as accuracy_m=None."""
    fake_video = tmp_path / "x.mov"
    fake_video.write_bytes(b"")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=_ffprobe_response(
            {
                "com.apple.quicktime.location.ISO6709": "+46.7701+023.5895/",
                "com.apple.quicktime.location.accuracy.horizontal": "-1",
            }
        ),
        stderr="",
    )
    with patch.object(video_metadata.shutil, "which", return_value="/usr/bin/ffprobe"), \
         patch.object(video_metadata.subprocess, "run", return_value=completed):
        meta = extract_video_gps_metadata(fake_video)
    assert meta is not None
    assert meta.accuracy_m is None


def test_extract_metadata_handles_missing_optional_tags(tmp_path) -> None:
    """Only the location tag is required — accuracy + recorded_at are optional."""
    fake_video = tmp_path / "x.mov"
    fake_video.write_bytes(b"")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=_ffprobe_response(
            {"com.apple.quicktime.location.ISO6709": "+46.7701+023.5895/"}
        ),
        stderr="",
    )
    with patch.object(video_metadata.shutil, "which", return_value="/usr/bin/ffprobe"), \
         patch.object(video_metadata.subprocess, "run", return_value=completed):
        meta = extract_video_gps_metadata(fake_video)
    assert meta is not None
    assert meta.accuracy_m is None
    assert meta.recorded_at is None


# -- universal video tag variants --------------------------------------------


def _patch_ffprobe(tags: dict[str, str]):
    """Context-manager pair (which, run) used by every video-extraction test."""
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=_ffprobe_response(tags), stderr=""
    )
    return (
        patch.object(video_metadata.shutil, "which", return_value="/usr/bin/ffprobe"),
        patch.object(video_metadata.subprocess, "run", return_value=completed),
    )


def test_android_location_tag_recognised(tmp_path) -> None:
    """Android Camera2 writes ``location`` as a plain ISO 6709 string."""
    fake = tmp_path / "android.mp4"
    fake.write_bytes(b"")
    which, run = _patch_ffprobe({"location": "+46.7701+023.5895/"})
    with which, run:
        meta = extract_media_gps_metadata(fake)
    assert meta is not None
    assert meta.latitude == pytest.approx(46.7701)
    assert meta.longitude == pytest.approx(23.5895)
    assert meta.source_format == "quicktime-iso6709"


def test_xyz_atom_alias_recognised(tmp_path) -> None:
    """The ASCII alias ``xyz`` maps to the same ISO 6709 path as ``©xyz``."""
    fake = tmp_path / "legacy.mp4"
    fake.write_bytes(b"")
    which, run = _patch_ffprobe({"xyz": "+46.7701+023.5895/"})
    with which, run:
        meta = extract_media_gps_metadata(fake)
    assert meta is not None
    assert meta.latitude == pytest.approx(46.7701)


def test_mkv_separate_lat_lon_tags(tmp_path) -> None:
    """Matroska sometimes splits the fix into two scalar tags rather than ISO 6709."""
    fake = tmp_path / "video.mkv"
    fake.write_bytes(b"")
    which, run = _patch_ffprobe({"latitude": "46.7701", "longitude": "23.5895"})
    with which, run:
        meta = extract_media_gps_metadata(fake)
    assert meta is not None
    assert meta.latitude == pytest.approx(46.7701)
    assert meta.longitude == pytest.approx(23.5895)
    assert meta.source_format == "mkv-tags"


def test_mkv_out_of_range_lat_rejected(tmp_path) -> None:
    fake = tmp_path / "video.mkv"
    fake.write_bytes(b"")
    which, run = _patch_ffprobe({"latitude": "120.0", "longitude": "23.5895"})
    with which, run:
        assert extract_media_gps_metadata(fake) is None


def test_iso6709_altitude_captured(tmp_path) -> None:
    """ISO 6709 H form's signed altitude component should populate altitude_m."""
    fake = tmp_path / "x.mp4"
    fake.write_bytes(b"")
    which, run = _patch_ffprobe(
        {"com.apple.quicktime.location.ISO6709": "+46.7701+023.5895+334.000/"}
    )
    with which, run:
        meta = extract_media_gps_metadata(fake)
    assert meta is not None
    assert meta.altitude_m == pytest.approx(334.0)


# -- image EXIF extraction ---------------------------------------------------


def _write_jpeg_with_gps_exif(
    path: Path,
    *,
    lat_dms: tuple[IFDRational, IFDRational, IFDRational],
    lat_ref: str,
    lon_dms: tuple[IFDRational, IFDRational, IFDRational],
    lon_ref: str,
    altitude: IFDRational | None = None,
    altitude_ref: int = 0,
    heading: IFDRational | None = None,
    accuracy: IFDRational | None = None,
) -> None:
    """Write a 4×4 JPEG with the requested GPS fields populated in EXIF.

    Pillow's GPSInfo IFD uses numeric tag ids — see the EXIF spec or
    ``PIL.ExifTags.GPSTAGS``.  Building the bytes here keeps the test
    independent of any binary fixture.
    """
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    exif = img.getexif()
    gps_ifd: dict[int, object] = {
        1: lat_ref,                 # GPSLatitudeRef
        2: lat_dms,                 # GPSLatitude
        3: lon_ref,                 # GPSLongitudeRef
        4: lon_dms,                 # GPSLongitude
    }
    if altitude is not None:
        gps_ifd[5] = altitude_ref   # GPSAltitudeRef (0=above, 1=below sea level)
        gps_ifd[6] = altitude       # GPSAltitude
    if heading is not None:
        gps_ifd[16] = "T"           # GPSImgDirectionRef ("T" = true north)
        gps_ifd[17] = heading       # GPSImgDirection
    if accuracy is not None:
        gps_ifd[31] = accuracy      # GPSHPositioningError
    exif[0x8825] = gps_ifd          # GPSInfo IFD pointer

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", exif=exif.tobytes())
    path.write_bytes(buffer.getvalue())


def test_extract_jpeg_exif_north_east(tmp_path) -> None:
    fake = tmp_path / "photo.jpg"
    _write_jpeg_with_gps_exif(
        fake,
        lat_dms=(_R(46), _R(46), _R(12)),    # 46°46'12"
        lat_ref="N",
        lon_dms=(_R(23), _R(35), _R(22, 1)), # 23°35'22"
        lon_ref="E",
        altitude=_R(334, 1),
        altitude_ref=0,
        heading=_R(90, 1),
        accuracy=_R(5, 1),
    )
    meta = extract_media_gps_metadata(fake)
    assert meta is not None
    assert meta.latitude == pytest.approx(46 + 46 / 60 + 12 / 3600, abs=1e-9)
    assert meta.longitude == pytest.approx(23 + 35 / 60 + 22 / 3600, abs=1e-9)
    assert meta.altitude_m == pytest.approx(334.0)
    assert meta.heading_deg == pytest.approx(90.0)
    assert meta.accuracy_m == pytest.approx(5.0)
    assert meta.source_format == "exif"


def test_extract_jpeg_exif_south_west_signs(tmp_path) -> None:
    """``S`` / ``W`` references must produce negative decimal coordinates."""
    fake = tmp_path / "photo.jpg"
    _write_jpeg_with_gps_exif(
        fake,
        lat_dms=(_R(33), _R(52), _R(7, 1)),
        lat_ref="S",
        lon_dms=(_R(151), _R(12), _R(33, 1)),
        lon_ref="W",
    )
    meta = extract_media_gps_metadata(fake)
    assert meta is not None
    assert meta.latitude < 0
    assert meta.longitude < 0


def test_extract_jpeg_below_sea_level(tmp_path) -> None:
    """``GPSAltitudeRef = 1`` flips altitude sign to below sea level."""
    fake = tmp_path / "photo.jpg"
    _write_jpeg_with_gps_exif(
        fake,
        lat_dms=(_R(46), _R(0), _R(0)),
        lat_ref="N",
        lon_dms=(_R(23), _R(0), _R(0)),
        lon_ref="E",
        altitude=_R(50, 1),
        altitude_ref=1,
    )
    meta = extract_media_gps_metadata(fake)
    assert meta is not None
    assert meta.altitude_m == pytest.approx(-50.0)


def test_extract_jpeg_without_exif_returns_none(tmp_path) -> None:
    fake = tmp_path / "plain.jpg"
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(fake, format="JPEG")
    assert extract_media_gps_metadata(fake) is None


def test_extract_unreadable_image_returns_none(tmp_path) -> None:
    fake = tmp_path / "broken.jpg"
    fake.write_bytes(b"not really a jpeg")
    assert extract_media_gps_metadata(fake) is None


# -- universal dispatch ------------------------------------------------------


def test_dispatch_uses_image_path_for_image_extension(tmp_path) -> None:
    fake = tmp_path / "shot.jpg"
    _write_jpeg_with_gps_exif(
        fake,
        lat_dms=(_R(46), _R(0), _R(0)),
        lat_ref="N",
        lon_dms=(_R(23), _R(0), _R(0)),
        lon_ref="E",
    )
    # ffprobe must NOT be invoked for an image file — patch ``shutil.which``
    # to a sentinel and ensure the result still comes back.
    with patch.object(video_metadata.shutil, "which") as which_mock:
        meta = extract_media_gps_metadata(fake)
    assert meta is not None
    which_mock.assert_not_called()


def test_dispatch_uses_video_path_for_video_extension(tmp_path) -> None:
    fake = tmp_path / "clip.mp4"
    fake.write_bytes(b"")
    which, run = _patch_ffprobe({"location": "+46.7701+023.5895/"})
    with which, run:
        meta = extract_media_gps_metadata(fake)
    assert meta is not None
    assert meta.source_format == "quicktime-iso6709"


def test_dispatch_unknown_extension_tries_image_then_video(tmp_path) -> None:
    """A file with no extension should still be probed by both backends."""
    fake = tmp_path / "noext"
    # Build it as a JPEG underneath so the image path succeeds.
    _write_jpeg_with_gps_exif(
        fake,
        lat_dms=(_R(46), _R(0), _R(0)),
        lat_ref="N",
        lon_dms=(_R(23), _R(0), _R(0)),
        lon_ref="E",
    )
    meta = extract_media_gps_metadata(fake)
    assert meta is not None


def test_video_gps_metadata_alias_returns_media_gps_type(tmp_path) -> None:
    """Old import paths still resolve and yield the new dataclass."""
    fake = tmp_path / "x.mp4"
    fake.write_bytes(b"")
    which, run = _patch_ffprobe({"location": "+46.7701+023.5895/"})
    with which, run:
        meta = extract_video_gps_metadata(fake)
    assert isinstance(meta, MediaGpsMetadata)
    assert meta is not None and meta.latitude == pytest.approx(46.7701)
