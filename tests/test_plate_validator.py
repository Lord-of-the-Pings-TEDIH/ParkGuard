from __future__ import annotations

import pytest

from app.pipeline.plate_validator import COUNTY_CODES
from app.pipeline.plate_validator import compact_plate
from app.pipeline.plate_validator import normalize_plate


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("MA1 I234S", "MAI 12345"),
        ("C0 I23 4S6", "CD 123 456"),
        ("4 I234", "A 1234"),
        ("8 I234S6", "B 123456"),
        ("CJ 0I234", "CJ 01234"),
        ("B-123.A8C", "B 123 ABC"),
        ("C1 01 XY2", "CJ 01 XYZ"),
    ],
)
def test_required_ocr_cases(raw: str, expected: str) -> None:
    assert normalize_plate(raw) == expected


def test_invalid_value_has_reason() -> None:
    assert normalize_plate("GARBAGE") == "INVALID: județ invalid"


def test_county_fixture_has_42_codes() -> None:
    assert len(COUNTY_CODES) == 42


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CJ12ABC", "CJ12ABC"),
        ("IS12ABC", "IS12ABC"),
        ("TM12ABC", "TM12ABC"),
        ("VL12ABC", "VL12ABC"),
        ("B12ABC", "B12ABC"),
    ],
)
def test_compact_plate_on_valid_standard_outputs(raw: str, expected: str) -> None:
    normalized = normalize_plate(raw)
    assert not normalized.startswith("INVALID:")
    assert compact_plate(normalized) == expected
