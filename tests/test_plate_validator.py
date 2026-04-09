from __future__ import annotations

import pytest

from app.pipeline.plate_validator import COUNTY_CODES
from app.pipeline.plate_validator import normalize_plate


@pytest.mark.parametrize(
    ("raw", "expected_normalized", "expected_valid"),
    [
        ("B 12 abc", "B12ABC", True),
        ("cj 05 xyz", "CJ05XYZ", True),
        ("B123ABC", "B123ABC", True),
        ("0J05XYZ", "CJ05XYZ", True),
        ("XX99AAA", "XX99AAA", False),
        ("", "", False),
        ("GARBAGE", "GARBAGE", False),
        ("B12ABC", "B12ABC", True),
        ("IS07GHI", "IS07GHI", True),
        (" B-123-abc ", "B123ABC", True),
        ("ab1oab0", "AB10ABO", True),
        ("08OI158", "OB01ISB", False),
        ("B1S8AB0", "B158ABO", True),
        ("8123AB0", "B123ABO", True),
        ("0S05XYZ", "CS05XYZ", True),
        ("0L05XYZ", "CL05XYZ", True),
        ("10O5XYZ", "IO05XYZ", False),
        ("TM9QQQ", "TM9QQQ", False),
        ("B1ABC", "B1ABC", False),
        ("B1234ABC", "B1234ABC", False),
        ("AB12AB", "AB12AB", False),
        ("B12AB0", "B12AB0", False),
        ("  ", "", False),
        ("VL00AAA", "VL00AAA", True),
        ("PHO1ABC", "PH01ABC", True),
    ],
)
def test_normalize_plate_cases(
    raw: str, expected_normalized: str, expected_valid: bool
) -> None:
    assert normalize_plate(raw) == (expected_normalized, expected_valid)


def test_county_fixture_has_42_codes() -> None:
    assert len(COUNTY_CODES) == 42


@pytest.mark.parametrize("county_code", sorted(COUNTY_CODES))
def test_all_romanian_county_codes_are_accepted(county_code: str) -> None:
    plate = "B12ABC" if county_code == "B" else f"{county_code}12ABC"
    assert normalize_plate(plate) == (plate, True)
