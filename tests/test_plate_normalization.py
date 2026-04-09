from __future__ import annotations

import pytest

from app.pipeline.plate_normalization import normalize_plate

COUNTY_CODES = (
    "AB",
    "AG",
    "AR",
    "B",
    "BC",
    "BH",
    "BN",
    "BR",
    "BT",
    "BV",
    "BZ",
    "CJ",
    "CL",
    "CS",
    "CT",
    "CV",
    "DB",
    "DJ",
    "GJ",
    "GL",
    "GR",
    "HD",
    "HR",
    "IF",
    "IL",
    "IS",
    "MH",
    "MM",
    "MS",
    "NT",
    "OT",
    "PH",
    "SB",
    "SJ",
    "SM",
    "SV",
    "TL",
    "TM",
    "TR",
    "VL",
    "VN",
    "VS",
)


def test_normalize_plate_strips_spacing_and_uppercases() -> None:
    assert normalize_plate("B 12 abc") == ("B12ABC", True)


def test_normalize_plate_applies_ocr_zero_to_c_correction() -> None:
    assert normalize_plate("0J05XYZ") == ("CJ05XYZ", True)


def test_normalize_plate_keeps_invalid_payload_and_marks_it_invalid() -> None:
    assert normalize_plate("GARBAGE") == ("GARBAGE", False)


def test_invalid_county_code_is_rejected() -> None:
    assert normalize_plate("XX12ABC") == ("XX12ABC", False)


def test_county_fixture_has_42_codes() -> None:
    assert len(COUNTY_CODES) == 42


@pytest.mark.parametrize("county_code", COUNTY_CODES)
def test_all_romanian_county_codes_are_accepted(county_code: str) -> None:
    plate = "B12ABC" if county_code == "B" else f"{county_code}12ABC"
    assert normalize_plate(plate) == (plate, True)
