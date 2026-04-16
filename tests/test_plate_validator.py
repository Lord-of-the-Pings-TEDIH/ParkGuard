from __future__ import annotations

import pytest

from app.pipeline.plate_validator import COUNTY_CODES
from app.pipeline.plate_validator import compact_plate
from app.pipeline.plate_validator import normalize_plate


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Institution plates
        ("MA1 I234S", "MAI 12345"),
        # Diplomatic plates
        ("C0 I23 4S6", "CD 123 456"),
        # Army plates
        ("4 I234", "A 1234"),
        ("8 I234S6", "B 123456"),  # hmm, this starts with "8" → "B" which is Bucharest
        # Temporary plates
        ("CJ 0I234", "CJ 01234"),
        # Standard plates
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


# ---------------------------------------------------------------------------
# Electric plates (E + standard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ECJ12ABC", "E CJ 12 ABC"),
        ("EB12ABC", "E B 12 ABC"),
        ("EB123XYZ", "E B 123 XYZ"),
        ("E CJ 05 XYZ", "E CJ 05 XYZ"),
    ],
)
def test_electric_plate_normalisation(raw: str, expected: str) -> None:
    result = normalize_plate(raw)
    assert result == expected, f"normalize_plate({raw!r}) = {result!r}, expected {expected!r}"


# ---------------------------------------------------------------------------
# Probe (test-drive) plates (P + county + digits)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("PCJ12345", "P CJ 12345"),
        ("PB1234", "P B 1234"),
        ("PB12345", "P B 12345"),
    ],
)
def test_probe_plate_normalisation(raw: str, expected: str) -> None:
    result = normalize_plate(raw)
    assert result == expected, f"normalize_plate({raw!r}) = {result!r}, expected {expected!r}"


# ---------------------------------------------------------------------------
# Standard vs Temporary priority order
# ---------------------------------------------------------------------------


def test_standard_before_temporary_with_letter_tail() -> None:
    """A 7-char plate with letters at the end should be Standard, not Temporary."""
    result = normalize_plate("CJ12ABC")
    assert result == "CJ 12 ABC"
    assert not result.startswith("INVALID:")


def test_temporary_only_when_no_letter_tail() -> None:
    """When the tail is genuinely numeric, the plate IS temporary."""
    result = normalize_plate("CJ12345")
    assert result == "CJ 12345"


# ---------------------------------------------------------------------------
# Confusion corrections
# ---------------------------------------------------------------------------


def test_confusion_3_maps_to_E_not_B() -> None:
    """Digit 3 should map to E (not B, which is already mapped from 8)."""
    from app.pipeline.plate_validator import NUM_TO_LET
    assert NUM_TO_LET["3"] == "E"
    assert NUM_TO_LET["8"] == "B"


def test_confusion_D_maps_to_0() -> None:
    from app.pipeline.plate_validator import LET_TO_NUM
    assert LET_TO_NUM["D"] == "0"


def test_no_lowercase_l_in_LET_TO_NUM() -> None:
    """Lowercase 'l' was dead code (cleaned text is always uppercased)."""
    from app.pipeline.plate_validator import LET_TO_NUM
    assert "l" not in LET_TO_NUM


# ---------------------------------------------------------------------------
# Color-based plate_type hint
# ---------------------------------------------------------------------------


def test_temporary_hint_forces_temporary_parsing() -> None:
    """When color says 'temporary' (red text), SJ07404 → SJ 07404."""
    result = normalize_plate("SJ07404", plate_type="temporary")
    assert result == "SJ 07404"


def test_temporary_hint_on_ambiguous_plate() -> None:
    """CJ12345 with temporary hint should skip standard and parse as temporary."""
    result = normalize_plate("CJ12345", plate_type="temporary")
    assert result == "CJ 12345"


def test_standard_hint_blocks_temporary_fallback() -> None:
    """When color says 'standard', don't fall back to temporary."""
    result = normalize_plate("CJ12ABC", plate_type="standard")
    assert result == "CJ 12 ABC"


def test_no_hint_uses_default_priority() -> None:
    """Without hint, normal priority applies."""
    result = normalize_plate("CJ12ABC")
    assert result == "CJ 12 ABC"


def test_standard_hint_on_invalid_returns_invalid() -> None:
    """When colour says standard but text doesn't match any format, return INVALID."""
    result = normalize_plate("GARBAGE", plate_type="standard")
    assert result.startswith("INVALID:")


def test_standard_hint_does_not_override_special_numeric_family() -> None:
    """County + 5 digits stays in special numeric parsing even with standard hint."""
    result = normalize_plate("SJ07404", plate_type="standard")
    assert result == "SJ 07404"


def test_standard_hint_does_not_force_county_plus_5_digits_to_standard() -> None:
    """County + 5 digits should remain temporary-style, not letter-tail standard."""
    result = normalize_plate("SJ05084", plate_type="standard")
    assert result == "SJ 05084"


def test_standard_hint_does_not_force_county_plus_6_digits_to_standard() -> None:
    """B + 6 digits is also special numeric (not coerced to a 3-letter tail)."""
    result = normalize_plate("B123456", plate_type="standard")
    assert result == "B 123456"


def test_temporary_with_4_digits_is_invalid() -> None:
    result = normalize_plate("SJ0740", plate_type="temporary")
    assert result.startswith("INVALID:")


def test_county_two_letters_needs_standard_2_digits_plus_3_letters() -> None:
    result = normalize_plate("SJ37A")
    assert result.startswith("INVALID:")
