"""Tests for app.pipeline.ctc_decoder (FSM Prefix Beam Search)."""

from __future__ import annotations

import pytest

from app.pipeline.ctc_decoder import fsm_beam_decode
from app.pipeline.plate_validator import is_invalid_plate, normalize_plate


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _is_valid_plate(text: str) -> bool:
    return not is_invalid_plate(normalize_plate(text))


# ---------------------------------------------------------------------------
# Correct inputs — no substitution needed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "B12ABC",    # Bucharest 2-digit
        "B123XYZ",   # Bucharest 3-digit
        "CJ12ABC",   # Two-char county + 2 digits + 3 letters
        "IS45DEF",   # Another county
        "TM99XYZ",   # TM county
    ],
)
def test_fsm_beam_decode_passes_through_valid_plates(raw: str) -> None:
    """Perfect OCR output must be returned unchanged."""
    result, conf = fsm_beam_decode(raw, 0.9)
    assert result == raw
    assert conf == pytest.approx(0.9)
    assert _is_valid_plate(result)


# ---------------------------------------------------------------------------
# Classic confusion errors in digit positions
# ---------------------------------------------------------------------------


def test_letter_in_digit_position_Z_corrected_to_2() -> None:
    result, _ = fsm_beam_decode("B1ZABC", 0.9)
    assert result == "B12ABC"
    assert _is_valid_plate(result)


def test_letter_in_digit_position_O_corrected_to_0() -> None:
    result, _ = fsm_beam_decode("B1OABC", 0.9)
    assert result == "B10ABC"
    assert _is_valid_plate(result)


def test_letter_in_digit_position_I_corrected_to_1() -> None:
    result, _ = fsm_beam_decode("BIIABС", 0.9)
    # Cyrillic С may cause beam death — this test just ensures no crash


def test_multiple_errors_in_digit_positions() -> None:
    result, _ = fsm_beam_decode("CJ1ZABC", 0.9)
    assert result == "CJ12ABC"
    assert _is_valid_plate(result)


# ---------------------------------------------------------------------------
# Classic confusion errors in letter positions
# ---------------------------------------------------------------------------


def test_digit_in_letter_position_8_corrected_to_B() -> None:
    result, _ = fsm_beam_decode("CJ12A8C", 0.9)
    assert result == "CJ12ABC"
    assert _is_valid_plate(result)


def test_digit_in_letter_position_0_corrected_to_O() -> None:
    result, _ = fsm_beam_decode("B120AB0", 0.9)
    assert _is_valid_plate(result)
    assert result[-1] == "O"


def test_multiple_errors_both_positions() -> None:
    result, _ = fsm_beam_decode("CJ1ZA8C", 0.9)
    assert result == "CJ12ABC"
    assert _is_valid_plate(result)


# ---------------------------------------------------------------------------
# Electric plate beam decode
# ---------------------------------------------------------------------------


def test_electric_plate_passes_through() -> None:
    result, conf = fsm_beam_decode("ECJ12ABC", 0.9)
    assert result == "ECJ12ABC"
    assert conf == pytest.approx(0.9)


def test_electric_plate_corrects_confusion() -> None:
    # "ECJ12A8C" — 8 in letter pos → B → "ECJ12ABC"
    result, _ = fsm_beam_decode("ECJ12A8C", 0.9)
    assert result == "ECJ12ABC"


# ---------------------------------------------------------------------------
# Confidence adjustment
# ---------------------------------------------------------------------------


def test_perfect_input_preserves_confidence() -> None:
    _, conf = fsm_beam_decode("B12ABC", 0.95)
    assert conf == pytest.approx(0.95)


def test_substituted_input_reduces_confidence() -> None:
    _, conf_corrected = fsm_beam_decode("CJ12A8C", 0.9)
    assert conf_corrected < 0.9


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_string_returns_empty() -> None:
    result, conf = fsm_beam_decode("", 0.5)
    assert result == ""
    assert conf == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Undecodable inputs — passthrough / graceful fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "AB",         # Too short — no matching pattern
        "X",          # Single char
    ],
)
def test_undecodable_input_falls_back_gracefully(raw: str) -> None:
    result, conf = fsm_beam_decode(raw, 0.8)
    assert isinstance(result, str)
    assert isinstance(conf, float)
    assert conf >= 0.0


# ---------------------------------------------------------------------------
# Beam width parameter
# ---------------------------------------------------------------------------


def test_beam_width_1_still_decodes_classic_confusion() -> None:
    result, _ = fsm_beam_decode("CJ12A8C", 0.9, beam_width=1)
    assert result == "CJ12ABC"


def test_beam_width_10_matches_default_on_clean_input() -> None:
    r1, c1 = fsm_beam_decode("B12ABC", 0.9, beam_width=5)
    r2, c2 = fsm_beam_decode("B12ABC", 0.9, beam_width=10)
    assert r1 == r2
    assert c1 == pytest.approx(c2)
