"""Tests for app.pipeline.fsm_plate."""

from __future__ import annotations

import pytest

from app.pipeline.fsm_plate import (
    ALL_PATTERNS,
    CONFUSION_SCORE,
    DIGITS,
    LETTERS,
    STANDARD_PATTERNS,
    char_candidates,
    char_type,
    slot_accepts,
)


# ---------------------------------------------------------------------------
# char_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("char", sorted(LETTERS))
def test_char_type_returns_L_for_all_letters(char: str) -> None:
    assert char_type(char) == "L"


@pytest.mark.parametrize("char", sorted(DIGITS))
def test_char_type_returns_D_for_all_digits(char: str) -> None:
    assert char_type(char) == "D"


@pytest.mark.parametrize("char", ["!", " ", "-", "_", "å"])
def test_char_type_returns_question_mark_for_non_alnum(char: str) -> None:
    assert char_type(char) == "?"


# ---------------------------------------------------------------------------
# slot_accepts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("char", ["A", "B", "Z"])
def test_slot_accepts_letter_slot_with_letters(char: str) -> None:
    assert slot_accepts("L", char) is True


@pytest.mark.parametrize("char", ["0", "1", "9"])
def test_slot_accepts_letter_slot_rejects_digits(char: str) -> None:
    assert slot_accepts("L", char) is False


@pytest.mark.parametrize("char", ["0", "1", "9"])
def test_slot_accepts_digit_slot_with_digits(char: str) -> None:
    assert slot_accepts("D", char) is True


@pytest.mark.parametrize("char", ["A", "Z"])
def test_slot_accepts_digit_slot_rejects_letters(char: str) -> None:
    assert slot_accepts("D", char) is False


def test_slot_accepts_unknown_slot_type_returns_false() -> None:
    assert slot_accepts("X", "A") is False


# ---------------------------------------------------------------------------
# Fixed-literal slots (E and P)
# ---------------------------------------------------------------------------


def test_slot_accepts_E_slot_only_accepts_E() -> None:
    assert slot_accepts("E", "E") is True
    assert slot_accepts("E", "A") is False
    assert slot_accepts("E", "3") is False


def test_slot_accepts_P_slot_only_accepts_P() -> None:
    assert slot_accepts("P", "P") is True
    assert slot_accepts("P", "A") is False
    assert slot_accepts("P", "0") is False


# ---------------------------------------------------------------------------
# char_candidates
# ---------------------------------------------------------------------------


def test_char_candidates_digit_zero_yields_letter_O() -> None:
    cands = char_candidates("0")
    chars = [c for c, _ in cands]
    assert "0" in chars
    assert "O" in chars


def test_char_candidates_digit_one_yields_letter_I() -> None:
    chars = [c for c, _ in char_candidates("1")]
    assert "1" in chars
    assert "I" in chars


def test_char_candidates_letter_I_yields_digit_one() -> None:
    chars = [c for c, _ in char_candidates("I")]
    assert "I" in chars
    assert "1" in chars


def test_char_candidates_letter_O_yields_digit_zero() -> None:
    chars = [c for c, _ in char_candidates("O")]
    assert "O" in chars
    assert "0" in chars


def test_char_candidates_original_always_first_with_score_one() -> None:
    for char in ("A", "5", "X", "0"):
        cands = char_candidates(char)
        assert cands[0] == (char, 1.0), f"Expected original first for {char!r}"


def test_char_candidates_confusion_score_is_correct() -> None:
    cands = char_candidates("0")
    scores = {c: s for c, s in cands}
    assert scores.get("O") == pytest.approx(CONFUSION_SCORE)


def test_char_candidates_no_confusion_for_unambiguous_chars() -> None:
    cands = char_candidates("X")
    assert len(cands) == 1
    assert cands[0][0] == "X"


# ---------------------------------------------------------------------------
# Updated confusion mappings (3→E, D→0)
# ---------------------------------------------------------------------------


def test_digit_3_maps_to_E_not_B() -> None:
    """Digit 3 should now map to 'E', not 'B' (avoids ambiguity with 8→B)."""
    cands = char_candidates("3")
    chars = [c for c, _ in cands]
    assert "E" in chars
    assert "B" not in chars


def test_letter_D_maps_to_digit_0() -> None:
    """Letter D is commonly misread as 0 by OCR."""
    cands = char_candidates("D")
    chars = [c for c, _ in cands]
    assert "D" in chars
    assert "0" in chars


# ---------------------------------------------------------------------------
# STANDARD_PATTERNS integrity
# ---------------------------------------------------------------------------


def test_standard_patterns_are_unique() -> None:
    assert len(set(STANDARD_PATTERNS)) == len(STANDARD_PATTERNS)


def test_standard_patterns_only_contain_L_and_D() -> None:
    for pattern in STANDARD_PATTERNS:
        assert all(c in ("L", "D") for c in pattern), pattern


def test_standard_patterns_expected_lengths() -> None:
    lengths = {len(p) for p in STANDARD_PATTERNS}
    assert lengths == {6, 7}


# ---------------------------------------------------------------------------
# ALL_PATTERNS extended patterns
# ---------------------------------------------------------------------------


def test_all_patterns_are_unique() -> None:
    assert len(set(ALL_PATTERNS)) == len(ALL_PATTERNS)


def test_all_patterns_contain_valid_slot_types() -> None:
    valid_slots = {"L", "D", "E", "P"}
    for pattern in ALL_PATTERNS:
        assert all(c in valid_slots for c in pattern), f"Invalid slot in {pattern!r}"


def test_all_patterns_includes_standard() -> None:
    for sp in STANDARD_PATTERNS:
        assert sp in ALL_PATTERNS, f"Standard pattern {sp!r} missing from ALL_PATTERNS"


def test_all_patterns_includes_electric() -> None:
    electric = [p for p in ALL_PATTERNS if p.startswith("E")]
    assert len(electric) == 3
    assert "ELDDLLL" in ALL_PATTERNS
    assert "ELDDDLLL" in ALL_PATTERNS
    assert "ELLDDLLL" in ALL_PATTERNS


def test_all_patterns_includes_probe() -> None:
    probe = [p for p in ALL_PATTERNS if p.startswith("P")]
    assert len(probe) >= 2


def test_all_patterns_includes_diplomatic() -> None:
    assert "LLDDDDDD" in ALL_PATTERNS


def test_all_patterns_includes_army() -> None:
    army_4 = [p for p in ALL_PATTERNS if p == "LDDD"]
    assert len(army_4) == 1
