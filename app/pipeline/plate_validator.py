"""Romanian OCR plate cleanup, normalization and parsing.

Supports:
  - Standard plates:      CJ 12 ABC, B 123 XYZ
  - Electric plates:      E CJ 12 ABC (prefix "E" + standard body)
  - Probe plates:         P CJ 12345 (prefix "P" + county + digits)
  - Temporary plates:     CJ 12345 (county + digits only)
  - Institutional plates: MAI 12345, SPP 12345, …
  - Diplomatic plates:    CD 123 456, TC 123 456, CO 123 456
  - Army plates:          A 12345
"""

from __future__ import annotations

import re
from typing import Final, Mapping

# ---------------------------------------------------------------------------
# Visual-confusion dictionaries
# ---------------------------------------------------------------------------

NUM_TO_LET: Final[dict[str, str]] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",  # was B — ambiguous with 8→B; E is visually closer
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "G",  # secondary: 9 resembles lowercase g → G
}

LET_TO_NUM: Final[dict[str, str]] = {
    "O": "0",
    "I": "1",
    "Z": "2",
    "E": "3",  # paired with 3→E above
    "B": "8",
    "S": "5",
    "G": "6",
    "T": "7",
    "A": "4",
    "Q": "0",
    "D": "0",  # OCR classic: D looks like 0
}

# ---------------------------------------------------------------------------
# County codes
# ---------------------------------------------------------------------------

COUNTY_CODES: Final[frozenset[str]] = frozenset(
    {
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
    }
)

_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Z0-9]")
_INSTITUTION_PREFIXES: Final[frozenset[str]] = frozenset({"MAI", "SPP", "GUV", "SEN"})
_DIPLO_PREFIXES: Final[frozenset[str]] = frozenset({"CD", "TC", "CO"})
_INSTITUTION_EQUIV: Final[dict[str, str]] = {"C": "G", **NUM_TO_LET}
_DIPLO_FORCED_PREFIX: Final[dict[str, str]] = {
    "CD": "CD",
    "TC": "TC",
    "CO": "CO",
    "C0": "CD",
    "T0": "TC",
}

# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------


def _invalid(reason: str) -> str:
    return f"INVALID: {reason}"


def is_invalid_plate(normalized: str) -> bool:
    return normalized.startswith("INVALID:")


def compact_plate(plate_text: str) -> str:
    return _NON_ALNUM_RE.sub("", plate_text.upper())


def apply_correction(text: str, mapping: Mapping[str, str]) -> str:
    return "".join(mapping.get(char, char) for char in text)


def is_valid_county(county: str) -> bool:
    return county in COUNTY_CODES


def extract_county(compact_plate_text: str) -> str | None:
    """Return the Romanian county code embedded in a compact plate, or None.

    Operates on the *compact* form (no spaces) — the same shape stored in
    ``Plate.normalized_text``.  Plate families that don't carry a county
    (institution / diplomatic / army) return ``None`` so the registry UI
    doesn't render a misleading "county" badge for them.
    """
    if not compact_plate_text:
        return None

    text = compact_plate_text.upper()

    # Institution / diplomatic / army plates have no county.
    if len(text) >= 3 and text[:3] in _INSTITUTION_PREFIXES:
        return None
    if len(text) >= 2 and text[:2] in _DIPLO_PREFIXES:
        return None
    if len(text) >= 2 and text[0] == "A" and text[1].isdigit():
        return None

    # Electric (E + body) and probe (P + body) — strip the prefix and recurse.
    if len(text) >= 4 and text[0] in {"E", "P"}:
        nested = extract_county(text[1:])
        if nested is not None:
            return nested

    # Bucharest plates always start with a single ``B`` followed by digits.
    if text[0] == "B" and len(text) >= 2 and text[1].isdigit():
        return "B"

    if len(text) >= 2 and text[:2] in COUNTY_CODES:
        return text[:2]
    return None


# ---------------------------------------------------------------------------
# County normalization (with fuzzy OCR correction)
# ---------------------------------------------------------------------------


def _county_char_candidates(char: str) -> tuple[str, ...]:
    base = [char]
    mapped = NUM_TO_LET.get(char)
    if mapped and mapped not in base:
        base.append(mapped)
    if char == "1" and "J" not in base:
        base.append("J")
    if char == "0":
        if "C" not in base:
            base.append("C")
        if "O" not in base:
            base.append("O")
    if char == "3" and "E" not in base:
        base.append("E")
    if char == "D" and "0" not in base:
        # D commonly misread as 0
        pass  # D is a letter, valid in county codes already
    return tuple(base)


def _normalize_county(raw_county: str) -> str:
    if not raw_county:
        return raw_county
    if raw_county in COUNTY_CODES:
        return raw_county
    if len(raw_county) == 1:
        return apply_correction(raw_county, NUM_TO_LET)

    first_options = _county_char_candidates(raw_county[0])
    second_options = _county_char_candidates(raw_county[1])
    for first in first_options:
        for second in second_options:
            candidate = f"{first}{second}"
            if candidate in COUNTY_CODES:
                return candidate
    return apply_correction(raw_county, NUM_TO_LET)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def _clean(raw_text: str) -> str:
    return _NON_ALNUM_RE.sub("", raw_text.upper())


# ---------------------------------------------------------------------------
# Institution plates: MAI 12345, SPP 12345, GUV 12345, SEN 12345
# ---------------------------------------------------------------------------


def _detect_institution_prefix(cleaned: str) -> str | None:
    if len(cleaned) < 3:
        return None
    candidate = apply_correction(cleaned[:3], _INSTITUTION_EQUIV)
    if candidate in _INSTITUTION_PREFIXES:
        return candidate
    return None


def _parse_institution(cleaned: str, prefix: str) -> str:
    digits = apply_correction(cleaned[3:], LET_TO_NUM)
    if not digits.isdigit():
        return _invalid("format instituție: sufix nenumeric")
    if not 3 <= len(digits) <= 5:
        return _invalid("format instituție: număr de cifre invalid")
    return f"{prefix} {digits}"


# ---------------------------------------------------------------------------
# Diplomatic plates: CD 123 456, TC 123 456, CO 123 456
# ---------------------------------------------------------------------------


def _detect_diplomatic_prefix(cleaned: str) -> str | None:
    if len(cleaned) < 2:
        return None
    return _DIPLO_FORCED_PREFIX.get(cleaned[:2])


def _parse_diplomatic(cleaned: str, prefix: str) -> str:
    digits = apply_correction(cleaned[2:], LET_TO_NUM)
    if not digits.isdigit():
        return _invalid("format diplomatic: sufix nenumeric")
    if len(digits) != 6:
        return _invalid("format diplomatic: necesare exact 6 cifre")
    return f"{prefix} {digits[:3]} {digits[3:]}"


# ---------------------------------------------------------------------------
# Army plates: A 12345
# ---------------------------------------------------------------------------


def _is_army_candidate(cleaned: str) -> bool:
    if not cleaned or cleaned[0] not in {"A", "4"}:
        return False
    if not 4 <= len(cleaned) <= 8:
        return False
    if len(cleaned) >= 6 and cleaned[-3:].isalpha():
        return False
    numeric_rest = apply_correction(cleaned[1:], LET_TO_NUM)
    digit_count = sum(char.isdigit() for char in numeric_rest)
    return digit_count >= len(numeric_rest) - 1


def _parse_army(cleaned: str) -> str:
    digits = apply_correction(cleaned[1:], LET_TO_NUM)
    if not digits.isdigit():
        return _invalid("format armată: sufix nenumeric")
    if not 3 <= len(digits) <= 7:
        return _invalid("format armată: număr de cifre invalid")
    return f"A {digits}"


# ---------------------------------------------------------------------------
# Electric plates: E + standard body  (e.g. ECJ12ABC → E CJ 12 ABC)
# ---------------------------------------------------------------------------


def _is_electric_candidate(cleaned: str) -> bool:
    """Detect 'E' prefix followed by a standard plate body."""
    if not cleaned or cleaned[0] not in {"E", "3"}:
        return False
    if len(cleaned) < 7:
        return False
    # Check if removing 'E' would leave a valid standard plate body
    body = cleaned[1:]
    # Body must end with 3 chars that can be letters
    tail3 = apply_correction(body[-3:], NUM_TO_LET)
    if not tail3.isalpha():
        return False
    prefix = body[:-3]
    if not prefix:
        return False
    # Must have county (1-2 chars) + digits (2-3)
    if prefix[0] in {"B", "8"}:
        digits = apply_correction(prefix[1:], LET_TO_NUM)
        return 2 <= len(digits) <= 3 and digits.isdigit()
    if len(prefix) >= 4:
        digits = apply_correction(prefix[2:], LET_TO_NUM)
        return len(digits) == 2 and digits.isdigit()
    return False


def _parse_electric(cleaned: str) -> str:
    """Parse an electric-vehicle plate: E + standard body."""
    body = cleaned[1:]
    standard_result = _parse_standard(body)
    if is_invalid_plate(standard_result):
        return _invalid("format electric: corp standard invalid")
    return f"E {standard_result}"


# ---------------------------------------------------------------------------
# Probe (test-drive) plates: P + county + digits (e.g. PCJ12345 → P CJ 12345)
# ---------------------------------------------------------------------------


def _is_probe_candidate(cleaned: str) -> bool:
    """Detect 'P' prefix followed by county + digits."""
    if not cleaned or cleaned[0] != "P":
        return False
    if len(cleaned) < 5:
        return False
    rest = cleaned[1:]
    # Check if rest starts with a valid county and the remainder is digits
    if rest[0] in {"B", "8"}:
        digits = apply_correction(rest[1:], LET_TO_NUM)
        return len(digits) >= 3 and digits.isdigit()
    if len(rest) >= 3:
        county = _normalize_county(rest[:2])
        if is_valid_county(county):
            digits = apply_correction(rest[2:], LET_TO_NUM)
            return len(digits) >= 3 and digits.isdigit()
    return False


def _parse_probe(cleaned: str) -> str:
    """Parse a probe (test-drive) plate: P + county + digits."""
    rest = cleaned[1:]
    if rest[0] in {"B", "8"}:
        county = "B"
        digits = apply_correction(rest[1:], LET_TO_NUM)
    else:
        county = _normalize_county(rest[:2])
        digits = apply_correction(rest[2:], LET_TO_NUM)

    if not is_valid_county(county):
        return _invalid("format probă: județ invalid")
    if not digits.isdigit():
        return _invalid("format probă: sufix nenumeric")
    if len(digits) < 3:
        return _invalid("format probă: prea puține cifre")
    return f"P {county} {digits}"


# ---------------------------------------------------------------------------
# Temporary plates: county + digits only (no trailing letters)
# ---------------------------------------------------------------------------


def _looks_like_temporary(cleaned: str) -> bool:
    if len(cleaned) < 4:
        return False
    # At this point, _parse_standard has already been tried and failed.
    # So we only need to check if the suffix looks numeric.
    suffix = cleaned[-3:]
    if not apply_correction(suffix, LET_TO_NUM).isdigit():
        return False
    return sum(char.isdigit() for char in suffix) >= 2


def _looks_like_county_plus_5_or_6_digits(cleaned: str) -> bool:
    """Detect county-prefixed plates that look like special numeric families.

    Examples:
      - SJ07404   -> SJ + 5 digits
      - B123456   -> B + 6 digits

    These should not be coerced into standard ``county + 2/3 digits + 3 letters``.
    """
    if not cleaned:
        return False

    if cleaned[0] in {"B", "8"}:
        suffix = cleaned[1:]
    else:
        if len(cleaned) < 7:
            return False
        county = _normalize_county(cleaned[:2])
        if not is_valid_county(county):
            return False
        suffix = cleaned[2:]

    if len(suffix) not in {5, 6}:
        return False

    corrected = apply_correction(suffix, LET_TO_NUM)
    if not corrected.isdigit():
        return False

    # Allow at most one non-digit OCR-confusable character in numeric temporary
    # families (e.g. O→0), but reject mixed alpha tails like "12AB0".
    non_digit_chars = [char for char in suffix if not char.isdigit()]
    if len(non_digit_chars) > 1:
        return False
    return all(char in LET_TO_NUM for char in non_digit_chars)


def _parse_temporary(cleaned: str) -> str:
    if cleaned[0] in {"B", "8"}:
        county = "B"
        digits = apply_correction(cleaned[1:], LET_TO_NUM)
    else:
        if len(cleaned) < 3:
            return _invalid("format provizoriu: prea scurt")
        county = _normalize_county(cleaned[:2])
        digits = apply_correction(cleaned[2:], LET_TO_NUM)

    if not is_valid_county(county):
        return _invalid("județ invalid")
    if not digits.isdigit():
        return _invalid("format provizoriu: sufix nenumeric")
    if len(digits) not in {5, 6}:
        return _invalid("format provizoriu: necesare 5 sau 6 cifre")
    return f"{county} {digits}"


# ---------------------------------------------------------------------------
# Standard plates: CJ 12 ABC, B 123 XYZ
# ---------------------------------------------------------------------------


def _parse_standard(cleaned: str, *, plate_type: str | None = None) -> str:
    if len(cleaned) < 6:
        return _invalid("format standard: prea scurt")

    tail3 = cleaned[-3:]

    # Guard: if the original tail has ≥2 digits, the conversion to letters
    # (e.g. "345"→"EAS") is almost certainly a false positive.  This plate
    # is temporary, not standard.
    tail3_digit_count = sum(c.isdigit() for c in tail3)
    if tail3_digit_count >= 2:
        # County + 5/6 digits is a dedicated numeric family (temporary-like),
        # never force it into a standard 3-letter tail even with a color hint.
        if _looks_like_county_plus_5_or_6_digits(cleaned):
            return _invalid("format standard: candidat special numeric (5/6 cifre)")
        # For non-standard color hints (or no hint), treat digit-heavy tails
        # as non-standard and allow temporary/special parsers to handle them.
        if plate_type != "standard":
            return _invalid("format standard: ultimele 3 caractere sunt predominant cifre")
    letters = apply_correction(tail3, NUM_TO_LET)
    if not letters.isalpha():
        return _invalid("format standard: ultimele 3 caractere trebuie litere")

    prefix = cleaned[:-3]
    if not prefix:
        return _invalid("format standard: prefix lipsă")

    if prefix[0] in {"B", "8"}:
        county = "B"
        digits = apply_correction(prefix[1:], LET_TO_NUM)
        if not 2 <= len(digits) <= 3:
            return _invalid("format standard București: necesare 2-3 cifre")
    else:
        if len(prefix) < 4:
            return _invalid("format standard: prefix prea scurt")
        county = _normalize_county(prefix[:2])
        digits = apply_correction(prefix[2:], LET_TO_NUM)
        if len(digits) != 2:
            return _invalid("format standard: necesare 2 cifre după județ")

    if not is_valid_county(county):
        return _invalid("județ invalid")
    if not digits.isdigit():
        return _invalid("format standard: segment numeric invalid")
    return f"{county} {digits} {letters}"


# ---------------------------------------------------------------------------
# Main entry point — priority order matters!
# ---------------------------------------------------------------------------


def normalize_plate(raw_text: str, *, plate_type: str | None = None) -> str:
    """Normalize raw OCR text into a canonical Romanian plate format.

    Parameters
    ----------
    raw_text:
        Raw OCR output string (may contain hyphens, dots, spaces).
    plate_type:
        Optional color-based plate type hint (e.g. ``"standard"``,
        ``"temporary"``, ``"diplomatic"``, ``"probe"``, ``"electric"``).
        When provided, the parser favours the hinted format, resolving
        ambiguity between standard (3-letter tail) and temporary (all-digit
        tail) plates.

    Priority order:
      1. Institution (MAI, SPP, GUV, SEN)
      2. Diplomatic (CD, TC, CO)
      3. Army (A + digits)
      4. Electric (E + standard body)
      5. Probe (P + county + digits)
      6. Standard (county + digits + 3 letters)  ← BEFORE temporary
      7. Temporary (county + digits only)

    The standard check is placed BEFORE temporary to prevent misclassifying
    plates like "CJ01234" as temporary when they are actually "CJ 01 EBA"
    with OCR confusions in the final 3 characters.

    When *plate_type* == ``"temporary"`` (detected via red text color),
    the standard check is skipped entirely → temporary is tried first.
    """
    cleaned = _clean(raw_text)
    if not cleaned:
        return _invalid("text gol după curățare")

    # 1. Institution
    institution_prefix = _detect_institution_prefix(cleaned)
    if institution_prefix:
        return _parse_institution(cleaned, institution_prefix)

    # 2. Diplomatic
    diplomatic_prefix = _detect_diplomatic_prefix(cleaned)
    if diplomatic_prefix:
        return _parse_diplomatic(cleaned, diplomatic_prefix)

    # 3. Army
    if _is_army_candidate(cleaned):
        return _parse_army(cleaned)

    # 4. Electric
    if _is_electric_candidate(cleaned):
        return _parse_electric(cleaned)

    # 5. Probe
    if _is_probe_candidate(cleaned):
        return _parse_probe(cleaned)

    # 5b. Special numeric families (county + 5/6 digits)
    # Do this before the "standard" branch so these cases are not coerced
    # into 3-letter tails.
    if _looks_like_county_plus_5_or_6_digits(cleaned):
        return _parse_temporary(cleaned)

    # --- Color hint short-circuits ---

    # When color says "temporary" (red text), skip standard entirely.
    if plate_type == "temporary":
        if _looks_like_temporary(cleaned):
            return _parse_temporary(cleaned)
        # Still fall through to standard as safety net.

    # When color says "standard" (black text), prefer standard but still
    # allow temporary fallback — the color classifier may misidentify
    # crops with lots of dark background around the plate.
    if plate_type == "standard":
        standard_result = _parse_standard(cleaned, plate_type=plate_type)
        if not is_invalid_plate(standard_result):
            return standard_result
        # Fall through to default logic (including temporary).

    # --- Default priority (no hint or unhandled hint) ---

    # 6. Standard — try FIRST before temporary
    standard_result = _parse_standard(cleaned, plate_type=plate_type)
    if not is_invalid_plate(standard_result):
        return standard_result

    # 7. Temporary — fallback when standard doesn't match
    if _looks_like_temporary(cleaned):
        return _parse_temporary(cleaned)

    # If nothing matched, return the standard result (which is INVALID: ...)
    return standard_result
