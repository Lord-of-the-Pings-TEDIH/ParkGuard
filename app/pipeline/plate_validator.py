"""Romanian OCR plate cleanup, normalization and parsing."""

from __future__ import annotations

import re
from typing import Final, Mapping

NUM_TO_LET: Final[dict[str, str]] = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "B",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}

LET_TO_NUM: Final[dict[str, str]] = {
    "O": "0",
    "I": "1",
    "Z": "2",
    "B": "8",
    "S": "5",
    "G": "6",
    "T": "7",
    "A": "4",
    "Q": "0",
    "l": "1",
}

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
_INSTITUTION_EQUIV: Final[dict[str, str]] = {"C": "G", **NUM_TO_LET}
_DIPLO_FORCED_PREFIX: Final[dict[str, str]] = {
    "CD": "CD",
    "TC": "TC",
    "CO": "CO",
    "C0": "CD",
    "T0": "TC",
}


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


def _clean(raw_text: str) -> str:
    return _NON_ALNUM_RE.sub("", raw_text.upper())


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


def _looks_like_temporary(cleaned: str) -> bool:
    if len(cleaned) < 4:
        return False
    suffix = cleaned[-3:]
    if not apply_correction(suffix, LET_TO_NUM).isdigit():
        return False
    return sum(char.isdigit() for char in suffix) >= 2


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
    if len(digits) < 3:
        return _invalid("format provizoriu: prea puține cifre")
    return f"{county} {digits}"


def _parse_standard(cleaned: str) -> str:
    if len(cleaned) < 6:
        return _invalid("format standard: prea scurt")

    letters = apply_correction(cleaned[-3:], NUM_TO_LET)
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


def normalize_plate(raw_text: str) -> str:
    cleaned = _clean(raw_text)
    if not cleaned:
        return _invalid("text gol după curățare")

    institution_prefix = _detect_institution_prefix(cleaned)
    if institution_prefix:
        return _parse_institution(cleaned, institution_prefix)

    diplomatic_prefix = _detect_diplomatic_prefix(cleaned)
    if diplomatic_prefix:
        return _parse_diplomatic(cleaned, diplomatic_prefix)

    if _is_army_candidate(cleaned):
        return _parse_army(cleaned)

    if _looks_like_temporary(cleaned):
        return _parse_temporary(cleaned)

    return _parse_standard(cleaned)
