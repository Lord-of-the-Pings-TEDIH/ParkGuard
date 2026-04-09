"""Romanian plate normalization and structural validation."""

from __future__ import annotations

import re
from typing import Final

COUNTY_CODES: Final[frozenset[str]] = frozenset(
    {
        "AB",
        "AR",
        "AG",
        "BC",
        "BH",
        "BN",
        "BT",
        "BV",
        "BR",
        "B",
        "BZ",
        "CS",
        "CL",
        "CJ",
        "CT",
        "CV",
        "DB",
        "DJ",
        "GL",
        "GR",
        "GJ",
        "HR",
        "HD",
        "IL",
        "IS",
        "IF",
        "MM",
        "MH",
        "MS",
        "NT",
        "OT",
        "PH",
        "SM",
        "SJ",
        "SB",
        "SV",
        "TR",
        "TM",
        "TL",
        "VS",
        "VL",
        "VN",
    }
)
STANDARD_RE: Final[re.Pattern[str]] = re.compile(
    r"^(B\d{2,3}[A-Z]{3}|[A-Z]{2}\d{2}[A-Z]{3})$"
)

_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Z0-9]")
_BUCHAREST_RE: Final[re.Pattern[str]] = re.compile(r"^B\d{2,3}[A-Z]{3}$")
_B_PREFIX_DIGIT_HINTS: Final[frozenset[str]] = frozenset("0123456789OISB")
_COUNTY_CODE_CORRECTIONS: Final[dict[str, str]] = {"0": "O", "1": "I", "8": "B"}
_DIGIT_CORRECTIONS: Final[dict[str, str]] = {"O": "0", "I": "1", "S": "5", "B": "8"}
_LETTER_CORRECTIONS: Final[dict[str, str]] = {"0": "O", "1": "I", "5": "S", "8": "B"}


def _replace_positions(
    chars: list[str], positions: tuple[int, ...], replacements: dict[str, str]
) -> None:
    for idx in positions:
        if idx < len(chars):
            chars[idx] = replacements.get(chars[idx], chars[idx])


def _county_char_candidates(char: str) -> tuple[str, ...]:
    if char == "0":
        return ("O", "C")
    corrected = _COUNTY_CODE_CORRECTIONS.get(char, char)
    return (corrected,)


def _correct_county_code(county_code: str) -> str:
    if len(county_code) != 2:
        return county_code

    first_candidates = _county_char_candidates(county_code[0])
    second_candidates = _county_char_candidates(county_code[1])
    for first_char in first_candidates:
        for second_char in second_candidates:
            candidate = f"{first_char}{second_char}"
            if candidate in COUNTY_CODES:
                return candidate

    return "".join(_COUNTY_CODE_CORRECTIONS.get(char, char) for char in county_code)


def _apply_ocr_corrections(text: str) -> str:
    if len(text) != 7:
        return text

    chars = list(text)
    is_b_prefix_shape = (
        _COUNTY_CODE_CORRECTIONS.get(chars[0], chars[0]) == "B"
        and all(chars[idx] in _B_PREFIX_DIGIT_HINTS for idx in (1, 2, 3))
    )
    is_standard_shape = all(chars[idx] in _B_PREFIX_DIGIT_HINTS for idx in (2, 3))

    if is_b_prefix_shape:
        _replace_positions(chars, (0,), _COUNTY_CODE_CORRECTIONS)
        _replace_positions(chars, (1, 2, 3), _DIGIT_CORRECTIONS)
    elif is_standard_shape:
        corrected_county = _correct_county_code("".join(chars[:2]))
        chars[0], chars[1] = corrected_county[0], corrected_county[1]
        _replace_positions(chars, (2, 3), _DIGIT_CORRECTIONS)
    else:
        return text

    _replace_positions(chars, (4, 5, 6), _LETTER_CORRECTIONS)

    return "".join(chars)


def _county_code(plate: str) -> str:
    if _BUCHAREST_RE.fullmatch(plate):
        return "B"
    return plate[:2]


def normalize_plate(raw: str) -> tuple[str, bool]:
    normalized = _NON_ALNUM_RE.sub("", raw.strip().upper())
    if not normalized:
        return normalized, False

    normalized = _apply_ocr_corrections(normalized)

    if not STANDARD_RE.fullmatch(normalized):
        return normalized, False

    return normalized, _county_code(normalized) in COUNTY_CODES
