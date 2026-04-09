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
    r"^(B\d{3}[A-Z]{3}|[A-Z]{2}\d{2}[A-Z]{3})$"
)

_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Z0-9]")
_BUCHAREST_RE: Final[re.Pattern[str]] = re.compile(r"^B\d{3}[A-Z]{3}$")
_B_PREFIX_DIGIT_HINTS: Final[frozenset[str]] = frozenset("0123456789O")


def _apply_corrections(plate: str, *, bucharest: bool) -> str:
    chars = list(plate)
    if bucharest:
        numeric_positions = (1, 2, 3)
        letter_positions = (4, 5, 6)
    else:
        numeric_positions = (2, 3)
        letter_positions = (0, 1, 4, 5, 6)

    for idx in numeric_positions:
        if idx < len(chars) and chars[idx] == "O":
            chars[idx] = "0"

    for idx in letter_positions:
        if idx < len(chars) and chars[idx] == "0":
            chars[idx] = "O"

    return "".join(chars)


def _county_code(plate: str) -> str:
    if _BUCHAREST_RE.fullmatch(plate):
        return "B"
    return plate[:2]


def normalize_plate(raw: str) -> tuple[str, bool]:
    normalized = _NON_ALNUM_RE.sub("", raw.strip().upper())
    if not normalized:
        return normalized, False

    is_bucharest_shape = (
        len(normalized) == 7
        and normalized.startswith("B")
        and normalized[1] in _B_PREFIX_DIGIT_HINTS
    )
    normalized = _apply_corrections(normalized, bucharest=is_bucharest_shape)

    if not STANDARD_RE.fullmatch(normalized):
        return normalized, False

    return normalized, _county_code(normalized) in COUNTY_CODES
