"""Romanian plate normalization and structural validation."""

from __future__ import annotations

import re
from typing import Final

ROMANIAN_COUNTY_CODES: Final[frozenset[str]] = frozenset(
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
_BUCHAREST_RE: Final[re.Pattern[str]] = re.compile(r"^B\d{2,3}[A-Z]{3}$")
_COUNTY_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{3}$")


def normalize_plate(raw_text: str) -> tuple[str, bool]:
    """Normalize OCR text and validate basic Romanian plate structure."""
    normalized = _NON_ALNUM_RE.sub("", raw_text.upper())
    if not normalized:
        return normalized, False

    # Common OCR confusion in county position: '0J' -> 'CJ'.
    if normalized[0] == "0":
        normalized = f"C{normalized[1:]}"

    if _BUCHAREST_RE.fullmatch(normalized):
        return normalized, True

    if _COUNTY_RE.fullmatch(normalized) and normalized[:2] in ROMANIAN_COUNTY_CODES:
        return normalized, True

    return normalized, False
