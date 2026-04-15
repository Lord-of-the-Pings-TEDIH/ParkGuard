from __future__ import annotations

import pytest

from app.pipeline.plate_validator import normalize_plate


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" B-123-abc ", "B 123 ABC"),
        ("cj 05 xyz", "CJ 05 XYZ"),
        ("8123AB0", "B 123 ABO"),
        ("C0 123 456", "CD 123 456"),
        ("A 1S8", "A 158"),
        ("", "INVALID: text gol după curățare"),
        ("TM9QQQ", "INVALID: format standard: prefix prea scurt"),
        ("XX12ABC", "INVALID: județ invalid"),
    ],
)
def test_normalize_plate_cases(raw: str, expected: str) -> None:
    assert normalize_plate(raw) == expected
