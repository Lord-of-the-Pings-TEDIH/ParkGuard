from __future__ import annotations

import pytest

from app.pipeline.plate_validator import normalize_plate


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Standard plates
        (" B-123-abc ", "B 123 ABC"),
        ("cj 05 xyz", "CJ 05 XYZ"),
        ("8123AB0", "B 123 ABO"),
        # Diplomatic
        ("C0 123 456", "CD 123 456"),
        # Army
        ("A 1S8", "A 158"),
        # Empty
        ("", "INVALID: text gol după curățare"),
        # Invalid county
        ("XX12ABC", "INVALID: județ invalid"),
        # Electric plates
        ("ECJ12ABC", "E CJ 12 ABC"),
        ("EB123XYZ", "E B 123 XYZ"),
        # Probe plates
        ("PCJ12345", "P CJ 12345"),
        ("PB1234", "P B 1234"),
        # Standard with OCR confusion in letters (0→O at end)
        ("CJ12AB0", "CJ 12 ABO"),
        # Temporary (genuinely numeric ending)
        ("CJ12345", "CJ 12345"),
    ],
)
def test_normalize_plate_cases(raw: str, expected: str) -> None:
    assert normalize_plate(raw) == expected


def test_temporary_4_digit_variant_is_rejected() -> None:
    result = normalize_plate("SJ0740")
    assert result.startswith("INVALID:")
