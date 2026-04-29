"""Pydantic schema validation for ParkingSpotIn and ParkingTicketIn.

Verifies that:
- assigned_plate / plate_text are normalised to canonical compact form
- invalid plate strings are rejected with 422-equivalent ValueError
- allowed_plates list is normalised element-by-element
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.parking import ParkingSpotIn, ParkingTicketIn
from datetime import datetime, timezone


NOW = datetime.now(timezone.utc)


class TestParkingSpotIn:
    def test_valid_plate_canonicalised(self) -> None:
        spot = ParkingSpotIn(
            spot_label="A-01",
            parking_lot_id=1,
            assigned_plate="B 11 AAA",
        )
        assert spot.assigned_plate == "B11AAA"

    def test_lowercase_plate_canonicalised(self) -> None:
        spot = ParkingSpotIn(
            spot_label="A-02",
            parking_lot_id=1,
            assigned_plate="cj 05 xyz",
        )
        assert spot.assigned_plate == "CJ05XYZ"

    def test_garbage_plate_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ParkingSpotIn(
                spot_label="A-03",
                parking_lot_id=1,
                assigned_plate="garbage123!!",
            )
        assert "assigned_plate" in str(exc_info.value).lower() or "valid" in str(exc_info.value).lower()

    def test_none_plate_accepted(self) -> None:
        spot = ParkingSpotIn(spot_label="A-04", parking_lot_id=1)
        assert spot.assigned_plate is None

    def test_allowed_plates_normalised(self) -> None:
        spot = ParkingSpotIn(
            spot_label="A-05",
            parking_lot_id=1,
            assigned_plate="B11AAA",
            allowed_plates=["B 22 BBB", "cj 05 xyz"],
        )
        assert spot.allowed_plates == ["B22BBB", "CJ05XYZ"]

    def test_allowed_plates_invalid_entry_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ParkingSpotIn(
                spot_label="A-06",
                parking_lot_id=1,
                assigned_plate="B11AAA",
                allowed_plates=["!!!!"],
            )

    def test_allowed_plates_none_accepted(self) -> None:
        spot = ParkingSpotIn(spot_label="A-07", parking_lot_id=1, allowed_plates=None)
        assert spot.allowed_plates is None


class TestParkingTicketIn:
    def test_valid_plate_canonicalised(self) -> None:
        ticket = ParkingTicketIn(
            plate_text="B 11 AAA",
            valid_from=NOW,
            valid_until=NOW,
            ticket_ref="TKT-X",
        )
        assert ticket.plate_text == "B11AAA"

    def test_garbage_plate_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ParkingTicketIn(
                plate_text="not-a-plate",
                valid_from=NOW,
                valid_until=NOW,
                ticket_ref="TKT-Y",
            )
