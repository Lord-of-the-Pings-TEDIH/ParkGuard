"""Minimum-events gate — _should_flag returns False when event_count is below
OCCUPANCY_MIN_EVENTS_FOR_FLAG, regardless of score."""

from __future__ import annotations

from unittest.mock import patch

import pytest

import app.services.occupancy as occ_mod
from app.services.occupancy import _should_flag, FLAG_THRESHOLD


class TestShouldFlag:
    def test_score_above_threshold_no_gate(self) -> None:
        with patch.object(occ_mod, "_MIN_EVENTS_FOR_FLAG", 0):
            assert _should_flag(FLAG_THRESHOLD + 1.0, event_count=1) is True

    def test_score_below_threshold(self) -> None:
        assert _should_flag(FLAG_THRESHOLD - 1.0, event_count=100) is False

    def test_gate_blocks_even_when_score_high(self) -> None:
        with patch.object(occ_mod, "_MIN_EVENTS_FOR_FLAG", 20):
            assert _should_flag(FLAG_THRESHOLD + 5.0, event_count=5) is False

    def test_gate_passes_when_count_meets_minimum(self) -> None:
        with patch.object(occ_mod, "_MIN_EVENTS_FOR_FLAG", 20):
            assert _should_flag(FLAG_THRESHOLD + 5.0, event_count=20) is True

    def test_gate_zero_disables_minimum(self) -> None:
        with patch.object(occ_mod, "_MIN_EVENTS_FOR_FLAG", 0):
            # Gate disabled → only score matters
            assert _should_flag(FLAG_THRESHOLD + 1.0, event_count=1) is True

    def test_gate_passes_at_exact_minimum(self) -> None:
        with patch.object(occ_mod, "_MIN_EVENTS_FOR_FLAG", 5):
            assert _should_flag(FLAG_THRESHOLD + 1.0, event_count=5) is True

    def test_gate_blocks_one_below_minimum(self) -> None:
        with patch.object(occ_mod, "_MIN_EVENTS_FOR_FLAG", 5):
            assert _should_flag(FLAG_THRESHOLD + 1.0, event_count=4) is False
