"""Backward-compatible imports for plate normalization."""

from __future__ import annotations

from app.pipeline.plate_validator import COUNTY_CODES as ROMANIAN_COUNTY_CODES
from app.pipeline.plate_validator import normalize_plate

__all__ = ["ROMANIAN_COUNTY_CODES", "normalize_plate"]
