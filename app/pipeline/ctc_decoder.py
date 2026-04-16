"""app/pipeline/ctc_decoder.py

Prefix Beam Search decoder guided by the Romanian plate FSM.

Instead of greedy argmax  →  post-hoc regex repair, this decoder maintains
multiple character-level hypotheses simultaneously.  At every step the FSM
restricts which characters are *permitted* (letter vs. digit vs. fixed
literal), so classic OCR confusions (0↔O, 1↔I, 8↔B …) are resolved
*mathematically* by the beam rather than by fragile substitution heuristics.

Supports all Romanian plate formats:
  - Standard (LDDLLL, LDDDLLL, LLDDLLL)
  - Electric (ELDDLLL, ELDDDLLL, ELLDDLLL)
  - Probe (PLDDDDD, PLLDDDDD, …)
  - Institutional (LLLDDD .. LLLDDDDD)
  - Diplomatic (LLDDDDDD)
  - Army (LDDD .. LDDDDDDD)

Usage
-----
    from app.pipeline.ctc_decoder import fsm_beam_decode

    corrected_text, adjusted_conf = fsm_beam_decode(raw_ocr_text, ocr_confidence)
"""

from __future__ import annotations

import logging
from typing import Final

from app.pipeline.fsm_plate import (
    ALL_PATTERNS,
    PATTERN_TAGS,
    char_candidates,
    slot_accepts,
)
from app.pipeline.plate_validator import is_invalid_plate, normalize_plate

logger = logging.getLogger(__name__)

_DEFAULT_BEAM_WIDTH: Final[int] = 5

# Beam element type alias for clarity:
#   (prefix_text, cumulative_similarity_score, active_pattern_states)
# active_pattern_states: list[(pattern_index, consumed_positions)]
_BeamElem = tuple[str, float, list[tuple[int, int]]]


def fsm_beam_decode(
    raw_text: str,
    confidence: float,
    *,
    beam_width: int = _DEFAULT_BEAM_WIDTH,
    plate_type: str | None = None,
) -> tuple[str, float]:
    """Decode *raw_text* using a Prefix Beam Search guided by the plate FSM.

    Parameters
    ----------
    raw_text:
        Compacted (no spaces, uppercase, alphanumeric only) OCR output.
    confidence:
        OCR confidence score (0–1) from the underlying model.
    beam_width:
        Maximum number of hypotheses maintained at each decoding step.
        Higher values improve accuracy at the cost of CPU time.

    Returns
    -------
    tuple[str, float]
        ``(best_text, adjusted_confidence)`` where ``best_text`` is the
        FSM-corrected compacted plate string and ``adjusted_confidence``
        is the original confidence scaled by the beam path's cumulative
        similarity score.

        Falls back to ``(raw_text, confidence)`` when all beam paths are
        killed by the FSM (truly undecodable input).
    plate_type:
        Optional color-based plate type hint (e.g. ``"standard"``,
        ``"temporary"``).  When provided, only patterns tagged with this
        type are considered by the beam, eliminating cross-type ambiguity.
    """
    if not raw_text:
        return raw_text, confidence

    n = len(raw_text)

    # Find all patterns that match this input length.
    active_pattern_states: list[tuple[int, int]] = [
        (pi, 0)
        for pi, pat in enumerate(ALL_PATTERNS)
        if len(pat) == n
    ]

    if not active_pattern_states:
        logger.debug(
            "FSM beam: length %d matches no pattern; passing through", n
        )
        return raw_text, confidence

    # Soft preference: when plate_type is provided, patterns that match
    # get a score bonus.  Non-matching patterns are NOT excluded — just
    # slightly penalized.  This avoids catastrophic errors when color
    # classification is wrong, while still biasing toward the right format.
    _HINT_BONUS: float = 1.3
    pattern_bonus: dict[int, float] = {}
    if plate_type:
        for pi, _ in active_pattern_states:
            tag = PATTERN_TAGS.get(ALL_PATTERNS[pi], "")
            pattern_bonus[pi] = _HINT_BONUS if tag == plate_type else 1.0

    # -----------------------------------------------------------------------
    # Initialise beam: one element with all compatible starting patterns.
    # -----------------------------------------------------------------------
    beam: list[_BeamElem] = [("", 1.0, active_pattern_states)]

    for char in raw_text:
        new_beam: list[_BeamElem] = []

        for prefix, sim_score, states in beam:
            for cand_char, cand_sim in char_candidates(char):
                # Keep only pattern states that accept this candidate char.
                next_states: list[tuple[int, int]] = [
                    (pi, pos + 1)
                    for pi, pos in states
                    if pos < len(ALL_PATTERNS[pi])
                    and slot_accepts(ALL_PATTERNS[pi][pos], cand_char)
                ]
                if next_states:
                    new_beam.append(
                        (prefix + cand_char, sim_score * cand_sim, next_states)
                    )

        if not new_beam:
            # All hypotheses were killed — FSM cannot decode this sequence.
            logger.debug(
                "FSM beam: all paths killed at char=%r in %r; falling back",
                char,
                raw_text,
            )
            return raw_text, confidence

        # Prune beam to top-k by cumulative similarity score.
        new_beam.sort(key=lambda e: e[1], reverse=True)
        beam = new_beam[:beam_width]

    # -----------------------------------------------------------------------
    # Select best *complete* (terminal) path that yields a valid plate.
    # Apply pattern_bonus for soft color-hint preference.
    # -----------------------------------------------------------------------
    best_text = ""
    best_score = 0.0

    for prefix, sim_score, states in beam:
        for pi, pos in states:
            if pos == len(ALL_PATTERNS[pi]):  # Pattern fully consumed
                normalized = normalize_plate(prefix, plate_type=plate_type)
                if not is_invalid_plate(normalized):
                    bonus = pattern_bonus.get(pi, 1.0)
                    adjusted = confidence * sim_score * bonus
                    if adjusted > best_score:
                        best_score = adjusted
                        best_text = prefix
                break  # At most one terminal state per beam element

    if best_text:
        logger.debug(
            "FSM beam: %r → %r  (conf %.3f → %.3f, hint=%s)",
            raw_text,
            best_text,
            confidence,
            best_score,
            plate_type,
        )
        return best_text, best_score

    # -----------------------------------------------------------------------
    # No valid plate found — return highest-scoring hypothesis and let the
    # downstream normaliser make the final decision.
    # -----------------------------------------------------------------------
    best_prefix, best_sim, _ = beam[0]
    logger.debug(
        "FSM beam: no valid plate in beam; best guess %r (sim=%.3f)",
        best_prefix,
        best_sim,
    )
    return best_prefix, confidence * best_sim

