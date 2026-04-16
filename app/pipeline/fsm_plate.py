"""app/pipeline/fsm_plate.py

Finite State Machine (FSM) for Romanian license-plate positional grammar.

Each plate pattern is encoded as a string of slot types:
  'L' — this position *must* be a letter (A-Z)
  'D' — this position *must* be a digit (0-9)

The FSM is used by the Prefix Beam Search decoder (ctc_decoder.py) to
constrain character choices at every position, eliminating the need for
post-hoc substitution guessing.

Supported formats
-----------------
- Standard:    LDDLLL (B 12 ABC),  LDDDLLL (B 123 ABC),  LLDDLLL (CJ 12 ABC)
- Electric:    ELDDLLL, ELDDDLLL, ELLDDLLL  (E-prefix + standard body)
- Probe:       PLDDDDD (P B 12345), PLLDDDDD (P CJ 12345)
- Institution: LLLDDD, LLLDDDD, LLLDDDDD  (MAI/SPP/GUV/SEN + 3-5 digits)
- Diplomatic:  LLDDDDDD  (CD/TC/CO + 6 digits)
- Army:        LDDD .. LDDDDDDD  (A + 3-7 digits)
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Character class sets
# ---------------------------------------------------------------------------

LETTERS: Final[frozenset[str]] = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS: Final[frozenset[str]] = frozenset("0123456789")

# ---------------------------------------------------------------------------
# Plate patterns (compacted, no spaces)
#
# Standard Romanian plates:
#   LDDLLL       B12ABC              B 12 ABC
#   LDDDLLL      B123ABC             B 123 ABC
#   LLDDLLL      CJ12ABC             CJ 12 ABC
#
# Electric (E + standard body):
#   ELDDLLL      EB12ABC             E B 12 ABC
#   ELDDDLLL     EB123ABC            E B 123 ABC
#   ELLDDLLL     ECJ12ABC            E CJ 12 ABC
#
# Probe (P + county + digits):
#   PLDDDD       PB1234              P B 1234
#   PLDDDDD      PB12345             P B 12345
#   PLLDDDD      PCJ1234             P CJ 1234
#   PLLDDDDD     PCJ12345            P CJ 12345
#
# Institutional (3 letters + 3-5 digits):
#   LLLDDD       MAI123              MAI 123
#   LLLDDDD      MAI1234             MAI 1234
#   LLLDDDDD     MAI12345            MAI 12345
#
# Diplomatic (2 letters + 6 digits):
#   LLDDDDDD     CD123456            CD 123 456
#
# Army (A + 3-7 digits):
#   LDDD         A123
#   LDDDD        A1234
#   LDDDDD       A12345
#   LDDDDDD      A123456
#   LDDDDDDD     A1234567
# ---------------------------------------------------------------------------

STANDARD_PATTERNS: Final[tuple[str, ...]] = (
    "LDDLLL",   # Bucharest – 2 digit suffix
    "LDDDLLL",  # Bucharest – 3 digit suffix
    "LLDDLLL",  # Two-char county + 2 digits + 3 letters
)

ALL_PATTERNS: Final[tuple[str, ...]] = (
    # Standard
    "LDDLLL",
    "LDDDLLL",
    "LLDDLLL",
    # Temporary (county + digits only)
    "LDDDD",      # B + 4 digits
    "LDDDDD",     # B + 5 digits
    "LLDDDD",     # County + 4 digits
    "LLDDDDD",    # County + 5 digits
    # Electric
    "ELDDLLL",
    "ELDDDLLL",
    "ELLDDLLL",
    # Probe
    "PLDDDD",
    "PLDDDDD",
    "PLLDDDD",
    "PLLDDDDD",
    # Institutional
    "LLLDDD",
    "LLLDDDD",
    "LLLDDDDD",
    # Diplomatic
    "LLDDDDDD",
    # Army
    "LDDD",
    "LDDDDDD",
    "LDDDDDDD",
)

#: Map each pattern to its plate type tag (used for color-based filtering).
PATTERN_TAGS: Final[dict[str, str]] = {
    # Standard
    "LDDLLL": "standard",
    "LDDDLLL": "standard",
    "LLDDLLL": "standard",
    # Temporary
    "LDDDD": "temporary",
    "LDDDDD": "temporary",
    "LLDDDD": "temporary",
    "LLDDDDD": "temporary",
    # Electric
    "ELDDLLL": "electric",
    "ELDDDLLL": "electric",
    "ELLDDLLL": "electric",
    # Probe
    "PLDDDD": "probe",
    "PLDDDDD": "probe",
    "PLLDDDD": "probe",
    "PLLDDDDD": "probe",
    # Institutional
    "LLLDDD": "standard",  # uses standard color scheme
    "LLLDDDD": "standard",
    "LLLDDDDD": "standard",
    # Diplomatic
    "LLDDDDDD": "diplomatic",
    # Army
    "LDDD": "army",
    "LDDDDDD": "army",
    "LDDDDDDD": "army",
}

# ---------------------------------------------------------------------------
# Visual confusion mappings (bidirectional, synced with plate_validator.py)
# ---------------------------------------------------------------------------

#: Map a digit to its visually similar letter(s) — ordered by similarity.
#: Multi-valued: some digits can be confused with multiple letters.
DIGIT_TO_LETTERS: Final[dict[str, list[tuple[str, float]]]] = {
    "0": [("O", 0.87), ("D", 0.75), ("Q", 0.65)],
    "1": [("I", 0.87)],
    "2": [("Z", 0.80)],
    "3": [("E", 0.80)],  # was B — 3 is visually closer to E; 8→B avoids ambiguity
    "4": [("A", 0.80)],
    "5": [("S", 0.78)],
    "6": [("G", 0.78)],
    "7": [("T", 0.75)],
    "8": [("B", 0.87)],
    "9": [("G", 0.70)],  # secondary: 9 resembles lowercase g
}

# Keep the old single-value dict for backward compatibility with plate_validator.
DIGIT_TO_LETTER: Final[dict[str, str]] = {
    k: v[0][0] for k, v in DIGIT_TO_LETTERS.items()
}

#: Map a letter to its visually similar digit(s).
LETTER_TO_DIGITS: Final[dict[str, list[tuple[str, float]]]] = {
    "O": [("0", 0.87)],
    "I": [("1", 0.87)],
    "Z": [("2", 0.80)],
    "E": [("3", 0.80)],
    "B": [("8", 0.87)],
    "S": [("5", 0.78)],
    "G": [("6", 0.78)],
    "T": [("7", 0.75)],
    "A": [("4", 0.80)],
    "Q": [("0", 0.65)],
    "D": [("0", 0.75)],  # OCR classic: D looks like 0
}

# Keep the old single-value dict for backward compatibility with plate_validator.
LETTER_TO_DIGIT: Final[dict[str, str]] = {
    k: v[0][0] for k, v in LETTER_TO_DIGITS.items()
}

# Old name kept for backward compat
CONFUSION_SCORE: Final[float] = 0.87

# ---------------------------------------------------------------------------
# FSM helper functions
# ---------------------------------------------------------------------------


def char_type(char: str) -> str:
    """Return ``'L'``, ``'D'``, or ``'?'`` for *char*."""
    if char in LETTERS:
        return "L"
    if char in DIGITS:
        return "D"
    return "?"


def slot_accepts(slot_type: str, char: str) -> bool:
    """Return ``True`` when *char* is valid for a pattern slot of *slot_type*.

    Slot types:
      ``'L'`` — letter position (A-Z)
      ``'D'`` — digit position (0-9)
      ``'E'`` — fixed literal 'E' (electric prefix)
      ``'P'`` — fixed literal 'P' (probe prefix)
    """
    if slot_type == "L":
        return char in LETTERS
    if slot_type == "D":
        return char in DIGITS
    # Fixed-literal slots for special prefixes
    if slot_type == "E":
        return char == "E"
    if slot_type == "P":
        return char == "P"
    return False


def char_candidates(char: str) -> list[tuple[str, float]]:
    """Return ``(candidate_char, similarity_score)`` pairs for *char*.

    The original character is always first with score ``1.0``.  Classic
    visual-confusion alternatives follow with their similarity scores.

    Examples
    --------
    >>> char_candidates("0")
    [("0", 1.0), ("O", 0.87), ("D", 0.75), ("Q", 0.65)]
    >>> char_candidates("I")
    [("I", 1.0), ("1", 0.87)]
    >>> char_candidates("X")
    [("X", 1.0)]
    """
    cands: list[tuple[str, float]] = [(char, 1.0)]

    if char in DIGITS:
        for alt_char, score in DIGIT_TO_LETTERS.get(char, []):
            cands.append((alt_char, score))
    elif char in LETTERS:
        for alt_char, score in LETTER_TO_DIGITS.get(char, []):
            cands.append((alt_char, score))

    return cands

