"""ONE definition of "correct detection", imported by the episode loop (for
takeover release), the grader, and every report/page builder — never
re-listed anywhere (skill §5.3, §7.2). A test greps consumers for hand-listed
verdict tuples.
"""
from __future__ import annotations

# verdict labels a detection flag can receive
CREDIT = "correct"          # named the injected error, or its in-window twin
EARLY = "early_correct"     # correct before the deviation was strictly observable
NEUTRAL = "neutral"         # a true statement about the world, not the injected error
FALSE_ALARM = "false_alarm"  # neither the error nor true of the world
MISS = "miss"               # error present, never flagged

# the set that ARMS takeover release and scores as a hit — one tuple, imported
CORRECT_DETECTION = frozenset({CREDIT, EARLY})
# verdicts that neither credit nor charge
NEUTRAL_VERDICTS = frozenset({NEUTRAL})


def is_correct(verdict: str) -> bool:
    return verdict in CORRECT_DETECTION
