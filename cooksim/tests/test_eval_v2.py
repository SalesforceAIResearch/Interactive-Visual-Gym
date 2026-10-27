"""eval v2 scorer (docs/PERSONA_EVAL_V2.md): formulas, null-propagation, and the
level-equals-weight rule, pinned against hand-computed values."""
import os
import sys

from cooksim_bench.paths import ROOT  # the CookSim release root (benchmark/, web/, serving/ live here)
from cooksim_bench.scoring import eval_v2 as E
REPORT = {
    "outcome": "won", "ticks": 120, "injected_at": 30,
    "flagged_valid": True, "latency": 8, "window": [38, 70],
    "flags": [{"verdict": "valid"}], "false_positives": [{}, {}],
    "pauses": [{"who": "vlm", "ticks": 12, "costs_game_time": False},
               {"who": "human", "ticks": 6, "costs_game_time": False}],
    "eval_qualitative": {"judge_mean": 8.2},
    "exit_survey": {"satisfaction": 6, "reuse": 5, "elicitation": "live"},
}


def test_worked_example_matches_hand_computation():
    ev = E.score(REPORT, a=1, b=1, c=3, ref_ticks=60)
    assert ev["task_success"] == 1.0
    TT, CC = ev["task_efficiency"]["TT"], ev["task_efficiency"]["CC"]
    assert abs(TT - 0.5) < 1e-9                      # 60/120
    assert abs(CC - (1 - 12 / 120)) < 1e-9           # only vlm speech counts
    assert abs(ev["task_efficiency"]["normalized"] - (1 * TT + 3 * CC) / 4) < 1e-9
    DT, DQ, SQ = (ev["assistance_quality"][k] for k in ("DT", "DQ", "SQ"))
    assert abs(DT - (0.9 - 0.4 * (8 / 32))) < 1e-9
    assert abs(DQ - (1 - 0.5 * (2 / 3.6))) < 1e-9    # hit=1; fa=2, budget=3.6
    assert abs(SQ - (8.2 - 1) / 9) < 1e-9
    assert abs(ev["assistance_quality"]["raw"] - (1 * (DT + DQ) + 2 * SQ)) < 1e-9
    assert abs(ev["assistance_quality"]["normalized"]
               - (1 * (DT + DQ) + 2 * SQ) / 4) < 1e-9
    assert abs(ev["interaction_quality"]["satisfaction"] - 5 / 6) < 1e-9


def test_weights_change_the_normalized_scores():
    lo = E.score(REPORT, a=1, b=1, c=1, ref_ticks=60)
    hi = E.score(REPORT, a=3, b=1, c=1, ref_ticks=60)
    # TT < CC here, so weighting TT up must LOWER normalized efficiency
    assert hi["task_efficiency"]["normalized"] < lo["task_efficiency"]["normalized"]


def test_not_measured_propagates_as_null_never_zero():
    d = dict(REPORT, outcome=None, aborted="wall_clock", injected_at=None,
             failed_at="injection_skipped")
    ev = E.score(d, a=2, b=2, c=2, ref_ticks=60)
    assert ev["task_success"] is None
    assert ev["assistance_quality"]["DT"] is None
    assert ev["assistance_quality"]["DQ"] is None
    # SQ alone still averages (weight 2), efficiency still measured
    assert ev["assistance_quality"]["normalized"] is not None


def test_missing_exit_survey_is_null():
    d = dict(REPORT)
    d.pop("exit_survey")
    ev = E.score(d, a=1, b=1, c=1, ref_ticks=60)
    assert ev["interaction_quality"]["satisfaction"] is None
    assert ev["interaction_quality"]["reuse"] is None


# --- SQ must survive BOTH judge shapes -------------------------------------------------
# eval_judge.py writes the six sub-scores nested under judge.scores and no top-level mean,
# so a reader that knows only the flat keys returns null on a report that has a real judge
# result. SQ is the only term grading delivery, so losing it silently flatters the score --
# the sweep's classic_expert published assistance_quality 0.915 from a user rating it 2/7.
# Both shapes are pinned here so neither writer can drift away from the reader again.

# verbatim shape written by tools/eval_judge.py --write
JUDGE_NESTED = {"schema": "assist-qual-v1", "model": "gemini-3.1-pro-preview",
                "questionnaire": {"answers": {"helpfulness": 6, "overall": 6}},
                "judge": {"scores": {"grounding": 6, "actionability": 7, "persuasion": 9,
                                     "timing_manners": 6, "persona_respect": 5,
                                     "language_economy": 8},
                          "rationales": {"grounding": "..."}}}


def test_sq_reads_the_nested_judge_scores_mean():
    d = dict(REPORT, eval_qualitative=JUDGE_NESTED)
    ev = E.score(d, a=1, b=1, c=1, ref_ticks=60)
    mean = (6 + 7 + 9 + 6 + 5 + 8) / 6                   # 6.8333...
    assert abs(ev["assistance_quality"]["SQ"] - (mean - 1) / 9) < 1e-9


def test_sq_still_reads_a_flat_mean_and_prefers_it():
    # older reports carry the flat key; an explicit summary wins over the sub-scores
    d = dict(REPORT, eval_qualitative=dict(JUDGE_NESTED, judge_mean=8.2))
    ev = E.score(d, a=1, b=1, c=1, ref_ticks=60)
    assert abs(ev["assistance_quality"]["SQ"] - (8.2 - 1) / 9) < 1e-9


def test_sq_is_null_without_any_judge_block():
    d = dict(REPORT)
    d.pop("eval_qualitative")
    ev = E.score(d, a=1, b=1, c=1, ref_ticks=60)
    assert ev["assistance_quality"]["SQ"] is None


def test_nested_judge_actually_reaches_the_headline():
    """The regression that shipped: SQ null quietly renormalises over 2b instead of 2b+2."""
    with_judge = E.score(dict(REPORT, eval_qualitative=JUDGE_NESTED),
                         a=1, b=1, c=1, ref_ticks=60)
    d = dict(REPORT); d.pop("eval_qualitative")
    without = E.score(d, a=1, b=1, c=1, ref_ticks=60)
    DT, DQ = with_judge["assistance_quality"]["DT"], with_judge["assistance_quality"]["DQ"]
    SQ = with_judge["assistance_quality"]["SQ"]
    assert abs(with_judge["assistance_quality"]["raw"] - (1 * (DT + DQ) + 2 * SQ)) < 1e-9
    assert abs(with_judge["assistance_quality"]["normalized"]
               - (1 * (DT + DQ) + 2 * SQ) / 4) < 1e-9     # /(2b+2), all three terms live
    assert abs(without["assistance_quality"]["normalized"] - (DT + DQ) / 2) < 1e-9


def test_early_correct_gets_partial_timing_credit():
    d = dict(REPORT, flagged_valid=False, prevented=None,
             flags=[{"verdict": "early_correct"}])
    ev = E.score(d, a=1, b=1, c=1, ref_ticks=60)
    assert ev["assistance_quality"]["DT"] == 0.5
