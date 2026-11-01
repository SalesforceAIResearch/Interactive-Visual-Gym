"""A stored judge entry with a `score` key but no Boolean item verdict is not a judgement (resume must re-call it)."""
from cooksim_bench.scoring.sq_v4_score import valid_judgement


def test_score_key_alone_is_not_a_judgement():
    assert not valid_judgement({"score": None, "items": {"T1": {"pass": None, "reason": ""}}})
    assert not valid_judgement({"score": 0.5, "items": {}})
    assert not valid_judgement({"error": "judge returned nothing"})


def test_boolean_verdict_or_documented_na_is_valid():
    assert valid_judgement({"score": 1.0, "items": {"T1": {"pass": True, "reason": "ok"}, "L2": {"pass": None, "reason": "n/a"}}})
    assert valid_judgement({"score": None, "items": {"T1": {"pass": None, "reason": "no claim"}}, "na_round": True})
