"""Design rule (user, 2026-09-15): forget_next_step never forgets the serve.

A cook next to a finished plate that it 'forgot' to hand in is not a plausible slip. The rule is
enforced in four places — the purity gate, the inventory generator, err_forget_next_step itself
and the live harness — and this file pins all of them plus the shipped suite."""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("COOKSIM_GRID", os.path.join(ROOT, os.path.join(ROOT, "benchmark", "grid.json")))


def _final_step(cell):
    from cooksim_bench.exp_interact import recipe_of
    from cooksim.core.plan_ir import segment_plan
    sids = [n.sid for n in segment_plan(recipe_of(cell))]
    return sids[sids.index("serve") - 1]


def test_engine_refuses_to_forget_the_serve():
    from cooksim.core.error_lib import err_forget_next_step
    from cooksim.core.plan_ir import segment_plan
    from cooksim_bench.exp_interact import recipe_of
    cell = "hard_nops_medium_map"
    plan = list(segment_plan(recipe_of(cell)))
    with pytest.raises(KeyError):
        err_forget_next_step(plan, after=_final_step(cell))
    err_forget_next_step(list(segment_plan(recipe_of(cell))), after="prep(potato)")   # a real step is fine


@pytest.mark.parametrize("path", ["tmp/interact_eval/cases_composite_v4.json", "benchmark/cases_composite_v4.json"])
def test_no_suite_case_forgets_the_serve(path):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        pytest.skip(f"{path} not present")
    bad = []
    for c in json.load(open(full))["cases"]:
        for s in c["subs"]:
            if s.get("err") == "forget_next_step" and (s.get("kwargs") or {}).get("after") == _final_step(s["cell"]):
                bad.append(f"{s['name']}: after={s['kwargs']['after']}")
    assert not bad, "forget_next_step anchored on the step before serve:\n" + "\n".join(bad)
