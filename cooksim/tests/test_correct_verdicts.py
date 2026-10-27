import cooksim_bench.e2e_stepin_rollout
import cooksim_bench.scoring.eval_v2
import cooksim_bench.scoring.eval_assistant
"""User ruling 2026-08-18: what counts as a CORRECT detection is ONE definition, consumed by
the episode (release-gate arming) and by every later evaluation alike.

The defect this pins: the scorer forgave an in-window twin name (`alias_in_window`) as
neutral while the release gate armed only on `valid`. An assistant that read the deviation
right under a twin name left the takeover unreleased; the cook's accepted repairs were
silently dropped and its hands served the wrong plate. Eight call sites had hand-listed
("valid", "early_correct").
"""
import re, os, sys
from cooksim_bench.paths import ROOT  # the CookSim release root (benchmark/, web/, serving/ live here)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from cooksim.interact.checks import CORRECT_VERDICTS, NEUTRAL_VERDICTS


def test_correct_verdicts_definition():
    assert CORRECT_VERDICTS == ("valid", "early_correct", "alias_in_window")
    # every correct verdict is also neutral (never charged); early_alias is neutral but NOT correct
    assert set(CORRECT_VERDICTS) <= set(NEUTRAL_VERDICTS)
    assert "early_alias" not in CORRECT_VERDICTS and "true_observation" not in CORRECT_VERDICTS


def test_no_consumer_hand_lists_the_correct_set():
    """No file may re-list ("valid", "early_correct") as the correct-flag test; import it."""
    bad = []
    for rel in (cooksim_bench.e2e_stepin_rollout.__file__, cooksim_bench.scoring.eval_v2.__file__, cooksim_bench.scoring.eval_assistant.__file__):
        src = open(rel).read()
        for m in re.finditer(r'in \(("valid"|"early_correct")(, ("valid"|"early_correct"))?\)', src):
            bad.append((rel, src[:m.start()].count("\n") + 1, m.group(0)))
        # arming must consult the shared set
        if rel.endswith("e2e_stepin_rollout.py"):
            assert "in CORRECT_VERDICTS" in src
            assert 'nv == "valid" and not armed' not in src
    assert not bad, bad


def test_max_hold_release_exists():
    """After the injected error COMMITS in the world, the takeover releases regardless of
    whether the assistant named it (user ruling 2026-08-18)."""
    src = open(cooksim_bench.e2e_stepin_rollout.__file__).read()
    assert 'report["max_hold_release"]' in src
    assert "deviated_block_executed" in src and "plate_committed" in src
    # the omit branch of plate_verdict is true from the first scoop; it must not be a release trigger.
    # B1 (fix-firing 2026-09-19): nor is wrong_order — it commits on its first scoop and releasing there dropped the
    # second one (the swapped pair became an omission); it releases when its whole block has run.
    assert '"add_foreign", "misidentify_assemble", "serve_raw"):' in src
    assert '"add_foreign", "misidentify_assemble", "serve_raw", "wrong_order"' not in src
