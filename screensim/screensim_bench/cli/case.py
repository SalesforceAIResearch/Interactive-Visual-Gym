"""screensim-case: list the benchmark tasks and the episodes designed from them.

    screensim-case                       # the 11 benchmark tasks
    screensim-case --episodes            # the 30 designed episodes, with their injected mistakes
    screensim-case --task hearing_setup  # one task in detail
"""
import argparse

from screensim.tasks import HARD_TASKS, by_id


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", action="store_true", help="show the designed episodes and their mistakes")
    ap.add_argument("--task", default="", help="one task id")
    ap.add_argument("--rl", action="store_true", help="the hand-written RL training library instead")
    a = ap.parse_args(argv)

    if a.rl:
        from screensim.task_splits import RL_TRAIN, RL_TRAIN_BY_BAND
        for band in sorted(RL_TRAIN_BY_BAND):
            print(f"band {band}: {len(RL_TRAIN_BY_BAND[band])} tasks")
            for t in RL_TRAIN_BY_BAND[band]:
                print(f"    {t}")
        print(f"\n{len(RL_TRAIN)} training tasks (held apart from the benchmark; see docs/TASKS.md)")
        return 0

    ids = [a.task] if a.task else list(HARD_TASKS)
    for tid in ids:
        t = by_id(tid)
        print(f"\n{t.id}   [{t.family}]   {len(t.plan)} gestures")
        print(f"  {t.instruction[:160]}")
        if t.trap:
            print(f"  trap: {t.trap[:140]}")
        if a.episodes or a.task:
            from screensim.interact.composite import design_composite_episodes
            _ir, eps = design_composite_episodes(t, per_task=3)
            for e in eps:
                kinds = ", ".join(f"{b.kind}@{b.node_idx}" for b in e.beats)
                print(f"    {e.id}: {kinds}")
    if not a.task:
        print(f"\n{len(HARD_TASKS)} benchmark tasks -> 30 designed episodes x 3 personas = 90 cells per seat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
