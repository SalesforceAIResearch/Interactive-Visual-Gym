"""What the engine designed for a task: the episodes, and the mistake forced into each. No API keys."""
from screensim.interact.composite import design_composite_episodes
from screensim.tasks import HARD_TASKS, by_id

for tid in HARD_TASKS[:3]:
    task = by_id(tid)
    ir, episodes = design_composite_episodes(task, per_task=3)
    print(f"\n{task.id}  ({len(task.plan)} gestures)")
    print(f"  {task.instruction[:120]}")
    for e in episodes:
        for b in e.beats:
            print(f"    {e.id}: {b.kind:13} instead of {b.correct_gesture!r} -> {b.alt_gesture!r}")
