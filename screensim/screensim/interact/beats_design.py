"""The designed mistakes of an episode record, as films and reports show them (kind, mode, the slip's
gesture, the intended gesture, the repair). Pilot records carry them as `beats_design`; sweep records may
not, so they are rebuilt from the episode design, which is deterministic."""


_CACHE = {}


def beats_design(record):
    if record.get("beats_design"):
        return record["beats_design"]
    from ..tasks import by_id
    from .composite import design_composite_episodes
    if record["task"] not in _CACHE:
        _CACHE[record["task"]] = design_composite_episodes(by_id(record["task"]), per_task=3)[1]
    eps = _CACHE[record["task"]]
    ep = next((e for e in eps if e.id == record["episode"]), None)
    if ep is None:
        return []
    return [{"kind": b.kind, "mode": b.mode, "alt": b.alt_gesture, "correct": b.correct_gesture,
             "repair": list(b.repair)} for b in ep.beats]
