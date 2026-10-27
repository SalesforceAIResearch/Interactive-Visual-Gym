"""Does a goal-level instruction map to exactly one set of setting changes?

For each task, a strong model is given EVERY setting that exists on the simulated phone (from
screensim/apps/inventory.py; the QuickPay form for payments) and the owner's request, and asked
which settings must change, which must stay, and which other settings could also plausibly do
part of the job. An instruction passes only if every model returns exactly the label-level
ground truth below (nothing more, nothing less) and names no alternative the wording fails to
rule out.

  python screensim_bench/suite/verify_instructions.py tmp/goal_instructions.json --out tmp/verify_round1.json
  python screensim_bench/suite/verify_instructions.py tmp/goal_instructions.json --tasks standby_bedside
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor


from screensim.interact.agents import _extract_json, make_client

MODELS = ["gpt-5.5", "gemini-3.1-pro-preview"]

# ------------------------------------------------------------------ label-level ground truth
# each item: (page must contain, setting label, target). A tuple of items is "any one of".
# Derived from the goal predicates in screensim/tasks.py.
TRUTH = {
    "lost_phone_lockdown": {
        "changes": [("Advanced Data Protection", "advanced data protection", "on"),
                    ("Stolen Device Protection", "stolen device protection", "on")],
        "keep": [("iCloud", "access icloud data on the web", "on"),
                 ("Stolen Device Protection", "away from familiar locations", "selected")]},
    "app_privacy_quarantine": {
        "changes": [("Location Services", "never", "selected"),
                    (("Background App Refresh", "tripsnap", "off"), ("Apps > TripSnap", "background app refresh", "off")),
                    ("Tracking", "tripsnap", "off")],
        "keep": [("Tracking", "allow apps to request to track", "on")]},
    "parent_phone_readability": {
        "changes": [("", "bold text", "on"),
                    ("", "text size", "increase"),
                    ("Display & Text Size", "increase contrast", "on"),
                    ("Display & Text Size", "button shapes", "on"),
                    ("Motion", "reduce motion", "on")],
        "keep": [],
        # "don't magnify": naming these as kept is right, leaving them out is fine too
        "keep_optional": [("Display Zoom", "display zoom", "default"), ("Zoom", "zoom", "off")]},
    "lock_screen_lockdown": {
        "changes": [("Allow Access When Locked", "control center", "off"),
                    ("Allow Access When Locked", "today view and search", "off"),
                    ("Allow Access When Locked", "wallet", "off"),
                    ("Allow Access When Locked", "return missed calls", "off")],
        "keep": []},
    "hearing_setup": {
        "changes": [("Audio & Visual", "led flash for alerts", "on"),
                    ("Audio & Visual", "mono audio", "on"),
                    ("Subtitles & Captioning", "closed captions + sdh", "on")],
        "keep": [("VoiceOver", "voiceover", "off")]},
    "stop_profiling": {
        "changes": [("Tracking", "allow apps to request to track", "off"),
                    ("Analytics & Improvements", "share iphone analytics", "off"),
                    ("Analytics & Improvements", "improve siri & dictation", "off")],
        "keep": [("Analytics & Improvements", "share icloud analytics", "on")]},
    "siri_lock_screen_privacy": {
        "changes": [("Siri", "allow siri when locked", "off"),
                    ("Allow Access When Locked", "notification center", "off"),
                    ("Allow Access When Locked", "reply with message", "off")],
        "keep": [("Allow Access When Locked", "control center", "on")]},
    "keyboard_stop_correcting": {
        "changes": [("Keyboard", "auto-correction", "off"),
                    ("Keyboard", "smart punctuation", "off"),
                    ("Keyboard", "predictive", "off")],
        "keep": [("Keyboard", "check spelling", "on"), ("Keyboard", "auto-capitalization", "on")]},
    "standby_bedside": {
        "changes": [("StandBy", "always on", "off"),
                    ("StandBy", "motion to wake", "off"),
                    ("StandBy", "show notifications", "off")],
        "keep": [("StandBy", "standby", "on"), ("StandBy", "night mode", "on"),
                 ("StandBy > Display", "turn display off", "automatically")]},
    "rent_share_standard": {
        "changes": [("QuickPay", "payee", "Alex Chen"), ("QuickPay", "amount", "40"),
                    ("QuickPay", "pay from", "QuickPay balance"), ("QuickPay", "speed", "1-3 business days"),
                    ("QuickPay", "note", "March rent share")],
        "keep": []},
    "rent_landlord_instant": {
        "changes": [("QuickPay", "payee", "Landlord — Sunset Apts"), ("QuickPay", "amount", "1450"),
                    ("QuickPay", "pay from", "QuickPay balance"), ("QuickPay", "speed", "Instant"),
                    ("QuickPay", "note", "April rent")],
        "keep": []},
}
PAYMENT_TASKS = {"rent_share_standard", "rent_landlord_instant"}

QUICKPAY = """[QuickPay > Send money]
  choose one payee:  Alex Chen (@alexc) · Alex Kim (@akim) · Jordan Diaz (@jdiaz) · Landlord — Sunset Apts (@sunsetapts)
[QuickPay > Send money > Amount]
  field   amount (dollars)
  field   note (optional, free text)
  pay from, choose one:  QuickPay balance · $5,284.55  |  Visa ···· 4821 · 3% fee
  speed, choose one:     Instant · arrives in seconds  |  1-3 business days · no fee
  (defaults: QuickPay balance, Instant)"""

PROMPT_SETTINGS = """You are auditing a benchmark task for a simulated iPhone. The phone's owner makes a
request; your job is to decide whether the request has exactly one correct reading.

Below is EVERY setting that exists on this phone, grouped by the page path that reaches it,
with its current value. Nothing outside this list exists.

{inventory}

The phone's assistant will also read this help text, written for a real iPhone. It may describe
controls or options that are not on this phone; only the list above exists.

{manual}

The owner says:
"{instruction}"

1. List exactly the settings that must be DIFFERENT afterwards to do what the owner asked — no
   more, no less. Use only settings from the list, named exactly as listed, with their page.
   For a slider use "increase" or "decrease"; for an option list use "selected".
2. List the settings the owner explicitly wants to STAY as they are.
3. List every OTHER setting from the list that someone might think does part of what was asked,
   would also be needed, or could be used instead. For each, decide: would changing it — instead
   of, or in addition to, your list — ALSO be a valid way to carry out the owner's request as
   worded? Answer true only if the owner could not fairly object to it. A setting that is already
   in the required state, does something different, or does only part of the job is false.
   A control marked "the same setting also appears under" is the same setting, and a page row
   that "opens the page listed as [...]" leads to that same page: neither is an alternative.
   A button that switches a feature on (e.g. "Turn On ...") is a change: put it in list 1, not here.
4. Under "manual_only", list every SETTING the help text describes that is NOT in the list above
   and that someone might think does part of what the owner asked. Skip physical actions, steps
   such as entering a passcode, and controls that exist in the list under another name or path.
   For each, decide as in 3 whether it would ALSO be a valid way to carry out the request.

Reply with JSON only:
{{"changes": [{{"page": "...", "setting": "...", "to": "on|off|selected|increase|decrease"}}],
  "keep": [{{"page": "...", "setting": "...", "value": "..."}}],
  "alternatives": [{{"page": "...", "setting": "...", "also_valid": false, "why": "..."}}],
  "manual_only": [{{"control": "...", "also_valid": false, "why": "..."}}]}}"""

PROMPT_PAYMENT = """You are auditing a benchmark task for a simulated phone payment app. The owner makes a
request; your job is to decide whether the request has exactly one correct reading.

This is EVERYTHING the QuickPay send-money form offers:

{form}

The owner says:
"{instruction}"

1. Under "changes", give ALL FIVE of payee, amount, pay from, speed and note, with the exact value
   the person must end up with — including a field whose correct value is already the default.
   If the request does not determine a field exactly, write "UNDETERMINED".
2. List every other choice in the form someone might pick instead for any field, and decide
   whether picking it would ALSO be a valid way to carry out the request as worded.

Reply with JSON only:
{{"changes": [{{"page": "QuickPay", "setting": "payee|amount|pay from|speed|note", "to": "..."}}],
  "keep": [],
  "alternatives": [{{"page": "QuickPay", "setting": "...", "also_valid": false, "why": "..."}}]}}"""


def norm(s):
    s = (s or "").lower().replace("’", "'").replace("“", "").replace("”", "").replace('"', "")
    return re.sub(r"\s+", " ", s).strip()


def to_norm(v):
    v = norm(str(v))
    if v in ("true", "enabled", "turn on", "on (enabled)"):
        return "on"
    if v in ("false", "disabled", "turn off"):
        return "off"
    if v.startswith("increase") or v in ("larger", "bigger", "raise"):
        return "increase"
    return v


def _match_one(item, got):
    page, label, target = item
    if label == "advanced data protection":           # reached from iCloud or Sign-In & Security
        return ("advanced data protection" in norm(got.get("setting"))
                or "advanced data protection" in norm(got.get("page")))
    if norm(page) and norm(page) not in norm(got.get("page")):
        return False
    gl, gt = norm(got.get("setting")), to_norm(got.get("to", got.get("value")))
    if label in ("payee", "amount", "pay from", "speed", "note"):
        if gl != label:
            return False
        want = norm(target)
        have = norm(str(got.get("to", ""))).replace("$", "").replace(",", "").replace(".00", "")
        return want in have or have in want if have else False
    if label == "never":               # Location Services > TripSnap > Never
        return ("tripsnap" in norm(got.get("page")) and
                (gl == "never" or (gl in ("tripsnap", "allow location access") and "never" in gt)))
    if label == "advanced data protection":           # a page with a "Turn On" button, not a switch
        return "advanced data protection" in gl or "advanced data protection" in norm(got.get("page"))
    if target == "automatically" and label == "turn display off":   # an option list kept as it is
        return ("turn display off" in gl + " " + norm(got.get("page"))) and "automatically" in (gt + " " + gl)
    if gl != label:
        return False
    return gt == target or (target == "selected" and gt in ("selected", "on"))


_CANON = None


def _canon_index():
    global _CANON
    if _CANON is None:
        import screensim.apps.inventory as inv
        inv.CANON.clear()
        inv.walk()
        _CANON = {}
        for key, places in inv.CANON.items():
            for page, label in places:
                _CANON[(norm(page), norm(label))] = key
    return _CANON


def same_setting(extra_str, matched):
    page, _, rest = extra_str.rpartition(" > ")
    label = rest.split(" -> ")[0]
    idx = _canon_index()
    a = idx.get((norm(page), norm(label)))
    b = idx.get((norm(matched.get("page")), norm(matched.get("setting"))))
    return a is not None and a == b


def grade(task, out):
    truth = TRUTH[task]
    changes = list(out.get("changes") or [])
    keeps = list(out.get("keep") or [])
    missing, used = [], set()
    for item in truth["changes"]:
        options = item if isinstance(item[0], tuple) else (item,)
        hit = next((i for i, g in enumerate(changes) if i not in used and any(_match_one(o, g) for o in options)), None)
        if hit is None:
            missing.append(options[0][1])
        else:
            used.add(hit)
    extra = [f"{c.get('page')} > {c.get('setting')} -> {c.get('to')}" for i, c in enumerate(changes) if i not in used]
    kmissing = [k[1] for k in truth["keep"] if not any(_match_one(k, g) for g in keeps)]
    kextra = [f"{k.get('page')} > {k.get('setting')} = {k.get('value')}" for k in keeps
              if not any(_match_one(t, k) for t in truth["keep"] + truth.get("keep_optional", []))]
    def _is_required(alt):                  # an "alternative" that IS one of the required changes
        probe = {"page": alt.get("page"), "setting": alt.get("setting"), "to": "on"}
        for item in truth["changes"]:
            options = item if isinstance(item[0], tuple) else (item,)
            for page, label, target in options:
                if _match_one((page, label, target), dict(probe, to=target)):
                    return True
                # the row on a parent page that opens the required control's own page
                opened = norm(f"{alt.get('page')} > {alt.get('setting')}").replace("[", "").replace("]", "")
                if norm(alt.get("setting")) == label and norm(page) and norm(page) in opened:
                    return True
        return False
    open_alts = [f"{a.get('page')} > {a.get('setting')}: {a.get('why')}" for a in (out.get("alternatives") or [])
                 if a.get("also_valid") is True and not _is_required(a)]
    # the same setting listed from two pages is one change, not an extra
    matched = [changes[i] for i in used]
    extra = [e for e in extra if not any(same_setting(e, m) for m in matched)]
    manual_only = [f"{m.get('control')}: {m.get('why')}" for m in (out.get("manual_only") or [])
                   if m.get("also_valid") is True]
    ok = not (missing or extra or kmissing or open_alts or manual_only)
    return {"pass": ok, "missing": missing, "extra": extra, "keep_missing": kmissing,
            "keep_extra": kextra, "open_alternatives": open_alts,
            "manual_only": manual_only}


def ask(model, task, instruction, inventory):
    if task in PAYMENT_TASKS:
        prompt = PROMPT_PAYMENT.format(form=QUICKPAY, instruction=instruction)
    else:
        from screensim.interact.manual import manual_context
        prompt = PROMPT_SETTINGS.format(inventory=inventory, instruction=instruction,
                                        manual=manual_context(task))
    client = make_client(model)
    raw = client.text(prompt, temperature=0.2, max_tokens=16000)
    return raw, _extract_json(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instructions", help="json {task_id: goal-level instruction}")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", default="tmp/verify_instructions.json")
    ap.add_argument("--inventory", default="tmp/settings_inventory.txt")
    a = ap.parse_args()
    instr = json.load(open(a.instructions))
    tasks = [t for t in (a.tasks.split(",") if a.tasks else instr) if t in instr]
    inventory = open(a.inventory).read()
    jobs = [(m, t) for t in tasks for m in MODELS]
    results = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(ask, m, t, instr[t], inventory): (m, t) for m, t in jobs}
        for f in futs:
            m, t = futs[f]
            try:
                raw, out = f.result()
                results[(t, m)] = {"raw": raw, "out": out, "grade": grade(t, out)}
            except Exception as e:
                results[(t, m)] = {"raw": "", "out": {}, "grade": {"pass": False, "error": str(e)}}
    report = {}
    for t in tasks:
        report[t] = {"instruction": instr[t], "models": {}}
        for m in MODELS:
            r = results[(t, m)]
            report[t]["models"][m] = {"grade": r["grade"], "out": r["out"]}
        verdict = all(report[t]["models"][m]["grade"].get("pass") for m in MODELS)
        report[t]["pass"] = verdict
        print(f"{'PASS' if verdict else 'FAIL'}  {t}")
        for m in MODELS:
            g = report[t]["models"][m]["grade"]
            if not g.get("pass"):
                print(f"    {m}: " + "; ".join(f"{k}={v}" for k, v in g.items() if k != "pass" and v))
    old = json.load(open(a.out)) if os.path.exists(a.out) else {}
    old.update(report)
    json.dump(old, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
