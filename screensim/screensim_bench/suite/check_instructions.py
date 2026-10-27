"""LABEL-LEVEL check (Task.instruction_label). Goal-level instructions are verified by
screensim_bench/suite/verify_instructions.py instead. Every goal requirement must be nameable from the task's instruction (the lesson of
T1–T3: an LLM person defends a step the instruction never asked for).

For each hard task: each commit step's control label and each 'keep' constraint must
share a content word with the instruction (or an obvious synonym listed here).

  python screensim_bench/suite/check_instructions.py
"""
from __future__ import annotations
import os, re, sys
from screensim.tasks import HARD_TASKS, by_id
from screensim.interact.plan_ir import compile_plan

STOP = {"the", "and", "for", "to", "on", "off", "a", "of", "with", "in", "my", "it", "is", "&", "tap", "turn", "set", "keep", "app", "apps"}
SYN = {"bold": {"bolder", "bold"}, "larger": {"bigger", "larger", "text"}, "contrast": {"contrast"}, "shapes": {"buttons", "button"},
       "motion": {"motion"}, "captions": {"captions", "caption"}, "mono": {"mono"}, "led": {"flash", "led"},
       "never": {"never", "location"}, "tracking": {"tracking", "track"}, "refresh": {"refreshing", "refresh", "background"},
       "analytics": {"analytics"}, "siri": {"siri"}, "dictation": {"siri", "recordings"}, "widgets": {"widgets"},
       "wallet": {"wallet"}, "calls": {"calls"}, "center": {"center"}, "notification": {"notification"}, "reply": {"reply"},
       "protection": {"protection"}, "advanced": {"advanced"}, "stolen": {"stolen"}, "web": {"web"},
       "always": {"always"}, "notifications": {"notifications"}, "wake": {"waking", "move", "wake"}, "night": {"night"},
       "standby": {"standby"}, "correction": {"auto-correction", "correction"}, "punctuation": {"punctuation"},
       "predictive": {"predictive"}, "spelling": {"spell"}, "capitalization": {"auto-capitalization", "capitalization"},
       "instant": {"instantly", "instant"}, "balance": {"balance"}, "days": {"1-3", "day"}, "review": {"send", "pay"},
       "confirm": {"send", "pay"}, "verify": {"send", "pay"}, "code": {"send", "pay"}}

def words(s):
    return {w for w in re.findall(r"[a-z0-9\-']+", s.lower()) if w not in STOP}

def main():
    bad = 0
    for tid in HARD_TASKS:
        t = by_id(tid); ir = compile_plan(t); iw = words(t.instruction_label or t.instruction)
        for n in ir.nodes:
            if n.kind != "commit":
                continue
            label = n.gestures[-1] if n.gestures else n.label
            lw = words(label.replace("tap ", "").replace("type ", ""))
            confirm = label.replace("tap ", "").strip().lower() in {"turn on", "turn off", "done", "confirm", "ok", "continue"}
            ok = confirm or any(w in iw or (SYN.get(w, set()) & iw) for w in lw) or label.startswith("type")
            if not ok:
                bad += 1
                print(f"  !! {tid}: step {label!r} not nameable from the instruction")
        print(f"{tid}: {sum(1 for n in ir.nodes if n.kind == 'commit')} commit steps checked")
    print("unnameable steps:", bad)
    return bad

if __name__ == "__main__":
    sys.exit(1 if main() else 0)
