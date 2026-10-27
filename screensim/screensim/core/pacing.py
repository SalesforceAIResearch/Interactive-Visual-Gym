"""Hazard pacing — one module, imported by the apps AND by the case
generators, so a case is always certified under the same clocks the graded
episode runs (the skill's Law-3 lesson: an inline pace patch in the harness
let cases certify outcomes the live run could not reach).

Defaults are the headless-replay clocks. `apply_conversation_pace()` relaxes
them for interactive episodes where every human conversational turn costs a
tick; `reset()` restores the defaults for plain replay and tests.
"""

DEFAULTS = {"OTP_TICKS": 30, "OTP_WARN": 22, "CART_TICKS": 40, "CART_WARN": 28}

OTP_TICKS = DEFAULTS["OTP_TICKS"]
OTP_WARN = DEFAULTS["OTP_WARN"]
CART_TICKS = DEFAULTS["CART_TICKS"]
CART_WARN = DEFAULTS["CART_WARN"]

CONVERSATION = {"OTP_TICKS": 90, "OTP_WARN": 70, "CART_TICKS": 120,
                "CART_WARN": 95}


def apply_conversation_pace():
    g = globals()
    for k, v in CONVERSATION.items():
        g[k] = v


def reset():
    g = globals()
    for k, v in DEFAULTS.items():
        g[k] = v
