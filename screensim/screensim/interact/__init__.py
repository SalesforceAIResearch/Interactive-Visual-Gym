"""The interaction stack: plan IR, error injection, personas, the human and
assistant agents, the episode loop, and grading — the port of the
interactive-sim-benchmark skill (§5-§7) onto the ScreenSim world engine.

Interface versions (skill §7.2 "version the interface before changing what an
agent sees or says"): read at call time, stamped in every report, gated on at
publication. The retired version must still run.

  sv1  frames only                    — perception under test, the honest default
  sv2  frames + relational text map   — a perception ablation (does the text help?)
"""

INTERFACE_VERSION = "sv1"

# The one model used for BOTH the human agent and the AI assistant. Do not
# use anything else in the interaction stack.
MODEL = "gemini-3.7-flash"

INTERFACES = {
    "sv1": {"frames": True, "relational": False,
            "desc": "frames only"},
    "sv2": {"frames": True, "relational": True,
            "desc": "frames + relational text map"},
}


def interface(version):
    if version not in INTERFACES:
        raise ValueError(f"unknown interface version {version!r}; "
                         f"known: {list(INTERFACES)}")
    return INTERFACES[version]
