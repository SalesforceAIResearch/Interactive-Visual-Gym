"""Interaction stage — user/supervisor status JSONs, system rules, oracles, protocol.

Canonical design: docs/INTERACTION_DESIGN.md. The system (this package) is the ONLY writer of
state; the human agent emits utterances + parse dicts, the supervisor emits reply envelopes.
"""
from .state import Profile, PERSONAS, UserState, VLMReply           # noqa: F401
from . import system, checks, protocol                              # noqa: F401
