"""The gesture action space and its grounding — the core of ScreenSim.

Per the project thesis: the action space *is* the conversation space between
the assistant and the user, so it must mirror how people actually talk about a
phone. A gesture is therefore split into two independent parts, and the hard
part is always the second:

    VERB      what the finger does:  tap · long_press · swipe · type · press_key
    GROUNDING which pixels / element / direction it applies to

Two grounding modes coexist, exactly as in real guidance:

  * SEMANTIC — name a target ("tap Continue", "the second one", "the toggle
    next to Bluetooth", "the back arrow"). The engine resolves the element and
    returns its bounds. Ambiguity is a first-class outcome, not a crash.
  * SPATIAL — a raw point / direction ("tap near the top-right", "swipe up from
    the bottom", "swipe left on the Alex row"). Direction and anchor carry the
    meaning; "swipe" with no direction is refused.

Every gesture ultimately reduces to a low-level event on a pixel or an element,
and every failure returns a spoken `reason` (the analog of CookSim's blocker
strings) so the human's feedback loop and the assistant's grounding both work.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .model import Rect, Widget

# ---- verbs ---------------------------------------------------------------
TAP = "tap"
LONG_PRESS = "long_press"
SWIPE = "swipe"
TYPE = "type"
CLEAR = "clear"
PRESS_KEY = "press_key"      # a hardware/nav key: back, home, recents
WAIT = "wait"

VERBS = {TAP, LONG_PRESS, SWIPE, TYPE, CLEAR, PRESS_KEY, WAIT}

DIRECTIONS = {"up", "down", "left", "right"}
NAV_KEYS = {"back", "home", "recents"}

# words that never disambiguate a target. Prepositions (to/in/on/at/of) are
# deliberately NOT here: they collide with real control labels ("To", "On"),
# and by the time target text is cleaned the verb/relation phrases are already
# stripped, so a leftover preposition is almost always part of a label.
_STOP = {"the", "a", "an", "button", "icon", "option", "item", "one",
         "please", "then", "and", "labelled", "labeled", "that", "says",
         "this", "very", "just"}

# politeness/filler prefixes stripped before verb detection, so
# "go ahead and tap X" parses like "tap X"
_FILLER = re.compile(r"^(?:please\s+|just\s+|now\s+|then\s+|ok(?:ay)?[,\s]+"
                     r"|go ahead and\s+|go ahead[,\s]+|can you\s+"
                     r"|could you\s+|would you\s+)+", re.I)

_ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
             "last": -1, "1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4,
             "top": 0, "bottom": -1}


@dataclass
class Gesture:
    verb: str
    # semantic grounding
    target_text: str = ""            # noun phrase naming the element
    ordinal: Optional[int] = None    # 0-based; -1 = last
    region: str = ""                 # top/bottom/left/right/middle
    relation: tuple = ()             # ("next_to"|"below"|"above"|"in", text)
    # spatial grounding
    point: Optional[tuple] = None    # (x, y) absolute
    direction: str = ""              # for swipe
    anchor_text: str = ""            # swipe "on the <row>"
    # payload
    text: str = ""                   # for type
    key: str = ""                    # for press_key
    raw: str = ""

    def describe(self) -> str:
        if self.verb == TYPE:
            return f'type "{self.text}"'
        if self.verb == PRESS_KEY:
            return f"press {self.key}"
        if self.verb == SWIPE:
            a = f" on {self.anchor_text}" if self.anchor_text else ""
            return f"swipe {self.direction}{a}"
        tgt = self.target_text or (f"({self.point[0]},{self.point[1]})"
                                   if self.point else "?")
        return f"{self.verb} {tgt}"


@dataclass
class Ground:
    """Result of grounding a gesture against the live screen."""
    ok: bool
    widget: Optional[Widget] = None
    point: Optional[tuple] = None
    reason: str = ""
    n_candidates: int = 0
    ambiguous: bool = False
    candidates: list = field(default_factory=list)


# ---- parsing -------------------------------------------------------------
def parse(text: str) -> Optional[Gesture]:
    """Parse a natural gesture command. Returns None only if no verb is found.

    The parser is deliberately permissive about phrasing but strict about the
    swipe-direction requirement (a swipe with no direction is not a gesture).
    """
    t = " ".join(text.strip().split())
    low = _FILLER.sub("", t.lower()).strip()
    if not low:
        return None

    # coordinate forms: "tap (30, 40)" or "tap at 30 40"
    coord = re.search(r"\(?\s*(\d{1,3})\s*[, ]\s*(\d{1,3})\s*\)?", low)

    if low in ("wait", "do nothing", "hold on"):
        return Gesture(WAIT, raw=t)

    for k in NAV_KEYS:
        # NOTE: "tap {k}" is deliberately NOT here — "tap Home" must ground to a
        # widget labelled Home (e.g. the iCloud row), not press the home key.
        # Hardware keys are named with press/go/button or bare.
        if low in (k, f"go {k}", f"press {k}", f"{k} button",
                   "back arrow" if k == "back" else k):
            return Gesture(PRESS_KEY, key=k, raw=t)
    if low.startswith("press ") and low.split()[-1] in NAV_KEYS:
        return Gesture(PRESS_KEY, key=low.split()[-1], raw=t)

    if low.startswith(("type ", "enter ", "input ")):
        m = re.search(r'["“](.+?)["”]', t)
        payload = m.group(1) if m else t.split(" ", 1)[1]
        return Gesture(TYPE, text=payload, raw=t)

    if low.startswith(("clear", "delete the text", "empty the field")):
        return Gesture(CLEAR, raw=t)

    # swipe only when it's the leading verb (or after a filler word) — a
    # label like "Swipe Options" must not hijack "tap Swipe Options"
    toks0 = low.split()
    if toks0[0] in ("drag", "slide", "move") and len(toks0) > 1:
        dm = re.search(r"\b(right|left|up|down)\b", low)
        if dm:
            lab = re.sub(r"^(drag|slide|move)\s+(the\s+)?", "", low)
            lab = re.sub(r"\s*(slider\s+)?(all the way\s+|a bit\s+|over\s+)?(to the\s+)?\b(right|left|up|down)\b.*$", "", lab)
            lab = re.sub(r"\s+slider$", "", lab).strip()
            g = Gesture(SWIPE, raw=t, direction=dm.group(1))
            if lab:
                g.anchor_text = _clean(lab)
            return g
    is_swipe = toks0[0] in ("swipe", "scroll") or (
        len(toks0) > 1 and toks0[0] in ("please", "now", "then", "just")
        and toks0[1] in ("swipe", "scroll"))
    if is_swipe:
        g = Gesture(SWIPE, raw=t)
        # edge/corner origin FIRST — "from the top right" contains a direction
        # word ("right") that must not be mistaken for the swipe direction.
        rest = low
        if re.search(r"from the top[- ]?right", rest):
            g.region = "top-right"
            rest = re.sub(r"from the top[- ]?right( corner| edge)?", " ", rest)
        elif re.search(r"from the top", rest):
            g.region = "top"
            rest = re.sub(r"from the top( edge)?", " ", rest)
        elif re.search(r"from the bottom", rest):
            g.region = "bottom"
            rest = re.sub(r"from the bottom( edge)?", " ", rest)
        for d in DIRECTIONS:
            if re.search(rf"\b{d}\b", rest):
                g.direction = d
                break
        # "scroll down" means content up; keep user's word, engine maps it
        if low.startswith("scroll") and not g.direction:
            g.direction = "down"
        m = re.search(r"\bon (?:the )?(.+?)(?: row| card)?$", rest)
        if m:
            g.anchor_text = _clean(m.group(1))
        if not g.direction:
            return g  # will be refused with a helpful reason
        return g

    verb = TAP
    rest = low
    if low.startswith("long press") or low.startswith("long-press") \
            or low.startswith("hold "):
        verb = LONG_PRESS
        rest = re.sub(r"^(long[ -]press|hold)\s*", "", low)
    elif low.startswith(("tap", "click", "press", "select", "choose",
                         "open", "hit", "toggle", "turn on", "turn off",
                         "enable", "disable", "go to")):
        rest = re.sub(r"^(tap on|tap|click on|click|press|select|choose|open|"
                      r"hit|toggle|turn on|turn off|enable|disable|go to)\s*",
                      "", low)

    g = Gesture(verb, raw=t)
    if coord and ("(" in low or " at " in low or "coordinate" in low):
        g.point = (int(coord.group(1)), int(coord.group(2)))
        return g

    # relation: "the toggle next to Bluetooth", "the field below Email".
    # The keyword must stand alone (start-or-whitespace before it) so the
    # "In" inside hyphenated labels like "Sign-In & Security" never parses
    # as a spatial relation.
    # bare "in" needs "the" after it ("in the Wi-Fi section") — labels like
    # "Sign in with Apple" carry a bare " in " that is not a relation
    rel = re.search(r"(?:^|\s)(?:right |directly |immediately |straight )?"
                    r"(next to|beside|below|under|above|inside|in the) "
                    r"(?:the )?(.+)$", rest)
    if rel:
        kind = {"next to": "next_to", "beside": "next_to", "below": "below",
                "under": "below", "above": "above", "inside": "in",
                "in the": "in"}[rel.group(1)]
        g.relation = (kind, _clean(rel.group(2)))
        rest = rest[:rel.start()].strip()

    # region: "near the top right", "at the bottom", "top-left corner".
    # Accumulate matched region words into a hyphen-joined set so
    # "top right" -> "top-right" (region_of expects hyphen-split parts).
    region_parts = []
    for reg in ("top-right", "top-left", "bottom-right", "bottom-left"):
        if reg in rest:
            region_parts.extend(reg.split("-"))
            rest = rest.replace(reg, " ")
    # bare region words count only inside a locative phrase ("near the top",
    # "at the bottom", "on the right") — otherwise they stay in the target,
    # so labels like "Game Center" or "Top Hits" keep their words
    _RW = r"(?:top|bottom|left|right|middle|center)"
    rm = re.search(rf"\b(?:near|at|in|on|from|towards?)\s+(?:the\s+)?"
                   rf"({_RW}(?:\s+{_RW})?)"
                   rf"(?:\s+(?:corner|edge|half|of the screen))?\b", rest)
    if rm:
        for p in rm.group(1).split():
            region_parts.append("middle" if p == "center" else p)
        rest = rest[:rm.start()] + " " + rest[rm.end():]
    if region_parts:
        # dedupe preserving order
        seen = []
        for p in region_parts:
            if p not in seen:
                seen.append(p)
        g.region = "-".join(seen)
    rest = re.sub(r"\b(near|at|the|towards?|corner)\b", " ", rest).strip()

    # ordinal
    toks = rest.split()
    for tok in list(toks):
        if tok in _ORDINALS:
            g.ordinal = _ORDINALS[tok]
            toks.remove(tok)
    rest = " ".join(toks)

    g.target_text = _clean(rest)
    return g


def _clean(s: str) -> str:
    toks = []
    for w in re.split(r"[^\w%$@.+-]+", s.lower()):
        # trailing sentence punctuation is not part of a token ("fire." ==
        # "fire"), but interior dots survive ("3.5", "192.168.1.1")
        w = w.rstrip(".,!?;:")
        if w and w not in _STOP:
            toks.append(w)
    return " ".join(toks)


# ---- grounding -----------------------------------------------------------
def _text_of(w: Widget) -> str:
    return _clean(f"{w.label} {w.sub} {w.meta.get('hint','')}")


# ultra-generic UI words: they appear in many labels and must not, on their
# own, make a match. "tap the settings box" should hinge on a distinctive word.
_GENERIC = {"field", "box", "button", "option", "settings", "setting", "menu",
            "item", "screen", "page", "bar", "icon", "area", "section", "tab",
            "list", "row", "card", "toggle", "switch", "control", "text"}

# generic nouns that name a widget KIND: "the second toggle" is an ordinal
# over toggles, not a label search for the word "toggle"
_TYPE_WORDS = {"toggle": "toggle", "switch": "toggle", "row": "row",
               "field": "field", "box": "field", "slider": "slider",
               "checkbox": "checkbox", "radio": "radio", "button": "button"}


def _subtokens(tokens: set) -> set:
    """Augment tokens with their [-_.] parts, so "CoffeeBar_Guest" also
    answers to "guest" and "HP-Print-a2" to "print"."""
    out = set(tokens)
    for t in tokens:
        for p in re.split(r"[-_.]", t):
            if len(p) > 1:
                out.add(p)
    return out


def _match_score(query: str, w: Widget) -> float:
    """Score how well a noun phrase names a widget.

    The score hinges on the query's DISTINCTIVE words (everything but generic
    UI nouns): "password field" matches on 'password', "quantum settings" would
    need 'quantum' to appear and so is rejected even though 'settings' is in the
    search box. When a query is entirely generic ("the field"), fall back to
    plain recall. Exact-label and all-in-label bonuses break ties.
    """
    if not query:
        return 0.0
    label = _clean(w.label)
    hay = _text_of(w)
    if not hay:
        return 0.0
    q = set(query.split())
    h = _subtokens(set(hay.split()))
    if not q:
        return 0.0
    inter = q & h
    if not inter:
        return 0.55 if query in hay else 0.0
    dq = q - _GENERIC
    if dq:
        base = len(inter & dq) / len(dq)            # must match distinctive words
        base += 0.1 * len(inter & (q & _GENERIC))   # small credit for generic hits
    else:
        base = len(inter) / len(q)                  # all-generic query: plain recall
    exact_label = 0.5 if label == query else 0.0
    label_recall = 0.15 if q <= set(label.split()) else 0.0
    penalty = min(0.1, 0.03 * len(h - q))
    return base + exact_label + label_recall - penalty


def ground(g: Gesture, screen_widgets: list) -> Ground:
    """Resolve a gesture to a widget or point on the current screen."""
    interactive = [w for w in screen_widgets if w.visible and w.interactive()]

    if g.verb in (WAIT, PRESS_KEY, CLEAR):
        return Ground(True)

    if g.verb == SWIPE:
        if not g.direction:
            return Ground(False, reason="which direction should I swipe?")
        if g.anchor_text:
            cands = [w for w in screen_widgets if w.visible
                     and _match_score(g.anchor_text, w) >= 0.5]
            if not cands:
                return Ground(False, reason=f"I don't see {g.anchor_text} to swipe on")
            if len(cands) > 1 and not _pick_ordinal(cands, g):
                # narrow before calling it ambiguous: an exact label match wins, then a
                # widget that actually has a swipe action for this direction
                want = (g.anchor_text or "").strip().lower()
                exact = [w for w in cands if (w.label or "").strip().lower() == want]
                if exact:
                    cands = exact
                if len(cands) > 1:
                    swipable = [w for w in cands if w.on_swipe and g.direction in w.on_swipe]
                    if swipable:
                        cands = swipable
            if len(cands) > 1 and not _pick_ordinal(cands, g):
                return Ground(False, ambiguous=True, n_candidates=len(cands),
                              candidates=cands,
                              reason=f"there are {len(cands)} rows like that — which one?")
            w = _pick_ordinal(cands, g) or cands[0]
            return Ground(True, widget=w, point=(w.bounds.cx, w.bounds.cy))
        return Ground(True)  # whole-screen scroll

    if g.verb == TYPE:
        return Ground(True)  # focus checked by executor

    # tap / long_press
    if g.point:
        hit = _hit_test(g.point, screen_widgets)
        if hit is None:
            return Ground(True, point=g.point)  # taps empty space, harmless
        return Ground(True, widget=hit, point=g.point)

    cands = interactive

    # relation filter
    rel_anchor = None
    if g.relation:
        kind, anchor = g.relation
        anchors = [w for w in screen_widgets if w.visible
                   and _match_score(anchor, w) >= 0.6]
        if not anchors:
            return Ground(False, reason=f"I can't find {anchor} on this screen")
        a = anchors[0]
        rel_anchor = a
        cands = [w for w in cands if _relation_holds(kind, a, w)]
        if not cands:
            return Ground(False,
                          reason=f"I don't see anything {kind.replace('_',' ')} {anchor}")

    # region filter — a PREFERENCE, not a hard wall: if the text match later
    # finds nothing inside the region, we retry against these
    pre_region = cands
    if g.region:
        from .layout import region_of
        want = set(g.region.split("-"))
        rc = [w for w in cands if want <= region_of(w.bounds)]
        if rc:
            cands = rc

    # text match
    if g.target_text:
        qtoks = g.target_text.split()
        # pure type reference — "the second row", "the toggle below Wi-Fi
        # Password": generic nouns act as a KIND filter, and selection comes
        # from ordinal / relation-proximity / region, never label text.
        # EXCEPT when the phrase names an actual visible label ("Switch
        # Control" is a row, not a request for a switch) — labels win.
        names_a_label = any(set(qtoks) <= _subtokens(set(_clean(w.label).split()))
                            for w in cands if w.label)
        if qtoks and not names_a_label \
                and all(w in _GENERIC or w in _TYPE_WORDS for w in qtoks):
            kinds = {_TYPE_WORDS[w] for w in qtoks if w in _TYPE_WORDS}
            tc = [w for w in cands if not kinds or w.kind in kinds]
            if not tc:
                return Ground(False,
                              reason=f"I don't see a {qtoks[-1]} like that here")
            picked = None
            if g.ordinal is not None:
                picked = _pick_ordinal(tc, g)
            elif rel_anchor is not None:
                # "the row below X" means the NEAREST such widget to X
                picked = min(tc, key=lambda w: abs(w.bounds.cy
                                                   - rel_anchor.bounds.cy))
            elif g.region:
                picked = _pick_region(tc, g)
            if picked is None and len(tc) == 1:
                picked = tc[0]
            if picked is None:
                return Ground(False, ambiguous=True, n_candidates=len(tc),
                              candidates=tc,
                              reason=f"there are {len(tc)} of those — "
                                     "which one?")
            return Ground(True, widget=picked,
                          point=(picked.bounds.cx, picked.bounds.cy))
        scored = [(w, _match_score(g.target_text, w)) for w in cands]
        scored = [(w, s) for w, s in scored if s > 0.3]
        if not scored and g.region and pre_region is not cands:
            # nothing matched inside the named region — the region was a
            # preference; a solid label match just outside it wins
            scored = [(w, _match_score(g.target_text, w)) for w in pre_region]
            scored = [(w, s) for w, s in scored if s > 0.3]
        if not scored:
            # maybe it names a non-interactive element the user sees but can't act on
            seen = [w for w in screen_widgets if w.visible
                    and _match_score(g.target_text, w) > 0.5]
            if seen:
                return Ground(False,
                              reason=f"'{g.target_text}' isn't something you can tap")
            return Ground(False, reason=f"I can't see '{g.target_text}' on this screen")
        scored.sort(key=lambda p: -p[1])
        top = scored[0][1]
        # if something OFFSCREEN matches decisively better than anything
        # visible, don't tap the visible lookalike — the honest answer is
        # that the target needs scrolling into view first
        hidden_best = max((_match_score(g.target_text, w)
                           for w in screen_widgets
                           if not w.visible and w.interactive()), default=0.0)
        if hidden_best >= top + 0.15:
            return Ground(False,
                          reason=f"'{g.target_text}' isn't on screen yet — "
                                 "try scrolling")
        # if visible prose names the phrase decisively better than any
        # control, the user is pointing at text, not a control — refuse
        # honestly instead of tapping a lookalike widget
        best_ni = max((_match_score(g.target_text, w) for w in screen_widgets
                       if w.visible and not w.interactive()), default=0.0)
        if best_ni >= top + 0.15:
            return Ground(False,
                          reason=f"'{g.target_text}' is just text on this "
                                 "screen, not something you can tap")
        near = [w for w, s in scored if s >= top - 0.15]
        if len(near) > 1:
            picked = _pick_ordinal(near, g) or _pick_region(near, g)
            if picked is None:
                return Ground(False, ambiguous=True, n_candidates=len(near),
                              candidates=near,
                              reason=f"there are {len(near)} things like "
                                     f"'{g.target_text}' — which one?")
            return Ground(True, widget=picked, point=(picked.bounds.cx, picked.bounds.cy))
        w = scored[0][0]
        return Ground(True, widget=w, point=(w.bounds.cx, w.bounds.cy))

    # ordinal / region only (no text)
    if g.ordinal is not None and cands:
        picked = _pick_ordinal(cands, g)
        if picked:
            return Ground(True, widget=picked, point=(picked.bounds.cx, picked.bounds.cy))
    if g.region and cands:
        return Ground(True, widget=cands[0], point=(cands[0].bounds.cx, cands[0].bounds.cy))

    return Ground(False, reason="I'm not sure what you want me to tap")


def _pick_ordinal(cands, g: Gesture):
    if g.ordinal is None:
        return None
    ordered = sorted(cands, key=lambda w: (w.bounds.y, w.bounds.x))
    try:
        return ordered[g.ordinal]
    except IndexError:
        return None


def _pick_region(cands, g: Gesture):
    if not g.region:
        return None
    from .layout import region_of
    want = set(g.region.split("-"))
    rc = [w for w in cands if want <= region_of(w.bounds)]
    return rc[0] if len(rc) == 1 else None


def _relation_holds(kind, anchor: Widget, w: Widget) -> bool:
    a, b = anchor.bounds, w.bounds
    if w is anchor:
        return False
    if kind == "next_to":
        return abs(a.cy - b.cy) < max(a.h, b.h) and abs(a.cx - b.cx) < 300
    if kind == "below":
        return b.y >= a.y2 - 4 and abs(a.cx - b.cx) < 220
    if kind == "above":
        return b.y2 <= a.y + 4 and abs(a.cx - b.cx) < 220
    if kind == "in":
        return a.intersects(b) and (b.w * b.h) < (a.w * a.h)
    return False


def _hit_test(point, widgets):
    x, y = point
    hits = [w for w in widgets if w.visible and w.interactive()
            and w.bounds.contains(x, y)]
    if not hits:
        return None
    return min(hits, key=lambda w: w.bounds.w * w.bounds.h)
