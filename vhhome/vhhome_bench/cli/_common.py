"""Shared pieces of the console scripts: the --assistant spec, key checks, user case files."""
import os, sys

OPENAI_URL = "https://api.openai.com/v1"

ASSISTANT_HELP = """\
--assistant takes one of:
  <roster tag>             an assistant of the published table, e.g. gemini-3.8-flash (vhhome-roster lists them)
  silent                   the control that never speaks; needs no key
  gemini:<model>           a Gemini model through generateContent            (GEMINI_API_KEY)
  openai:<model>           any OpenAI-compatible chat endpoint: --base-url (default api.openai.com) and
                           IV_ASSIST_API_KEY or OPENAI_API_KEY. This is how a model you serve yourself is evaluated.
  live:<model>             Gemini Live, polled every two ticks               (GEMINI_API_KEY)
  live-native:<model>      Gemini Live in its native streaming setting       (GEMINI_API_KEY + OPENAI_API_KEY for the person's voice)
  realtime:<model>         GPT Realtime, polled every two ticks              (OPENAI_API_KEY)
  realtime-native:<model>  GPT Realtime in its native streaming setting      (OPENAI_API_KEY)
The simulated person is always a Gemini model, so GEMINI_API_KEY is needed for every assistant but the
offline tools. Keys: environment variables, or ~/.config/vhhome/keys.env (one KEY=value per line).
"""


def assistant_entry(spec, base_url=None, profile=None, image_scale=None, max_tok=None, tag=None):
    """Turn an --assistant spec into a roster-style entry {tag, kind, needs, env}."""
    from vhhome_bench import roster
    try:
        e = {k: (dict(v) if isinstance(v, dict) else v) for k, v in roster.by_tag(spec).items()}
    except KeyError:
        kind, _, model = spec.partition(":")
        if spec == "silent":
            e = dict(tag="silent", kind="control", needs=[], env={"IV_ASSIST_MODEL": "silent"})
        elif kind == "gemini" and model:
            e = dict(tag=model, kind="api", needs=["GEMINI_API_KEY"], env={"IV_ASSIST_MODEL": model})
        elif kind == "openai" and model:
            prof = profile or ("gpt5" if model.startswith(("gpt-5", "gpt-6", "o")) else "default")
            e = dict(tag=model, kind="api", needs=[] if base_url else ["OPENAI_API_KEY"],
                     env={"IV_ASSIST_MODEL": model, "IV_ASSIST_BASE_URL": base_url or OPENAI_URL,
                          "IV_ASSIST_PROFILE": prof})
        elif kind == "live" and model:
            e = dict(tag=model, kind="api", needs=["GEMINI_API_KEY"],
                     env={"IV_ASSIST_BACKEND": "live", "IV_ASSIST_MODEL": model})
        elif kind == "live-native" and model:
            e = dict(tag=model + "__native", kind="native", needs=["GEMINI_API_KEY", "OPENAI_API_KEY"],
                     env={"IV_ASSIST_BACKEND": "live_native", "IV_ASSIST_MODEL": model})
        elif kind == "realtime" and model:
            e = dict(tag=model, kind="api", needs=["OPENAI_API_KEY"],
                     env={"IV_ASSIST_BACKEND": "realtime", "IV_ASSIST_MODEL": model})
        elif kind == "realtime-native" and model:
            e = dict(tag=model + "__native", kind="native", needs=["OPENAI_API_KEY"],
                     env={"IV_ASSIST_BACKEND": "realtime_native", "IV_ASSIST_MODEL": model})
        else:
            sys.exit(f"unknown assistant {spec!r}\n\n{ASSISTANT_HELP}")
    if e.get("status") in ("not_ported", "dropped"):
        sys.exit(f"{spec}: roster status '{e['status']}' — see vhhome-roster")
    env = e.setdefault("env", {})
    if base_url:
        env["IV_ASSIST_BASE_URL"] = base_url
    if profile:
        env["IV_ASSIST_PROFILE"] = profile
    if image_scale:
        env["IV_ASSIST_IMAGE_SCALE"] = str(image_scale)
    if max_tok:
        env["IV_ASSIST_MAX_TOK"] = str(max_tok)
    if tag:
        e["tag"] = tag
    if "GEMINI_API_KEY" not in e.setdefault("needs", []):
        e["needs"] = list(e["needs"]) + ["GEMINI_API_KEY"]        # the simulated person
    return e


def require_keys(entry):
    from vhhome_bench import roster
    miss = roster.missing_keys(entry)
    if miss:
        sys.exit(f"{entry['tag']}: missing {', '.join(miss)} — export it or add it to ~/.config/vhhome/keys.env")


def use_cases(path):
    """Make a user case file (vhhome-case new) visible to this process and its episode subprocesses."""
    if not path:
        return
    p = os.path.abspath(path)
    if not os.path.exists(p):
        sys.exit(f"case file not found: {p}")
    cur = [x for x in os.environ.get("VHHOME_CASES", "").split(os.pathsep) if x]
    if p not in cur:
        os.environ["VHHOME_CASES"] = os.pathsep.join(cur + [p])


def add_assistant_args(ap, default=None):
    ap.add_argument("--assistant", required=default is None, default=default,
                    help="who watches: roster tag | silent | gemini:<m> | openai:<m> | live:<m> | realtime:<m> | ...-native:<m>")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint for openai:<model> (your own server)")
    ap.add_argument("--profile", default=None, help="request profile for openai:<model> (default | gpt5 | ...; see protocol._PROFILES)")
    ap.add_argument("--image-scale", type=float, default=None, help="downscale the frames sent to the assistant (0.5 for small contexts)")
    ap.add_argument("--max-tok", type=int, default=None, help="the assistant's max output tokens per turn (default 4000)")
    ap.add_argument("--tag", default=None, help="name of the record folder (default: the model or roster tag)")
    ap.add_argument("--latency", choices=["2", "measured", "0"], default=None,
                    help="2: fixed two-tick delay (main benchmark); measured: inference-time delay; 0: custom instantaneous setting. Omitted: ASSIST_LATENCY or measured.")
    ap.add_argument("--cases", default=None, help="a case file written by vhhome-case, in addition to the shipped cases")
