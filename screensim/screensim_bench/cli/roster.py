"""screensim-roster: the assistant seats this release knows how to drive."""
import argparse

from screensim.interact.roster import ROSTER


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.json:
        import json
        print(json.dumps(ROSTER, indent=1, default=str))
        return 0
    for e in ROSTER:
        print(f'{e["tag"]:26} backend={str(e.get("backend")):10} model={e.get("model")}')
    print(f"\n{len(ROSTER)} seats. Any OpenAI-compatible server works too: "
          f"--assistant openai:<model> with --base-url.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
