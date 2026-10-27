#!/usr/bin/env python3
"""Aggregate python -m cooksim_bench.backends.controller_eval results: one row per (backend, mode) with win rate,
ticks vs the reference plan, decisions, latency, tokens and $/episode (from a price table).

    $PY python -m cooksim_bench.scoring.controller_eval_report tmp/ctrl_eval/*/ --prices tmp/ctrl_eval/prices.json [--html out.html]

prices.json: {"<model>": {"in": $/1M input tokens, "out": $/1M output tokens, "think": $/1M thinking (default out)}}
"""
import argparse, glob, json, os, statistics as st, sys, html

import os
from cooksim_bench.paths import ROOT  # the CookSim release root (benchmark/, web/, serving/ live here)
def load(dirs):
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if r.get("schema") != "controller-eval-v1":
                continue
            r["_file"] = f; rows.append(r)
    return rows

def ref_ticks():
    try:
        return json.load(open(os.path.join(ROOT, "benchmark/ref_ticks.json")))
    except Exception:
        return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--prices", default=None)
    ap.add_argument("--html", default=None)
    a = ap.parse_args()
    prices = json.load(open(a.prices)) if a.prices and os.path.exists(a.prices) else {}
    rows = load(a.dirs); ref = ref_ticks()
    groups = {}
    for r in rows:
        groups.setdefault((r["backend"], r["mode"]), []).append(r)
    out = []
    for (b, m), rs in sorted(groups.items()):
        n = len(rs); won = sum(1 for r in rs if r["won"])
        lat = [r["latency_ms"]["median"] for r in rs if r["latency_ms"]["median"] is not None]
        p90 = [r["latency_ms"]["p90"] for r in rs if r["latency_ms"]["p90"] is not None]
        dec = [r["decisions"] for r in rs]
        tt = []
        for r in rs:
            rt = ref.get(r["cell"]) if isinstance(ref, dict) else None
            rt = rt.get("ticks") if isinstance(rt, dict) else rt
            if r["won"] and rt:
                tt.append(min(1.0, rt / max(1, r["ticks"])))
        ptok = sum(r["tokens"]["prompt"] for r in rs); ctok = sum(r["tokens"]["completion"] for r in rs); ttok = sum(r["tokens"]["thinking"] for r in rs)
        pr = prices.get(b) or {}
        cost = None
        if pr:
            cost = (ptok * pr.get("in", 0) + ctok * pr.get("out", 0) + ttok * pr.get("think", pr.get("out", 0))) / 1e6
        out.append(dict(backend=b, mode=m, n=n, won=won, win_rate=round(won / n, 2),
                        mean_ticks=round(st.mean([r["ticks"] for r in rs]), 1),
                        TT=round(st.mean(tt), 2) if tt else None,
                        mean_decisions=round(st.mean(dec), 1),
                        lat_median_ms=round(st.median(lat)) if lat else None,
                        lat_p90_ms=round(st.median(p90)) if p90 else None,
                        prompt_tok_per_ep=round(ptok / n), completion_tok_per_ep=round(ctok / n), thinking_tok_per_ep=round(ttok / n),
                        usd_per_episode=(round(cost / n, 4) if cost is not None else None),
                        errors=sum(r.get("backend_errors", 0) for r in rs),
                        outcomes={o: sum(1 for r in rs if r["outcome"] == o) for o in ("won", "wrong_serve", "fire", "burned", "timeout")}))
    cols = ["backend", "mode", "n", "won", "win_rate", "mean_ticks", "TT", "mean_decisions", "lat_median_ms", "lat_p90_ms",
            "prompt_tok_per_ep", "completion_tok_per_ep", "thinking_tok_per_ep", "usd_per_episode", "errors", "outcomes"]
    w = {c: max(len(c), *(len(str(o[c])) for o in out)) for c in cols} if out else {}
    print(" | ".join(c.ljust(w[c]) for c in cols))
    for o in out:
        print(" | ".join(str(o[c]).ljust(w[c]) for c in cols))
    if a.html:
        per = []
        for r in sorted(rows, key=lambda r: (r["backend"], r["mode"], r["cell"])):
            per.append(f"<tr><td>{html.escape(r['backend'])}</td><td>{html.escape(r['mode'])}</td><td>{html.escape(r['cell'])}</td>"
                       f"<td>{r['outcome']}</td><td class=n>{r['ticks']}</td><td class=n>{r['decisions']}</td>"
                       f"<td class=n>{r['latency_ms']['median']}</td><td class=n>{r['tokens']['prompt']}</td><td class=n>{r['tokens']['completion']}</td>"
                       f"<td class=n>{r['dispositions'].get('refused',0)}/{r['dispositions'].get('grounding',0)}</td></tr>")
        summ = "".join("<tr>" + "".join(f"<td class={'n' if isinstance(o[c],(int,float)) else ''}>{html.escape(str(o[c]))}</td>" for c in cols) + "</tr>" for o in out)
        page = f"""<title>Controller Eval</title><style>body{{font-family:ui-serif,Georgia,serif;max-width:120ch;margin:40px auto;padding:0 20px;color:#191B1A;background:#FAFAF7}}
h1{{font-weight:500}}h2{{font:600 12px ui-monospace,monospace;letter-spacing:.14em;text-transform:uppercase;color:#1F5F4B;margin-top:40px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}}th,td{{border-bottom:1px solid #DCE0DA;padding:6px 9px;text-align:left;vertical-align:top}}
th{{font:600 10.5px ui-monospace,monospace;letter-spacing:.1em;text-transform:uppercase;color:#5C635F;background:#F2F3EF}}td.n{{font-family:ui-monospace,monospace;text-align:right}}
.scroll{{overflow-x:auto}}p{{max-width:72ch}}</style>
<h1>VLM as controller — model comparison</h1>
<p>The cook driven by one model, no assistant, no injection, on the benchmark's 10 cells (5 recipes × medium/hard maps). Decision every 2 ticks; the model's own JSON is its only memory; full engine feedback; strict v2 grounding. Two observation modes: the cook's text state map (ids + coordinates) and agent+++ (8 frames + exact readout). Cost uses the price table where known.</p>
<h2>Summary per model × mode</h2><div class=scroll><table><thead><tr>{''.join(f'<th>{c}</th>' for c in cols)}</tr></thead><tbody>{summ}</tbody></table></div>
<h2>Every episode</h2><div class=scroll><table><thead><tr><th>backend</th><th>mode</th><th>cell</th><th>outcome</th><th>ticks</th><th>decisions</th><th>lat med ms</th><th>prompt tok</th><th>completion tok</th><th>refused/grounding</th></tr></thead><tbody>{''.join(per)}</tbody></table></div>"""
        open(a.html, "w").write(page); print("wrote", a.html)

if __name__ == "__main__":
    main()
