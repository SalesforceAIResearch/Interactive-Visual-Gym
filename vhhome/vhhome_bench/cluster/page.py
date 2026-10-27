"""The progress page: what will be submitted, in order; what is done; what the verifier said."""
import datetime, html, json, os, zoneinfo

from vhhome_bench.cluster import plan as P

E = html.escape
PT = zoneinfo.ZoneInfo("America/Los_Angeles")
CHIP = {"done": "ok", "pass": "ok", "running": "run", "submitted": "run", "queued": "wait", "pending": "wait",
        "waiting": "wait", "warn": "warn", "fail": "bad", "failed": "bad", "blocked": "bad", "needs approval": "wait"}
CSS = """
:root{--bg:#F4F2ED;--surface:#fff;--ink:#1C2126;--muted:#667079;--line:#D9D5CC;--accent:#1F6F8B;--ok:#2E7D4F;--bad:#B4462E;--warn:#9A7B1C;--run:#1F6F8B}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15181B;--surface:#1D2226;--ink:#E6E8EA;--muted:#9AA3AC;--line:#30363C;--accent:#6BB6D1;--ok:#6CC08F;--bad:#E48268;--warn:#D6BE6A;--run:#6BB6D1}}
:root[data-theme="dark"]{--bg:#15181B;--surface:#1D2226;--ink:#E6E8EA;--muted:#9AA3AC;--line:#30363C;--accent:#6BB6D1;--ok:#6CC08F;--bad:#E48268;--warn:#D6BE6A;--run:#6BB6D1}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Source Sans 3","Segoe UI",system-ui,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 80px}
h1{font:600 1.9rem/1.2 "Fraunces",Georgia,serif;margin:0 0 8px}h2{font:600 1.25rem/1.3 "Fraunces",Georgia,serif;margin:0}
h3{font:600 1rem/1.3 "Source Sans 3",sans-serif;margin:18px 0 6px}
.lede{color:var(--muted);max-width:76ch}.mono{font:500 .82rem "JetBrains Mono",ui-monospace,monospace}
.phase{background:var(--surface);border:1px solid var(--line);margin:14px 0;padding:14px 16px}
.phase header{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;justify-content:space-between}
.phase .note{color:var(--muted);margin:4px 0 0;font-size:.92rem}
.chip{font:600 .72rem "JetBrains Mono",ui-monospace,monospace;padding:2px 8px;border:1px solid currentColor;white-space:nowrap;text-transform:uppercase}
.chip.ok{color:var(--ok)}.chip.bad{color:var(--bad)}.chip.warn{color:var(--warn)}.chip.run{color:var(--run)}.chip.wait{color:var(--muted)}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.88rem;margin:6px 0}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--muted);font-weight:500;font-size:.78rem}
.n{text-align:right;font-variant-numeric:tabular-nums}.bar{display:inline-block;width:90px;height:8px;background:var(--line);margin-right:6px;vertical-align:1px}
.bar i{display:block;height:100%;background:var(--accent)}.small{font-size:.82rem;color:var(--muted)}
.steps{counter-reset:s;list-style:none;padding:0;margin:14px 0}.steps li{counter-increment:s;display:grid;grid-template-columns:28px 1fr auto;gap:8px;padding:6px 0;border-bottom:1px solid var(--line)}
.steps li::before{content:counter(s);color:var(--muted);font:600 .85rem "JetBrains Mono",monospace}
.log{font:12px/1.5 "JetBrains Mono",ui-monospace,monospace;background:var(--surface);border:1px solid var(--line);padding:10px;max-height:320px;overflow:auto;white-space:pre-wrap}
a{color:var(--accent)}
"""


def pt(ts):
    return datetime.datetime.fromtimestamp(ts, PT).strftime("%b %-d %H:%M PT") if ts else "–"


def chip(s):
    return f'<span class="chip {CHIP.get(s, "wait")}">{E(str(s))}</span>'


def bar(n, tot):
    return f'<span class="bar"><i style="width:{100 * n / max(tot, 1):.0f}%"></i></span>{n}/{tot}'


def seat_rows(ph_state, lane, verify):
    vmap = {s["tag"]: s for s in (verify or {}).get("seats", [])}
    rows = []
    for tag, s in (ph_state.get(lane) or {}).items():
        v = vmap.get(tag) or {}
        where = ""
        if lane == "gpu" and s.get("exp"):
            where = f'<a href="https://beaker.org/ex/{E(s["exp"])}">Beaker</a> {E(str(s.get("jobs") or ""))}'
        elif lane == "api" and s.get("where") == "beaker":
            where = (f'<a href="https://beaker.org/ex/{E(s.get("exp") or "")}">Beaker</a> CPU job, the A100 cluster / the CPU cluster, no GPU '
                     f'{E(str(s.get("jobs") or ""))}')
        elif lane == "api" and s.get("ports"):
            where = f'local engines {s["ports"][0]}–{s["ports"][-1]}'
        notes = "; ".join((v.get("problems") or []) + (v.get("warnings") or [])) or E(s.get("note") or "")
        oc = v.get("outcomes") or s.get("outcomes") or {}
        ocs = " · ".join(f"{k} {n}" for k, n in sorted(oc.items(), key=lambda kv: -kv[1]))
        rows.append(f'<tr><td class="mono">{E(tag)}</td><td>{chip(s.get("status", "pending"))}</td>'
                    f'<td class="n">{bar(s.get("n", 0), P.N)}</td><td class="small">{E(ocs)}</td>'
                    f'<td>{chip(v["status"]) if v else ""}</td><td class="small">{where}</td><td class="small">{notes}</td></tr>')
    return "".join(rows)


def results_section():
    """The final table, once vhhome_bench/scoring/final_table.py has written it (three runs pooled)."""
    f = os.path.join(P.ENG, "tmp", "bench2", "final_results.json")
    if not os.path.exists(f):
        return ""
    d = json.load(open(f))

    def num(x, k=2):
        return "—" if x is None else f"{x:.{k}f}"

    def pm(r, k):
        return "—" if r.get(k) is None else f"{r[k]:.2f} <span class=small>± {num(r.get(k + '_spread'))}</span>"
    rows = []
    for r in d["rows"]:
        aq = ("—" if r["AQ"] is None else
              f"<b>{r['AQ']:.3f}</b> <span class=small>[{num(r['AQ_lo'])}, {num(r['AQ_hi'])}]</span>")
        runs = " / ".join(num(x, 3) for x in r["AQ_runs"])
        rows.append(f'<tr><td class="mono">{E(r["seat"])}</td><td>{aq}</td><td class="small">{runs}</td><td>{pm(r, "won")}</td>'
                    f'<td>{pm(r, "S")}</td><td>{pm(r, "EFF")}</td><td>{pm(r, "DET")}</td><td class="small">{pm(r, "DT")}</td>'
                    f'<td class="small">{pm(r, "F1")}</td></tr>')
    notes = "".join(f"<li>{E(n)}</li>" for n in d["notes"])
    return (f'<section class="phase" id="results"><header><h2>Final table — three independent runs pooled</h2>{chip("done")}</header>'
            f'<div class="tbl"><table><tr><th>seat</th><th>AQ [95% CI]</th><th>AQ r1 / r2 / r3</th><th>won</th><th>S (in-time)</th>'
            f'<th>EFF</th><th>DET</th><th>DT</th><th>F1</th></tr>{"".join(rows)}</table></div><ul class="small">{notes}</ul></section>')


def render(st):
    phases = P.phases()
    steps, blocks = [], []
    for ph in phases:
        s = st["phases"].get(ph["id"], {})
        status = s.get("status", "pending")
        if ph.get("needs_approval") and ph["id"] not in st.get("approved", []) and status == "pending":
            status = "needs approval"
        gpu_only = bool(s.get("gpu_only"))
        steps.append(f'<li><span><b>{E(ph["title"])}</b><br><span class="small">{E(ph["note"])}</span></span>'
                     f'<span>{chip(status)}<br><span class="small">{pt(s.get("done_at")) if s.get("done_at") else ""}</span></span></li>')
        body = ""
        if ph["kind"] == "preflight":
            pf = st.get("preflight", {})
            loc = "".join(f'<tr><td class="mono">{E(k)}</td><td>{chip(v.get("status", "pending"))}</td><td class="small">{E(v.get("note", ""))}</td></tr>'
                          for k, v in (pf.get("local") or {}).items())
            gpu = "".join(f'<tr><td class="mono">{E(k)}</td><td>{chip(v.get("status", "pending"))}</td>'
                          f'<td class="small">{("<a href=https://beaker.org/ex/" + E(v["exp"]) + ">Beaker</a> ") if v.get("exp") else ""}{E(str(v.get("jobs") or ""))}</td>'
                          f'<td class="small">{E(v.get("note", ""))}</td></tr>' for k, v in (pf.get("gpu") or {}).items())
            body = (f'<h3>Local episodes</h3><div class="tbl"><table><tr><th>episode</th><th>status</th><th>what the verifier saw</th></tr>{loc}</table></div>'
                    f'<h3>Check job per open-weight seat (the H100 cluster, H100)</h3><div class="tbl"><table><tr><th>seat</th><th>status</th><th>job</th><th>what the verifier saw</th></tr>{gpu}</table></div>')
        elif ph["kind"] == "bench" and s:
            head = ('<tr><th>seat</th><th>lane status</th><th class="n">episodes</th><th>outcomes</th><th>verifier</th><th>where</th><th>notes</th></tr>')
            v = s.get("verify")
            api_html = ('<p class="small">API lane: not started — only the GPU lane of this run has been approved.</p>' if gpu_only else
                        f'<h3>API lane — local, {P.API_CONCURRENT} seats at a time</h3><div class="tbl"><table>{head}{seat_rows(s, "api", v)}</table></div>')
            body = (api_html
                    + f'<h3>GPU lane — the H100 cluster H100, {P.BEAKER["replicas"]} replicas per seat, latency charged as measured</h3>'
                    f'<div class="tbl"><table>{head}{seat_rows(s, "gpu", v)}</table></div>'
                    + (f'<p class="small">Verifier: {chip(v["status"])} run {pt(v.get("at"))}</p>' if v else ""))
        elif ph["kind"] == "score" and s:
            body = '<div class="tbl"><table><tr><th>step</th><th>status</th><th>note</th></tr>' + "".join(
                f'<tr><td>{E(k)}</td><td>{chip(v.get("status", "pending"))}</td><td class="small">{E(v.get("note", ""))}</td></tr>'
                for k, v in (s.get("steps") or {}).items()) + "</table></div>"
        if body:
            blocks.append(f'<section class="phase" id="{E(ph["id"])}"><header><h2>{E(ph["title"])}</h2>{chip(status)}</header>'
                          f'<p class="note">{E(ph["note"])}</p>{body}</section>')
    log = "\n".join(st.get("log", [])[-60:][::-1])
    b = P.BEAKER
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VH-Home formal run 2</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style></head><body><main>
<h1>VH-Home benchmark, formal run on the no-self-repair person</h1>
<p class="lede">VH-Home puts a simulated person in a 3D house to do a chore while an assistant watches and speaks; the engine
makes the person commit two mistakes per episode. In the earlier runs the person often repaired its own mistakes, so an
episode could be won with no help at all. This run uses the engine in which a mistake stays until the assistant gets
the person to fix it. It repeats the same 75 episodes (25 cases, three personas each) for 17 assistants. The plan below
is carried out top to bottom by a daemon; a phase is marked done only after the verifier has read every record.</p>
<div class="tbl"><table>
<tr><th>engine stamp</th><td class="mono">{E(", ".join(st["stamp"]) if isinstance(st.get("stamp"), list) else str(st.get("stamp")))} · commit {E(str(st.get("commit")))}</td></tr>
<tr><th>stamp note</th><td>f125 softens one rule: an episode is ended for a looping assistant only after 30 ticks of standstill (before: at once). Episodes the old rule never ended play out the same, so their f124 records stand; the ones it did end are re-run.</td></tr>
<tr><th>API lane</th><td>three polled API assistants, the two streaming APIs in their native setting (a live session pushed one frame per tick and the person's speech as audio; the model speaks when it chooses) and the silent control, run on local engines, {P.API_CONCURRENT} seats at a time, {P.ENGINES_PER_SEAT} engines per seat</td></tr>
<tr><th>GPU lane</th><td>eleven open-weight assistants on {E(b["cluster"])} (H100), workspace {E(b["workspace"])}, {b["replicas"]} replicas per seat; generation time charged as measured, no A100 factor</td></tr>
<tr><th>started</th><td>{pt(st.get("started"))} · page written {pt(st.get("rendered"))}</td></tr>
</table></div>
{results_section()}
<h3>The plan, in order</h3><ol class="steps">{''.join(steps)}</ol>
{''.join(blocks)}
<h3>What the verifier checks</h3>
<p class="small">Blocking: all 75 records of a seat present and readable, none voided; the run's engine stamp on every record; an outcome on
every record; for open-weight seats an H100 and a latency factor of 1.0. Reported: the assistant spoke in at least half
its episodes; at least 60% of designed mistakes fired; timeouts at most 30%; and for the silent control, at most 10% of
episodes with an outcome-breaking mistake still won (47% on the old engine).</p>
<h3>Daemon log (newest first)</h3><div class="log">{E(log)}</div>
</main></body></html>"""
