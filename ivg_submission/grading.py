"""Maintainer-only grading handoff. Submitted programs are never run here."""

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

from .bundle import manifest, validate
from .common import (
    catalog,
    digest,
    encoded,
    file_hash,
    loads,
    read_json,
    require,
    write_new,
)

CATS = {
    "factual_grounding": ("T1", "T2"),
    "situational_relevance": ("S1", "S2"),
    "actionable_guidance": ("H1", "H2", "H3"),
    "user_intent_uptake": ("L1", "L2"),
    "guidance_conciseness": ("E1",),
}
JUDGES = ("gemini", "gpt")
AGGREGATION = "five-categories-per-judge-factual-gate-then-two-judge-mean-v1"


def native_module(name):
    return importlib.import_module(name)


def round_score(row):
    """Gate EACH judge's round, then average the two scores (not AND across judges)."""
    scores, category_scores = [], defaultdict(list)
    for judge in JUDGES:
        verdict = (row.get("judges") or {}).get(judge)
        require(
            isinstance(verdict, dict) and not verdict.get("error"),
            f"Missing or failed {judge} judgment",
        )
        items = verdict.get("items")
        require(isinstance(items, dict), "Missing rubric items")
        flags = {}
        for cat, keys in CATS.items():
            for key in keys:
                require(
                    key in items
                    and isinstance(items[key], dict)
                    and "pass" in items[key],
                    f"Incomplete rubric: {key}",
                )
                require(
                    items[key]["pass"] is None or type(items[key]["pass"]) is bool,
                    "Rubric verdict must be true, false, or null",
                )
            applicable = [
                items[k]["pass"] for k in keys if items[k]["pass"] is not None
            ]
            if applicable:
                flags[cat] = float(all(applicable))
                category_scores[cat].append(flags[cat])
        if flags:
            gate = not any(items[k]["pass"] is False for k in ("T1", "T2"))
            scores.append(fmean(flags.values()) if gate else 0.0)
        else:
            scores.append(None)
    # Keep the score's denominator explicit when a judge marks the entire turn N/A.
    available = [v for v in scores if v is not None]
    return {
        "score": fmean(available) if available else None,
        "judges_applicable": len(available),
        "categories": {k: fmean(v) for k, v in category_scores.items()},
    }


def native_path(work, e):
    work = Path(work)
    run, case, persona = e["run"], e["case"], e["persona"]
    if e["engine"] == "cooksim":
        tag = "entry" + (f"__r{run}" if run > 1 else "")
        return work / "data/cooksim/frames" / persona / tag / f"{case}.json"
    if e["engine"] == "vhhome":
        return work / f"data/vhhome/run-{run}/entry/entry__{case}__{persona}.json"
    return work / f"data/screensim/run-{run}/entry.records.jsonl"


def judge_path(work, e):
    p = native_path(work, e)
    if e["engine"] == "cooksim":
        root = Path(work) / "data/cooksim"
        return root / "sq_v4" / p.relative_to(root)
    if e["engine"] == "vhhome":
        return p.parent.parent / "sq_v4" / p.parent.name / p.name
    return p.parent / "aq_v4five/entry" / f"{e['case']}__{e['persona']}.json"


def stage(bundle, work, allow_preview=False):
    report = validate(bundle, allow_preview=allow_preview)
    m = manifest(bundle)
    work = Path(work).resolve()
    require(
        not work.exists(), "Grading workspace exists; use resume instead of restaging"
    )
    work.mkdir(parents=True)
    with zipfile.ZipFile(bundle) as z:
        for e in m["episodes"]:
            target = native_path(work, e)
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = z.read(e["record"])
            if e["engine"] == "screensim":
                record = loads(raw)
                record["contestant"] = (
                    "entry"  # fixed safe directory tag; identity remains in manifest
                )
                with target.open("ab") as f:
                    f.write(json.dumps(record, allow_nan=False).encode() + b"\n")
            else:
                with target.open("xb") as f:
                    f.write(raw)
                for kind, name in e["evidence"].items():
                    suffix = ".mp4" if kind == "video" else ".states.jsonl"
                    with (
                        z.open(name) as src,
                        target.with_suffix(suffix).open("xb") as dst,
                    ):
                        for block in iter(lambda: src.read(1024 * 1024), b""):
                            dst.write(block)
    write_new(work / "manifest.json", m)
    write_new(work / "validation.json", report)
    plan = commands(work, m)
    # Pin all inputs and trusted scoring source, so resume cannot silently reuse another grading version.
    write_new(
        work / "job.json",
        {
            "manifest_sha256": report["manifest_sha256"],
            "aggregation": AGGREGATION,
            "judges": catalog()["judges"],
            "commands": plan,
            "inputs": {
                str(p.relative_to(work)): file_hash(p)
                for p in (work / "data").rglob("*")
                if p.is_file()
            },
        },
    )
    return {"work": str(work), "jobs": len(plan), "status": "staged-not-graded"}


def commands(work, m):
    """Fixed module allowlist. No command, import, shell snippet, or URL from a submitter."""
    work = Path(work).resolve()
    pairs = sorted({(e["engine"], e["run"]) for e in m["episodes"]})
    result = []
    if any(e == "cooksim" for e, _ in pairs):
        root = str(work / "data/cooksim")
        for module, args in (
            ("regrade_qa", ["--workers", "4", "--model", "gemini-3.7-flash"]),
            ("regrade_proactive_v2", ["--model", "gemini-3.7-flash"]),
            ("eval_v2_sweep", []),
            (
                "sq_v4_score",
                ["--judge", "gemini,gpt", "--categories", "all", "--workers", "4"],
            ),
        ):
            result.append(
                {
                    "module": "cooksim_bench.scoring." + module,
                    "args": args,
                    "env": {"ASSIST_ROOT": root},
                }
            )
    for engine, run in pairs:
        root = str(work / f"data/{engine}/run-{run}")
        if engine == "vhhome":
            result.append(
                {
                    "module": "vhhome_bench.scoring.rubric",
                    "args": [
                        root,
                        "--subset",
                        "--judge",
                        "gemini,gpt",
                        "--workers",
                        "4",
                    ],
                    "env": {},
                }
            )
        elif engine == "screensim":
            result.append(
                {
                    "module": "screensim_bench.scoring.rubric",
                    "args": [
                        root,
                        "--subset",
                        "five",
                        "--judge",
                        "gemini,gpt",
                        "--workers",
                        "4",
                    ],
                    "env": {},
                }
            )
    return result


def check_workspace(work):
    work = Path(work).resolve()
    job, m = read_json(work / "job.json"), read_json(work / "manifest.json")
    require(job["manifest_sha256"] == digest(encoded(m)), "Staged manifest changed")
    require(
        m["protocol"] == catalog()["version"]
        and m["catalog_sha256"] == digest(encoded(catalog())),
        "Staged protocol differs from the current frozen catalog",
    )
    require(
        job["aggregation"] == AGGREGATION and job["judges"] == catalog()["judges"],
        "Grading version changed",
    )
    require(job["commands"] == commands(work, m), "Unexpected grading commands")
    for name, want in job["inputs"].items():
        p = work / name
        require(
            p.resolve().is_relative_to(work) and not p.is_symlink() and p.is_file(),
            "Unsafe staged input",
        )
        require(file_hash(p) == want, "Staged input changed; create a fresh job")
    return job, m


def scorer_fingerprint():
    """Pin installed native scoring code and benchmark assets, not only a human-readable tag."""
    hashes, roots = {}, {}
    for name in (
        "ivg_submission",
        "cooksim",
        "cooksim_bench",
        "vhhome",
        "vhhome_bench",
        "screensim",
        "screensim_bench",
    ):
        spec = importlib.util.find_spec(name)
        require(
            spec is not None and spec.submodule_search_locations,
            f"Install trusted package {name} before grading",
        )
        # From the monorepo root, the outer engine directory can be a namespace
        # package; resolve the installed package's actual __init__.py instead.
        initializer = (
            importlib.util.find_spec(name + ".__init__")
            if spec.origin is None
            else spec
        )
        require(
            initializer is not None and initializer.origin,
            f"Cannot locate installed package {name}",
        )
        root = Path(initializer.origin).parent
        roots[name] = root
        for p in sorted(root.rglob("*.py")):
            hashes[f"{name}/{p.relative_to(root)}"] = file_hash(p)
        assets = root.parent / "benchmark"
        if assets.is_dir():
            for p in sorted(assets.rglob("*")):
                if p.is_file() and p.suffix in (".json", ".jsonl"):
                    hashes[f"{name}/benchmark/{p.relative_to(assets)}"] = file_hash(p)
    for relative, expected in catalog()["source_sha256"].items():
        engine, rest = relative.split("/", 1)
        require(
            file_hash(roots[engine].parent / rest) == expected,
            "Suite definitions differ from the frozen catalog; create a new protocol version",
        )
    return digest(encoded(hashes))


def execute(work, allow_paid=False):
    require(
        allow_paid,
        "Judging makes paid API calls; use --allow-paid-judging after reviewing the job",
    )
    work = Path(work).resolve()
    job, _ = check_workspace(work)
    fingerprint_path = work / "scorer.json"
    fingerprint = {
        "source_sha256": scorer_fingerprint(),
        "python": sys.version.split()[0],
    }
    if fingerprint_path.exists():
        require(
            read_json(fingerprint_path) == fingerprint,
            "Scoring code changed; do not reuse cached judgments",
        )
    else:
        write_new(fingerprint_path, fingerprint)
    # Only a maintainer-owned, isolated worker should run this command. No inference code is mounted.
    for i, command in enumerate(job["commands"]):
        with (work / f"job-{i:02d}.log").open("ab") as log:
            done = subprocess.run(
                [sys.executable, "-m", command["module"], *command["args"]],
                cwd=work,
                env=dict(os.environ, **command["env"]),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                timeout=24 * 60 * 60,
            )
        require(
            done.returncode == 0,
            f"Grading job {i} failed; inspect its private log before resuming",
        )
    return {
        "status": "judge-jobs-finished",
        "next": "ivg-submit collect --work <workspace>",
    }


def collect(work):
    """Recompute objective metrics and strict two-judge quality; never read submitter score summaries."""
    work = Path(work).resolve()
    _, m = check_workspace(work)
    provenance = read_json(work / "scorer.json")
    require(
        provenance["source_sha256"] == scorer_fingerprint(),
        "Scoring source differs from the executed job",
    )
    by_run = defaultdict(list)
    screen_cache = {}
    cook_objective = None
    cook_cases, qa_grades, pro_grades = {}, {}, {}
    evidence_hashes = {}
    for e in m["episodes"]:
        engine, run = e["engine"], e["run"]
        p = native_path(work, e)
        if engine == "screensim":
            if p not in screen_cache:
                screen_cache[p] = {
                    (r["episode"], r["persona"]): r
                    for r in (
                        loads(line)
                        for line in p.read_bytes().splitlines()
                        if line.strip()
                    )
                }
            record = screen_cache[p][(e["case"], e["persona"])]
        else:
            record = read_json(p)
        rubric_module = {
            "cooksim": "cooksim_bench.scoring.sq_v4_score",
            "vhhome": "vhhome_bench.scoring.rubric",
            "screensim": "screensim_bench.scoring.rubric",
        }[engine]
        expected = native_module(rubric_module).rounds_of(record)
        graded_path = judge_path(work, e)
        require(
            graded_path.is_file(),
            f"Missing judged episode: {engine}/{run}/{e['case']}/{e['persona']}",
        )
        graded = read_json(graded_path)
        rows = graded.get("rounds_detail" if engine == "screensim" else "rounds")
        require(
            isinstance(rows, list) and len(rows) == len(expected),
            "Missing or extra judged rounds",
        )
        require(
            [r.get("round") for r in rows] == [r.get("round") for r in expected],
            "Round identity mismatch",
        )
        evidence_hashes[str(graded_path.relative_to(work))] = file_hash(graded_path)
        quality = [round_score(r) for r in rows]
        if engine == "cooksim":
            if cook_objective is None:
                objective_path = work / "data/cooksim/eval_v2.json"
                cook_objective = read_json(objective_path)["episodes"]
                evidence_hashes[str(objective_path.relative_to(work))] = file_hash(
                    objective_path
                )
                qa_path = work / "data/cooksim/qa_regrade.json"
                pro_path = work / "data/cooksim/proactive_v2.json"
                qa_grades, pro_grades = read_json(qa_path), read_json(pro_path)
                evidence_hashes[str(qa_path.relative_to(work))] = file_hash(qa_path)
                evidence_hashes[str(pro_path.relative_to(work))] = file_hash(pro_path)
                from cooksim_bench.paths import SUITE

                cook_cases = {c["name"]: c for c in read_json(SUITE)["cases"]}
            key = str(p.relative_to(work / "data/cooksim").with_suffix(""))
            metric = cook_objective[key]
            require(
                metric["v3"]["cap_ticks"] is not None,
                "Missing CookSim reference duration",
            )
            objective = {
                "success": metric["v3"]["in_time_success"],
                "detected": metric["v3"]["tp"],
                "triggered": metric["evidence"]["errors_fired"],
                "false_alarms": metric["v3"]["fp"],
            }
            qa_lib = native_module("cooksim_bench.scoring.qa_v3_lib")
            answers = []
            questions_reached = proactive_reached = 0
            for event in record["injections"]:
                if not event.get("injected") or event.get("family") not in (
                    "qa_probe",
                    "qa_proactive",
                ):
                    continue
                event_key = f"{key}#{event['idx']}"
                if event["family"] == "qa_probe":
                    questions_reached += 1
                    require(
                        event_key in qa_grades
                        and not qa_grades[event_key].get("error")
                        and qa_grades[event_key].get("verdict") != "judge_error",
                        "Missing or failed independent QA grade",
                    )
                    g = qa_grades[event_key]
                    score = qa_lib.qa_score(
                        g.get("verdict"), qa_lib.family_of(event.get("gt_kind") or "")
                    )
                else:
                    proactive_reached += 1
                    require(
                        event_key in pro_grades
                        and not pro_grades[event_key].get("error"),
                        "Missing independent proactive grade",
                    )
                    score = pro_grades[event_key].get("score")
                require(
                    score is None or (type(score) in (int, float) and 0 <= score <= 1),
                    "Invalid QA score",
                )
                answers.append((event["family"], score))
            # Frozen definitions determine scheduled counts; reached/scored depend on the rollout.
            case = cook_cases[e["case"]]
            objective.update(
                errors_scheduled=sum(s.get("family") == "error" for s in case["subs"])
            )
            qa = {
                "scores": [s for _, s in answers if s is not None],
                "questions": {
                    "scheduled": sum(
                        s.get("family") == "qa_probe" for s in case["subs"]
                    ),
                    "reached": questions_reached,
                    "scored": sum(
                        f == "qa_probe" and s is not None for f, s in answers
                    ),
                },
                "proactive": {
                    "scheduled": sum(
                        s.get("family") == "qa_proactive" for s in case["subs"]
                    ),
                    "reached": proactive_reached,
                    "scored": sum(
                        f == "qa_proactive" and s is not None for f, s in answers
                    ),
                },
            }
        elif engine == "vhhome":
            metric = native_module("vhhome_bench.scoring.metrics").episode_metrics(
                record
            )
            objective = {
                "success": metric["S"],
                "detected": metric["tp"],
                "triggered": metric["fired"],
                "false_alarms": metric["fp"],
                "errors_scheduled": metric["phases"],
            }
            qa = None
        else:
            metric = native_module("screensim_bench.scoring.scores").scores(record)
            # ScreenSim detection still depends on native per-beat engine verdicts. Retain these for audit.
            fired = [b for b in record["beats"] if b.get("injected_at") is not None]
            objective = {
                "success": metric["S"],
                "triggered": len(fired),
                "detected": sum(b.get("detected") is True for b in fired),
                "false_alarms": record.get("false_flags"),
                "errors_scheduled": len(record["beats"]),
            }
            qa = None
        by_run[(engine, run)].append(
            {
                "case": e["case"],
                "persona": e["persona"],
                "objective": objective,
                "rounds": quality,
                "qa": qa,
            }
        )
    engines = {}
    for engine in sorted({e for e, _ in by_run}):
        runs = {}
        for (eng, run), rows in sorted(by_run.items()):
            if eng != engine:
                continue
            q = [r for ep in rows for r in ep["rounds"] if r["score"] is not None]
            triggered = sum(ep["objective"]["triggered"] for ep in rows)
            detected = [ep["objective"]["detected"] for ep in rows]
            runs[str(run)] = {
                "episodes": len(rows),
                "success": fmean(ep["objective"]["success"] for ep in rows),
                "interaction_quality": fmean(r["score"] for r in q) if q else None,
                "rounds_scored": len(q),
                "rounds_reached": sum(len(ep["rounds"]) for ep in rows),
                "errors_triggered": triggered,
                "errors_scheduled": sum(
                    ep["objective"]["errors_scheduled"] for ep in rows
                ),
                "false_alarms": sum(
                    ep["objective"]["false_alarms"] or 0 for ep in rows
                ),
                "detection_recall": sum(detected) / triggered
                if triggered and all(v is not None for v in detected)
                else None,
                "categories": {
                    c: fmean(vals)
                    if (vals := [r["categories"][c] for r in q if c in r["categories"]])
                    else None
                    for c in CATS
                },
            }
            qa = [ep["qa"] for ep in rows if ep["qa"] is not None]
            answers = [s for ep in qa for s in ep["scores"]]
            runs[str(run)]["qa_accuracy"] = fmean(answers) if answers else None
            runs[str(run)]["qa_coverage"] = (
                {
                    family: {
                        k: sum(ep[family][k] for ep in qa)
                        for k in ("scheduled", "reached", "scored")
                    }
                    for family in ("questions", "proactive")
                }
                if qa
                else None
            )
        summary = {}
        for key in (
            "success",
            "interaction_quality",
            "detection_recall",
            "qa_accuracy",
        ):
            vals = [r[key] for r in runs.values() if r[key] is not None]
            summary[key] = {
                "mean": fmean(vals) if vals else None,
                "sd": pstdev(vals) if len(vals) > 1 else None,
                "runs_scored": len(vals),
            }
        engines[engine] = {"runs": runs, "summary": summary}
    result = {
        "schema": 1,
        "submission_id": m["submission_id"],
        "protocol": m["protocol"],
        "manifest_sha256": digest(encoded(m)),
        "scope": m["scope"],
        "assistant": m["assistant"],
        "aggregation": AGGREGATION,
        "judges": catalog()["judges"],
        "status": "scored-awaiting-review",
        "engines": engines,
        "evidence_sha256": evidence_hashes,
        "scorer": provenance,
        "publication_blockers": [],
    }
    write_new(work / "result.json", result)
    return result
