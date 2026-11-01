import copy
import stat
import sys
import tempfile
import unittest
from types import SimpleNamespace
import zipfile
from pathlib import Path
from unittest.mock import patch

from ivg_submission.bundle import check_record, metadata, prepare, validate
from ivg_submission.common import (
    Invalid,
    catalog,
    digest,
    encoded,
    expected_cells,
    file_hash,
    loads,
    read_json,
)
from ivg_submission.grading import (
    AGGREGATION,
    CATS,
    check_workspace,
    collect,
    commands,
    judge_path,
    native_path,
    round_score,
    stage,
)
from ivg_submission.intake import check_url, make_request, check_request
from ivg_submission.publishing import approve, audit_plan, build_ledger


def config():
    return {
        "submission_id": "test-model-v1",
        "assistant": {
            "name": "Test assistant",
            "revision": "v1",
            "family": "open-nonstreaming",
            "inference_code": "https://github.com/example/model",
            "inference_commit": "a" * 40,
            "hardware": "test fixture",
        },
        "settings": {
            e: {**s, "code_commit": "b" * 40, "assistant_model_id": "model"}
            for e, s in catalog()["settings"].items()
        },
    }


def record(engine, case, persona):
    if engine == "screensim":
        return {
            "episode": case,
            "persona": persona,
            "final_tick": 20,
            "timeline": [],
            "transcript": [],
            "beats": [],
            "contestant": "model",
            "human_model": catalog()["settings"][engine]["human_model"],
        }
    common = {
        "scenario": case,
        "persona": persona,
        "ticks": 20,
        "conversation": [],
        "outcome": "timeout",
    }
    if engine == "cooksim":
        common.update(
            moves=[],
            injections=[],
            obs_mode="frames",
            cook_backend={"model": "gemini-3.7-flash"},
            assist_latency=[{"ticks": 2}],
            assistant_backend={"model": "model"},
        )
    else:
        common.update(
            action_log=[],
            phases=[],
            assistant_latency=[{"ticks": 2}],
            assistant={"model": "model"},
        )
    return common


def bundle_data(full=True):
    m = metadata(config(), not full)
    members = {}
    cells = sorted(expected_cells())
    for engine, run, case, persona in cells if full else cells[:1]:
        name = f"records/{engine}/{run}/{case}__{persona}.json"
        raw = encoded(record(engine, case, persona))
        members[name] = raw
        m["files"][name] = {"bytes": len(raw), "sha256": digest(raw)}
        m["episodes"].append(
            {
                "engine": engine,
                "run": run,
                "case": case,
                "persona": persona,
                "record": name,
                "evidence": {},
            }
        )
    members["manifest.json"] = encoded(m)
    return m, members


def write_zip(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, raw in members.items():
            z.writestr(name, raw)


class SubmissionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / "submission.zip"

    def check_bad(self, mutate, full=True):
        m, members = bundle_data(full)
        mutate(m, members)
        members["manifest.json"] = encoded(m)
        write_zip(self.bundle, members)
        with self.assertRaises(Invalid):
            validate(self.bundle)

    def test_exact_catalog(self):
        self.assertEqual(len(expected_cells()), 945)
        self.assertEqual(
            {e: len(v) for e, v in catalog()["engines"].items()},
            {"cooksim": 150, "vhhome": 75, "screensim": 90},
        )

    def test_valid_full_bundle(self):
        _, members = bundle_data()
        write_zip(self.bundle, members)
        report = validate(self.bundle)
        self.assertEqual(report["episodes"], 945)
        self.assertEqual(report["status"], "validated-not-scored")

    def test_missing_case(self):
        self.check_bad(lambda m, _: m["episodes"].pop())

    def test_duplicate_case(self):
        self.check_bad(lambda m, _: m["episodes"].append(m["episodes"][0]))

    def test_hash_mismatch(self):
        self.check_bad(
            lambda m, files: files.__setitem__(m["episodes"][0]["record"], b"{}")
        )

    def test_path_traversal(self):
        self.check_bad(lambda m, files: files.__setitem__("../evil.py", b"pass"))

    def test_absolute_path(self):
        self.check_bad(lambda m, files: files.__setitem__("/tmp/evil", b"pass"))

    def test_backslash(self):
        self.check_bad(lambda m, files: files.__setitem__("..\\evil", b"pass"))

    def test_undeclared_program(self):
        self.check_bad(
            lambda m, files: files.__setitem__("program.py", b"raise Exception()")
        )

    def test_changed_protocol(self):
        self.check_bad(lambda m, _: m.__setitem__("protocol", "another-version"))

    def test_no_credential_metadata(self):
        self.check_bad(
            lambda m, _: m["assistant"].__setitem__("api_key", "not-a-real-key")
        )

    def test_preview_not_publishable(self):
        _, members = bundle_data(False)
        write_zip(self.bundle, members)
        with self.assertRaises(Invalid):
            validate(self.bundle)
        self.assertEqual(validate(self.bundle, allow_preview=True)["episodes"], 1)

    def test_symlink(self):
        _, members = bundle_data(False)
        write_zip(self.bundle, members)
        with zipfile.ZipFile(self.bundle, "a") as z:
            i = zipfile.ZipInfo("link")
            i.create_system = 3
            i.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(i, "/etc/passwd")
        with self.assertRaises(Invalid):
            validate(self.bundle, allow_preview=True)

    def test_duplicate_json_key(self):
        with self.assertRaises(Invalid):
            loads(b'{"a":1,"a":2}')

    def test_nonfinite_json(self):
        with self.assertRaises(Invalid):
            loads(b'{"score":NaN}')

    def test_changed_record_identity_even_with_correct_hash(self):
        def mutate(m, files):
            name = m["episodes"][0]["record"]
            r = loads(files[name])
            r["scenario"] = "fake-case"
            files[name] = encoded(r)
            m["files"][name] = {
                "bytes": len(files[name]),
                "sha256": digest(files[name]),
            }

        self.check_bad(mutate)

    def test_mismatched_latency(self):
        r = record("cooksim", "case", "baseline")
        r["assist_latency"][0]["ticks"] = 7
        with self.assertRaises(Invalid):
            check_record(
                "cooksim", "case", "baseline", r, config()["settings"]["cooksim"]
            )

    def test_wrong_user_model(self):
        r = record("cooksim", "case", "baseline")
        r["cook_backend"]["model"] = "different"
        with self.assertRaises(Invalid):
            check_record(
                "cooksim", "case", "baseline", r, config()["settings"]["cooksim"]
            )

    def test_wrong_assistant(self):
        r = record("cooksim", "case", "baseline")
        r["assistant_backend"]["model"] = "another-model"
        with self.assertRaises(Invalid):
            check_record(
                "cooksim", "case", "baseline", r, config()["settings"]["cooksim"]
            )

    def test_native_interface_rejected(self):
        r = record("cooksim", "case", "baseline")
        r["assistant_mode"] = "native"
        with self.assertRaises(Invalid):
            check_record(
                "cooksim", "case", "baseline", r, config()["settings"]["cooksim"]
            )

    def test_prepare_native_partial(self):
        c = config()
        case = catalog()["engines"]["cooksim"][0]
        p = (
            self.root
            / "native/frames"
            / case["persona"]
            / "model"
            / (case["case"] + ".json")
        )
        p.parent.mkdir(parents=True)
        p.write_bytes(encoded(record("cooksim", case["case"], case["persona"])))
        c["sources"] = [
            {
                "engine": "cooksim",
                "run": 1,
                "tag": "model",
                "path": str(self.root / "native"),
            }
        ]
        cp = self.root / "config.json"
        cp.write_bytes(encoded(c))
        report = prepare(cp, self.bundle, partial=True)
        self.assertEqual(report["episodes"], 1)
        with self.assertRaises(Invalid):
            prepare(cp, self.bundle, partial=True)

    def test_stage_clean_and_tamper_detection(self):
        _, members = bundle_data(False)
        write_zip(self.bundle, members)
        work = self.root / "work"
        stage(self.bundle, work, allow_preview=True)
        job, m = check_workspace(work)
        self.assertEqual(len(job["commands"]), 4)
        self.assertFalse(list(work.rglob("*.py")))
        inp = work / next(iter(job["inputs"]))
        inp.write_text("{}")
        with self.assertRaises(Invalid):
            check_workspace(work)

    def test_command_allowlist(self):
        m, _ = bundle_data()
        plan = commands(self.root, m)
        self.assertEqual(len(plan), 10)
        self.assertTrue(
            all(
                x["module"].endswith(
                    (
                        "rubric",
                        "sq_v4_score",
                        "regrade_qa",
                        "regrade_proactive_v2",
                        "eval_v2_sweep",
                    )
                )
                for x in plan
            )
        )

    def test_audit_sample(self):
        _, members = bundle_data()
        write_zip(self.bundle, members)
        plan = audit_plan(
            self.bundle, self.root / "audit.json", per_engine=3, seed="test"
        )
        self.assertEqual(len(plan["selected"]), 9)
        self.assertEqual(plan["status"], "planned-not-executed")
        self.assertEqual(len({(r["engine"], r["case"]) for r in plan["selected"]}), 9)

    def test_public_release_request(self):
        _, members = bundle_data()
        write_zip(self.bundle, members)
        p = self.root / "request.json"
        make_request(
            self.bundle,
            "https://github.com/org/data/releases/download/v1/submission.zip",
            p,
        )
        self.assertEqual(check_request(p)["sha256"], file_hash(self.bundle))

    def test_download_url_restrictions(self):
        for url in (
            "http://github.com/a/b/releases/download/v1/a.zip",
            "https://localhost/a",
            "file:///etc/passwd",
            "https://github.com.evil.test/a",
            "https://user:password@github.com/a",
            "https://github.com:444/a",
        ):
            with self.subTest(url=url), self.assertRaises(Invalid):
                check_url(url, initial=True)

    def test_preview_approval_rejected(self):
        p = self.root / "result.json"
        p.write_bytes(encoded({"status": "scored-awaiting-review", "scope": "preview"}))
        with self.assertRaises(Invalid):
            approve(p, self.root / "approval.json", "maintainer", "Reviewed")

    def test_empty_ledger_does_not_invent_results(self):
        out = self.root / "feed.json"
        build_ledger(self.root / "reviewed", out)
        self.assertEqual(read_json(out)["entries"], [])

    def test_approval_binds_exact_result_bytes(self):
        d = self.root / "reviewed/test-model-v1"
        d.mkdir(parents=True)
        r = {
            "schema": 1,
            "submission_id": "test-model-v1",
            "scope": "full",
            "status": "scored-awaiting-review",
            "protocol": catalog()["version"],
            "aggregation": AGGREGATION,
            "judges": catalog()["judges"],
            "assistant": config()["assistant"],
            "manifest_sha256": "c" * 64,
            "publication_blockers": [],
            "engines": {
                e: {"runs": {str(n): {"episodes": len(cells)} for n in (1, 2, 3)}}
                for e, cells in catalog()["engines"].items()
            },
        }
        (d / "result.json").write_bytes(encoded(r))
        approve(
            d / "result.json",
            d / "approval.json",
            "test-maintainer",
            "Synthetic test fixture only",
        )
        build_ledger(d.parent, self.root / "feed.json")
        self.assertEqual(len(read_json(self.root / "feed.json")["entries"]), 1)
        r["assistant"]["name"] = "changed"
        (d / "result.json").write_bytes(encoded(r))
        with self.assertRaises(Invalid):
            build_ledger(d.parent, self.root / "tampered.json")

    def test_collect_strict_judges_and_run_mean(self):
        # Synthetic one-episode preview, no native imports, no model/provider calls.
        m, members = bundle_data(False)
        write_zip(self.bundle, members)
        work = self.root / "work"
        stage(self.bundle, work, allow_preview=True)
        (work / "scorer.json").write_bytes(
            encoded({"source_sha256": "fixture", "python": "fixture"})
        )
        e = m["episodes"][0]
        p = native_path(work, e)
        key = str(p.relative_to(work / "data/cooksim").with_suffix(""))
        metric = {
            "v3": {"cap_ticks": 100, "in_time_success": 1, "tp": 1, "fp": 0},
            "evidence": {"errors_fired": 1},
        }
        for name, obj in (
            ("eval_v2.json", {"episodes": {key: metric}}),
            ("qa_regrade.json", {}),
            ("proactive_v2.json", {}),
        ):
            (work / "data/cooksim" / name).write_bytes(encoded(obj))
        jp = judge_path(work, e)
        jp.parent.mkdir(parents=True)
        row = JudgeTest().row()
        row["round"] = 1
        jp.write_bytes(encoded({"rounds": [row]}))
        suite = self.root / "suite.json"
        suite.write_bytes(
            encoded({"cases": [{"name": e["case"], "subs": [{"family": "error"}]}]})
        )
        module = SimpleNamespace(rounds_of=lambda r: [{"round": 1}])
        with (
            patch("ivg_submission.grading.scorer_fingerprint", return_value="fixture"),
            patch("ivg_submission.grading.native_module", return_value=module),
            patch.dict(
                sys.modules, {"cooksim_bench.paths": SimpleNamespace(SUITE=str(suite))}
            ),
        ):
            result = collect(work)
        self.assertEqual(
            result["engines"]["cooksim"]["summary"]["interaction_quality"]["mean"], 1
        )
        self.assertIsNone(
            result["engines"]["cooksim"]["summary"]["qa_accuracy"]["mean"]
        )
        self.assertEqual(result["scope"], "preview")
        with self.assertRaises(Invalid):
            approve(
                work / "result.json", self.root / "approval.json", "maintainer", "No"
            )


class JudgeTest(unittest.TestCase):
    def row(self):
        items = {k: {"pass": True} for ids in CATS.values() for k in ids}
        return {
            "judges": {j: {"items": copy.deepcopy(items)} for j in ("gemini", "gpt")}
        }

    def test_two_judge_mean_not_merged_and(self):
        row = self.row()
        row["judges"]["gpt"]["items"]["T1"]["pass"] = False
        result = round_score(row)
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["categories"]["factual_grounding"], 0.5)

    def test_category_all_applicable_must_pass(self):
        row = self.row()
        row["judges"]["gpt"]["items"]["H2"]["pass"] = False
        self.assertEqual(round_score(row)["score"], 0.9)

    def test_missing_judge_blocks(self):
        row = self.row()
        row["judges"].pop("gpt")
        with self.assertRaises(Invalid):
            round_score(row)

    def test_failed_judge_blocks(self):
        row = self.row()
        row["judges"]["gpt"] = {"error": "timeout"}
        with self.assertRaises(Invalid):
            round_score(row)

    def test_missing_item_blocks(self):
        row = self.row()
        row["judges"]["gpt"]["items"].pop("H3")
        with self.assertRaises(Invalid):
            round_score(row)

    def test_not_applicable_is_not_zero(self):
        row = self.row()
        for judge in row["judges"].values():
            for item in judge["items"].values():
                item["pass"] = None
        self.assertIsNone(round_score(row)["score"])

    def test_string_false_is_invalid(self):
        row = self.row()
        row["judges"]["gpt"]["items"]["T1"]["pass"] = "false"
        with self.assertRaises(Invalid):
            round_score(row)


if __name__ == "__main__":
    unittest.main()
