"""Package native records; validate data without importing or executing submitter code."""

import hashlib
import re
import stat
import zipfile
from collections import Counter
from pathlib import Path

from .common import (
    Invalid,
    MAX_FILE,
    MAX_JSON,
    MAX_MEMBERS,
    MAX_TOTAL,
    SHA,
    catalog,
    digest,
    encoded,
    expected_cells,
    loads,
    read_json,
    require,
    safe_id,
    safe_path,
)

FAMILIES = ("frontier-api", "streaming-api", "open-nonstreaming", "open-streaming")


def metadata(config, partial=False):
    c = catalog()
    model = config.get("assistant", {})
    require(isinstance(model, dict), "assistant must be an object")
    require(
        set(model)
        == {
            "name",
            "revision",
            "family",
            "inference_code",
            "inference_commit",
            "hardware",
        },
        "Unknown or missing assistant metadata fields (do not include credentials)",
    )
    for key in (
        "name",
        "revision",
        "family",
        "inference_code",
        "inference_commit",
        "hardware",
    ):
        require(
            isinstance(model.get(key), str) and 0 < len(model[key]) <= 500,
            f"Missing or invalid assistant.{key}",
        )
    require(model["family"] in FAMILIES, "Unknown assistant family")
    require(
        re.fullmatch(r"[0-9a-f]{40}", model["inference_commit"]),
        "Use a full inference code commit SHA",
    )
    require(
        model["inference_code"].startswith("https://"),
        "inference_code must be an HTTPS repository URL",
    )
    require(isinstance(config.get("settings"), dict), "Missing per-engine settings")
    for engine, want in c["settings"].items():
        actual = config["settings"].get(engine, {})
        require(isinstance(actual, dict), f"Invalid settings for {engine}")
        require(
            set(actual) == {*want, "code_commit", "assistant_model_id"},
            f"Unknown or missing settings for {engine}",
        )
        require(
            isinstance(actual["assistant_model_id"], str)
            and 0 < len(actual["assistant_model_id"]) <= 300,
            f"{engine}: specify the actual model ID stamped in the native record",
        )
        for key, value in want.items():
            require(actual.get(key) == value, f"{engine}: {key} must be {value}")
        require(
            re.fullmatch(r"[0-9a-f]{40}", str(actual.get("code_commit", ""))),
            f"{engine}: provide a full code_commit SHA",
        )
    return {
        "schema": 1,
        "submission_id": safe_id(config.get("submission_id")),
        "protocol": c["version"],
        "catalog_sha256": digest(encoded(c)),
        "scope": "preview" if partial else "full",
        "assistant": model,
        "settings": config["settings"],
        "episodes": [],
        "files": {},
    }


def check_record(engine, case, persona, record, settings):
    require(
        isinstance(record, dict) and "error" not in record,
        "Failed or invalid episode record",
    )
    require(
        record.get("assistant_mode") != "native" and "native_assistant" not in record,
        "Native streaming records do not belong to the pulled-interface track",
    )
    actual_case = (
        record.get("episode") if engine == "screensim" else record.get("scenario")
    )
    require(actual_case == case, f"{engine}/{case}: record case differs from manifest")
    actual_persona = record.get("persona")
    if engine == "cooksim" and isinstance(actual_persona, str):
        actual_persona = actual_persona.removeprefix("pv2_")
    require(
        actual_persona == persona,
        f"{engine}/{case}: record persona differs from manifest",
    )
    fields = {
        "cooksim": ("conversation", "moves", "injections"),
        "vhhome": ("conversation", "action_log", "phases"),
        "screensim": ("transcript", "timeline", "beats"),
    }[engine]
    for field in fields:
        require(
            isinstance(record.get(field), list),
            f"{engine}/{case}: missing native {field} evidence",
        )
    ticks = record.get("final_tick" if engine == "screensim" else "ticks")
    require(
        type(ticks) is int and ticks >= 0, f"{engine}/{case}: invalid episode duration"
    )
    if engine == "cooksim":
        assistant = (record.get("assistant_backend") or {}).get("model")
        human = (record.get("cook_backend") or {}).get("model")
        require(
            record.get("obs_mode") == "frames", "CookSim must use visual observations"
        )
        latency = record.get("assist_latency") or []
    elif engine == "vhhome":
        assistant = (record.get("assistant") or {}).get("model")
        human = record.get("person_model") or record.get("human_model")
        latency = record.get("assistant_latency") or []
    else:
        assistant = record.get("contestant")
        human = record.get("human_model")
        latency = []
    require(
        assistant == settings["assistant_model_id"],
        f"{engine}/{case}: recorded assistant differs from the declared model",
    )
    if human is not None:
        require(
            human == settings["human_model"],
            f"{engine}/{case}: incompatible recorded user model",
        )
    for entry in latency:
        require(
            isinstance(entry, dict) and entry.get("ticks") == 2,
            f"{engine}/{case}: recorded delay is not fixed at two ticks",
        )
    # Some historical records do not stamp all configuration. Absence is NOT evidence of compliance.
    warnings = []
    if human is None:
        warnings.append("user model is declared, not stamped in the native record")
    if engine != "screensim" and not latency:
        warnings.append("no recorded latency calls; settings require maintainer review")
    return warnings


def _local(path):
    p = Path(path)
    require(
        p.is_file() and not p.is_symlink(),
        "Input must be a regular file, not a symbolic link",
    )
    require(p.stat().st_size <= MAX_FILE, "Input file exceeds size limit")
    return p


def source_records(source):
    """Yield explicit selected native episodes, not logs, cached scores, or backup files."""
    engine, run = source.get("engine"), source.get("run")
    require(
        engine in catalog()["engines"] and type(run) is int and run in (1, 2, 3),
        "Invalid source engine/run",
    )
    root = Path(source["path"]).resolve()
    tag = safe_id(source.get("tag"))
    if engine == "screensim":
        require(
            root.name.endswith(".records.jsonl"),
            "ScreenSim source must be one .records.jsonl file",
        )
        with _local(root).open("rb") as f:
            while True:
                line = f.readline(MAX_JSON + 1)
                if not line:
                    break
                require(len(line) <= MAX_JSON, "Oversized ScreenSim record")
                if not line.strip():
                    continue
                d = loads(line)
                require(isinstance(d, dict), "ScreenSim line must be an object")
                if d.get("contestant") == tag:
                    yield d.get("episode"), d.get("persona"), d, None
    else:
        paths = (
            root.glob(f"frames/*/{tag}/b3_*.json")
            if engine == "cooksim"
            else root.glob(f"{tag}/{tag}__*.json")
        )
        for p in sorted(paths):
            if p.name.endswith((".states.json", ".lane.json")):
                continue
            d = read_json(_local(p))
            require(isinstance(d, dict), "Episode must be an object")
            persona = d.get("persona")
            if engine == "cooksim" and isinstance(persona, str):
                persona = persona.removeprefix("pv2_")
            yield d.get("scenario"), persona, d, p


def prepare(config_path, out, partial=False):
    config = read_json(config_path)
    manifest = metadata(config, partial)
    expected = expected_cells()
    seen, members, total = set(), set(), 0
    sources = config.get("sources")
    require(isinstance(sources, list) and sources, "Provide native run sources")
    require(len(sources) <= 100, "Too many sources")
    dest = Path(out)
    require(not dest.exists(), "Output exists; choose a new bundle name")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation; on failure preserve the incomplete file for inspection, never overwrite a good bundle.
    with zipfile.ZipFile(
        dest, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3
    ) as z:

        def add(name, raw=None, path=None):
            nonlocal total
            require(name not in members, "Duplicate artifact name")
            members.add(name)
            size = len(raw) if raw is not None else _local(path).stat().st_size
            total += size
            require(
                size <= MAX_FILE and total <= MAX_TOTAL and len(members) < MAX_MEMBERS,
                "Bundle exceeds limits",
            )
            h = hashlib.sha256()
            with z.open(name, "w", force_zip64=True) as target:
                if raw is not None:
                    target.write(raw)
                    h.update(raw)
                else:
                    with path.open("rb") as f:
                        for block in iter(lambda: f.read(1024 * 1024), b""):
                            target.write(block)
                            h.update(block)
            manifest["files"][name] = {"bytes": size, "sha256": h.hexdigest()}

        for source in sources:
            engine, run = source["engine"], source["run"]
            n = 0
            for case, persona, record, path in source_records(source):
                safe_id(case)
                safe_id(persona)
                key = (engine, run, case, persona)
                require(key in expected, "Unknown case/persona combination")
                require(
                    key not in seen,
                    "Duplicate case/persona in a run; do not mix retries",
                )
                seen.add(key)
                n += 1
                check_record(
                    engine, case, persona, record, manifest["settings"][engine]
                )
                prefix = f"records/{engine}/{run}/{case}__{persona}"
                entry = {
                    "engine": engine,
                    "run": run,
                    "case": case,
                    "persona": persona,
                    "record": prefix + ".json",
                    "evidence": {},
                }
                add(entry["record"], raw=encoded(record))
                if path is not None:
                    for kind, suffix in (
                        ("states", ".states.jsonl"),
                        ("video", ".mp4"),
                    ):
                        evidence = path.with_suffix(suffix)
                        if evidence.exists():
                            name = prefix + suffix
                            add(name, path=evidence)
                            entry["evidence"][kind] = name
                manifest["episodes"].append(entry)
            require(
                n,
                f"No records selected for {engine} run {run}; check source path and tag",
            )
        require(
            partial or seen == expected,
            f"Incomplete suite: {len(seen)} of {len(expected)} episodes",
        )
        manifest["episodes"].sort(
            key=lambda x: (x["engine"], x["run"], x["case"], x["persona"])
        )
        z.writestr("manifest.json", encoded(manifest))
    return validate(dest, allow_preview=partial)


def validate(path, allow_preview=False):
    """Check ZIP members BEFORE reading or extracting. No network requests, simulator imports, or shell calls."""
    try:
        return _validate(path, allow_preview)
    except (
        zipfile.BadZipFile,
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,
    ) as e:
        raise Invalid("Malformed submission archive or manifest") from e


def _validate(path, allow_preview):
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        require(0 < len(infos) <= MAX_MEMBERS, "Invalid archive member count")
        names = [x.filename for x in infos]
        require(len(names) == len(set(names)), "Duplicate ZIP member")
        require(sum(x.file_size for x in infos) <= MAX_TOTAL, "Archive too large")
        for info in infos:
            safe_path(info.filename)
            mode = info.external_attr >> 16
            require(
                not info.is_dir() and stat.S_IFMT(mode) in (0, stat.S_IFREG),
                "Non-regular archive member",
            )
            require(
                info.file_size <= MAX_FILE and not info.flag_bits & 1,
                "Oversized or encrypted member",
            )
            require(
                info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                "Unsupported compression",
            )
        require("manifest.json" in names, "Missing manifest.json")
        require(z.getinfo("manifest.json").file_size <= MAX_JSON, "Oversized manifest")
        m = loads(z.read("manifest.json"))
        require(isinstance(m, dict) and m.get("schema") == 1, "Unknown manifest schema")
        require(m.get("scope") in ("preview", "full"), "Unknown submission scope")
        require(
            m["scope"] == "full" or allow_preview,
            "Preview bundles cannot enter the leaderboard",
        )
        checked = metadata(m, m["scope"] == "preview")
        for field in ("protocol", "catalog_sha256"):
            require(m.get(field) == checked[field], f"Incompatible {field}")
        require(isinstance(m.get("files"), dict), "Missing file manifest")
        require(
            set(names) == {"manifest.json", *m["files"]}, "Undeclared or missing file"
        )
        for name, info in m["files"].items():
            require(
                isinstance(info, dict)
                and type(info.get("bytes")) is int
                and SHA.fullmatch(str(info.get("sha256", ""))),
                "Invalid file descriptor",
            )
            require(z.getinfo(name).file_size == info["bytes"], "File length mismatch")
            h = hashlib.sha256()
            count = 0
            with z.open(name) as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
                    count += len(block)
                    require(count <= MAX_FILE, "Expanded file too large")
            require(
                count == info["bytes"] and h.hexdigest() == info["sha256"],
                "File checksum mismatch",
            )
        require(isinstance(m.get("episodes"), list) and m["episodes"], "No episodes")
        expected, seen, referenced, warnings = expected_cells(), set(), set(), Counter()
        for e in m["episodes"]:
            require(isinstance(e, dict), "Invalid episode entry")
            key = (e.get("engine"), e.get("run"), e.get("case"), e.get("persona"))
            require(
                type(e.get("run")) is int and key in expected,
                "Unexpected episode membership",
            )
            require(key not in seen, "Duplicate episode")
            seen.add(key)
            prefix = f"records/{key[0]}/{key[1]}/{key[2]}__{key[3]}"
            require(e.get("record") == prefix + ".json", "Noncanonical record path")
            require(
                z.getinfo(e["record"]).file_size <= MAX_JSON, "Oversized episode record"
            )
            record = loads(z.read(e["record"]))
            warnings.update(
                check_record(key[0], key[2], key[3], record, m["settings"][key[0]])
            )
            referenced.add(e["record"])
            require(isinstance(e.get("evidence"), dict), "Invalid evidence descriptor")
            for kind, name in e["evidence"].items():
                suffix = {"states": ".states.jsonl", "video": ".mp4"}.get(kind)
                require(
                    suffix and name == prefix + suffix and name in m["files"],
                    "Unsupported evidence file",
                )
                referenced.add(name)
            if key[0] in ("cooksim", "vhhome") and "video" not in e["evidence"]:
                warnings[
                    "visual evidence absent; visual grading requires maintainer review"
                ] += 1
        require(referenced == set(m["files"]), "Unreferenced files are not accepted")
        require(
            m["scope"] == "preview" or seen == expected,
            f"Incomplete suite: {len(seen)}/{len(expected)}",
        )
        return {
            "valid": True,
            "submission_id": m["submission_id"],
            "scope": m["scope"],
            "episodes": len(seen),
            "expected": len(expected),
            "manifest_sha256": digest(encoded(m)),
            "counts": dict(Counter(f"{e}/run-{r}" for e, r, _, _ in seen)),
            "review_notes": dict(warnings),
            "status": "validated-not-scored",
        }


def manifest(path):
    with zipfile.ZipFile(path) as z:
        return loads(z.read("manifest.json"))
