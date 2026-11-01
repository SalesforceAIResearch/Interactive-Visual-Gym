"""Bounded, strict data IO. Never load executable formats from a submission."""

import hashlib
import json
import re
from importlib.resources import files
from pathlib import Path, PurePosixPath

MAX_JSON = 64 * 1024 * 1024
MAX_FILE = 2 * 1024**3
MAX_TOTAL = 24 * 1024**3
MAX_MEMBERS = 4000
ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,159}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


class Invalid(ValueError):
    """Invalid submission; safe to display without dumping source records."""


def require(condition, message):
    if not condition:
        raise Invalid(message)


def safe_id(value):
    require(isinstance(value, str) and ID.fullmatch(value), "Invalid identifier")
    return value


def safe_path(value):
    require(
        isinstance(value, str) and value and "\\" not in value and "\x00" not in value,
        "Invalid archive path",
    )
    p = PurePosixPath(value)
    require(
        not p.is_absolute()
        and all(x not in (".", "..", "") for x in value.split("/"))
        and ":" not in value,
        "Unsafe archive path",
    )
    return p


def _pairs(items):
    d = {}
    for k, v in items:
        require(k not in d, "Duplicate JSON object key")
        d[k] = v
    return d


def loads(raw):
    require(len(raw) <= MAX_JSON, "JSON record exceeds size limit")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                Invalid("Non-finite JSON number")
            ),
        )
    except (ValueError, UnicodeError, RecursionError) as e:
        raise Invalid("Invalid JSON") from e


def read_json(path):
    with Path(path).open("rb") as f:
        return loads(f.read(MAX_JSON + 1))


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        + "\n"
    ).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def catalog():
    return loads(files("ivg_submission").joinpath("catalog.json").read_bytes())


def write_new(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        f.write(encoded(data))


def expected_cells(c=None):
    c = c or catalog()
    return {
        (e, run, x["case"], x["persona"])
        for e, rows in c["engines"].items()
        for run in c["runs"]
        for x in rows
    }
