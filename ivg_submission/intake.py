"""Small PR requests refer to external data; never put full trajectories into Git history."""

import hashlib
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from .bundle import validate
from .common import SHA, catalog, file_hash, read_json, require, safe_id, write_new

MAX_DOWNLOAD = 2 * 1024**3
HOSTS = {
    "github.com",
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
}


def check_url(url, initial=False):
    p = urlsplit(url)
    require(
        p.scheme == "https"
        and p.hostname in HOSTS
        and not p.username
        and not p.password
        and p.port in (None, 443)
        and not p.fragment,
        "Use a public HTTPS GitHub Release asset URL",
    )
    if initial:
        parts = p.path.split("/")
        require(
            p.hostname == "github.com"
            and len(parts) >= 7
            and parts[3:5] == ["releases", "download"]
            and not p.query,
            "Expected github.com/owner/repo/releases/download/tag/file.zip",
        )
    return url


def check_request(path):
    d = read_json(path)
    require(
        isinstance(d, dict)
        and set(d)
        == {"schema", "submission_id", "protocol", "artifact_url", "sha256", "bytes"},
        "Invalid submission request fields",
    )
    require(
        d["schema"] == 1 and d["protocol"] == catalog()["version"],
        "Unknown request version",
    )
    safe_id(d["submission_id"])
    check_url(d["artifact_url"], initial=True)
    require(SHA.fullmatch(str(d["sha256"])), "Invalid artifact SHA-256")
    require(
        type(d["bytes"]) is int and 0 < d["bytes"] <= MAX_DOWNLOAD,
        "Artifact must be at most 2 GiB",
    )
    return d


def make_request(bundle, url, out):
    report = validate(bundle)
    check_url(url, initial=True)
    require(
        Path(bundle).stat().st_size <= MAX_DOWNLOAD,
        "Use a bundle below 2 GiB for hosted validation",
    )
    d = {
        "schema": 1,
        "submission_id": report["submission_id"],
        "protocol": catalog()["version"],
        "artifact_url": url,
        "sha256": file_hash(bundle),
        "bytes": Path(bundle).stat().st_size,
    }
    write_new(out, d)
    return d


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_request(request_path, out):
    d = check_request(request_path)
    opener = urllib.request.build_opener(
        SafeRedirect(), urllib.request.ProxyHandler({})
    )
    h, n = hashlib.sha256(), 0
    with (
        opener.open(d["artifact_url"], timeout=60) as response,
        Path(out).open("xb") as dest,
    ):
        check_url(response.url)
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            n += len(block)
            require(n <= d["bytes"], "Download exceeds declared size")
            h.update(block)
            dest.write(block)
    require(
        n == d["bytes"] and h.hexdigest() == d["sha256"],
        "Downloaded artifact differs from the submitted hash",
    )
    report = validate(out)
    require(
        report["submission_id"] == d["submission_id"],
        "Artifact submission identity differs from request",
    )
    return report
