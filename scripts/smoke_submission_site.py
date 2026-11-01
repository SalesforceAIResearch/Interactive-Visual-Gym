"""Browser checks for the static leaderboard; synthetic data is intercepted, never published."""

import argparse
import functools
import http.server
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Handler, directory=str(ROOT / "leaderboard"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route(
                "**/feed.json", lambda route: route.fulfill(json={"entries": []})
            )
            url = f"http://127.0.0.1:{server.server_port}"
            page.goto(url)
            page.get_by_text(
                "No reviewed results have been published for this protocol."
            ).wait_for()
            page.screenshot(path=str(a.out / "leaderboard-empty.png"), full_page=True)
            page.unroute("**/feed.json")
            example = {
                "id": "browser-test",
                "assistant": {
                    "name": "Test assistant <img src=x onerror=alert(1)>",
                    "family": "frontier-api",
                },
                "protocol": "test-only",
                "aggregation": "test-only",
                "verification": "officially-scored",
                "engines": {},
            }
            for e, n in (("cooksim", 150), ("vhhome", 75), ("screensim", 90)):
                example["engines"][e] = {
                    "summary": {
                        k: {"mean": v, "sd": 0.01}
                        for k, v in (
                            ("success", 0.4),
                            ("interaction_quality", 0.6),
                            ("detection_recall", 0.5),
                            ("qa_accuracy", None),
                        )
                    },
                    "runs": {
                        str(r): {
                            "episodes": n,
                            "errors_scheduled": 10,
                            "errors_triggered": 8,
                            "rounds_reached": 12,
                            "rounds_scored": 12,
                            "qa_coverage": None,
                        }
                        for r in (1, 2, 3)
                    },
                }
            page.route(
                "**/feed.json", lambda route: route.fulfill(json={"entries": [example]})
            )
            page.reload()
            page.locator("#rows button").wait_for()
            assert page.locator("#rows img").count() == 0
            assert "<img" in page.locator("#rows button").inner_text()
            page.locator("#rows button").click()
            assert page.locator("#details tr").count() == 4
            page.get_by_role("button", name="ScreenSim", exact=True).click()
            assert (
                page.get_by_role("button", name="ScreenSim", exact=True).get_attribute(
                    "aria-pressed"
                )
                == "true"
            )
            page.locator("#rows button").click()
            assert "90" in page.locator("#details").inner_text()
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            )
            page.screenshot(
                path=str(a.out / "leaderboard-mobile-test.png"), full_page=True
            )
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    print(
        "PASS empty state, reviewed rows, engine tabs, coverage expansion, HTML escaping, mobile overflow, JS errors"
    )


if __name__ == "__main__":
    main()
