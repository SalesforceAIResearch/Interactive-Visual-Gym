"""Headless capture: HTML -> PNG bytes via Playwright, at exactly SCREEN_W×H.

A single browser/page is reused across frames for speed. `Capturer` is a
context manager; `snap(dev)` renders the device's current screen to PNG bytes.
"""
from __future__ import annotations

from ..core.model import SCREEN_H, SCREEN_W
from .renderer import render


class Capturer:
    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self._pw = None
        self._browser = None
        self._page = None

    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(args=["--no-sandbox",
                                                        "--force-color-profile=srgb"])
        self._page = self._browser.new_page(
            viewport={"width": SCREEN_W, "height": SCREEN_H},
            device_scale_factor=self.scale)
        return self

    def snap(self, dev, fx_override=None) -> bytes:
        html = render(dev, fx_override=fx_override)
        self._page.set_content(html, wait_until="load")
        return self._page.screenshot(type="png")

    def snap_html(self, html: str) -> bytes:
        self._page.set_content(html, wait_until="load")
        return self._page.screenshot(type="png")

    def __exit__(self, *a):
        try:
            self._browser.close()
        finally:
            self._pw.stop()


def render_to_file(dev, path: str, scale: float = 2.0):
    with Capturer(scale=scale) as cap:
        png = cap.snap(dev)
    with open(path, "wb") as f:
        f.write(png)
    return path
