"""A stale source pin must fail CI before it blocks maintainer grading."""

import hashlib
from pathlib import Path
import unittest

from ivg_submission.common import catalog


class CatalogTests(unittest.TestCase):
    def test_frozen_sources_match_shipped_release(self):
        root = Path(__file__).resolve().parents[1]
        for relative, expected in catalog()["source_sha256"].items():
            with self.subTest(source=relative):
                self.assertEqual(
                    hashlib.sha256((root / relative).read_bytes()).hexdigest(),
                    expected,
                    "Rebuild the catalog and version the protocol when frozen sources change",
                )


if __name__ == "__main__":
    unittest.main()
