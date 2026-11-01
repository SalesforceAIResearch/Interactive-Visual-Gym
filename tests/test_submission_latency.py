"""Exercise the exact runner expressions without Unity or API calls."""

import ast
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "vhhome/vhhome_bench/episode.py"


class FixedDelayTests(unittest.TestCase):
    def check_mode(self, mode, seconds, expected):
        tree = ast.parse(PATH.read_text())
        wanted = {
            "_LAT_ENV",
            "ASSIST_LATENCY",
            "ASSIST_FIXED_TICKS",
            "ASSIST_TICK_S",
            "ASSIST_MAX_TICKS",
        }
        assigns = [
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets)
        ]
        native = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_nat_ticks"
        )
        ns = {"os": os, "math": math}
        with patch.dict(
            os.environ,
            {"ASSIST_LATENCY": mode, "ASSIST_TICK_S": "1.2", "ASSIST_MAX_TICKS": "15"},
        ):
            exec(
                compile(
                    ast.Module(body=assigns + [native], type_ignores=[]),
                    str(PATH),
                    "exec",
                ),
                ns,
            )
        self.assertEqual(ns["_nat_ticks"](seconds), expected)
        polled = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ("_y_raw", "_y") for t in n.targets
            ):
                # The override assignment is nested in the following If, not executed twice here.
                if len(n.targets) == 1:
                    polled.append(n)
            if (
                isinstance(n, ast.If)
                and ast.unparse(n.test) == "ASSIST_FIXED_TICKS is not None"
                and any(isinstance(x, ast.Assign) for x in n.body)
            ):
                polled.append(n)
        polled.sort(key=lambda n: n.lineno)
        ns["_x_charged"] = seconds
        exec(compile(ast.Module(body=polled, type_ignores=[]), str(PATH), "exec"), ns)
        self.assertEqual(ns["_y"], expected)

    def test_fixed_two_ignores_wall_latency(self):
        for seconds in (0.01, 1.2, 4.8, 100):
            self.check_mode("2", seconds, 2)

    def test_measured_default_unchanged(self):
        for seconds, expected in ((0.1, 1), (1.2, 1), (4.8, 4), (100, 15)):
            self.check_mode("1", seconds, expected)
            self.check_mode("measured", seconds, expected)

    def test_zero_unchanged(self):
        self.check_mode("0", 100, 0)


if __name__ == "__main__":
    unittest.main()
