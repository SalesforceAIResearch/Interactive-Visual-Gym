"""Timing modes must reach the runners without changing their delay calculations."""
import ast
import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'cooksim'), str(ROOT / 'vhhome')]


class TimingCLITests(unittest.TestCase):
    def test_cooksim_fixed_and_measured_override_exported_setting(self):
        mod = importlib.import_module('cooksim_bench.cli.benchmark')
        for mode in ['2', 'measured']:
            observed = []
            with patch.dict(os.environ, {'COOKSIM_ASSIST_LATENCY': '0'}), \
                 patch.object(mod, 'require_renderer'), \
                 patch.object(mod, 'assistant_flags', return_value=([], 'example')), \
                 patch.object(mod.runpy, 'run_module', side_effect=lambda *a, **k: observed.append(os.environ['COOKSIM_ASSIST_LATENCY'])), \
                 patch.object(sys, 'argv', ['cooksim-benchmark']):
                mod.main(['--assistant', 'gemini:example', '--out', '/tmp/ivg-cli-test', '--latency', mode])
            self.assertEqual(observed, [mode])

    def test_vhhome_benchmark_passes_delay_to_episode_subprocess(self):
        mod = importlib.import_module('vhhome_bench.cli.benchmark')
        package = importlib.import_module('vhhome_bench')
        for mode in ['2', 'measured']:
            seen = []
            entry = {'tag': 'example', 'kind': 'api', 'env': {}, 'needs': []}
            def record(e, cell, port, root, log):
                seen.append(e['env']['ASSIST_LATENCY'])
                log.append({'ok': True})
            lane = types.SimpleNamespace(load_cells=lambda *a: [{}], run_cell=record)
            with tempfile.TemporaryDirectory() as out, \
                 patch.object(sys, 'argv', ['vhhome-benchmark', '--assistant', 'example', '--out', out, '--latency', mode]), \
                 patch.object(mod, 'assistant_entry', return_value=entry), \
                 patch.object(mod, 'require_keys'), \
                 patch.object(package, 'lane', lane, create=True):
                self.assertEqual(mod.main(), 0)
            self.assertEqual(seen, [mode])

    def test_vhhome_single_episode_sets_delay_before_import(self):
        mod = importlib.import_module('vhhome_bench.cli.episode')
        package = importlib.import_module('vhhome_bench')
        for mode in ['2', 'measured']:
            seen = []
            def run(*args):
                seen.append(os.environ['ASSIST_LATENCY'])
                return {'success': True}
            entry = {'tag': 'example', 'kind': 'api', 'env': {}, 'needs': []}
            with tempfile.TemporaryDirectory() as out, \
                 patch.dict(os.environ, {'ASSIST_LATENCY': '0'}), \
                 patch.object(sys, 'argv', ['vhhome-episode', '--case', 'example', '--out', out, '--latency', mode]), \
                 patch.object(mod, 'assistant_entry', return_value=entry), \
                 patch.object(mod, 'require_keys'), \
                 patch.object(package, 'lane', types.SimpleNamespace(BASE_ENV={}), create=True), \
                 patch.object(package, 'episode', types.SimpleNamespace(run=run), create=True):
                self.assertEqual(mod.main(), 0)
            self.assertEqual(seen, [mode])

    def test_omitted_vhhome_flag_preserves_exported_setting(self):
        from vhhome_bench.cli._common import add_assistant_args
        import argparse
        ap = argparse.ArgumentParser()
        add_assistant_args(ap, default='silent')
        self.assertIsNone(ap.parse_args([]).latency)
        for mode in ['2', 'measured', '0']:
            self.assertEqual(ap.parse_args(['--latency', mode]).latency, mode)

    def test_cooksim_actual_delay_formula(self):
        path = ROOT / 'cooksim/cooksim_bench/e2e_stepin_rollout.py'
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'assist_latency_ticks')
        ns = {'os': os, 'TICK_SECONDS': 1.2}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), ns)
        for mode, expected in [('2', [2, 2, 2, 2]), ('measured', [1, 1, 4, 30]), ('0', [0, 0, 0, 0])]:
            with patch.dict(os.environ, {'COOKSIM_ASSIST_LATENCY': mode, 'COOKSIM_ASSIST_LATENCY_MAX': '30'}):
                actual = [ns['assist_latency_ticks'](seconds) for seconds in [0.01, 1.2, 4.8, 100]]
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
