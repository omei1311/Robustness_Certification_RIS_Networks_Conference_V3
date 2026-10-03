"""Seed-level parallel execution must be invisible to the science.

Run this suite via ``python -m unittest discover`` (as documented): spawn
workers re-import the parent ``__main__``, and the unittest console entry
point is import-safe, unlike pytest's console script.
"""
import csv
import json
import sys
import time
import unittest
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from journal_sim.config import JournalConfig, certificate_context_fingerprint, config_fingerprint
from journal_sim.experiments.common import (ExecutionOptions, execute, _resolve_run_seed_module,
                                            _run_seed_worker)
from journal_sim.experiments.exp3_dynamic_reconfiguration import run_seed as exp3_run_seed
from .helpers import tiny_config


def run_seed(cfg, seed):
    """Module-level worker callable: staggered completion, one injected failure.

    Later seeds sleep less, so with workers>1 the actual completion order is
    the reverse of cfg.seeds; one seed fails outright.
    """
    if seed == 60002:
        raise RuntimeError("injected seed failure")
    rank = list(cfg.seeds).index(seed)
    time.sleep(max(0.0, 1.5 - 0.5 * rank))
    return dict(records=[dict(value=seed % 100, note="x")], arrays=dict(marker=[float(seed)]))


def noop_plot(rows, details, output, cfg):
    (output / "plot.png").write_bytes(b"test placeholder")


def read_records(path):
    with open(path / "records.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class ExecutionOptionTests(unittest.TestCase):
    def test_workers_is_runtime_only_and_never_a_fingerprint_input(self):
        cfg = JournalConfig()
        self.assertNotIn("workers", [f.name for f in fields(cfg)])
        self.assertEqual(config_fingerprint(cfg), config_fingerprint(JournalConfig()))
        self.assertEqual(certificate_context_fingerprint(cfg), certificate_context_fingerprint(JournalConfig()))
        ExecutionOptions(workers=8).validate()
        with self.assertRaises(ValueError):
            ExecutionOptions(workers=0).validate()

    def test_module_resolution_for_python_m_execution(self):
        self.assertEqual(_resolve_run_seed_module(run_seed), __name__)
        class MainOnly:  # emulates a __main__-defined callable without a spec
            pass
        MainOnly.__module__ = "__main__"
        with patch.object(sys, "modules", {"__main__": SimpleNamespace()}):
            with self.assertRaises(ValueError):
                _resolve_run_seed_module(MainOnly)


class ParallelEqualityTests(unittest.TestCase):
    def test_workers_1_and_2_produce_identical_science(self):
        # One shared config (hence one fingerprint); the two runs land in
        # different timestamped directories under the same output root.
        with TemporaryDirectory() as tmp:
            cfg = tiny_config(output_root=tmp, epsilon_est=.01, pool_size=1, pool_attempts=1,
                              time_steps=1, policies=("certificate_triggered", "static"),
                              mobility_regimes=("slow",), c0_db=90., alpha_br=2.0, alpha_ru=2.0,
                              p_max_dbm=40., design_gamma_mult=(2.,), power_slack_grid=(1.,),
                              seeds=(60001, 60002))
            serial_root = execute("par_equal", cfg, exp3_run_seed, noop_plot, workers=1)[0]
            parallel_root = execute("par_equal", cfg, exp3_run_seed, noop_plot, workers=2)[0]
            serial_rows = {(r["seed"], r["policy"]): r for r in read_records(serial_root)}
            parallel_rows = {(r["seed"], r["policy"]): r for r in read_records(parallel_root)}
            self.assertEqual(set(serial_rows), set(parallel_rows))
            scientific = ("trajectory_id", "csi_trajectory_id", "configuration_id", "qos_hold",
                          "design_status", "rho_total", "trigger_threshold", "sinr_min")
            for key in serial_rows:
                for column in scientific:
                    self.assertEqual(serial_rows[key][column], parallel_rows[key][column],
                                     f"{key} {column}")
            serial_manifest = json.loads((serial_root / "manifest.json").read_text(encoding="utf-8"))
            parallel_manifest = json.loads((parallel_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(serial_manifest["fingerprint"], parallel_manifest["fingerprint"])
            self.assertFalse(serial_manifest["execution"]["parallel"])
            self.assertTrue(parallel_manifest["execution"]["parallel"])
            self.assertFalse(parallel_manifest["execution"]["runtime_measurement_valid"])
            for a, b in zip(serial_manifest["seed_details"], parallel_manifest["seed_details"]):
                self.assertEqual(a["seed"], b["seed"])
                self.assertEqual(a["status"], b["status"])
                summary_a = {(s["policy"], s["trajectory_id"], s["csi_trajectory_id"],
                              s["number_of_reconfigurations"], s["qos_outage_rate"]) for s in a["summary"]}
                summary_b = {(s["policy"], s["trajectory_id"], s["csi_trajectory_id"],
                              s["number_of_reconfigurations"], s["qos_outage_rate"]) for s in b["summary"]}
                self.assertEqual(summary_a, summary_b)


class ParallelOrderAndFailureTests(unittest.TestCase):
    def test_order_failure_and_staging_behaviour(self):
        cfg = tiny_config(output_root=None, seeds=(60001, 60002, 60003))
        with TemporaryDirectory() as tmp:
            cfg = cfg.with_overrides(output_root=tmp)
            output, success = execute("par_order", cfg, run_seed, noop_plot, workers=2)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(success)
            self.assertEqual([d["seed"] for d in manifest["seed_details"]], list(cfg.seeds))
            self.assertEqual([d["status"] for d in manifest["seed_details"]],
                             ["COMPLETED", "FAILED", "COMPLETED"])
            self.assertIn("injected seed failure", json.dumps(manifest["seed_details"]))
            # CSV order follows cfg.seeds, not the reversed completion order.
            seeds_seen = [r.get("seed") for r in read_records(output)]
            self.assertEqual([s for i, s in enumerate(seeds_seen) if i == 0 or seeds_seen[i - 1] != s],
                             ["60001", "60002", "60003"])
            import numpy as np
            with np.load(output / "arrays.npz") as arrays:
                self.assertIn("seed60001_marker", arrays.files)
                self.assertNotIn("seed60002_marker", arrays.files)
                self.assertIn("seed60003_marker", arrays.files)
            # Failed runs keep the staging directory for post-mortem recovery.
            self.assertTrue((output / "_worker_staging").exists())

    def test_worker_writes_only_its_own_staging_file(self):
        cfg = tiny_config(seeds=(60001,))
        with TemporaryDirectory() as tmp:
            staging = Path(tmp) / "stage" / "seed_60001_arrays.npz"
            payload = dict(module_name=__name__, cfg=cfg, seed=60001, staging_path=str(staging))
            result = _run_seed_worker(payload)
            self.assertNotIn("arrays", result)
            self.assertTrue(staging.exists())
            for shared in ("records.csv", "manifest.json", "arrays.npz", "plot.png"):
                self.assertFalse((Path(tmp) / shared).exists(), shared)


class SerialBenchmarkGuardTests(unittest.TestCase):
    def test_exp4_rejects_workers_above_one(self):
        from journal_sim.experiments import exp4_runtime_scaling
        with patch.object(sys, "argv", ["exp4_runtime_scaling.py", "--workers", "2"]):
            with self.assertRaises(SystemExit) as caught:
                exp4_runtime_scaling.main()
        self.assertIn("serial benchmark", str(caught.exception))
        self.assertIn("--workers 1", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
