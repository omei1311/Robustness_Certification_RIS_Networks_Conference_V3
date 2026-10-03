import unittest
from unittest.mock import patch
import numpy as np
from journal_sim.config import smoke_config
from journal_sim.evaluation.dynamic import paired_observations, run_paired_policies
from journal_sim.dynamics.reconfiguration import PolicyRunner
from journal_sim.core.models import array_digest
from journal_sim.tests.helpers import tiny_config


class PairedTests(unittest.TestCase):
    def test_paired_same_trajectory_and_noise(self):
        cfg = tiny_config(time_steps=3, pool_size=1, pool_attempts=1, calibration_samples=10)
        trajectory, estimates = paired_observations(cfg, 60001)
        # Force no candidates; every policy/time must still be retained.
        empty_stats = dict(attempts=[dict(status="GENERATION_FAILED")], candidate_generation_runtime=0.,
                           certificate_runtime=0., strict_validation_runtime=0., fast_oracle_calls=0, strict_oracle_calls=0)
        seen = []
        def empty(channel, cfg, seed, t):
            seen.append((t, id(channel), array_digest(channel.h_bu, channel.h_ru)))
            return [], empty_stats
        with patch("journal_sim.dynamics.reconfiguration.build_candidate_pool", side_effect=empty):
            out = run_paired_policies(cfg, 60001, trajectory, estimates)
        self.assertEqual(len(out["records"]), len(cfg.policies) * cfg.time_steps)
        for t in range(cfg.time_steps):
            self.assertEqual(len({(i, digest) for time, i, digest in seen if time == t}), 1)
        self.assertEqual(len({r["trajectory_id"] for r in out["records"]}), 1)
        self.assertEqual(len({r["csi_trajectory_id"] for r in out["records"]}), 1)
        self.assertTrue(all(r["design_status"] == "NO_ELIGIBLE_CANDIDATE" for r in out["records"]))
        self.assertTrue(all(r["rate"] == 0 and not r["qos_hold"] for r in out["records"]))
        self.assertTrue(all(s["failed_design_steps"] == cfg.time_steps for s in out["summaries"]))

    def test_design_cannot_import_evaluation_or_legacy(self):
        import ast
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        for directory in ("design", "certification", "dynamics"):
            for path in (root / directory).glob("*.py"):
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else [x.name for x in node.names] if isinstance(node, ast.Import) else []
                    for module in modules:
                        self.assertFalse(module.startswith("src") or "evaluation" in module, str(path))
