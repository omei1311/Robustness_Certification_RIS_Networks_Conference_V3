import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from journal_sim.experiments.common import execute, load_config
from journal_sim.core.channels import generate_channel
from journal_sim.evaluation.dynamic import run_paired_policies
from journal_sim.config import config_fingerprint
from .helpers import tiny_config


class ExperimentTests(unittest.TestCase):
    def test_failed_seeds_preserved_and_config_roundtrip(self):
        with TemporaryDirectory() as tmp:
            cfg = tiny_config(output_root=tmp, seeds=(60001, 60002))
            def run(local, seed):
                if seed == 60001:
                    raise RuntimeError("injected seed failure")
                return dict(records=[dict(value=2)], arrays=dict(array=np.ones(2)))
            def plot(rows, details, output, local):
                (output / "plot.png").write_bytes(b"test placeholder")
            output, success = execute("test", cfg, run, plot)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(success)
            self.assertEqual([x["status"] for x in manifest["seed_details"]], ["FAILED", "COMPLETED"])
            self.assertIn("60001", (output / "records.csv").read_text())
            self.assertEqual(config_fingerprint(cfg), config_fingerprint(load_config(output / "manifest.json")))
            self.assertEqual(manifest["timestamp"][-6:], "+08:00")
            with np.load(output / "arrays.npz") as arrays:
                np.testing.assert_array_equal(arrays["seed60002_array"], np.ones(2))

    def test_successful_policies_reuse_and_energy(self):
        cfg = tiny_config(pool_size=2, pool_attempts=2, time_steps=3, T_period=2,
                          calibration_samples=10, noise_power_dbm=30, p_max_dbm=40,
                          estimation_snr_db=80, design_gamma_mult=(2.,), power_slack_grid=(1.,))
        c = generate_channel(cfg, 60001)
        c = c.with_links(h_bu=np.full_like(c.h_bu, 10), h_ru=np.zeros_like(c.h_ru))
        out = run_paired_policies(cfg, 60001, (c,) * 3, (c,) * 3)
        summaries = {s["policy"]: s for s in out["summaries"]}
        self.assertEqual(summaries["always_reconfigure"]["number_of_reconfigurations"], 2)
        self.assertEqual(summaries["periodic_reconfigure"]["number_of_reconfigurations"], 1)
        self.assertEqual(summaries["certificate_triggered"]["number_of_reconfigurations"], 0)
        self.assertEqual(summaries["static"]["number_of_reconfigurations"], 0)
        for s in summaries.values():
            self.assertEqual(s["qos_outage_rate"], 0)
            self.assertGreater(s["long_term_ee"], 0)
            self.assertEqual(s["number_of_initializations"], 1)
        cert_rows = [r for r in out["records"] if r["policy"] == "certificate_triggered"]
        self.assertEqual(len({r["configuration_id"] for r in cert_rows}), 1)
        self.assertTrue(all(r["certified_budget_hold"] for r in cert_rows))
        designs_t0 = [d for d in out["designs"] if d["time_index"] == 0]
        self.assertEqual(len({tuple(c["candidate_seed"] for c in d["candidates"]) for d in designs_t0}), 1)

    def test_ground_truth_cannot_affect_design(self):
        cfg = tiny_config(pool_size=1, pool_attempts=1, time_steps=1, policies=("certificate_triggered",),
                          calibration_samples=10, noise_power_dbm=30, p_max_dbm=40,
                          estimation_snr_db=80, design_gamma_mult=(2.,), power_slack_grid=(1.,))
        c = generate_channel(cfg, 60001).with_links(h_bu=np.full((1, 1, 1, 1), 10.), h_ru=np.zeros((1, 1, 2)))
        a = run_paired_policies(cfg, 60001, (c,), (c,))
        dark = c.with_links(h_bu=np.zeros_like(c.h_bu))
        b = run_paired_policies(cfg, 60001, (dark,), (c,))
        np.testing.assert_array_equal(a["arrays"]["certificate_triggered_w"], b["arrays"]["certificate_triggered_w"])
        self.assertNotEqual(a["records"][0]["qos_hold"], b["records"][0]["qos_hold"])
