import unittest
import numpy as np
from journal_sim.config import config_fingerprint, JournalConfig
from journal_sim.dynamics.csi_estimation import calibrate_csi, estimate_effective
from .helpers import tiny_config


class ReproducibilityTests(unittest.TestCase):
    def test_csi_calibration_reproducibility(self):
        cfg = tiny_config(calibration_samples=500)
        H = np.ones((1, 1, 1, 1), complex)
        a, b = calibrate_csi(H, cfg, 52001), calibrate_csi(H, cfg, 52001)
        np.testing.assert_array_equal(a["relative_error_distribution"], b["relative_error_distribution"])
        self.assertLess(a["epsilon_90"], a["epsilon_95"])
        self.assertLess(a["epsilon_95"], a["epsilon_99"])
        self.assertAlmostEqual(a["measured_nmse"], 1e-4, delta=2e-5)
        np.testing.assert_array_equal(estimate_effective(H, cfg, 1), estimate_effective(H, cfg, 1))
        self.assertFalse(np.array_equal(estimate_effective(H, cfg, 1), estimate_effective(H, cfg, 2)))

    def test_nmse_parameter(self):
        cfg = tiny_config(nmse_db=-20, calibration_samples=500)
        out = calibrate_csi(np.ones((1, 1, 1, 1)), cfg, 52001)
        self.assertAlmostEqual(out["measured_nmse"], .01, delta=.002)

    def test_fingerprint_all_parameters(self):
        from dataclasses import fields, replace
        cfg = JournalConfig()
        fp = config_fingerprint(cfg)
        self.assertEqual(fp, config_fingerprint(cfg))
        for f in fields(cfg):
            value = getattr(cfg, f.name)
            changed = (not value if isinstance(value, bool) else value + 1 if isinstance(value, (int, float)) else
                       value + "changed" if isinstance(value, str) else value + ("changed",) if isinstance(value, tuple) else 1.)
            self.assertNotEqual(fp, config_fingerprint(replace(cfg, **{f.name: changed})), f.name)
