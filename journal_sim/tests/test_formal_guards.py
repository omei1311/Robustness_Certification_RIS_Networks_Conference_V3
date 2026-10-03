"""Formal-mode guards: smoke artifacts and seed leakage must stop the run.

Guards raise SystemExit with an explicit message; smoke configs always pass.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from journal_sim.dynamics.calibration_guards import formal_calibration_guard
from journal_sim.dynamics.drift_calibration import bind_drift_artifact, drift_scope_fingerprint
from journal_sim.dynamics.offline_calibration import bind_artifact, calibration_scope, scope_fingerprint
from .helpers import tiny_config
from .test_drift_calibration import write_drift_artifact


def write_csi_artifact(path, cfg, mode="formal", joint99=.03, joint95=.02, joint90=.01):
    payload = dict(schema="journal_offline_joint_radius_v1", mode=mode, complete=True,
                   interpretation="test artifact",
                   entries=[dict(scope_fingerprint=scope_fingerprint(cfg), scope=calibration_scope(cfg),
                                 epsilon_joint_90=joint90, epsilon_joint_95=joint95, epsilon_joint_99=joint99,
                                 per_seed=[dict(seed=cfg.calibration_seeds[0], epsilon_joint_90=joint90,
                                                epsilon_joint_95=joint95, epsilon_joint_99=joint99)],
                                 aggregate=dict(epsilon_joint_90=joint90, epsilon_joint_95=joint95,
                                                epsilon_joint_99=joint99))])
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
    return path


def formal_cfg(tmp, csi_mode="formal", drift_mode="formal", with_csi=True, with_drift=True):
    # tiny_config pins epsilon_est=0.; the guard tests need it unset so the
    # artifact paths are actually exercised.
    cfg = tiny_config(epsilon_est=None)
    if with_csi:
        cfg = bind_artifact(cfg, str(write_csi_artifact(Path(tmp) / "csi.json", cfg, mode=csi_mode)))
    if with_drift:
        cfg = bind_drift_artifact(cfg, str(write_drift_artifact(Path(tmp) / "drift.json", cfg,
                                                                mode=drift_mode)))
    return cfg


class FormalGuardTests(unittest.TestCase):
    def test_formal_exp3_accepts_valid_formal_artifacts(self):
        with TemporaryDirectory() as tmp:
            cfg = formal_cfg(tmp)
            formal_calibration_guard(cfg, "exp3")  # must not raise

    def test_formal_exp3_rejects_smoke_csi_artifact(self):
        with TemporaryDirectory() as tmp:
            cfg = formal_cfg(tmp, csi_mode="smoke")
            with self.assertRaises(SystemExit) as caught:
                formal_calibration_guard(cfg, "exp3")
            self.assertIn("smoke artifact supplied", str(caught.exception))

    def test_formal_exp3_rejects_smoke_drift_artifact(self):
        with TemporaryDirectory() as tmp:
            cfg = formal_cfg(tmp, drift_mode="smoke")
            with self.assertRaises(SystemExit) as caught:
                formal_calibration_guard(cfg, "exp3")
            self.assertIn("smoke artifact supplied", str(caught.exception))

    def test_formal_exp3_rejects_missing_drift_artifact(self):
        with TemporaryDirectory() as tmp:
            cfg = formal_cfg(tmp, with_drift=False)
            with self.assertRaises(SystemExit) as caught:
                formal_calibration_guard(cfg, "exp3")
            self.assertIn("requires a formal drift calibration artifact", str(caught.exception))

    def test_formal_exp3_rejects_missing_csi_artifact_and_explicit_radius_is_allowed(self):
        with TemporaryDirectory() as tmp:
            cfg = formal_cfg(tmp, with_csi=False)
            with self.assertRaises(SystemExit) as caught:
                formal_calibration_guard(cfg, "exp3")
            self.assertIn("requires a formal CSI calibration artifact", str(caught.exception))
            explicit = cfg.with_overrides(epsilon_est=.02, csi_calibration_file=None,
                                          csi_calibration_sha256=None)
            formal_calibration_guard(explicit, "exp3")  # declared parameter, must not raise

    def test_formal_exp2_soft_drift_warns_instead_of_failing(self):
        with TemporaryDirectory() as tmp:
            cfg = formal_cfg(tmp, with_drift=False)
            formal_calibration_guard(cfg, "exp2", drift="soft")  # must not raise
            smoke_legacy = cfg.with_overrides(smoke=True, epsilon_est=.02,
                                              csi_calibration_file=None, csi_calibration_sha256=None)
            formal_calibration_guard(smoke_legacy, "exp3")  # smoke bypasses everything

    def test_tampered_csi_artifact_fails_sha_check(self):
        # Binding snapshots the artifact; a corrupted hash is detected. (An
        # in-place file edit after binding is masked by the read cache within
        # one process, which is exactly the consistent-snapshot guarantee.)
        with TemporaryDirectory() as tmp:
            cfg = formal_cfg(tmp).with_overrides(csi_calibration_sha256="0" * 64)
            with self.assertRaises(SystemExit) as caught:
                formal_calibration_guard(cfg, "exp3")
            self.assertIn("hash mismatch", str(caught.exception))

    def test_drift_artifact_missing_mobility_blocks_formal(self):
        import hashlib
        with TemporaryDirectory() as tmp:
            cfg = formal_cfg(tmp)
            path = write_drift_artifact(Path(tmp) / "partial.json", cfg, mobilities=("slow", "fast"))
            sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            cfg2 = cfg.with_overrides(drift_calibration_file=str(path), drift_calibration_sha256=sha)
            with self.assertRaises(SystemExit) as caught:
                formal_calibration_guard(cfg2, "exp3")
            self.assertIn("lacks mobility regime medium", str(caught.exception))

    def test_evaluation_seed_overlap_with_calibration_seeds_is_fatal(self):
        with TemporaryDirectory() as tmp:
            cfg = formal_cfg(tmp).with_overrides(seeds=(53001,))
            with self.assertRaises(SystemExit) as caught:
                formal_calibration_guard(cfg, "exp3")
            self.assertIn("overlap calibration seeds", str(caught.exception))
            legacy = tiny_config()
            self.assertFalse(set(legacy.seeds) &
                             (set(legacy.calibration_seeds) | set(legacy.drift_calibration_seeds)))


if __name__ == "__main__":
    unittest.main()
