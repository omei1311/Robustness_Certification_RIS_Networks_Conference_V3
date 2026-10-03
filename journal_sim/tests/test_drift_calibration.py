"""Drift calibration: trigger-identical rho definition, units, scope, priority.

nu is an empirical predictor input only; nothing here may change the online
trigger or the certificate definition.
"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
from journal_sim.config import smoke_config
from journal_sim.core.channels import effective_channels
from journal_sim.core.models import Configuration
from journal_sim.core.uncertainty import relative_drift
from journal_sim.design.selection import certified_lifetime, select_lifetime_aware
from journal_sim.dynamics import drift_calibration, trigger
from journal_sim.dynamics.channel_process import channel_trajectory
from journal_sim.dynamics.csi_estimation import estimate_physical, csi_observation_seed
from journal_sim.dynamics.drift_calibration import (drift_samples, summarize_drift, rho_per_second,
                                                    drift_scope_fingerprint, bind_drift_artifact,
                                                    offline_drift_rate, effective_drift_rate)
from journal_sim.experiments.exp1b_drift_calibration import run_seed as drift_run_seed
from .helpers import tiny_config


def write_drift_artifact(path, cfg, mobilities=("slow", "medium", "fast"), mode="formal", nu=0.1):
    payload = dict(schema_version=1, calibration_type="journal_offline_drift_rate_v1", mode=mode,
                   quantile=cfg.drift_calibration_q, unit="1/s", complete=True,
                   scope_fingerprint=drift_scope_fingerprint(cfg),
                   mobility={m: dict(nu_50=nu / 2, nu_90=nu, nu_95=nu, nu_99=2 * nu,
                                     nu_selected=nu, sample_count=10) for m in mobilities},
                   interpretation="test artifact")
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
    return path


class DriftDefinitionTests(unittest.TestCase):
    def test_uses_the_triggers_own_relative_drift(self):
        # Structural: the calibration module must call the trigger's function,
        # never a reimplementation that could drift apart later.
        self.assertIs(drift_calibration.relative_drift, trigger.relative_drift)

    def test_drift_sample_matches_manual_pipeline_replay(self):
        # Recompute one calibration cell through the documented pipeline and
        # require exact agreement with drift_samples().
        cfg = smoke_config().with_overrides(drift_calibration_slots=6, drift_reference_stride=3,
                                            drift_horizon_slots=(3,))
        rows = drift_samples(cfg, cfg.drift_calibration_seeds[0])
        local = cfg.with_overrides(mobility_level="slow", time_steps=cfg.drift_calibration_slots)
        seed = cfg.drift_calibration_seeds[0]
        trajectory = channel_trajectory(local, seed)
        estimates = [estimate_physical(c, local, csi_observation_seed(local, seed, t))
                     for t, c in enumerate(trajectory)]
        theta = drift_calibration.phase_profile(local, seed, 0)
        H_hat = [effective_channels(est, theta, local) for est in estimates]
        expected = relative_drift(H_hat[3], H_hat[0], local)
        cell = next(r for r in rows if r["mobility"] == "slow" and r["phase_profile"] == 0
                    and r["reference_index"] == 0 and r["delta_slots"] == 3)
        self.assertEqual(cell["rho_obs"], expected)
        self.assertAlmostEqual(cell["rho_over_time"], expected / (3 * cfg.slot_duration))

    def test_nu_unit_is_per_second(self):
        cfg = tiny_config(slot_duration=.1)
        self.assertAlmostEqual(rho_per_second(.02, 2, cfg), .1)
        with self.assertRaises(ValueError):
            rho_per_second(.02, 0, cfg)

    def test_lifetime_orders_by_mobility_nu(self):
        # Same budget, faster mobility => shorter predicted reuse. Formula
        # check only; no performance claim.
        slow = certified_lifetime(.06, .02, .05, .9)
        fast = certified_lifetime(.06, .02, .20, .9)
        self.assertGreater(slow, fast)
        cfg = tiny_config(drift_rate_nu=1., eta_trigger=.9)
        cand = SimpleNamespace(index=0, rate=10., system_power=1., wee=10., epsilon_cert=.06,
                               epsilon_est=.02, valid_certificate=True,
                               configuration=Configuration(np.ones((1, 1, 1)), np.ones(2)),
                               certificate=SimpleNamespace(status="VALIDATED"))
        t_slow = select_lifetime_aware([cand], np.ones(2), cfg, epsilon_est=.02, drift_rate_nu=.05)
        t_fast = select_lifetime_aware([cand], np.ones(2), cfg, epsilon_est=.02, drift_rate_nu=.20)
        self.assertGreater(t_slow.scores[0]["predicted_lifetime"],
                           t_fast.scores[0]["predicted_lifetime"])


class DriftScopeTests(unittest.TestCase):
    def test_scope_excludes_experiment_budget(self):
        base = tiny_config()
        for change in (dict(pool_size=9), dict(time_steps=77), dict(seeds=(99,)), dict(policies=("static",)),
                       dict(solver_tol=1e-3), dict(eta_trigger=.5)):
            self.assertEqual(drift_scope_fingerprint(base),
                             drift_scope_fingerprint(base.with_overrides(**change)), str(change))

    def test_scope_tracks_drift_physics(self):
        base = tiny_config()
        for change in (dict(mobility_correlations=(0.9, 0.995, 0.95)), dict(estimation_snr_db=25.),
                       dict(slot_duration=.2), dict(alpha_ru=3.0), dict(quasi_static_br=False),
                       dict(rician_k=6.)):
            self.assertNotEqual(drift_scope_fingerprint(base),
                                drift_scope_fingerprint(base.with_overrides(**change)), str(change))


class DriftArtifactTests(unittest.TestCase):
    def test_binding_scope_sha_and_mobility_gates(self):
        cfg = tiny_config(epsilon_est=.01)
        with TemporaryDirectory() as tmp:
            path = write_drift_artifact(Path(tmp) / "drift.json", cfg)
            bound = bind_drift_artifact(cfg, str(path))
            nu, provenance = offline_drift_rate(bound, "fast")
            self.assertEqual(nu, .1)
            self.assertEqual(provenance["quantile"], cfg.drift_calibration_q)
            # Scope change rejects the artifact.
            moved = bound.with_overrides(estimation_snr_db=17.)
            with self.assertRaises(ValueError):
                offline_drift_rate(moved, "slow")
            # A corrupted binding (wrong SHA) is rejected; binding takes a
            # snapshot, so this is checked against a mismatched hash rather
            # than an in-place file edit.
            with self.assertRaises(ValueError):
                offline_drift_rate(bound.with_overrides(drift_calibration_sha256="0" * 64), "slow")
            # An artifact missing a regime is rejected already at binding time.
            partial = write_drift_artifact(Path(tmp) / "partial.json", cfg, mobilities=("slow", "fast"))
            with self.assertRaises(ValueError):
                bind_drift_artifact(cfg, str(partial))

    def test_effective_drift_rate_priority_order(self):
        cfg = tiny_config(drift_rate_nu=.002)
        nu, provenance = effective_drift_rate(cfg, "slow")
        self.assertEqual((nu, provenance["source"]), (.002, "legacy_manual_constant"))
        manual = cfg.with_overrides(drift_rate_nu_by_mobility=(.05, .1, .2))
        self.assertEqual([effective_drift_rate(manual, m)[0] for m in ("slow", "medium", "fast")],
                         [.05, .1, .2])
        with TemporaryDirectory() as tmp:
            bound = bind_drift_artifact(cfg, str(write_drift_artifact(Path(tmp) / "d.json", cfg, nu=.123)))
            self.assertEqual(effective_drift_rate(bound, "medium")[0], .123)
            # Manual values and an artifact may not be mixed.
            with self.assertRaises(ValueError):
                bound.with_overrides(drift_rate_nu_by_mobility=(1, 1, 1)).validate()


class DriftSmokeTests(unittest.TestCase):
    def test_mini_exp1b_seed_yields_positive_finite_nu(self):
        cfg = smoke_config()  # smoke budgets: 12 slots, horizons (1,2,4), 1 seed set
        rows = drift_run_seed(cfg, cfg.drift_calibration_seeds[0])["records"]
        self.assertTrue(rows)
        summary = summarize_drift(rows, cfg)
        for mobility in ("slow", "medium", "fast"):
            entry = summary[mobility]
            self.assertTrue(np.isfinite(entry["nu_selected"]) and entry["nu_selected"] > 0)
            self.assertLessEqual(entry["nu_50"], entry["nu_95"])
            self.assertLessEqual(entry["nu_95"], entry["nu_99"])


if __name__ == "__main__":
    unittest.main()
