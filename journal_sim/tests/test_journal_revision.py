import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from journal_sim.config import JournalConfig, certificate_context_fingerprint, config_fingerprint
from journal_sim.certification.certificate import certificate_bisection
from journal_sim.core.channels import generate_channel
from journal_sim.dynamics.offline_calibration import offline_radius, bind_artifact, calibration_scope, scope_fingerprint
from journal_sim.dynamics.reconfiguration import PolicyRunner
from journal_sim.experiments.exp0_oracle_validation import choose_representatives
from journal_sim.experiments.exp3_dynamic_reconfiguration import run_seed
from .helpers import tiny_config, scalar_case


class JournalRevisionTests(unittest.TestCase):
    def test_certificate_context_excludes_experiment_settings(self):
        cfg, w, H = scalar_case()
        theta = np.ones(2)
        cert = certificate_bisection(w, H, cfg, theta)
        other = cfg.with_overrides(output_root="another", seeds=(90001,), mobility_level="fast", pool_size=12)
        self.assertTrue(cert.matches(w, H, theta, other))
        self.assertNotEqual(config_fingerprint(cfg), config_fingerprint(other))
        for change in (dict(gamma=2.), dict(noise_power_dbm=-20.), dict(radius_floor=1e-9), dict(strict_eig_tol=1e-7)):
            self.assertNotEqual(certificate_context_fingerprint(cfg), certificate_context_fingerprint(cfg.with_overrides(**change)))
        self.assertFalse(cert.matches(w * 1.01, H, theta, other))

    def test_offline_scope_hash_and_joint_quantile(self):
        cfg = tiny_config(epsilon_est=None)
        payload = dict(schema="journal_offline_joint_radius_v1", complete=True,
                       interpretation="empirical joint quantiles", entries=[dict(scope=calibration_scope(cfg),
                       scope_fingerprint=scope_fingerprint(cfg), epsilon_joint_95=.02, epsilon_joint_99=.03)])
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "joint_radius.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            bound = bind_artifact(cfg, str(path))
            value, provenance = offline_radius(bound)
            self.assertEqual(value, .03)
            self.assertEqual(provenance["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(offline_radius(bound.with_overrides(calibration_q=.95))[0], .02)
            with self.assertRaises(ValueError):
                offline_radius(bound.with_overrides(estimation_snr_db=10))
            with self.assertRaises(ValueError):
                offline_radius(bound.with_overrides(csi_calibration_sha256="corrupted"))
            with self.assertRaises(ValueError):
                offline_radius(cfg)

    def test_online_never_calls_monte_carlo_calibration(self):
        cfg = tiny_config(epsilon_est=.01, pool_size=1, pool_attempts=1, design_gamma_mult=(2.,),
                          power_slack_grid=(1.,), noise_power_dbm=30, p_max_dbm=40)
        channel = generate_channel(cfg, 60001).with_links(h_bu=np.full((1, 1, 1, 1), 10.), h_ru=np.zeros((1, 1, 2)))
        runner = PolicyRunner("always_reconfigure", cfg, 60001)
        with patch("journal_sim.dynamics.csi_estimation.calibrate_physical", side_effect=AssertionError("online calibration forbidden")) as calibration:
            for t in range(2):
                _, event = runner.step(channel, t)
                self.assertEqual(event["calibration_runtime"], 0)
                self.assertEqual(event["epsilon_est_offline"], .01)
            self.assertEqual(calibration.call_count, 0)
        self.assertTrue(runner.state.valid_for(cfg))

    def test_three_journal_representatives_preselected_by_rank(self):
        pool = [SimpleNamespace(index=i, epsilon_cert=e, valid_certificate=v) for i, e, v in
                ((0, 0., False), (1, .02, True), (2, .04, True), (3, .08, True), (4, .01, True))]
        reps = choose_representatives(pool)
        self.assertEqual([(label, c.index) for label, c in reps],
                         [("journal_fragile", 4), ("journal_medium", 2), ("journal_robust", 3)])
        self.assertEqual(len(pool), 5)

    def test_three_mobility_five_policy_formal_defaults(self):
        cfg = JournalConfig(epsilon_est=.01).validate()
        self.assertEqual((cfg.L, cfg.K, cfg.N, cfg.bits, cfg.pool_size, cfg.time_steps, len(cfg.seeds)), (2, 3, 32, 2, 12, 100, 10))
        self.assertEqual(len(cfg.policies), 5)
        self.assertLess(cfg.periodic_short, cfg.periodic_long)
        result = dict(records=[], summaries=[], designs=[], arrays={}, end_to_end_runtime=1.)
        with patch("journal_sim.experiments.exp3_dynamic_reconfiguration.run_paired_policies", return_value=result) as run:
            output = run_seed(cfg, 60001)
        self.assertEqual([c.args[0].mobility_level for c in run.call_args_list], ["slow", "medium", "fast"])
        self.assertTrue(all(c.args[0].policies == cfg.policies for c in run.call_args_list))
        self.assertEqual(output["end_to_end_runtime"], 3.)

    def test_two_periodic_schedules(self):
        from journal_sim.core.models import Configuration
        from journal_sim.dynamics.trigger import reset_reference, trigger_decision
        cfg, w, H = scalar_case()
        cfg = cfg.with_overrides(periodic_short=2, periodic_long=5)
        X = Configuration(w, np.ones(2))
        cert = certificate_bisection(w, H, cfg, X.theta)
        state = reset_reference(X, cert, H, 0, 0., cfg)
        self.assertTrue(trigger_decision("periodic_short", state, H, 2, cfg).triggered)
        self.assertFalse(trigger_decision("periodic_long", state, H, 2, cfg).triggered)
        self.assertTrue(trigger_decision("periodic_long", state, H, 5, cfg).triggered)
