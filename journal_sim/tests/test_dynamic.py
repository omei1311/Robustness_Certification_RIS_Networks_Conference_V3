import unittest
from dataclasses import replace
import numpy as np
from journal_sim.core.models import Configuration
from journal_sim.certification.certificate import certificate_bisection
from journal_sim.dynamics.trigger import reset_reference, trigger_decision
from .helpers import scalar_case


class TriggerTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.w, self.H = scalar_case()
        self.X = Configuration(self.w, np.ones(2))
        self.cert = certificate_bisection(self.w, self.H, self.cfg, self.X.theta)
        self.ref = reset_reference(self.X, self.cert, self.H, 0, 0., self.cfg)

    def test_no_trigger_before_threshold(self):
        threshold = self.cfg.eta_trigger * self.cert.epsilon_cert
        d = trigger_decision("certificate_triggered", self.ref, self.H * (1 + .9 * threshold), 1, self.cfg)
        self.assertFalse(d.triggered)
        self.assertLess(d.rho_total, d.threshold)

    def test_trigger_after_threshold(self):
        threshold = self.cfg.eta_trigger * self.cert.epsilon_cert
        d = trigger_decision("certificate_triggered", self.ref, self.H * (1 + 1.1 * threshold), 1, self.cfg)
        self.assertTrue(d.triggered)
        self.assertGreater(d.rho_total, d.threshold)

    def test_equality_triggers(self):
        # Exact comparison exercised with zero drift and estimation budget
        # equal to the stored threshold (avoids floating subtraction in drift).
        threshold = self.cfg.eta_trigger * self.cert.epsilon_cert
        ref = replace(self.ref, epsilon_est_calibrated=threshold)
        d = trigger_decision("certificate_triggered", ref, self.H, 1, self.cfg)
        self.assertEqual(d.rho_total, d.threshold)
        self.assertTrue(d.triggered)

    def test_change_resets_reference(self):
        new_H = self.H * 1.2
        new_X = Configuration(self.w * 1.1, -self.X.theta)
        new_cert = certificate_bisection(new_X.w, new_H, self.cfg, new_X.theta)
        ref = reset_reference(new_X, new_cert, new_H, 7, 0., self.cfg)
        d = trigger_decision("certificate_triggered", ref, new_H, 8, self.cfg)
        self.assertEqual(d.rho_obs, 0)
        self.assertEqual(ref.reference_time, 7)
        self.assertNotEqual(ref.configuration.configuration_id, self.ref.configuration.configuration_id)

    def test_old_certificate_invalidated_after_w_change(self):
        changed = replace(self.ref, configuration=Configuration(self.w * 1.01, self.X.theta))
        self.assertFalse(changed.valid_for(self.cfg))
        self.assertTrue(trigger_decision("certificate_triggered", changed, self.H, 1, self.cfg).triggered)
        with self.assertRaises(ValueError):
            reset_reference(changed.configuration, self.cert, self.H, 1, 0., self.cfg)

    def test_policy_schedules(self):
        self.assertTrue(trigger_decision("always_reconfigure", self.ref, self.H, 1, self.cfg).triggered)
        self.assertFalse(trigger_decision("periodic_reconfigure", self.ref, self.H, 1, self.cfg).triggered)
        self.assertTrue(trigger_decision("periodic_reconfigure", self.ref, self.H, self.cfg.T_period, self.cfg).triggered)
        self.assertFalse(trigger_decision("static", self.ref, self.H, 99, self.cfg).triggered)

    def test_estimation_budget_reference_scale(self):
        ref = replace(self.ref, epsilon_est_calibrated=.02)
        d = trigger_decision("certificate_triggered", ref, self.H * 1.1, 1, self.cfg)
        self.assertAlmostEqual(d.epsilon_est, .022)
        self.assertAlmostEqual(d.rho_total, .122)
