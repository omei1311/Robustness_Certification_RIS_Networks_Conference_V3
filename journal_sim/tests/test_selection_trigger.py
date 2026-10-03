"""Selection must predict reuse time with exactly the trigger's accounting.

Regression guard for the old inconsistency where selection used
(eps_cert - eps_est)/nu while the trigger fires at
eps_est*(1+rho)+rho >= eta*eps_cert, making selection believe in long reuse
while the trigger fired on the very next slot.
"""
import unittest
from types import SimpleNamespace
import numpy as np
from journal_sim.core.models import Configuration
from journal_sim.design.selection import certified_lifetime, select_lifetime_aware
from journal_sim.dynamics.trigger import reset_reference, trigger_decision
from journal_sim.certification.certificate import certificate_bisection
from .helpers import tiny_config, scalar_case


def fake_candidate(index, eps_cert, eps_est, theta):
    return SimpleNamespace(index=index, rate=10., system_power=1., wee=10.,
                           epsilon_cert=eps_cert, epsilon_est=eps_est, valid_certificate=True,
                           configuration=Configuration(np.ones((1, 1, 1)), theta),
                           certificate=SimpleNamespace(status="VALIDATED"))


class CertifiedLifetimeTests(unittest.TestCase):
    def test_case_a_no_budget_at_installation(self):
        # eta * eps_cert = 0.9 * 0.030 = 0.027 < eps_est = 0.028: the candidate
        # exceeds the trigger threshold at rho = 0 and can never be eligible.
        self.assertEqual(certified_lifetime(.030, .028, .002, .9), 0.0)
        cfg = tiny_config(drift_rate_nu=.002, eta_trigger=.9)
        out = select_lifetime_aware([fake_candidate(0, .030, .028, np.ones(2))], np.ones(2), cfg)
        self.assertIsNone(out.candidate)
        self.assertEqual(out.status, "NO_ELIGIBLE_CANDIDATE")
        score = out.scores[0]
        self.assertFalse(score["eligible"])
        self.assertEqual(score["predicted_lifetime"], 0.0)
        self.assertIsNone(score["EE_life"])

    def test_case_b_exact_positive_lifetime(self):
        expected = (0.9 * 0.04 - 0.02) / ((1 + 0.02) * 0.002)
        self.assertAlmostEqual(certified_lifetime(.04, .02, .002, .9), expected, places=12)

    def test_case_c_lifetime_lands_on_trigger_boundary(self):
        # rho = nu * T_pred must make the trigger accounting hit the threshold.
        eps_cert, eps_est, nu, eta = .052, .0286, .002, .9
        T = certified_lifetime(eps_cert, eps_est, nu, eta)
        rho = nu * T
        rho_total = eps_est * (1 + rho) + rho  # exact trigger.py accounting
        self.assertAlmostEqual(rho_total, eta * eps_cert, places=9)
        # One drift step short of T stays inside, one step beyond triggers.
        for margin, expected_inside in ((-.1, True), (+.1, False)):
            t = T * (1 + margin)
            total = eps_est * (1 + nu * t) + nu * t
            self.assertEqual(total < eta * eps_cert, expected_inside)

    def test_argument_guards(self):
        for args in ((.1, .02, 0., .9), (.1, -.1, .002, .9), (.1, .02, .002, 0.),
                     (.1, .02, .002, 1.2), (-.1, .02, .002, .9)):
            with self.assertRaises(ValueError):
                certified_lifetime(*args)

    def test_smoke_configured_candidates_follow_same_rule(self):
        # Smoke calibration eps_est = 0.0286 (40 dB joint 99%): a barely
        # certified candidate (0.030 < 0.0286/0.9) is ineligible at install;
        # the strongest smoke candidate (0.052) predicts ~9 reuse slots.
        self.assertEqual(certified_lifetime(.030, .0286, .002, .9), 0.0)
        T = certified_lifetime(.052, .0286, .002, .9)
        self.assertGreater(T, 5)
        self.assertLess(T, 10)


class SelectionTriggerAgreementTests(unittest.TestCase):
    def test_real_certificate_lifetime_hits_trigger_boundary(self):
        # End-to-end: install a certified configuration, scale the channel to
        # realize rho = nu * T_pred, and require trigger agreement with the
        # prediction. Scaling H by (1+r) realizes rho = r exactly (one user).
        cfg, w, H = scalar_case()
        theta = np.ones(2)
        cert = certificate_bisection(w, H, cfg, theta)
        eps_est = .02
        state = reset_reference(Configuration(w, theta), cert, H, 0, eps_est, cfg)
        T = certified_lifetime(cert.epsilon_cert, eps_est, cfg.drift_rate_nu, cfg.eta_trigger)
        self.assertGreater(T, 0)
        rho = cfg.drift_rate_nu * T
        decision = trigger_decision("certificate_triggered", state, H * (1 + rho), 1, cfg)
        self.assertAlmostEqual(decision.rho_total, decision.threshold, places=9)
        # The prediction lands exactly on the boundary; floating-point
        # rounding may sit an ulp below the >= comparison, so assert firing
        # strictly beyond and holding strictly before the predicted time.
        beyond = trigger_decision("certificate_triggered", state, H * (1 + rho * (1 + 1e-6)), 1, cfg)
        self.assertTrue(beyond.triggered)
        earlier = trigger_decision("certificate_triggered", state, H * (1 + .99 * rho), 1, cfg)
        self.assertFalse(earlier.triggered)

    def test_selection_prefers_higher_certificate_under_trigger_budget(self):
        # With eps_est close to eta*eps_cert of the low-cert candidate, the
        # lifetime-aware rule must move to the higher-certificate candidate
        # even though its instantaneous WEE is lower.
        cfg = tiny_config(drift_rate_nu=.002, eta_trigger=.9)
        pool = [fake_candidate(0, .030, .028, -np.ones(2)),   # budget exhausted
                fake_candidate(1, .045, .028, np.ones(2))]    # budget positive
        out = select_lifetime_aware(pool, np.ones(2), cfg)
        self.assertEqual(out.candidate.index, 1)
        self.assertFalse(out.scores[0]["eligible"])
        self.assertTrue(out.scores[1]["eligible"])
