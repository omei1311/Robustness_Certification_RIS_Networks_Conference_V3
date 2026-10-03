import unittest
from journal_sim.certification.certificate import certificate_bisection, VALIDATED, LOWER_BOUND_CENSORED, NOMINAL_INFEASIBLE, NUMERICALLY_UNCERTAIN
from journal_sim.certification.oracle import StrictOracle, STRICT_FEASIBLE
from .helpers import scalar_case


class CertificateTests(unittest.TestCase):
    def test_strict_lower_endpoint(self):
        cfg, w, H = scalar_case()
        result = certificate_bisection(w, H, cfg)
        self.assertEqual(result.status, VALIDATED)
        self.assertTrue(result.strict_validation_passed)
        self.assertLessEqual(result.epsilon_cert, .2)
        self.assertGreater(result.epsilon_cert, .19)
        self.assertTrue(all(r.status == STRICT_FEASIBLE for r in StrictOracle(cfg).check(w, H, result.epsilon_cert)))
        self.assertGreater(result.strict_oracle_calls, 1)

    def test_monotonicity(self):
        cfg, w, H = scalar_case()
        statuses = [StrictOracle(cfg).check_user(w, H, x, (0, 0)).status == STRICT_FEASIBLE for x in (.01, .05, .1, .15, .25, .3)]
        self.assertEqual(statuses, sorted(statuses, reverse=True))

    def test_censored_and_nominal_infeasible(self):
        cfg, w, H = scalar_case()
        result = certificate_bisection(w, H, cfg.with_overrides(eps_hi=.01, eps_cap=.02))
        self.assertEqual(result.status, LOWER_BOUND_CENSORED)
        self.assertEqual(result.epsilon_cert, .02)
        self.assertTrue(result.strict_validation_passed)
        self.assertEqual(certificate_bisection(w * .01, H, cfg).status, NOMINAL_INFEASIBLE)

    def test_uncertainty_never_masquerades_as_certificate(self):
        cfg, w, H = scalar_case()
        result = certificate_bisection(w, H, cfg.with_overrides(solver="UNAVAILABLE", fallback_solver=None,
                                                                strict_recovery_factors=(1.0,), strict_refinement_steps=0))
        self.assertEqual(result.status, NUMERICALLY_UNCERTAIN)
        self.assertFalse(result.strict_validation_passed)
        self.assertEqual(result.epsilon_cert, 0)

    def test_identity_binds_w_theta_channel_and_config(self):
        import numpy as np
        cfg, w, H = scalar_case()
        theta = np.ones(2)
        result = certificate_bisection(w, H, cfg, theta)
        self.assertTrue(result.matches(w, H, theta, cfg))
        self.assertFalse(result.matches(w * (1 + 1e-12), H, theta, cfg))
        self.assertFalse(result.matches(w, H * 1.01, theta, cfg))
        self.assertFalse(result.matches(w, H, -theta, cfg))
