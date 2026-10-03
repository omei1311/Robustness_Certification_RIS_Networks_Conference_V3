import unittest
from pathlib import Path
import numpy as np
from journal_sim.config import JournalConfig
from journal_sim.tests.helpers import tiny_config
from journal_sim.certification.certificate import certificate_bisection, CertificateResult, VALIDATED, NUMERICALLY_UNCERTAIN
from journal_sim.certification.oracle import StrictOracle, STRICT_FEASIBLE
from journal_sim.evaluation.monte_carlo import sample_unit_ball, sampled_sinr, counterexample_check


class FragileTests(unittest.TestCase):
    def test_positive_fragile_all_five_factors(self):
        cfg = tiny_config(noise_power_dbm=30, p_max_dbm=40)
        true_boundary = .0005
        H = np.ones((1, 1, 1, 1), complex)
        w = np.full((1, 1, 1), 1 / (1 - true_boundary), complex)
        result = certificate_bisection(w, H, cfg)
        self.assertEqual(result.status, VALIDATED)
        self.assertGreater(result.epsilon_cert, 0)
        self.assertLess(result.epsilon_cert, 1e-3)
        directions = sample_unit_ball(100, cfg, 41001)
        tested = []
        for alpha in (.25, .5, .9, 1., 1.1):
            eps = alpha * result.epsilon_cert
            status = StrictOracle(cfg).check_user(w, H, eps, (0, 0)).status
            worst_direct = (1 - eps) ** 2 * abs(w.item()) ** 2
            mc = sampled_sinr(w, H, eps, directions, cfg)
            if alpha <= 1:
                self.assertEqual(status, STRICT_FEASIBLE)
                self.assertGreaterEqual(worst_direct, cfg.gamma)
                self.assertTrue(np.all(mc >= cfg.gamma))
            else:
                self.assertLess(worst_direct, cfg.gamma)
                self.assertNotEqual(status, STRICT_FEASIBLE)
            tested.append(alpha)
        self.assertEqual(len(tested), 5)

    def test_real_archived_candidate26(self):
        fixture = Path(__file__).parent / "fixtures/conference_representatives.npz"
        with np.load(fixture, allow_pickle=False) as data:
            i = list(data["indices"]).index(26)
            w, H, theta = data["w"][i], data["H"][i], data["theta"][i]
        cfg = JournalConfig(include_direct_intercell=False).validate()
        result = certificate_bisection(w, H, cfg, theta)
        self.assertEqual(result.status, NUMERICALLY_UNCERTAIN)
        self.assertEqual(result.epsilon_cert, 0.)
        self.assertFalse(result.strict_validation_passed)
        old_epsilon = .0009521484374999999
        dirs = sample_unit_ball(300, cfg, 41001)
        # All five factors are evaluated even though the new radius is zero.
        for alpha in (.25, .5, .9, 1., 1.1):
            checks = StrictOracle(cfg).check(w, H, alpha * old_epsilon)
            self.assertFalse(all(x.status == STRICT_FEASIBLE for x in checks))
        violation = sampled_sinr(w, H, .125 * old_epsilon, dirs, cfg).min(axis=(1, 2)) < cfg.gamma
        self.assertGreater(int(violation.sum()), 0)

    def test_counterexample_downgrades_but_zero_does_not_prove(self):
        cfg = tiny_config(noise_power_dbm=30, p_max_dbm=40)
        w = np.ones((1, 1, 1), complex)
        H = np.ones((1, 1, 1, 1), complex)
        false = CertificateResult(epsilon_cert=.1, status=VALIDATED, strict_validation_passed=True)
        directions = -np.ones((1, 1, 1, 1), complex)
        checked, stats = counterexample_check(false, w, H, .01, directions, cfg)
        self.assertEqual(stats["violation_count"], 1)
        self.assertEqual(checked.status, NUMERICALLY_UNCERTAIN)
        self.assertFalse(checked.strict_validation_passed)
        uncertain = CertificateResult(status=NUMERICALLY_UNCERTAIN)
        checked, _ = counterexample_check(uncertain, w * 2, H, .01, directions, cfg)
        self.assertEqual(checked.status, NUMERICALLY_UNCERTAIN)
