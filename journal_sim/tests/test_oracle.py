import unittest
import numpy as np
import cvxpy as cp
from journal_sim.certification.lmi import make_lmi, quadratic_matrix
from journal_sim.certification.oracle import FastOracle, StrictOracle, validate_solution, STRICT_FEASIBLE, STRICT_INFEASIBLE, NUMERICALLY_UNCERTAIN
from journal_sim.core.sinr import compute_sinr
from journal_sim.config import JournalConfig
from .helpers import scalar_case


class OracleTests(unittest.TestCase):
    def test_scalar_analytic_crosscheck(self):
        cfg, w, H = scalar_case()
        for eps, expect in ((.05, True), (.15, True), (.25, False)):
            fast = FastOracle(cfg).check_user(w, H, eps, (0, 0))
            strict = StrictOracle(cfg).check_user(w, H, eps, (0, 0))
            direct = abs(w.item()) ** 2 * (1 - eps) ** 2 >= cfg.noise_power
            self.assertEqual(direct, expect)
            self.assertEqual(fast.feasible_fast, expect)
            self.assertEqual(strict.status, STRICT_FEASIBLE if expect else STRICT_INFEASIBLE)
            if expect:
                p = make_lmi(w, H, eps, (0, 0), cfg)
                self.assertAlmostEqual(strict.raw_min_eig, np.linalg.eigvalsh(p.raw(strict.lambda_value))[0])

    def test_congruence_preserves_lmi(self):
        cfg, w, H = scalar_case()
        H *= 3
        p = make_lmi(w, H, .1, (0, 0), cfg)
        lam = 0.03
        D = np.diag([p.h_scale, 1])
        np.testing.assert_allclose(p.balanced(lam * p.h_scale ** 2 / p.scale), D @ p.raw(lam) @ D / p.scale)

    def test_quadratic_matches_sinr(self):
        cfg = JournalConfig(M=2, K=2, N=2).validate()
        rng = np.random.default_rng(73)
        w = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
        H = rng.normal(size=(2, 2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2, 2))
        sinr = compute_sinr(w, H, cfg)
        for l in range(2):
            for k in range(2):
                h = H[:, l, k].ravel()
                q = np.vdot(h, quadratic_matrix(w, (l, k), cfg) @ h).real - cfg.gamma * cfg.noise_power
                self.assertEqual(q >= 0, sinr[l, k] >= cfg.gamma)

    def test_inaccurate_and_false_optimal_rejected(self):
        cfg, w, H = scalar_case()
        p = make_lmi(w, H, .1, (0, 0), cfg)
        self.assertEqual(validate_solution(p, cp.OPTIMAL_INACCURATE, 1., cfg).status, NUMERICALLY_UNCERTAIN)
        self.assertEqual(validate_solution(p, cp.OPTIMAL, -1., cfg).status, NUMERICALLY_UNCERTAIN)
        self.assertEqual(validate_solution(p, cp.OPTIMAL, 0., cfg).status, NUMERICALLY_UNCERTAIN)

    def test_solver_failure_preserved(self):
        cfg, w, H = scalar_case()
        oracle = StrictOracle(cfg.with_overrides(solver="UNAVAILABLE", fallback_solver=None))
        self.assertEqual(oracle.check_user(w, H, .1, (0, 0)).status, NUMERICALLY_UNCERTAIN)
        self.assertEqual(len(oracle.history), 1)

    def test_rust_solver_panic_preserved(self):
        from unittest.mock import patch
        PanicException = type("PanicException", (BaseException,), {})
        cfg, w, H = scalar_case()
        oracle = StrictOracle(cfg)
        with patch("cvxpy.Problem.solve", side_effect=PanicException("Eigval error")):
            result = oracle.check_user(w, H, .1, (0, 0))
        self.assertEqual(result.status, NUMERICALLY_UNCERTAIN)
        self.assertIn("PanicException", result.reason)
        self.assertEqual(len(oracle.history), 1)
