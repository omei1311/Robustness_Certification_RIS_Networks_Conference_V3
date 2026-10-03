"""Bounded strict recovery replaces the old 40 x 0.98 shrink ladder.

Scripted oracles put a band of NUMERICALLY_UNCERTAIN radii around the
boundary (feasible strictly below, uncertain inside, infeasible above) and
verify that certificate_bisection (i) still returns a conservative positive
validated lower bound, (ii) never treats uncertainty as infeasibility, and
(iii) obeys a hard, small bound on strict rounds with the final acceptance
re-checked on all users without fail-fast.
"""
import unittest
from types import SimpleNamespace
from journal_sim.certification.certificate import certificate_bisection, VALIDATED, NUMERICALLY_UNCERTAIN
from journal_sim.certification.oracle import StrictResult, STRICT_FEASIBLE, NUMERICALLY_UNCERTAIN as UNCERTAIN
from .helpers import scalar_case


class ScriptedFast:
    def __init__(self, boundary):
        self.boundary = boundary
        self.calls = 0
        self.runtime = 0.0

    def check(self, w, H_hat, eps):
        self.calls += 1
        return [SimpleNamespace(feasible_fast=eps < self.boundary)]


class ScriptedStrict:
    """STRICT_FEASIBLE below `low`, uncertain in [low, high], infeasible above."""

    def __init__(self, low, high):
        self.low, self.high = low, high
        self.calls = 0
        self.runtime = 0.0
        self.history = []
        self.primary_solver_calls = 0
        self.fallback_solver_calls = 0
        self.solver_error_count = 0
        self.solver_inaccurate_count = 0
        self.log = []  # (epsilon, fail_fast)

    def _status(self, eps):
        if eps < self.low:
            return STRICT_FEASIBLE
        if eps <= self.high:
            return UNCERTAIN
        return "STRICT_INFEASIBLE"

    def check(self, w, H_hat, eps, fail_fast=False):
        self.calls += 1
        self.log.append((float(eps), fail_fast))
        status = self._status(eps)
        return [StrictResult(status, "SCRIPTED", .5, 1e-4, 1e-4, (0, 0), 1., 1e-4, "", "SCRIPTED")]


class StrictRecoveryTests(unittest.TestCase):
    def test_uncertain_band_yields_bounded_validated_lower_bound(self):
        cfg, w, H = scalar_case()
        fast, strict = ScriptedFast(.55), ScriptedStrict(.40, .60)
        result = certificate_bisection(w, H, cfg, fast=fast, strict=strict)
        self.assertEqual(result.status, VALIDATED)
        self.assertTrue(result.strict_validation_passed)
        # Accepted radius sits between the last passing retreat (0.70*lo) and
        # the uncertainty band; the refinements push it toward 0.40 from below.
        self.assertGreater(result.epsilon_cert, .70 * .55 * .98)
        self.assertLess(result.epsilon_cert, .40)
        # Hard bound: nominal + upper + factors + refinements + final acceptance.
        bound = 2 + len(cfg.strict_recovery_factors) + cfg.strict_refinement_steps + 1
        self.assertLessEqual(strict.calls, bound)
        self.assertLess(strict.calls, 40)  # old worst-case shrink ladder
        self.assertEqual(result.strict_trial_count, len(cfg.strict_recovery_factors[:5]) + cfg.strict_refinement_steps + 1)
        # Recovery search may fail fast; the accepted endpoint must not.
        search_flags = [fail_fast for eps, fail_fast in strict.log]
        self.assertTrue(all(search_flags[2:-1]))
        self.assertFalse(search_flags[-1])
        self.assertFalse(search_flags[0])  # nominal
        # Uncertainty near the boundary is not claimed as an infeasible bracket.
        self.assertEqual(result.upper_endpoint_status, NUMERICALLY_UNCERTAIN)

    def test_all_factors_uncertain_keeps_uncertain_status(self):
        cfg, w, H = scalar_case()
        fast, strict = ScriptedFast(.5), ScriptedStrict(.05, 10.)
        result = certificate_bisection(w, H, cfg, fast=fast, strict=strict)
        self.assertEqual(result.status, NUMERICALLY_UNCERTAIN)
        self.assertFalse(result.strict_validation_passed)
        self.assertEqual(result.epsilon_cert, 0)
        self.assertLessEqual(strict.calls, 2 + len(cfg.strict_recovery_factors))
        self.assertLess(strict.calls, 40)

    def test_immediate_pass_at_full_endpoint(self):
        # No numerical difficulty: factor 1.0 already passes, so the accepted
        # radius equals the bisection lower endpoint and is fully re-validated.
        cfg, w, H = scalar_case()
        fast, strict = ScriptedFast(.55), ScriptedStrict(.70, .80)
        result = certificate_bisection(w, H, cfg, fast=fast, strict=strict)
        self.assertEqual(result.status, VALIDATED)
        self.assertGreater(result.epsilon_cert, .54)
        self.assertLess(result.epsilon_cert, .55)
        self.assertEqual(strict.log[-1], (result.epsilon_cert, False))
        # One recovery round at factor 1.0 plus the final full re-validation.
        self.assertEqual(result.strict_trial_count, 2)
