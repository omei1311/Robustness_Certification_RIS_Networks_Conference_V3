"""Primary (CLARABEL) -> fallback (SCS) behaviour of StrictOracle.

Solver attempts are scripted by replacing StrictOracle._attempt, the single
seam through which every SDP solve passes. Acceptance policy (independent
eigvalsh guards) is unchanged and covered by test_oracle.
"""
import unittest
from unittest.mock import patch
from journal_sim.certification.oracle import (StrictOracle, StrictResult, STRICT_FEASIBLE,
                                              STRICT_INFEASIBLE, NUMERICALLY_UNCERTAIN)
from .helpers import scalar_case

PanicException = type("PanicException", (BaseException,), {})


def scripted(status, solver, lam=.5, raw=1e-4, normalized=1e-4, reason=""):
    return StrictResult(status, "SCRIPTED", lam, raw, normalized, (0, 0), 1.0,
                        raw, reason, solver_name=solver)


class SolverFallbackTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.w, self.H = scalar_case()

    def script(self, oracle, results):
        """results: list of StrictResult (or Exception) returned per attempt."""
        calls = []

        def fake(p, solver, fallback):
            outcome = results[len(calls)]
            calls.append((solver, fallback))
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome, []
        oracle._attempt = fake
        return calls

    def test_primary_success_skips_fallback(self):
        oracle = StrictOracle(self.cfg)
        calls = self.script(oracle, [scripted(STRICT_FEASIBLE, "CLARABEL")])
        result = oracle.check_user(self.w, self.H, .1, (0, 0))
        self.assertEqual(result.status, STRICT_FEASIBLE)
        self.assertEqual(calls, [("CLARABEL", False)])
        self.assertEqual(oracle.primary_solver_calls, 1)
        self.assertEqual(oracle.fallback_solver_calls, 0)
        self.assertEqual(len(oracle.history[0]["attempts"]), 1)

    def test_primary_inaccurate_falls_back_to_feasible(self):
        oracle = StrictOracle(self.cfg)
        calls = self.script(oracle, [scripted(NUMERICALLY_UNCERTAIN, "CLARABEL",
                                              reason="non-optimal or inaccurate solve"),
                                     scripted(STRICT_FEASIBLE, "SCS")])
        result = oracle.check_user(self.w, self.H, .1, (0, 0))
        self.assertEqual(result.status, STRICT_FEASIBLE)
        self.assertEqual(result.solver_name, "SCS")
        self.assertEqual(calls, [("CLARABEL", False), ("SCS", True)])
        self.assertEqual(oracle.fallback_solver_calls, 1)
        attempts = oracle.history[0]["attempts"]
        self.assertEqual([a["solver"] for a in attempts], ["CLARABEL", "SCS"])
        self.assertEqual([a["status"] for a in attempts], [NUMERICALLY_UNCERTAIN, STRICT_FEASIBLE])

    def test_primary_panic_escalates_to_fallback(self):
        # A Rust PanicException is converted to SOLVER_ERROR/UNCERTAIN inside
        # _attempt's catch (proven with a real raised panic in
        # test_oracle.test_rust_solver_panic_preserved). Here the same
        # converted outcome drives the fallback, which really solves and must
        # clear the independent eigvalsh guards before acceptance.
        oracle = StrictOracle(self.cfg)
        calls = self.script(oracle, [
            StrictResult(NUMERICALLY_UNCERTAIN, "SOLVER_ERROR", None, None, None, (0, 0), 1.,
                         reason="PanicException: Eigval error", solver_name="CLARABEL"),
            scripted(STRICT_FEASIBLE, "SCS"),
        ])
        result = oracle.check_user(self.w, self.H, .15, (0, 0))
        self.assertEqual(result.status, STRICT_FEASIBLE)
        self.assertEqual(calls, [("CLARABEL", False), ("SCS", True)])
        self.assertEqual(oracle.fallback_solver_calls, 1)
        attempts = oracle.history[0]["attempts"]
        self.assertEqual(attempts[0]["status"], NUMERICALLY_UNCERTAIN)
        self.assertIn("PanicException", attempts[0]["reason"])

    def test_real_scs_primary_passes_guards(self):
        # The fallback solver's own accuracy is sufficient to clear the
        # independent eigvalsh guards on this case when run as primary.
        oracle = StrictOracle(self.cfg.with_overrides(solver="SCS", fallback_solver=None))
        self.assertEqual(oracle.check_user(self.w, self.H, .15, (0, 0)).status, STRICT_FEASIBLE)

    def test_keyboard_interrupt_is_never_swallowed(self):
        oracle = StrictOracle(self.cfg)
        calls = self.script(oracle, [KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            oracle.check_user(self.w, self.H, .1, (0, 0))
        self.assertEqual(calls, [("CLARABEL", False)])
        self.assertEqual(oracle.fallback_solver_calls, 0)

    def test_fallback_uncertain_stays_uncertain(self):
        oracle = StrictOracle(self.cfg)
        self.script(oracle, [scripted(NUMERICALLY_UNCERTAIN, "CLARABEL", reason="invalid multiplier"),
                             scripted(NUMERICALLY_UNCERTAIN, "SCS",
                                      reason="independent eigvalsh did not clear positive guards")])
        result = oracle.check_user(self.w, self.H, .1, (0, 0))
        self.assertEqual(result.status, NUMERICALLY_UNCERTAIN)
        self.assertEqual(len(oracle.history[0]["attempts"]), 2)
        self.assertEqual(oracle.fallback_solver_calls, 1)

    def test_primary_infeasible_stands_without_fallback(self):
        # A decided infeasible primary result is a proof, not a numerical fluke.
        oracle = StrictOracle(self.cfg)
        calls = self.script(oracle, [scripted(STRICT_INFEASIBLE, "CLARABEL")])
        result = oracle.check_user(self.w, self.H, .1, (0, 0))
        self.assertEqual(result.status, STRICT_INFEASIBLE)
        self.assertEqual(calls, [("CLARABEL", False)])

    def test_fallback_disabled_and_same_solver_rules(self):
        disabled = StrictOracle(self.cfg.with_overrides(fallback_solver=None))
        self.script(disabled, [scripted(NUMERICALLY_UNCERTAIN, "CLARABEL")])
        self.assertEqual(disabled.check_user(self.w, self.H, .1, (0, 0)).status, NUMERICALLY_UNCERTAIN)
        self.assertEqual(disabled.fallback_solver_calls, 0)
        # Same solver configured twice must not be called twice.
        twin = StrictOracle(self.cfg.with_overrides(solver="SCS", fallback_solver="SCS"))
        self.assertEqual(twin._solver_sequence(), ["SCS"])

    def test_real_solvers_reach_consistent_boundary(self):
        # Real CLARABEL->(SCS) chain on the analytic scalar case: eps=0.15
        # feasible, eps=0.25 infeasible (boundary at 0.2).
        oracle = StrictOracle(self.cfg)
        self.assertEqual(oracle.check_user(self.w, self.H, .15, (0, 0)).status, STRICT_FEASIBLE)
        self.assertEqual(oracle.check_user(self.w, self.H, .25, (0, 0)).status, STRICT_INFEASIBLE)
        self.assertGreaterEqual(oracle.primary_solver_calls, 2)
        history = oracle.history[-1]
        self.assertIn("attempts", history)
        self.assertEqual(history["final_status"], STRICT_INFEASIBLE)
