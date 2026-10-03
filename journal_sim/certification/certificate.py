"""Strictly validated numerical lower bounds, with full endpoint provenance.

Recovery from a numerically uncertain endpoint uses a small fixed ladder of
conservative retreat factors plus at most `strict_refinement_steps`
deterministic bisection refinements; the search may stop at the first
failing user, but the accepted endpoint is always re-checked on every user.
NUMERICALLY_UNCERTAIN is never treated as a proof of infeasibility.
"""
from dataclasses import dataclass, asdict, field
import numpy as np
from journal_sim.config import certificate_context_fingerprint
from journal_sim.core.models import array_digest
from .oracle import FastOracle, StrictOracle, STRICT_FEASIBLE, STRICT_INFEASIBLE

VALIDATED = "VALIDATED"
LOWER_BOUND_CENSORED = "LOWER_BOUND_CENSORED"
NOMINAL_INFEASIBLE = "NOMINAL_INFEASIBLE"
NUMERICALLY_UNCERTAIN = "NUMERICALLY_UNCERTAIN"


def context_id(w, H_hat, theta, cfg):
    return array_digest(np.asarray(w, complex), np.asarray(H_hat, complex),
                        np.asarray([] if theta is None else theta, complex)) + certificate_context_fingerprint(cfg)


@dataclass
class CertificateResult:
    epsilon_cert: float = 0.0
    epsilon_lo: float = 0.0
    epsilon_hi: float = 0.0
    binding_user: tuple | None = None
    binding_margin: float | None = None
    fast_oracle_calls: int = 0
    strict_oracle_calls: int = 0
    primary_solver_calls: int = 0
    fallback_solver_calls: int = 0
    solver_error_count: int = 0
    solver_inaccurate_count: int = 0
    strict_trial_count: int = 0
    strict_validation_passed: bool = False
    status: str = NUMERICALLY_UNCERTAIN
    context_id: str = ""
    upper_endpoint_status: str = "UNVERIFIED_FAST_PROPOSAL"
    strict_validation_runtime: float = 0.0
    validation_history: list = field(default_factory=list)
    note: str = ""

    def to_dict(self):
        return asdict(self)

    def matches(self, w, H_hat, theta, cfg):
        return self.context_id == context_id(w, H_hat, theta, cfg)

    @property
    def reusable(self):
        return self.strict_validation_passed and self.status in (VALIDATED, LOWER_BOUND_CENSORED) and self.epsilon_cert > 0


def certificate_bisection(w, H_hat, cfg, theta=None, fast=None, strict=None):
    cfg.validate()
    fast = fast or FastOracle(cfg)
    strict = strict or StrictOracle(cfg)
    f0, s0, r0 = fast.calls, strict.calls, strict.runtime
    p0 = getattr(strict, "primary_solver_calls", 0)
    fb0 = getattr(strict, "fallback_solver_calls", 0)
    se0 = getattr(strict, "solver_error_count", 0)
    si0 = getattr(strict, "solver_inaccurate_count", 0)
    history0 = len(strict.history)
    result = CertificateResult(context_id=context_id(w, H_hat, theta, cfg))

    def finish():
        result.fast_oracle_calls = fast.calls - f0
        result.strict_oracle_calls = strict.calls - s0
        result.primary_solver_calls = getattr(strict, "primary_solver_calls", 0) - p0
        result.fallback_solver_calls = getattr(strict, "fallback_solver_calls", 0) - fb0
        result.solver_error_count = getattr(strict, "solver_error_count", 0) - se0
        result.solver_inaccurate_count = getattr(strict, "solver_inaccurate_count", 0) - si0
        result.strict_validation_runtime = strict.runtime - r0
        result.validation_history = strict.history[history0:]
        return result

    def strict_round(eps, fail_fast):
        # Recovery search may fail fast; final acceptance never does.
        return strict.check(w, H_hat, eps, fail_fast=fail_fast)

    nominal = strict_round(0., False)
    if any(r.status == STRICT_INFEASIBLE for r in nominal):
        result.status = NOMINAL_INFEASIBLE
        result.note = "nominal quadratic QoS fails"
        return finish()
    if not all(r.status == STRICT_FEASIBLE for r in nominal):
        result.note = "nominal QoS lies within numerical guard; no positive radius asserted"
        return finish()

    def fast_ok(eps):
        return all(r.feasible_fast for r in fast.check(w, H_hat, eps))

    hi, lo = cfg.eps_hi, 0.
    censored = False
    while fast_ok(hi):
        lo = hi
        if hi >= cfg.eps_cap:
            censored = True
            break
        hi = min(2 * hi, cfg.eps_cap)
    if not censored:
        for _ in range(cfg.certificate_max_iter):
            if hi - lo <= max(cfg.certificate_abs_tol, cfg.certificate_rel_tol * lo):
                break
            mid = (lo + hi) / 2
            if fast_ok(mid):
                lo = mid
            else:
                hi = mid
    result.epsilon_hi = hi
    # A fast upper endpoint is a search proposal, not an infeasibility proof.
    upper = strict_round(hi, False)
    result.upper_endpoint_status = (STRICT_INFEASIBLE if any(r.status == STRICT_INFEASIBLE for r in upper) else
                                    STRICT_FEASIBLE if all(r.status == STRICT_FEASIBLE for r in upper) else NUMERICALLY_UNCERTAIN)

    # Bounded conservative recovery: fixed factor ladder, then at most
    # strict_refinement_steps bisection steps between the last passing and
    # first blocked radius. Hard upper bound on strict rounds:
    # 1 nominal + 1 upper + len(factors) + refinements + 1 final acceptance.
    recovery_rounds = 0
    accepted, blocked = None, None
    for factor in cfg.strict_recovery_factors:
        trial = lo * factor
        if trial <= 0:
            break
        checks = strict_round(trial, True)
        recovery_rounds += 1
        if all(r.status == STRICT_FEASIBLE for r in checks):
            accepted = trial
            break
        blocked = trial
    if accepted is not None and blocked is not None and cfg.strict_refinement_steps > 0:
        low, high = accepted, blocked
        for _ in range(cfg.strict_refinement_steps):
            mid = (low + high) / 2
            checks = strict_round(mid, True)
            recovery_rounds += 1
            if all(r.status == STRICT_FEASIBLE for r in checks):
                low = mid
            else:
                high = mid
        accepted = low
    if accepted is not None:
        final = strict_round(accepted, False)
        recovery_rounds += 1
        if all(r.status == STRICT_FEASIBLE for r in final):
            binding = min(final, key=lambda r: r.normalized_min_eig)
            result.epsilon_cert = result.epsilon_lo = float(accepted)
            result.binding_user, result.binding_margin = binding.user_index, binding.normalized_min_eig
            result.strict_validation_passed = True
            result.status = LOWER_BOUND_CENSORED if censored and accepted == cfg.eps_cap else VALIDATED
            result.strict_trial_count = recovery_rounds
            result.note = "numerically validated conservative lower bound; inspect upper_endpoint_status before interpreting a bracket"
            return finish()
        result.note = "final full all-user validation contradicted the fail-fast recovery; no endpoint accepted"
        result.strict_trial_count = recovery_rounds
        return finish()
    result.strict_trial_count = recovery_rounds
    result.note = "no positive endpoint cleared independent strict guards within the bounded recovery budget"
    return finish()
