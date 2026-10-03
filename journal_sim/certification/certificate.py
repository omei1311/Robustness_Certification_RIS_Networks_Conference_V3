"""Strictly validated numerical lower bounds, with full endpoint provenance."""
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
    history0 = len(strict.history)
    result = CertificateResult(context_id=context_id(w, H_hat, theta, cfg))

    def finish():
        result.fast_oracle_calls = fast.calls - f0
        result.strict_oracle_calls = strict.calls - s0
        result.strict_validation_runtime = strict.runtime - r0
        result.validation_history = strict.history[history0:]
        return result

    nominal = strict.check(w, H_hat, 0.)
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
    upper = strict.check(w, H_hat, hi)
    result.upper_endpoint_status = (STRICT_INFEASIBLE if any(r.status == STRICT_INFEASIBLE for r in upper) else
                                    STRICT_FEASIBLE if all(r.status == STRICT_FEASIBLE for r in upper) else NUMERICALLY_UNCERTAIN)
    trial = lo
    for _ in range(cfg.strict_shrink_steps):
        if trial <= 0:
            break
        checks = strict.check(w, H_hat, trial)
        if all(r.status == STRICT_FEASIBLE for r in checks):
            binding = min(checks, key=lambda r: r.normalized_min_eig)
            result.epsilon_cert = result.epsilon_lo = float(trial)
            result.binding_user, result.binding_margin = binding.user_index, binding.normalized_min_eig
            result.strict_validation_passed = True
            result.status = LOWER_BOUND_CENSORED if censored and trial == cfg.eps_cap else VALIDATED
            result.note = "numerically validated conservative lower bound; inspect upper_endpoint_status before interpreting a bracket"
            return finish()
        trial *= cfg.strict_shrink_factor
    result.note = "no positive endpoint cleared independent strict guards within shrink budget"
    return finish()
