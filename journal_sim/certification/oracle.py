"""Fast search proposes endpoints; only independent strict checks validate them.

StrictOracle tries a primary SDP solver first and falls back to a second
solver on numerical difficulty (panic, solver error, inaccurate status or a
failed independent eigenvalue guard). A solve only ever *proposes* a
multiplier; final acceptance is always the independent NumPy eigvalsh check
in validate_solution, regardless of which solver found the multiplier.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
import warnings
import numpy as np
from scipy.optimize import minimize_scalar
import cvxpy as cp
from .lmi import make_lmi

STRICT_FEASIBLE = "STRICT_FEASIBLE"
STRICT_INFEASIBLE = "STRICT_INFEASIBLE"
NUMERICALLY_UNCERTAIN = "NUMERICALLY_UNCERTAIN"


@dataclass(frozen=True)
class FastResult:
    lambda_star: float
    normalized_min_eigenvalue: float
    raw_min_eigenvalue: float
    scale: float
    feasible_fast: bool
    user_index: tuple
    search_cap_reached: bool = False


class FastOracle:
    def __init__(self, cfg):
        self.cfg = cfg.validate()
        self.calls = 0
        self.runtime = 0.0

    def check_user(self, w, H_hat, epsilon, user):
        start = perf_counter()
        self.calls += 1
        p = make_lmi(w, H_hat, epsilon, user, self.cfg)
        # epsilon=0 is a singleton, where the usual strict Slater condition
        # for the ball fails. Report its direct quadratic margin instead.
        if epsilon == 0:
            raw = p.bottom_base
            result = FastResult(0., raw / p.scale, raw, p.scale,
                                raw / p.scale >= -self.cfg.fast_tol, user)
        else:
            def margin(mu):
                return float(np.linalg.eigvalsh(p.balanced(mu))[0])
            hi = 10.0
            while hi < self.cfg.fast_lambda_cap and margin(hi) > margin(hi / 2):
                hi = min(hi * 2, self.cfg.fast_lambda_cap)
            opt = minimize_scalar(lambda mu: -margin(mu), bounds=(0., hi), method="bounded",
                                  options={"xatol": self.cfg.fast_search_tol})
            mu = max((0., float(opt.x), hi), key=margin)
            lam = p.lambda_from_mu(mu)
            normalized = margin(mu)
            result = FastResult(lam, normalized, float(np.linalg.eigvalsh(p.raw(lam))[0]),
                                p.scale, normalized >= -self.cfg.fast_tol, user,
                                hi >= self.cfg.fast_lambda_cap)
        self.runtime += perf_counter() - start
        return result

    def check(self, w, H_hat, epsilon):
        return [self.check_user(w, H_hat, epsilon, (l, k))
                for l in range(self.cfg.L) for k in range(self.cfg.K)]


@dataclass(frozen=True)
class StrictResult:
    status: str
    solver_status: str
    lambda_value: float | None
    raw_min_eig: float | None
    normalized_min_eig: float | None
    user_index: tuple
    scale: float
    raw_normalized_min_eig: float | None = None
    reason: str = ""
    solver_name: str = ""

    def to_dict(self):
        return asdict(self)


def validate_solution(p, solver_status, lambda_value, cfg, solver_name=""):
    """Public, testable post-solve check. Never trust inaccurate statuses."""
    if solver_status == cp.INFEASIBLE:
        return StrictResult(STRICT_INFEASIBLE, solver_status, None, None, None,
                            p.user_index, p.scale, reason="solver reports infeasible", solver_name=solver_name)
    if solver_status != cp.OPTIMAL:
        return StrictResult(NUMERICALLY_UNCERTAIN, str(solver_status), lambda_value, None, None,
                            p.user_index, p.scale, reason="non-optimal or inaccurate solve", solver_name=solver_name)
    if lambda_value is None or not np.isfinite(lambda_value) or lambda_value < 0:
        return StrictResult(NUMERICALLY_UNCERTAIN, solver_status, lambda_value, None, None,
                            p.user_index, p.scale, reason="invalid multiplier", solver_name=solver_name)
    raw = float(np.linalg.eigvalsh(p.raw(lambda_value))[0])
    mu = lambda_value * p.h_scale ** 2 / p.scale
    normalized = float(np.linalg.eigvalsh(p.balanced(mu))[0])
    # Positive guards deliberately spend tolerance as a safety margin.
    # Accepting small negative eigenvalues would invent a fragile radius.
    passed = raw >= cfg.strict_eig_tol and normalized >= cfg.strict_normalized_tol
    return StrictResult(STRICT_FEASIBLE if passed else NUMERICALLY_UNCERTAIN,
                        solver_status, float(lambda_value), raw, normalized, p.user_index,
                        p.scale, raw / p.scale, "independent eigvalsh passed" if passed else
                        "independent eigvalsh did not clear positive guards", solver_name)


class StrictOracle:
    def __init__(self, cfg):
        self.cfg = cfg.validate()
        self.calls = 0
        self.runtime = 0.0
        self.history = []
        self.primary_solver_calls = 0
        self.fallback_solver_calls = 0
        self.solver_error_count = 0
        self.solver_inaccurate_count = 0

    def _solver_sequence(self):
        """Primary first; a distinct configured solver is the fallback."""
        names = [self.cfg.solver]
        if self.cfg.fallback_solver is not None and self.cfg.fallback_solver != self.cfg.solver:
            names.append(self.cfg.fallback_solver)
        return names

    def _attempt(self, p, solver, fallback):
        """One SDP feasibility solve; returns (StrictResult, warning messages).

        Kept as a single method so tests can script solver behaviour and so
        that the numerical-failure policy below stays auditable in one place.
        """
        mu = cp.Variable(nonneg=True)
        d = len(p.h)
        top = p.h_scale ** 2 * p.A / p.scale + mu * np.eye(d)
        cross = (p.h_scale * p.A @ p.h / p.scale).reshape(-1, 1)
        bottom = p.bottom_base / p.scale - mu * (p.radius / p.h_scale) ** 2
        M = cp.bmat([[top, cross], [cross.conj().T, cp.reshape(bottom, (1, 1), order="C")]])
        # A feasibility solve finds an interior multiplier when one exists.
        # Optimizing the smallest eigenvalue needlessly drives the solver
        # onto repeated-eigenvalue faces and can produce inaccurate status.
        problem = cp.Problem(cp.Minimize(0), [M >> 0])
        tol = self.cfg.fallback_solver_tol if fallback else self.cfg.solver_tol
        max_iter = self.cfg.fallback_solver_max_iter if fallback else self.cfg.solver_max_iter
        options = (dict(max_iter=max_iter, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol) if solver == "CLARABEL"
                   else dict(eps=tol, max_iters=max_iter) if solver == "SCS" else {})
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                problem.solve(solver=solver, **options)
        except BaseException as exc:
            # PyO3 exposes a Rust solver panic as BaseException rather
            # than Exception. Preserve it as a numerical failure while
            # keeping cancellation/exit semantics intact.
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            if not isinstance(exc, Exception) and type(exc).__name__ != "PanicException":
                raise
            self.solver_error_count += 1
            return (StrictResult(NUMERICALLY_UNCERTAIN, "SOLVER_ERROR", None, None, None,
                                 p.user_index, p.scale,
                                 reason=f"{type(exc).__name__}: {exc}", solver_name=solver),
                    [str(x.message) for x in caught])
        if problem.status == cp.OPTIMAL_INACCURATE:
            self.solver_inaccurate_count += 1
        lam = None if mu.value is None else p.lambda_from_mu(float(mu.value))
        return validate_solution(p, problem.status, lam, self.cfg, solver), [str(x.message) for x in caught]

    def check_user(self, w, H_hat, epsilon, user):
        start = perf_counter()
        self.calls += 1
        p = make_lmi(w, H_hat, epsilon, user, self.cfg)
        attempts = []
        if epsilon == 0:
            # A radius-zero ball is checked directly, not with a degenerate SDP.
            raw, normalized = p.bottom_base, p.bottom_base / p.scale
            status = (STRICT_FEASIBLE if normalized > self.cfg.nominal_normalized_tol else
                      STRICT_INFEASIBLE if normalized < -self.cfg.nominal_normalized_tol else NUMERICALLY_UNCERTAIN)
            result = StrictResult(status, "NOMINAL_DIRECT", None, raw, normalized, user, p.scale,
                                  normalized, "singleton uncertainty set")
            attempts.append(dict(solver="NOMINAL_DIRECT", status=result.status,
                                 solver_status=result.solver_status, reason=result.reason,
                                 lambda_value=None, raw_min_eig=raw, normalized_min_eig=normalized, warnings=[]))
        else:
            result = None
            for index, solver in enumerate(self._solver_sequence()):
                fallback = index > 0
                if fallback:
                    self.fallback_solver_calls += 1
                else:
                    self.primary_solver_calls += 1
                result, solver_warnings = self._attempt(p, solver, fallback)
                attempts.append(dict(solver=solver, status=result.status,
                                     solver_status=result.solver_status, reason=result.reason,
                                     lambda_value=result.lambda_value, raw_min_eig=result.raw_min_eig,
                                     normalized_min_eig=result.normalized_min_eig, warnings=solver_warnings))
                # A decided primary result stands; only numerical uncertainty escalates.
                if result.status != NUMERICALLY_UNCERTAIN:
                    break
        self.runtime += perf_counter() - start
        self.history.append(dict(result.to_dict(), epsilon=float(epsilon), user_index=list(user),
                                 attempts=attempts, final_status=result.status))
        return result

    def check(self, w, H_hat, epsilon, fail_fast=False):
        """Check all users; fail-fast stops at the first non-feasible user.

        fail_fast=True is only for bounded-recovery search inside
        certificate_bisection. Final certificate acceptance always re-runs
        the full all-user check with fail_fast=False.
        """
        results = []
        for l in range(self.cfg.L):
            for k in range(self.cfg.K):
                r = self.check_user(w, H_hat, epsilon, (l, k))
                results.append(r)
                if fail_fast and r.status != STRICT_FEASIBLE:
                    return results
        return results
