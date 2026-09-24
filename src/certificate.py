"""Configuration-specific robustness certificate epsilon_cert (new; Section 4).

For a FIXED configuration X = (w, theta) that is nominally feasible, the
robustness-feasibility indicator is

    F_X(eps) = 1  if  X satisfies all robust QoS constraints at the
                    relative uncertainty radius                              (10)
                    r_lk(X, eps) = eps * max(||h_hat_lk(X)||, radius_floor)
              = 0  otherwise,

and the certificate is the certified *scaling factor* of the per-user
nominal channel norms:

    epsilon_cert(X) = sup { eps >= 0 : F_X(eps) = 1 }.                    (11)

NOTE on semantics: the bisection variable is the dimensionless relative
uncertainty factor epsilon, NOT an absolute channel radius.  Each user's
actual ball radius is r_lk = eps * ||h_hat_lk||, i.e. it scales with that
user's own (Theta-dependent) nominal channel norm, matching the first
paper's uncertainty convention.  Because the balls are nested in eps,
F_X is monotone non-increasing and epsilon_cert is computed by bisection
(Eq. (12)-(13)); the returned value is the conservative lower endpoint of
the final bracket.  The oracle at every bisection step is the reused
statewise S-procedure LMI feasibility test
(``ris_base.robustness.robust_check``); no beamforming or RIS
re-optimization happens here.

``r_cert`` is kept as a backward-compatible alias for ``epsilon_cert``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .ris_base import SimConfig, compute_sinr_from_H, robust_check
from .ris_base.robustness import optimize_lambda


def robust_feasibility_indicator(
    w: np.ndarray,
    H: np.ndarray,
    epsilon: float,
    cfg: SimConfig,
) -> bool:
    """F_X(eps) in Eq. (10) via the statewise LMI test (Eq. (9)).

    ``epsilon`` is the dimensionless relative uncertainty factor; the
    per-user ball radius is eps * max(||h_hat_lk||, radius_floor).
    """
    return bool(robust_check(w, H, epsilon, cfg).feasible)


@dataclass
class CertificateResult:
    r_cert: float                     # conservative certificate (final eps_lo)
    feasible_at_zero: bool            # nominal feasibility flag F_X(0)
    n_feasibility_checks: int         # oracle calls (complexity Eq. (21)-(23))
    n_bisection_iter: int
    bracket: tuple                    # final (eps_lo, eps_hi)
    binding_user: Optional[tuple] = None      # (l, k) with the smallest margin at eps_cert
    binding_margin: Optional[float] = None
    per_user_margins: Optional[np.ndarray] = None  # (L,K) margins at eps_lo

    @property
    def epsilon_cert(self) -> float:
        """Primary name: the certified relative uncertainty scaling factor."""
        return self.r_cert


def epsilon_cert_bisection(
    w: np.ndarray,
    H: np.ndarray,
    cfg: SimConfig,
    eps_hi: float = 0.60,
    eps_hi_max: float = 1.20,
    tol: float = 1.0e-4,
    max_iter: int = 40,
) -> CertificateResult:
    """Compute epsilon_cert(X) by bisection with the statewise LMI oracle.

    The bisection variable is the relative uncertainty factor eps (each
    user's ball radius is eps * max(||h_hat_lk||, radius_floor)).  The
    bracket starts at [0, eps_hi]; the upper end is doubled (up to
    ``eps_hi_max``) until F_X is 0 there, guaranteeing a valid bracket.
    Iteration stops when eps_hi - eps_lo <= tol (Eq. (13)); the conservative
    lower endpoint is returned as the certificate.
    """
    n_checks = 0

    feasible0 = robust_feasibility_indicator(w, H, 0.0, cfg)
    n_checks += 1
    if not feasible0:
        # Only nominally feasible configurations enter certification
        # (Section 9 protocol); a nominally infeasible one gets eps_cert = 0.
        return CertificateResult(
            r_cert=0.0,
            feasible_at_zero=False,
            n_feasibility_checks=n_checks,
            n_bisection_iter=0,
            bracket=(0.0, 0.0),
        )

    # Ensure the upper endpoint is infeasible (expand if necessary).
    hi = float(eps_hi)
    while robust_feasibility_indicator(w, H, hi, cfg):
        n_checks += 1
        if hi >= eps_hi_max:
            # Still feasible at the cap: report the cap as a lower bound.
            return CertificateResult(
                r_cert=hi,
                feasible_at_zero=True,
                n_feasibility_checks=n_checks,
                n_bisection_iter=0,
                bracket=(hi, hi),
            )
        hi = min(2.0 * hi, eps_hi_max)
    n_checks += 1

    lo = 0.0
    it = 0
    while (hi - lo) > tol and it < max_iter:
        mid = 0.5 * (lo + hi)                                   # Eq. (12)
        if robust_feasibility_indicator(w, H, mid, cfg):
            lo = mid
        else:
            hi = mid
        n_checks += 1
        it += 1

    # Diagnostics at the certified endpoint: which user binds first.
    margins = np.zeros((cfg.L, cfg.K), dtype=float)
    for l in range(cfg.L):
        for k in range(cfg.K):
            _, margin = optimize_lambda(w, H, l, k, lo, cfg)
            margins[l, k] = margin
            n_checks += 0  # optimize_lambda is not a full robust_check call
    idx = np.unravel_index(np.argmin(margins), margins.shape)
    return CertificateResult(
        r_cert=float(lo),
        feasible_at_zero=True,
        n_feasibility_checks=n_checks,
        n_bisection_iter=it,
        bracket=(float(lo), float(hi)),
        binding_user=(int(idx[0]), int(idx[1])),
        binding_margin=float(margins[idx]),
        per_user_margins=margins,
    )


# Backward-compatible alias: R_cert(X) == epsilon_cert(X) under the
# relative-radius convention adopted by this framework.
r_cert_bisection = epsilon_cert_bisection


def robustness_profile(
    w: np.ndarray,
    H: np.ndarray,
    eps_grid: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """Sample the monotone robustness profile F_X(eps) on a grid.

    Used to visualize the nested feasible-radius property mentioned in
    Section 4 (indicator is 1 up to epsilon_cert and 0 beyond, up to the
    bisection tolerance).
    """
    return np.asarray(
        [robust_feasibility_indicator(w, H, float(e), cfg) for e in eps_grid],
        dtype=float,
    )


def worst_case_sinr_at_radius(
    w: np.ndarray,
    H: np.ndarray,
    epsilon: float,
    cfg: SimConfig,
    gamma_hi: float | None = None,
    gamma_lo: float = 1.0e-9,
    tol: float = 1.0e-3,
    max_iter: int = 30,
) -> float:
    """Deterministic worst-case SINR of the binding user at radius epsilon.

    Bisection on the QoS target gamma: the largest gamma_bar for which the
    statewise LMI oracle still declares X robustly feasible at radius
    epsilon.  Because the S-procedure is exact for one ball constraint, this
    equals the min over users of the worst-case SINR over the uncertainty
    balls, up to the bisection tolerance.  Used by Experiment 1 to overlay
    the deterministic degradation curve on the sampled statistics.
    """
    if gamma_hi is None:
        gamma_hi = float(np.min(compute_sinr_from_H(w, H, cfg))) * 1.05 + 1e-6

    def feasible_at(gamma: float) -> bool:
        return robust_check(w, H, epsilon, cfg.with_overrides(gamma=float(gamma))).feasible

    if not feasible_at(gamma_lo):
        return 0.0
    lo, hi = float(gamma_lo), float(gamma_hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if feasible_at(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, lo):
            break
    return float(lo)


def worst_case_sinr_curve(
    w: np.ndarray,
    H: np.ndarray,
    radii: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """Deterministic worst-case SINR evaluated on a radius grid."""
    return np.asarray(
        [worst_case_sinr_at_radius(w, H, float(r), cfg) for r in radii],
        dtype=float,
    )
