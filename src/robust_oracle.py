"""Optional cvxpy realization of the statewise LMI oracle (new).

The default oracle everywhere is the reused eigendecomposition multiplier
search (``ris_base.robustness.robust_check``), which decides feasibility of
Eq. (9) by maximizing the normalized minimum eigenvalue over lambda >= 0.
This module provides an independent SDP formulation of the *same* LMI with
cvxpy, used by the unit tests to cross-validate the oracle and available for
spot checks in the experiments.  It is not on the default certification path
because bisection needs thousands of oracle calls.
"""

from __future__ import annotations

import numpy as np

from .ris_base import SimConfig
from .ris_base.robustness import (
    certificate_channel_vector,
    certificate_quadratic_matrix,
)
from .ris_base.uncertainty import channel_radius, uncertainty_dimension


def user_lmi_feasible_cvxpy(
    w: np.ndarray,
    H: np.ndarray,
    l: int,
    k: int,
    epsilon: float,
    cfg: SimConfig,
    solver: str = "CLARABEL",
    verbose: bool = False,
) -> bool:
    """Solve the per-user feasibility problem of Eq. (9) with cvxpy.

    Finds lambda_lk >= 0 such that
        [[A + lambda I,      A h_hat     ],
         [h_hat^H A,   h^H A h - gamma sigma^2 - lambda r^2]]  >> 0.
    """
    import cvxpy as cp

    h = certificate_channel_vector(H, l, k, cfg).reshape(-1, 1)
    A = certificate_quadratic_matrix(w, l, k, cfg)
    r = channel_radius(h, epsilon, cfg)
    dim = uncertainty_dimension(cfg)

    if cfg.uncertainty_model == "equivalent_aggregate_l2":
        inter = 0.0
    else:
        inter = float(
            sum(
                abs(np.vdot(H[n, l, k], w[n, j])) ** 2
                for n in range(w.shape[0])
                if n != l
                for j in range(w.shape[1])
            )
        )

    A_c = cp.Constant(A)
    h_c = cp.Constant(h)
    lam = cp.Variable(nonneg=True)
    top_left = A_c + lam * np.eye(dim)
    top_right = A_c @ h_c
    bottom_const = float(
        np.real((h.conj().T @ A @ h).item())
        - cfg.gamma * (inter + cfg.noise_power_watt)
    )
    bottom = cp.Constant(bottom_const) - lam * float(r**2)
    M = cp.bmat(
        [
            [top_left, top_right],
            [top_right.H, cp.reshape(bottom, (1, 1), order="C")],
        ]
    )
    prob = cp.Problem(cp.Minimize(0), [M >> 0])
    try:
        prob.solve(solver=solver, verbose=verbose)
    except cp.error.SolverError:
        return False
    return prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


def robust_feasible_cvxpy(
    w: np.ndarray,
    H: np.ndarray,
    epsilon: float,
    cfg: SimConfig,
    solver: str = "CLARABEL",
) -> bool:
    """F_X(eps) of Eq. (10) with every user's LMI solved as an SDP."""
    for l in range(cfg.L):
        for k in range(cfg.K):
            if not user_lmi_feasible_cvxpy(w, H, l, k, epsilon, cfg, solver=solver):
                return False
    return True
