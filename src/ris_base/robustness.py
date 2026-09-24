"""Statewise S-procedure robust feasibility (reused from first paper).

For a fixed configuration X = (w, theta), each user's robust QoS constraint
(V3 paper Eq. (4)) is the quadratic inequality Eq. (7)

    (h_hat + Delta h)^H A_lk(X) (h_hat + Delta h) - gamma_bar * sigma^2 >= 0
    for all ||Delta h||_2 <= r,

whose S-procedure equivalent (Eq. (9)) is the existence of a multiplier
lambda_lk >= 0 with

    M_lk = [[A + lambda I,       A h_hat            ],
            [h_hat^H A,   h_hat^H A h_hat - gamma_bar sigma^2 - lambda r^2]] >= 0.

``certificate_quadratic_matrix`` builds A_lk(X) in the aggregate L*M channel
ordering, ``sprocedure_matrix`` builds M_lk(lambda), and ``optimize_lambda``
maximizes the scale-normalized minimum eigenvalue over lambda >= 0, giving
the feasibility decision used as the oracle F_X(r) by the certification layer.
The scipy scalar search is the first paper's; the caching/timing wrappers are
dropped because they only served the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from .config import SimConfig
from .models import compute_sinr_from_H
from .uncertainty import channel_radius, uncertainty_dimension


@dataclass
class RobustCheck:
    penalty: float
    max_violation: float
    min_margin: float
    feasible: bool
    lambdas: np.ndarray
    nominal_min_sinr: float


def _q_matrix(w: np.ndarray, l: int, k: int, gamma: float) -> np.ndarray:
    q = np.outer(w[l, k], w[l, k].conj())
    for j in range(w.shape[1]):
        if j != k:
            q -= gamma * np.outer(w[l, j], w[l, j].conj())
    return 0.5 * (q + q.conj().T)


def certificate_channel_vector(
    H: np.ndarray,
    l: int,
    k: int,
    cfg: SimConfig,
) -> np.ndarray:
    """Return the exact ordered channel vector used by the certificate.

    Aggregate ordering is BS0(M entries),...,BS(L-1)(M entries).
    """
    if cfg.uncertainty_model == "equivalent_aggregate_l2":
        return np.asarray(H[:, l, k, :], dtype=np.complex128).reshape(
            cfg.L * cfg.M
        )
    return np.asarray(H[l, l, k], dtype=np.complex128).reshape(cfg.M)


def certificate_quadratic_matrix(
    w: np.ndarray,
    l: int,
    k: int,
    cfg: SimConfig,
) -> np.ndarray:
    """SINR quadratic form A_lk(X) in the same order as certificate_channel_vector."""
    if cfg.uncertainty_model != "equivalent_aggregate_l2":
        return _q_matrix(w, l, k, cfg.gamma)
    out = np.zeros((cfg.L * cfg.M, cfg.L * cfg.M), dtype=np.complex128)
    for n in range(cfg.L):
        block = np.zeros((cfg.M, cfg.M), dtype=np.complex128)
        if n == l:
            block += np.outer(w[n, k], w[n, k].conj())
            interferers = [j for j in range(cfg.K) if j != k]
        else:
            interferers = list(range(cfg.K))
        for j in interferers:
            block -= cfg.gamma * np.outer(w[n, j], w[n, j].conj())
        sl = slice(n * cfg.M, (n + 1) * cfg.M)
        out[sl, sl] = 0.5 * (block + block.conj().T)
    return out


def _interference(w: np.ndarray, H: np.ndarray, l: int, k: int) -> float:
    return float(
        sum(
            abs(np.vdot(H[n, l, k], w[n, j])) ** 2
            for n in range(w.shape[0])
            if n != l
            for j in range(w.shape[1])
        )
    )


def sprocedure_matrix(
    w: np.ndarray,
    H: np.ndarray,
    l: int,
    k: int,
    epsilon: float,
    lam: float,
    cfg: SimConfig,
) -> np.ndarray:
    """Assemble the LMI M_lk(X, r, lambda) of Eq. (9)."""
    h = certificate_channel_vector(H, l, k, cfg).reshape(-1, 1)
    q = certificate_quadratic_matrix(w, l, k, cfg)
    r = channel_radius(h, epsilon, cfg)
    inter = (
        0.0
        if cfg.uncertainty_model == "equivalent_aggregate_l2"
        else _interference(w, H, l, k)
    )
    top_left = q + float(lam) * np.eye(uncertainty_dimension(cfg))
    top_right = q @ h
    bottom = (
        (h.conj().T @ q @ h).item()
        - cfg.gamma * (inter + cfg.noise_power_watt)
        - float(lam) * r**2
    )
    out = np.block([[top_left, top_right], [top_right.conj().T, np.asarray([[bottom]])]])
    return 0.5 * (out + out.conj().T)


def _lmi_components(
    w: np.ndarray,
    H: np.ndarray,
    l: int,
    k: int,
    epsilon: float,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, float, complex, float, np.ndarray]:
    """Precompute the lambda-invariant terms of one user's S-procedure LMI."""
    h = certificate_channel_vector(H, l, k, cfg).reshape(-1, 1)
    q = certificate_quadratic_matrix(w, l, k, cfg)
    radius = channel_radius(h, epsilon, cfg)
    inter = (
        0.0
        if cfg.uncertainty_model == "equivalent_aggregate_l2"
        else _interference(w, H, l, k)
    )
    scale = max(
        float(np.linalg.norm(q, "fro") * (np.linalg.norm(h) ** 2 + radius**2)),
        cfg.gamma * (inter + cfg.noise_power_watt),
        1e-10,
    )
    qh = q @ h
    bottom_base = (
        (h.conj().T @ qh).item()
        - cfg.gamma * (inter + cfg.noise_power_watt)
    )
    eye = np.eye(q.shape[0], dtype=np.complex128)
    return q, qh, float(radius**2), bottom_base, float(scale), eye


def _lmi_from_components(
    q: np.ndarray,
    qh: np.ndarray,
    radius_sq: float,
    bottom_base: complex,
    eye: np.ndarray,
    lam: float,
) -> np.ndarray:
    dim = int(q.shape[0])
    out = np.empty((dim + 1, dim + 1), dtype=np.complex128)
    out[:dim, :dim] = q + float(lam) * eye
    out[:dim, dim:] = qh
    out[dim:, :dim] = qh.conj().T
    out[dim, dim] = bottom_base - float(lam) * radius_sq
    return 0.5 * (out + out.conj().T)


def optimize_lambda(
    w: np.ndarray,
    H: np.ndarray,
    l: int,
    k: int,
    epsilon: float,
    cfg: SimConfig,
) -> Tuple[float, float]:
    """Maximize the minimum normalized LMI eigenvalue over lambda >= 0."""
    if epsilon <= 1e-14:
        sinr = compute_sinr_from_H(w, H, cfg)[l, k]
        return 0.0, float(sinr / cfg.gamma - 1.0)

    q, qh, radius_sq, bottom_base, scale, eye = _lmi_components(
        w, H, l, k, epsilon, cfg
    )
    q_norm = np.linalg.norm(q, 2)
    upper = max(10.0 * q_norm, 10.0, 1e-6)

    def margin_at(lam: float) -> float:
        matrix = _lmi_from_components(
            q, qh, radius_sq, bottom_base, eye, lam
        )
        return float(np.min(np.linalg.eigvalsh(matrix / scale)))

    # Expand the search interval when the margin is still improving at the
    # upper endpoint. This avoids falsely declaring infeasibility for small
    # uncertainty radii, where the optimal multiplier can be relatively large.
    previous = margin_at(upper / 10.0)
    current = margin_at(upper)
    for _ in range(5):
        if current <= previous + 1e-8 or upper >= 1e7:
            break
        previous = current
        upper *= 10.0
        current = margin_at(upper)

    def score_log(log_lam: float) -> float:
        return -margin_at(float(np.exp(log_lam)))

    lo, hi = np.log(1e-12), np.log(upper)
    result = minimize_scalar(score_log, bounds=(lo, hi), method="bounded", options={"xatol": 1e-5})
    candidates = [0.0, float(np.exp(result.x)), upper]
    best_lam, best_margin = 0.0, -np.inf
    for lam in candidates:
        margin = margin_at(lam)
        if margin > best_margin:
            best_lam, best_margin = lam, margin
    return best_lam, best_margin


def robust_check(
    w: np.ndarray,
    H: np.ndarray,
    epsilon: float,
    cfg: SimConfig,
) -> RobustCheck:
    """Statewise robust feasibility F_X(eps) for all users (Section 3)."""
    nominal_sinr = compute_sinr_from_H(w, H, cfg)
    lambdas = np.zeros((cfg.L, cfg.K), dtype=float)
    violations = []
    signed_margins = []
    for l in range(cfg.L):
        for k in range(cfg.K):
            lam, margin = optimize_lambda(w, H, l, k, epsilon, cfg)
            lambdas[l, k] = lam
            signed_margins.append(float(margin))
            violations.append(max(0.0, -margin))
    arr = np.asarray(violations, dtype=float)
    penalty = float(np.sum(arr**2))
    max_violation = float(np.max(arr)) if arr.size else 0.0
    min_margin = float(np.min(np.asarray(signed_margins, dtype=float))) if signed_margins else 0.0
    return RobustCheck(
        penalty=penalty,
        max_violation=max_violation,
        min_margin=min_margin,
        feasible=bool(max_violation <= cfg.feasibility_tol),
        lambdas=lambdas,
        nominal_min_sinr=float(np.min(nominal_sinr)),
    )
