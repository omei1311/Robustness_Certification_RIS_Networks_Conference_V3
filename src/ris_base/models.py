"""RIS phase configuration, SINR/QoS, and WEE power model (reused, no torch).

Mapping to the V3 paper:
  - phase_set / quantize_theta / random_theta ...... Eq. (1) discrete set Q_B
  - compute_sinr_from_H ........................... Eq. (2) SINR, QoS checking
  - user_weights / utility_power_wee_from_H ........ Eq. (5) WEE (U, V, U/V)
  - total_power / ris_power ........................ P_tot in Eq. (5)
  - initialize_mrt / initialize_rzf ................ beamforming initializers
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .channels import ChannelDrop, effective_channels
from .config import SimConfig


def phase_set(bits: int) -> np.ndarray:
    q = 2 ** int(bits)
    return np.exp(1j * 2.0 * np.pi * np.arange(q) / q)


def phase_indices(theta: np.ndarray, bits: int) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.complex128).reshape(-1)
    states = phase_set(bits)
    diff = np.angle(theta[:, None] * states[None, :].conj())
    return np.argmin(np.abs(diff), axis=1)


def quantize_theta(theta: np.ndarray, bits: int) -> np.ndarray:
    states = phase_set(bits)
    return states[phase_indices(theta, bits)]


def random_theta(N: int, bits: int, rng: np.random.Generator) -> np.ndarray:
    states = phase_set(bits)
    return states[rng.integers(0, len(states), size=int(N))]


def gray_code(i: int) -> int:
    return int(i) ^ (int(i) >> 1)


def physical_state_power_costs(bits: int, cfg: SimConfig) -> np.ndarray:
    """Return the physical Gray-code-dependent per-element power costs."""
    q = 2 ** int(bits)
    return np.asarray(
        [
            cfg.p_cell_idle + cfg.p_diode_on * bin(gray_code(i)).count("1")
            for i in range(q)
        ],
        dtype=float,
    )


def state_power_costs(bits: int, cfg: SimConfig) -> np.ndarray:
    """Return the RIS state costs seen by the power model."""
    physical = physical_state_power_costs(bits, cfg)
    model = str(getattr(cfg, "ris_power_model", "state_dependent"))
    if model == "state_dependent":
        return physical
    if model == "constant_average":
        return np.full_like(physical, float(np.mean(physical)))
    if model == "controller_only":
        return np.zeros_like(physical)
    raise ValueError(f"Unknown ris_power_model: {model}")


def ris_power(theta: np.ndarray, cfg: SimConfig) -> float:
    costs = state_power_costs(cfg.bits, cfg)
    return float(cfg.p_ris_controller + np.sum(costs[phase_indices(theta, cfg.bits)]))


def radiated_power(w: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(w)) ** 2))


def total_power(w: np.ndarray, theta: np.ndarray, cfg: SimConfig) -> float:
    ue_factor = cfg.K if cfg.ue_power_per_user else 1
    static = cfg.L * (cfg.p_bs + cfg.p_loss + ue_factor * cfg.p_ue)
    return float(radiated_power(w) / cfg.pa_efficiency + static + ris_power(theta, cfg))


def project_power(w: np.ndarray, p_max: float) -> np.ndarray:
    out = np.array(w, dtype=np.complex128, copy=True)
    for l in range(out.shape[0]):
        p = float(np.sum(np.abs(out[l]) ** 2))
        if p > p_max and p > 0.0:
            out[l] *= np.sqrt(p_max / p)
    return out


def compute_sinr_from_H(w: np.ndarray, H: np.ndarray, cfg: SimConfig) -> np.ndarray:
    w = np.asarray(w, dtype=np.complex128)
    out = np.zeros((cfg.L, cfg.K), dtype=float)
    for l in range(cfg.L):
        for k in range(cfg.K):
            desired = abs(np.vdot(H[l, l, k], w[l, k])) ** 2
            intra = sum(
                abs(np.vdot(H[l, l, k], w[l, j])) ** 2
                for j in range(cfg.K)
                if j != k
            )
            inter = sum(
                abs(np.vdot(H[n, l, k], w[n, j])) ** 2
                for n in range(cfg.L)
                if n != l
                for j in range(cfg.K)
            )
            out[l, k] = float(desired / max(intra + inter + cfg.noise_power_watt, 1e-15))
    return out


def compute_sinr(w: np.ndarray, theta: np.ndarray, drop: ChannelDrop, cfg: SimConfig) -> np.ndarray:
    return compute_sinr_from_H(w, effective_channels(drop, theta, cfg), cfg)


def user_weights(drop: ChannelDrop, cfg: SimConfig) -> np.ndarray:
    gamma_mat = np.full((cfg.L, cfg.K), cfg.gamma, dtype=float)
    dist = np.zeros_like(gamma_mat)
    for l in range(cfg.L):
        for k in range(cfg.K):
            dist[l, k] = np.linalg.norm(cfg.bs_positions[l] - drop.ue_positions[l, k])
    g = gamma_mat / max(float(np.sum(gamma_mat)), 1e-12)
    d = dist / max(float(np.sum(dist)), 1e-12)
    omega = cfg.omega_eta * g + (1.0 - cfg.omega_eta) * d
    return omega / max(float(np.sum(omega)), 1e-12)


def weighted_sum_rate_from_sinr(sinr: np.ndarray, omega: np.ndarray) -> float:
    return float(np.sum(omega * np.log2(1.0 + np.maximum(sinr, 0.0))))


def utility_power_wee(
    w: np.ndarray,
    theta: np.ndarray,
    drop: ChannelDrop,
    cfg: SimConfig,
    omega: np.ndarray | None = None,
) -> Tuple[float, float, float]:
    H = effective_channels(drop, theta, cfg)
    return utility_power_wee_from_H(w, theta, drop, H, cfg, omega=omega)


def utility_power_wee_from_H(
    w: np.ndarray,
    theta: np.ndarray,
    drop: ChannelDrop,
    H: np.ndarray,
    cfg: SimConfig,
    omega: np.ndarray | None = None,
) -> Tuple[float, float, float]:
    """Evaluate WEE from a caller-supplied equivalent channel tensor."""
    if omega is None:
        omega = user_weights(drop, cfg)
    u = weighted_sum_rate_from_sinr(compute_sinr_from_H(w, H, cfg), omega)
    v = total_power(w, theta, cfg)
    return u, v, float(u / max(v, 1e-15))


def initialize_mrt(
    drop: ChannelDrop,
    theta: np.ndarray,
    cfg: SimConfig,
    power_fraction: float | None = None,
) -> np.ndarray:
    H = effective_channels(drop, theta, cfg)
    frac = cfg.init_power_fraction if power_fraction is None else float(power_fraction)
    w = np.zeros((cfg.L, cfg.K, cfg.M), dtype=np.complex128)
    p_user = cfg.p_max_watt * frac / cfg.K
    for l in range(cfg.L):
        for k in range(cfg.K):
            h = H[l, l, k]
            norm = np.linalg.norm(h)
            if norm <= 1e-14:
                w[l, k, 0] = np.sqrt(p_user)
            else:
                w[l, k] = np.sqrt(p_user) * h / norm
    return project_power(w, cfg.p_max_watt)


def initialize_rzf(drop: ChannelDrop, theta: np.ndarray, cfg: SimConfig) -> np.ndarray:
    H = effective_channels(drop, theta, cfg)
    w = np.zeros((cfg.L, cfg.K, cfg.M), dtype=np.complex128)
    for l in range(cfg.L):
        A = np.column_stack([H[l, l, k] for k in range(cfg.K)])
        reg = cfg.noise_power_watt / max(cfg.p_max_watt, 1e-12)
        V = np.linalg.solve(A @ A.conj().T + reg * np.eye(cfg.M), A)
        for k in range(cfg.K):
            norm = np.linalg.norm(V[:, k])
            if norm > 1e-14:
                w[l, k] = np.sqrt(cfg.p_max_watt / cfg.K) * V[:, k] / norm
    return project_power(w, cfg.p_max_watt)
