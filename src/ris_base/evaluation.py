"""Monte Carlo SINR evaluation under norm-bounded channel errors (reused).

The first paper's ``evaluate_solution`` reduced to the parts the
certification layer needs: per-user sampled SINRs when the aggregate
equivalent channel is perturbed inside its uncertainty ball.  This is the
empirical companion of the deterministic LMI oracle used by Experiment 1
(certificate validation); it never replaces the certificate computation.
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig
from .uncertainty import channel_radius, uncertainty_dimension


def sinr_under_error_samples(
    w: np.ndarray,
    H: np.ndarray,
    epsilon: float,
    error_directions: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """Sampled per-user SINRs for perturbed channels h_hat + Delta h.

    Parameters
    ----------
    w : (L,K,M) beamformers of the fixed configuration.
    H : (L,L,K,M) nominal equivalent channels.
    epsilon : uncertainty radius in the ``relative_radius`` convention.
    error_directions : (S,L,K,dim) samples from the complex unit ball, where
        dim = uncertainty_dimension(cfg); each is scaled to the user's radius.

    Returns
    -------
    sinr : (S,L,K) sampled SINRs.
    """
    w = np.asarray(w, dtype=np.complex128)
    H = np.asarray(H, dtype=np.complex128)
    S = int(error_directions.shape[0])
    expected_dim = uncertainty_dimension(cfg)
    expected_shape = (S, cfg.L, cfg.K, expected_dim)
    if error_directions.shape != expected_shape:
        raise ValueError(
            "error_directions do not match the certificate model: "
            f"got {error_directions.shape}, expected {expected_shape}"
        )

    sinr = np.zeros((S, cfg.L, cfg.K), dtype=float)
    for l in range(cfg.L):
        for k in range(cfg.K):
            if cfg.uncertainty_model == "equivalent_aggregate_l2":
                h0 = H[:, l, k, :].reshape(cfg.L * cfg.M)
                radius = channel_radius(h0, epsilon, cfg)
                h_samples = (
                    h0[None, :] + radius * error_directions[:, l, k, :]
                )
                blocks = h_samples.reshape(S, cfg.L, cfg.M)
                desired_amps = blocks[:, l, :].conj() @ w[l].T
                signal = np.abs(desired_amps[:, k]) ** 2
                intra = np.sum(np.abs(desired_amps) ** 2, axis=1) - signal
                inter = np.zeros(S, dtype=float)
                for n in range(cfg.L):
                    if n == l:
                        continue
                    amps_n = blocks[:, n, :].conj() @ w[n].T
                    inter += np.sum(np.abs(amps_n) ** 2, axis=1)
            else:
                h0 = H[l, l, k]
                radius = channel_radius(h0, epsilon, cfg)
                h_samples = (
                    h0[None, :] + radius * error_directions[:, l, k, :]
                )
                # np.vdot is not batched; h_samples.conj() @ w[l].T is.
                amps = h_samples.conj() @ w[l].T
                signal = np.abs(amps[:, k]) ** 2
                intra = np.sum(np.abs(amps) ** 2, axis=1) - signal
                inter = sum(
                    abs(np.vdot(H[n, l, k], w[n, j])) ** 2
                    for n in range(cfg.L)
                    if n != l
                    for j in range(cfg.K)
                )
            sinr[:, l, k] = signal / np.maximum(
                intra + inter + cfg.noise_power_watt, 1e-15
            )
    return sinr
