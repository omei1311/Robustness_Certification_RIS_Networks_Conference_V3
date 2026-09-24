"""Uncertainty-radius conventions and ball sampling (reused from first paper).

The per-user uncertainty ball of the V3 paper, ||Delta h_lk|| <= r, is
realized here either as an absolute radius (relative_radius=False) or, by
default exactly as in the first paper, as a *relative* radius
``r_lk = eps * ||h_lk||`` (relative_radius=True).  All certification results
(R_cert, the prescribed design radius epsilon, and the drift-rate bound nu)
are expressed in this same convention.
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig


def uncertainty_dimension(cfg: SimConfig) -> int:
    """Return the complex dimension certified for each user."""
    if cfg.uncertainty_model == "equivalent_aggregate_l2":
        return int(cfg.L * cfg.M)
    return int(cfg.M)


def uncertainty_direction_shape(cfg: SimConfig) -> tuple[int, int, int]:
    """Shape of one Monte Carlo direction array without its sample axis."""
    return int(cfg.L), int(cfg.K), uncertainty_dimension(cfg)


def sample_complex_unit_ball(
    n_samples: int,
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform directions/radii in a complex Euclidean unit ball.

    The returned array has shape ``(n_samples, *shape)``.  The final dimension
    is treated as the channel-vector dimension and is normalized jointly.
    """
    z = (
        rng.standard_normal((n_samples, *shape))
        + 1j * rng.standard_normal((n_samples, *shape))
    ) / np.sqrt(2.0)
    d = int(shape[-1])
    norms = np.linalg.norm(z, axis=-1, keepdims=True)
    z = z / np.maximum(norms, 1e-15)
    # A complex d-vector has 2d real dimensions.
    radii = rng.uniform(0.0, 1.0, size=(n_samples, *shape[:-1], 1)) ** (1.0 / (2.0 * d))
    return z * radii


def channel_radius(h_hat: np.ndarray, epsilon: float, cfg: SimConfig) -> float:
    if cfg.relative_radius:
        return float(epsilon) * max(float(np.linalg.norm(h_hat)), cfg.radius_floor)
    return max(float(epsilon), cfg.radius_floor if epsilon > 0 else 0.0)
