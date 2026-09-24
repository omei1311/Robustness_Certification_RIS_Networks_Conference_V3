"""Channel generation (reused from the first paper, torch helpers removed).

One ``ChannelDrop`` holds a physical realization: direct BS-UE links,
BS-RIS links, and RIS-UE links, all with the first paper's path-loss and
Rician small-scale model.  ``effective_channels`` composes them through the
RIS reflection vector theta into the equivalent channels H[n, l, k, :]
(BS n -> user (l,k)) used everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import SimConfig


@dataclass
class ChannelDrop:
    """One physical channel realization.

    Shapes
    ------
    h_bu : (L,L,K,M), BS n -> UE (l,k)
    h_br : (L,N,M), BS n -> RIS
    h_ru : (L,K,N), RIS -> UE (l,k)
    ue_positions : (L,K,2)
    """

    h_bu: np.ndarray
    h_br: np.ndarray
    h_ru: np.ndarray
    ue_positions: np.ndarray

    @property
    def L(self) -> int:
        return int(self.h_br.shape[0])

    @property
    def N(self) -> int:
        return int(self.h_br.shape[1])

    @property
    def M(self) -> int:
        return int(self.h_br.shape[2])

    @property
    def K(self) -> int:
        return int(self.h_ru.shape[1])


def complex_normal(shape, rng: np.random.Generator) -> np.ndarray:
    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) / np.sqrt(2.0)


def steering_vector(n: int, angle: float) -> np.ndarray:
    idx = np.arange(n, dtype=float)
    return np.exp(1j * np.pi * idx * np.sin(angle))


def pathloss(distance: float, alpha: float, cfg: SimConfig) -> float:
    distance = max(float(distance), 1.0)
    c0 = 10.0 ** (cfg.c0_db / 10.0)
    return c0 * distance ** (-float(alpha))


def _sample_user_positions(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((cfg.L, cfg.K, 2), dtype=float)
    centers = cfg.ue_centers
    for l in range(cfg.L):
        for k in range(cfg.K):
            angle = rng.uniform(0.0, 2.0 * np.pi)
            radius = cfg.ue_radius * np.sqrt(rng.uniform(0.0, 1.0))
            out[l, k] = centers[l] + radius * np.array(
                [np.cos(angle), np.sin(angle)]
            )
    return out


def _ris_aperture_scale(N: int, cfg: SimConfig) -> float:
    if cfg.ris_size_model == "growing_aperture":
        return 1.0
    return np.sqrt(float(cfg.ris_reference_N) / float(N))


def generate_channel_drop(
    cfg: SimConfig,
    rng: np.random.Generator,
    N: Optional[int] = None,
) -> ChannelDrop:
    N = int(cfg.N if N is None else N)
    L, K, M = cfg.L, cfg.K, cfg.M
    scale = float(cfg.channel_scale)
    ris_scale = np.sqrt(scale) * _ris_aperture_scale(N, cfg)
    direct_scale = scale * float(cfg.direct_serving_attenuation)

    ue_positions = _sample_user_positions(cfg, rng)
    h_bu = np.zeros((L, L, K, M), dtype=np.complex128)
    h_br = np.zeros((L, N, M), dtype=np.complex128)
    h_ru = np.zeros((L, K, N), dtype=np.complex128)

    for n in range(L):
        d_br = np.linalg.norm(cfg.bs_positions[n] - cfg.ris_position)
        pl = pathloss(d_br, cfg.alpha_br, cfg)
        aoa = rng.uniform(-np.pi / 2.0, np.pi / 2.0)
        aod = rng.uniform(-np.pi / 2.0, np.pi / 2.0)
        los = np.outer(steering_vector(N, aoa), steering_vector(M, aod).conj())
        nlos = complex_normal((N, M), rng)
        kf = cfg.rician_k
        h_br[n] = ris_scale * np.sqrt(pl) * (
            np.sqrt(kf / (kf + 1.0)) * los
            + np.sqrt(1.0 / (kf + 1.0)) * nlos
        )

    for l in range(L):
        for k in range(K):
            pos = ue_positions[l, k]
            d_ru = np.linalg.norm(cfg.ris_position - pos)
            pl_ru = pathloss(d_ru, cfg.alpha_ru, cfg)
            angle = rng.uniform(-np.pi / 2.0, np.pi / 2.0)
            los = steering_vector(N, angle)
            nlos = complex_normal((N,), rng)
            kf = cfg.rician_k
            h_ru[l, k] = ris_scale * np.sqrt(pl_ru) * (
                np.sqrt(kf / (kf + 1.0)) * los
                + np.sqrt(1.0 / (kf + 1.0)) * nlos
            )

            for n in range(L):
                if n != l and not cfg.include_direct_intercell:
                    continue
                d_bu = np.linalg.norm(cfg.bs_positions[n] - pos)
                pl_bu = pathloss(d_bu, cfg.alpha_bu, cfg)
                h_bu[n, l, k] = (
                    direct_scale * np.sqrt(pl_bu) * complex_normal((M,), rng)
                )

    drop = ChannelDrop(h_bu=h_bu, h_br=h_br, h_ru=h_ru, ue_positions=ue_positions)
    validate_drop(drop, cfg)
    return drop


def validate_drop(drop: ChannelDrop, cfg: SimConfig | None = None) -> None:
    L, N, M = drop.h_br.shape
    if drop.h_bu.shape != (L, L, drop.K, M):
        raise ValueError(f"invalid h_bu shape {drop.h_bu.shape}")
    if drop.h_ru.shape != (L, drop.K, N):
        raise ValueError(f"invalid h_ru shape {drop.h_ru.shape}")
    if drop.ue_positions.shape != (L, drop.K, 2):
        raise ValueError(f"invalid UE position shape {drop.ue_positions.shape}")
    if cfg is not None and (L != cfg.L or M != cfg.M or drop.K != cfg.K):
        raise ValueError("channel dimensions disagree with configuration")


def without_ris(drop: ChannelDrop) -> ChannelDrop:
    """Return a direct-link-only copy used by No-RIS references."""
    return ChannelDrop(
        h_bu=np.array(drop.h_bu, copy=True),
        h_br=np.zeros_like(drop.h_br),
        h_ru=np.zeros_like(drop.h_ru),
        ue_positions=np.array(drop.ue_positions, copy=True),
    )


def effective_channel_ris_terms(
    drop: ChannelDrop, cfg: SimConfig
) -> np.ndarray:
    """Return each RIS element's additive contribution to ``H``.

    The result has shape ``(N,L,L,K,M)``: element i contributes
    ``conj(h_br[n,i,:]) * h_ru[l,k,i]`` to the BS n -> UE (l,k) channel.
    """
    terms = np.einsum(
        "nim,lki->inlkm",
        np.asarray(drop.h_br, dtype=np.complex128).conj(),
        np.asarray(drop.h_ru, dtype=np.complex128),
        optimize=True,
    )
    factors = np.full(
        (drop.L, drop.L), float(cfg.inter_ris_attenuation), dtype=float
    )
    np.fill_diagonal(factors, 1.0)
    return terms * factors[None, :, :, None, None]


def effective_channels(drop: ChannelDrop, theta: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Return H[n,l,k,:], the equivalent channel from BS n to UE (l,k)."""
    theta = np.asarray(theta, dtype=np.complex128).reshape(drop.N)
    terms = effective_channel_ris_terms(drop, cfg)
    cascaded = np.tensordot(theta.conj(), terms, axes=(0, 0))
    return np.asarray(drop.h_bu, dtype=np.complex128) + cascaded
