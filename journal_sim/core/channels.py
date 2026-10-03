"""Rician links with V3 path loss; ablation masks composition, not RNG draws."""
from dataclasses import dataclass, replace
import numpy as np


def complex_normal(shape, rng):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


@dataclass(frozen=True)
class PhysicalChannel:
    h_bu: np.ndarray
    h_br: np.ndarray
    h_ru: np.ndarray
    ue_positions: np.ndarray
    mean_bu: np.ndarray
    mean_br: np.ndarray
    mean_ru: np.ndarray
    std_bu: np.ndarray
    std_br: np.ndarray
    std_ru: np.ndarray

    def with_links(self, **kwargs):
        return replace(self, **kwargs)


def generate_channel(cfg, seed):
    cfg.validate()
    rng = np.random.default_rng(seed)
    bs = np.column_stack((cfg.bs_x, cfg.bs_y))
    centers = np.column_stack((cfg.ue_center_x, cfg.ue_center_y))
    ris = np.array((cfg.ris_x, cfg.ris_y))
    positions = np.empty((cfg.L, cfg.K, 2))
    for l in range(cfg.L):
        for k in range(cfg.K):
            angle = rng.uniform(0, 2 * np.pi)
            radius = cfg.ue_radius * np.sqrt(rng.uniform())
            positions[l, k] = centers[l] + radius * np.array((np.cos(angle), np.sin(angle)))
    def loss(distance, alpha):
        return 10 ** (cfg.c0_db / 10) * max(float(distance), 1) ** (-alpha)
    def steering(n, angle):
        return np.exp(1j * np.pi * np.arange(n) * np.sin(angle))
    aperture = 1 if cfg.ris_size_model == "growing_aperture" else np.sqrt(cfg.ris_reference_N / cfg.N)
    ris_scale = np.sqrt(cfg.channel_scale) * aperture
    br_mean = np.empty((cfg.L, cfg.N, cfg.M), complex)
    br_std = np.empty((cfg.L, 1, 1))
    br = np.empty_like(br_mean)
    ru_mean = np.empty((cfg.L, cfg.K, cfg.N), complex)
    ru_std = np.empty((cfg.L, cfg.K, 1))
    ru = np.empty_like(ru_mean)
    bu_std = np.empty((cfg.L, cfg.L, cfg.K, 1))
    bu = np.empty((cfg.L, cfg.L, cfg.K, cfg.M), complex)
    kf = cfg.rician_k
    for i in range(cfg.L):
        amp = ris_scale * np.sqrt(loss(np.linalg.norm(bs[i] - ris), cfg.alpha_br))
        los = np.outer(steering(cfg.N, rng.uniform(-np.pi / 2, np.pi / 2)),
                       steering(cfg.M, rng.uniform(-np.pi / 2, np.pi / 2)).conj())
        br_mean[i] = amp * np.sqrt(kf / (kf + 1)) * los
        br_std[i] = amp / np.sqrt(kf + 1)
        br[i] = br_mean[i] + br_std[i] * complex_normal((cfg.N, cfg.M), rng)
    for l in range(cfg.L):
        for k in range(cfg.K):
            amp = ris_scale * np.sqrt(loss(np.linalg.norm(positions[l, k] - ris), cfg.alpha_ru))
            ru_mean[l, k] = amp * np.sqrt(kf / (kf + 1)) * steering(cfg.N, rng.uniform(-np.pi / 2, np.pi / 2))
            ru_std[l, k] = amp / np.sqrt(kf + 1)
            ru[l, k] = ru_mean[l, k] + ru_std[l, k] * complex_normal((cfg.N,), rng)
            for i in range(cfg.L):
                bu_std[i, l, k] = cfg.channel_scale * cfg.direct_link_attenuation * np.sqrt(loss(np.linalg.norm(bs[i] - positions[l, k]), cfg.alpha_bu))
                bu[i, l, k] = bu_std[i, l, k] * complex_normal((cfg.M,), rng)
    return PhysicalChannel(bu, br, ru, positions, np.zeros_like(bu), br_mean, ru_mean,
                           bu_std, br_std, ru_std)


def effective_channels(channel, theta, cfg):
    theta = np.asarray(theta)
    if theta.shape != (cfg.N,):
        raise ValueError("theta shape mismatch")
    reflected = np.einsum("inm,lkn,n->ilkm", channel.h_br.conj(), channel.h_ru, theta.conj(), optimize=True)
    factors = np.full((cfg.L, cfg.L), cfg.inter_ris_attenuation)
    np.fill_diagonal(factors, 1)
    direct = np.array(channel.h_bu, copy=True)
    if not cfg.include_direct_intercell:
        direct[np.arange(cfg.L)[:, None] != np.arange(cfg.L)[None, :]] = 0
    return direct + reflected * factors[:, :, None, None]
