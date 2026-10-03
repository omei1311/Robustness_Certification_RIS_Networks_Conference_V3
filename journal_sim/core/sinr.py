import numpy as np


def compute_sinr(w, H, cfg):
    if np.shape(w) != (cfg.L, cfg.K, cfg.M) or np.shape(H) != (cfg.L, cfg.L, cfg.K, cfg.M):
        raise ValueError("SINR array shape mismatch")
    gains = abs(np.einsum("ilkm,ijm->lkij", np.asarray(H).conj(), w, optimize=True)) ** 2
    result = np.empty((cfg.L, cfg.K))
    for l in range(cfg.L):
        for k in range(cfg.K):
            mask = np.ones((cfg.L, cfg.K), bool)
            mask[l, k] = False
            result[l, k] = gains[l, k, l, k] / (cfg.noise_power + gains[l, k][mask].sum())
    return result


def spectral_rate(sinr):
    return float(np.log2(1 + np.asarray(sinr)).sum())


def rate(w, H, cfg):
    """bit/s; multiply bandwidth once, consistently for lifetime and energy."""
    return cfg.bandwidth_hz * spectral_rate(compute_sinr(w, H, cfg))
