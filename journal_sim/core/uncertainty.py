import numpy as np


def stack_users(H):
    """(BS,cell,user,antenna) -> (cell,user,BS*antenna)."""
    return np.transpose(H, (1, 2, 0, 3)).reshape(H.shape[1], H.shape[2], -1)


def channel_radii(H, epsilon, cfg):
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    return epsilon * np.maximum(np.linalg.norm(stack_users(H), axis=-1), cfg.radius_floor)


def relative_drift(current, reference, cfg):
    a, b = stack_users(current), stack_users(reference)
    return float(np.max(np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), cfg.radius_floor)))
