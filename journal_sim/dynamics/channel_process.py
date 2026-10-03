"""Stationary Gauss-Markov fading about the fixed Rician LOS mean."""
import numpy as np
from journal_sim.core.channels import generate_channel, complex_normal
from journal_sim.core.models import array_digest


def evolve_channel(channel, cfg, rng):
    rho = cfg.correlation
    updates = {}
    for name, mean_name, std_name in (("h_bu", "mean_bu", "std_bu"),
                                      ("h_ru", "mean_ru", "std_ru"),
                                      ("h_br", "mean_br", "std_br")):
        if name == "h_br" and cfg.quasi_static_br:
            continue
        mean = getattr(channel, mean_name)
        current = getattr(channel, name)
        std = getattr(channel, std_name)
        updates[name] = mean + rho * (current - mean) + np.sqrt(1 - rho ** 2) * std * complex_normal(current.shape, rng)
    return channel.with_links(**updates)


def channel_trajectory(cfg, seed):
    initial = generate_channel(cfg, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1701]))
    trajectory = [initial]
    for _ in range(1, cfg.time_steps):
        trajectory.append(evolve_channel(trajectory[-1], cfg, rng))
    return tuple(trajectory)


def trajectory_id(trajectory):
    return array_digest(*(getattr(c, name) for c in trajectory for name in ("h_bu", "h_br", "h_ru")))
