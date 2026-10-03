"""Ball samples can refute numerical certificates, never establish them."""
from dataclasses import replace
import numpy as np
from journal_sim.core.channels import complex_normal
from journal_sim.core.uncertainty import stack_users, channel_radii
from journal_sim.core.sinr import compute_sinr
from journal_sim.certification.certificate import NUMERICALLY_UNCERTAIN


def sample_unit_ball(n_samples, cfg, seed):
    rng = np.random.default_rng(seed)
    shape = (n_samples, cfg.L, cfg.K, cfg.L * cfg.M)
    z = complex_normal(shape, rng)
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    radii = rng.uniform(size=shape[:-1] + (1,)) ** (1 / (2 * cfg.L * cfg.M))
    return z * radii


def sampled_sinr(w, H_hat, epsilon, directions, cfg):
    directions = np.asarray(directions)
    if directions.shape[1:] != (cfg.L, cfg.K, cfg.L * cfg.M):
        raise ValueError("Monte Carlo directions shape mismatch")
    if not np.isfinite(directions).all() or np.any(np.linalg.norm(directions, axis=-1) > 1 + 1e-12):
        raise ValueError("sample outside unit uncertainty ball")
    samples = stack_users(H_hat)[None] + channel_radii(H_hat, epsilon, cfg)[None, :, :, None] * directions
    H = samples.reshape(-1, cfg.L, cfg.K, cfg.L, cfg.M).transpose(0, 3, 1, 2, 4)
    return np.stack([compute_sinr(w, x, cfg) for x in H])


def counterexample_check(certificate, w, H_hat, epsilon, directions, cfg):
    sinrs = sampled_sinr(w, H_hat, epsilon, directions, cfg)
    worst = sinrs.min(axis=(1, 2))
    violations = int(np.count_nonzero(worst < cfg.gamma))
    checked = certificate
    if epsilon <= certificate.epsilon_cert and violations:
        checked = replace(certificate, status=NUMERICALLY_UNCERTAIN, strict_validation_passed=False,
                          note="sampled counterexample inside reported radius; original endpoint retained for audit",
                          validation_history=certificate.validation_history + [dict(evaluation_counterexamples=violations, epsilon=epsilon)])
    return checked, dict(violation_count=violations, sample_count=len(worst),
                         violation_rate=violations / len(worst), sinr_min=float(worst.min()),
                         sinr_p5=float(np.quantile(worst, .05)))
