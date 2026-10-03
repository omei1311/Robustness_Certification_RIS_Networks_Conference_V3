"""One estimated physical snapshot; nominal linear power control + certification.

This is a transparent candidate generator, not a globally optimal robust WEE
solver. All failed attempts and all accepted/dominated candidates are retained.
"""
from time import perf_counter
import numpy as np
from journal_sim.config import config_fingerprint
from journal_sim.core.channels import effective_channels
from journal_sim.core.models import Configuration, phase_set
from journal_sim.core.sinr import compute_sinr, rate
from journal_sim.core.power import system_power
from journal_sim.certification.certificate import certificate_bisection
from .candidate import Candidate


def zero_forcing_directions(H_hat, cfg):
    directions = np.empty((cfg.L, cfg.K, cfg.M), complex)
    for i in range(cfg.L):
        rows = H_hat[i].reshape(cfg.L * cfg.K, cfg.M).conj()
        inverse = np.linalg.pinv(rows, rcond=cfg.zf_rcond)
        for k in range(cfg.K):
            v = inverse[:, i * cfg.K + k]
            norm = np.linalg.norm(v)
            if norm <= cfg.zf_rcond:
                raise ValueError("degenerate ZF direction")
            directions[i, k] = v / norm
    return directions


def nominal_power_control(H_hat, directions, gamma_design, cfg):
    amplitudes = abs(np.einsum("ilkm,ijm->lkij", H_hat.conj(), directions, optimize=True)) ** 2
    d = cfg.L * cfg.K
    gains = amplitudes.reshape(d, d)
    desired = np.diag(gains)
    if np.any(desired <= 0):
        raise ValueError("zero desired channel gain")
    coupling = gamma_design * gains / desired[:, None]
    np.fill_diagonal(coupling, 0.)
    if np.max(abs(np.linalg.eigvals(coupling))) >= 1:
        raise ValueError("power control interference limited")
    rhs = gamma_design * cfg.noise_power / desired
    powers = np.linalg.solve(np.eye(d) - coupling, rhs)
    if np.any(powers < 0) or not np.isfinite(powers).all():
        raise ValueError("nonfinite or negative powers")
    if np.linalg.norm((np.eye(d) - coupling) @ powers - rhs) > cfg.power_control_tol * max(np.linalg.norm(rhs), 1):
        raise ValueError("power control residual")
    return powers.reshape(cfg.L, cfg.K)


def candidate_seed(cfg, channel_seed, time_index, attempt):
    # Same seed at the same time for every policy, independent of prior calls.
    return int(np.random.SeedSequence([cfg.pool_seed, channel_seed, time_index, attempt]).generate_state(1)[0])


def build_candidate_pool(estimated_channel, cfg, channel_seed, time_index=0):
    start = perf_counter()
    pool, attempts = [], []
    generation_time = certificate_time = strict_time = 0.
    for attempt in range(cfg.pool_attempts):
        if len(pool) >= cfg.pool_size:
            break
        seed = candidate_seed(cfg, channel_seed, time_index, attempt)
        record = dict(attempt=attempt, candidate_seed=seed, seed=channel_seed,
                      time_index=time_index, config_fingerprint=config_fingerprint(cfg))
        generation_start = perf_counter()
        try:
            rng = np.random.default_rng(seed)
            theta = np.ones(cfg.N, complex) if attempt == 0 else phase_set(cfg.bits)[rng.integers(2 ** cfg.bits, size=cfg.N)]
            H_hat = effective_channels(estimated_channel, theta, cfg)
            directions = zero_forcing_directions(H_hat, cfg)
            gamma_mult = float(rng.choice(cfg.design_gamma_mult))
            slack = float(rng.choice(cfg.power_slack_grid))
            powers = nominal_power_control(H_hat, directions, cfg.gamma * gamma_mult, cfg) * slack
            X = Configuration(directions * np.sqrt(powers)[:, :, None], theta).validate(cfg)
            sinr_min = float(compute_sinr(X.w, H_hat, cfg).min())
            nominal_ok = sinr_min >= cfg.gamma
            power = system_power(X.w, theta, cfg).total
            nominal_rate = rate(X.w, H_hat, cfg)
            generation_time += perf_counter() - generation_start
            cert_start = perf_counter()
            cert = certificate_bisection(X.w, H_hat, cfg, theta)
            cert_elapsed = perf_counter() - cert_start
            certificate_time += cert_elapsed
            strict_time += cert.strict_validation_runtime
            candidate = Candidate(len(pool), seed, X, H_hat, nominal_rate, power,
                                  nominal_rate / power, sinr_min, nominal_ok, cert,
                                  metadata=dict(cfg=cfg, seed=channel_seed, time_index=time_index,
                                                gamma_design=cfg.gamma * gamma_mult, power_slack=slack,
                                                config_fingerprint=config_fingerprint(cfg), certificate_runtime=cert_elapsed))
            pool.append(candidate)
            record.update(status=cert.status, candidate_index=candidate.index,
                          configuration_id=X.configuration_id, nominal_feasible=nominal_ok)
        except (ValueError, np.linalg.LinAlgError) as exc:
            generation_time += perf_counter() - generation_start
            record.update(status="GENERATION_FAILED", error=f"{type(exc).__name__}: {exc}")
        attempts.append(record)
    stats = dict(attempts=attempts, actual_pool_size=len(pool), target_pool_size=cfg.pool_size,
                 short_pool=len(pool) < cfg.pool_size, candidate_generation_runtime=generation_time,
                 certificate_runtime=certificate_time, strict_validation_runtime=strict_time,
                 end_to_end_runtime=perf_counter() - start,
                 fast_oracle_calls=sum(c.certificate.fast_oracle_calls for c in pool),
                 strict_oracle_calls=sum(c.certificate.strict_oracle_calls for c in pool))
    return pool, stats
