"""Explicit CN estimation noise; empirical quantiles are not bounded errors."""
import numpy as np
from journal_sim.core.channels import complex_normal
from journal_sim.core.uncertainty import stack_users


def error_ratio(cfg):
    """Target E||e||² / ||h_true||². NMSE overrides SNR when specified."""
    db = -cfg.estimation_snr_db if cfg.nmse_db is None else cfg.nmse_db
    return float(10 ** (db / 10))


def estimate_effective(H_true, cfg, seed):
    """Per-user isotropic stacked effective-channel noise for calibration."""
    rng = np.random.default_rng(seed)
    h = stack_users(H_true)
    sigma = np.linalg.norm(h, axis=-1, keepdims=True) * np.sqrt(error_ratio(cfg) / h.shape[-1])
    observed = h + sigma * complex_normal(h.shape, rng)
    return observed.reshape(cfg.L, cfg.K, cfg.L, cfg.M).transpose(2, 0, 1, 3)


def estimate_physical(channel, cfg, seed):
    """Noisy BS-UE/RIS-UE pilots with known quasi-static BS-RIS link.

    Effective errors are colored after composition. Dynamic experiments
    therefore calibrate this physical estimator, not isotropic ball samples.
    """
    rng = np.random.default_rng(seed)
    updates = {}
    for name in ("h_bu", "h_ru"):
        true = getattr(channel, name)
        sigma = np.sqrt(np.mean(abs(true) ** 2, axis=-1, keepdims=True) * error_ratio(cfg))
        updates[name] = true + sigma * complex_normal(true.shape, rng)
    return channel.with_links(**updates)


def summarize_calibration(relative, squared_errors, squared_signal, cfg, seed):
    relative = np.asarray(relative)
    joint = relative.max(axis=(1, 2))
    nmse = float(np.sum(squared_errors) / np.sum(squared_signal))
    result = dict(calibration_seed=seed, estimation_snr_db=cfg.estimation_snr_db,
                  nmse_db_target=cfg.nmse_db, target_nmse=error_ratio(cfg),
                  measured_nmse=nmse, measured_nmse_db=float(10 * np.log10(nmse)),
                  sample_count=len(joint), relative_error_distribution=relative,
                  joint_relative_error_distribution=joint,
                  interpretation="empirically calibrated per-slot radius; Gaussian errors are unbounded; no deterministic or horizon guarantee")
    for q in cfg.calibration_quantiles:
        result[f"epsilon_{round(100 * q)}"] = float(np.quantile(relative, q))
        result[f"epsilon_joint_{round(100 * q)}"] = float(np.quantile(joint, q))
    result["epsilon_est"] = float(np.quantile(joint, cfg.calibration_q))
    return result


def calibrate_csi(H_true, cfg, seed):
    rng = np.random.default_rng(seed)
    h = stack_users(H_true)
    sigma = np.linalg.norm(h, axis=-1, keepdims=True) * np.sqrt(error_ratio(cfg) / h.shape[-1])
    errors = sigma[None] * complex_normal((cfg.calibration_samples,) + h.shape, rng)
    observed = h[None] + errors
    relative = np.linalg.norm(errors, axis=-1) / np.maximum(np.linalg.norm(observed, axis=-1), cfg.radius_floor)
    return summarize_calibration(relative, abs(errors) ** 2, np.broadcast_to(abs(h) ** 2, errors.shape), cfg, seed)


def calibrate_physical(channel, theta, cfg, seed):
    """Calibrate fixed-theta effective errors of the physical pilot model."""
    from journal_sim.core.channels import effective_channels
    reference = stack_users(effective_channels(channel, theta, cfg))
    seeds = np.random.SeedSequence(seed).generate_state(cfg.calibration_samples)
    errors, ratios = [], []
    for child_seed in seeds:
        observed = stack_users(effective_channels(estimate_physical(channel, cfg, int(child_seed)), theta, cfg))
        error = observed - reference
        errors.append(error)
        ratios.append(np.linalg.norm(error, axis=-1) / np.maximum(np.linalg.norm(observed, axis=-1), cfg.radius_floor))
    errors = np.asarray(errors)
    return summarize_calibration(ratios, abs(errors) ** 2, np.broadcast_to(abs(reference) ** 2, errors.shape), cfg, seed)
