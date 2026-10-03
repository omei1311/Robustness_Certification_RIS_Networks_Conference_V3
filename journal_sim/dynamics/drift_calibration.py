"""Offline mobility/drift calibration for the lifetime predictor.

The lifetime predictor assumes rho(t) ~ nu * t. This module estimates an
empirical, mobility-specific drift-rate proxy nu from *calibration* drops
that replay the exact online observation pipeline:

    true Gauss-Markov channel -> per-slot physical CSI estimator ->
    estimated effective channel -> trigger's relative_drift().

nu is ONLY used for candidate lifetime prediction / selection. It never
enters the online trigger: the trigger keeps comparing the actually observed
rho_total = eps_est*(1+rho_obs)+rho_obs against eta*eps_cert, and nu never
changes the certificate's safety definition.

Calibration seeds (drift_calibration_seeds) are disjoint from the CSI
calibration seeds and from the formal evaluation seeds; calibration/evaluation
leakage is impossible by construction.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
from journal_sim.core.channels import effective_channels
from journal_sim.core.models import phase_set
from journal_sim.core.uncertainty import relative_drift
from .channel_process import channel_trajectory
from .csi_estimation import estimate_physical, csi_observation_seed

ROOT = Path(__file__).resolve().parents[2]
MOBILITY_ORDER = ("slow", "medium", "fast")
DRIFT_SCHEMA = "journal_offline_drift_rate_v1"

# Quantiles reported in the artifact alongside the selected one.
DRIFT_REPORT_QUANTILES = (0.50, 0.90, 0.95, 0.99)


def drift_calibration_scope(cfg):
    """Only parameters that shape the observed drift process.

    Deliberately excludes pool/seeds/policies/solver/workers/certificate
    tolerances so a formal dynamic experiment stays compatible with an
    artifact produced under a different evaluation budget.
    """
    names = ("L", "K", "M", "N", "bits", "bs_x", "bs_y", "ue_center_x", "ue_center_y",
             "ue_radius", "ris_x", "ris_y", "c0_db", "alpha_bu", "alpha_br", "alpha_ru",
             "rician_k", "channel_scale", "direct_link_attenuation", "include_direct_intercell",
             "inter_ris_attenuation", "ris_size_model", "ris_reference_N",
             "mobility_correlations", "quasi_static_br", "estimation_snr_db", "nmse_db",
             "slot_duration", "radius_floor")
    return json.loads(json.dumps({name: getattr(cfg, name) for name in names}))


def drift_scope_fingerprint(cfg):
    return hashlib.sha256(json.dumps(drift_calibration_scope(cfg), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def rho_per_second(rho, delta_slots, cfg):
    """Empirical drift-rate sample in 1/s; never per-slot units."""
    if delta_slots < 1:
        raise ValueError("delta_slots must be a positive slot count")
    return float(rho) / (delta_slots * cfg.slot_duration)


def phase_profile(cfg, seed, phase):
    """Deterministic RIS phase profile: all-ones first, random discrete after."""
    phase_seed = int(np.random.SeedSequence([cfg.calibration_seed, seed, phase]).generate_state(1)[0])
    theta = np.ones(cfg.N, complex) if phase == 0 else phase_set(cfg.bits)[
        np.random.default_rng(phase_seed).integers(2 ** cfg.bits, size=cfg.N)]
    return theta


def drift_samples(cfg, seed):
    """One calibration drop: observed relative drift on estimated channels.

    Uses the trigger's own relative_drift() on H_hat(t) under representative
    discrete RIS phase profiles, with references reset every
    drift_reference_stride slots - mirroring how a reused configuration
    re-anchors at each reinstallation event.
    """
    rows = []
    for mobility in cfg.mobility_regimes:
        local = cfg.with_overrides(mobility_level=mobility, time_steps=cfg.drift_calibration_slots)
        trajectory = channel_trajectory(local, seed)
        estimates = [estimate_physical(c, local, csi_observation_seed(local, seed, t))
                     for t, c in enumerate(trajectory)]
        for phase in range(cfg.calibration_phase_profiles):
            theta = phase_profile(local, seed, phase)
            H_hat = [effective_channels(est, theta, local) for est in estimates]
            for reference in range(0, local.time_steps, cfg.drift_reference_stride):
                for delta in cfg.drift_horizon_slots:
                    current = reference + delta
                    if current >= local.time_steps:
                        continue
                    rho = relative_drift(H_hat[current], H_hat[reference], local)
                    rows.append(dict(mobility=mobility, seed=seed, phase_profile=phase,
                                     reference_index=reference, delta_slots=delta,
                                     delta_time=delta * local.slot_duration, rho_obs=rho,
                                     rho_over_time=rho_per_second(rho, delta, local)))
    return rows


def summarize_drift(rows, cfg):
    """Quantiles of the pooled rho/delta_t samples, per mobility regime."""
    summary = {}
    for mobility in cfg.mobility_regimes:
        rates = np.array([r["rho_over_time"] for r in rows if r["mobility"] == mobility], float)
        if not len(rates):
            raise ValueError(f"no drift samples for mobility {mobility}")
        entry = {f"nu_{round(100 * q)}": float(np.quantile(rates, q)) for q in DRIFT_REPORT_QUANTILES}
        entry["nu_selected"] = float(np.quantile(rates, cfg.drift_calibration_q))
        entry["sample_count"] = int(len(rates))
        summary[mobility] = entry
    return summary


def artifact_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


@lru_cache(maxsize=16)
def _read_drift_artifact(path, expected_sha256):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError("offline drift calibration artifact hash mismatch")
    payload = json.loads(data)
    if payload.get("schema_version") != 1 or payload.get("calibration_type") != DRIFT_SCHEMA:
        raise ValueError("unsupported drift calibration artifact schema")
    if not payload.get("complete"):
        raise ValueError("offline drift calibration run is incomplete; failed seeds must be resolved")
    return payload


def read_drift_artifact(cfg):
    """SHA-verified artifact payload; scope/mobility checks happen per read."""
    if cfg.drift_calibration_file is None or cfg.drift_calibration_sha256 is None:
        raise ValueError("no drift calibration artifact bound (file and SHA256 required)")
    return _read_drift_artifact(str(artifact_path(cfg.drift_calibration_file)), cfg.drift_calibration_sha256)


def offline_drift_rate(cfg, mobility):
    """Validated artifact nu for one mobility, with full provenance."""
    payload = read_drift_artifact(cfg)
    expected = drift_scope_fingerprint(cfg)
    if payload.get("scope_fingerprint") != expected:
        raise ValueError("offline drift calibration does not cover the current channel/CSI model")
    block = payload.get("mobility", {})
    if mobility not in block:
        raise ValueError(f"drift calibration artifact lacks mobility regime {mobility}")
    entry = block[mobility]
    nu = entry.get("nu_selected")
    if nu is None or not np.isfinite(nu) or nu <= 0:
        raise ValueError(f"drift calibration artifact has no usable nu for {mobility}")
    return float(nu), dict(source=str(cfg.drift_calibration_file), sha256=cfg.drift_calibration_sha256,
                           quantile=payload.get("quantile"), unit=payload.get("unit"),
                           scope_fingerprint=expected, mobility=mobility,
                           interpretation=payload.get("interpretation"))


def bind_drift_artifact(cfg, path):
    resolved = artifact_path(path)
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    bound = cfg.with_overrides(drift_calibration_file=str(path), drift_calibration_sha256=digest)
    for mobility in bound.mobility_regimes:
        offline_drift_rate(bound, mobility)
    return bound


def effective_drift_rate(cfg, mobility=None):
    """Lifetime-predictor nu with a single resolution order.

    Bound artifact -> manual per-mobility rates -> legacy constant. Formal
    experiments never reach the fallbacks: the formal guard rejects them at
    startup, so a legacy value can only appear in smoke/debug runs.
    """
    mobility = cfg.mobility_level if mobility is None else mobility
    if cfg.drift_rate_nu_by_mobility is not None:
        if mobility not in MOBILITY_ORDER:
            raise ValueError(f"no manual drift rate for mobility {mobility}")
        value = float(cfg.drift_rate_nu_by_mobility[MOBILITY_ORDER.index(mobility)])
        return value, dict(source="manual_per_mobility", mobility=mobility, unit="1/s")
    if cfg.drift_calibration_file is not None or cfg.drift_calibration_sha256 is not None:
        return offline_drift_rate(cfg, mobility)
    return float(cfg.drift_rate_nu), dict(source="legacy_manual_constant", mobility=mobility,
                                          value=cfg.drift_rate_nu, unit="1/s")
