"""Read offline joint radii with scope/hash checks; never run Monte Carlo."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def calibration_scope(cfg):
    """Pilot error distribution and effective-channel composition scope."""
    names = ("L", "K", "M", "N", "bits", "bs_x", "bs_y", "ue_center_x", "ue_center_y",
             "ue_radius", "ris_x", "ris_y", "c0_db", "alpha_bu", "alpha_br", "alpha_ru",
             "rician_k", "channel_scale", "direct_link_attenuation", "include_direct_intercell",
             "inter_ris_attenuation", "ris_size_model", "ris_reference_N", "radius_floor",
             "estimation_snr_db", "nmse_db")
    return json.loads(json.dumps({name: getattr(cfg, name) for name in names}))


def scope_fingerprint(cfg):
    return hashlib.sha256(json.dumps(calibration_scope(cfg), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifact_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


@lru_cache(maxsize=16)
def _read_artifact(path, expected_sha256):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ValueError("offline calibration artifact hash mismatch")
    payload = json.loads(data)
    if payload.get("schema") != "journal_offline_joint_radius_v1":
        raise ValueError("unsupported calibration artifact schema")
    if not payload.get("complete"):
        raise ValueError("offline calibration run is incomplete; failed seeds must be resolved")
    return payload


def read_calibration_artifact(cfg):
    """SHA-verified CSI artifact payload; scope resolution happens per read."""
    if cfg.csi_calibration_file is None or cfg.csi_calibration_sha256 is None:
        raise ValueError("no CSI calibration artifact bound (file and SHA256 required)")
    return _read_artifact(str(artifact_path(cfg.csi_calibration_file)), cfg.csi_calibration_sha256)


def artifact_mode(payload):
    """Artifacts that predate explicit mode marking count as smoke-only."""
    return payload.get("mode", "smoke")


def offline_radius(cfg):
    if cfg.epsilon_est is not None:
        return float(cfg.epsilon_est), dict(source="explicit_precalibrated_parameter", quantile=cfg.calibration_q)
    if cfg.csi_calibration_file is None or cfg.csi_calibration_sha256 is None:
        raise ValueError("online simulation requires an Exp1 offline calibration artifact with SHA256, or explicit epsilon_est")
    payload = _read_artifact(str(artifact_path(cfg.csi_calibration_file)), cfg.csi_calibration_sha256)
    matches = [entry for entry in payload["entries"] if entry["scope_fingerprint"] == scope_fingerprint(cfg)]
    if len(matches) != 1:
        raise ValueError("offline CSI calibration does not cover the current physical model / pilot SNR")
    entry = matches[0]
    key = f"epsilon_joint_{round(cfg.calibration_q * 100)}"
    if key not in entry or not np.isfinite(entry[key]) or entry[key] < 0:
        raise ValueError("requested joint calibration quantile unavailable")
    return float(entry[key]), dict(source=str(cfg.csi_calibration_file), sha256=cfg.csi_calibration_sha256,
                                  quantile=cfg.calibration_q, scope_fingerprint=entry["scope_fingerprint"],
                                  interpretation=payload["interpretation"])


def bind_artifact(cfg, path):
    resolved = artifact_path(path)
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    bound = cfg.with_overrides(csi_calibration_file=str(path), csi_calibration_sha256=digest, epsilon_est=None)
    offline_radius(bound)
    return bound
