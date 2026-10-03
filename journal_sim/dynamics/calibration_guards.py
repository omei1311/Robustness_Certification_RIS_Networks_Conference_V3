"""Startup guards for formal experiments: no smoke artifacts, no seed leakage.

Formal dynamic experiments (exp3/exp4) must run on a formal CSI calibration
artifact (or an explicit pre-calibrated radius) and a formal drift
calibration artifact, with calibration seeds disjoint from the evaluation
seeds. Smoke runs keep full manual freedom for debugging. Guards raise
SystemExit with an explicit message instead of silently falling back to
legacy constants.
"""
from .offline_calibration import offline_radius, read_calibration_artifact, artifact_mode
from .drift_calibration import offline_drift_rate, read_drift_artifact


def _check_seed_isolation(cfg, experiment):
    calibration = set(cfg.calibration_seeds) | set(cfg.drift_calibration_seeds)
    overlap = sorted(set(cfg.seeds) & calibration)
    if overlap:
        raise SystemExit(f"formal {experiment}: evaluation seeds overlap calibration seeds {overlap}; "
                         "calibration/evaluation leakage is forbidden")


def _require_formal_csi(cfg, experiment):
    if cfg.epsilon_est is not None:
        # Explicit pre-calibrated radius: a declared parameter, not an artifact.
        print(f"[{experiment}] formal CSI radius: explicit epsilon_est={cfg.epsilon_est}", flush=True)
        return
    if cfg.csi_calibration_file is None or cfg.csi_calibration_sha256 is None:
        raise SystemExit(f"formal {experiment} requires a formal CSI calibration artifact "
                         "(--csi-calibration) or an explicit --epsilon-est")
    try:
        offline_radius(cfg)  # SHA, schema, completeness and scope checks
    except ValueError as exc:
        raise SystemExit(f"formal {experiment}: CSI calibration artifact rejected: {exc}") from exc
    mode = artifact_mode(read_calibration_artifact(cfg))
    if mode != "formal":
        raise SystemExit(f"formal {experiment} requires a formal CSI calibration artifact; "
                         f"smoke artifact supplied (mode={mode})")


def _require_formal_drift(cfg, experiment, hard=True):
    bound = cfg.drift_calibration_file is not None or cfg.drift_calibration_sha256 is not None
    if not bound:
        if hard:
            raise SystemExit(f"formal {experiment} requires a formal drift calibration artifact "
                             "(--drift-calibration); legacy constant drift_rate_nu is smoke-only")
        print(f"WARNING: formal {experiment} runs without a drift artifact; lifetime ranking uses the "
              f"legacy constant nu={cfg.drift_rate_nu}. Primary dynamic evidence comes from exp3.",
              flush=True)
        return
    try:
        for mobility in cfg.mobility_regimes:
            offline_drift_rate(cfg, mobility)  # SHA, scope, mobility-key checks
    except ValueError as exc:
        raise SystemExit(f"formal {experiment}: drift calibration artifact rejected: {exc}") from exc
    mode = artifact_mode(read_drift_artifact(cfg))
    if mode != "formal":
        raise SystemExit(f"formal {experiment} requires a formal drift calibration artifact; "
                         f"smoke artifact supplied (mode={mode})")


def formal_calibration_guard(cfg, experiment, drift="hard"):
    """drift: 'hard' (exp3/exp4) rejects a missing artifact; 'soft' (exp2) warns."""
    if cfg.smoke:
        return
    drift = "hard" if drift == "hard" else "soft"
    _check_seed_isolation(cfg, experiment)
    _require_formal_csi(cfg, experiment)
    _require_formal_drift(cfg, experiment, hard=drift == "hard")
