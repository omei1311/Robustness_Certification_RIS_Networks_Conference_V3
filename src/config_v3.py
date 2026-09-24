"""Certification-layer experiment settings (new in the V3 paper).

These parameters control the NEW parts of the framework only -- candidate
pool generation, R_cert bisection, dominance filtering, stability-aware
selection, and the three numerical experiments of Section 9.  The physical
system model stays in ``ris_base.SimConfig`` (reused from the first paper).

All radii follow the ``relative_radius`` convention of the reused layer:
a radius value ``eps`` means ``eps * ||h_lk||`` per user.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class CertConfig:
    # ------------------------------------------------------------------ #
    # Candidate pool generation (Section 9 protocol)                     #
    # ------------------------------------------------------------------ #
    pool_seed: int = 31001            # pool-generation RNG (independent of channel seed)
    pool_target_size: int = 40        # paper: ~20-50 unique feasible configurations
    pool_max_attempts: int = 400      # generation attempts budget
    channel_seed: int = 20260706      # nominal channel drop (shared comparison axis)

    # Design-space knobs sampled per candidate for pool diversity.
    # "zf_full" (intra-cell ZF + inter-cell leakage nulling) is the reliable
    # workhorse: with the shared RIS coupling both cells at full strength,
    # pure mrt/rzf direction sets are structurally interference-limited and
    # almost never admit a feasible power allocation at the QoS target.
    direction_choices: Tuple[str, ...] = ("zf_full",)
    align_frac_grid: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    align_jitter_grid: Tuple[float, ...] = (0.0, 0.15, 0.3, 0.6, 1.0)  # radians, half-width
    # The paper fixes B = 2; bit resolution is NOT a diversity knob.
    # Diversity comes from phase initialization, alignment fraction/jitter,
    # beamforming initialization, design epsilon/gamma, power slack/
    # perturbation, seeds.
    bits_grid: Tuple[int, ...] = (2,)
    # QoS design targets (multiples of the required gamma_bar).
    design_gamma_mult: Tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 7.0)
    # Design uncertainty radius choices: 0 = nominal design, eps = robust design.
    design_eps_grid: Tuple[float, ...] = (0.0, 0.025, 0.05, 0.075, 0.10)
    # Non-minimal power operation factors (wasteful designs that trade WEE
    # for extra SINR margin; part of the pool-diversity protocol).
    power_slack_grid: Tuple[float, ...] = (1.0, 1.1, 1.2, 1.3, 1.5, 1.8)
    # Small per-user relative power perturbations around the power-control
    # solution (0 = none). Perturbed allocations are never accepted blindly:
    # they must re-pass nominal QoS, and robust QoS at the design radius
    # whenever design_eps > 0.
    power_perturb_grid: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)

    # Prescribed design uncertainty radius epsilon reported in Section 9.
    epsilon_design: float = 0.05

    # Power-control fixed point
    pc_max_iter: int = 400
    pc_tol: float = 1.0e-10

    # ------------------------------------------------------------------ #
    # R_cert bisection (Section 4, Eq. (12)-(13))                        #
    # ------------------------------------------------------------------ #
    bisection_eps_hi: float = 0.60    # initial upper bound (relative radius)
    bisection_eps_hi_max: float = 1.20  # bracket-expansion cap
    bisection_tol: float = 1.0e-4     # delta_r on the relative radius
    bisection_max_iter: int = 40

    # ------------------------------------------------------------------ #
    # Experiment 1 -- certificate validation (Monte Carlo sweep)          #
    # ------------------------------------------------------------------ #
    exp1_n_configs: int = 4
    exp1_radius_extend: float = 3.0   # sweep realized radius up to 3 * R_cert
    exp1_radius_points: int = 25
    exp1_mc_samples: int = 300
    exp1_seed: int = 41001

    # ------------------------------------------------------------------ #
    # Experiment 3 -- selection sensitivity                               #
    # ------------------------------------------------------------------ #
    r_min_fracs: Tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 0.90, 0.95)
    # Linear drift-rate bound nu (relative radius per second); used ONLY for
    # the conditional T_cert corollary: T_cert = R_cert / nu  (Eq. (16)).
    drift_rate_nu: float = 2.0e-3
    exp3_check_samples: int = 500
    exp3_seed: int = 43001

    # ------------------------------------------------------------------ #
    # Experiment 3 generalization check (independent channel seeds)       #
    # ------------------------------------------------------------------ #
    gen_check_seed_base: int = 60001
    gen_check_n_seeds: int = 12

    def validate(self) -> "CertConfig":
        if self.pool_target_size <= 0 or self.pool_max_attempts < self.pool_target_size:
            raise ValueError("pool sizes inconsistent")
        if self.bisection_tol <= 0.0 or self.bisection_tol >= self.bisection_eps_hi:
            raise ValueError("bisection tolerance must lie inside the bracket")
        if self.epsilon_design < 0.0:
            raise ValueError("epsilon_design must be nonnegative")
        if self.drift_rate_nu <= 0.0:
            raise ValueError("drift_rate_nu must be positive")
        return self

    def with_overrides(self, **kwargs: Any) -> "CertConfig":
        return replace(self, **kwargs).validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_cert_config() -> CertConfig:
    return CertConfig().validate()


def config_fingerprint(cfg, cert_cfg) -> str:
    """Stable sha256 fingerprint of every parameter that determines the
    certified pool: system model (topology, channel, QoS, power, uncertainty
    conventions) and the full pool/certification/experiment configuration.

    The pool cache may only be reused when this fingerprint matches exactly;
    any mismatch forces a rebuild.  Note that ``SimConfig.seed`` is
    deliberately excluded: the experiment channel is generated from
    ``cert_cfg.channel_seed``, which is included below.
    """
    payload = {
        "simconfig": {
            "L": cfg.L, "K": cfg.K, "M": cfg.M, "N": cfg.N,
            "gamma": cfg.gamma, "bits": cfg.bits,
            "uncertainty_model": cfg.uncertainty_model,
            "relative_radius": cfg.relative_radius,
            "radius_floor": cfg.radius_floor,
            "feasibility_tol": cfg.feasibility_tol,
            "geometry_channel": {
                "bs_x": tuple(cfg.bs_x), "bs_y": tuple(cfg.bs_y),
                "ue_center_x": tuple(cfg.ue_center_x),
                "ue_center_y": tuple(cfg.ue_center_y),
                "ue_radius": cfg.ue_radius,
                "ris_x": cfg.ris_x, "ris_y": cfg.ris_y,
                "c0_db": cfg.c0_db,
                "alpha_bu": cfg.alpha_bu, "alpha_br": cfg.alpha_br,
                "alpha_ru": cfg.alpha_ru,
                "rician_k": cfg.rician_k,
                "channel_scale": cfg.channel_scale,
                "direct_serving_attenuation": cfg.direct_serving_attenuation,
                "include_direct_intercell": cfg.include_direct_intercell,
                "inter_ris_attenuation": cfg.inter_ris_attenuation,
                "ris_size_model": cfg.ris_size_model,
                "ris_reference_N": cfg.ris_reference_N,
            },
            "power_model": {
                "noise_power_dbm": cfg.noise_power_dbm,
                "p_max_dbm": cfg.p_max_dbm,
                "pa_efficiency": cfg.pa_efficiency,
                "p_bs": cfg.p_bs, "p_ue": cfg.p_ue, "p_loss": cfg.p_loss,
                "p_ris_controller": cfg.p_ris_controller,
                "p_cell_idle": cfg.p_cell_idle,
                "p_diode_on": cfg.p_diode_on,
                "ue_power_per_user": cfg.ue_power_per_user,
                "ris_power_model": cfg.ris_power_model,
                "omega_eta": cfg.omega_eta,
                "bandwidth_hz": cfg.bandwidth_hz,
            },
        },
        "certconfig": {
            "channel_seed": cert_cfg.channel_seed,
            "pool_seed": cert_cfg.pool_seed,
            "pool_target_size": cert_cfg.pool_target_size,
            "pool_max_attempts": cert_cfg.pool_max_attempts,
            "direction_choices": tuple(cert_cfg.direction_choices),
            "align_frac_grid": tuple(cert_cfg.align_frac_grid),
            "align_jitter_grid": tuple(cert_cfg.align_jitter_grid),
            "bits_grid": tuple(cert_cfg.bits_grid),
            "design_gamma_mult": tuple(cert_cfg.design_gamma_mult),
            "design_eps_grid": tuple(cert_cfg.design_eps_grid),
            "power_slack_grid": tuple(cert_cfg.power_slack_grid),
            "power_perturb_grid": tuple(cert_cfg.power_perturb_grid),
            "epsilon_design": cert_cfg.epsilon_design,
            "pc_max_iter": cert_cfg.pc_max_iter,
            "pc_tol": cert_cfg.pc_tol,
            "bisection_eps_hi": cert_cfg.bisection_eps_hi,
            "bisection_eps_hi_max": cert_cfg.bisection_eps_hi_max,
            "bisection_tol": cert_cfg.bisection_tol,
            "bisection_max_iter": cert_cfg.bisection_max_iter,
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
