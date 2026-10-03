"""All model, algorithm and experiment parameters are fingerprinted."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math


@dataclass(frozen=True)
class JournalConfig:
    L: int = 2
    K: int = 3
    M: int = 12
    N: int = 32
    bits: int = 2
    gamma: float = 2.0
    bs_x: tuple = (-100.0, 100.0)
    bs_y: tuple = (0.0, 0.0)
    ue_center_x: tuple = (-15.0, 15.0)
    ue_center_y: tuple = (0.0, 0.0)
    ue_radius: float = 7.0
    ris_x: float = 0.0
    ris_y: float = 0.0
    c0_db: float = -30.0
    alpha_bu: float = 4.5
    alpha_br: float = 2.2
    alpha_ru: float = 2.2
    rician_k: float = 3.0
    channel_scale: float = 1e5
    direct_serving_attenuation: float = 0.25
    include_direct_intercell: bool = True
    inter_ris_attenuation: float = 1.0
    ris_size_model: str = "growing_aperture"
    ris_reference_N: int = 32
    noise_power_dbm: float = -100.0
    p_max_dbm: float = 30.0
    pa_efficiency: float = 0.38
    p_bs: float = 6.0
    p_ue: float = 0.1
    p_loss: float = 0.5
    p_ris_controller: float = 0.5
    p_cell_idle: float = 0.001
    p_diode_on: float = 0.004
    bandwidth_hz: float = 1e7
    radius_floor: float = 1e-10
    strict_eig_tol: float = 1e-8
    strict_normalized_tol: float = 1e-8
    nominal_normalized_tol: float = 1e-12
    solver: str = "CLARABEL"
    solver_tol: float = 1e-10
    solver_max_iter: int = 200
    fast_tol: float = 0.0
    fast_lambda_cap: float = 1e10
    fast_search_tol: float = 1e-9
    eps_hi: float = 0.2
    eps_cap: float = 1.2
    certificate_abs_tol: float = 1e-7
    certificate_rel_tol: float = 1e-4
    certificate_max_iter: int = 60
    strict_shrink_factor: float = 0.98
    strict_shrink_steps: int = 40
    pool_size: int = 40
    pool_attempts: int = 200
    design_gamma_mult: tuple = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0)
    power_slack_grid: tuple = (1.0, 1.1, 1.3, 1.8)
    power_control_tol: float = 1e-10
    zf_rcond: float = 1e-12
    epsilon_min: float = 0.05
    estimation_snr_db: float = 40.0
    nmse_db: float | None = None
    calibration_samples: int = 2000
    calibration_quantiles: tuple = (0.90, 0.95, 0.99)
    calibration_snr_grid: tuple = (10.0, 20.0, 30.0, 40.0)
    calibration_q: float = 0.99
    time_steps: int = 100
    slot_duration: float = 0.1
    channel_correlation: float | None = None
    mobility_level: str = "slow"
    mobility_correlations: tuple = (0.9999, 0.995, 0.95)
    quasi_static_br: bool = True
    eta_trigger: float = 0.9
    T_period: int = 10
    drift_rate_nu: float = 0.002
    energy_per_bit_switch: float = 1e-5
    E_controller_fixed: float = 1e-3
    policies: tuple = ("always_reconfigure", "periodic_reconfigure", "certificate_triggered", "static")
    selection_rule: str = "lifetime_aware"
    seeds: tuple = (60001, 60002)
    channel_seed: int = 20260706
    pool_seed: int = 31001
    csi_seed: int = 51001
    calibration_seed: int = 52001
    mc_seed: int = 41001
    mc_samples: int = 300
    alpha_grid: tuple = (0.25, 0.50, 0.90, 1.00, 1.10)
    legacy_indices: tuple = (34, 6, 26)
    legacy_fixture: str = "journal_sim/tests/fixtures/conference_representatives.npz"
    lambda_grid_points: int = 400
    scaling_cases: tuple = ((16, 2), (32, 2), (64, 2), (32, 3))
    runtime_time_steps: int = 10
    runtime_policies: tuple = ("certificate_triggered",)
    output_root: str = "journal_results"
    smoke: bool = False

    def validate(self):
        if min(self.L, self.K, self.M, self.N, self.bits) < 1:
            raise ValueError("positive system dimensions required")
        if any(len(x) != self.L for x in (self.bs_x, self.bs_y, self.ue_center_x, self.ue_center_y)):
            raise ValueError("geometry lengths must equal L")
        positive = (self.gamma, self.channel_scale, self.radius_floor, self.pa_efficiency,
                    self.bandwidth_hz, self.slot_duration, self.drift_rate_nu, self.solver_tol,
                    self.strict_eig_tol, self.strict_normalized_tol, self.nominal_normalized_tol,
                    self.certificate_abs_tol, self.certificate_rel_tol, self.fast_search_tol)
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("positive finite physical/algorithm parameters required")
        if not 0 < self.eta_trigger <= 1 or not 0 < self.strict_shrink_factor < 1:
            raise ValueError("invalid trigger/shrink factor")
        if not 0 < self.eps_hi <= self.eps_cap or self.fast_lambda_cap <= 0:
            raise ValueError("invalid search bounds")
        if self.mobility_level not in ("slow", "medium", "fast"):
            raise ValueError("unknown mobility regime")
        if len(self.mobility_correlations) != 3 or any(not 0 <= x <= 1 for x in self.mobility_correlations):
            raise ValueError("three valid mobility correlations required")
        if self.channel_correlation is not None and not 0 <= self.channel_correlation <= 1:
            raise ValueError("invalid channel correlation")
        if self.ris_size_model not in ("growing_aperture", "fixed_aperture"):
            raise ValueError("unknown aperture model")
        if min(self.time_steps, self.T_period, self.pool_size, self.calibration_samples,
               self.mc_samples, self.solver_max_iter, self.certificate_max_iter, self.strict_shrink_steps) < 1:
            raise ValueError("positive experiment budgets required")
        if self.pool_attempts < self.pool_size or not self.seeds:
            raise ValueError("invalid pool/seed budget")
        if any(x < 0 for x in (self.energy_per_bit_switch, self.E_controller_fixed,
                               self.p_bs, self.p_ue, self.p_loss, self.p_ris_controller,
                               self.p_cell_idle, self.p_diode_on, self.epsilon_min)):
            raise ValueError("nonnegative powers, energies and threshold required")
        if any(not 0 < x < 1 for x in (*self.calibration_quantiles, self.calibration_q)):
            raise ValueError("invalid calibration quantile")
        if self.selection_rule not in ("lifetime_aware", "wee_only", "robustness_only", "stability_aware"):
            raise ValueError("unknown selection rule")
        if not set(self.policies) <= {"always_reconfigure", "periodic_reconfigure", "certificate_triggered", "static"}:
            raise ValueError("unknown policy")
        return self

    @property
    def noise_power(self):
        return 10 ** ((self.noise_power_dbm - 30) / 10) * self.channel_scale ** 2

    @property
    def p_max(self):
        return 10 ** ((self.p_max_dbm - 30) / 10)

    @property
    def correlation(self):
        return self.channel_correlation if self.channel_correlation is not None else self.mobility_correlations[("slow", "medium", "fast").index(self.mobility_level)]

    def with_overrides(self, **kwargs):
        return replace(self, **kwargs).validate()

    def to_dict(self):
        return asdict(self)


def config_fingerprint(cfg):
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def smoke_config(**kwargs):
    return JournalConfig(pool_size=4, pool_attempts=20, time_steps=6, runtime_time_steps=2,
                         calibration_samples=100, mc_samples=50, seeds=(60001,), smoke=True).with_overrides(**kwargs)
