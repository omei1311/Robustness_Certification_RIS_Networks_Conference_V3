"""Certification-layer experiment settings (new in the V3 paper).

These parameters control the NEW parts of the framework only -- candidate
pool generation, R_cert bisection, dominance filtering, stability-aware
selection, and the three numerical experiments of Section 9.  The physical
system model stays in ``ris_base.SimConfig`` (reused from the first paper).

All radii follow the ``relative_radius`` convention of the reused layer:
a radius value ``eps`` means ``eps * ||h_lk||`` per user.
"""

from __future__ import annotations

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
    align_frac_grid: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    align_jitter_grid: Tuple[float, ...] = (0.0, 0.3, 0.6, 1.0)  # radians, half-width
    bits_grid: Tuple[int, ...] = (1, 2, 3)
    # QoS design targets (multiples of the required gamma_bar).
    design_gamma_mult: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)
    # Design uncertainty radius choices: 0 = nominal design, eps = robust design.
    design_eps_grid: Tuple[float, ...] = (0.0, 0.05)
    # Non-minimal power operation factors (wasteful designs that trade WEE
    # for extra SINR margin; part of the pool-diversity protocol).
    power_slack_grid: Tuple[float, ...] = (1.0, 1.3, 1.8)

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
