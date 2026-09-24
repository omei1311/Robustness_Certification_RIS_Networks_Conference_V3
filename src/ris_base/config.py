"""System parameters (reused from the first paper's SimConfig, trimmed).

Kept fields are exactly those referenced by the reused channel-generation,
RIS-configuration, WEE, and S-procedure modules.  Solver-budget knobs
(AO/Dinkelbach/GA/figure settings) and the yaml dependency are dropped.

Conventions inherited from the first paper:
  - channels are generated with ``channel_scale = 1e5`` and the receiver
    noise power is rescaled by ``channel_scale**2`` so that link budgets are
    dimensionally consistent;
  - the V6.5 aggregate uncertainty model ("equivalent_aggregate_l2") stacks
    BS0..BS(L-1) equivalent channels of each user into one L*M complex
    vector, which is exactly the per-user channel vector of the V3 paper;
  - ``relative_radius = True`` makes the uncertainty radius a fraction of
    the user's own channel norm (r_lk = eps * ||h_lk||).  All radii in this
    framework -- the prescribed design factor epsilon and the certificate
    epsilon_cert -- follow this convention unless ``relative_radius`` is
    disabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class SimConfig:
    """System-model configuration shared by the reuse layer."""

    # Reproducibility
    seed: int = 20260706

    # Network
    L: int = 2
    K: int = 3
    M: int = 12
    N: int = 32
    bs_x: tuple = (-100.0, 100.0)
    bs_y: tuple = (0.0, 0.0)
    ue_center_x: tuple = (-15.0, 15.0)
    ue_center_y: tuple = (0.0, 0.0)
    ue_radius: float = 7.0
    ris_x: float = 0.0
    ris_y: float = 0.0

    # Channel
    c0_db: float = -30.0
    alpha_bu: float = 4.5
    alpha_br: float = 2.2
    alpha_ru: float = 2.2
    rician_k: float = 3.0
    channel_scale: float = 1.0e5
    direct_serving_attenuation: float = 0.25
    include_direct_intercell: bool = False
    inter_ris_attenuation: float = 1.0
    ris_size_model: str = "growing_aperture"  # growing_aperture | fixed_aperture
    ris_reference_N: int = 32

    # Noise / power
    noise_power_dbm: float = -100.0
    p_max_dbm: float = 30.0
    pa_efficiency: float = 0.38
    p_bs: float = 6.0
    p_ue: float = 0.1
    p_loss: float = 0.5
    p_ris_controller: float = 0.5
    p_cell_idle: float = 0.001
    p_diode_on: float = 0.004
    ue_power_per_user: bool = True
    ris_power_model: str = "state_dependent"  # state_dependent | constant_average | controller_only

    # QoS / WEE
    bits: int = 2
    gamma: float = 2.0
    omega_eta: float = 0.5
    bandwidth_hz: float = 10.0e6

    # Uncertainty (design-stage settings; certification adds its own)
    uncertainty_model: str = "equivalent_aggregate_l2"
    relative_radius: bool = True
    radius_floor: float = 1.0e-10
    eps_list: tuple = (0.0, 0.10, 0.20, 0.25, 0.30)

    # Robust-feasibility acceptance threshold for the LMI oracle.
    feasibility_tol: float = 2.0e-4

    # MRT initialization power fraction (reused initializer)
    init_power_fraction: float = 0.7

    @property
    def noise_power_watt(self) -> float:
        raw = 10.0 ** ((self.noise_power_dbm - 30.0) / 10.0)
        return raw * self.channel_scale**2

    @property
    def p_max_watt(self) -> float:
        return 10.0 ** ((self.p_max_dbm - 30.0) / 10.0)

    @property
    def bs_positions(self) -> np.ndarray:
        return np.column_stack([np.asarray(self.bs_x), np.asarray(self.bs_y)])

    @property
    def ue_centers(self) -> np.ndarray:
        return np.column_stack([np.asarray(self.ue_center_x), np.asarray(self.ue_center_y)])

    @property
    def ris_position(self) -> np.ndarray:
        return np.asarray([self.ris_x, self.ris_y], dtype=float)

    def validate(self) -> "SimConfig":
        if min(self.L, self.K, self.M, self.N) <= 0:
            raise ValueError("L, K, M, and N must be positive")
        if len(self.bs_x) != self.L or len(self.bs_y) != self.L:
            raise ValueError("bs_x/bs_y lengths must equal L")
        if len(self.ue_center_x) != self.L or len(self.ue_center_y) != self.L:
            raise ValueError("UE-center lengths must equal L")
        if self.bits <= 0 or self.gamma <= 0:
            raise ValueError("bits and gamma must be positive")
        if tuple(sorted(self.eps_list)) != tuple(self.eps_list) or any(x < 0 for x in self.eps_list):
            raise ValueError("eps_list must be sorted and nonnegative")
        if self.uncertainty_model not in {"equivalent_desired_l2", "equivalent_aggregate_l2"}:
            raise ValueError(
                "uncertainty_model must be equivalent_desired_l2 or equivalent_aggregate_l2"
            )
        if self.ris_size_model not in {"growing_aperture", "fixed_aperture"}:
            raise ValueError("invalid ris_size_model")
        if self.ris_power_model not in {"state_dependent", "constant_average", "controller_only"}:
            raise ValueError("invalid ris_power_model")
        if self.feasibility_tol <= 0.0:
            raise ValueError("feasibility_tol must be positive")
        return self

    def with_overrides(self, **kwargs: Any) -> "SimConfig":
        return replace(self, **kwargs).validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
