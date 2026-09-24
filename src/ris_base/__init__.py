"""Reusable base layer adapted from the first paper's robust WEE simulator.

Source: ``ris_robust_simulation`` (V6.5 multi-cell discrete-RIS robust WEE
design project).  The modules below are trimmed copies that keep the channel
generation, beamforming initialization, RIS phase-configuration, WEE power
model, S-procedure LMI robustness check, and uncertainty-sampling conventions
identical to the original, while removing torch/yaml/solver dependencies that
the certification layer does not need.

Mapping to the V3 paper ("Configuration-Specific Robustness Certification
for Stability-Aware Selection of Discrete RIS Configurations"):
  - channels.effective_channels ........ effective channel h in Eq. (2)
  - models.compute_sinr_from_H .......... Eq. (2) SINR / QoS checking
  - models.utility_power_wee_from_H ..... Eq. (5) WEE
  - models.phase_set/quantize_theta ..... Eq. (1) discrete RIS phases Q_B
  - robustness.certificate_* ........... Eq. (7) quadratic form A_lk(X)
  - robustness.sprocedure_matrix ....... Eq. (9) statewise LMI M_lk
  - robustness.robust_check ............ statewise robust feasibility oracle
  - uncertainty.channel_radius ......... relative/absolute radius convention
"""

from .config import SimConfig
from .channels import ChannelDrop, generate_channel_drop, effective_channels
from .models import (
    phase_set,
    phase_indices,
    quantize_theta,
    random_theta,
    ris_power,
    total_power,
    compute_sinr_from_H,
    compute_sinr,
    user_weights,
    utility_power_wee,
    utility_power_wee_from_H,
    initialize_mrt,
    initialize_rzf,
)
from .uncertainty import (
    uncertainty_dimension,
    sample_complex_unit_ball,
    channel_radius,
)
from .robustness import (
    certificate_channel_vector,
    certificate_quadratic_matrix,
    sprocedure_matrix,
    optimize_lambda,
    robust_check,
)
from .evaluation import sinr_under_error_samples

__all__ = [
    "SimConfig",
    "ChannelDrop",
    "generate_channel_drop",
    "effective_channels",
    "phase_set",
    "phase_indices",
    "quantize_theta",
    "random_theta",
    "ris_power",
    "total_power",
    "compute_sinr_from_H",
    "compute_sinr",
    "user_weights",
    "utility_power_wee",
    "utility_power_wee_from_H",
    "initialize_mrt",
    "initialize_rzf",
    "uncertainty_dimension",
    "sample_complex_unit_ball",
    "channel_radius",
    "certificate_channel_vector",
    "certificate_quadratic_matrix",
    "sprocedure_matrix",
    "optimize_lambda",
    "robust_check",
    "sinr_under_error_samples",
]
