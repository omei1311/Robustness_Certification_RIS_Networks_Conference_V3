"""Numerical framework for the V3 conference paper.

Configuration-Specific Robustness Certification for Stability-Aware
Selection of Discrete RIS Configurations.

Pipeline (paper Eq. (20)):
    Candidate pool -> statewise robust feasibility -> R_cert ->
    WEE-robustness Pareto/dominance filtering -> stability-aware selection
    -> (conditional) T_cert interpretation.
"""

from .ris_base import SimConfig
from .config_v3 import CertConfig, default_cert_config

__all__ = ["SimConfig", "CertConfig", "default_cert_config"]
