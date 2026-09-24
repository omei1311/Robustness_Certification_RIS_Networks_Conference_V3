"""Small helper shared by the experiment scripts."""

from __future__ import annotations

from src.ris_base import SimConfig
from src.ris_base.uncertainty import uncertainty_dimension


def uncertainty_shape(cfg: SimConfig) -> tuple:
    """Per-sample error-direction shape (L, K, dim) for the aggregate model."""
    return (cfg.L, cfg.K, uncertainty_dimension(cfg))
