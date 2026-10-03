from dataclasses import dataclass
import numpy as np
from .models import phase_indices


def gray_code(index):
    return int(index) ^ (int(index) >> 1)


def ris_power(theta, cfg):
    count = sum(gray_code(s).bit_count() for s in phase_indices(theta, cfg.bits))
    return cfg.p_ris_controller + cfg.N * cfg.p_cell_idle + cfg.p_diode_on * count


@dataclass(frozen=True)
class PowerBreakdown:
    transmission_circuit: float
    ris_static_state: float

    @property
    def total(self):
        return self.transmission_circuit + self.ris_static_state


def system_power(w, theta, cfg):
    tx = float(np.sum(abs(w) ** 2)) / cfg.pa_efficiency
    circuit = cfg.L * (cfg.p_bs + cfg.p_loss + cfg.K * cfg.p_ue)
    return PowerBreakdown(tx + circuit, ris_power(theta, cfg))


@dataclass(frozen=True)
class TransitionEnergy:
    n_changed_elements: int
    n_changed_bits: int
    switching_energy: float
    controller_energy: float

    @property
    def total(self):
        return self.switching_energy + self.controller_energy


def transition_energy(theta_old, theta_new, cfg):
    old, new = np.asarray(theta_old), np.asarray(theta_new)
    if old.shape != (cfg.N,) or new.shape != (cfg.N,):
        raise ValueError("transition shape mismatch")
    from .models import quantize_theta
    if not (np.allclose(old, quantize_theta(old, cfg.bits), atol=1e-12, rtol=0) and
            np.allclose(new, quantize_theta(new, cfg.bits), atol=1e-12, rtol=0)):
        raise ValueError("transition states must be discrete")
    a, b = phase_indices(old, cfg.bits), phase_indices(new, cfg.bits)
    changed = int(np.count_nonzero(a != b))
    bits = sum((gray_code(x) ^ gray_code(y)).bit_count() for x, y in zip(a, b))
    # Controller cost is per redesign event, even if theta stays unchanged.
    return TransitionEnergy(changed, bits, bits * cfg.energy_per_bit_switch, cfg.E_controller_fixed)
