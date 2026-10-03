"""Threshold comparisons on observed CSI only, with immutable reference X."""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from journal_sim.core.models import Configuration, array_digest
from journal_sim.core.uncertainty import relative_drift


@dataclass(frozen=True)
class ReferenceState:
    configuration: Configuration
    certificate: object
    H_hat_ref: np.ndarray
    reference_time: int
    epsilon_est_calibrated: float
    channel_digest: str

    def valid_for(self, cfg):
        return (self.certificate.reusable and
                self.channel_digest == array_digest(self.H_hat_ref) and
                self.certificate.matches(self.configuration.w, self.H_hat_ref, self.configuration.theta, cfg))


def reset_reference(configuration, certificate, H_hat, time_index, epsilon_est, cfg):
    configuration.validate(cfg)
    if not certificate.matches(configuration.w, H_hat, configuration.theta, cfg) or not certificate.reusable:
        raise ValueError("new X needs its own positive strict certificate")
    ref = np.array(H_hat, complex, copy=True)
    ref.setflags(write=False)
    return ReferenceState(configuration, deepcopy(certificate), ref, time_index, float(epsilon_est), array_digest(ref))


@dataclass(frozen=True)
class TriggerDecision:
    triggered: bool
    reason: str
    rho_obs: float
    epsilon_est: float
    rho_total: float
    threshold: float


def trigger_decision(policy, state, H_hat_current, time_index, cfg):
    if state is None:
        return TriggerDecision(True, "initial_design", 0., 0., 0., 0.)
    rho = relative_drift(H_hat_current, state.H_hat_ref, cfg)
    # If ||e_t|| <= q ||hhat_t||, triangle inequality gives
    # ||e_t||/||hhat_ref|| <= q(1+rho). q+rho alone drops this factor.
    epsilon_est = state.epsilon_est_calibrated * (1 + rho)
    total = epsilon_est + rho
    threshold = cfg.eta_trigger * state.certificate.epsilon_cert
    if not state.valid_for(cfg):
        return TriggerDecision(True, "configuration_or_certificate_invalidated", rho, epsilon_est, total, threshold)
    if policy == "always_reconfigure":
        triggered, reason = True, "every_csi_update"
    elif policy == "periodic_reconfigure":
        triggered = (time_index - state.reference_time) >= cfg.T_period
        reason = "period_elapsed" if triggered else "period_pending"
    elif policy == "certificate_triggered":
        triggered = total >= threshold
        reason = "certificate_threshold" if triggered else "within_certificate_threshold"
    elif policy == "static":
        triggered, reason = False, "static_baseline"
    else:
        raise ValueError("unknown policy")
    return TriggerDecision(triggered, reason, rho, epsilon_est, total, threshold)
