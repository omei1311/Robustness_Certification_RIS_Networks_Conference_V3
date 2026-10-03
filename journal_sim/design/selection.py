"""Full-pool comparisons; transition costs can make a dominated point useful.

The predicted reuse time is derived from exactly the same accounting the
online trigger uses (dynamics/trigger.py): reuse while

    rho_total(t) = epsilon_est * (1 + rho(t)) + rho(t) < eta_trigger * eps_cert.

Under the linear drift proxy rho(t) = nu * t this boundary is reached at

    T_pred = (eta_trigger * eps_cert - epsilon_est) / ((1 + epsilon_est) * nu),

and T_pred = 0 whenever epsilon_est >= eta_trigger * eps_cert, i.e. the
candidate has no trigger safety margin even at installation time (rho = 0)
and can never be eligible for the lifetime-aware rule.
"""
from dataclasses import dataclass
import numpy as np
from journal_sim.core.power import transition_energy


@dataclass
class SelectionResult:
    rule: str
    candidate: object | None
    scores: list
    status: str


def certified_lifetime(epsilon_cert, epsilon_est, nu, eta_trigger):
    """Trigger-consistent predicted reuse time under rho(t) = nu * t."""
    if nu <= 0 or epsilon_est < 0 or epsilon_cert < 0:
        raise ValueError("positive drift rate and nonnegative radii/certificate required")
    if not 0 < eta_trigger <= 1:
        raise ValueError("trigger safety factor must lie in (0, 1]")
    budget = eta_trigger * epsilon_cert - epsilon_est
    if budget <= 0:
        return 0.0
    return budget / ((1.0 + epsilon_est) * nu)


def _select(pool, rule, cfg, theta_old=None, epsilon_est=None):
    scores = []
    eligible = []
    for c in pool:
        q = c.epsilon_est if epsilon_est is None else epsilon_est
        lifetime = certified_lifetime(c.epsilon_cert, q, cfg.drift_rate_nu, cfg.eta_trigger)
        energy = 0. if theta_old is None else transition_energy(theta_old, c.configuration.theta, cfg).total
        ee_life = c.rate * lifetime / (c.system_power * lifetime + energy) if lifetime > 0 else None
        valid = c.valid_certificate
        if rule == "lifetime_aware":
            valid = valid and lifetime > 0
            score = ee_life
        elif rule == "wee_only":
            score = c.wee
        elif rule == "robustness_only":
            score = c.epsilon_cert
        elif rule == "stability_aware":
            valid = valid and c.epsilon_cert >= cfg.epsilon_min
            score = c.wee
        else:
            raise ValueError("unknown selection rule")
        scores.append(dict(candidate_index=c.index, eligible=bool(valid), score=score,
                           predicted_lifetime=lifetime, EE_life=ee_life, transition_energy=energy,
                           certificate_status=c.certificate.status))
        if valid and score is not None and np.isfinite(score):
            eligible.append((score, -c.index, c))
    winner = max(eligible, key=lambda row: row[:2])[2] if eligible else None
    return SelectionResult(rule, winner, scores, "SELECTED" if winner is not None else "NO_ELIGIBLE_CANDIDATE")


def select_lifetime_aware(pool, theta_old, cfg, epsilon_est=None):
    return _select(pool, "lifetime_aware", cfg, theta_old, epsilon_est)


def select_wee_only(pool, cfg):
    return _select(pool, "wee_only", cfg)


def select_robustness_only(pool, cfg):
    return _select(pool, "robustness_only", cfg)


def select_stability_aware(pool, cfg):
    return _select(pool, "stability_aware", cfg)
