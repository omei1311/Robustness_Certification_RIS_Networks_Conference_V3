"""Online policies consume estimated physical CSI; no true channels or errors."""
from dataclasses import asdict
from time import perf_counter
import numpy as np
from journal_sim.core.channels import effective_channels
from journal_sim.core.power import transition_energy
from journal_sim.design.candidate_pool import build_candidate_pool
from journal_sim.design import selection
from .offline_calibration import offline_radius
from .trigger import trigger_decision, reset_reference


class PolicyRunner:
    def __init__(self, policy, cfg, seed):
        self.policy, self.cfg, self.seed = policy, cfg, seed
        self.state = None
        self.design_records = []
        self.candidate_arrays = {}
        self.epsilon_est, self.calibration_provenance = offline_radius(cfg)

    def step(self, estimated_channel, time_index):
        start = perf_counter()
        old = self.state
        old_theta = np.ones(self.cfg.N, complex) if old is None else old.configuration.theta
        current_H = effective_channels(estimated_channel, old_theta, self.cfg)
        decision = trigger_decision(self.policy, old, current_H, time_index, self.cfg)
        event = dict(seed=self.seed, time_index=time_index, time=time_index * self.cfg.slot_duration,
                     policy=self.policy, old_configuration=None if old is None else old.configuration.configuration_id,
                     new_configuration=None, epsilon_cert_old=None if old is None else old.certificate.epsilon_cert,
                     **asdict(decision), trigger_threshold=decision.threshold, installed=False,
                     reconfigured=False, design_status="REUSE", switching_energy=0., controller_energy=0.,
                     n_changed_elements=0, n_changed_bits=0, algorithm_calls=0,
                     fast_oracle_calls=0, strict_oracle_calls=0,
                     candidate_generation_runtime=0., certificate_runtime=0., strict_validation_runtime=0.,
                     calibration_runtime=0.)
        if decision.triggered:
            event["algorithm_calls"] = 1
            pool, stats = build_candidate_pool(estimated_channel, self.cfg, self.seed, time_index)
            for c in pool:
                c.epsilon_est = self.epsilon_est
            rule = self.cfg.selection_rule
            chooser = getattr(selection, "select_" + rule)
            outcome = chooser(pool, old_theta, self.cfg) if rule == "lifetime_aware" else chooser(pool, self.cfg)
            c = outcome.candidate
            event["design_status"] = outcome.status
            if c is not None:
                self.state = reset_reference(c.configuration, c.certificate, c.H_hat,
                                             time_index, c.epsilon_est, self.cfg)
                transition = transition_energy(old_theta, c.configuration.theta, self.cfg)
                event.update(installed=True, reconfigured=old is not None,
                             **asdict(transition), new_configuration=c.configuration.configuration_id)
            elif old is not None:
                # Keep the physical old X, but record the failed redesign and
                # expired budget. Never relabel that reuse as certified.
                event["new_configuration"] = old.configuration.configuration_id
            self.design_records.append(dict(seed=self.seed, time_index=time_index, policy=self.policy,
                                            stats=stats, selection_status=outcome.status, scores=outcome.scores,
                                            candidates=[c.to_record() for c in pool],
                                            offline_calibration=self.calibration_provenance))
            for c in pool:
                key = f"t{time_index}_c{c.index}"
                self.candidate_arrays[key + "_w"] = c.configuration.w
                self.candidate_arrays[key + "_theta"] = c.configuration.theta
                self.candidate_arrays[key + "_H_hat"] = c.H_hat
            for name in ("candidate_generation_runtime", "certificate_runtime", "strict_validation_runtime",
                         "fast_oracle_calls", "strict_oracle_calls"):
                event[name] = stats[name]
        state = self.state
        event.update(configuration_id=None if state is None else state.configuration.configuration_id,
                     epsilon_est_offline=self.epsilon_est, offline_calibration=self.calibration_provenance,
                     epsilon_cert=0. if state is None else state.certificate.epsilon_cert,
                     epsilon_est_calibrated=None if state is None else state.epsilon_est_calibrated,
                     reference_time=None if state is None else state.reference_time,
                     certificate_status=None if state is None else state.certificate.status)
        if state is not None:
            post = trigger_decision("certificate_triggered", state,
                                    effective_channels(estimated_channel, state.configuration.theta, self.cfg), time_index, self.cfg)
            event.update(rho_obs_after=post.rho_obs, rho_total_after=post.rho_total,
                         epsilon_est_after=post.epsilon_est,
                         certified_budget_hold=state.valid_for(self.cfg) and post.rho_total < state.certificate.epsilon_cert)
        else:
            event.update(rho_obs_after=None, rho_total_after=None, epsilon_est_after=None, certified_budget_hold=False)
        event["runtime"] = perf_counter() - start
        return None if state is None else state.configuration, event
