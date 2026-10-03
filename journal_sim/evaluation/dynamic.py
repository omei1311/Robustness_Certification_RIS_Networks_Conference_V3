"""Pair true trajectories/noise once; online policy runners see estimates only."""
from time import perf_counter
import numpy as np
from journal_sim.core.channels import effective_channels
from journal_sim.core.models import array_digest
from journal_sim.core.sinr import compute_sinr, rate
from journal_sim.core.power import system_power
from journal_sim.dynamics.channel_process import channel_trajectory, trajectory_id
from journal_sim.dynamics.csi_estimation import estimate_physical, csi_observation_seed
from journal_sim.dynamics.reconfiguration import PolicyRunner
from .metrics import long_term_metrics


def paired_observations(cfg, seed):
    trajectory = channel_trajectory(cfg, seed)
    estimates = tuple(estimate_physical(c, cfg, csi_observation_seed(cfg, seed, t))
                      for t, c in enumerate(trajectory))
    return trajectory, estimates


def evaluate_configuration(X, true_channel, cfg):
    theta = np.ones(cfg.N) if X is None else X.theta
    w = np.zeros((cfg.L, cfg.K, cfg.M), complex) if X is None else X.w
    H_true = effective_channels(true_channel, theta, cfg)
    sinr_min = float(compute_sinr(w, H_true, cfg).min())
    powers = system_power(w, theta, cfg)
    R = rate(w, H_true, cfg)
    return dict(sinr_min=sinr_min, qos_hold=sinr_min >= cfg.gamma, rate=R,
                system_power=powers.total, transmission_circuit_power=powers.transmission_circuit,
                ris_static_state_power=powers.ris_static_state, instantaneous_wee=R / powers.total)


def run_paired_policies(cfg, seed, trajectory=None, estimates=None):
    start = perf_counter()
    if trajectory is None and estimates is None:
        trajectory, estimates = paired_observations(cfg, seed)
    if trajectory is None or estimates is None or len(trajectory) != cfg.time_steps or len(estimates) != cfg.time_steps:
        raise ValueError("paired trajectory and observations must span the full horizon")
    trajectory_digest, csi_digest = trajectory_id(trajectory), trajectory_id(estimates)
    records, summaries, designs, arrays = [], [], [], {}
    for name, sequence in (("true", trajectory), ("estimated", estimates)):
        for link in ("h_bu", "h_br", "h_ru"):
            arrays[name + "_" + link] = np.stack([getattr(c, link) for c in sequence])
    for policy in cfg.policies:
        runner = PolicyRunner(policy, cfg, seed)
        policy_records, ws, thetas = [], [], []
        for t in range(cfg.time_steps):
            # Shared observation is identical by object and contents; no error
            # realization is passed to the optimizer/certifier.
            X, event = runner.step(estimates[t], t)
            event.update(evaluate_configuration(X, trajectory[t], cfg),
                         trajectory_id=trajectory_digest, csi_trajectory_id=csi_digest,
                         observation_id=array_digest(estimates[t].h_bu, estimates[t].h_ru, estimates[t].h_br))
            event["csi_noise_seed"] = csi_observation_seed(cfg, seed, t)
            policy_records.append(event)
            ws.append(np.zeros((cfg.L, cfg.K, cfg.M), complex) if X is None else X.w)
            thetas.append(np.ones(cfg.N, complex) if X is None else X.theta)
        records.extend(policy_records)
        summaries.append(dict(seed=seed, policy=policy, trajectory_id=trajectory_digest,
                              csi_trajectory_id=csi_digest, **long_term_metrics(policy_records, cfg)))
        designs.extend(runner.design_records)
        arrays[policy + "_w"] = np.stack(ws)
        arrays[policy + "_theta"] = np.stack(thetas)
        arrays.update({policy + "_" + k: v for k, v in runner.candidate_arrays.items()})
    return dict(records=records, summaries=summaries, designs=designs, arrays=arrays,
                end_to_end_runtime=perf_counter() - start)
