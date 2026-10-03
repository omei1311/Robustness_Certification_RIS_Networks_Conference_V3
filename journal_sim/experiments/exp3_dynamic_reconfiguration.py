"""Exp3: paired policies on shared physical trajectories and pilot noise."""
import json
from pathlib import Path
from journal_sim.evaluation.dynamic import run_paired_policies
from journal_sim.dynamics.offline_calibration import offline_radius
from journal_sim.dynamics.calibration_guards import formal_calibration_guard
import numpy as np
from .common import execute, cli_config, pyplot, save_plot, serializable


def run_mobility(cfg, seed, mobility):
    """One mobility regime end to end: paired trajectory, all policies, shared pool cache."""
    local = cfg.with_overrides(mobility_level=mobility)
    result = run_paired_policies(local, seed)
    return dict(
        records=[dict(mobility=mobility, channel_correlation=local.correlation, **r) for r in result["records"]],
        summaries=[dict(mobility=mobility, channel_correlation=local.correlation, **r) for r in result["summaries"]],
        designs=[dict(mobility=mobility, **d) for d in result["designs"]],
        arrays={mobility + "_" + key: value for key, value in result["arrays"].items()},
        pool_cache=result.get("pool_cache", dict(candidate_pool_requests=0, candidate_pool_builds=0,
                                                 candidate_pool_cache_hits=0, candidate_pool_cache_hit_rate=0.)),
        end_to_end_runtime=result["end_to_end_runtime"])


def _stage_mobility(staging_dir, seed, mobility, result):
    """Atomic per-mobility checkpoint: only the worker's own staging is written."""
    directory = Path(staging_dir) / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in result.items() if key != "arrays"}
    result_tmp = directory / f"{mobility}_result.json.tmp"
    result_tmp.write_text(json.dumps(serializable(payload), ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")
    result_tmp.replace(directory / f"{mobility}_result.json")
    arrays_tmp = directory / f"{mobility}_arrays.tmp.npz"
    np.savez_compressed(arrays_tmp, **result["arrays"])
    arrays_tmp.replace(directory / f"{mobility}_arrays.npz")


def run_seed(cfg, seed, staging_dir=None):
    offline_radius(cfg)  # Validate offline provenance before any redesign.
    records, summaries, designs, arrays = [], [], [], {}
    runtime = 0.
    pool_stats = []
    for mobility in cfg.mobility_regimes:
        result = run_mobility(cfg, seed, mobility)
        records.extend(result["records"])
        summaries.extend(result["summaries"])
        designs.extend(result["designs"])
        arrays.update(result["arrays"])
        runtime += result["end_to_end_runtime"]
        pool_stats.append(dict(mobility=mobility, **result["pool_cache"]))
        if staging_dir is not None:
            _stage_mobility(staging_dir, seed, mobility, result)
    return dict(records=records, summaries=summaries, designs=designs, arrays=arrays,
                pool_cache=pool_stats, end_to_end_runtime=runtime)


def plot_trajectory(rows, details, output, cfg):
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = ("#0072B2", "#E69F00", "#CC79A7", "#8D913D", "#6B6B6B")
    styles = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))
    seed = cfg.seeds[0]
    for policy, color, style in zip(cfg.policies, colors, styles):
        group = [r for r in rows if r.get("policy") == policy and r["seed"] == seed and r.get("mobility") == cfg.mobility_level]
        if not group:
            continue
        time = [r["time"] for r in group]
        axes[0, 0].plot(time, [r["sinr_min"] for r in group], ls=style, color=color, label=policy)
        cumulative_bits = cumulative_energy = n = 0
        ees, ns = [], []
        for r in group:
            cumulative_bits += r["rate"] * cfg.slot_duration
            cumulative_energy += r["system_power"] * cfg.slot_duration + r["switching_energy"] + r["controller_energy"]
            n += int(r["reconfigured"])
            ees.append(cumulative_bits / cumulative_energy)
            ns.append(n)
        axes[0, 1].plot(time, ees, ls=style, color=color, label=policy)
        axes[1, 0].step(time, ns, where="post", ls=style, color=color, label=policy)
    summary_rows = [s for detail in details for s in (detail.get("summary") or []) if s.get("mobility") == cfg.mobility_level]
    for i, (policy, color) in enumerate(zip(cfg.policies, colors)):
        ys = [100 * s["qos_outage_rate"] for s in summary_rows if s["policy"] == policy]
        axes[1, 1].scatter([i] * len(ys), ys, color=color, s=36)
    axes[1, 1].set_xticks(range(len(cfg.policies)), [p.replace("_reconfigure", "").replace("_", "\n") for p in cfg.policies])
    axes[1, 1].set_ylim(-2, 102)
    axes[1, 1].set_ylabel("QoS outage (%) per requested seed")
    axes[0, 0].axhline(cfg.gamma, color="#333333", lw=1, label="QoS target")
    axes[0, 0].set_ylabel("Worst-user SINR (linear)")
    axes[0, 1].set_ylabel("Cumulative long-term EE (bit/J)")
    axes[1, 0].set_ylabel("Reconfigurations after initialization")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Paired dynamic policies: {cfg.mobility_level}; trajectory seed {seed}; all seeds in CSV")
    save_plot(fig, output, "trajectory_" + cfg.mobility_level)
    plt.close(fig)


def plot(rows, details, output, cfg):
    plt = pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    summaries = [s for d in details for s in (d.get("summary") or [])]
    metrics = (("qos_outage_rate", "QoS outage (%)", 100), ("long_term_ee", "Long-term EE (bit/J)", 1),
               ("number_of_reconfigurations", "Reconfigurations", 1), ("mean_reuse_interval", "Mean reuse interval (s)", 1),
               ("ris_switching_energy", "RIS switching energy (J)", 1), ("runtime", "Online wall-clock time (s)", 1))
    colors = ("#0072B2", "#E69F00", "#CC79A7", "#8D913D", "#6B6B6B")
    markers = ("o", "s", "^", "D", "x")
    for ax, (field, ylabel, scale) in zip(axes.flat, metrics):
        for i, (policy, color, marker) in enumerate(zip(cfg.policies, colors, markers)):
            offset = (i - (len(cfg.policies) - 1) / 2) * .12
            for j, mobility in enumerate(cfg.mobility_regimes):
                ys = [s[field] * scale for s in summaries if s["policy"] == policy and s["mobility"] == mobility and s[field] is not None]
                if ys:
                    ax.scatter([j + offset] * len(ys), ys, color=color, alpha=.3, s=18, marker=marker)
                    ax.scatter([j + offset], [float(np.median(ys))], color=color, s=50, marker=marker,
                               label=policy if j == 0 else None)
        ax.set_xticks(range(len(cfg.mobility_regimes)), cfg.mobility_regimes)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .98), ncol=5, fontsize=8)
    fig.suptitle(f"Three mobility regimes; raw seed observations and medians (requested seeds={len(cfg.seeds)})", y=1.025)
    save_plot(fig, output)
    plt.close(fig)
    for mobility in cfg.mobility_regimes:
        plot_trajectory(rows, details, output, cfg.with_overrides(mobility_level=mobility))


if __name__ == "__main__":
    cfg, execution = cli_config(__doc__)
    formal_calibration_guard(cfg, "exp3")  # formal runs need formal CSI + drift artifacts
    _, success = execute("exp3_dynamic", cfg, run_seed, plot, workers=execution.workers)
    raise SystemExit(0 if success else 1)
