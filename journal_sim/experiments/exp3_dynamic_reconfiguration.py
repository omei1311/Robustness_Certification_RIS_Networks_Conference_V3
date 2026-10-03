"""Exp3: paired policies on shared physical trajectories and pilot noise."""
from journal_sim.evaluation.dynamic import run_paired_policies
from .common import execute, cli_config, pyplot, save_plot


def plot(rows, details, output, cfg):
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = ("#0072B2", "#E69F00", "#CC79A7", "#6B6B6B")
    styles = ("-", "--", "-.", ":")
    seed = cfg.seeds[0]
    for policy, color, style in zip(cfg.policies, colors, styles):
        group = [r for r in rows if r.get("policy") == policy and r["seed"] == seed]
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
            ees.append(cumulative_bits / cumulative_energy / cfg.bandwidth_hz)
            ns.append(n)
        axes[0, 1].plot(time, ees, ls=style, color=color, label=policy)
        axes[1, 0].step(time, ns, where="post", ls=style, color=color, label=policy)
    summary_rows = [s for detail in details for s in (detail.get("summary") or [])]
    for i, (policy, color) in enumerate(zip(cfg.policies, colors)):
        ys = [100 * s["qos_outage_rate"] for s in summary_rows if s["policy"] == policy]
        axes[1, 1].scatter([i] * len(ys), ys, color=color, s=36)
    axes[1, 1].set_xticks(range(len(cfg.policies)), [p.replace("_reconfigure", "").replace("_", "\n") for p in cfg.policies])
    axes[1, 1].set_ylim(-2, 102)
    axes[1, 1].set_ylabel("QoS outage (%) per requested seed")
    axes[0, 0].axhline(cfg.gamma, color="#333333", lw=1, label="QoS target")
    axes[0, 0].set_ylabel("Worst-user SINR (linear)")
    axes[0, 1].set_ylabel("Cumulative spectral EE (bit/s/Hz/J)")
    axes[1, 0].set_ylabel("Reconfigurations after initialization")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Paired dynamic policies: {cfg.mobility_level}; trajectory seed {seed}; all seeds in CSV")
    save_plot(fig, output)
    plt.close(fig)


if __name__ == "__main__":
    _, success = execute("exp3_dynamic", cli_config(__doc__), run_paired_policies, plot)
    raise SystemExit(0 if success else 1)
