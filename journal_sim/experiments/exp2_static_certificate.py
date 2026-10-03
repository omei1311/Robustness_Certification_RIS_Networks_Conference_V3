"""Exp2: full static candidate pool and four independent selection rules."""
import numpy as np
from journal_sim.core.channels import generate_channel
from journal_sim.dynamics.csi_estimation import estimate_physical
from journal_sim.dynamics.offline_calibration import offline_radius
from journal_sim.design.candidate_pool import build_candidate_pool
from journal_sim.design.selection import select_wee_only, select_robustness_only, select_stability_aware, select_lifetime_aware
from .common import execute, cli_config, pyplot, save_plot


def run_seed(cfg, seed):
    cfg = cfg.with_overrides(pool_size=cfg.static_pool_size)
    epsilon_est, provenance = offline_radius(cfg)
    true = generate_channel(cfg, seed)
    estimated = estimate_physical(true, cfg, cfg.csi_seed + seed)
    pool, stats = build_candidate_pool(estimated, cfg, seed)
    arrays = dict(true_h_bu=true.h_bu, true_h_br=true.h_br, true_h_ru=true.h_ru,
                  estimated_h_bu=estimated.h_bu, estimated_h_ru=estimated.h_ru)
    for c in pool:
        c.epsilon_est = epsilon_est
        arrays[f"c{c.index}_w"] = c.configuration.w
        arrays[f"c{c.index}_theta"] = c.configuration.theta
        arrays[f"c{c.index}_H_hat"] = c.H_hat
    outcomes = [select_wee_only(pool, cfg), select_robustness_only(pool, cfg), select_stability_aware(pool, cfg),
                select_lifetime_aware(pool, np.ones(cfg.N, complex), cfg)]
    rows = [dict(row_type="candidate", **c.to_record(), selected_by=[o.rule for o in outcomes if o.candidate is c]) for c in pool]
    rows.extend(dict(row_type="attempt", **a) for a in stats["attempts"])
    rows.extend(dict(row_type="selection", rule=o.rule, selection_status=o.status,
                     selected_index=None if o.candidate is None else o.candidate.index, scores=o.scores) for o in outcomes)
    return dict(records=rows, arrays=arrays, details=[dict(stats=stats, offline_calibration=provenance, local_config=cfg.to_dict())],
                summary=dict(pool_size=len(pool), selections={o.rule: None if o.candidate is None else o.candidate.index for o in outcomes},
                             statuses={o.rule: o.status for o in outcomes}))


def plot(rows, details, output, cfg):
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for seed in cfg.seeds:
        group = [r for r in rows if r.get("row_type") == "candidate" and r["seed"] == seed]
        valid = [r for r in group if r["strict_validation_passed"]]
        invalid = [r for r in group if not r["strict_validation_passed"]]
        ax.scatter([r["epsilon_cert"] for r in valid], [r["spectral_wee"] for r in valid], facecolors="none", edgecolors="#0072B2", label=f"Strict lower bounds, seed {seed}")
        ax.scatter([r["epsilon_cert"] for r in invalid], [r["spectral_wee"] for r in invalid], marker="x", color="#6B6B6B", label=f"Uncertain/infeasible, seed {seed}")
        for rule, color, marker in (("wee_only", "#E69F00", "^"), ("robustness_only", "#CC79A7", "s"), ("lifetime_aware", "#8D913D", "D")):
            chosen = [r for r in group if rule in r["selected_by"]]
            ax.scatter([r["epsilon_cert"] for r in chosen], [r["spectral_wee"] for r in chosen], marker=marker, color=color, s=65, label=f"{rule}, seed {seed}")
    ax.set_xlabel("Validated relative-radius lower bound")
    ax.set_ylabel("Spectral WEE (bit/s/Hz/W)")
    ax.set_title(f"Full candidate pool (direct intercell {'ON' if cfg.include_direct_intercell else 'OFF'})")
    ax.legend(fontsize=8, loc="best")
    save_plot(fig, output)
    plt.close(fig)


if __name__ == "__main__":
    cfg, execution = cli_config(__doc__)
    _, success = execute("exp2_static", cfg, run_seed, plot, workers=execution.workers)
    raise SystemExit(0 if success else 1)
