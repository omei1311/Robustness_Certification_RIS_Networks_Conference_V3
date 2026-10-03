"""Exp4: representative N/B cases, end-to-end dynamic computation costs."""
import numpy as np
from journal_sim.config import config_fingerprint
from journal_sim.evaluation.dynamic import run_paired_policies
from .common import execute, cli_config, pyplot, save_plot


def run_seed(cfg, seed):
    records, details, arrays = [], [], {}
    for N, bits in cfg.scaling_cases:
        local = cfg.with_overrides(N=N, bits=bits, time_steps=cfg.runtime_time_steps, policies=cfg.runtime_policies)
        result = run_paired_policies(local, seed)
        rows = result["records"]
        records.append(dict(N=N, bits=bits, seed=seed, local_fingerprint=config_fingerprint(local),
                            candidate_generation_runtime=sum(r["candidate_generation_runtime"] for r in rows),
                            certificate_runtime=sum(r["certificate_runtime"] for r in rows),
                            strict_validation_runtime=sum(r["strict_validation_runtime"] for r in rows),
                            calibration_runtime=sum(r["calibration_runtime"] for r in rows),
                            dynamic_average_computation_cost=sum(r["runtime"] for r in rows) / len(rows),
                            end_to_end_runtime=result["end_to_end_runtime"],
                            fast_oracle_calls=sum(r["fast_oracle_calls"] for r in rows),
                            strict_oracle_calls=sum(r["strict_oracle_calls"] for r in rows),
                            candidate_generation_calls=sum(r["algorithm_calls"] for r in rows),
                            failed_design_steps=sum(r["design_status"] == "NO_ELIGIBLE_CANDIDATE" for r in rows),
                            uncertainty_dimension=local.L * local.M,
                            note="certificate_runtime includes strict_validation_runtime; runtime is not physical energy"))
        details.append(dict(N=N, bits=bits, config=local.to_dict(), fingerprint=config_fingerprint(local),
                            summaries=result["summaries"], dynamic_records=rows, designs=result["designs"]))
        arrays.update({f"N{N}_B{bits}_{key}": value for key, value in result["arrays"].items()})
    return dict(records=records, details=details, arrays=arrays)


def plot(rows, details, output, cfg):
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [f"N={N}\nB={bits}" for N, bits in cfg.scaling_cases]
    for seed in cfg.seeds:
        group = [r for r in rows if r.get("seed") == seed and "N" in r]
        for field, color, marker in (("candidate_generation_runtime", "#0072B2", "o"),
                                     ("certificate_runtime", "#E69F00", "s"),
                                     ("strict_validation_runtime", "#CC79A7", "^")):
            axes[0].plot(range(len(group)), [r[field] for r in group], color=color, marker=marker,
                         label=f"{field.replace('_runtime', '')}, seed {seed}")
        axes[1].plot(range(len(group)), [r["end_to_end_runtime"] for r in group], "o-", color="#0072B2", label=f"End to end, seed {seed}")
        axes[1].plot(range(len(group)), [r["dynamic_average_computation_cost"] for r in group], "s--", color="#E69F00", label=f"Mean step, seed {seed}")
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
    axes[0].set_title("Total component runtimes (strict is a subset)")
    axes[1].set_title("Dynamic end-to-end and per-step costs")
    save_plot(fig, output)
    plt.close(fig)


if __name__ == "__main__":
    _, success = execute("exp4_runtime", cli_config(__doc__), run_seed, plot)
    raise SystemExit(0 if success else 1)
