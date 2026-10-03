"""Exp1: estimation SNR to empirical relative-radius quantiles."""
import numpy as np
from journal_sim.core.channels import generate_channel, effective_channels
from journal_sim.dynamics.csi_estimation import calibrate_csi, calibrate_physical
from .common import execute, cli_config, pyplot, save_plot


def run_seed(cfg, seed):
    channel = generate_channel(cfg, seed)
    theta = np.ones(cfg.N, complex)
    H = effective_channels(channel, theta, cfg)
    rows, arrays = [], dict(h_bu=channel.h_bu, h_br=channel.h_br, h_ru=channel.h_ru, theta=theta)
    for i, snr in enumerate(cfg.calibration_snr_grid):
        local = cfg.with_overrides(estimation_snr_db=snr)
        for estimator, function in (("isotropic_effective", calibrate_csi), ("physical_pilot", calibrate_physical)):
            calibration_seed = int(np.random.SeedSequence([cfg.calibration_seed, seed, i]).generate_state(1)[0])
            result = function(H, local, calibration_seed) if estimator == "isotropic_effective" else function(channel, theta, local, calibration_seed)
            rows.append(dict(estimator=estimator, **{k: v for k, v in result.items() if not k.endswith("distribution")}))
            arrays[f"snr{i}_{estimator}_relative_errors"] = result["relative_error_distribution"]
            arrays[f"snr{i}_{estimator}_joint_relative_errors"] = result["joint_relative_error_distribution"]
    return dict(records=rows, arrays=arrays)


def plot(rows, details, output, cfg):
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, estimator in zip(axes, ("isotropic_effective", "physical_pilot")):
        for q, color, marker in ((90, "#0072B2", "o"), (95, "#E69F00", "s"), (99, "#CC79A7", "^")):
            for seed in cfg.seeds:
                group = [r for r in rows if r.get("estimator") == estimator and r["seed"] == seed]
                if group:
                    ax.plot([r["estimation_snr_db"] for r in group], [r[f"epsilon_{q}"] for r in group],
                            color=color, marker=marker, label=f"{q}% (seed {seed})")
        ax.set_title(estimator.replace("_", " "))
        ax.set_xlabel("Estimation SNR (dB)")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
    axes[0].set_ylabel("Empirical relative radius (per-user quantile)")
    save_plot(fig, output)
    plt.close(fig)


if __name__ == "__main__":
    _, success = execute("exp1_csi", cli_config(__doc__), run_seed, plot)
    raise SystemExit(0 if success else 1)
