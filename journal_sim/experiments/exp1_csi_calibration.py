"""Exp1: estimation SNR to empirical relative-radius quantiles."""
import numpy as np
from journal_sim.core.channels import generate_channel, effective_channels
from journal_sim.dynamics.csi_estimation import calibrate_csi, calibrate_physical
from journal_sim.dynamics.offline_calibration import calibration_scope, scope_fingerprint
from journal_sim.core.models import phase_set
from .common import execute, cli_config, pyplot, save_plot, save_json


def run_seed(cfg, seed):
    rows, arrays = [], {}
    for N, bits in cfg.calibration_cases or ((cfg.N, cfg.bits),):
        base = cfg.with_overrides(N=N, bits=bits)
        channel = generate_channel(base, seed)
        prefix = f"N{N}_B{bits}"
        arrays.update({prefix + "_" + link: getattr(channel, link) for link in ("h_bu", "h_br", "h_ru")})
        for phase in range(cfg.calibration_phase_profiles):
            phase_seed = int(np.random.SeedSequence([cfg.calibration_seed, seed, N, bits, phase]).generate_state(1)[0])
            theta = np.ones(N, complex) if phase == 0 else phase_set(bits)[np.random.default_rng(phase_seed).integers(2 ** bits, size=N)]
            H = effective_channels(channel, theta, base)
            arrays[f"{prefix}_phase{phase}_theta"] = theta
            for i, snr in enumerate(cfg.calibration_snr_grid):
                local = base.with_overrides(estimation_snr_db=snr)
                for estimator, function in (("isotropic_effective", calibrate_csi), ("physical_pilot", calibrate_physical)):
                    calibration_seed = int(np.random.SeedSequence([phase_seed, i]).generate_state(1)[0])
                    result = function(H, local, calibration_seed) if estimator == "isotropic_effective" else function(channel, theta, local, calibration_seed)
                    rows.append(dict(estimator=estimator, N=N, bits=bits, phase_profile=phase,
                                     phase_seed=phase_seed, scope=calibration_scope(local), scope_fingerprint=scope_fingerprint(local),
                                     **{k: v for k, v in result.items() if not k.endswith("distribution")}))
                    key = f"{prefix}_phase{phase}_snr{i}_{estimator}"
                    arrays[key + "_relative_errors"] = result["relative_error_distribution"]
                    arrays[key + "_joint_relative_errors"] = result["joint_relative_error_distribution"]
    return dict(records=rows, arrays=arrays)


def plot(rows, details, output, cfg):
    # Publish offline parameters once, after all requested calibration seeds.
    # System-level value = max over calibration seeds of the per-seed joint
    # quantile (per-seed value = max over phase profiles). Averaging would
    # dilute the safety budget and is deliberately not used.
    grouped = {}
    for r in rows:
        if r.get("estimator") == "physical_pilot":
            grouped.setdefault(r["scope_fingerprint"], []).append(r)
    entries = []
    for fingerprint, group in grouped.items():
        seeds = sorted({r["seed"] for r in group})
        entry = dict(scope_fingerprint=fingerprint, scope=group[0]["scope"],
                     mode="smoke" if cfg.smoke else "formal",
                     calibration_seeds=seeds, profile_count=len(group) // len(seeds),
                     samples_per_profile=cfg.calibration_samples)
        per_seed = []
        for seed in seeds:
            record = dict(seed=seed)
            for q in cfg.calibration_quantiles:
                key = f"epsilon_joint_{round(q * 100)}"
                record[key] = max(r[key] for r in group if r["seed"] == seed)
            per_seed.append(record)
        aggregate = {k: max(record[k] for record in per_seed)
                     for k in per_seed[0] if k != "seed"}
        entry["per_seed"] = per_seed
        entry["aggregate"] = aggregate
        entry.update(aggregate)  # flat keys kept for the offline_radius reader
        entries.append(entry)
    save_json(output / "joint_radius.json", dict(schema="journal_offline_joint_radius_v1", entries=entries,
              interpretation="Offline maximum over calibration seeds of empirical all-user joint relative-error quantiles (per seed: max over phase profiles). This is an empirical offline calibration radius based on finite calibration drops; it is not a deterministic future-horizon guarantee and not a robust certificate.",
              source="records.csv", mode="smoke" if cfg.smoke else "formal", config=cfg.to_dict(),
              requested_seed_count=len(cfg.seeds), completed_seed_count=sum(d["status"] == "COMPLETED" for d in details),
              complete=all(d["status"] == "COMPLETED" for d in details)))
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    styles = {90: ("#009E73", "^"), 95: ("#0072B2", "o"), 99: ("#E69F00", "s")}
    for ax, estimator in zip(axes, ("isotropic_effective", "physical_pilot")):
        group = [r for r in rows if r.get("estimator") == estimator and r["N"] == cfg.N and r["bits"] == cfg.bits]
        snrs = sorted({r["estimation_snr_db"] for r in group})
        for q, (color, marker) in styles.items():
            if snrs and any(f"epsilon_joint_{q}" in r for r in group):
                ax.plot(snrs, [max(r[f"epsilon_joint_{q}"] for r in group if r["estimation_snr_db"] == snr)
                               for snr in snrs], color=color, marker=marker, label=f"joint {q}% (max over drops)")
        if estimator == "physical_pilot" and snrs:
            q = round(cfg.calibration_q * 100)
            operating = [max(r[f"epsilon_joint_{q}"] for r in group if r["estimation_snr_db"] == snr)
                         for snr in snrs]
            ax.plot(snrs, operating, color="#CC79A7", marker="*", ms=13, ls="none",
                    label=f"operating point (joint {q}%)")
            ax.axvline(cfg.estimation_snr_db, color="#999999", lw=1, ls="--")
        ax.set_title(estimator.replace("_", " "))
        ax.set_xlabel("Estimation SNR (dB)")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
    axes[0].set_ylabel("Empirical all-user joint relative radius")
    save_plot(fig, output)
    plt.close(fig)


if __name__ == "__main__":
    cfg, execution = cli_config(__doc__)
    # Calibration never touches evaluation seeds: exp1 always runs on the
    # dedicated CSI calibration seed set (--n-seeds selects its prefix).
    count = len(cfg.seeds)
    seeds = cfg.calibration_seeds if count >= len(cfg.calibration_seeds) else cfg.calibration_seeds[:count]
    cfg = cfg.with_overrides(seeds=seeds)
    _, success = execute("exp1_csi", cfg, run_seed, plot, workers=execution.workers)
    raise SystemExit(0 if success else 1)
