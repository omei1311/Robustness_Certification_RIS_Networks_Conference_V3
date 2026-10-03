"""Exp1b: mobility-specific drift-rate calibration for the lifetime predictor.

Replays the exact online observation pipeline on dedicated calibration drops
(disjoint from CSI calibration and evaluation seeds) and reports, per
mobility regime, empirical quantiles of rho(delta_t)/delta_t - where rho is
the trigger's own relative_drift() on estimated effective channels. The
selected nu (default joint 95% quantile) feeds ONLY the trigger-consistent
lifetime predictor used for candidate ranking; the online trigger keeps
deciding on the actually observed rho_total.
"""
import numpy as np
from journal_sim.dynamics.drift_calibration import (drift_samples, summarize_drift, drift_scope_fingerprint,
                                                    drift_calibration_scope)
from .common import execute, cli_config, pyplot, save_plot, save_json


def run_seed(cfg, seed):
    return dict(records=drift_samples(cfg, seed), arrays={})


def plot(rows, details, output, cfg):
    mobility_summary = summarize_drift(rows, cfg)
    save_json(output / "drift_rate.json", dict(
        schema_version=1, calibration_type="journal_offline_drift_rate_v1",
        mode="smoke" if cfg.smoke else "formal",
        quantile=cfg.drift_calibration_q, unit="1/s",
        mobility=mobility_summary,
        seeds=sorted({r["seed"] for r in rows}),
        reference_stride=cfg.drift_reference_stride, horizons_slots=list(cfg.drift_horizon_slots),
        slot_duration=cfg.slot_duration, scope=drift_calibration_scope(cfg),
        scope_fingerprint=drift_scope_fingerprint(cfg),
        interpretation="Empirical offline drift-rate proxy over estimated-CSI relative drift, used only for lifetime prediction and candidate ranking; the online trigger uses the actually observed drift and the certificate definition is unchanged.",
        source="records.csv", config=cfg.to_dict(),
        requested_seed_count=len(cfg.seeds), completed_seed_count=sum(d["status"] == "COMPLETED" for d in details),
        complete=all(d["status"] == "COMPLETED" for d in details)))

    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = {"slow": "#0072B2", "medium": "#E69F00", "fast": "#CC79A7"}
    for mobility in cfg.mobility_regimes:
        group = [r for r in rows if r["mobility"] == mobility]
        horizons = sorted({r["delta_time"] for r in group})
        median = [float(np.median([r["rho_obs"] for r in group if r["delta_time"] == h])) for h in horizons]
        upper = [float(np.quantile([r["rho_obs"] for r in group if r["delta_time"] == h], .95)) for h in horizons]
        color = colors[mobility]
        ax.plot(horizons, median, color=color, marker="o", label=f"{mobility}: median rho")
        ax.plot(horizons, upper, color=color, marker="s", ls="--", alpha=.7, label=f"{mobility}: 95% rho")
        nu = mobility_summary[mobility]["nu_selected"]
        ax.plot(horizons, [nu * h for h in horizons], color=color, ls=":", lw=2,
                label=f"{mobility}: nu·dt ({nu:.3g}/s)")
    ax.set_xlabel("Reuse interval dt [s]")
    ax.set_ylabel("Observed relative drift rho (estimated CSI)")
    ax.set_title(f"Drift calibration over {len({r['seed'] for r in rows})} calibration seeds; "
                 f"linear proxy nu at q={cfg.drift_calibration_q}")
    ax.legend(fontsize=8)
    save_plot(fig, output)
    plt.close(fig)


if __name__ == "__main__":
    cfg, execution = cli_config(__doc__)
    # Drift calibration runs on its own seed set, never on evaluation seeds
    # and never on the CSI calibration seeds (--n-seeds selects its prefix).
    count = len(cfg.seeds)
    seeds = (cfg.drift_calibration_seeds if count >= len(cfg.drift_calibration_seeds)
             else cfg.drift_calibration_seeds[:count])
    cfg = cfg.with_overrides(seeds=seeds)
    _, success = execute("exp1b_drift", cfg, run_seed, plot, workers=execution.workers)
    raise SystemExit(0 if success else 1)
