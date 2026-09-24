"""Experiment 1 -- Certificate validation (Section 9).

For several representative feasible configurations, sweep the realized
channel-deviation radius over a range covering 0 to ~1.5 R_cert and
evaluate the worst-user SINR over independent uncertainty samples drawn
uniformly from the per-user balls.  Panel (a): sampled worst-user SINR with
the QoS target and the R_cert markers; panel (b): empirical QoS violation
rate.  The Monte Carlo sweep empirically validates the deterministic LMI
certificate; it does not replace the certificate calculation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import manifest, save_csv, save_json  # noqa: E402
from src.pareto import pareto_mask  # noqa: E402
from src.certificate import worst_case_sinr_curve  # noqa: E402
from src.ris_base import sample_complex_unit_ball, sinr_under_error_samples  # noqa: E402
from src.uncertainty_helpers import uncertainty_shape  # noqa: E402
from src.plotting import plot_exp1  # noqa: E402
from shared import build_and_certify, make_system  # noqa: E402


def pick_representatives(pool, wee, rcert, mask, n: int = 4):
    """WEE-max, robustness-max, a mid nondominated point, and a fragile one."""
    chosen = {}
    chosen["max-WEE"] = int(np.argmax(wee))
    chosen["max-Rcert"] = int(np.argmax(rcert))
    nd = np.flatnonzero(mask)
    if nd.size >= 3:
        mid = nd[np.argsort(wee[nd])[nd.size // 2]]
        chosen["mid-Pareto"] = int(mid)
    chosen["low-Rcert"] = int(np.argmin(rcert))
    items = list(chosen.items())[:n]
    return items


def main() -> dict:
    cfg, cc = make_system()
    pool, meta = build_and_certify(cfg, cc)

    wee = np.array([c.wee for c in pool])
    rcert = np.array([c.r_cert for c in pool])
    mask = pareto_mask(wee, rcert)
    reps = pick_representatives(pool, wee, rcert, mask, n=cc.exp1_n_configs)

    rng = np.random.default_rng(cc.exp1_seed)
    shape = (cfg.L, cfg.K, cfg.L * cfg.M)  # aggregate uncertainty model
    configs = []
    all_rows = []

    for label, idx in reps:
        cand = pool[idx]
        rc = float(cand.r_cert)
        radii = np.linspace(
            0.0, cc.exp1_radius_extend * rc, cc.exp1_radius_points
        )
        # Deterministic worst-case SINR curve from the same LMI oracle
        # (bisection on the QoS target at each radius).
        wc = worst_case_sinr_curve(cand.w, cand.H, radii, cfg)
        mean_arr, p5_arr, min_arr, viol_arr = [], [], [], []
        viol_tol = cfg.gamma * (1.0 - 1e-6)
        for j, eps_real in enumerate(radii):
            dirs = sample_complex_unit_ball(cc.exp1_mc_samples, shape, rng)
            sinr = sinr_under_error_samples(cand.w, cand.H, eps_real, dirs, cfg)
            worst = sinr.min(axis=(1, 2))
            mean_arr.append(float(worst.mean()))
            p5_arr.append(float(np.percentile(worst, 5)))
            min_arr.append(float(worst.min()))
            viol_arr.append(float(np.mean(worst < viol_tol)))
            all_rows.append(
                {
                    "label": label,
                    "candidate": idx,
                    "r_cert": rc,
                    "eps_real": float(eps_real),
                    "eps_over_rcert": float(eps_real / rc) if rc > 0 else np.inf,
                    "worst_sinr_mean": mean_arr[-1],
                    "worst_sinr_p5": p5_arr[-1],
                    "worst_sinr_min": min_arr[-1],
                    "worst_sinr_deterministic": float(wc[j]),
                    "violation_rate": viol_arr[-1],
                }
            )
        configs.append(
            {
                "label": f"{label} (idx {idx}, R={rc:.3f})",
                "r_cert": rc,
                "radii": radii,
                "sinr_mean": np.array(mean_arr),
                "sinr_p5": np.array(p5_arr),
                "sinr_min": np.array(min_arr),
                "sinr_wc": wc,
                "violation": np.array(viol_arr),
            }
        )
        # violation onset relative to the certificate
        v = np.array(viol_arr)
        onset = radii[np.flatnonzero(v > 0.0)[0]] if np.any(v > 0.0) else np.inf
        below = wc < cfg.gamma
        wc_cross = radii[np.flatnonzero(below)[0]] if np.any(below) else np.inf
        print(f"[exp1] {label}: idx={idx} R_cert={rc:.4f} "
              f"worst-case-curve crosses target at {wc_cross:.4f} "
              f"(ratio {wc_cross / rc if rc > 0 else float('inf'):.2f}); "
              f"first sampled violation at {onset:.4f}")

    fig_path = plot_exp1(configs, cfg.gamma)
    save_csv("exp1_mc_sweep", all_rows)
    summary = {
        "representatives": [
            {"label": label, "index": idx, "r_cert": float(pool[idx].r_cert),
             "wee": float(pool[idx].wee)}
            for label, idx in reps
        ],
        "mc_samples": cc.exp1_mc_samples,
        "radius_points": cc.exp1_radius_points,
        "radius_extend": cc.exp1_radius_extend,
    }
    save_json("exp1_summary", {"manifest": manifest("exp1"), **summary})

    print("=== Experiment 1: certificate validation ===")
    print(f"representatives: {[lbl for lbl, _ in reps]}")
    print(f"samples/radius : {cc.exp1_mc_samples}, radii/config: {cc.exp1_radius_points}")
    print(f"figure -> {fig_path}")
    return summary


if __name__ == "__main__":
    main()
