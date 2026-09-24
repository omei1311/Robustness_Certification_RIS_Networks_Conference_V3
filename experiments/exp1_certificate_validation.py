"""Experiment 1 -- Certificate validation (Section 9).

Two complementary layers, kept strictly separate:

(a) DETERMINISTIC certificate consistency check.  For each representative
    configuration, the certified epsilon_cert is evaluated on the normalized
    grid alpha = epsilon / epsilon_cert with key points
    {0.90, 0.95, 0.99, 1.00, 1.01, 1.05, 1.10}.  At each point we record the
    LMI feasibility decision, the minimum normalized LMI margin, and the
    deterministic worst-case SINR (bisection on the QoS target with the same
    LMI oracle).  Expected pattern, up to the numerical tolerance
    (feasibility_tol and the bisection width delta_eps):
      alpha < 1  -> robustly feasible (certificate holds)
      alpha = 1  -> feasible (conservative lower-bound certificate)
      alpha > 1  -> typically the first infeasible grid points appear;
                    boundary statements are reported as "first infeasible
                    grid point" / "deterministic feasibility boundary is
                    consistent with epsilon_cert within numerical
                    tolerance" -- never as a claimed true crossing point.

(b) EMPIRICAL Monte Carlo sweep (validation only; never used to estimate
    epsilon_cert): realized radius swept over 0-3 epsilon_cert, worst-user
    SINR statistics and empirical violation rate from independent ball
    samples, plus the deterministic worst-case SINR curve on the same grid.
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
from src.certificate import (  # noqa: E402
    worst_case_sinr_at_radius,
    worst_case_sinr_curve,
)
from src.ris_base import (  # noqa: E402
    robust_check,
    sample_complex_unit_ball,
    sinr_under_error_samples,
)
from src.uncertainty_helpers import uncertainty_shape  # noqa: E402
from src.plotting import plot_exp1  # noqa: E402
from shared import build_and_certify, make_system  # noqa: E402

# Normalized key points of the deterministic certificate consistency check.
ALPHA_KEY_POINTS = (0.90, 0.95, 0.99, 1.00, 1.01, 1.05, 1.10)


def pick_representatives(pool, wee, rcert, mask, n: int = 4):
    """WEE-max, robustness-max, a mid nondominated point, and a fragile one."""
    chosen = {}
    chosen["max-WEE"] = int(np.argmax(wee))
    chosen["max-eps_cert"] = int(np.argmax(rcert))
    nd = np.flatnonzero(mask)
    if nd.size >= 3:
        mid = nd[np.argsort(wee[nd])[nd.size // 2]]
        chosen["mid-Pareto"] = int(mid)
    chosen["low-eps_cert"] = int(np.argmin(rcert))
    items = list(chosen.items())[:n]
    return items


def certificate_consistency_rows(cand, cfg, label):
    """Deterministic LMI check at alpha * epsilon_cert (no Monte Carlo)."""
    rc = float(cand.r_cert)
    rows = []
    for alpha in ALPHA_KEY_POINTS:
        eps = float(alpha) * rc
        chk = robust_check(cand.w, cand.H, eps, cfg)
        wc_sinr = worst_case_sinr_at_radius(cand.w, cand.H, eps, cfg)
        rows.append(
            {
                "label": label,
                "candidate": cand.index,
                "epsilon_cert": rc,
                "alpha": float(alpha),
                "epsilon": eps,
                "robust_feasible": bool(chk.feasible),
                "minimum_lmi_margin": float(chk.min_margin),
                "worst_case_sinr": float(wc_sinr),
            }
        )
    return rows


def summarize_consistency(rows):
    """Consistency verdict against the certificate (tolerance-aware).

    alpha <= 0.99 must be feasible, alpha = 1.00 must be feasible (the
    certificate is the conservative lower endpoint of the bisection
    bracket), alpha = 1.10 is expected infeasible; alpha = 1.01 may be
    either, because the bisection bracket width (delta_eps = 1e-4) and the
    feasibility tolerance are of that order.
    """
    by_alpha = {r["alpha"]: r for r in rows}
    ok_below = all(by_alpha[a]["robust_feasible"] for a in (0.90, 0.95, 0.99))
    ok_at_one = by_alpha[1.00]["robust_feasible"]
    infeasible_above = [
        a for a in ALPHA_KEY_POINTS if a > 1.0 and not by_alpha[a]["robust_feasible"]
    ]
    first_infeasible = min(infeasible_above) if infeasible_above else None
    consistent = bool(ok_below and ok_at_one and first_infeasible is not None)
    return {
        "feasible_below_one": ok_below,
        "feasible_at_one": ok_at_one,
        "first_infeasible_alpha": first_infeasible,
        "consistent_with_certificate": consistent,
    }


def main() -> dict:
    cfg, cc = make_system()
    pool, meta = build_and_certify(cfg, cc)

    wee = np.array([c.wee for c in pool])
    rcert = np.array([c.r_cert for c in pool])
    mask = pareto_mask(wee, rcert)
    reps = pick_representatives(pool, wee, rcert, mask, n=cc.exp1_n_configs)

    # ---------------- (a) deterministic consistency table -------------- #
    print("=== Experiment 1(a): deterministic certificate consistency ===")
    print(f"{'label':<16s} {'cand':>4s} {'alpha':>6s} {'eps':>8s} {'feas':>5s} "
          f"{'min_margin':>11s} {'wc_SINR':>8s}")
    all_consistency = []
    all_rows = []
    for label, idx in reps:
        cand = pool[idx]
        rows = certificate_consistency_rows(cand, cfg, label)
        all_rows.extend(rows)
        verdict = summarize_consistency(rows)
        verdict.update({"label": label, "candidate": idx,
                        "epsilon_cert": float(cand.r_cert)})
        all_consistency.append(verdict)
        for r in rows:
            print(f"{label:<16s} {idx:>4d} {r['alpha']:>6.2f} {r['epsilon']:>8.5f} "
                  f"{str(r['robust_feasible']):>5s} {r['minimum_lmi_margin']:>11.3e} "
                  f"{r['worst_case_sinr']:>8.3f}")
        if verdict["consistent_with_certificate"]:
            first = verdict["first_infeasible_alpha"]
            print(f"  -> [{label}] first infeasible grid point at alpha="
                  f"{first:.2f}; deterministic feasibility boundary is "
                  f"consistent with epsilon_cert within numerical tolerance.")
        else:
            print(f"  -> [{label}] consistency flags: below_one="
                  f"{verdict['feasible_below_one']}, at_one="
                  f"{verdict['feasible_at_one']}, first_infeasible="
                  f"{verdict['first_infeasible_alpha']}")

    # ---------------- (b) empirical MC sweep (validation only) --------- #
    print("\n=== Experiment 1(b): empirical Monte Carlo sweep (validation only) ===")
    rng = np.random.default_rng(cc.exp1_seed)
    shape = uncertainty_shape(cfg)
    configs = []
    mc_rows = []
    for label, idx in reps:
        cand = pool[idx]
        rc = float(cand.r_cert)
        radii = np.linspace(0.0, cc.exp1_radius_extend * rc, cc.exp1_radius_points)
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
            mc_rows.append(
                {
                    "label": label,
                    "candidate": idx,
                    "r_cert": rc,
                    "epsilon_cert": rc,
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
                "label": f"{label} (idx {idx}, eps_cert={rc:.3f})",
                "r_cert": rc,
                "radii": radii,
                "sinr_mean": np.array(mean_arr),
                "sinr_p5": np.array(p5_arr),
                "sinr_min": np.array(min_arr),
                "sinr_wc": wc,
                "violation": np.array(viol_arr),
            }
        )
        v = np.array(viol_arr)
        first_v = radii[np.flatnonzero(v > 0.0)[0]] if np.any(v > 0.0) else np.inf
        print(f"[exp1-MC] {label}: idx={idx} eps_cert={rc:.4f}; "
              f"first sampled violation at eps={first_v:.4f} "
              f"(alpha={first_v / rc if np.isfinite(first_v) and rc > 0 else float('inf'):.2f}) "
              f"-- empirical validation only, NOT a certificate estimate")

    fig_path = plot_exp1(configs, cfg.gamma)
    save_csv("exp1_certificate_consistency", all_rows)
    save_csv("exp1_mc_sweep", mc_rows)
    summary = {
        "alpha_key_points": list(ALPHA_KEY_POINTS),
        "consistency": all_consistency,
        "representatives": [
            {"label": label, "index": idx, "r_cert": float(pool[idx].r_cert),
             "epsilon_cert": float(pool[idx].r_cert),
             "wee": float(pool[idx].wee)}
            for label, idx in reps
        ],
        "mc_samples": cc.exp1_mc_samples,
        "radius_points": cc.exp1_radius_points,
        "radius_extend": cc.exp1_radius_extend,
        "note": (
            "epsilon_cert is obtained by deterministic LMI bisection only; "
            "the Monte Carlo sweep is an empirical validation and is never "
            "used to estimate epsilon_cert."
        ),
    }
    save_json("exp1_summary", {"manifest": manifest("exp1"), **summary})

    n_consistent = sum(v["consistent_with_certificate"] for v in all_consistency)
    print(f"\ncertificate consistency: {n_consistent}/{len(all_consistency)} "
          f"representatives consistent (first infeasible grid points at "
          f"alpha > 1, feasible at alpha <= 1 within tolerance)")
    print(f"figure -> {fig_path}")
    return summary


if __name__ == "__main__":
    main()
