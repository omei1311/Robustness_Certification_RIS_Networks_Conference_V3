"""Experiment 3 -- Stability-aware selection and requirement sensitivity.

Compares the three selection rules (WEE-only Eq. (17), robustness-only
reference, and the proposed robustness-constrained WEE rule Eq. (18)),
sweeps the minimum robustness requirement R_min over the normalized grid
{0, 0.25, 0.50, 0.75, 0.90} R_max, and reports selected WEE, R_cert, the
conditional T_cert under the stated linear drift model, and the number of
candidates remaining after filtering.  A compact independent-sample check
at the prescribed design radius epsilon closes the experiment.
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
from src.ris_base import sample_complex_unit_ball, sinr_under_error_samples  # noqa: E402
from src.selection import (  # noqa: E402
    r_min_sensitivity,
    select_robustness_only,
    select_stability_aware,
    select_wee_only,
    t_cert,
)
from src.uncertainty_helpers import uncertainty_shape  # noqa: E402
from src.plotting import plot_exp3  # noqa: E402
from shared import build_and_certify, make_system  # noqa: E402


def main() -> dict:
    cfg, cc = make_system()
    pool, meta = build_and_certify(cfg, cc)

    wee = np.array([c.wee for c in pool])
    rcert = np.array([c.r_cert for c in pool])
    r_max = float(rcert.max())
    nu = cc.drift_rate_nu

    # ---- three selection rules ---------------------------------------- #
    out_wee = select_wee_only(wee, rcert, nu)
    out_rob = select_robustness_only(wee, rcert, nu)
    out_sel = select_stability_aware(wee, rcert, cc.epsilon_design, nu)

    rule_rows = []
    for out in (out_wee, out_rob, out_sel):
        rule_rows.append(
            {
                "rule": out.rule,
                "selected_index": out.index,
                "selected_wee": out.wee,
                "selected_rcert": out.rcert,
                "selected_epsilon_cert": out.rcert,
                "selected_t_cert_s": out.t_cert,
                "n_remaining": out.n_remaining,
            }
        )

    # ---- R_min sensitivity sweep --------------------------------------- #
    rows = r_min_sensitivity(wee, rcert, cc.r_min_fracs, nu)

    # ---- independent-sample robustness check at the design radius ------ #
    rng = np.random.default_rng(cc.exp3_seed)
    shape = uncertainty_shape(cfg)
    check = {}
    for name, idx in (
        ("wee_only", out_wee.index),
        ("robustness_only", out_rob.index),
        ("stability_aware", out_sel.index),
    ):
        cand = pool[idx]
        dirs = sample_complex_unit_ball(cc.exp3_check_samples, shape, rng)
        sinr = sinr_under_error_samples(
            cand.w, cand.H, cc.epsilon_design, dirs, cfg
        )
        worst = sinr.min(axis=(1, 2))
        hold = float(np.mean(worst >= cfg.gamma))
        check[name] = {
            "index": int(idx),
            "r_cert": float(cand.r_cert),
            "epsilon_cert": float(cand.r_cert),
            "epsilon_design": cc.epsilon_design,
            "qos_hold_rate": hold,
            "worst_sinr_p5": float(np.percentile(worst, 5)),
            "meets_epsilon_design": bool(cand.r_cert >= cc.epsilon_design),
        }

    rule_points = {r["rule"]: r for r in rule_rows}
    fig_path = plot_exp3(rows, rule_points, nu)

    save_csv("exp3_selection_rules", rule_rows)
    save_csv("exp3_rmin_sweep", rows)
    summary = {
        "rules": rule_rows,
        "r_min_sweep": rows,
        "r_max": r_max,
        "epsilon_design": cc.epsilon_design,
        "drift_rate_nu": nu,
        "independent_check": check,
        "pareto_n_nondominated": int(pareto_mask(wee, rcert).sum()),
    }
    save_json("exp3_summary", {"manifest": manifest("exp3"), **summary})

    print("=== Experiment 3: selection and sensitivity ===")
    for r in rule_rows:
        print(f"{r['rule']:<16s} idx={r['selected_index']:>3d} "
              f"WEE={r['selected_wee']:.5f} eps_cert={r['selected_rcert']:.4f} "
              f"T_cert={r['selected_t_cert_s']:.1f}s")
    print(f"R_max={r_max:.4f}, epsilon_design={cc.epsilon_design}, nu={nu}/s")
    for frac, r in zip(cc.r_min_fracs, rows):
        sel = "-" if r["selected_index"] is None else f"idx {r['selected_index']}"
        w = "n/a" if not r["feasible"] else f"{r['selected_wee']:.5f}"
        t = "n/a" if not r["feasible"] else f"{r['selected_t_cert_s']:.1f}s"
        print(f"  R_min={frac:.2f}R_max -> {sel:<8s} WEE={w:<9s} T_cert={t:<8s} "
              f"remaining={r['n_remaining']}")
    for name, c in check.items():
        print(f"  check[{name}]: hold@eps={c['qos_hold_rate']:.3f} "
              f"(eps_cert {'>=' if c['meets_epsilon_design'] else '<'} eps)")
    print(f"figure -> {fig_path}")
    return summary


if __name__ == "__main__":
    main()
