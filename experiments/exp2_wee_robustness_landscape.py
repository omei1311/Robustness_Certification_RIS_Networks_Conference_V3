"""Experiment 2 -- WEE-robustness landscape and Pareto structure (Section 9).

Builds the candidate pool (~20-50 unique feasible configurations) on the
shared nominal drop, certifies every member (R_cert by bisection with the
statewise LMI oracle), visualizes the joint (WEE, R_cert) distribution,
marks the WEE-only / robustness-only choices and the nondominated Pareto
set, and reports representative dominated points.  The prescribed design
radius epsilon is drawn separately from the post-optimization certificates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import spearmanr  # noqa: E402

from src.io_utils import manifest, save_csv, save_json  # noqa: E402
from src.pareto import pareto_mask  # noqa: E402
from src.selection import (  # noqa: E402
    select_robustness_only,
    select_stability_aware,
    select_wee_only,
)
from src.plotting import plot_exp2  # noqa: E402
from shared import build_and_certify, make_system  # noqa: E402


def main(rebuild: bool = False) -> dict:
    cfg, cc = make_system()
    pool, meta = build_and_certify(cfg, cc, rebuild=rebuild)

    wee = np.array([c.wee for c in pool])
    rcert = np.array([c.r_cert for c in pool])
    mask = pareto_mask(wee, rcert)

    out_wee = select_wee_only(wee, rcert, cc.drift_rate_nu)
    out_rob = select_robustness_only(wee, rcert, cc.drift_rate_nu)
    r_max = float(rcert.max())
    out_sel = select_stability_aware(wee, rcert, 0.90 * r_max, cc.drift_rate_nu)

    rho, pval = spearmanr(wee, rcert)

    rows = []
    for c in pool:
        rows.append(
            {
                "index": c.index,
                "bits": c.bits,
                "direction": c.direction_kind,
                "align_frac": c.align_frac,
                "design_gamma": c.design_gamma,
                "design_eps": c.design_eps,
                "wee": c.wee,
                "r_cert": c.r_cert,
                "nondominated": bool(mask[c.index]),
                "min_sinr_nominal": float(np.min(c.sinr_nominal)),
                "power_per_bs_w": float(np.sum(c.power_per_bs)),
            }
        )

    summary = {
        "pool_size": len(pool),
        "n_nondominated": int(mask.sum()),
        "n_dominated": int((~mask).sum()),
        "wee_min": float(wee.min()),
        "wee_max": float(wee.max()),
        "rcert_min": float(rcert.min()),
        "rcert_max": r_max,
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "epsilon_design": cc.epsilon_design,
        "n_below_epsilon_design": int(np.sum(rcert < cc.epsilon_design)),
        "wee_only": out_wee.__dict__,
        "robustness_only": out_rob.__dict__,
        "stability_aware_0p9rmax": out_sel.__dict__,
    }

    save_csv("exp2_candidate_map", rows)
    save_json("exp2_summary", {"manifest": manifest("exp2"), **summary, "meta": meta})
    fig_path = plot_exp2(
        wee, rcert, mask,
        i_wee=out_wee.index,
        i_rob=out_rob.index,
        i_sel=out_sel.index if out_sel.index != out_wee.index else None,
        epsilon_design=cc.epsilon_design,
    )

    print("=== Experiment 2: WEE-robustness landscape ===")
    print(f"pool size            : {summary['pool_size']}")
    print(f"nondominated / dominated : {summary['n_nondominated']} / {summary['n_dominated']}")
    print(f"WEE range            : [{summary['wee_min']:.5f}, {summary['wee_max']:.5f}]")
    print(f"R_cert range         : [{summary['rcert_min']:.5f}, {summary['rcert_max']:.5f}]")
    print(f"Spearman(WEE, R_cert): rho={rho:.3f} (p={pval:.2g})")
    print(f"candidates below prescribed epsilon={cc.epsilon_design}: "
          f"{summary['n_below_epsilon_design']}")
    print(f"WEE-only choice        : idx={out_wee.index} WEE={out_wee.wee:.5f} R={out_wee.rcert:.4f}")
    print(f"robustness-only choice : idx={out_rob.index} WEE={out_rob.wee:.5f} R={out_rob.rcert:.4f}")
    print(f"stability-aware (0.9R_max): idx={out_sel.index} WEE={out_sel.wee:.5f} "
          f"R={out_sel.rcert:.4f} T_cert={out_sel.t_cert:.1f}s")
    print(f"figure -> {fig_path}")
    return summary


if __name__ == "__main__":
    main(rebuild="--rebuild" in sys.argv)
