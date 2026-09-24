"""Experiment 3 -- generalization (holdout) check across channel seeds.

Compact holdout validation, NOT a new main experiment and NOT a redefinition
of the paper contribution: it verifies that the WEE / epsilon_cert / Pareto /
stability-aware-selection behaviour observed on the single nominal channel of
the main experiments is not an artifact of that one realization.

Per independent channel seed (seed = gen_check_seed_base + i, fixed system
parameters, fixed candidate-generation protocol, B = 2):
  1. generate_channel_drop(...)
  2. build candidate pool (record the ACTUAL pool size; never fabricate 40)
  3. certify every candidate (epsilon_cert by deterministic LMI bisection)
  4. compute pareto_mask
  5. WEE-only selection (full pool)
  6. proposed stability-aware selection:
     Pareto -> epsilon_cert >= epsilon_design (= 0.05) -> max WEE

No seed tuning, no candidate removal, no selective reporting.  Per-seed
Pareto frontiers are never merged into one WEE-epsilon_cert plot (each seed
is a different channel realization), so the outputs are machine-readable
tables only:
    results/exp3_generalization_records.csv
    results/exp3_generalization_summary.json

Run standalone (deliberately NOT part of run_all.py):
    .venv\\Scripts\\python.exe experiments\\exp3_generalization_check.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_utils import manifest, save_csv, save_json  # noqa: E402
from src.pareto import pareto_mask  # noqa: E402
from src.certificate import epsilon_cert_bisection  # noqa: E402
from src.selection import (  # noqa: E402
    select_stability_aware,
    select_wee_only,
)
from src.candidate_pool import build_pool  # noqa: E402
from shared import make_system, nominal_drop  # noqa: E402


def certify_pool(pool, cfg, cc):
    for c in pool:
        res = epsilon_cert_bisection(
            c.w, c.H, cfg,
            eps_hi=cc.bisection_eps_hi,
            eps_hi_max=cc.bisection_eps_hi_max,
            tol=cc.bisection_tol,
            max_iter=cc.bisection_max_iter,
        )
        c.r_cert = res.r_cert
        c.cert_info = {
            "status": res.status,
            "bracket": list(res.bracket),
            "n_bisection_iter": res.n_bisection_iter,
            "n_feasibility_checks": res.n_feasibility_checks,
            "binding_user": list(res.binding_user) if res.binding_user else None,
            "binding_margin": res.binding_margin,
        }
    return pool


def _stats_block(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
    }


def main() -> dict:
    cfg, cc = make_system()
    assert cc.bits_grid == (2,), "generalization check requires fixed B=2"
    nu = cc.drift_rate_nu
    eps_min = cc.epsilon_design
    seeds = list(
        range(cc.gen_check_seed_base, cc.gen_check_seed_base + cc.gen_check_n_seeds)
    )

    rows = []
    t0 = time.time()
    for seed in seeds:
        drop = nominal_drop(cfg, seed)
        pool, gen_stats = build_pool(drop, cfg, cc)
        if not pool:
            rows.append(
                {
                    "channel_seed": seed, "pool_size": 0, "n_pareto": 0,
                    "wee_only_index": None, "wee_only_wee": float("nan"),
                    "wee_only_epsilon_cert": float("nan"),
                    "proposed_index": None, "proposed_wee": float("nan"),
                    "proposed_epsilon_cert": float("nan"),
                    "selection_same": None,
                    "epsilon_design": eps_min,
                    "n_after_rmin": 0, "r_max": float("nan"),
                    "unique_theta": 0, "unique_X": 0,
                }
            )
            print(f"[gen seed {seed}] empty pool, recorded as invalid")
            continue
        certify_pool(pool, cfg, cc)
        wee = np.array([c.wee for c in pool])
        rcert = np.array([c.r_cert for c in pool])
        mask = pareto_mask(wee, rcert)
        r_max = float(rcert.max())

        out_wee = select_wee_only(wee, rcert, nu, nondominated=mask)
        try:
            out_prop = select_stability_aware(wee, rcert, eps_min, nu,
                                              nondominated=mask)
            prop = {
                "proposed_index": out_prop.index,
                "proposed_wee": out_prop.wee,
                "proposed_epsilon_cert": out_prop.rcert,
                "n_after_rmin": out_prop.n_after_rmin,
            }
            same = bool(out_prop.index == out_wee.index)
        except ValueError:
            # No nondominated candidate meets eps_min on this realization.
            prop = {
                "proposed_index": None,
                "proposed_wee": float("nan"),
                "proposed_epsilon_cert": float("nan"),
                "n_after_rmin": 0,
            }
            same = None
        rows.append(
            {
                "channel_seed": seed,
                "pool_size": len(pool),
                "n_pareto": int(mask.sum()),
                "wee_only_index": out_wee.index,
                "wee_only_wee": out_wee.wee,
                "wee_only_epsilon_cert": out_wee.rcert,
                **prop,
                "selection_same": same,
                "epsilon_design": eps_min,
                "r_max": r_max,
                "unique_theta": gen_stats.get("unique_theta_count", len(pool)),
                "unique_X": gen_stats.get("unique_configuration_count", len(pool)),
            }
        )
        print(f"[gen seed {seed}] pool={len(pool)} pareto={int(mask.sum())} "
              f"eps_max={r_max:.4f} | wee_only idx={out_wee.index} "
              f"WEE={out_wee.wee:.4f} eps_cert={out_wee.rcert:.4f} | "
              f"proposed idx={prop['proposed_index']} "
              f"WEE={prop['proposed_wee']:.4f} "
              f"eps_cert={prop['proposed_epsilon_cert']:.4f} "
              f"n_after_rmin={prop['n_after_rmin']} same={same}")

    # ---------------- aggregate statistics ---------------- #
    valid = [r for r in rows if r["pool_size"] > 0]
    decided = [r for r in valid if r["selection_same"] is not None]
    sel_wee = np.array([r["proposed_wee"] for r in decided], dtype=float)
    sel_eps = np.array([r["proposed_epsilon_cert"] for r in decided], dtype=float)
    wee_eps = np.array([r["wee_only_epsilon_cert"] for r in valid], dtype=float)
    same_flags = [bool(r["selection_same"]) for r in decided]

    summary = {
        "valid_seed_count": len(valid),
        "n_seeds_requested": len(seeds),
        "epsilon_design": eps_min,
        "pool_size": _stats_block(np.array([r["pool_size"] for r in valid], dtype=float)),
        "n_pareto": _stats_block(np.array([r["n_pareto"] for r in valid], dtype=float)),
        "selected_wee_proposed": _stats_block(sel_wee),
        "selected_epsilon_cert_proposed": _stats_block(sel_eps),
        "epsilon_cert_wee_only": _stats_block(wee_eps),
        "selection_same_count": int(np.sum(same_flags)) if same_flags else 0,
        "selection_changed_count": int(len(same_flags) - np.sum(same_flags)) if same_flags else 0,
        "n_seeds_no_eligible_candidate": len(valid) - len(decided),
        "note": (
            "Compact holdout validation on independent channel seeds; "
            "per-seed Pareto frontiers are never merged into one plot. "
            "Reference results for the paper come from the single nominal "
            "channel; this check is separate."
        ),
    }
    summary["selection_changed_rate"] = (
        float(summary["selection_changed_count"]) / float(len(same_flags))
        if same_flags else None
    )

    save_csv("exp3_generalization_records", rows)
    save_json(
        "exp3_generalization_summary",
        {"manifest": manifest("exp3_generalization"), **summary},
    )

    print("\n=== Experiment 3 generalization (holdout) check ===")
    print(f"valid seeds: {summary['valid_seed_count']}/{summary['n_seeds_requested']}")
    print(f"pool_size : mean={summary['pool_size']['mean']:.2f} "
          f"median={summary['pool_size']['median']:.1f} std={summary['pool_size']['std']:.2f}")
    print(f"n_pareto  : mean={summary['n_pareto']['mean']:.2f} "
          f"median={summary['n_pareto']['median']:.1f} std={summary['n_pareto']['std']:.2f}")
    print(f"proposed selected WEE      : mean={summary['selected_wee_proposed']['mean']:.4f} "
          f"median={summary['selected_wee_proposed']['median']:.4f} "
          f"std={summary['selected_wee_proposed']['std']:.4f}")
    print(f"proposed selected eps_cert : mean={summary['selected_epsilon_cert_proposed']['mean']:.4f} "
          f"median={summary['selected_epsilon_cert_proposed']['median']:.4f} "
          f"std={summary['selected_epsilon_cert_proposed']['std']:.4f}")
    print(f"WEE-only eps_cert          : mean={summary['epsilon_cert_wee_only']['mean']:.4f} "
          f"median={summary['epsilon_cert_wee_only']['median']:.4f} "
          f"std={summary['epsilon_cert_wee_only']['std']:.4f}")
    print(f"selection same/changed     : {summary['selection_same_count']}/"
          f"{summary['selection_changed_count']} "
          f"(changed rate {summary['selection_changed_rate']})")
    print(f"seeds with no eligible candidate: {summary['n_seeds_no_eligible_candidate']}")
    print(f"total time: {time.time() - t0:.1f}s")
    return summary


if __name__ == "__main__":
    main()
