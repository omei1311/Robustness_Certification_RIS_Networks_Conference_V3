"""Exp3: paired multi-channel policy comparison (standalone, not run_all).

Seeds and thresholds are fixed in CertConfig. Every seed uses one certified
pool shared by all policies. T_cert is conditional on rho(t)=nu*t, never an
empirically measured QoS failure time. See --help for smoke/MC options.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import warnings

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.shared import build_and_certify, make_system
from src.config_v3 import config_fingerprint
from src.generalization import aggregate, compare_pool, POLICIES
from src.io_utils import RESULTS_DIR, manifest, save_csv, save_json
from src.plotting import plot_generalization
from src.ris_base import sample_complex_unit_ball, sinr_under_error_samples
from src.uncertainty_helpers import uncertainty_shape


def add_mc(rows, pool, cfg, cc):
    """Independent RNG from pool generation; common errors pair the policies."""
    seed = cc.generalization_mc_seed_base + cc.channel_seed
    dirs = sample_complex_unit_ball(cc.generalization_mc_samples, uncertainty_shape(cfg),
                                    np.random.default_rng(seed))
    by_index = {c.index: c for c in pool}
    checked = {}
    issues = []
    for row in rows:
        idx = row["selected_candidate_index"]
        if row["threshold_type"] == "normalized" or idx is None:
            continue
        if idx not in checked:
            c = by_index[idx]
            sinr = sinr_under_error_samples(c.w, c.H, cc.epsilon_design, dirs, cfg)
            margins = sinr.min(axis=(1, 2)) - cfg.gamma
            count = int(np.sum(margins >= 0))
            checked[idx] = dict(mc_seed=seed, mc_samples=len(margins), qos_hold_count=count,
                                qos_violation_count=len(margins)-count, qos_hold_rate=count/len(margins),
                                worst_margin=float(margins.min()))
            if c.epsilon_cert >= cc.epsilon_design and count != len(margins):
                issues.append(dict(check="MC_violation_inside_reported_certificate", candidate_index=idx,
                                   note="Empirical discrepancy; inspect numerical oracle tolerance. Certificate unchanged."))
        row.update(checked[idx])
    return issues


def save_tables(summary):
    table = []
    labels = {"wee_only": "WEE-only", "proposed": f"Proposed @ {summary['epsilon_design']}",
              "robustness_only": "Robustness-only"}
    for policy in POLICIES:
        b = summary["fixed_threshold"][policy]
        row = {"Policy": labels[policy], "Valid seeds": b["valid_seed_count"]}
        for metric, name in (("wee", "WEE"), ("epsilon_cert", "epsilon_cert"), ("t_cert", "T_cert")):
            for stat in ("mean", "median", "std"):
                row[f"{stat.title()} {name}"] = b[metric][stat]
        row["Feasibility rate"] = b.get("feasibility_rate")
        row["Selection change rate"] = b.get("selection_changed_rate")
        table.append(row)
    save_csv("exp3_generalization_table", table)
    def fmt(x):
        return "N/A" if x is None else f"{x:.6g}" if isinstance(x, float) else str(x)
    headers = list(table[0])
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"]
    lines += ["| " + " | ".join(fmt(row[h]) for h in headers) + " |" for row in table]
    lines += ["", f"Requested seeds: {summary['n_requested_seeds']}; processed: {summary['n_processed_seeds']}; "
              f"failed: {summary['pool_statistics']['failed_seed_count']}.",
              "WEE: bit/s/Hz/W. epsilon_cert: dimensionless. Std: population (ddof=0).",
              summary["interpretation"]["rate_denominators"], summary["interpretation"]["t_cert"]]
    (RESULTS_DIR / "exp3_generalization_table.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def checkpoint(rows, details, cc, seeds, cfg, with_mc):
    summary = aggregate(rows, details, cc, seeds)
    summary.update(manifest=manifest("exp3_generalization_check"),
                   config=cc.to_dict(), system_config=cfg.to_dict(), with_mc=with_mc,
                   run_complete=len(details) == len(seeds))
    save_csv("exp3_generalization_records", rows)
    save_json("exp3_generalization_summary", summary)
    save_tables(summary)
    return summary


def main(n_seeds=None, with_mc=False, rebuild=False):
    cfg, cc = make_system()
    if n_seeds is not None:
        cc = replace(cc, gen_check_n_seeds=n_seeds).validate()
    if cfg.bits != 2 or cc.bits_grid != (2,) or not cfg.relative_radius:
        raise ValueError("This experiment requires B=2 and relative uncertainty")
    seeds = list(range(cc.gen_check_seed_base, cc.gen_check_seed_base + cc.gen_check_n_seeds))
    rows, details = [], []
    for number, seed in enumerate(seeds, 1):
        seed_cc = replace(cc, channel_seed=seed).validate()
        fingerprint = config_fingerprint(cfg, seed_cc)
        prefix = RESULTS_DIR / "generalization_cache" / f"seed_{seed}_{fingerprint}"
        pool, meta = [], {}
        detail = dict(channel_seed=seed, config_fingerprint=fingerprint, status="valid", pool_size=None,
                      checks_completed=False, sanity_failures=[])
        seed_rows = []
        try:
            pool, meta = build_and_certify(cfg, seed_cc, rebuild=rebuild,
                                          log=lambda _: None, cache_prefix=prefix, allow_empty=True)
            gen = meta["generation_stats"]
            detail.update(pool_size=len(pool), generation_stats=gen,
                          configuration_signatures=[c.configuration_signature for c in pool],
                          cache_prefix=str(prefix))
            if len(pool) < cc.pool_target_size:
                warnings.warn(f"seed={seed}: actual pool={len(pool)}, target={cc.pool_target_size}")
            seed_rows, failures = compare_pool(pool, seed_cc, gen)
            detail["sanity_failures"] = failures
            detail["checks_completed"] = bool(pool)
            if not pool:
                raise RuntimeError("empty candidate pool")
            if failures:
                raise AssertionError(f"selection sanity failures: {failures}")
            if with_mc:
                mc_issues = add_mc(seed_rows, pool, cfg, seed_cc)
                detail["mc_issues"] = mc_issues
                if mc_issues:
                    warnings.warn(f"seed={seed}: {mc_issues}")
        except Exception as exc:
            detail.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            warnings.warn(f"seed={seed}: {detail['error']}; retaining seed and continuing")
            if not seed_rows:
                seed_rows, _ = compare_pool(pool, seed_cc, meta.get("generation_stats", {}), error=detail["error"])
            for row in seed_rows:
                row.update(seed_status="failed", error=detail["error"], pool_size=detail["pool_size"])
        for row in seed_rows:
            row["config_fingerprint"] = fingerprint
        rows.extend(seed_rows)
        details.append(detail)
        summary = checkpoint(rows, details, cc, seeds, cfg, with_mc)
        fixed = [r for r in seed_rows if r["threshold_type"] != "normalized"]
        parts = []
        for r in fixed:
            val = (f"WEE={r['selected_wee']:.6f}, eps={r['selected_epsilon_cert']:.6f}"
                   if r["selected_wee"] is not None else "no selection")
            parts.append(f"{r['policy']}: {val}, changed={r['selection_changed_vs_wee_only']}")
        print(f"[{number:02d}/{len(seeds)}] seed={seed} pool={detail['pool_size']} "
              f"status={detail['status']} | " + " | ".join(parts), flush=True)
    paths = plot_generalization(rows, cc.epsilon_design, cc.eps_min_fracs, len(seeds))
    print("\n=== Generalization Summary ===")
    print(f"Valid seeds: {summary['valid_seed_count']}; failed seeds: {summary['pool_statistics']['failed_seed_count']}")
    print(f"Fixed epsilon_min = {cc.epsilon_design}")
    for policy in POLICIES:
        b = summary["fixed_threshold"][policy]
        print(f"{policy}: n={b['valid_seed_count']}; WEE {b['wee']}; epsilon_cert {b['epsilon_cert']}")
        if policy == "proposed":
            print(f"  feasible rate={b['feasibility_rate']}; selection changed rate={b['selection_changed_rate']}")
    print("Paired proposed vs WEE-only:", summary["fixed_threshold"]["paired_vs_wee_only"])
    print(summary["interpretation"]["t_cert"])
    print("Results saved to:")
    for name in ("records.csv", "summary.json", "table.csv", "table.md"):
        print(RESULTS_DIR / ("exp3_generalization_" + name))
    for path in paths:
        print(path)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=None, help="Consecutive seeds from config seed base (default 20)")
    parser.add_argument("--with-mc", action="store_true", help="Independent 100-sample sanity check at epsilon_design")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the same predetermined pools")
    args = parser.parse_args()
    result = main(args.n_seeds, args.with_mc, args.rebuild)
    if result["failed_seeds"]:
        sys.exit(1)  # All seeds were processed and results saved before signaling failure.
