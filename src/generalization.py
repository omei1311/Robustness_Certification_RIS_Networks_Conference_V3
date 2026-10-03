"""Paired policy records and aggregation; no generation or certification here."""
from __future__ import annotations

import hashlib
import numpy as np

from .pareto import pareto_mask
from .selection import select_wee_only, select_robustness_only, select_stability_aware

EPS_DENOM = 1e-8
SELECTION_TOL = 1e-9
POLICIES = ("wee_only", "proposed", "robustness_only")


def ratio(numerator, denominator):
    if numerator is None or denominator is None:
        return None
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= EPS_DENOM:
        return None
    return float(numerator / denominator)


def stats(values):
    a = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    return dict(n=int(a.size), mean=float(a.mean()) if a.size else None,
                median=float(np.median(a)) if a.size else None,
                std=float(a.std()) if a.size else None)


def threshold_specs(cc):
    return [("wee_only", "none", None), ("robustness_only", "none", None),
            ("proposed", "fixed", None)] + [
                ("proposed", "normalized", float(f)) for f in cc.eps_min_fracs]


def compare_pool(pool, cc, generation_stats, error=None):
    """Apply every policy to the same immutable metric arrays, once per seed.

    Returns long-format records and diagnostic failures. Infeasible thresholds
    keep a None selection; no fallback. Full-pool eligibility is reported
    separately from Pareto eligibility. Empty/failed seeds retain every row.
    """
    wee = np.asarray([c.wee for c in pool], dtype=float)
    eps = np.asarray([c.epsilon_cert for c in pool], dtype=float)
    if pool and error is None and (not np.all(np.isfinite(wee)) or not np.all(np.isfinite(eps))):
        raise ValueError("nonfinite pool metrics")
    wee.setflags(write=False)
    eps.setflags(write=False)
    signatures = [c.configuration_signature for c in pool]
    pool_id = hashlib.sha256("\n".join(sorted(signatures)).encode()).hexdigest()
    mask = pareto_mask(wee, eps)
    usable = bool(pool) and error is None
    base = select_wee_only(wee, eps, cc.drift_rate_nu, mask) if usable else None
    rob = select_robustness_only(wee, eps, cc.drift_rate_nu, mask) if usable else None
    eps_max = float(eps.max()) if usable else None
    counts = {status: sum(c.cert_info.get("status") == status for c in pool) for status in
              ("EXACT_BRACKET", "LOWER_BOUND_CENSORED", "NOMINAL_INFEASIBLE")}
    rows, failures = [], []
    previous = None
    for policy, threshold_type, frac in threshold_specs(cc):
        threshold = (cc.epsilon_design if threshold_type == "fixed" else
                     frac * eps_max if threshold_type == "normalized" and eps_max is not None else None)
        eligible = int(np.sum(eps >= threshold)) if usable and threshold is not None else None
        out = None
        if usable:
            if policy == "wee_only":
                out = base
            elif policy == "robustness_only":
                out = rob
            elif eligible:
                out = select_stability_aware(wee, eps, threshold, cc.drift_rate_nu, mask)
        cand = pool[out.index] if out else None
        delta_w = out.wee - base.wee if out else None
        delta_e = out.epsilon_cert - base.epsilon_cert if out else None
        pct_w = ratio(delta_w, base.wee) if base else None
        pct_e = ratio(delta_e, base.epsilon_cert) if base else None
        row = dict(
            channel_seed=cc.channel_seed, pool_size=len(pool), pool_signature=pool_id,
            config_fingerprint=pool[0].config_fingerprint if pool else None,
            unique_theta_count=generation_stats.get("unique_theta_count"),
            unique_configuration_count=generation_stats.get("unique_configuration_count"),
            policy=policy, threshold_type=threshold_type, epsilon_min=threshold, eps_min_frac=frac,
            epsilon_max=eps_max, selected_candidate_index=cand.index if cand else None,
            configuration_signature=cand.configuration_signature if cand else None,
            selected_wee=out.wee if out else None,
            selected_epsilon_cert=out.epsilon_cert if out else None,
            t_cert=out.t_cert if out else None, eligible_count=eligible,
            n_pareto=int(mask.sum()), n_after_threshold_pareto=out.n_after_rmin if out and policy == "proposed" else None,
            proposed_feasible=bool(out) if policy == "proposed" and usable else None,
            wee_only_index=pool[base.index].index if base else None,
            wee_only_wee=base.wee if base else None,
            wee_only_epsilon_cert=base.epsilon_cert if base else None,
            delta_wee_vs_wee_only=delta_w, delta_wee_percent_vs_wee_only=100*pct_w if pct_w is not None else None,
            delta_epsilon_vs_wee_only=delta_e,
            delta_epsilon_percent_vs_wee_only=100*pct_e if pct_e is not None else None,
            wee_retention_ratio=ratio(out.wee, base.wee) if out else None,
            selected_epsilon_ratio=ratio(out.epsilon_cert, eps_max) if out else None,
            selection_changed_vs_wee_only=(cand.configuration_signature != pool[base.index].configuration_signature) if out else None,
            certificate_status=cand.cert_info.get("status") if cand else None,
            candidate_generation_attempts=generation_stats.get("attempts"),
            n_exact_bracket=counts["EXACT_BRACKET"], n_lower_bound_censored=counts["LOWER_BOUND_CENSORED"],
            n_nominal_infeasible=counts["NOMINAL_INFEASIBLE"],
            seed_status="valid" if usable else "failed", error=error or ("empty pool" if not pool else None),
            mc_seed=None, mc_samples=None, qos_hold_count=None, qos_violation_count=None,
            qos_hold_rate=None, worst_margin=None,
        )
        if out and policy == "proposed":
            checks = {
                "A_unconstrained_wee_bound": base.wee >= out.wee - SELECTION_TOL,
                "B_threshold_satisfied": out.epsilon_cert >= threshold - SELECTION_TOL,
            }
            if frac == 0:
                checks["D_zero_threshold_matches_wee"] = abs(out.wee-base.wee) <= SELECTION_TOL
            if threshold_type == "normalized":
                if previous is not None:
                    checks["E_wee_nonincreasing"] = out.wee <= previous + SELECTION_TOL
                previous = out.wee
            failures.extend(dict(check=k, eps_min_frac=frac) for k, ok in checks.items() if not ok)
        rows.append(row)
    if rob and abs(rob.epsilon_cert - eps_max) > SELECTION_TOL:
        failures.append(dict(check="C_robustness_maximum"))
    return rows, failures


def aggregate(rows, seed_details, cc, seeds):
    """Finite-value statistics (population std); paired deltas only on valid pairs."""
    valid = [r for r in rows if r.get("seed_status") == "valid"]
    valid_seeds = {r["channel_seed"] for r in valid}
    failed = [d for d in seed_details if d["status"] == "failed"]

    def block(group):
        good = [r for r in group if r.get("selected_wee") is not None and
                np.isfinite(r["selected_wee"]) and r.get("selected_epsilon_cert") is not None and
                np.isfinite(r["selected_epsilon_cert"])]
        out = {"valid_seed_count": len(good)}
        for label, field in (("wee", "selected_wee"), ("epsilon_cert", "selected_epsilon_cert"), ("t_cert", "t_cert")):
            out[label] = stats(r.get(field) for r in good)
        return out, good

    fixed = {}
    for policy in POLICIES:
        group = [r for r in valid if r["policy"] == policy and r["threshold_type"] != "normalized"]
        out, good = block(group)
        if policy == "proposed":
            out.update(feasible_seed_count=len(good),
                       infeasible_seed_count=sum(r.get("proposed_feasible") is False for r in group),
                       feasibility_rate=len(good)/len(valid_seeds) if valid_seeds else None,
                       feasible_fraction_requested=len(good)/len(seeds) if seeds else None,
                       selection_changed_count=sum(r.get("selection_changed_vs_wee_only") is True for r in good),
                       selection_changed_rate=stats(r.get("selection_changed_vs_wee_only") for r in good)["mean"])
        fixed[policy] = out
    paired = [r for r in valid if r["policy"] == "proposed" and r["threshold_type"] == "fixed"
              and r.get("proposed_feasible") is True]
    fixed["paired_vs_wee_only"] = {field: stats(r.get(field) for r in paired) for field in
        ("delta_wee_vs_wee_only", "delta_wee_percent_vs_wee_only", "delta_epsilon_vs_wee_only",
         "delta_epsilon_percent_vs_wee_only")}
    fixed["paired_vs_wee_only"]["n_pairs"] = len(paired)
    sensitivity = {}
    for frac in cc.eps_min_fracs:
        group = [r for r in valid if r["threshold_type"] == "normalized" and r["eps_min_frac"] == frac]
        out, good = block(group)
        out.update(n_valid_seeds=len(group), n_feasible_seeds=len(good),
                   wee_retention_ratio=stats(r.get("wee_retention_ratio") for r in good),
                   absolute_epsilon_gain=stats(r.get("delta_epsilon_vs_wee_only") for r in good),
                   selected_epsilon_ratio=stats(r.get("selected_epsilon_ratio") for r in good),
                   selection_changed_rate=stats(r.get("selection_changed_vs_wee_only") for r in good)["mean"])
        sensitivity[f"{frac:.2f}"] = out
    sizes = [d.get("pool_size") for d in seed_details if d.get("pool_size") is not None]
    issues = [dict(channel_seed=d["channel_seed"], **f) for d in seed_details for f in d.get("sanity_failures", [])]
    return dict(
        experiment="exp3_generalization_check", n_requested_seeds=len(seeds), seed_base=cc.gen_check_seed_base,
        channel_seeds=seeds, n_processed_seeds=len(seed_details), valid_seed_count=len(valid_seeds),
        epsilon_design=cc.epsilon_design, eps_min_fracs=list(cc.eps_min_fracs), drift_rate_nu=cc.drift_rate_nu,
        pool_statistics=dict(mean_pool_size=stats(sizes)["mean"], min_pool_size=min(sizes) if sizes else None,
                             max_pool_size=max(sizes) if sizes else None, failed_seed_count=len(failed),
                             short_pool_seeds=[d["channel_seed"] for d in seed_details
                                               if d.get("pool_size") is not None and d["pool_size"] < cc.pool_target_size]),
        fixed_threshold=fixed, threshold_sensitivity=sensitivity,
        sanity_checks=dict(tolerance=SELECTION_TOL, failed_check_count=len(issues), failures=issues,
                           checked_seed_count=sum(d.get("checks_completed", False) for d in seed_details)),
        failed_seeds=failed, seed_details=seed_details,
        interpretation=dict(
            t_cert="Derived conditional interpretation under relative drift rho(t)=nu*t; not actual QoS failure time.",
            statistics="Population std (ddof=0); finite observations only; each block reports n.",
            rate_denominators="Feasibility rate: valid seeds. Selection change rate: feasible valid pairs. Failed seeds separate; feasible_fraction_requested includes all requested seeds.",
            certificate="LOWER_BOUND_CENSORED values remain lower bounds; normalized sweep uses maximum reported certificate, not an uncensored boundary.",
            percentage_denominator_min=EPS_DENOM,
            selection="Certificate-aware constraint controls selection; Pareto preprocessing does not improve the constrained max-WEE objective."))
