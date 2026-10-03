"""Phase smoke report: trigger/lifetime activation and solver-call economics."""
import csv
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
cfg = m["config"]
print("run:", run)
print("run_complete:", m.get("run_complete"), "| success:", m.get("success"))
print("execution:", json.dumps(m.get("execution")))
print("cfg: solver=%s@%g fallback=%s@%g steps=%d pool=%d policies=%s regimes=%s seeds=%s eta=%g nu=%g eps_est=%s" % (
    cfg["solver"], cfg["solver_tol"], cfg["fallback_solver"], cfg["fallback_solver_tol"],
    cfg["time_steps"], cfg["pool_size"], cfg["policies"], cfg["mobility_regimes"],
    list(cfg["seeds"]), cfg["eta_trigger"], cfg["drift_rate_nu"],
    cfg["epsilon_est"] if cfg["epsilon_est"] is not None else "(artifact)"))

designs = m.get("details", [])
print("\n--- designs ---")
for d in designs:
    stats = d.get("stats", {})
    print(f"t={d.get('time_index')} policy={d.get('policy')} selection={d.get('selection_status')} "
          f"pool={stats.get('actual_pool_size')} fast={stats.get('fast_oracle_calls')} "
          f"strict={stats.get('strict_oracle_calls')} primary={stats.get('primary_solver_calls')} "
          f"fallback={stats.get('fallback_solver_calls')} errors={stats.get('solver_error_count')} "
          f"inaccurate={stats.get('solver_inaccurate_count')} cert_rt={stats.get('certificate_runtime', 0):.2f}s")
    for s in d.get("scores", []):
        print(f"   cand {s.get('candidate_index')}: eligible={s.get('eligible')} "
              f"T_pred={s.get('predicted_lifetime')} EE_life={s.get('EE_life')}")
    for c in d.get("candidates", []):
        print(f"   cand {c.get('candidate_index')}: eps_cert={c.get('epsilon_cert'):.5g} "
              f"status={c.get('status')} strict_calls={c.get('strict_oracle_calls')} "
              f"trials={c.get('strict_trial_count')} primary={c.get('primary_solver_calls')} "
              f"fallback={c.get('fallback_solver_calls')} sinr_min={c.get('sinr_min'):.3g}")

with open(run / "records.csv", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
print("\n--- per-slot events ---")
for r in rows:
    if "policy" not in r:
        continue
    print(f"m={r.get('mobility','?'):6s} t={r.get('time_index')} {r.get('policy','?'):22s} "
          f"trig={r.get('triggered')} {r.get('reason','')[:30]:30s} "
          f"rho_total={r.get('rho_total_after')} thr={r.get('trigger_threshold')} "
          f"inst={r.get('installed')} cfg={str(r.get('configuration_id'))[:8]} "
          f"trigHold={r.get('trigger_budget_hold')} certHold={r.get('certified_budget_hold')} "
          f"sinr={r.get('sinr_min')} qos={r.get('qos_hold')} "
          f"nu={r.get('drift_rate_nu')} "
          f"Tpred={r.get('predicted_reuse_slots')}slots Tactual={r.get('actual_reuse_slots')}slots")

print("\n--- summaries ---")
for sd in m.get("seed_details", []):
    for s in sd.get("summary") or []:
        print(f"m={s.get('mobility','?'):6s} {s.get('policy','?'):22s} outage={s.get('qos_outage_rate')} "
              f"EE_LT={s.get('long_term_ee'):.6g} reconf={s.get('number_of_reconfigurations')} "
              f"failedDesign={s.get('failed_design_steps')} "
              f"strict={s.get('strict_oracle_calls')} primary={s.get('primary_solver_calls')} "
              f"fallback={s.get('fallback_solver_calls')} rt={s.get('runtime'):.1f}s")

# 39: per-candidate solver-call economics
calls, runtimes, fallbacks, errors, inaccurates = [], [], 0, 0, 0
for d in designs:
    for c in d.get("candidates", []):
        calls.append(c.get("strict_oracle_calls", 0))
        runtimes.append(c.get("certificate_runtime", 0.))
        fallbacks += c.get("fallback_solver_calls", 0)
        errors += c.get("solver_error_count", 0)
        inaccurates += c.get("solver_inaccurate_count", 0)
if calls:
    print("\n--- per-candidate certificate economics (all designs) ---")
    print(f"candidates={len(calls)} avg strict/cand={sum(calls)/len(calls):.1f} max={max(calls)}")
    print(f"avg cert runtime={sum(runtimes)/len(runtimes):.2f}s max={max(runtimes):.2f}s")
    print(f"fallback calls={fallbacks} solver errors={errors} optimal_inaccurate={inaccurates}")
