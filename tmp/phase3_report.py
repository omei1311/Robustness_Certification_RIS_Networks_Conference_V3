"""Phase 3 checks: paired trajectory identity across policies + per-regime trends."""
import csv
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
cfg = m["config"]
print("run:", run)
print("run_complete:", m.get("run_complete"), "| success:", m.get("success"))
print("cfg: solver=%s fallback=%s steps=%d pool=%d policies=%s" % (
    cfg["solver"], cfg["fallback_solver"], cfg["time_steps"], cfg["pool_size"], cfg["policies"]))

rows = [r for r in csv.DictReader(open(run / "records.csv", encoding="utf-8")) if "policy" in r]

print("\n--- paired identity: trajectory/csi ids must match across policies ---")
ok = True
for mobility in cfg["mobility_regimes"]:
    for field in ("trajectory_id", "csi_trajectory_id"):
        values = {r[field] for r in rows if r["mobility"] == mobility}
        same = len(values) == 1
        ok = ok and same
        print(f"{mobility:6s} {field}: unique={len(values)} {'OK' if same else 'MISMATCH'}")
print("paired:", "OK" if ok else "FAILED")

print("\n--- per-regime summary (reconfig / outage / EE / failed designs / reuse) ---")
for sd in m.get("seed_details", []):
    for s in sd.get("summary") or []:
        print(f"{s.get('mobility'):6s} {s.get('policy'):22s} reconf={s.get('number_of_reconfigurations'):2d} "
              f"outage={100 * (s.get('qos_outage_rate') or 0):5.1f}% EE_LT={s.get('long_term_ee'):.4g} "
              f"failed={s.get('failed_design_steps')} reuse={s.get('mean_reuse_interval')} "
              f"swE={s.get('ris_switching_energy'):.3g} strict={s.get('strict_oracle_calls')} "
              f"primary={s.get('primary_solver_calls')} fallback={s.get('fallback_solver_calls')} "
              f"errs={s.get('solver_error_count')} inacc={s.get('solver_inaccurate_count')} rt={s.get('runtime'):.0f}s")

print("\n--- certificate_triggered per-slot (config changes & holds) ---")
cert = [r for r in rows if r["policy"] == "certificate_triggered"]
for r in cert:
    print(f"{r['mobility']:6s} t={r['time_index']} trig={r['triggered']:5s} {r['reason'][:30]:30s} "
          f"rho={float(r['rho_total_after'] or 0):.4f} thr={float(r['trigger_threshold'] or 0):.4f} "
          f"cfg={str(r.get('configuration_id'))[:8]} hold={r['trigger_budget_hold']}")

designs = m.get("details", [])
calls = [c.get("strict_oracle_calls", 0) for d in designs for c in d.get("candidates", [])]
runtimes = [c.get("certificate_runtime", 0.) for d in designs for c in d.get("candidates", [])]
fb = sum(c.get("fallback_solver_calls", 0) for d in designs for c in d.get("candidates", []))
errs = sum(c.get("solver_error_count", 0) for d in designs for c in d.get("candidates", []))
inacc = sum(c.get("solver_inaccurate_count", 0) for d in designs for c in d.get("candidates", []))
if calls:
    print(f"\n--- certificate economics: n={len(calls)} avg strict/cand={sum(calls)/len(calls):.1f} "
          f"max={max(calls)} avg rt={sum(runtimes)/len(runtimes):.2f}s max rt={max(runtimes):.2f}s")
    print(f"fallback calls={fb} solver errors={errs} optimal_inaccurate={inacc}")
    over = [c for c in calls if c > 2 + len(cfg["strict_recovery_factors"]) + cfg["strict_refinement_steps"] + 1]
    print(f"candidates over hard bound (incl nominal/upper rounds): {len(over)}")
