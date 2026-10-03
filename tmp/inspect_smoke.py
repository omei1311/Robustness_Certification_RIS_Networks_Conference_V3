"""Read-only inspection of journal_results smoke runs (no framework code touched)."""
import csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JR = ROOT / "journal_results"


def rows_of(csv_path):
    with open(csv_path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------- latest runs per experiment ----------
latest = {}
for exp_dir in sorted(JR.glob("exp*")):
    for mode_dir in exp_dir.glob("*"):
        runs = sorted(p for p in mode_dir.iterdir() if p.is_dir())
        if runs:
            latest[exp_dir.name + "/" + mode_dir.name] = runs[-1]

for key, run in latest.items():
    m = jload(run / "manifest.json")
    header(f"{key} -> {run.name}")
    print("run_complete:", m.get("run_complete"), "| success:", m.get("success"),
          "| plot_error:", m.get("plot_error"))
    for d in m.get("seed_details", []):
        print(f"  seed={d.get('seed')} status={d.get('status')}",
              (d.get("error") or "")[:200])
    # staleness: compare source_hashes recorded in manifest vs current files
    current = {}
    for p in (ROOT / "journal_sim").rglob("*"):
        if p.is_file() and p.suffix in (".py", ".npz", ".json") and "__pycache__" not in p.parts:
            import hashlib
            current[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    recorded = m.get("source_hashes", {})
    changed = [k for k, v in recorded.items() if current.get(k) != v]
    added = [k for k in current if k not in recorded]
    print("  source files changed since run:", changed if changed else "none")
    if added:
        print("  files added after run:", added)

# ---------- failed exp3 runs ----------
header("exp3 runs that produced manifest only (aborted/failed)")
for run in sorted((JR / "exp3_dynamic" / "smoke").iterdir()):
    if not (run / "records.csv").exists():
        if not (run / "manifest.json").exists():
            print(f"\n{run.name}: directory exists but no manifest (aborted at creation)")
            continue
        m = jload(run / "manifest.json")
        print(f"\n{run.name}: run_complete={m.get('run_complete')} success={m.get('success')}")
        for d in m.get("seed_details", []):
            print("  ", d.get("seed"), d.get("status"), (d.get("error") or "")[:300])

# ---------- EXP0: certificate validation trends ----------
run = latest.get("exp0_oracle/smoke")
if run:
    header("EXP0 certificate validation (per label x alpha, aggregated over users)")
    rows = [r for r in rows_of(run / "records.csv") if r.get("row_type") == "validation"]
    agg = {}
    for r in rows:
        key = (r["label"], float(r["alpha"]))
        agg.setdefault(key, []).append(r)
    for (label, alpha), group in sorted(agg.items()):
        n = len(group)
        fast_ok = sum(str(r["fast_feasible"]).lower() == "true" for r in group)
        strict = {}
        for r in group:
            strict[r["strict_status"]] = strict.get(r["strict_status"], 0) + 1
        vrate = [float(r["violation_rate"]) for r in group]
        sinr_min = min(float(r["sinr_min"]) for r in group)
        margins = [float(r["fast_margin"]) for r in group]
        neig = [float(r["normalized_min_eig"]) for r in group if r["normalized_min_eig"] not in ("", None)]
        print(f"{label:18s} alpha={alpha:4.2f} users={n} fast_ok={fast_ok}/{n} strict={strict} "
              f"viol%={100*sum(vrate)/n:6.2f} worstSINR={sinr_min:9.3g} "
              f"fastMargin[min]={min(margins):9.3g} strictNormEig[min]={min(neig) if neig else float('nan'):9.3g}")
    # candidate pool summary
    cands = [r for r in rows_of(run / "records.csv") if r.get("row_type") == "candidate"]
    print(f"\npool: {len(cands)} candidates, "
          f"validated={sum(str(c['strict_validation_passed']).lower()=='true' for c in cands)}, "
          f"nominal_feasible={sum(str(c['nominal_feasible']).lower()=='true' for c in cands)}")
    certs = sorted(float(c['epsilon_cert']) for c in cands)
    print("epsilon_cert values:", [f"{c:.4g}" for c in certs])

# ---------- EXP1: calibration monotonicity ----------
run = latest.get("exp1_csi/smoke")
if run:
    header("EXP1 CSI calibration: joint epsilon vs estimation SNR (per estimator, N=32 B=2)")
    rows = [r for r in rows_of(run / "records.csv")
            if r.get("N") == "32" and r.get("bits") == "2"]
    for estimator in sorted({r["estimator"] for r in rows}):
        sub = [r for r in rows if r["estimator"] == estimator]
        for snr in sorted({float(r["estimation_snr_db"]) for r in sub}):
            g = [r for r in sub if float(r["estimation_snr_db"]) == snr]
            e90 = max(float(r["epsilon_joint_90"]) for r in g)
            e95 = max(float(r["epsilon_joint_95"]) for r in g)
            e99 = max(float(r["epsilon_joint_99"]) for r in g)
            nm = max(float(r["measured_nmse"]) for r in g)
            print(f"{estimator:20s} SNR={snr:5.1f}dB  e90={e90:.4f} e95={e95:.4f} e99={e99:.4f} measuredNMSE={nm:.2e}")
    jr_file = run / "joint_radius.json"
    if jr_file.exists():
        payload = jload(jr_file)
        print(f"\njoint_radius.json: schema={payload.get('schema')} complete={payload.get('complete')} "
              f"entries={len(payload['entries'])}")
        for e in payload["entries"][:6]:
            print(f"  N={e['scope']['N']} B={e['scope']['bits']} snr={e['scope']['estimation_snr_db']}"
                  f" -> est(99 joint, max)={e.get('epsilon_joint_99'):.4f}")

# ---------- EXP2: static pool + selections ----------
run = latest.get("exp2_static/smoke")
if run:
    header("EXP2 static pool / selection rules")
    rows = rows_of(run / "records.csv")
    cands = [r for r in rows if r.get("row_type") == "candidate"]
    sels = [r for r in rows if r.get("row_type") == "selection"]
    print(f"pool={len(cands)} validated={sum(str(c['strict_validation_passed']).lower()=='true' for c in cands)}")
    for c in sorted(cands, key=lambda r: float(r["epsilon_cert"])):
        print(f"  c{c['candidate_index']}: eps_cert={float(c['epsilon_cert']):.4f} "
              f"sinr_min={float(c['sinr_min']):.3g} wee={float(c['spectral_wee']):.4g} "
              f"status={c['status']} selected_by={c.get('selected_by')}")
    for s in sels:
        print(f"  rule={s['rule']:16s} -> idx={s['selected_index']} status={s['selection_status']}")

# ---------- EXP3: dynamic policy comparison ----------
run = latest.get("exp3_dynamic/smoke")
if run:
    header("EXP3 dynamic policies (latest complete smoke run)")
    m = jload(run / "manifest.json")
    cfg = m["config"]
    print(f"cfg: time_steps={cfg['time_steps']} mobility_regimes={cfg['mobility_regimes']} "
          f"policies={cfg['policies']} seeds={cfg['seeds']} pool_size={cfg['pool_size']} "
          f"epsilon_est={cfg['epsilon_est']} solver={cfg['solver']}@{cfg['solver_tol']} "
          f"csi_file={cfg['csi_calibration_file']}")
    for d in m.get("details", []):
        for s in d.get("summary", []):
            print(f"\n  mobility={s['mobility']:6s} policy={s['policy']:22s} "
                  f"outage={100 * (s['qos_outage_rate'] or 0):5.1f}% "
                  f"EE_LT={s['long_term_ee']:.3g}bit/J reconf={s['number_of_reconfigurations']} "
                  f"init={s['number_of_initializations']} reuseMean={s['mean_reuse_interval']} "
                  f"swE={s['ris_switching_energy']:.3g} ctrlE={s['controller_energy']:.3g} "
                  f"failedDesign={s['failed_design_steps']} emptyCfg={s['empty_configuration_steps']}")
    # per-slot trajectory snapshot for seed 1
    recs = rows_of(run / "records.csv")
    header("EXP3 per-slot events (t, policy, mobility, triggered?, reason, rho_total, threshold, installed, sinr_min, qos)")
    for r in recs:
        print(f"  m={r.get('mobility','?'):6s} t={r.get('time_index')} {r.get('policy'):22s} "
              f"trig={r.get('triggered')} {r.get('reason','')[:28]:28s} "
              f"rho={num(r.get('rho_total'))} thr={num(r.get('trigger_threshold'))} "
              f"inst={r.get('installed')} sinr={num(r.get('sinr_min'))} qos={r.get('qos_hold')}")

# ---------- EXP4 ----------
header("EXP4 runtime scaling")
e4 = JR / "exp4_runtime"
if e4.exists():
    for run in sorted(e4.rglob("manifest.json")):
        m = jload(run)
        print(run.parent.name, "complete:", m.get("run_complete"), "success:", m.get("success"))
else:
    print("no exp4 results directory -> exp4 smoke has not been run")

# ---------- smoke json configs vs framework defaults ----------
header("Root smoke_*.json configs")
for name in ("smoke_fast.json", "smoke_scs.json"):
    cfg = jload(ROOT / name)["config"]
    print(f"{name}: solver={cfg['solver']}@{cfg['solver_tol']} max_iter={cfg['solver_max_iter']} "
          f"pool={cfg['pool_size']} steps={cfg['time_steps']} policies={cfg['policies']} "
          f"mobility={cfg['mobility_regimes']} epsilon_est={cfg['epsilon_est']} csi_file={cfg['csi_calibration_file']}")
