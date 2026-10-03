"""Read the latest smoke/formal calibration artifacts."""
import json
from pathlib import Path

for pattern, key in (("exp1_csi", "joint_radius.json"), ("exp1b_drift", "drift_rate.json")):
    runs = sorted(Path("journal_results").glob(f"{pattern}/smoke/*/{key}"))
    run = runs[-1]
    p = json.loads(run.read_text(encoding="utf-8"))
    print("=" * 72)
    print(run)
    print("mode:", p.get("mode"), "| complete:", p.get("complete"),
          "| seeds:", p.get("seeds", p.get("requested_seed_count")))
    if key.endswith("joint_radius.json"):
        for e in p["entries"]:
            if e["scope"]["N"] == 32 and e["scope"]["bits"] == 2:
                print(f"  N=32 B=2 snr={e['scope']['estimation_snr_db']:5.1f}: "
                      f"agg joint99={e['aggregate']['epsilon_joint_99']:.4f} "
                      f"per-seed={[round(s['epsilon_joint_99'], 4) for s in e['per_seed']]}")
    else:
        for m, block in p["mobility"].items():
            print(f"  {m:6s}: nu50={block['nu_50']:.3f} nu90={block['nu_90']:.3f} "
                  f"nu95={block['nu_95']:.3f} nu99={block['nu_99']:.3f} "
                  f"selected(q={p['quantile']})={block['nu_selected']:.3f} n={block['sample_count']}")
        print("  unit:", p["unit"], "| slot_duration:", p["slot_duration"],
              "| horizons:", p["horizons_slots"], "| stride:", p["reference_stride"])
