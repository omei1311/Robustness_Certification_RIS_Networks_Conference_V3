"""Smoke C before/after comparison."""
import json
from pathlib import Path


def totals(run):
    m = json.loads((Path(run) / "manifest.json").read_text(encoding="utf-8"))
    summaries = [x for sd in m["seed_details"] for x in (sd.get("summary") or [])]
    strict = sum(x["strict_oracle_calls"] for x in summaries)
    rt = sum(x["runtime"] for x in summaries)
    pool = next((x for x in summaries if "candidate_pool_requests" in x), {})
    return m["success"], strict, rt, pool, summaries


for label, run in (("BEFORE (no cache)", "journal_results/exp3_dynamic/smoke/20261003T220217389823_483c764bb209"),
                   ("AFTER  (cache)   ", "journal_results/exp3_dynamic/smoke/20261003T230432486046_483c764bb209")):
    ok, strict, rt, pool, _ = totals(run)
    print(f"{label} success={ok} strict={strict:5d} policy_rt={rt:5.0f}s "
          f"pool(req/build/hit)={pool.get('candidate_pool_requests')}/"
          f"{pool.get('candidate_pool_builds')}/{pool.get('candidate_pool_cache_hits')}")

ok, strict, rt, pool, summaries = totals("journal_results/exp3_dynamic/smoke/20261003T230432486046_483c764bb209")
for s in summaries:
    print(f"  {s['mobility']:6s} {s['policy']:22s} reconf={s['number_of_reconfigurations']:2d} "
          f"outage={100 * (s['qos_outage_rate'] or 0):4.0f}% EE={s['long_term_ee']:.4g} "
          f"strict={s['strict_oracle_calls']:4d} rt={s['runtime']:.0f}s")
