"""Unit/validation tests for the certification framework.

Run:  .venv\\Scripts\\python.exe tests\\run_tests.py
(no pytest dependency; plain asserts with readable output)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import SimConfig, default_cert_config  # noqa: E402
from src.candidate_pool import build_pool  # noqa: E402
from src.certificate import (  # noqa: E402
    epsilon_cert_bisection,
    r_cert_bisection,
    robust_feasibility_indicator,
    robustness_profile,
)
from src.config_v3 import config_fingerprint  # noqa: E402
from src.pareto import pareto_mask  # noqa: E402
from src.robust_oracle import robust_feasible_cvxpy  # noqa: E402
from src.selection import (  # noqa: E402
    r_min_sensitivity,
    select_stability_aware,
    select_wee_only,
    t_cert,
)
from src.ris_base import robust_check  # noqa: E402

PASSED = []
FAILED = []


def check(name: str, fn) -> None:
    t0 = time.time()
    try:
        fn()
        PASSED.append(name)
        print(f"[PASS] {name}  ({time.time() - t0:.1f}s)")
    except AssertionError as exc:
        FAILED.append((name, str(exc)))
        print(f"[FAIL] {name}: {exc}")
    except Exception as exc:  # pragma: no cover
        FAILED.append((name, repr(exc)))
        print(f"[ERROR] {name}: {exc!r}")


def make_pool(n: int = 3):
    cfg = SimConfig().validate()
    cc = default_cert_config()
    rng = np.random.default_rng(cfg.seed)
    from src.ris_base import generate_channel_drop

    drop = generate_channel_drop(cfg, rng)
    pool, _ = build_pool(
        drop, cfg, cc.with_overrides(pool_target_size=n, pool_max_attempts=200)
    )
    assert len(pool) >= 1, "pool generation unexpectedly empty"
    return cfg, pool


# ---------------------------------------------------------------------- #

def test_t_cert():
    assert abs(t_cert(0.10, 0.002) - 50.0) < 1e-12
    try:
        t_cert(0.1, 0.0)
        raise AssertionError("nu=0 must raise")
    except ValueError:
        pass


def test_pareto_and_selection():
    wee = np.array([1.0, 0.9, 0.5, 0.3])
    rc = np.array([0.1, 0.5, 0.4, 0.05])
    mask = pareto_mask(wee, rc)
    # idx1 (0.9,0.5) dominates idx2 (0.5,0.4); idx3 dominated by all but idx2? no:
    # (0.3,0.05) dominated by (0.9,0.5) and (0.5,0.4) and (1.0,0.1)
    assert mask.tolist() == [True, True, False, False], mask
    out = select_wee_only(wee, rc, 1e-3)
    assert out.index == 0
    # Pareto-first proposed rule: nondominated {0,1}, threshold 0.2 keeps
    # only idx1 (rc=0.1 < 0.2), so n_pareto=2 and n_after_rmin=1.
    sel = select_stability_aware(wee, rc, 0.2, 1e-3)
    assert sel.index == 1 and sel.n_pareto == 2 and sel.n_after_rmin == 1
    assert sel.n_remaining == sel.n_after_rmin
    sel2 = select_stability_aware(wee, rc, 0.45, 1e-3)
    assert sel2.index == 1
    rows = r_min_sensitivity(wee, rc, [0.0, 0.5, 0.99], 1e-3)
    assert rows[0]["selected_index"] == 0
    assert rows[1]["selected_index"] == 1  # 0.5*eps_max = 0.25 excludes idx0 (rc=0.1)
    assert rows[2]["selected_index"] == 1


def test_fingerprint_and_metadata():
    """Cache fingerprint determinism/sensitivity + candidate provenance."""
    cfg = SimConfig().validate()
    cc = default_cert_config()
    fp1 = config_fingerprint(cfg, cc)
    assert fp1 == config_fingerprint(cfg, cc), "fingerprint must be stable"
    assert fp1 != config_fingerprint(cfg, cc.with_overrides(pool_target_size=5))
    assert fp1 != config_fingerprint(cfg.with_overrides(gamma=2.5), cc)
    assert fp1 != config_fingerprint(cfg, cc.with_overrides(align_jitter_grid=(0.0, 0.5)))
    # Paper fixes B = 2: bit resolution must not be a diversity knob.
    assert cc.bits_grid == (2,), f"bits_grid must be (2,), got {cc.bits_grid}"

    # Alias compatibility: r_cert == epsilon_cert under relative radius.
    assert r_cert_bisection is epsilon_cert_bisection

    from src.ris_base import generate_channel_drop

    cc_small = cc.with_overrides(pool_target_size=2, pool_max_attempts=120)
    drop = generate_channel_drop(cfg, np.random.default_rng(cc.channel_seed))
    pool, _ = build_pool(drop, cfg, cc_small)
    assert len(pool) >= 1, "pool unexpectedly empty"
    fp_small = config_fingerprint(cfg, cc_small)
    for c in pool:
        assert c.channel_seed == cc.channel_seed, "channel seed not stamped"
        assert c.config_fingerprint == fp_small, "fingerprint not stamped"
        assert c.align_jitter in cc.align_jitter_grid
        c.r_cert = 0.123  # simulate post-certification stamping
        assert c.epsilon_cert == c.r_cert  # compat alias
        rec = c.to_record()
        for key in ("epsilon_cert", "channel_seed", "align_jitter",
                    "config_fingerprint"):
            assert key in rec, f"to_record missing {key}"


def test_configuration_signature():
    """X = (W, Theta) identity: stable, W-sensitive, rounding-tolerant."""
    from src.candidate_pool import configuration_signature
    from src.ris_base import random_theta

    cfg = SimConfig().validate()
    rng = np.random.default_rng(99)
    w = (
        rng.standard_normal((cfg.L, cfg.K, cfg.M))
        + 1j * rng.standard_normal((cfg.L, cfg.K, cfg.M))
    ) / np.sqrt(2.0)
    theta = random_theta(cfg.N, 2, rng)

    sig = configuration_signature(w, theta, 2)
    # identical W and Theta -> identical signature
    assert configuration_signature(w.copy(), theta.copy(), 2) == sig
    # same Theta, different W -> different signature
    assert configuration_signature(w * 1.5, theta, 2) != sig
    assert configuration_signature(w + 0.25, theta, 2) != sig
    # same W, different Theta -> different signature
    assert configuration_signature(w, np.roll(theta, 1), 2) != sig
    # numerical perturbation below the rounding threshold -> same signature
    assert configuration_signature(w + 1e-9, theta, 2) == sig
    assert configuration_signature(w * (1.0 + 1e-9), theta, 2) == sig
    # a different phase resolution changes the index encoding -> different
    assert configuration_signature(w, theta, 3) != sig


def test_certificate_status():
    """CertificateResult.status semantics (three states)."""
    cfg, pool = make_pool(2)
    cc = default_cert_config()

    # Certify each candidate once; pick the most robust one.
    certs = [
        epsilon_cert_bisection(
            c.w, c.H, cfg,
            eps_hi=cc.bisection_eps_hi, eps_hi_max=cc.bisection_eps_hi_max,
            tol=cc.bisection_tol, max_iter=cc.bisection_max_iter,
        )
        for c in pool
    ]
    # Normal robust candidates on this system: boundary well below eps_hi.
    assert all(r.status == "EXACT_BRACKET" for r in certs), \
        [r.status for r in certs]
    assert all(r.bracket[1] > r.bracket[0] for r in certs)

    # LOWER_BOUND_CENSORED: shrink the bracket cap below the true boundary.
    c = pool[int(np.argmax([r.r_cert for r in certs]))]
    res_cens = epsilon_cert_bisection(
        c.w, c.H, cfg, eps_hi=1.0e-3, eps_hi_max=2.0e-3,
        tol=cc.bisection_tol, max_iter=cc.bisection_max_iter,
    )
    assert res_cens.status == "LOWER_BOUND_CENSORED"
    assert res_cens.bracket == (2.0e-3, 2.0e-3)
    assert res_cens.r_cert == 2.0e-3  # lower bound, not an exact certificate

    # NOMINAL_INFEASIBLE: uniformly attenuated beams miss the QoS target.
    res_bad = epsilon_cert_bisection(
        c.w * 0.01, c.H, cfg,
        eps_hi=cc.bisection_eps_hi, eps_hi_max=cc.bisection_eps_hi_max,
        tol=cc.bisection_tol, max_iter=cc.bisection_max_iter,
    )
    assert res_bad.status == "NOMINAL_INFEASIBLE"
    assert res_bad.r_cert == 0.0


def test_certificate_boundary():
    """0.99*eps_cert feasible, 1.10*eps_cert infeasible, 1.00* feasible.

    1.01 is deliberately NOT asserted infeasible: the bisection width
    (delta_eps = 1e-4) and feasibility tolerance permit boundary error
    around it.
    """
    cfg, pool = make_pool(3)
    cc = default_cert_config()
    certs = [
        epsilon_cert_bisection(
            c.w, c.H, cfg,
            eps_hi=cc.bisection_eps_hi, eps_hi_max=cc.bisection_eps_hi_max,
            tol=cc.bisection_tol, max_iter=cc.bisection_max_iter,
        )
        for c in pool
    ]
    # Use a well-separated certificate so that 10% clearly exceeds tol.
    pairs = [(c, r) for c, r in zip(pool, certs)
             if r.status == "EXACT_BRACKET" and r.r_cert > 0.01]
    assert pairs, "no candidate with a well-separated certificate"
    for c, r in pairs:
        eps = r.epsilon_cert
        assert robust_feasibility_indicator(c.w, c.H, 0.99 * eps, cfg) is True
        assert robust_feasibility_indicator(c.w, c.H, 1.00 * eps, cfg) is True
        assert robust_feasibility_indicator(c.w, c.H, 1.10 * eps, cfg) is False


def test_profile_monotone():
    cfg, pool = make_pool(3)
    grid = np.linspace(0.0, 0.4, 21)
    for c in pool:
        prof = robustness_profile(c.w, c.H, grid, cfg)
        diffs = np.diff(prof)
        assert np.all(diffs <= 1e-12), f"F_X not monotone: {prof}"
        assert prof[0] == 1.0, "nominally feasible candidate must have F(0)=1"


def test_bisection_matches_grid():
    cfg, pool = make_pool(2)
    cc = default_cert_config()
    grid = np.linspace(0.0, 0.4, 401)
    for c in pool:
        res = r_cert_bisection(
            c.w, c.H, cfg, eps_hi=cc.bisection_eps_hi, eps_hi_max=cc.bisection_eps_hi_max,
            tol=cc.bisection_tol, max_iter=cc.bisection_max_iter,
        )
        prof = robustness_profile(c.w, c.H, grid, cfg)
        feasible = grid[prof > 0.5]
        grid_boundary = feasible.max() if feasible.size else 0.0
        # both estimates approach the true boundary from below: the grid is
        # bounded by the grid step, the bisection by tol, so they must agree
        # within step + tol on either side
        step = grid[1] - grid[0]
        slack = 2.0 * step + 4.0 * cc.bisection_tol
        assert abs(res.r_cert - grid_boundary) <= slack, (
            f"cert {res.r_cert:.5f} disagrees with grid boundary {grid_boundary:.5f}"
        )


def test_lmi_cvxpy_crosscheck():
    cfg, pool = make_pool(2)
    for c in pool:
        res = r_cert_bisection(c.w, c.H, cfg, eps_hi=0.6, eps_hi_max=1.2,
                               tol=1e-4, max_iter=40)
        rc = res.r_cert
        if rc < 1e-3:
            continue  # fragile config: skip borderline radii
        for factor, expect in ((0.7, True), (1.4, False)):
            eps = factor * rc
            eig_ok = robust_feasibility_indicator(c.w, c.H, eps, cfg)
            sdp_ok = robust_feasible_cvxpy(c.w, c.H, eps, cfg)
            assert eig_ok == expect, f"eig oracle wrong at {factor}*eps_cert"
            assert sdp_ok == expect, f"cvxpy oracle disagrees at {factor}*eps_cert (got {sdp_ok})"


def test_robust_design_guarantee():
    """Candidates designed at radius eps_d must certify feasible at eps_d.

    The worst-case Cauchy-Schwarz power control guarantees robust QoS at the
    design radius, so the LMI oracle must agree (F_X(eps_d) = 1).
    """
    cfg = SimConfig().validate()
    cc = default_cert_config()
    rng = np.random.default_rng(cfg.seed)
    from src.ris_base import generate_channel_drop

    drop = generate_channel_drop(cfg, rng)
    found = 0
    seed = int(cc.pool_seed) + 5000
    for _ in range(120):
        seed += 1
        from src.candidate_pool import generate_candidate

        cand, _ = generate_candidate(drop, cfg, cc, seed)
        if cand is not None and cand.design_eps > 0:
            found += 1
            chk = robust_check(cand.w, cand.H, cand.design_eps, cfg)
            assert chk.feasible, (
                f"robust-designed candidate (eps_d={cand.design_eps}) "
                f"infeasible at its design radius"
            )
            if found >= 3:
                break
    assert found >= 1, "no robust-designed candidate generated in 120 attempts"


def toy_pool():
    from types import SimpleNamespace
    return [SimpleNamespace(index=i, wee=w, epsilon_cert=e,
                            configuration_signature=chr(65+i), config_fingerprint="test",
                            cert_info={"status": "EXACT_BRACKET"})
            for i, (w, e) in enumerate(((10, .02), (9, .06), (7, .10)))]


def test_generalization_thresholds():
    from src.generalization import compare_pool
    for threshold, expected in ((0, 0), (.05, 1), (.08, 2), (.20, None)):
        cc = default_cert_config().with_overrides(epsilon_design=threshold)
        rows, failures = compare_pool(toy_pool(), cc, {})
        row = next(r for r in rows if r["threshold_type"] == "fixed")
        assert row["selected_candidate_index"] == expected
        assert row["proposed_feasible"] == (expected is not None)
        assert not failures


def test_generalization_normalized():
    from src.generalization import compare_pool
    rows, failures = compare_pool(toy_pool(), default_cert_config(), {})
    sweep = [r for r in rows if r["threshold_type"] == "normalized"]
    assert sweep[0]["selected_wee"] == rows[0]["selected_wee"] == 10
    assert all(a["selected_wee"] >= b["selected_wee"] for a, b in zip(sweep, sweep[1:]))
    assert not failures


def test_generalization_same_pool():
    from unittest.mock import patch
    import src.generalization as g
    seen = []
    def wrapper(fn):
        def call(wee, eps, *args, **kwargs):
            seen.append((id(wee), id(eps), tuple(zip(wee, eps))))
            return fn(wee, eps, *args, **kwargs)
        return call
    pool = toy_pool()
    with patch.object(g, "select_wee_only", wrapper(g.select_wee_only)), \
         patch.object(g, "select_robustness_only", wrapper(g.select_robustness_only)), \
         patch.object(g, "select_stability_aware", wrapper(g.select_stability_aware)):
        rows, _ = g.compare_pool(pool, default_cert_config(), {})
    assert len(seen) == 9 and len(set(seen)) == 1
    assert len({r["pool_signature"] for r in rows}) == 1
    assert {r["configuration_signature"] for r in rows} <= {c.configuration_signature for c in pool}


def test_generalization_missing_aggregation():
    from src.generalization import aggregate, compare_pool, stats
    cc = default_cert_config().with_overrides(epsilon_design=.20)
    rows, _ = compare_pool(toy_pool(), cc, {})
    empty, _ = compare_pool([], cc.with_overrides(channel_seed=cc.channel_seed+1), {}, error="empty")
    rows[0]["t_cert"] = float("nan")
    details = [{"channel_seed": cc.channel_seed, "pool_size": 3, "status": "valid"},
               {"channel_seed": cc.channel_seed+1, "pool_size": 0, "status": "failed"}]
    summary = aggregate(rows+empty, details, cc, [cc.channel_seed, cc.channel_seed+1])
    assert summary["fixed_threshold"]["proposed"]["infeasible_seed_count"] == 1
    assert summary["fixed_threshold"]["proposed"]["feasibility_rate"] == 0
    assert summary["fixed_threshold"]["paired_vs_wee_only"]["n_pairs"] == 0
    assert summary["pool_statistics"]["failed_seed_count"] == 1
    assert stats([None, float("nan")])["mean"] is None
    assert aggregate([], [], cc, []) ["valid_seed_count"] == 0


def test_generalization_cache_isolation():
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    from experiments.shared import build_and_certify
    from src.candidate_pool import save_pool
    cfg, pool = make_pool(1)
    cc = default_cert_config().with_overrides(channel_seed=pool[0].channel_seed)
    fp = config_fingerprint(cfg, cc)
    pool[0].config_fingerprint = fp
    pool[0].r_cert = .1
    pool[0].cert_info = {"status": "EXACT_BRACKET"}
    with TemporaryDirectory() as tmp:
        prefix = str(Path(tmp)/"pool")
        save_pool(prefix, pool, {"channel_seed": cc.channel_seed, "config_fingerprint": fp})
        with patch("experiments.shared.build_pool", return_value=([], {"attempts": 400})) as build:
            loaded, _ = build_and_certify(cfg, cc, cache_prefix=prefix, log=lambda _: None)
            assert len(loaded) == 1 and build.call_count == 0
            other, meta = build_and_certify(cfg, cc.with_overrides(channel_seed=60002),
                                           cache_prefix=prefix, allow_empty=True, log=lambda _: None)
            assert not other and build.call_count == 1
            assert meta["channel_seed"] == 60002
        pool[0].configuration_signature = "corrupted"
        save_pool(prefix, pool, {"channel_seed": cc.channel_seed, "config_fingerprint": fp})
        with patch("experiments.shared.build_pool", return_value=([], {})) as build:
            build_and_certify(cfg, cc, cache_prefix=prefix, allow_empty=True, log=lambda _: None)
            assert build.call_count == 1


def test_generalization_zero_and_ties():
    from src.generalization import compare_pool, ratio
    pool = toy_pool()
    pool[1].wee = pool[0].wee
    pool[0].epsilon_cert = 0
    rows, failures = compare_pool(pool, default_cert_config(), {})
    assert not failures
    assert rows[2]["delta_epsilon_percent_vs_wee_only"] is None
    assert ratio(1, 1e-10) is None
    for c in pool:
        c.epsilon_cert = 0
    rows, failures = compare_pool(pool, default_cert_config(), {})
    assert not failures
    assert all(r["selected_epsilon_ratio"] is None for r in rows)


def test_generalization_failure_continuation():
    from unittest.mock import patch
    import warnings
    from experiments import exp3_generalization_check as exp
    # First seed raises, second has a short (nonempty) pool, third is empty.
    cc = default_cert_config()
    pool = toy_pool()
    outputs = [RuntimeError("injected failure"), (pool, {"generation_stats": {
        "attempts": 400, "accepted": 3, "unique_theta_count": 3, "unique_configuration_count": 3}}),
        ([], {"generation_stats": {"attempts": 400, "accepted": 0}})]
    with patch.object(exp, "build_and_certify", side_effect=outputs) as build, \
         patch.object(exp, "save_csv") as csv, patch.object(exp, "save_json"), \
         patch.object(exp, "save_tables"), patch.object(exp, "plot_generalization", return_value=[]), \
         patch("builtins.print"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = exp.main(n_seeds=3)
    assert build.call_count == 3 and csv.call_count == 3
    assert result["run_complete"] and result["valid_seed_count"] == 1
    assert len(result["failed_seeds"]) == 2
    assert result["pool_statistics"]["short_pool_seeds"] == [60002, 60003]
    assert result["channel_seeds"] == [60001, 60002, 60003]
    records = csv.call_args.args[1]
    assert len(records) == 27
    assert records[0]["pool_size"] is None  # unknown, not a fabricated empty pool
    assert records[-1]["pool_size"] == 0


def main() -> None:
    print("=== certification framework tests ===")
    check("t_cert arithmetic", test_t_cert)
    check("pareto + selection logic", test_pareto_and_selection)
    check("fingerprint + candidate metadata", test_fingerprint_and_metadata)
    check("configuration signature (X = (W, Theta))", test_configuration_signature)
    check("certificate status semantics", test_certificate_status)
    check("certificate boundary (0.99/1.00/1.10)", test_certificate_boundary)
    check("F_X monotonicity", test_profile_monotone)
    check("bisection vs grid boundary", test_bisection_matches_grid)
    check("LMI eig vs cvxpy SDP", test_lmi_cvxpy_crosscheck)
    check("robust-design guarantee", test_robust_design_guarantee)
    check("generalization fixed thresholds + infeasibility", test_generalization_thresholds)
    check("generalization normalized zero + monotonicity", test_generalization_normalized)
    check("generalization same-pool calls", test_generalization_same_pool)
    check("generalization missing/NaN/infeasible aggregation", test_generalization_missing_aggregation)
    check("generalization cache seed + signature isolation", test_generalization_cache_isolation)
    check("generalization zero denominators + ties", test_generalization_zero_and_ties)
    check("generalization failure checkpoint + continuation", test_generalization_failure_continuation)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name, msg in FAILED:
            print(f"  FAILED: {name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
