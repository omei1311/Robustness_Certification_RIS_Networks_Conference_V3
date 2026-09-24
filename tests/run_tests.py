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
    r_cert_bisection,
    robust_feasibility_indicator,
    robustness_profile,
)
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
    sel = select_stability_aware(wee, rc, 0.2, 1e-3)
    assert sel.index == 1 and sel.n_remaining == 2
    sel2 = select_stability_aware(wee, rc, 0.45, 1e-3)
    assert sel2.index == 1
    rows = r_min_sensitivity(wee, rc, [0.0, 0.5, 0.99], 1e-3)
    assert rows[0]["selected_index"] == 0
    assert rows[1]["selected_index"] == 1  # 0.5*R_max = 0.25 excludes idx0 (rc=0.1)
    assert rows[2]["selected_index"] == 1


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
            assert eig_ok == expect, f"eig oracle wrong at {factor}R_cert"
            assert sdp_ok == expect, f"cvxpy oracle disagrees at {factor}R_cert (got {sdp_ok})"


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


def main() -> None:
    print("=== certification framework tests ===")
    check("t_cert arithmetic", test_t_cert)
    check("pareto + selection logic", test_pareto_and_selection)
    check("F_X monotonicity", test_profile_monotone)
    check("bisection vs grid boundary", test_bisection_matches_grid)
    check("LMI eig vs cvxpy SDP", test_lmi_cvxpy_crosscheck)
    check("robust-design guarantee", test_robust_design_guarantee)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name, msg in FAILED:
            print(f"  FAILED: {name}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
