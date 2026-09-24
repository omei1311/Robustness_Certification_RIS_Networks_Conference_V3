"""Shared experiment helpers: certified-pool construction with caching."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import SimConfig, default_cert_config  # noqa: E402
from src.candidate_pool import build_pool, load_pool, save_pool  # noqa: E402
from src.certificate import r_cert_bisection  # noqa: E402
from src.io_utils import RESULTS_DIR, ensure_dirs  # noqa: E402
from src.ris_base import generate_channel_drop  # noqa: E402

POOL_PREFIX = str(RESULTS_DIR / "pool_cache")


def make_system() -> Tuple[SimConfig, object]:
    cfg = SimConfig().validate()
    cc = default_cert_config()
    return cfg, cc


def nominal_drop(cfg: SimConfig):
    """The shared nominal channel drop (common comparison axis, Section 9)."""
    rng = np.random.default_rng(cfg.seed)
    return generate_channel_drop(cfg, rng)


def build_and_certify(
    cfg: SimConfig,
    cert_cfg,
    rebuild: bool = False,
    log=print,
) -> Tuple[List, Dict]:
    """Build the candidate pool, certify every member, cache to results/."""
    ensure_dirs()
    cache_ok = False
    if not rebuild and Path(POOL_PREFIX + ".npz").exists():
        try:
            pool, meta = load_pool(POOL_PREFIX)
            if (
                meta.get("pool_target_size") == cert_cfg.pool_target_size
                and meta.get("channel_seed") == cert_cfg.channel_seed
                and meta.get("cfg_seed") == cfg.seed
                and all(np.isfinite(c.r_cert) for c in pool)
                and len(pool) > 0
            ):
                cache_ok = True
                log(f"[pool] loaded {len(pool)} certified candidates from cache")
        except Exception as exc:  # pragma: no cover - cache self-heal
            log(f"[pool] cache unreadable ({exc}); rebuilding")

    if not cache_ok:
        t0 = time.time()
        drop = nominal_drop(cfg)
        pool, stats = build_pool(drop, cfg, cert_cfg)
        if not pool:
            raise RuntimeError(
                "candidate pool is empty: loosen the generation protocol"
            )
        log(f"[pool] accepted {len(pool)} / {stats['attempts']} attempts "
            f"({stats['reject_reasons']}) in {time.time() - t0:.1f}s")
        t0 = time.time()
        n_checks = 0
        for c in pool:
            res = r_cert_bisection(
                c.w, c.H, cfg,
                eps_hi=cert_cfg.bisection_eps_hi,
                eps_hi_max=cert_cfg.bisection_eps_hi_max,
                tol=cert_cfg.bisection_tol,
                max_iter=cert_cfg.bisection_max_iter,
            )
            c.r_cert = res.r_cert
            c.cert_info = {
                "bracket": list(res.bracket),
                "n_bisection_iter": res.n_bisection_iter,
                "n_feasibility_checks": res.n_feasibility_checks,
                "binding_user": list(res.binding_user) if res.binding_user else None,
                "binding_margin": res.binding_margin,
            }
            n_checks += res.n_feasibility_checks
        log(f"[cert] certified {len(pool)} candidates "
            f"({n_checks} oracle calls) in {time.time() - t0:.1f}s")
        meta = {
            "pool_target_size": cert_cfg.pool_target_size,
            "channel_seed": cert_cfg.channel_seed,
            "cfg_seed": cfg.seed,
            "generation_stats": stats,
            "n_oracle_calls": n_checks,
        }
        npz, jsn = save_pool(POOL_PREFIX, pool, meta)
        log(f"[pool] cached -> {npz}, {jsn}")

    return pool, meta
