"""Stability-aware configuration selection (new; Sections 5-6).

Selection rules on a certified candidate set:
  - WEE-only rule            X_WEE = argmax WEE                        (17)
  - robustness-only rule     X_R   = argmax R_cert  (reference only)
  - proposed rule            X_sel = argmax WEE  s.t. R_cert >= R_min  (18)

Under the linear drift model rho(t) = nu t of Section 5, the minimum
robustness requirement can equivalently be written as a minimum certified
reuse horizon, R_cert(X) >= nu * T_min  <=>  T_cert(X) >= T_min        (19),
with the conditional corollary

    T_cert(X) = R_cert(X) / nu.                                          (16)

T_cert is a conservative certified reuse horizon under the stated drift
model -- not a prediction of the actual QoS failure time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


def t_cert(rcert: float, nu: float) -> float:
    """Certified reuse horizon T_cert = R_cert / nu  (Eq. (16))."""
    if nu <= 0.0:
        raise ValueError("drift-rate bound nu must be positive")
    return float(rcert) / float(nu)


@dataclass
class SelectionOutcome:
    rule: str
    index: int
    wee: float
    rcert: float
    t_cert: float
    n_candidates: int
    n_remaining: int          # candidates passing the robustness filter


def _argbest(wee: np.ndarray, rcert: np.ndarray, mask: np.ndarray, key: str, rule: str) -> SelectionOutcome:
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        raise ValueError(f"no candidate satisfies the {rule} filter")
    if key == "wee":
        best = idxs[np.argmax(wee[idxs])]
    else:
        best = idxs[np.argmax(rcert[idxs])]
    return int(best)


def select_wee_only(
    wee: Sequence[float], rcert: Sequence[float], nu: float
) -> SelectionOutcome:
    """Eq. (17): the conventional WEE-maximizing configuration."""
    wee = np.asarray(wee, dtype=float)
    rcert = np.asarray(rcert, dtype=float)
    best = _argbest(wee, rcert, np.ones(wee.size, dtype=bool), "wee", "wee_only")
    return SelectionOutcome(
        rule="wee_only",
        index=best,
        wee=float(wee[best]),
        rcert=float(rcert[best]),
        t_cert=t_cert(rcert[best], nu),
        n_candidates=int(wee.size),
        n_remaining=int(wee.size),
    )


def select_robustness_only(
    wee: Sequence[float], rcert: Sequence[float], nu: float
) -> SelectionOutcome:
    """Reference rule argmax R_cert (reported for comparison, Section 6)."""
    wee = np.asarray(wee, dtype=float)
    rcert = np.asarray(rcert, dtype=float)
    best = _argbest(wee, rcert, np.ones(wee.size, dtype=bool), "rcert", "robustness_only")
    return SelectionOutcome(
        rule="robustness_only",
        index=best,
        wee=float(wee[best]),
        rcert=float(rcert[best]),
        t_cert=t_cert(rcert[best], nu),
        n_candidates=int(wee.size),
        n_remaining=int(wee.size),
    )


def select_stability_aware(
    wee: Sequence[float],
    rcert: Sequence[float],
    r_min: float,
    nu: float,
) -> SelectionOutcome:
    """Eq. (18): max WEE subject to R_cert >= R_min."""
    wee = np.asarray(wee, dtype=float)
    rcert = np.asarray(rcert, dtype=float)
    mask = rcert >= r_min
    if not np.any(mask):
        raise ValueError(
            f"no candidate satisfies R_cert >= R_min = {r_min:.4f}"
        )
    best = _argbest(wee, rcert, mask, "wee", "stability_aware")
    return SelectionOutcome(
        rule="stability_aware",
        index=best,
        wee=float(wee[best]),
        rcert=float(rcert[best]),
        t_cert=t_cert(rcert[best], nu),
        n_candidates=int(wee.size),
        n_remaining=int(np.sum(mask)),
    )


def r_min_sensitivity(
    wee: Sequence[float],
    rcert: Sequence[float],
    r_min_fracs: Sequence[float],
    nu: float,
) -> List[dict]:
    """Experiment 3 sweep: R_min = frac * R_max over the normalized grid.

    Returns one row per fraction with the selected configuration's WEE,
    R_cert, T_cert and the number of candidates remaining after filtering.
    Fractions that filter out the whole pool are reported with index = None.
    """
    wee = np.asarray(wee, dtype=float)
    rcert = np.asarray(rcert, dtype=float)
    r_max = float(np.max(rcert)) if rcert.size else 0.0
    rows: List[dict] = []
    for frac in r_min_fracs:
        r_min = float(frac) * r_max
        try:
            out = select_stability_aware(wee, rcert, r_min, nu)
            rows.append(
                {
                    "r_min_frac": float(frac),
                    "r_min": r_min,
                    "selected_index": out.index,
                    "selected_wee": out.wee,
                    "selected_rcert": out.rcert,
                    "selected_t_cert_s": out.t_cert,
                    "n_remaining": out.n_remaining,
                    "feasible": True,
                }
            )
        except ValueError:
            rows.append(
                {
                    "r_min_frac": float(frac),
                    "r_min": r_min,
                    "selected_index": None,
                    "selected_wee": float("nan"),
                    "selected_rcert": float("nan"),
                    "selected_t_cert_s": float("nan"),
                    "n_remaining": 0,
                    "feasible": False,
                }
            )
    return rows
