"""Stability-aware configuration selection (new; Sections 5-6).

Paper pipeline:

    Candidate Pool -> epsilon_cert -> Dominance/Pareto Filtering
    -> Robustness Threshold -> max WEE.

Selection rules on a certified candidate set.  The robustness axis is the
certified relative uncertainty scaling factor epsilon_cert(X) (see
``certificate.py``); ``R_cert`` is retained as a compatibility alias.

  - WEE-only rule (reference):  X_WEE = argmax WEE over the FULL pool (17)
  - robustness-only (reference): argmax eps_cert over the FULL pool
  - proposed stability-aware rule (Eq. (18), Pareto-first):
        pareto_candidates = candidates[pareto_mask]
        eligible          = pareto_candidates[epsilon_cert >= epsilon_min]
        selected          = argmax WEE(eligible)

All policies operate on the SAME candidate pool; no candidate regeneration
happens inside selection.

Under the linear drift model rho(t) = nu t of Section 5 -- with nu in the
same relative-radius units per second -- the minimum robustness requirement
can equivalently be written as a minimum certified reuse horizon,
eps_cert(X) >= nu * T_min  <=>  T_cert(X) >= T_min                     (19),
with the conditional corollary

    T_cert(X) = eps_cert(X) / nu.                                        (16)

T_cert is a conservative certified reuse horizon under the stated drift
model -- not a prediction of the actual QoS failure time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .pareto import pareto_mask


def t_cert(rcert: float, nu: float) -> float:
    """Certified reuse horizon T_cert = eps_cert / nu  (Eq. (16)).

    ``rcert`` is the certified relative uncertainty factor; ``nu`` is the
    relative channel-drift-rate bound (relative radius per second).
    """
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
    n_pareto: int = 0         # nondominated candidates in the pool
    n_after_rmin: int = 0     # eligible after Pareto + threshold filtering

    @property
    def epsilon_cert(self) -> float:
        """Primary name for the certified robustness value."""
        return self.rcert

    @property
    def n_total(self) -> int:
        """Alias of n_candidates (pool size the policy was applied to)."""
        return self.n_candidates


def _mask_or_compute(
    wee: np.ndarray, rcert: np.ndarray, nondominated: Optional[np.ndarray]
) -> np.ndarray:
    if nondominated is None:
        return pareto_mask(wee, rcert)
    return np.asarray(nondominated, dtype=bool)


def select_wee_only(
    wee: Sequence[float],
    rcert: Sequence[float],
    nu: float,
    nondominated: Optional[Sequence[bool]] = None,
) -> SelectionOutcome:
    """Eq. (17): the conventional WEE-maximizing configuration (full pool)."""
    wee = np.asarray(wee, dtype=float)
    rcert = np.asarray(rcert, dtype=float)
    mask = _mask_or_compute(wee, rcert, nondominated)
    best = int(np.argmax(wee))
    return SelectionOutcome(
        rule="wee_only",
        index=best,
        wee=float(wee[best]),
        rcert=float(rcert[best]),
        t_cert=t_cert(rcert[best], nu),
        n_candidates=int(wee.size),
        n_remaining=int(wee.size),
        n_pareto=int(mask.sum()),
        n_after_rmin=int(wee.size),
    )


def select_robustness_only(
    wee: Sequence[float],
    rcert: Sequence[float],
    nu: float,
    nondominated: Optional[Sequence[bool]] = None,
) -> SelectionOutcome:
    """Reference rule argmax eps_cert (reported for comparison, Section 6)."""
    wee = np.asarray(wee, dtype=float)
    rcert = np.asarray(rcert, dtype=float)
    mask = _mask_or_compute(wee, rcert, nondominated)
    best = int(np.argmax(rcert))
    return SelectionOutcome(
        rule="robustness_only",
        index=best,
        wee=float(wee[best]),
        rcert=float(rcert[best]),
        t_cert=t_cert(rcert[best], nu),
        n_candidates=int(wee.size),
        n_remaining=int(wee.size),
        n_pareto=int(mask.sum()),
        n_after_rmin=int(wee.size),
    )


def select_stability_aware(
    wee: Sequence[float],
    rcert: Sequence[float],
    r_min: float,
    nu: float,
    nondominated: Optional[Sequence[bool]] = None,
) -> SelectionOutcome:
    """Eq. (18), Pareto-first: max WEE subject to eps_cert >= r_min.

    Filtering order follows the paper pipeline: dominance/Pareto filtering
    first, then the minimum-robustness requirement, then WEE maximization
    among the eligible nondominated candidates.  ``r_min`` is a threshold on
    the certified relative uncertainty factor.
    """
    wee = np.asarray(wee, dtype=float)
    rcert = np.asarray(rcert, dtype=float)
    mask = _mask_or_compute(wee, rcert, nondominated)
    pareto_idx = np.flatnonzero(mask)
    eligible = pareto_idx[rcert[pareto_idx] >= r_min]
    if eligible.size == 0:
        raise ValueError(
            f"no nondominated candidate satisfies eps_cert >= {r_min:.4f}"
        )
    best = int(eligible[np.argmax(wee[eligible])])
    return SelectionOutcome(
        rule="stability_aware",
        index=best,
        wee=float(wee[best]),
        rcert=float(rcert[best]),
        t_cert=t_cert(rcert[best], nu),
        n_candidates=int(wee.size),
        n_remaining=int(eligible.size),
        n_pareto=int(mask.sum()),
        n_after_rmin=int(eligible.size),
    )


def r_min_sensitivity(
    wee: Sequence[float],
    rcert: Sequence[float],
    eps_min_fracs: Sequence[float],
    nu: float,
    nondominated: Optional[Sequence[bool]] = None,
) -> List[dict]:
    """Experiment 3 sweep: eps_min = frac * eps_max over the normalized grid.

    Returns one row per fraction with the selected configuration's WEE,
    epsilon_cert, T_cert and the full filtering statistics
    (n_total / n_pareto / n_after_rmin).  Fractions that filter out the
    whole Pareto set are reported with index = None.
    """
    wee = np.asarray(wee, dtype=float)
    rcert = np.asarray(rcert, dtype=float)
    mask = _mask_or_compute(wee, rcert, nondominated)
    eps_max = float(np.max(rcert)) if rcert.size else 0.0
    rows: List[dict] = []
    for frac in eps_min_fracs:
        eps_min = float(frac) * eps_max
        try:
            out = select_stability_aware(wee, rcert, eps_min, nu, nondominated=mask)
            rows.append(
                {
                    "eps_min_frac": float(frac),
                    "eps_min": eps_min,
                    "selected_index": out.index,
                    "selected_wee": out.wee,
                    "selected_rcert": out.rcert,
                    "selected_epsilon_cert": out.rcert,
                    "selected_t_cert_s": out.t_cert,
                    "n_total": out.n_candidates,
                    "n_pareto": out.n_pareto,
                    "n_after_rmin": out.n_after_rmin,
                    "n_remaining": out.n_remaining,
                    "feasible": True,
                }
            )
        except ValueError:
            rows.append(
                {
                    "eps_min_frac": float(frac),
                    "eps_min": eps_min,
                    "selected_index": None,
                    "selected_wee": float("nan"),
                    "selected_rcert": float("nan"),
                    "selected_epsilon_cert": float("nan"),
                    "selected_t_cert_s": float("nan"),
                    "n_total": int(wee.size),
                    "n_pareto": int(mask.sum()),
                    "n_after_rmin": 0,
                    "n_remaining": 0,
                    "feasible": False,
                }
            )
    return rows
