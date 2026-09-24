"""Dominance filtering in the WEE-robustness plane (new; Section 6).

A configuration X_i dominates X_j when

    WEE(X_i) >= WEE(X_j)   and   R_cert(X_i) >= R_cert(X_j)             (*)

with at least one strict inequality.  Candidates satisfying (*) are removed
before threshold-based selection because no policy that is nondecreasing in
both WEE and certified robustness can ever prefer them.  The remaining
nondominated set is the WEE-robustness Pareto frontier of Experiment 2.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def dominates(
    wee_i: float, rcert_i: float, wee_j: float, rcert_j: float
) -> bool:
    """Strict Pareto dominance (maximize both objectives), Eq. (*) above."""
    ge = (wee_i >= wee_j) and (rcert_i >= rcert_j)
    strict = (wee_i > wee_j) or (rcert_i > rcert_j)
    return bool(ge and strict)


def pareto_mask(
    wee: Sequence[float], rcert: Sequence[float]
) -> np.ndarray:
    """Boolean mask of nondominated candidates (True = on the frontier).

    Numerical ties are treated as equality: candidates identical in both
    coordinates are all kept (none dominates the other, no strict inequality).
    """
    wee = np.asarray(wee, dtype=float)
    rcert = np.asarray(rcert, dtype=float)
    n = wee.size
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            if dominates(wee[j], rcert[j], wee[i], rcert[i]):
                mask[i] = False
                break
    return mask


def frontier_points(
    wee: Sequence[float], rcert: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Nondominated (WEE, R_cert) pairs sorted by increasing R_cert."""
    wee = np.asarray(wee, dtype=float)
    rcert = np.asarray(rcert, dtype=float)
    mask = pareto_mask(wee, rcert)
    order = np.argsort(rcert[mask], kind="stable")
    return wee[mask][order], rcert[mask][order]
