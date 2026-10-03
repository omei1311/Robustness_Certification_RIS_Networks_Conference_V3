"""Publication-style figures for Experiments 1-3 (new).

Conventions: colorblind-friendly Okabe-Ito palette, one color per semantic
role (candidate / Pareto / selection markers), direct annotation instead of
heavy legends where possible, 300 dpi PNG output through io_utils.save_fig.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .io_utils import save_fig

# Okabe-Ito colorblind-safe palette
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"
C_GREY = "#7F7F7F"
C_BLACK = "#000000"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


# ---------------------------------------------------------------------- #
# Experiment 1: certificate validation                                    #
# ---------------------------------------------------------------------- #

def plot_exp1(
    configs: List[Dict],
    gamma_target: float,
    out_name: str = "exp1_certificate_validation",
    subdir: str = "",
):
    """Two-panel certificate validation figure (Section 9, Experiment 1).

    configs: list of dicts with keys
        label, r_cert, radii (grid), sinr_mean, sinr_p5, sinr_min (arrays,
        worst-user SINR statistics), violation (array of empirical QoS
        violation rates).
    """
    n = len(configs)
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_RED][:n]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    def xvals(c):
        """Normalized factor eps / epsilon_cert so all configs share one axis."""
        rc = c["r_cert"]
        return np.asarray(c["radii"]) / rc if rc > 0 else np.asarray(c["radii"])

    ax = axes[0]
    for c, col in zip(configs, colors):
        x = xvals(c)
        ax.plot(x, 10.0 * np.log10(np.maximum(c["sinr_mean"], 1e-12)),
                color=col, lw=1.6, label=c["label"])
        ax.plot(x, 10.0 * np.log10(np.maximum(c["sinr_p5"], 1e-12)),
                color=col, lw=1.0, ls="--", alpha=0.8)
        if c.get("sinr_wc") is not None:
            ax.plot(x, 10.0 * np.log10(np.maximum(c["sinr_wc"], 1e-12)),
                    color=col, lw=1.1, ls="-.", alpha=0.9)
    ax.axvline(1.0, color=C_GREY, lw=1.0, ls=":", alpha=0.9)
    ax.annotate(r"$\varepsilon = \varepsilon_{\rm cert}$", xy=(1.0, ax.get_ylim()[1]),
                xytext=(-4, -12), textcoords="offset points", fontsize=8,
                ha="right")
    tgt_db = 10.0 * np.log10(gamma_target)
    ax.axhline(tgt_db, color=C_BLACK, lw=1.2, alpha=0.8)
    ax.annotate(f"QoS target ({tgt_db:.0f} dB)",
                xy=(0.02, tgt_db), xycoords=("axes fraction", "data"),
                xytext=(2, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(r"realized uncertainty factor $\varepsilon / \varepsilon_{\rm cert}$")
    ax.set_ylabel("worst-user SINR (dB)")
    ax.set_title("(a) worst-user SINR (solid mean, dashed p5, dash-dot worst case)")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    for c, col in zip(configs, colors):
        ax.plot(xvals(c), 100.0 * np.asarray(c["violation"]),
                color=col, lw=1.6, label=c["label"])
    ax.axvline(1.0, color=C_GREY, lw=1.0, ls=":", alpha=0.9)
    ax.set_ylim(-2, 102)
    ax.set_xlabel(r"realized uncertainty factor $\varepsilon / \varepsilon_{\rm cert}$")
    ax.set_ylabel("empirical QoS violation rate (%)")
    ax.set_title(r"(b) violation rate; dotted line marks $\varepsilon = \varepsilon_{\rm cert}$")
    fig.tight_layout()
    path = save_fig(fig, out_name, subdir=subdir)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- #
# Experiment 2: WEE-robustness landscape                                  #
# ---------------------------------------------------------------------- #

def plot_exp2(
    wee: Sequence[float],
    rcert: Sequence[float],
    nondominated: np.ndarray,
    i_wee: int,
    i_rob: int,
    i_sel: Optional[int],
    epsilon_design: float,
    out_name: str = "exp2_wee_robustness_landscape",
    subdir: str = "",
):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    dom = ~nondominated
    ax.scatter(np.asarray(rcert)[dom], np.asarray(wee)[dom],
               s=26, c=C_GREY, alpha=0.75, label="dominated", zorder=2)
    ax.scatter(np.asarray(rcert)[nondominated], np.asarray(wee)[nondominated],
               s=46, marker="o", facecolors="none", edgecolors=C_BLUE, lw=1.6,
               label="nondominated (Pareto)", zorder=3)

    # Pareto step line
    rc = np.asarray(rcert)[nondominated]
    wr = np.asarray(wee)[nondominated]
    if rc.size:
        order = np.argsort(rc)
        ax.plot(rc[order], wr[order], color=C_BLUE, lw=1.0, alpha=0.6, zorder=2)

    def mark(i, color, symbol, text, dy):
        ax.scatter([rcert[i]], [wee[i]], s=120, c=color, marker=symbol,
                   zorder=5, edgecolor=C_BLACK, linewidth=0.6)
        ax.annotate(text, xy=(rcert[i], wee[i]), xytext=(6, dy),
                    textcoords="offset points", fontsize=8, color=color)

    mark(i_wee, C_ORANGE, "^", "WEE-only choice", 8)
    mark(i_rob, C_GREEN, "s", "robustness-only", -14)
    if i_sel is not None and i_sel != i_wee:
        mark(i_sel, C_RED, "D",
             r"stability-aware ($\varepsilon_{\min}=0.9\varepsilon_{\max}$)", 10)

    ax.axvline(epsilon_design, color=C_BLACK, lw=1.0, ls="--", alpha=0.7)
    ax.annotate(r"design radius $\varepsilon$",
                xy=(epsilon_design, ax.get_ylim()[0]),
                xytext=(-8, 14), textcoords="offset points", rotation=90,
                fontsize=8, ha="right")

    ax.set_xlabel(r"robustness certificate $\varepsilon_{\rm cert}$ (relative uncertainty factor)")
    ax.set_ylabel(r"WEE (bit/s/Hz per watt)")
    ax.set_title("Candidate pool in the WEE-robustness plane")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = save_fig(fig, out_name, subdir=subdir)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- #
# Experiment 3: selection sensitivity                                     #
# ---------------------------------------------------------------------- #

def plot_exp3(
    rows: List[dict],
    rule_points: Dict[str, Dict],
    nu: float,
    out_name: str = "exp3_selection_sensitivity",
    subdir: str = "",
):
    """Sensitivity figure: selected WEE / T_cert / remaining count vs eps_min."""
    fr = np.asarray([r["eps_min_frac"] for r in rows], dtype=float)
    wee = np.asarray([r["selected_wee"] for r in rows], dtype=float)
    tcert = np.asarray([r["selected_t_cert_s"] for r in rows], dtype=float)
    rem = np.asarray([r["n_remaining"] for r in rows], dtype=float)
    feas = np.asarray([bool(r["feasible"]) for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.3))

    ax = axes[0]
    m = feas
    ax.plot(fr[m], wee[m], color=C_BLUE, lw=1.6, marker="o", ms=4)
    if (~m).any():
        ax.scatter(fr[~m], np.full(int((~m).sum()), np.nanmin(wee[m]),
                                   dtype=float), marker="x", c=C_RED, s=40)
        ax.annotate("infeasible\n(no candidate)", xy=(fr[~m][0], np.nanmin(wee[m])),
                    xytext=(6, 10), textcoords="offset points", fontsize=8,
                    color=C_RED)
    for name, col, sym in (
        ("wee_only", C_ORANGE, "^"),
        ("robustness_only", C_GREEN, "s"),
        ("stability_aware", C_RED, "D"),
    ):
        p = rule_points.get(name)
        if p is not None:
            ax.axhline(p["selected_wee"], color=col, lw=0.9, ls=":", alpha=0.8)
    ax.set_xlabel(r"$\varepsilon_{\min}/\varepsilon_{\max}$")
    ax.set_ylabel("selected WEE")
    ax.set_title("(a) selected WEE vs requirement")

    ax = axes[1]
    ax.plot(fr[m], tcert[m], color=C_PURPLE, lw=1.6, marker="o", ms=4)
    ax.set_xlabel(r"$\varepsilon_{\min}/\varepsilon_{\max}$")
    ax.set_ylabel(r"$T_{\rm cert}$ (s)")
    ax.set_title(f"(b) certified reuse horizon ($\\nu={nu*1e3:.1f}$e-3/s)")

    ax = axes[2]
    ax.step(fr, rem, where="post", color=C_GREEN, lw=1.6)
    ax.plot(fr, rem, "o", color=C_GREEN, ms=4)
    ax.set_xlabel(r"$\varepsilon_{\min}/\varepsilon_{\max}$")
    ax.set_ylabel("candidates remaining")
    ax.set_ylim(bottom=0)
    ax.set_title("(c) pool size after filtering")

    fig.tight_layout()
    path = save_fig(fig, out_name, subdir=subdir)
    plt.close(fig)
    return path


def plot_generalization(rows, epsilon_design, fracs, n_requested):
    """Paired raw observations and median/IQR curves; retain every outlier."""
    valid = [r for r in rows if r.get("seed_status") == "valid"]
    fixed = [r for r in valid if r["threshold_type"] != "normalized"]
    policies = ("wee_only", "proposed", "robustness_only")
    labels = ("WEE-only", f"Proposed @ {epsilon_design:g}", "Robustness-only")
    colors = (C_ORANGE, C_RED, C_GREEN)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, field, ylabel in zip(axes, ("selected_wee", "selected_epsilon_cert"),
                                ("WEE (bit/s/Hz/W)", r"$\varepsilon_{\rm cert}$")):
        for seed in sorted({r["channel_seed"] for r in fixed}):
            group = {r["policy"]: r for r in fixed if r["channel_seed"] == seed}
            y = [group[p][field] if group[p][field] is not None else np.nan for p in policies]
            ax.plot(range(3), y, color=C_GREY, alpha=0.35, lw=0.8, zorder=1)
        for i, (policy, color) in enumerate(zip(policies, colors)):
            y = [r[field] for r in fixed if r["policy"] == policy and r[field] is not None]
            ax.scatter([i]*len(y), y, s=22, color=color, alpha=0.8, zorder=2)
        ax.set_xticks(range(3), labels)
        ax.set_ylabel(ylabel)
    axes[1].axhline(epsilon_design, color=C_BLACK, ls=":", lw=1, label="design requirement")
    axes[1].legend(fontsize=8)
    n_valid = len({r["channel_seed"] for r in fixed})
    n_prop = sum(r["policy"] == "proposed" and r["selected_wee"] is not None for r in fixed)
    fig.suptitle(f"Paired selection: {n_valid}/{n_requested} valid seeds; proposed feasible {n_prop}/{n_valid}")
    fig.tight_layout()
    first = save_fig(fig, "exp3_generalization_policy_comparison")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for ax, field, ylabel, color in zip(axes, ("wee_retention_ratio", "selected_epsilon_ratio"),
                                      ("WEE retention", r"Selected $\varepsilon_{\rm cert}/\varepsilon_{\max}$"),
                                      (C_BLUE, C_GREEN)):
        quantiles = []
        counts = []
        for frac in fracs:
            y = [r[field] for r in valid if r["threshold_type"] == "normalized" and
                 r["eps_min_frac"] == frac and r[field] is not None and np.isfinite(r[field])]
            counts.append(len(y))
            quantiles.append(np.percentile(y, [25, 50, 75]) if y else [np.nan]*3)
        q = np.asarray(quantiles)
        ax.plot(fracs, q[:, 1], "o-", color=color, lw=1.6, ms=4, label="median")
        ax.fill_between(fracs, q[:, 0], q[:, 2], color=color, alpha=0.2, label="25%-75% quantiles")
        ax.set_xticks(fracs, [f"{f:g}" for f in fracs], rotation=40, ha="right", fontsize=9)
        ax.set_xlabel(r"$\varepsilon_{\min}/\varepsilon_{\max}$")
        ax.set_ylabel(ylabel)
        ax.set_title("Valid ratios per threshold: " + ", ".join(map(str, counts)), fontsize=9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    second = save_fig(fig, "exp3_threshold_sensitivity")
    plt.close(fig)
    return first, second
