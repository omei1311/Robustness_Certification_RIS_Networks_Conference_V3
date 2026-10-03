"""Exp0: analytic tiny ball, dense lambda grid and real conference regression."""
from pathlib import Path
import json
import numpy as np
from journal_sim.core.models import Configuration
from journal_sim.config import JournalConfig
from journal_sim.certification.certificate import certificate_bisection, CertificateResult, VALIDATED
from journal_sim.certification.lmi import make_lmi
from journal_sim.certification.oracle import FastOracle, StrictOracle, STRICT_FEASIBLE
from journal_sim.evaluation.monte_carlo import sample_unit_ball, counterexample_check
from .common import execute, cli_config, pyplot, save_plot, ROOT


def run_seed(cfg, seed):
    rows, details, arrays = [], [], {}
    # Analytic scalar-channel tiny system: exact radius = 0.0005.
    tiny = cfg.with_overrides(L=1, K=1, M=1, N=2, bits=1, gamma=1., channel_scale=1.,
                             noise_power_dbm=30., p_max_dbm=40., bs_x=(-100.,), bs_y=(0.,),
                             ue_center_x=(15.,), ue_center_y=(0.,))
    cases = [("tiny_fragile", tiny, np.full((1, 1, 1), 1 / (1 - .0005), complex),
              np.ones((1, 1, 1, 1), complex), np.ones(2, complex), .0005)]
    fixture = Path(cfg.legacy_fixture)
    if not fixture.is_absolute():
        fixture = ROOT / fixture
    metadata = json.loads(fixture.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(fixture, allow_pickle=False) as data:
        for index in cfg.legacy_indices:
            i = list(data["indices"]).index(index)
            legacy_cfg = cfg.with_overrides(L=2, K=3, M=12, N=32, bits=2, gamma=2.,
                                            channel_scale=1e5, noise_power_dbm=-100., include_direct_intercell=False,
                                            bs_x=(-100., 100.), bs_y=(0., 0.),
                                            ue_center_x=(-15., 15.), ue_center_y=(0., 0.))
            old_eps = next(r["epsilon_cert"] for r in metadata["candidates"] if r["index"] == index)
            cases.append((f"conference_{index}", legacy_cfg, data["w"][i].copy(), data["H"][i].copy(), data["theta"][i].copy(), old_eps))
    failures = 0
    for label, local, w, H, theta, old_eps in cases:
        cert = certificate_bisection(w, H, local, theta)
        reported_before_eval = cert.to_dict()
        directions = sample_unit_ball(local.mc_samples, local, local.mc_seed + seed)
        # Archived zero-margin case is evaluated against its OLD claim too.
        tested_eps = cert.epsilon_cert if cert.epsilon_cert > 0 else old_eps
        if label == "conference_26":
            alphas = (.125,) + local.alpha_grid
        else:
            alphas = local.alpha_grid
        for alpha in alphas:
            eps = alpha * tested_eps
            fast, strict = FastOracle(local), StrictOracle(local)
            strict_results = strict.check(w, H, eps)
            fast_results = fast.check(w, H, eps)
            checked, mc = counterexample_check(cert, w, H, eps, directions, local)
            if cert.strict_validation_passed and not checked.strict_validation_passed:
                failures += 1
            cert = checked
            for f, s in zip(fast_results, strict_results):
                p = make_lmi(w, H, eps, s.user_index, local)
                grid = np.geomspace(1e-8, min(local.fast_lambda_cap, max(100., 2 * f.lambda_star * p.h_scale ** 2 / p.scale)), local.lambda_grid_points)
                dense_margin = max(float(np.linalg.eigvalsh(p.balanced(mu))[0]) for mu in np.r_[0., grid])
                rows.append(dict(label=label, candidate=26 if label == "conference_26" else label,
                                 epsilon_cert=cert.epsilon_cert, old_epsilon_claim=old_eps, tested_radius=eps,
                                 alpha=alpha, radius_basis="new_validated_lower" if tested_eps == cert.epsilon_cert else "old_unvalidated_claim",
                                 user_index=s.user_index, strict_status=s.status, solver_status=s.solver_status,
                                 lambda_value=s.lambda_value, raw_min_eig=s.raw_min_eig,
                                 normalized_min_eig=s.normalized_min_eig, fast_feasible=f.feasible_fast,
                                 mc_seed=local.mc_seed + seed,
                                 fast_margin=f.normalized_min_eigenvalue, dense_grid_margin=dense_margin,
                                 analytic_worst_sinr=(abs(w.item()) ** 2 * max(0, 1 - eps) ** 2 / local.noise_power) if label == "tiny_fragile" else None,
                                 **mc))
        details.append(dict(label=label, local_config=local.to_dict(), certificate_before_evaluation=reported_before_eval,
                            certificate=cert.to_dict(), source="analytic scalar" if label == "tiny_fragile" else metadata))
        arrays[label + "_w"], arrays[label + "_H_hat"], arrays[label + "_theta"] = w, H, theta
        arrays[label + "_ball_directions"] = directions
    return dict(records=rows, details=details, arrays=arrays, summary=dict(counterexamples_inside_new_certificate=failures),
                status="VALIDATION_FAILED" if failures else "COMPLETED")


def plot(rows, details, output, cfg):
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ("#0072B2", "#E69F00", "#CC79A7", "#6B6B6B")
    labels = list(dict.fromkeys(r["label"] for r in rows if "label" in r))
    for label, color in zip(labels, colors):
        by_alpha = {}
        for r in rows:
            if r.get("label") == label:
                by_alpha.setdefault(r["alpha"], []).append(r)
        x = sorted(by_alpha)
        axes[0].plot(x, [min(r["fast_margin"] for r in by_alpha[a]) for a in x], "o-", color=color, label=label)
        axes[1].plot(x, [100 * by_alpha[a][0]["violation_rate"] for a in x], "o-", color=color, label=label)
    axes[0].axhline(0, color="#333333", lw=1)
    axes[0].set_ylabel("Minimum balanced LMI margin")
    axes[1].set_ylabel("Sampled QoS violation (%)")
    axes[1].set_ylim(-2, 102)
    for ax in axes:
        ax.set_xlabel("Radius / tested certificate claim")
        ax.legend(fontsize=8)
    axes[0].set_title("Fast / strict crosscheck (see raw CSV)")
    axes[1].set_title(f"Monte Carlo counterexamples (n={cfg.mc_samples}/seed)")
    save_plot(fig, output)
    plt.close(fig)


if __name__ == "__main__":
    _, success = execute("exp0_oracle", cli_config(__doc__), run_seed, plot)
    raise SystemExit(0 if success else 1)
