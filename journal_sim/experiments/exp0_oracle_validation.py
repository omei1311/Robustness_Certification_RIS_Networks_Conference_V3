"""Exp0/Fig.2: journal direct-intercell-ON validation; legacy cases are tests only."""
from dataclasses import replace
import numpy as np
from journal_sim.core.channels import generate_channel
from journal_sim.dynamics.csi_estimation import estimate_physical
from journal_sim.design.candidate_pool import build_candidate_pool
from journal_sim.certification.certificate import certificate_bisection, NUMERICALLY_UNCERTAIN
from journal_sim.certification.lmi import make_lmi
from journal_sim.certification.oracle import FastOracle, StrictOracle, STRICT_FEASIBLE
from journal_sim.evaluation.monte_carlo import sample_unit_ball, counterexample_check
from .common import execute, cli_config, pyplot, save_plot


def choose_representatives(pool):
    ordered = sorted((c for c in pool if c.valid_certificate), key=lambda c: (c.epsilon_cert, c.index))
    if len(ordered) < 3:
        return []
    return list(zip(("journal_fragile", "journal_medium", "journal_robust"),
                    (ordered[0], ordered[len(ordered) // 2], ordered[-1])))


def check_case(label, cfg, w, H, theta, cert, seed, alphas, rows, arrays, details):
    before, radius = cert.to_dict(), cert.epsilon_cert
    dirs = sample_unit_ball(cfg.mc_samples, cfg, cfg.mc_seed + seed)
    failures = 0
    for alpha in alphas:
        eps = alpha * radius
        strict = StrictOracle(cfg).check(w, H, eps)
        fast = FastOracle(cfg).check(w, H, eps)
        checked, mc = counterexample_check(cert, w, H, eps, dirs, cfg)
        if cert.strict_validation_passed and not checked.strict_validation_passed:
            failures += 1
        if alpha <= 1 and before["strict_validation_passed"] and not all(s.status == STRICT_FEASIBLE for s in strict):
            failures += 1
            checked = replace(checked, status=NUMERICALLY_UNCERTAIN, strict_validation_passed=False,
                              note="repeated independent strict check failed; original endpoint retained")
        cert = checked
        for f, s in zip(fast, strict):
            dense = None
            if label == "tiny_fragile":
                p = make_lmi(w, H, eps, s.user_index, cfg)
                hi = min(cfg.fast_lambda_cap, max(100., 2 * f.lambda_star * p.h_scale ** 2 / p.scale))
                grid = np.r_[0., np.geomspace(1e-8, hi, cfg.lambda_grid_points)]
                dense = max(float(np.linalg.eigvalsh(p.balanced(mu))[0]) for mu in grid)
            rows.append(dict(row_type="validation", label=label, include_direct_intercell=cfg.include_direct_intercell,
                             epsilon_cert=radius, tested_radius=eps, alpha=alpha, user_index=s.user_index,
                             strict_status=s.status, solver_status=s.solver_status, lambda_value=s.lambda_value,
                             raw_min_eig=s.raw_min_eig, normalized_min_eig=s.normalized_min_eig,
                             fast_feasible=f.feasible_fast, fast_margin=f.normalized_min_eigenvalue,
                             dense_grid_margin=dense, mc_seed=cfg.mc_seed + seed,
                             analytic_worst_sinr=(abs(w.item()) ** 2 * max(0, 1 - eps) ** 2 / cfg.noise_power) if label == "tiny_fragile" else None,
                             **mc))
    details.append(dict(label=label, local_config=cfg.to_dict(), certificate_before_evaluation=before,
                        certificate=cert.to_dict(), source="analytic scalar" if label == "tiny_fragile" else "journal estimated-CSI pool, direct intercell ON"))
    for key, value in (("w", w), ("H_hat", H), ("theta", theta), ("ball_directions", dirs)):
        arrays[label + "_" + key] = value
    return failures


def run_seed(cfg, seed):
    if not cfg.include_direct_intercell:
        raise ValueError("Main Exp0/Fig.2 requires direct intercell ON; archived OFF cases are regression tests only")
    rows, details, arrays = [], [], {}
    tiny = cfg.with_overrides(L=1, K=1, M=1, N=2, bits=1, gamma=1., channel_scale=1.,
                             noise_power_dbm=30., p_max_dbm=40., bs_x=(-100.,), bs_y=(0.,),
                             ue_center_x=(15.,), ue_center_y=(0.,))
    w, H, theta = np.full((1, 1, 1), 1 / (1 - .0005), complex), np.ones((1, 1, 1, 1), complex), np.ones(2, complex)
    failures = check_case("tiny_fragile", tiny, w, H, theta, certificate_bisection(w, H, tiny, theta),
                          seed, cfg.alpha_grid, rows, arrays, details)
    local = cfg.with_overrides(pool_size=cfg.validation_pool_size)
    true = generate_channel(local, seed)
    csi_seed = int(np.random.SeedSequence([cfg.csi_seed, seed, 0]).generate_state(1)[0])
    estimated = estimate_physical(true, local, csi_seed)
    pool, stats = build_candidate_pool(estimated, local, seed)
    for c in pool:
        rows.append(dict(row_type="candidate", **c.to_record()))
        for key, value in (("w", c.configuration.w), ("theta", c.configuration.theta), ("H_hat", c.H_hat)):
            arrays[f"pool_c{c.index}_{key}"] = value
    rows.extend(dict(row_type="attempt", **a) for a in stats["attempts"])
    details.append(dict(label="journal_pool", stats=stats, csi_noise_seed=csi_seed, local_config=local.to_dict()))
    for kind, channel in (("true", true), ("estimated", estimated)):
        arrays.update({kind + "_" + link: getattr(channel, link) for link in ("h_bu", "h_br", "h_ru")})
    reps = choose_representatives(pool)
    for label, c in reps:
        failures += check_case(label, local, c.configuration.w, c.H_hat, c.configuration.theta,
                               c.certificate, seed, cfg.validation_alpha_grid, rows, arrays, details)
    status = "VALIDATION_FAILED" if failures else "COMPLETED" if len(reps) == 3 else "INSUFFICIENT_VALIDATED_REPRESENTATIVES"
    return dict(records=rows, details=details, arrays=arrays,
                summary=dict(counterexamples_or_strict_recheck_failures=failures,
                             representative_indices={label: c.index for label, c in reps}), status=status)


def plot(rows, details, output, cfg):
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for label, color, marker in (("journal_fragile", "#0072B2", "o"), ("journal_medium", "#E69F00", "s"), ("journal_robust", "#CC79A7", "^")):
        by_alpha = {}
        for r in rows:
            if r.get("row_type") == "validation" and r["label"] == label:
                by_alpha.setdefault(r["alpha"], []).append(r)
        x = sorted(by_alpha)
        if x:
            axes[0].plot(x, [min(r["fast_margin"] for r in by_alpha[a]) for a in x], marker=marker, color=color, label=label)
            axes[1].plot(x, [min((r["normalized_min_eig"] for r in by_alpha[a] if r["normalized_min_eig"] is not None), default=np.nan) for a in x], marker=marker, color=color, label=label)
            axes[2].plot(x, [100 * np.mean([r["violation_rate"] for r in by_alpha[a]]) for a in x], marker=marker, color=color, label=label)
    for ax in axes[:2]:
        ax.axhline(0, color="#333333", lw=1)
        ax.set_ylabel("Balanced minimum eigenvalue")
    axes[0].set_title("Fast search margin")
    axes[1].set_title("Independent strict SDP residual")
    axes[2].set_title(f"Sampled QoS violations (n={cfg.mc_samples}/seed)")
    axes[2].set_ylabel("QoS violation (%)")
    axes[2].set_ylim(-2, 102)
    for ax in axes:
        ax.set_xlabel("Radius / validated lower bound")
    axes[0].legend(fontsize=8)
    fig.suptitle("Journal model certificate validation: direct inter-cell ON")
    save_plot(fig, output)
    plt.close(fig)


if __name__ == "__main__":
    _, success = execute("exp0_oracle", cli_config(__doc__), run_seed, plot)
    raise SystemExit(0 if success else 1)
