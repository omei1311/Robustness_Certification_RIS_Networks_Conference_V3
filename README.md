# Configuration-Specific Robustness Certification for Stability-Aware Selection of Discrete RIS Configurations

Numerical experiment framework for the V3 conference paper.  The framework is
**not** another robust WEE optimizer: it attaches a configuration-level
*certification and selection layer* on top of an already-generated pool of
QoS-feasible RIS/beamforming configurations (paper Eq. (20)):

```
Candidate pool -> statewise robust feasibility -> R_cert
              -> WEE-robustness Pareto/dominance filtering
              -> stability-aware selection -> (conditional) T_cert
```

## Environment

```powershell
# already created in this project: .venv (Python 3.10)
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Running

```powershell
# unit / validation tests (LMI eig-vs-cvxpy cross-check, monotonicity, ...)
.venv\Scripts\python.exe tests\run_tests.py

# all three experiments (pool built once and cached in results/)
.venv\Scripts\python.exe experiments\run_all.py            # uses cache
.venv\Scripts\python.exe experiments\run_all.py --rebuild  # fresh pool

# individually
.venv\Scripts\python.exe experiments\exp2_wee_robustness_landscape.py
.venv\Scripts\python.exe experiments\exp1_certificate_validation.py
.venv\Scripts\python.exe experiments\exp3_selection_sensitivity.py
```

Outputs: `results/*.json`, `results/*.csv`, `figures/*.png`.

## Code layout

### Reuse layer (`src/ris_base/`, adapted from the first paper's simulator)

Trimmed copies of the robust-WEE design project (`ris_robust_simulation`,
V6.5): identical formulas and conventions, torch/yaml/solver dependencies
removed.  Paper mapping:

| module | paper element |
|---|---|
| `config.py` (`SimConfig`) | Section 2 system parameters (L=2, K=3, M=12, N=32, 2-bit, gamma=2) |
| `channels.py` | channel generation; `effective_channels` = h in Eq. (2) |
| `models.py` | Eq. (1) discrete phases `Q_B`; Eq. (2) SINR / QoS check; Eq. (5) WEE & power model; MRT/RZF initializers |
| `uncertainty.py` | radius convention (`relative_radius`: r = eps*||h_lk||) + ball sampling |
| `robustness.py` | Eq. (7) quadratic form `A_lk(X)`, Eq. (9) S-procedure LMI `M_lk`, statewise feasibility oracle (`robust_check`) |
| `evaluation.py` | Monte Carlo SINR under sampled channel errors |

### New certification layer (`src/`)

| module | paper element |
|---|---|
| `config_v3.py` (`CertConfig`) | pool/certification/experiment settings (Section 9) |
| `certificate.py` | **R_cert by bisection** (Eq. (10)-(13)), robustness profile, deterministic worst-case SINR curve |
| `robust_oracle.py` | cvxpy SDP realization of Eq. (9) for cross-validation |
| `candidate_pool.py` | candidate-pool generation protocol (Section 7 stage (i)): jittered aligned/random quantized phases + MRT/RZF/ZF directions + worst-case (Cauchy-Schwarz) QoS power control at design radius |
| `pareto.py` | dominance filtering, WEE-robustness Pareto frontier (Section 6) |
| `selection.py` | WEE-only / robustness-only / stability-aware rules (Eq. (17)-(18)), R_min sweep, T_cert = R_cert/nu (Eq. (16)) |
| `plotting.py`, `io_utils.py` | figures and result persistence |

### Experiments (Section 9)

| script | experiment |
|---|---|
| `exp1_certificate_validation.py` | Exp 1: radius sweep 0-3 R_cert, sampled worst-user SINR + violation rate + deterministic worst-case curve, R_cert markers |
| `exp2_wee_robustness_landscape.py` | Exp 2: ~40-configuration pool, (WEE, R_cert) map, Pareto set, selection markers, Spearman alignment |
| `exp3_selection_sensitivity.py` | Exp 3: three selection rules, R_min in {0,...,0.95}R_max, selected WEE/R_cert/T_cert/remaining count, independent-sample check at the design radius |

## Design decisions worth knowing

- **Radius convention.** Following the first paper, the per-user
  uncertainty ball is `eps * ||h_lk||` (relative radius).  The prescribed
  design radius `epsilon = 0.05` and every certificate are in these units.
- **Oracle.** `F_X(eps)` is decided by the first paper's eigendecomposition
  multiplier search over the Eq. (9) LMI; `robust_oracle.py` re-solves the
  same LMI with cvxpy and the tests assert both agree at 0.7 R_cert
  (feasible) and 1.4 R_cert (infeasible).
- **Candidate generation is decoupled from certification** (Section 7).
  It uses only reused initializers plus a worst-case power-control fixed
  point, *not* the first paper's WEE optimizer.  Diversity knobs:
  alignment fraction/jitter, B-bit resolution, design SINR target,
  nominal-vs-robust design radius, and a non-minimal-power slack factor.
- **`zf_full` beamforming.** With the shared RIS coupling both cells at full
  strength, per-cell MRT/RZF are structurally interference-limited; the pool
  protocol therefore uses centralized ZF that also nulls inter-cell leakage
  (M=12 >= L*K-1).  This is a *generation* choice only; certification treats
  each candidate as a fixed X.
- **T_cert is conditional** (Section 5): reported under rho(t) = nu t with
  nu = 2e-3 relative-radius/s; it is a certified reuse horizon, not a
  predicted failure time.

## Reference results (this seed)

- Pool: 40 accepted / 105 attempts; WEE in [0.105, 0.228];
  R_cert in [0.0011, 0.1285]; 8 candidates below the prescribed
  epsilon = 0.05; Spearman(WEE, R_cert) ~ 0.92 (strongly aligned pool,
  reported honestly per Section 9; dominance filtering still removes 38/40).
- Exp 1: deterministic worst-case SINR curve crosses the QoS target at
  1.00-1.13 x R_cert; sampled violations start near 2 x R_cert
  (certificate conservative, as stated in Section 4).
- Exp 3: stability-aware selection keeps the WEE-optimal configuration up to
  R_min = 0.90 R_max and switches at 0.95 R_max (WEE -7.5%, T_cert +4 s);
  selected configurations hold QoS on 100% of independent samples at
  epsilon = 0.05.
