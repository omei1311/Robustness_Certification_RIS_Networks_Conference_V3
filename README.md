# Configuration-Specific Robustness Certification for Stability-Aware Selection of Discrete RIS Configurations

**V3 会议论文数值实验框架（论文提交版）**

> 定位：在已生成的 QoS 可行 RIS 配置（beamforming + 离散相位）之上，增加**配置级鲁棒性认证**与**稳定性感知选择**层。本框架**不是**鲁棒 WEE 优化器；第一篇论文的优化器不在此重新实现——其信道/SINR/WEE/相位/S-procedure LMI 模块以同源精简方式复用（`src/ris_base/`）。

---

## 1. 核心定义与术语（与论文一致）

### 不确定集与证书

每个用户 (l,k) 的等效堆叠信道 ĥ_lk(X)（BS0..BS_{L-1} 拼接，L·M 维，Θ 的函数）服从相对不确定球：

```
r_lk(X, eps) = eps * max(||h_hat_lk(X)||, radius_floor)
```

对固定配置 **X = (W, Θ)**，可行性指示与证书为：

```
F_X(eps) = 1   若 X 在所有用户的相对不确定球下满足全部鲁棒 QoS 约束
epsilon_cert(X) = sup { eps >= 0 : F_X(eps) = 1 }
```

**epsilon_cert 是无量纲的相对不确定缩放因子（certified dimensionless relative uncertainty scaling factor），不是绝对信道半径。**

证书状态（`CertificateResult.status`）：

| status | 含义 |
|---|---|
| `EXACT_BRACKET` | 正常找到不可行上界；bracket=(eps_lo, eps_hi) 夹住边界，返回保守下端点 eps_lo |
| `LOWER_BOUND_CENSORED` | 扩张到 eps_hi_max 仍可行；返回值仅为下界（ε_cert ≥ eps_hi_max），非精确证书 |
| `NOMINAL_INFEASIBLE` | F_X(0)=0；ε_cert = 0 |

### 设计半径 vs 证书（必须区分）

- **ε_design = 0.05**：设计阶段预设的相对不确定因子（prescribed design factor）
- **ε_cert**：对固定配置的**后验**配置级证书（post-optimization, configuration-level）

两者分别报告（Exp2 输出 `candidates below epsilon_design`）。

### WEE

```
WEE(X) = sum_{l,k} log2(1 + SINR_lk) / P_tot(X)     [bit/s/Hz/W]
```

无权重和速率 / 总功率（`wee_unweighted_from_H`）。**bandwidth 不参与计算**：B_w 对所有候选相同，不改变候选排序与选择，因此论文统一使用频谱效率单位。旧加权路径（`user_weights`/`omega_eta`）仅作 legacy 兼容，不在任何主流程使用。

### T_cert（条件推论）

`T_cert = epsilon_cert / nu`，仅在**假定**线性相对漂移模型 ρ(t)=νt、ν=2×10⁻³/s 下可解释，是 certified reuse horizon，**不是**实际 QoS 失效时刻的预测。

### 选择规则（论文链条）

```
Candidate Pool -> epsilon_cert -> Dominance/Pareto Filtering
-> Robustness Threshold -> max WEE
```

- WEE-only / robustness-only：全池参考基线
- **proposed（Pareto-first）**：`pareto_candidates = pool[pareto_mask]` → `eligible = pareto_candidates[eps_cert >= eps_min]` → `selected = argmax WEE(eligible)`，统计输出 `n_total / n_pareto / n_after_rmin`

---

## 2. 环境

```powershell
uv venv --python 3.10 .venv                                   # 已建好
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## 3. 运行

```powershell
.venv\Scripts\python.exe tests\run_tests.py                    # 10 项测试
.venv\Scripts\python.exe experiments\run_all.py                # 主实验（缓存池）
.venv\Scripts\python.exe experiments\run_all.py --rebuild      # 强制重建池
.venv\Scripts\python.exe experiments\exp3_generalization_check.py   # 独立 holdout（~4 min，不进 run_all）
```

exp2 是建池/认证入口（写入 `results/pool_cache.*`），exp1/exp3 从缓存加载；缓存由全参数 sha256 指纹门禁（`config_fingerprint`），任何参数不一致自动重建。

---

## 4. 代码结构

```
src/ris_base/        复用层（第一篇论文 V6.5 同源精简；robustness.py 的 S-procedure/LMI/optimize_lambda 数学原样保留）
  config.py          SimConfig 系统参数（L=2,K=3,M=12,N=32,B=2,gamma=2）
  channels.py        信道生成 + effective_channels(θ)
  models.py          Q_B 离散相位、SINR、不加权 WEE（+legacy 加权兼容）、zf_full/MRT/RZF 方向、功率模型
  uncertainty.py     相对半径约定 r=eps*max(||h||,floor) + 复球采样
  robustness.py      Eq.(7)/(9) S-procedure LMI + 状态可行性 oracle（冻结，勿改）
  evaluation.py      误差样本下的 MC SINR

src/                 认证层
  config_v3.py       CertConfig 实验参数 + config_fingerprint（sha256 全参数指纹）
  certificate.py     epsilon_cert_bisection（+r_cert 兼容别名）、status 三态、
                     鲁棒性剖面、确定性最坏情形 SINR 曲线
  candidate_pool.py  候选池：configuration_signature(X=(W,Θ))、生成协议、缓存
  pareto.py          支配过滤 / Pareto 前沿
  selection.py       Pareto-first 选择规则 + 敏感性扫描 + T_cert
  robust_oracle.py   cvxpy SDP 交叉验证（测试用）
  plotting.py io_utils.py uncertainty_helpers.py

experiments/         shared(建池+指纹缓存) exp1 exp2 exp3 run_all
                     exp3_generalization_check.py（独立 holdout）
tests/run_tests.py   10 项测试
results/ figures/    JSON/CSV/NPZ + 300dpi PNG
```

论文公式 ↔ 代码映射见 `src/ris_base/__init__.py` 模块 docstring 与各函数 docstring（Eq.(1)–(23) 逐一标注）。

---

## 5. 候选池生成协议（与代码一致）

**固定**：L=2, K=3, M=12, N=32, **B=2（相位分辨率不是多样性旋钮）**、同一硬件功耗模型、同一信道实现（`channel_seed=20260706`）。

**波束方向**：`direction_choices=("zf_full",)` —— 集中式 ZF（本区零强迫 + 跨区泄漏置零）。共享 RIS 使两小区全强度耦合，MRT/RZF 方向族结构性干扰受限、功率控制发散，因此不在采样集合中（其实现保留于 `unit_directions` 作为 legacy 路径）。

**每个候选独立采样**（`rng` 由候选种子驱动）：

| 旋钮 | 网格 |
|---|---|
| `align_frac_grid` | 0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0 |
| `align_jitter_grid` | 0 / 0.15 / 0.3 / 0.6 / 1.0 rad |
| `design_gamma_mult` | 1.0 / 1.25 / 1.5 / 2.0 / 3.0 / 5.0 / 7.0（×γ̄） |
| `design_eps_grid` | 0 / 0.025 / 0.05 / 0.075 / 0.10 |
| `power_slack_grid` | 1.0 / 1.1 / 1.2 / 1.3 / 1.5 / 1.8 |
| `power_perturb_grid` | 0 / 0.05 / 0.10 / 0.20（相对逐用户扰动） |

流程：θ（对齐+抖动+随机混合）→ zf_full 方向 → **最坏情形（Cauchy–Schwarz）QoS 功率控制不动点**（在 ε_d 下保证鲁棒可行）→ slack 缩放 → 逐用户小扰动（**必须复检名义 QoS；ε_d>0 时复检鲁棒 QoS；失败回落 base**）→ 复用 QoS 检查与不加权 WEE。

**候选身份 = X=(W,Θ)**：`configuration_signature` = sha256( W 实/虚部按 1e-6 舍入 ‖ Θ 离散相位索引 )。池去重按完整签名；同一 Θ 不同 W 是两个配置。统计输出 `unique_theta_count` / `unique_configuration_count`。多样性目标是自然产生多个可行操作点，**不人为制造 WEE–ε_cert 权衡、不删改被支配点**。

---

## 6. 三个实验

- **Exp1 证书验证**：(a) α=ε/ε_cert ∈ {0.90,0.95,0.99,1.00,1.01,1.05,1.10} 的**确定性** LMI 一致性表（feasible/margin/最坏情形 SINR；α<1 可行、α=1 保守下端点可行、α>1 出现首个不可行网格点，容差内允许 1.01 附近边界误差）；(b) 0–3× 的 MC sweep —— **仅作经验验证，绝不用于估计 ε_cert**。输出措辞使用 "first infeasible grid point" / "boundary consistent within numerical tolerance"。
- **Exp2 景观**：40 候选 (WEE, ε_cert) 平面、Pareto 前沿、支配过滤、三选择点、ε_design 参考线、Spearman 相关。
- **Exp3 选择敏感性**：三规则（同一池）+ ε_min ∈ {0,0.25,0.50,0.75,0.90,0.95}×ε_max 扫描，逐档输出 selected_id / WEE / ε_cert / n_total / n_pareto / n_after_rmin + ε_design 独立样本检查。

**Exp3 泛化检查**（`exp3_generalization_check.py`，独立 holdout，非第四主实验）：12 个独立信道种子（60001+i），每种子完整执行建池→认证→Pareto→两策略选择，统计 pool_size/n_pareto/选中 WEE/选中与 WEE-only 的 ε_cert 的 mean/median/std 与 selection_changed_rate。**逐种子 Pareto 前沿绝不合并成一张图**；输出 `results/exp3_generalization_records.csv` + `exp3_generalization_summary.json`。

---

## 7. 参考结果（单信道，channel_seed=20260706）

**Pool**：40 accepted / 125 attempts；unique X = unique Θ = 40（逐用户扰动使全部 W 唯一）。

| 指标 | 数值 |
|---|---|
| WEE range | **[0.631933, 1.279232]** bit/s/Hz/W |
| epsilon_cert range | **[0.000952, 0.113965]** |
| nondominated | **2 / 40** |
| Spearman ρ | **0.85591**（p ≈ 1.95e-12） |
| below ε_design=0.05 | **10** |

**选择**：

| 规则 | idx | WEE | ε_cert |
|---|---|---|---|
| WEE-only | 34 | 1.279232 | 0.090894 |
| robustness-only | 6 | 1.246020 | 0.113965 |
| stability-aware @ 0.90·ε_max | 6 | 1.246020 | 0.113965（T_cert = 56.98 s） |

**Exp3 敏感性**：0 / 0.25 / 0.50 / 0.75·ε_max → idx 34；0.90 / 0.95·ε_max → idx 6。

**Exp1 一致性**：α≤1.00 全部 feasible（α=1.00 处 margin ≈ −1.9e-4，在 feasibility_tol=2e-4 内，保守下端点语义）；首个不可行网格点在 α=1.01–1.05；MC 首个采样违例在 α≈1.9–2.1（经验验证）。

**说明**：当前参考结果来自**单一信道实现**；多种子泛化结论以独立的 holdout 检查为准（见 §6）。证书认证开销：40 配置 ≈ 600 次 oracle 调用 ≈ 18 s（复杂度对应论文 Eq.(21)–(23)）。

---

## 8. 复现性

- 种子：信道 `channel_seed=20260706`（`nominal_drop(cfg, seed)`，与 `cfg.seed` 解耦）、池 `pool_seed=31001`、MC `exp1_seed=41001` / `exp3_seed=43001`、holdout `gen_check_seed_base=60001`
- 池缓存指纹门禁：任何 SimConfig/CertConfig 参数变化 → 自动重建；候选落盘 `channel_seed / config_fingerprint / configuration_signature / cert_info{status,...}`，可脱离代码审计来源
- 不修改 `src/ris_base/robustness.py` 的 S-procedure / LMI / `optimize_lambda` 数学实现（冻结）

## 9. 与第一篇论文项目的关系

`src/ris_base/` 由 `C:\Users\57756\Desktop\ris_robust_simulation`（`ris_sim` V6.5）精简移植：公式、相对半径约定、种子口径一致；仅去除 torch/yaml/求解器预算字段。候选生成刻意轻量（zf_full + 最坏情形功率控制），**不调用**其优化器；认证层只把候选当固定 X 处理，与第一篇论文的可行性判定同一 LMI 口径（cvxpy SDP 交叉验证见测试）。
