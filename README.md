# Configuration-Specific Robustness Certification for Stability-Aware Selection of Discrete RIS Configurations

**V3 会议论文数值实验框架（论文提交版）**

---

## 0. journal_sim（IET Communications 期刊修订框架）

`journal_sim/` 是面向期刊稿（Validated Robustness Certification and Event-Triggered
Reconfiguration）的独立仿真框架：离线 CSI 校准 → 候选生成 → 证书（fast oracle 提议 +
CLARABEL 主求解器 / SCS 回退 + 独立 eigvalsh 验证）→ lifetime-aware 选择 → 证书触发
重配置 → 真信道评估 → CSV/manifest/arrays。

### 关键一致性约定

- **选择与触发同一套账目**：触发器在 `rho_total = eps_est*(1+rho)+rho >= eta*eps_cert`
  时重设计；`certified_lifetime = (eta*eps_cert - eps_est)/((1+eps_est)*nu)` 由同一公式
  反解，`eps_est >= eta*eps_cert` 的候选 predicted_lifetime=0、不可入选（不放宽）。
- **证书永远保守**：`NUMERICALLY_UNCERTAIN` 不当作不可行证明；接受端点必须通过
  全用户、非 fail-fast 的独立 eigvalsh 复检；bounded recovery
  （`strict_recovery_factors` + `strict_refinement_steps`）取代旧的 40×0.98 收缩。
- **求解器回退可审计**：primary/fallback 调用数、solver error / optimal_inaccurate
  计数随 candidate 统计与 manifest 落盘。

### 两阶段离线校准（正式实验前置）

```text
Exp1  formal CSI calibration   (calibration_seeds 52001-52005, 2000 samples,
                                4 phase profiles) -> joint_radius.json
                                  -> epsilon_est = max over seeds of joint 99%
Exp1b formal drift calibration (drift_calibration_seeds 53001-53005,
                                60-slot trajectories, references every 10 slots,
                                horizons 1/2/5/10/20) -> drift_rate.json
                                  -> nu_slow / nu_medium / nu_fast (q=95%, 1/s)
```

- Exp1b 复用**在线管线本身**（真实 Gauss-Markov 信道 → 物理导频估计 → 估计等效信道），
  并直接调用 trigger 的 `relative_drift()`（不另立公式）；`nu` 只来自离线校准轨迹，
  与正式 evaluation seeds（60001–60010）、CSI 校准 seeds 完全隔离。
- 三种量**不可混淆**：`epsilon_cert` 是 configuration-specific 的 deterministic robust
  lower bound；`epsilon_est` 是有限样本 empirical CSI 误差半径；`nu_m` 是仅用于
  lifetime prediction 的 empirical drift proxy。`epsilon_est / nu / T_pred` 都不是
  deterministic guarantee。
- **在线 trigger 永远只用实际观测**：`rho_total = epsilon_est*(1+rho_obs)+rho_obs` 与
  `eta_trigger*epsilon_cert` 比较；drift calibration 不进入 trigger，也不改变证书定义。
- formal Exp3/Exp4 启动时由 `formal_calibration_guard` 防呆：必须绑定 **formal** 模式的
  CSI 与 drift artifact（SHA256 + scope fingerprint + 三 mobility 齐全），传 smoke
  artifact、缺 artifact、或 evaluation seeds 与校准 seeds 重叠都会**直接拒绝**；
  smoke 模式允许手工 `epsilon_est` / `drift_rate_nu` 便于调试。

### 正式实验顺序

```powershell
# 0. 工作区必须干净（正式结果要求 manifest 中 git.dirty=false）
git status --porcelain

# 1. 单元测试（spawn 并行依赖可导入的 __main__，请用 unittest 入口）
.venv\Scripts\python.exe -m unittest discover -s journal_sim\tests -t . -v

# 2. formal CSI calibration（--workers 只并行独立校准 seed）
.venv\Scripts\python.exe -m journal_sim.experiments.exp1_csi_calibration --workers 4

# 3. formal drift calibration
.venv\Scripts\python.exe -m journal_sim.experiments.exp1b_drift_calibration --workers 4

# 4. 证书边界验证（无需校准 artifact）
.venv\Scripts\python.exe -m journal_sim.experiments.exp0_oracle_validation --workers 1

# 5. 静态池 / 选择规则对比（CSI artifact 必需；drift artifact 软性要求）
.venv\Scripts\python.exe -m journal_sim.experiments.exp2_static_certificate --workers 4 --csi-calibration "<formal joint_radius.json>"

# 6. Exp3 formal（CSI + drift artifact 均为硬性要求）
.venv\Scripts\python.exe -m journal_sim.experiments.exp3_dynamic_reconfiguration --workers 4 --csi-calibration "<formal joint_radius.json>" --drift-calibration "<formal drift_rate.json>"

# 7. Exp4 runtime 基准（强制串行）
.venv\Scripts\python.exe -m journal_sim.experiments.exp4_runtime_scaling --workers 1 --csi-calibration "<formal joint_radius.json>" --drift-calibration "<formal drift_rate.json>"
```

Smoke 示例（模式由 `--smoke` 或 config 的 `smoke` 字段决定）：

```powershell
$csi = "journal_results\exp1_csi\smoke\<run>\joint_radius.json"
$drift = "journal_results\exp1b_drift\smoke\<run>\drift_rate.json"
.venv\Scripts\python.exe -m journal_sim.experiments.exp1_csi_calibration --smoke --workers 1
.venv\Scripts\python.exe -m journal_sim.experiments.exp1b_drift_calibration --smoke --workers 1
.venv\Scripts\python.exe -m journal_sim.experiments.exp3_dynamic_reconfiguration --smoke --workers 1 --config .\smoke_fast.json --csi-calibration "$csi" --drift-calibration "$drift"
.venv\Scripts\python.exe -m journal_sim.experiments.exp3_dynamic_reconfiguration --smoke --workers 1 --config .\smoke_trend.json --csi-calibration "$csi" --drift-calibration "$drift"
```

说明：

- `workers` 只改变执行并行度，**不改变随机种子、scientific config 或 fingerprint**
  （它位于 `ExecutionOptions`，不在 `JournalConfig` 内）；`workers=1` 走原始串行路径。
  每个 seed 的全部 mobility/policy/候选/SDP 在单个 worker 内顺序完成；结果严格按
  `cfg.seeds` 顺序汇总，worker 完成顺序不影响输出顺序。
- **并行 Exp3 的 runtime 字段仅供 debug**（manifest 中 `execution.runtime_measurement_valid=false`）；
  论文 runtime 数据一律来自串行 `Exp4 --workers 1`。
- 大型 arrays 由 worker 写入 `_worker_staging/seed_<seed>_arrays.npz`，parent 合并；
  失败/中断时保留 staging 以便排查；Ctrl+C 不被吞掉，已完成 seed 的记录保留，
  `run_complete=false`、`success` 不会伪造为 true。
- manifest 记录 `git`（commit/dirty/method）+ `source_hashes` + 校准 artifact SHA256；
  formal 运行若工作区 dirty 会打印 WARNING，正式论文结果要求 clean tree。

### 跨 policy 候选池缓存与 mobility 级 checkpoint

- **池缓存**：同一 `(seed, mobility, time_index, observation_id, config_fingerprint)`
  下的候选池（生成 + 认证）只执行一次，所有 policy 共享同一批只读候选；
  trigger、selection、切换成本仍完全 policy 独立。缓存生命周期 = 一次
  `run_paired_policies`（一个 seed × 一个 mobility），不跨 seed/mobility/运行。
  cache hit 不重复计费（solver 调用与 runtime 记 0，原始构建成本单独保留审计），
  summary/record 新增 `candidate_pool_{requests,builds,cache_hits,cache_hit_rate}`
  与 `candidate_pool_cache_{hit,key,build_runtime,original_runtime}` 字段。
- **mobility checkpoint**：exp3 的 seed worker 每完成一个 mobility 就把
  `slow/medium/fast_result.json` + `*_arrays.npz` 原子写入
  `_worker_staging/seed_<seed>/`；中断或失败后 staging 保留，manifest 的
  seed 条目记录 `completed_mobility / pending_mobility`。
  **当前不提供 `--resume`**：中断的 seed 需要整 seed 重跑（partial staging
  仅用于诊断与证据保留），恢复运行必须满足指纹/artifact 完全一致前不会引入。
- **`--mobility slow|medium|fast`** 显式指定时同时收窄 `mobility_regimes`，
  真正只跑单档；不指定则默认三档全跑。
- **workers 建议**（不要在代码中硬编码）：8C/16T → `--workers 4`；
  12C/24T → `--workers 4~6`；16C/32T → `--workers 6~8`。注意 CVXPY/SCS 的
  内存占用可能先于 CPU 成为瓶颈；正式运行期间不要在同一机器并行其他
  numpy 重载任务（会触发 OpenBLAS 内存分配失败或严重缺页变慢）。

---

## 1.（会议论文 V3 框架说明，以下为原内容）

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
.venv\Scripts\python.exe tests\run_tests.py                    # 17 项测试
.venv\Scripts\python.exe experiments\run_all.py                # 主实验（缓存池）
.venv\Scripts\python.exe experiments\run_all.py --rebuild      # 强制重建池
.venv\Scripts\python.exe experiments\exp3_generalization_check.py --n-seeds 2   # smoke test
.venv\Scripts\python.exe experiments\exp3_generalization_check.py --n-seeds 20  # 正式多信道实验
.venv\Scripts\python.exe experiments\exp3_generalization_check.py --n-seeds 20 --with-mc  # 可选独立 MC
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
  generalization.py  同池多策略 long-format 记录、边界检查与 paired 汇总
  robust_oracle.py   cvxpy SDP 交叉验证（测试用）
  plotting.py io_utils.py uncertainty_helpers.py

experiments/         shared(建池+指纹缓存) exp1 exp2 exp3 run_all
                     exp3_generalization_check.py（独立 holdout）
tests/run_tests.py   17 项测试（10 项原有 + 7 项多信道测试）
results/ figures/    JSON/CSV/NPZ + 300dpi PNG
```

论文公式 ↔ 代码映射见 `src/ris_base/__init__.py` 模块 docstring 与各函数 docstring（Eq.(1)–(23) 逐一标注）。

---

## 5. 候选池生成协议（与代码一致）

**固定**：L=2, K=3, M=12, N=32, **B=2（相位分辨率不是多样性旋钮）**、同一硬件功耗模型。单信道实验使用 `channel_seed=20260706`；多信道实验逐 seed 更换 realization，同 seed 的全部策略共享同一信道与 candidate pool。

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

实验层次：

```text
Exp1: Certificate validation
Exp2: WEE–robustness landscape and Pareto analysis
Exp3: Stability-aware selection
  ├─ single-channel illustrative comparison
  └─ multi-channel generalization check
```

**Exp3 多信道验证**（`exp3_generalization_check.py`，仍属于 Exp3）：正式默认使用 **20 independent channel realizations**，连续种子 **60001–60020**，由 `CertConfig.gen_check_seed_base/gen_check_n_seeds` 管理。每个 realization 仅生成一次目标为 40 的 candidate pool、认证一次，WEE-only、robustness-only 和 proposed 共享同一池。系统参数、B=2、候选生成协议完全相同；使用 `dataclasses.replace` 写入当前 channel seed。robustness-only 是 robustness extreme 辅助参考。

- **固定需求**：所有 seed 均使用 `epsilon_design=0.05`；无 eligible candidate 时记录 `proposed_feasible=False` 和空选择，不 fallback、不降低阈值。
- **normalized threshold sensitivity**：预先固定 `epsilon_min/epsilon_max = (0, 0.25, 0.50, 0.75, 0.90, 0.95)`。报告 WEE retention、绝对 epsilon_cert 增益和 selection change rate，不根据结果调阈值。
- 逐 seed paired comparison 报告 ΔWEE、Δepsilon_cert 及百分比；分母绝对值不大于 `1e-8` 时百分比缺失。主指标为 WEE 和 epsilon_cert；T_cert 仅为线性 relative drift 模型下的条件推论，不是真实 QoS failure time。
- Pareto filtering 是预处理；constrained max-WEE 的选择变化来自 certificate-aware stability constraint，不把性能变化归因于 Pareto filtering。
- 实际 pool 不足 40 时警告并保留；空池/异常 seed 保留失败记录，其余 seed 继续。每个 seed 后保存 CSV、JSON 和表格。summary 保存 attempts、accepted、去重统计、拒绝原因和全部 configuration signatures。失败运行最终返回非零退出码，但已完成记录保留。
- 缓存位于 `results/generalization_cache/seed_<seed>_<fingerprint>.*`，复用现有建池/认证接口。加载校验 channel seed、config fingerprint、每个配置签名和证书状态。`--rebuild` 可重新计算同一预定实验；不会搜索替代 seed。
- `--with-mc` 在固定 ε=0.05 下使用独立 RNG `90000+channel_seed`，默认 100 个复球样本，同 seed 的策略共享误差方向。记录 QoS hold/violation count/rate 与 `worst_margin=min(SINR)-gamma`；采样不替代证书，发现证书内采样违例会警告并原样记录。
- mean/median/std 使用有限值，std 为总体标准差（ddof=0）。feasibility rate 分母为有效 seed；selection change rate 分母为可行 paired seed；失败 seed 单列，同时报告以全部请求 seed 为分母的 feasible fraction。各指标报告有效样本数。censored certificate 保留为下界，normalized sweep 使用报告值的最大值；epsilon_max 为零时归一化比值缺失。

输出（每次运行更新，smoke test 不代表正式结果）：

```text
results/exp3_generalization_records.csv
results/exp3_generalization_summary.json
results/exp3_generalization_table.csv
results/exp3_generalization_table.md
figures/exp3_generalization_policy_comparison.png
figures/exp3_threshold_sensitivity.png
```

Figure A 分两个 panel 显示 WEE 与 epsilon_cert，同 seed 配对连线；不可行选择留空。Figure B 分两个 panel 显示 WEE retention 与 selected epsilon_cert/epsilon_max 的 median 和 25%–75% 分位区间，不删除 outlier。`run_all.py` 保持原有 Exp1/Exp2/Exp3 single-channel 行为，多信道实验需单独运行。正式结论由实际 20-seed summary 支持后填写，此处不预设 selection_changed_rate 或提升结论。

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

**Exp1 一致性**：α≤1.00 全部 feasible（α=1.00 处 margin ≈ −1.9e-4，在 feasibility_tol=2e-4 内，保守下端点语义）；首个不可行网格点在 α=1.01–1.05。当前 `exp1_summary.json` 的 `mc_first_sampled_violation` 显示：高证书代表 max-WEE idx34 和 max-ε_cert idx6 首个采样违例分别在 α=**2.0** 和 **1.875**；fragile low-ε_cert idx26 则在 α=**0.125** 已出现违例。因此不能笼统说所有配置都到 1.9–2.1 才违例。Monte Carlo 仅作经验 sanity check，不定义 certificate boundary；低证书配置的采样结果应结合 oracle 数值容差解读。

**说明**：当前参考结果来自**单一信道实现**；多种子泛化结论以独立的 holdout 检查为准（见 §6）。证书认证开销：40 配置 ≈ 600 次 oracle 调用 ≈ 18 s（复杂度对应论文 Eq.(21)–(23)）。

---

## 8. 复现性

- 种子：信道 `channel_seed=20260706`（`nominal_drop(cfg, seed)`，与 `cfg.seed` 解耦）、池 `pool_seed=31001`、MC `exp1_seed=41001` / `exp3_seed=43001`、holdout `gen_check_seed_base=60001`
- 池缓存指纹门禁：任何 SimConfig/CertConfig 参数变化 → 自动重建；候选落盘 `channel_seed / config_fingerprint / configuration_signature / cert_info{status,...}`，可脱离代码审计来源
- 不修改 `src/ris_base/robustness.py` 的 S-procedure / LMI / `optimize_lambda` 数学实现（冻结）

## 9. 与第一篇论文项目的关系

`src/ris_base/` 由 `C:\Users\57756\Desktop\ris_robust_simulation`（`ris_sim` V6.5）精简移植：公式、相对半径约定、种子口径一致；仅去除 torch/yaml/求解器预算字段。候选生成刻意轻量（zf_full + 最坏情形功率控制），**不调用**其优化器；认证层只把候选当固定 X 处理，与第一篇论文的可行性判定同一 LMI 口径（cvxpy SDP 交叉验证见测试）。
