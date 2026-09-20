# 面向 PA 状态漂移与动态负载失配的 LUT 检索方法

## 旧主任务详细交接文档

生成日期：2026-09-19  
来源：ChatGPT 项目“面向PA 状态漂移与动态负载失配的LUT检索方法”中的“旧主任务”

---

## 1. 文档用途与证据边界

本文用于把旧主任务交接给新的对话窗口、新模型或后续 Codex 任务。文档不是最近几条消息的摘要，而是对话中研究目标、工程演化、任务结果、文件线索、冻结规则、未完成事项和下一步工作的系统整理。

当前网页中可识别到：

- 96 条带有消息角色标记的语义消息；
- 90 条用户消息；
- 6 条直接助手消息；
- 581 个网页内部项目、附件、控件或虚拟列表项；
- 日期节点包括 8 月 25 日、8 月 26 日、9 月 8 日和 9 月 18 日。

大量内容以用户上传文件、粘贴执行日志或文件预览的形式出现。部分附件只显示文件名，或显示“无法显示预览”。因此：

- 只看到文件名，不等于已经读取二进制内容；
- 只看到用户上传的执行摘要，不等于当前工作树已经重新验证；
- Windows 历史路径不能直接视为当前 macOS 路径；
- 对话中写着“已完成”，需要区分脚本完成、任务完成、定向测试完成和论文结论冻结。

当前工程事实应以后续读取的 AGENTS.md、README.md、environment.yml、pyproject.toml、当前 Git 工作树和当前结果目录为准。

---

## 2. 论文主线

论文研究的问题不是“如何训练 DPD”，而是：

> 当 PA 工作条件发生未知变化，传统基于硬件状态或物理状态索引的 LUT 无法直接查询时，能否利用低带宽观测到的 PA 非线性行为构造行为指纹，并据此检索对应的 DPD 条目。

传统 LUT：

~~~text
PA 物理工作条件
        ↓
LUT 地址
        ↓
DPD 条目
~~~

本文方法：

~~~text
PA 当前未知状态
        ↓
采集输入/输出等效基带信号
        ↓
提取 PA 正向行为模型
        ↓
使用公共 B 段探测信号
        ↓
生成行为指纹
        ↓
与 LUT 行为指纹进行 CNMSE 匹配
        ↓
选择最相似状态
        ↓
调用对应 DPD 条目
~~~

核心映射是：

~~~text
PA 非线性行为指纹 → DPD
~~~

而不是：

~~~text
PA 物理状态 → DPD
~~~

---

## 3. 两个实验场景

### 3.1 Scenario 1：PA 状态漂移

Scenario 1 的物理含义是：

> PA 原本在 nominal 工作点附近运行，负载环境、偏置或前级输入功率发生变化；PA 仍在设计裕度内，但原 nominal DPD 发生失配。

三个状态轴：

- 负载阻抗；
- 载波路偏置电压；
- PA 输入功率。

目标联合规模：

~~~text
7 个 bias 状态
× 7 个 input-power 状态
× 7 个 load-drift 状态
= 343 个 PA 状态
~~~

旧主任务最后完成的是其中 7 个负载状态的选择。

### 3.2 Scenario 2：动态负载失配

Scenario 2 只改变负载阻抗，参数为：

~~~text
funMng
funAng
secMng
secAng
~~~

425 个状态包括：

- 1 个完全匹配状态；
- 24 个仅基波失配状态；
- 16 个仅二次谐波失配状态；
- 384 个基波和二次谐波联合失配状态。

Scenario 2 是 Scenario 1 负载状态筛选的真实数据来源。

---

## 4. 采集系统和数据链路

主要采集条件：

~~~text
PA 类型：Doherty PA
有效信号带宽：20 MHz
采样率：100 MHz
载波频率：2500 MHz
原始信号长度：24576 点
Scenario 2 vCarrier = 2.30 V
Scenario 2 vPeak = 5.50 V
Scenario 2 Pin = -21.00 dBm
~~~

信号链路：

~~~text
个人电脑
→ 矢量信号收发器
→ 驱动放大器
→ PA
→ 耦合器、衰减器、功率计
→ 矢量信号收发器
→ 个人电脑
~~~

负载网络：

~~~text
PA 输出端
→ 基波二端口网络
→ 二次谐波二端口网络
→ 50 Ω 负载
~~~

---

## 5. 三类数据与行为定义

### 5.1 X 类：无 DPD 行为

~~~text
xin → PA → yout_withoutdpd
~~~

主要变量：

~~~text
xin
yout_withoutdpd
yout_withoutdpd_ori
nmse_withoutdpd
acpr_low_withoutdpd
acpr_upper_withoutdpd
outputPower_withoutdpd
i_withoutdpd
dcPower_withoutdpd
efficiency_withoutdpd
~~~

定义：

~~~text
X：xin → yout_withoutdpd
~~~

### 5.2 Y 类：当前状态 DPD 链路

每个状态执行 ILC，并保存每次迭代：

~~~text
xin_pd_ori_ilc(:, n)
yout_withdpd_ori_ilc(:, n)
~~~

最终 DPD 条目通常为：

~~~text
xin_pd_ori_ilc(:, end)
~~~

Y 类行为：

~~~text
Y：xin_pd_ori_ilc(:, n) → yout_withdpd_ori_ilc(:, n)
~~~

自训练 DPD 成功条件：

~~~text
acpr_low_withdpd <= -50 dBc
且
acpr_upper_withdpd <= -50 dBc
~~~

### 5.3 stale-DPD 链路

先在 nominal 状态得到 nominal DPD，再固定它作用于所有状态：

~~~text
xin_pd_nominal
→ 不同 PA 状态
→ yout_withdpd_stale_ori
~~~

用于计算：

~~~text
nmse_withdpd_stale
acpr_low_withdpd_stale
acpr_upper_withdpd_stale
outputPower_withdpd_stale
i_withdpd_stale
~~~

该链路回答：

> nominal DPD 在状态变化后失配了多少？

---

## 6. A/B/C 分段和行为指纹

角色定义：

~~~text
A 段：离线建立 LUT
B 段：公共行为探测
C 段：在线未知状态行为提取
~~~

离线 LUT：

~~~text
A 段
→ 提取行为模型
→ 公共 B probe
→ y_A,B
~~~

在线 Query：

~~~text
C 段
→ 提取行为模型
→ 公共 B probe
→ y_C,B
~~~

匹配：

~~~text
CNMSE(y_C,B, y_A,B,n)
→ 选择距离最小的 LUT 状态 n
~~~

对话中记录的 ABC 边界：

~~~text
A：[0, 12288)
B：[12288, 17203)
C：[17203, 24576)
~~~

ownership 长度：

~~~text
A = 12288
B = 4915
C = 7373
~~~

模型有效长度：

~~~text
A = 12286
B = 4913
C = 7371
~~~

所有正式任务都应优先复用 canonical Rough Align、Fine Align、复增益调整和 CNMSE 处理，不能在单个任务中重新定义 preprocessing。

---

## 7. State 0 X/Y 行为等效性

### 7.1 模型结构

~~~text
X-A：1 个
Y-A：5 个
Y-C：5 个
总计：11 个
~~~

Frozen MP：

~~~text
orders = [1, 2, 3, 5, 7, 9]
memory_depth = {1:3, 2:2, 3:2, 5:1, 7:1, 9:1}
coefficient_count = 10
max_delay = 2
solver = np.linalg.lstsq(..., rcond=None)
~~~

### 7.2 Train / B 泛化结果

| 模型 | Train NMSE | B 泛化 NMSE |
|---|---:|---:|
| X-A | -42.5208 dB | -41.4771 dB |
| Y-A-1 | -41.9000 dB | -40.7672 dB |
| Y-A-2 | -41.3239 dB | -40.2163 dB |
| Y-A-3 | -41.3251 dB | -40.3140 dB |
| Y-A-4 | -41.4199 dB | -40.2750 dB |
| Y-A-5 | -41.3770 dB | -40.3538 dB |
| Y-C-1 | -42.1103 dB | -37.0435 dB |
| Y-C-2 | -41.9599 dB | -35.9402 dB |
| Y-C-3 | -41.9744 dB | -35.9010 dB |
| Y-C-4 | -41.8597 dB | -35.9429 dB |
| Y-C-5 | -42.1896 dB | -35.9514 dB |

### 7.3 CNMSE

~~~text
X vs Real：cnmse(real_B, X_response)
Y vs X：cnmse(X_response, Y_response)
Y vs Real：cnmse(real_B, Y_response)
~~~

| 模型 | Y vs X | Y vs Real |
|---|---:|---:|
| Y-A-1 | -42.7351 dB | -38.8347 dB |
| Y-A-2 | -39.8958 dB | -37.3637 dB |
| Y-A-3 | -39.7956 dB | -37.3041 dB |
| Y-A-4 | -39.8216 dB | -37.3169 dB |
| Y-A-5 | -39.8154 dB | -37.3110 dB |
| Y-C-1 | -38.5149 dB | -36.1831 dB |
| Y-C-2 | -36.3570 dB | -34.6975 dB |
| Y-C-3 | -36.3527 dB | -34.6984 dB |
| Y-C-4 | -36.3486 dB | -34.6935 dB |
| Y-C-5 | -36.2698 dB | -34.6376 dB |

### 7.4 结论

- Y-A 的 B 泛化整体好于 Y-C；
- Y-C 的 Train NMSE 较好但跨段泛化弱；
- 不能只用 Train NMSE 证明 X/Y 等效；
- 公共 B probe 后的 CNMSE 是更重要的判据；
- 本次只证明 State 0 范围内的诊断结果。

验证记录：

~~~text
_core：7 项 PASS
data_manager：5 项 PASS
signal_segmentation：12 项 PASS
原 behavior_model：5 项 PASS
State0 等效逻辑：7 项 PASS
合计：36 项检查通过
~~~

---

## 8. Ridge 诊断

目标：诊断 Y-C 泛化问题是否与 OLS 对局部 C 段数据过于敏感有关。

目标函数：

~~~text
theta_Ridge =
argmin_theta [
    (1/N) ||y - Phi theta||_2^2
    + lambda ||theta||_2^2
]
~~~

扫描 16 个 lambda，完成 176 次模型拟合。

lambda = 1e-8 的主要结果：

~~~text
X B-gen：
-41.4771 → -41.4597 dB

Y-A B-gen median：
-40.3140 → -40.4350 dB
改善约 0.1210 dB

Y-C B-gen median：
-35.9429 → -40.2894 dB
改善约 4.3465 dB

Y-C vs X median：
-36.3527 → -40.0248 dB
改善约 3.6722 dB

Y-C vs Real median：
-34.6975 → -37.5604 dB
改善约 2.8629 dB
~~~

早期诊断明确：

- lambda = 1e-8 尚未写入正式配置；
- 不能称为论文最终最优 lambda；
- 需要独立开发集和验证集；
- 同一个 B 段同时用于诊断和评价，存在数据泄漏风险。

---

## 9. 全 425 状态模型搜索与 5B retrieval

### 9.1 候选池

~~~text
Round 0～4 候选日志完整保留
candidate_count = 1064
Round 5 只生成 276 个候选壳
formal_model_bank_complete = false
common-support validation = 未完成
retrieval = 未启动
~~~

满足 Train 和 B-gen 门槛的状态数：

~~~text
Aend：344/425
C2：196/425
~~~

### 9.2 selected models

~~~text
Aend：425 个
C2：425 个
总计：850 个
~~~

其中：

~~~text
Aend：344 native-feasible，81 fallback
C2：196 native-feasible，229 fallback
~~~

不存在 joint-feasible candidate 时使用 minimax fallback：

~~~text
max(train_NMSE, B_NMSE)
~~~

### 9.3 Common-support

~~~text
M_common = 9
common A length = 12279
common B length = 4906
common C length = 7364
fingerprint length = 4906

LUT fingerprints   = (425, 4906)
Query fingerprints = (425, 4906)
CNMSE matrix       = (425, 425)
~~~

### 9.4 5B statewise retrieval

~~~text
statewise-best Y-C2 Query
→ statewise-best Y-Aend LUT
→ fingerprint CNMSE Top-1
→ State_Q 固定
→ canonical 5B OFF Real-B lookup
~~~

Real-B 没有参与模型选择、State_Q 选择、reranking 或 candidate selection。

记录的压缩结果：

| 指标 | 第一组 | 第二组 | 变化列 |
|---|---:|---:|---:|
| Exact | 161 | 218 | -57 |
| Shareable | 389 | 411 | -22 |
| Failure | 36 | 14 | +22 |
| Non-exact | 264 | 207 | — |
| Non-exact shareable | 228 | 193 | — |

由于网页压缩后的表头方向不完整，第一组和第二组的具体含义必须以后直接读取 retrieval_summary.json 和 CSV 确认。

Shareable 判据：

~~~text
retrieved_real_B_CNMSE_dB < -40 dB
~~~

自身命中保留 -Inf。

---

## 10. 低带宽 1B / 0.5B

对话中确认上传过：

~~~text
retrieval_summary_1B.json
scenario_2_C2_to_Aend_1B_retrieval.xlsx
retrieval_summary_0p5B.json
scenario_2_C2_to_Aend_0p5B_retrieval.xlsx
~~~

0.5B 任务定义：

~~~text
LUT fingerprint =
xin_pd_ori_ilc(:, ilc_A_end)
-
yout_withdpd_ori_ilc(:, ilc_A_end)

Online query =
xin_pd_ori_ilc(:, ilc_C_2)
-
yout_withdpd_ori_ilc(:, ilc_C_2)
~~~

当前对话中部分低带宽文件显示“无法显示预览”，因此完整低带宽结论必须重新读取 JSON、Excel 和 validation 文件。

---

## 11. 行为聚类与 Type III 压缩

### 11.1 A 段 Complete-Link

预处理顺序：

~~~text
完整 xin / yout_withoutdpd_ori
→ 完整记录 Rough Align
→ 完整记录 Fine Align
→ 切 A=[0:12288)
→ A 段 complex-gain adjustment
→ 两两 CNMSE
→ deterministic Complete-Link
~~~

阈值：

~~~text
T_CNMSE = -40 dB
D_ij < -40 dB
~~~

结果：

~~~text
状态数：425
unique state pairs：90100
cluster 数量：77
最大 cluster 大小：12
最小 cluster 大小：1
中位 cluster 大小：6
singleton 数量：4
全局最差类内 CNMSE：-40.084066914 dB
全局最佳类内 CNMSE：-45.179648783 dB
~~~

### 11.2 代表状态

采用 deterministic minimax medoid：

~~~text
r_k = argmin_i max_j D_ij
~~~

得到 77 个 cluster 和 77 个代表状态。三个代表 ID 完全一致：

~~~text
representative_state_id
representative_fingerprint_state_id
representative_dpd_state_id
~~~

### 11.3 Type III

~~~text
425 个状态
→ 77 个行为 cluster
→ 77 个代表 fingerprint
→ 77 个代表 DPD
~~~

主要结果：

~~~text
压缩率：81.8823529%
Exact State 命中：63/425
同簇命中：295/425
检索指纹 CNMSE 中位数：-46.394812461 dB
Real-B 严格低于 -40 dB：402/425
非 Exact 且有限 Real-B CNMSE 中位数：-43.101315844 dB
有限 Real-B CNMSE 最差值：-35.479619692 dB
~~~

本轮没有执行 DPD replay、LUT-I/LUT-II 对比、Real-B 参与重排或低带宽处理。

---

## 12. Scenario 1 负载状态筛选

### 12.1 候选定义

只保留基波负载漂移：

~~~text
secMng = 0
secAng = 0
funMng ∈ {0, 0.1, 0.2}
~~~

候选共 17 个：

~~~text
1 个 matched
8 个 |Gamma|=0.1
8 个 |Gamma|=0.2
~~~

### 12.2 筛选顺序

~~~text
设计裕度
    ↓
own-DPD 可恢复
    ↓
stale-DPD 失配层级
    ↓
真实 PA 行为变化层级
    ↓
候选行为互补性
    ↓
相位方向覆盖
~~~

不是选择失配最严重的 7 个点，也不是只按 Smith 圆几何均匀性选择。

### 12.3 最终推荐

~~~text
0
0.1∠0°
0.1∠90°
0.1∠180°
0.2∠0°
0.2∠90°
0.2∠180°
~~~

对应 State：

~~~text
{0, 17, 51, 85, 153, 187, 221}
~~~

统一：

~~~text
Gamma_2f = 0
~~~

### 12.4 代表性指标

~~~text
Nominal:
stale NMSE ≈ -41.99 dB
stale ACPR_avg ≈ -51.04 dBc

0.1∠0°:
stale ACPR ≈ -47.60 dBc
B 段 CNMSE ≈ -40.54 dB

0.1∠90°:
stale ACPR ≈ -43.58 dBc
B 段 CNMSE ≈ -35.83 dB

0.1∠180°:
B 段 CNMSE ≈ -33.19 dB

0.2∠0°:
stale ACPR ≈ -41.63 dBc
B 段 CNMSE ≈ -33.14 dB

0.2∠90°:
stale NMSE ≈ -25.65 dB
stale ACPR ≈ -35.04 dBc
own NMSE ≈ -44.42 dB
own ACPR ≈ -54.43 dBc
ILC iterations = 4
B 段 CNMSE ≈ -26.21 dB

0.2∠180°:
stale ACPR ≈ -38.45 dBc
B 段 CNMSE ≈ -28.88 dB
~~~

最终分析明确记录：

- 17 个候选全部 own-DPD target reached；
- 没有超过 37 dBm 保护阈值；
- 行为距离覆盖约 -40.5 dB 到 -26.2 dB；
- 0.2∠90°是较强的设计裕度边界状态。

### 12.5 早期假设替换原因

早期假设：

~~~text
45° / 180° / 270°
~~~

最终改为：

~~~text
0° / 90° / 180°
~~~

原因：

- 270°方向过于温和；
- 0.1∠270° stale ACPR 约为 -51.08 dBc，接近 nominal 的 -51.04 dBc；
- 0.2∠270° stale ACPR 约为 -47.49 dBc；
- 90°方向更敏感；
- 0°比45°更适合提供轻漂移端。

行为分散度：

~~~text
[0°, 90°, 180°] closest-pair CNMSE ≈ -36.04 dB
[45°, 180°, 270°] closest-pair CNMSE ≈ -37.21 dB

[0°, 90°, 180°] pairwise median ≈ -29.08 dB
[45°, 180°, 270°] pairwise median ≈ -31.52 dB
~~~

建议称为“基于当前 Scenario 2 实测 PA 行为选出的代表性负载方向”，不要称为所有条件下的绝对最优负载。

---

## 13. 工程规则与安全边界

当前工程固定模块：

~~~text
core
data_management
signal_segmentation
pa_performance_evaluation
behavior_modeling
behavior_fingerprint_retrieval
retrieval_oriented_model_selection
behavior_fingerprint_ranking_consistency
low_bandwidth_behavior_analysis
lut_clustering_compression
~~~

三棵目录树：

~~~text
scripts/
results/
work_logs/
~~~

任务映射：

~~~text
scripts/<module_name>/<task_name>/
results/<module_name>/<task_name>/
work_logs/<module_name>/<task_name>/
~~~

安全规则：

- data/raw 默认只读；
- 不覆盖历史结果；
- 不自动进入 Round 5；
- 不重新采集 PA 数据；
- 不重新运行仪器控制；
- 不无授权重新训练 ILC；
- 不执行 git add、commit、push、reset、restore、clean、checkout；
- 不建立新的一级模块；
- 不形成 Task-to-Task 依赖；
- 跨模块复用应依赖 shared 层。

---

## 14. 主要文件索引

初始 MATLAB 来源：

~~~text
adjust.m
Fine_Align.m
Rough_Align.m
Align_Rough.m
Discard_sameVector.m
integer_delay.m
aclr.m
chan_pow.m
dbp.m
nmsecount_old.m
nmseCount.m
Matrix_3st_order.m
Matrix_5st_order.m
Matrix_Dynamic.m
Matrix_Emempolyn.m
Matrix_LagPolyn.m
Matrix_leadPoyn.m
Matrix_MemPolyn.m
Matrix_Static.m
top.m
top_test.m
~~~

State 0：

~~~text
xy_equivalence.py
run_state0_xy_equivalence.py
test_state0_xy_equivalence.py
state0_xy_equivalence_summary.csv
common_B_responses.npz
validation.json
~~~

Ridge：

~~~text
ridge.py
ridge_scan.py
run_state0_ridge_scan.py
test_state0_ridge_scan.py
state0_ridge_scan_models.csv
state0_ridge_scan_summary.csv
ridge_scan_validation.json
theta_ridge_scan.npz
~~~

5B 模型选择和 retrieval：

~~~text
candidate_model_grid.csv.gz
candidate_search_Aend.csv.gz
candidate_search_C2.csv.gz
partial_validation.json
search_progress.json
statewise_adaptive_best_metrics.xlsx
selected_Aend_coefficients.npz
selected_C2_coefficients.npz
selected_model_descriptors.json
selection_validation.json
selected_Aend_models.csv
selected_C2_models.csv
selected_models.csv
common_support_diagnostics.csv
retrieval_results_statewise_best_round0_4_5B.csv
top1_fingerprint_retrieval.csv
retrieval_summary.json
validation.json
~~~

聚类：

~~~text
pairwise_cnmse_5B_A_preprocessed.npz
cluster_assignments.csv
cluster_summary.csv
cluster_representatives.csv
type3_cluster_compressed_retrieval.xlsx
~~~

Scenario 1 任务规范要求的输出：

~~~text
scenario_1_load_drift_candidate_selection.xlsx
01_candidate_summary.csv
02_phase_comparison.csv
03_phase_triplet_evaluation.csv
04_recommended_7.csv
05_pairwise_B_CNMSE.csv
06_selection_summary.json
07_final_result_summary.txt
08_validation_checks.json
~~~

---

## 15. 已完成、部分完成与未完成事项

### 15.1 明确报告为已完成

- 数据管理基础能力；
- canonical A/B/C；
- State 0 的 11 个行为模型；
- State 0 X/Y 等效性诊断；
- State 0 Ridge 扫描；
- Round 0–4 候选池整理；
- Aend/C2 850 个 selected models；
- 5B common-support；
- 5B statewise retrieval；
- 5B A 段 Complete-Link 聚类；
- 77 个代表状态；
- Type III 425→77 压缩检索；
- Scenario 1 负载轴选择。

### 15.2 部分完成

- Round 5 formal model bank 未完成；
- selected models 中存在 fallback；
- 1B / 0.5B 文件已上传但完整数值未在当前对话中全部展开；
- State 0 不能直接等价为 425 状态全量 X/Y 证明；
- 历史 Windows 路径与当前 macOS checkout 尚未完全映射。

### 15.3 尚未完成或未确认

- 正式 Ridge lambda 的独立验证；
- 低带宽全量数值复核；
- cluster representative 的 DPD replay；
- LUT-I / LUT-II 对比；
- Real-B 参与的候选重排；
- Scenario 1 的 7 个 bias 状态；
- Scenario 1 的 7 个 input-power 状态；
- 343 个联合 Scenario 1 状态的最终冻结；
- 论文最终图表和证据链的统一核验。

---

## 16. 新接手者第一步

### Step 1：确认当前工程

~~~text
pwd
git status --short
git rev-parse HEAD
~~~

读取 AGENTS.md、README.md、environment.yml、pyproject.toml 和 work_logs/core/codex_handoff/codex_handoff.txt。

### Step 2：建立历史路径映射

~~~text
旧 behavior_model
→ 当前 behavior_modeling

旧 clustering
→ 当前 lut_clustering_compression

旧 _core
→ 当前 core
~~~

不要直接复制 Windows 路径。

### Step 3：确认当前数据和结果

确认 data/raw、MAT 数量、状态数、canonical state ordering、A/B/C segmentation、Frozen MP、Ridge、5B、1B 和 0.5B 结果。

### Step 4：低带宽复核

直接读取：

~~~text
retrieval_summary_1B.json
retrieval_summary_0p5B.json
validation.json
~~~

补齐 Exact、Shareable、Failure、Top-k、Real-B CNMSE、ranking consistency 和最差状态。

### Step 5：完成 Scenario 1 另外两个轴

当前负载轴：

~~~text
0
0.1∠0°
0.1∠90°
0.1∠180°
0.2∠0°
0.2∠90°
0.2∠180°
~~~

继续确定 7 个 bias 状态、7 个 input-power 状态、三轴联合保护边界以及 343 个联合状态是否全部采集。

### Step 6：避免未经授权的大规模实验

不要自动重新采集 PA 数据、重新运行仪器控制、重新训练 ILC、重新执行模型搜索、重新执行 LUT retrieval、修改 raw MAT 或覆盖历史结果。

---

## 17. 最终交接结论

旧主任务已经形成了较完整的研究闭环：

1. 研究目标是未知 PA 状态下的行为索引，而不是单纯 DPD 训练；
2. 已定义 X 类、Y 类和 Real-B 行为；
3. 已建立 A/B/C 分段和公共 B probe；
4. 已完成 State 0 的 11 模型 X/Y 等效性诊断；
5. 已发现 OLS 下 Y-C 跨段泛化较弱；
6. Ridge 诊断显示适度正则可以改善 Y-C；
7. 已完成 425 状态的多轮模型候选、statewise 选择和 5B retrieval；
8. 已完成 5B A 段 -40 dB Complete-Link 聚类；
9. 已生成 77 个行为代表状态；
10. 已完成 Type III 425→77 聚类压缩检索；
11. 已上传并规划 1B / 0.5B 低带宽 retrieval；
12. 已根据 Scenario 2 实测数据完成 Scenario 1 负载轴选择；
13. 当前推荐 State 为：

~~~text
{0, 17, 51, 85, 153, 187, 221}
~~~

对应：

~~~text
0
0.1∠0°
0.1∠90°
0.1∠180°
0.2∠0°
0.2∠90°
0.2∠180°
~~~

最重要的后续工作：

- 将历史 Windows 结果与当前 macOS 工程逐一映射；
- 重新确认低带宽结果完整数值；
- 明确当前正式 Ridge / Frozen MP 合同；
- 完成 Scenario 1 的 bias 和 input-power 两个状态轴；
- 区分历史结果和当前验证结果，形成论文级最终证据链。

---

## 18. 文档生成说明

本文件根据“旧主任务”网页对话中可读取的消息文本、附件名称、执行摘要和结果说明整理生成。

本次只新增本文件，没有修改 Python 源代码、MATLAB 源代码、data/raw、results 中的历史结果、其他工作日志或 Git 提交历史。

只显示文件名或“无法显示预览”的附件，已按证据边界记录，不能视为已经完成二进制内容核验。
