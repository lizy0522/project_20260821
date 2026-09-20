# project_20260821

面向 PA 动态工作条件研究的标准 Python 科研工程。

当前工程长期服务两个场景：

```text
scenario_1 = PA state drift（PA 状态漂移）
scenario_2 = dynamic load mismatch（动态负载失配）
```

同时支持：

```text
cross_scenario
```

用于真正同时使用或比较两个场景的科研任务。

工程采用固定的 **10 个一级模块**，并统一使用场景感知目录结构。

普通科研模块：

```text
Module -> Scenario -> Task
```

特殊的 `retrieval_oriented_model_selection`：

```text
Module -> Route -> Scenario -> Task
```

原始/处理中间数据：

```text
Data -> Scenario -> Experiment
```

详细 Agent/Codex 执行规则以根目录 `AGENTS.md` 为准。

---

# 配置文件职责

```text
AGENTS.md
= Agent/Codex 的工程执行规则

environment.yml
= Conda/Python 环境唯一事实来源

pyproject.toml
= Python 工具配置（Ruff、pytest 等）

README.md
= 工程结构、使用方式和快速上手说明

.gitignore
= Git 排除规则
```

README 不维护独立的完整依赖版本表，避免与 `environment.yml` 漂移。

---

# Quick Start

## 1. 创建环境

```bash
conda env create -f environment.yml
```

当前环境名称由 `environment.yml` 的 `name` 字段定义。

## 2. 激活环境

```bash
conda activate project_20260821
```

## 3. 同步已有环境

```bash
conda env update -f environment.yml
```

默认不使用：

```bash
--prune
```

## 4. 基础验证

```bash
python -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.system()); print(platform.machine())"
python -c "import numpy, scipy; print(numpy.__version__); print(scipy.__version__); numpy.__config__.show()"
python -m pip check
ruff check --no-cache scripts
pytest
```

正式科研任务还应运行其已有 task-specific test（任务专属测试）或最小 smoke test（冒烟测试）。

---

# 工程目录

```text
project_20260821/
├── data/
│   ├── raw/
│   │   ├── scenario_1/
│   │   │   └── <experiment>/
│   │   └── scenario_2/
│   │       └── <experiment>/
│   └── processed/
│       ├── scenario_1/
│       └── scenario_2/
│
├── scripts/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── shared/
│   │   └── <engineering_task>/
│   ├── data_management/
│   │   └── shared/
│   ├── signal_segmentation/
│   │   ├── shared/
│   │   └── scenario_2/
│   │       └── <task>/
│   ├── pa_performance_evaluation/
│   │   ├── shared/
│   │   ├── scenario_1/
│   │   ├── scenario_2/
│   │   └── cross_scenario/
│   ├── behavior_modeling/
│   │   ├── shared/
│   │   ├── scenario_1/
│   │   ├── scenario_2/
│   │   └── cross_scenario/
│   ├── behavior_fingerprint_retrieval/
│   │   ├── shared/
│   │   ├── scenario_1/
│   │   ├── scenario_2/
│   │   └── cross_scenario/
│   ├── retrieval_oriented_model_selection/
│   │   ├── shared/
│   │   ├── self_hit_oriented/
│   │   │   └── scenario_2/
│   │   │       └── <task>/
│   │   └── dpd_shareability_oriented/
│   │       └── scenario_2/
│   │           └── <task>/
│   ├── behavior_fingerprint_ranking_consistency/
│   │   ├── shared/
│   │   ├── scenario_1/
│   │   ├── scenario_2/
│   │   └── cross_scenario/
│   ├── low_bandwidth_behavior_analysis/
│   │   ├── shared/
│   │   ├── scenario_1/
│   │   ├── scenario_2/
│   │   └── cross_scenario/
│   └── lut_clustering_compression/
│       ├── shared/
│       ├── scenario_1/
│       ├── scenario_2/
│       └── cross_scenario/
│
├── results/
│   └── 与实际科研 task 对应的 module/scenario/task
│       或 retrieval module/route/scenario/task
│
├── work_logs/
│   └── 与实际科研 task 对应的 module/scenario/task
│       或 retrieval module/route/scenario/task
│
├── AGENTS.md
├── README.md
├── environment.yml
└── pyproject.toml
```

说明：场景目录按需存在。没有真实任务时，不为了目录对称创建空 task。

一级模块必须始终满足：

```text
Modules(scripts) = Modules(results) = Modules(work_logs)
```

---

# 固定 10 个模块

| 模块 | 主要职责 |
|---|---|
| `core` | 工程路径、模块注册表、通用底层工具、validator、工程维护和交接 |
| `data_management` | 场景/实验路径、MAT 加载、状态索引、canonical ordering、manifest |
| `signal_segmentation` | A/B/C 分段、valid samples、dmax 边界、公共 B 段和预处理 |
| `pa_performance_evaluation` | NMSE、ACPR、输出功率、DC power、效率和 DPD on/off 实测评价 |
| `behavior_modeling` | PA 正向行为建模，以及建模精度/泛化精度导向的模型选择 |
| `behavior_fingerprint_retrieval` | 模型已确定后的行为指纹、LUT、distance、Top-k 和 Real-B 验证 |
| `retrieval_oriented_model_selection` | LUT 检索效果导向的 basis/Ridge/model structure 选择 |
| `behavior_fingerprint_ranking_consistency` | ranking consistency 和跨状态/模型/带宽排序分析 |
| `low_bandwidth_behavior_analysis` | 5B→nB、低带宽行为和跨带宽保持性 |
| `lut_clustering_compression` | 状态聚类、代表状态选择及 LUT/指纹/DPD 条目压缩 |

---

# Scenario Scope（场景作用域）

普通科研任务使用：

```text
scenario_1
scenario_2
cross_scenario
```

其中：

```text
scenario_1
= PA 状态漂移

scenario_2
= 动态负载失配

cross_scenario
= 科研问题本身明确同时涉及两个场景
```

例如：

```text
scripts/behavior_modeling/scenario_2/<task>/
```

跨场景任务：

```text
scripts/pa_performance_evaluation/cross_scenario/<task>/
```

新任务的 `task_name` 原则上不重复场景前缀，因为场景已经由目录表达；历史任务已有 `scenario_2_` 前缀时保留原名，不做无必要的批量重命名。

---

# Retrieval-Oriented Model Selection

`retrieval_oriented_model_selection` 是唯一 route-aware（研究路线感知）模块。

固定 route：

```text
self_hit_oriented
= 以检索到真实状态自身为第一目标

dpd_shareability_oriented
= 以检索出的 DPD 条目可共享率为第一目标
```

正式路径：

```text
scripts/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/
results/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/
work_logs/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/
```

模块公共代码始终位于：

```text
scripts/retrieval_oriented_model_selection/shared/
```

route 下不建立 `shared/`。

路径解析优先使用：

```text
scripts/core/shared/project_paths.py
scripts/retrieval_oriented_model_selection/shared/route_paths.py
```

当前统一接口包括：

```text
get_raw_scenario_root(...)
get_raw_experiment_root(...)
get_processed_scenario_root(...)
get_task_paths(module_name, task_name, scenario_scope=...)
get_route_task_paths(route, scenario_scope, task_name)
```

不要通过固定 `.parent` 层数推断 route/module/task，也不要在新活动源码中重新硬编码机器绝对路径。

---

# Behavior Modeling 与 Retrieval-Oriented Selection 的边界

## `behavior_modeling`

主要优化：

```text
Train NMSE
Validation NMSE
Generalization NMSE
Generalization Gap
model capacity
numerical conditioning
```

回答：

> 什么样的行为模型能够更准确地描述和泛化 PA 行为？

## `retrieval_oriented_model_selection`

主要优化：

```text
Top-1
N_self
N_shareable
N_valid
retrieval margin
MRR
Top-k
```

回答：

> 什么样的行为模型最适合 LUT 检索？

不能仅根据任务名中出现 `basis`、`model` 或 `ridge` 判断模块。

---

# Shared Code 与任务隔离

每个源码模块固定使用：

```text
scripts/<module>/shared/
```

`shared/` 是模块稳定公共 API。

推荐依赖：

```text
Task -> Shared
Module A Task -> Module B Shared
```

原则上禁止：

```text
Task -> Task
Shared -> Task
Module A -> Module B concrete task
```

如果某个任务实现需要被多个任务复用，应把真正公共的部分提升到所属模块 `shared/`。

`results/` 和 `work_logs/` 不建立模块公共 `shared/`。

---

# 数据目录

## 原始数据

```text
data/raw/<scenario>/<experiment>/
```

当前历史 Scenario 2 实验的正式路径为：

```text
data/raw/scenario_2/experiment_2026_0816/
```

`data/raw/` 默认只读。

除非明确执行原始数据目录架构迁移，否则不修改、不覆盖、不删除、不重命名原始实验文件。

工程为历史 checkpoint/cache 保留 `legacy_raw_manifest()` 兼容能力，但它只用于识别架构迁移前的历史科研状态。新任务应使用当前 `scenario + experiment` 数据身份，不应继续以旧 raw 目录结构作为工程逻辑。

## 可复用处理中间数据

```text
data/processed/scenario_1/
data/processed/scenario_2/
```

只服务单个任务的中间产物优先放：

```text
results/<module>/<scenario_scope>/<task>/
```

`data/processed/` 不建立 `cross_scenario/`。

---

# Results 与 Work Logs

## Results

保存任务正式科研产物，例如：

- CSV/Excel；
- JSON summary；
- MAT/NPZ/model；
- figures/PDF；
- final metrics；
- final model definition。

## Work Logs

保存：

- `execution_log.txt`；
- checkpoint/resume；
- pre-search validation；
- runtime diagnostics；
- screening partitions；
- CandidateScore cache；
- engineering audit。

因此 `work_logs/` 中可能存在非常重要的科研中间状态，不能机械清理。

总体工程交接：

```text
work_logs/core/codex_handoff/codex_handoff.txt
```

---

# Python 环境

环境唯一事实来源：

```text
environment.yml
```

当前原则：

```text
Python 3.11
conda-forge
OpenBLAS
```

在 Apple Silicon Mac 上使用原生 arm64 Conda/Python 环境；除非某个明确依赖只能通过 Rosetta/x86_64 运行且用户确认，否则避免 Rosetta Python。

正式科研计算不要使用 Conda `base`、系统 Python 或另一个工程的环境。

---

# CPU-Heavy 任务默认并行规范

正式 CPU-heavy 搜索默认：

```text
10 个 spawn worker processes
每 worker 1 个 BLAS/OpenBLAS thread
不设置 CPU affinity
GPU disabled
禁止 nested multiprocessing
```

调试/reference gate 可以显式使用更少 worker。

正式长任务应具有 checkpoint/resume、进度、吞吐、ETA、worker error 和 fallback 监控。

---

# 代码检查

基础静态检查：

```bash
python -m compileall -q scripts
ruff check --no-cache scripts
python -m pip check
```

工程级架构/路径维护优先运行直接相关的小型测试，例如：

```bash
pytest scripts/core/shared/tests scripts/data_management/shared/tests
```

完整：

```bash
pytest
```

用于需要全量回归时执行。历史任务中部分 regression test 可能依赖此前本来就不存在的科研 artifact；遇到这种情况应报告真实缺失原因，不得为了测试变绿而生成、复制或伪造科研结果。

涉及架构维护时还应运行项目 layout validator，并确认：

```text
Modules(scripts) = Modules(results) = Modules(work_logs)
```

以及：

- 普通科研 task 位于 scenario scope；
- Retrieval task 位于 route/scenario scope；
- `shared/` 只位于模块根；
- route/scenario 下不存在 `shared/`；
- `results/` 和 `work_logs/` 不存在模块公共 `shared/`；
- 不存在 task-to-task 依赖；
- 活动源码没有依赖旧目录结构。

如果某些历史回归测试依赖已经不存在的科研 artifact，应报告实际缺失原因，不得伪造结果来让测试通过。

当前最终架构已经完成整合。后续普通科研开发应直接在现有 Scenario/Route 结构内新增任务，不再为历史任务做无必要的大规模目录搬迁或批量去除 `scenario_2_` 前缀。

---

# Git Repository Skeleton

Git 本身只跟踪文件，不跟踪空目录。本工程对少量正式结构目录使用空的
.gitkeep，使 clone 后仍能看到固定模块、数据场景，以及当前已经存在的
results 或 retrieval route/scenario 结构。

.gitkeep 只表示目录骨架，不表示目录中的实际科研内容会被提交。

## 保留在 Git 中的内容

scripts/                         源码、任务脚本、测试和 shared API
AGENTS.md                        工程执行规则
README.md                        工程说明
environment.yml                  受控 Python/Conda 环境
pyproject.toml                   Python 工具配置
.gitignore / .gitattributes      Git 边界和文本属性
必要的文本型 work_logs/handoff   可审查的工程记录
.gitkeep                         正式工程目录骨架

## 默认保留在本地的内容

data/raw/                        实际原始实验数据
data/processed/                  实际处理中间数据
results/                         实际科研结果
checkpoint/                      可恢复科研状态
candidate_score_cache/           候选评分缓存
screening/                       筛选运行产物
runtime/                         运行时数组和诊断
大型 JSON/JSONL 科学缓存          高频生成的候选和分区记录
.npy/.npz/.mat 等二进制文件       科研数据和数值缓存

不要使用大量 task 级 .gitkeep 复刻历史科研运行现场。clone 后保留的是正式
工程设计层级，而不是每个历史 task、round、checkpoint 或 cache 子目录。

当前正式路径仍然是：

普通科研任务：Module -> Scenario -> Task
Retrieval：   Module -> Route -> Scenario -> Task
数据：        Data -> Scenario -> Experiment

如果一个目录已经有 __init__.py、execution_log.txt 或其他被 Git 跟踪的文件，
不需要再添加 .gitkeep。如果要调整骨架规则，应先检查当前实际目录，再用
git check-ignore -v 验证真实数据/结果仍然被忽略；目录骨架维护不会自动执行
git add、git commit 或 git push。

## Git Repository Boundary

Git 保存工程定义、手写源码、测试、环境和工具配置、必要的文本型工程知识以及
正式目录骨架；Git 默认不保存科研运行现场。

因此 clone 后可以看到：

scripts/                       源码、测试和 shared API
固定 10 个 module 根目录
现有的正式 Scenario/Route/Scenario 骨架
必要的 work_logs 文本和 core 小型审计

但不保证出现：

data/raw 的实际 Experiment 文件
data/processed 的生成数据
results 的实际科研输出
CandidateScore、checkpoint、screening、runtime 和大型 JSON/JSONL cache
distance matrices、fingerprints、NPY/NPZ/MAT 等科学运行产物

work_logs 不会整体忽略。execution_log.txt、codex_handoff.txt、Markdown 和必要的
小型 core 工程审计可以按需跟踪；CandidateScore、checkpoint、screening、runtime、
分区缓存、scientific JSONL 和科学二进制文件默认留在本地。

当前 DPD-shareability 任务的 optimization、pre_search_validation 和
execution_log.txt 也属于本地科研运行现场；Core 中的 migration_plan.json 和
source_patch_manifest.json 是保留的少量工程 provenance，其余包含本机路径、raw
manifest、文件 hash 或 cache 清理明细的架构 JSON 默认不进入 Git。

.gitkeep 不是 Git 的特殊语法，而是本工程用于保留正式空目录的约定文件。它不
代表原始数据、Experiment、科研结果或历史 task 已经进入 Git。具体 task、round、
checkpoint 和 cache 子目录不会通过大量 .gitkeep 复刻到仓库。

如果某类历史文件已经被 Git 跟踪，新增 .gitignore 规则不会自动取消跟踪；这类
冲突需要单独审计和用户授权，不在普通仓库边界维护中自动处理。

---

# Git

Git 安全规则以 `AGENTS.md` 为准。

常用只读检查：

```bash
git status --short --branch
git diff
git diff --check
git log
```

除用户明确要求外，不要使用：

```text
git reset --hard
git clean -fd
git restore 覆盖用户修改
git add -A
git add .
```

工程默认不把 `data/`、`results/` 和大型运行产物加入新的 Git 提交。

---

# Agent 使用

Codex 或其他 Agent 在本工程中工作时，应先读取：

```text
AGENTS.md
```

其中定义：

- 固定 10 模块；
- Module/Scenario/Task 架构；
- Retrieval 的 Module/Route/Scenario/Task 架构；
- Data/Scenario/Experiment 架构；
- 模块职责；
- Shared/Task 边界；
- 新任务与持续维护规则；
- `data/raw` 保护；
- checkpoint/cache 保护；
- CPU-heavy 并行规则；
- Python 环境；
- Git 安全；
- 完成前检查。

README 只负责说明工程如何使用，不重复维护全部 Agent 细则。
