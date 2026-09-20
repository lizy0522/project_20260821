# Codex 工程执行规则

## 适用范围

本文件适用于工程：

```text
project_20260821
```

以及由同一模板整理的标准 Python 科研工程。

工程根目录的核心业务结构为：

```text
data/
scripts/
results/
work_logs/
```

工程级配置文件包括：

```text
AGENTS.md
README.md
environment.yml
pyproject.toml
.gitignore
```

Codex 每次执行任务时，必须先识别：

1. 本次工作属于哪个一级模块；
2. 是否属于具体 PA 场景；
3. 是否属于 `retrieval_oriented_model_selection` 的某条 route；
4. 是新任务还是已有任务的持续维护；
5. 是否涉及科研结果、checkpoint、cache、原始数据或工程架构。

除用户明确授权外，不得把普通任务升级为工程级重构，不得顺带修改与当前目标无关的目录、API、依赖或 Git 状态。

---

# 零、工程架构

## 1. 固定 10 个一级模块

工程固定使用以下 10 个一级模块：

```text
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
```

名称必须保持小写英文 `snake_case`。

除非用户明确批准工程架构变更，否则不得：

- 新增第 11 个一级模块；
- 删除现有模块；
- 修改模块名称；
- 使用近义目录替代正式模块；
- 把具体科研任务直接放到 `scripts/`、`results/`、`work_logs/` 根目录。

模块注册表应维护在：

```text
scripts/core/shared/module_registry.py
```

并始终满足：

```text
Modules(scripts) = Modules(results) = Modules(work_logs)
```

一级模块集合必须严格一致。

## 2. 场景定义

当前工程长期服务两个 PA 动态场景：

```text
scenario_1 = PA state drift（PA 状态漂移）
scenario_2 = dynamic load mismatch（动态负载失配）
```

科研任务还允许：

```text
cross_scenario
```

表示同时使用两个场景、比较两个场景，或明确由一个场景服务另一个场景的任务。

场景目录固定使用：

```text
scenario_1
scenario_2
cross_scenario
```

不得创建近义作用域，例如：

```text
state_drift/
load_mismatch/
scenario1/
scenario2/
both_scenarios/
```

场景的物理含义可以写在 README、任务说明和日志中，但路径中的 canonical ID（规范标识）只使用上述名称。

## 3. 普通科研模块的正式路径

除 `core` 和特殊 route-aware 模块外，科研任务默认使用：

```text
scripts/<module>/<scenario_scope>/<task>/
results/<module>/<scenario_scope>/<task>/
work_logs/<module>/<scenario_scope>/<task>/
```

其中：

```text
scenario_scope ∈ {scenario_1, scenario_2, cross_scenario}
```

源码模块级公共代码仍只允许位于：

```text
scripts/<module>/shared/
```

不得建立：

```text
scripts/<module>/scenario_1/shared/
scripts/<module>/scenario_2/shared/
scripts/<module>/cross_scenario/shared/
```

`results/` 和 `work_logs/` 中也不得建立模块公共 `shared/`。

只为真实存在的任务创建场景目录。不得为了目录对称创建空的 `scenario_1/`、`scenario_2/` 或 `cross_scenario/` 任务树。

### 3.1 `core` 例外

`core` 保存工程级公共能力和工程维护任务，不强制增加场景层。

允许：

```text
scripts/core/shared/
scripts/core/<engineering_task>/
work_logs/core/<engineering_task>/
```

工程级维护不是 PA 科研场景任务，不应被强行归入 `scenario_1`、`scenario_2` 或 `cross_scenario`。

### 3.2 `data_management` 说明

`data_management/shared/` 保存跨场景数据管理公共能力。

如果未来出现真正的场景专属数据管理任务，可以使用：

```text
scripts/data_management/<scenario_scope>/<task>/
```

但不得为了形式完整创建空任务。

## 4. `retrieval_oriented_model_selection` 的 route + scenario 架构

该模块是当前唯一 route-aware（研究路线感知）模块。

固定 route：

```text
self_hit_oriented
dpd_shareability_oriented
```

正式路径为：

```text
scripts/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/
results/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/
work_logs/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/
```

其中：

```text
route ∈ {self_hit_oriented, dpd_shareability_oriented}
scenario_scope ∈ {scenario_1, scenario_2, cross_scenario}
```

当前两个 route 的含义固定为：

```text
self_hit_oriented
= 以检索到真实状态自身为第一研究目标

dpd_shareability_oriented
= 以检索出的 DPD 条目可共享率为第一研究目标
```

route 不是一级模块，也不是公共代码层。

模块公共代码仍只允许：

```text
scripts/retrieval_oriented_model_selection/shared/
```

禁止：

```text
scripts/retrieval_oriented_model_selection/<route>/shared/
scripts/retrieval_oriented_model_selection/<route>/<scenario_scope>/shared/
```

路径解析优先复用：

```text
scripts/retrieval_oriented_model_selection/shared/route_paths.py
scripts/core/shared/project_paths.py
```

不得在任务中依赖固定 `.parent` 层数假设 `task.parent == module`。

## 5. 数据目录的正式架构

### 5.1 原始数据

正式数据结构为：

```text
data/raw/
├── scenario_1/
│   └── <experiment>/
└── scenario_2/
    └── <experiment>/
```

例如：

```text
data/raw/scenario_2/experiment_2026_0816/
```

其中：

```text
scenario = 科研场景
experiment = 某次具体采集批次
```

二者不得混为同一级概念。

`data/raw/` 默认只读。除非用户明确要求进行数据目录架构迁移或原始数据维护，否则不得修改、覆盖、删除、重命名或移动原始实验文件。

### 5.2 可复用处理中间数据

使用：

```text
data/processed/
├── scenario_1/
└── scenario_2/
```

`data/processed/` 不建立 `cross_scenario/`。跨场景任务的专属中间产物应写入对应任务 `results/`；只有真正可被多个后续任务复用的数据才进入 `data/processed/`。

### 5.3 数据完整性

涉及 `data/raw/` 的显式授权迁移时，必须以实验目录内部文件树、文件数量、字节数和 per-file SHA256（逐文件 SHA256）验证内容未变化。

如果目录层级变化导致全局 manifest hash（清单哈希）变化，不得误判为原始科学数据发生变化；必须区分：

```text
global path-sensitive manifest
experiment-content manifest
```

当前历史 Scenario 2 数据的正式路径为：

```text
data/raw/scenario_2/experiment_2026_0816/
```

后续新增实验必须继续遵守：

```text
data/raw/<scenario>/<experiment>/
```

### 5.4 历史 raw manifest 兼容层

工程可以在：

```text
scripts/core/shared/project_paths.py
```

中保留类似：

```text
legacy_raw_manifest()
```

的 compatibility layer（兼容层），用于识别架构迁移前已经写入旧 checkpoint / cache / scientific fingerprint 的历史 raw manifest。

该兼容接口只服务于**历史科研状态恢复与验证**，不得被解释为旧目录仍然存在，也不得作为新任务的数据路径规范。

新任务和新的科学 fingerprint 应优先使用当前 canonical（规范）场景化数据身份：

```text
scenario_scope
experiment_id
data/raw/<scenario>/<experiment>/
```

不得为了让新任务复用旧 hash 而人为抹去新的场景层语义。

---

# 一、模块内部代码组织

## 1. `shared/` 与 task 的硬边界

`shared/` 表示模块稳定、可复用的公共 API。

适合放入 `shared/` 的内容包括：

- 多个任务复用的代码；
- 其他模块需要调用的稳定能力；
- 公共 loader、solver、model、basis dictionary；
- 公共 metric、signal operator、fingerprint builder；
- clustering、ranking、validation helper；
- 模块级数据结构与配置对象；
- 删除任何一个具体任务后仍具有长期价值的实现。

任务目录只保存任务专属代码，例如：

- `run_*.py`；
- 任务专属 backend；
- 任务专属 plot/exporter；
- 任务专属 validator/test；
- checkpoint/resume；
- 只服务该实验或扫描的实现。

推荐依赖：

```text
Task -> Shared
Module A Task -> Module B Shared
```

原则上禁止：

```text
Task -> Task
Shared -> Task
Module A -> Module B 的具体 Task
```

如果某个任务中的实现被第二个任务长期复用，应先把真正公共部分提升到所属模块 `shared/`。

## 2. 模块根目录

普通科研模块根目录原则上只允许：

```text
__init__.py
shared/
scenario_1/
scenario_2/
cross_scenario/
__pycache__/
```

以及经过明确识别的工程级例外。

不得把：

```text
run_xxx.py
backend.py
plot_xxx.py
export_xxx.mjs
validation.py
```

散落在模块根目录。

`retrieval_oriented_model_selection` 根目录只允许：

```text
__init__.py
shared/
self_hit_oriented/
dpd_shareability_oriented/
__pycache__/
```

两个 route 下只能继续进入场景层，不允许平铺新 task。

## 3. 测试归属

模块公共代码测试：

```text
scripts/<module>/shared/tests/
```

普通任务专属测试：

```text
scripts/<module>/<scenario_scope>/<task>/tests/
```

或任务目录内的 `test_*.py`。

Retrieval 任务专属测试：

```text
scripts/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/tests/
```

不得把公共测试散落在模块根目录。

---

# 二、10 个模块的职责边界

## 1. `core`

负责：

- 工程路径；
- 模块注册表；
- 通用 IO；
- 通用日志；
- 通用 validator；
- 跨领域基础 metric/signal utility；
- 工程架构维护；
- Codex 工程交接。

总体交接固定：

```text
work_logs/core/codex_handoff/codex_handoff.txt
```

## 2. `data_management`

负责：

- raw/processed 路径解析；
- experiment 与 scenario 定位；
- MAT 加载；
- state index；
- canonical state ordering；
- 文件索引；
- 数据访问；
- manifest。

## 3. `signal_segmentation`

负责：

- A/B/C 分段；
- valid sample；
- `dmax` 边界；
- 公共 B 段；
- 样点归属；
- 与分段直接相关的规范预处理和验证。

## 4. `pa_performance_evaluation`

负责：

- NMSE；
- ACPR；
- 输出功率；
- DC power；
- efficiency；
- DPD on/off；
- 实测 PA 性能统计与比较。

## 5. `behavior_modeling`

负责 PA 正向行为建模以及**建模精度/泛化精度导向**的模型选择，包括：

- MP / Envelope / GMP / Volterra 等基函数；
- OLS / Ridge；
- 模型训练与预测；
- Train / Validation / Generalization NMSE；
- Generalization Gap；
- model capacity；
- numerical conditioning；
- 以建模误差和泛化为目标的 basis/model/Ridge 选择。

该模块回答：

> 什么样的行为模型能够更准确地描述和泛化 PA 行为？

## 6. `behavior_fingerprint_retrieval`

负责在模型结构已经确定后执行 LUT 检索，包括：

- common-B 响应；
- behavior fingerprint；
- LUT 构建；
- CNMSE distance；
- Top-1 / Top-k；
- Real-B 后验验证；
- Self / Fallback / Fail；
- DPD shareability 验证；
- 检索结果统计。

该模块不负责搜索最优模型结构。

## 7. `retrieval_oriented_model_selection`

负责**LUT 检索效果导向**的模型结构与 Ridge 选择，例如：

- Forward Add；
- Backward Delete；
- Beam/Floating/Swap；
- Retrieval-oriented Ridge；
- Joint support + Ridge；
- Final Dense Ridge；
- retrieval-oriented validation；
- `N_self`、`N_shareable`、margin、MRR、Top-k 等检索目标。

该模块回答：

> 什么样的行为模型最适合 LUT 检索？

必须与 `behavior_modeling` 严格区分。

## 8. `behavior_fingerprint_ranking_consistency`

负责：

- ranking consistency；
- 不同带宽排序保持；
- 不同模型距离排序一致性；
- Query/LUT ranking；
- 跨状态一致性分布；
- Spearman/Kendall 等排序指标。

## 9. `low_bandwidth_behavior_analysis`

负责：

- `5B -> nB`；
- 低带宽观测；
- 低带宽行为指纹；
- 低带宽与完整带宽关系；
- 跨带宽行为保持性；
- 低带宽检索可行性。

## 10. `lut_clustering_compression`

负责：

- 状态聚类；
- Complete-Link；
- representative selection；
- LUT 压缩；
- behavior fingerprint 压缩；
- DPD 条目压缩；
- Type III 等压缩研究。

---

# 三、工作模式

## 1. 新任务模式

满足以下条件时使用新任务模式：

> 会产生新的、可以独立解释、独立保存和独立复现的分析、实验或研究结果。

例如：

- 新的数据分析；
- 新实验；
- 新参数扫描；
- 新模型训练；
- 新评价指标分析；
- 新独立科研问题。

创建前依次确定：

### 普通科研任务

```text
module_name
scenario_scope
task_name
```

### Retrieval 任务

```text
module_name
route_name
scenario_scope
task_name
```

`task_name` 使用小写英文 `snake_case`，必须有实际语义。

新任务在已有场景目录内时，原则上**不要重复添加** `scenario_1_` 或 `scenario_2_` 前缀，因为场景已经由目录表达。

例如新任务优先：

```text
behavior_modeling/scenario_2/unified_model_capacity_scan/
```

而不是：

```text
behavior_modeling/scenario_2/scenario_2_unified_model_capacity_scan/
```

历史任务已有 `scenario_2_` 前缀时保持原名，不进行无必要的批量重命名。

## 2. 持续维护模式

如果只是：

- 修 Bug；
- 修改已有任务配置；
- 优化实现但科研问题不变；
- 补充已有图表；
- 增加测试；
- 更新公共代码；
- 整理已有任务文档；

则使用持续维护模式。

不得为了每次维护创建新任务。

详细记录追加到原任务：

```text
work_logs/<module>/<scenario_scope>/<task>/execution_log.txt
```

Retrieval：

```text
work_logs/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/execution_log.txt
```

## 3. Cross-scenario 判定

只有任务的科研问题本身明确同时涉及两个场景时才使用：

```text
cross_scenario
```

例如：

- Scenario 1 与 Scenario 2 的直接比较；
- 用 Scenario 2 数据为 Scenario 1 选择状态；
- 两个场景共享模型/指标的迁移分析。

仅仅 import 了另一个场景的公共代码，不构成 `cross_scenario`。

---

# 四、新任务标准流程

1. 判断是否真的是新任务；
2. 根据主要科研目标确定唯一 `module_name`；
3. 确定 `scenario_scope`；
4. 如果是 Retrieval，再确定唯一 `route_name`；
5. 检查同 scope 下是否已有同研究问题任务；
6. 优先复用本模块和其他模块 `shared/`；
7. 创建实际需要的 task 目录；
8. 实现代码，不形成 task-to-task 依赖；
9. 结果写入对应 `results`；
10. 详细执行过程写入对应 `work_logs`；
11. 运行 Ruff、相关测试/smoke test；
12. 追加 `core/codex_handoff` 摘要。

普通需要脚本的任务：

```text
scripts/<module>/<scenario_scope>/<task>/
results/<module>/<scenario_scope>/<task>/
work_logs/<module>/<scenario_scope>/<task>/
```

Retrieval：

```text
scripts/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/
results/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/
work_logs/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/
```

如果没有独立脚本，只创建实际需要的 results/work_logs；不得为了形式完整创建空脚本任务目录。

---

# 五、结果、日志与 checkpoint

## 1. `results/`

保存正式科研结果和任务对外使用的稳定产物，例如：

- CSV/Excel；
- JSON summary；
- NPZ/MAT 模型；
- figures；
- PDF；
- final metrics；
- final model definition；
- 任务正式 checkpoint（如果该任务设计如此）。

## 2. `work_logs/`

不仅保存文本日志，也可以保存**任务执行和预搜索所必需的工程中间状态**，例如：

- `execution_log.txt`；
- pre-search validation；
- runtime diagnostics；
- screening partitions；
- CandidateScore cache；
- checkpoint/resume metadata；
- engineering audit；
- task-specific validation artifact。

因此不得把 `work_logs/` 简单理解为“可以随意清理的日志”。

特别是包含：

```text
checkpoint/
candidate_score_cache/
screening/
runtime/
pre_search_validation/
```

的任务目录，在确认科研任务不再需要恢复前不得删除、压缩替换或批量清理。

## 3. 日志规则

任务详细记录：

```text
work_logs/<module>/<scenario_scope>/<task>/execution_log.txt
```

Retrieval：

```text
work_logs/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/execution_log.txt
```

工程级交接：

```text
work_logs/core/codex_handoff/codex_handoff.txt
```

历史日志中的旧路径属于 provenance（来源记录），目录迁移后原则上不批量改写。

## 4. 架构迁移后的历史 artifact 兼容

目录架构变化后，历史 checkpoint、cache、JSON/JSONL、summary 或 execution log 中保存的旧路径原则上保持原内容。

如果旧路径只是：

```text
history
provenance
informational metadata
```

不得为了“统一路径”而批量重写。

如果旧路径会被 active loader（活动加载逻辑）实际用于 resume/read，应优先通过路径兼容层或 relocation resolver（重定位解析器）处理，而不是直接修改历史科研 artifact。

架构维护不得改变：

```text
scientific fingerprint
CandidateScore
support
Ridge lambda
distance matrix
checkpoint scientific state
```

仅允许修复文件系统定位。

---

# 六、Python 路径规则

## 1. 禁止机器绝对路径作为工程逻辑

不得在新的活动源码中硬编码：

```text
/Users/<name>/...
C:\Users\<name>\...
```

工程路径应由：

```text
scripts/core/shared/project_paths.py
```

或模块正式路径 helper 统一解析。

## 2. 场景与任务路径

活动源码应显式携带：

```text
scenario_scope
experiment_id（数据任务需要时）
```

不得仅靠任务名字符串猜测场景。

当前工程统一路径接口位于：

```text
scripts/core/shared/project_paths.py
```

优先使用其正式接口，例如：

```text
get_raw_scenario_root("scenario_1")
get_raw_scenario_root("scenario_2")
get_raw_experiment_root("scenario_2", "experiment_2026_0816")
get_processed_scenario_root("scenario_1")
get_processed_scenario_root("scenario_2")

get_task_paths(
    module_name,
    task_name,
    scenario_scope="scenario_2",
)
```

新代码不得重新通过机器绝对路径或固定 `.parents[n]` 层数复制一套路径逻辑。

`legacy_raw_manifest()` 仅用于历史 checkpoint/cache 兼容，不是新任务的数据路径 API。

## 3. Retrieval route 路径

Retrieval 路径应显式携带：

```text
route_name
scenario_scope
task_name
```

优先使用：

```text
scripts/retrieval_oriented_model_selection/shared/route_paths.py
```

当前正式接口语义为：

```text
get_route_task_paths(
    route,
    scenario_scope,
    task_name,
)
```

其结果应解析到：

```text
scripts/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/
results/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/
work_logs/retrieval_oriented_model_selection/<route>/<scenario_scope>/<task>/
```

不得用固定 `.parents[n]` 推断 route/module。

---

# 七、工程 validator 要求

正式 layout validator 至少必须检查：

1. 一级模块严格为固定 10 个；
2. `Modules(scripts) = Modules(results) = Modules(work_logs)`；
3. 普通科研模块只允许合法 scenario scope；
4. 普通模块 `shared/` 只位于模块根部；
5. `results/`、`work_logs/` 不存在模块公共 `shared/`；
6. Retrieval route 集合严格为两个正式 route；
7. Retrieval task 必须位于 `<route>/<scenario_scope>/<task>`；
8. route 下和 route/scenario 下不存在 `shared/`；
9. 模块根不存在平铺 Retrieval task；
10. `data/raw/`、`data/processed/` 场景目录只使用 `scenario_1`、`scenario_2`；
11. 不存在 `task -> task`、`shared -> task` 和跨模块 concrete-task 依赖；
12. 普通模块根目录没有散落业务源文件；
13. 架构维护完成后，active source（活动源码）不应继续依赖迁移前的旧 raw/retrieval 目录；
14. 历史日志、checkpoint provenance 和旧 summary 中保留旧路径不属于 layout violation。

历史 task 名中保留 `scenario_2_` 前缀不属于 layout violation。

---

# 八、科研计算资源规则

正式 CPU-heavy（CPU 密集型）科研搜索默认：

```text
10 个 spawn worker processes
每 worker 固定 1 个 BLAS/OpenBLAS thread
不设置 CPU affinity
GPU 不参与
禁止 worker 内嵌套 multiprocessing pool
```

实际物理核由操作系统调度。

调试、reference gate、synthetic validation 可以显式使用 1 或 2 worker。

worker 数属于运行配置，不属于科研 candidate ID 或科学 fingerprint，除非某任务明确另有定义。

正式搜索或长时间计算必须支持：

- checkpoint/resume；
- worker error 监控；
- fallback 统计；
- progress/throughput/ETA；
- 安全停止和恢复。

未经用户明确要求，不得因为完成了预搜索验证而自动启动下一阶段正式搜索。

---

# 九、Python 环境与依赖

## 1. Single Source of Truth

根目录：

```text
environment.yml
```

是 Conda/Python 环境唯一事实来源。

`AGENTS.md` 和 `README.md` 只描述原则，不维护平行依赖清单。

## 2. 正式环境

必须使用工程受控环境执行正式科研任务。

不得默认使用：

- Conda `base`；
- 系统 Python；
- 用户全局 Python；
- 其他工程环境；
- 临时 `.venv`。

当前工程以 Python 3.11、conda-forge 和 OpenBLAS 为正式环境约束，具体声明以 `environment.yml` 为准。

在 Apple Silicon Mac 上应使用原生 arm64 Conda/Python 环境；除非明确依赖只能通过 Rosetta/x86_64 运行且用户确认，否则不要使用 Rosetta Python。

## 3. 环境创建与同步

首次：

```bash
conda env create -f environment.yml
```

已有环境：

```bash
conda env update -f environment.yml
```

默认不使用 `--prune`。

不得无依据执行：

```bash
conda update --all
```

## 4. 新依赖

优先 conda-forge。

只把工程直接依赖写入 `environment.yml`。

pip 仅作为 Conda 无法满足需求时的受控例外，并必须写入 `environment.yml` 的 `pip:` 子项。

---

# 十、Git 安全规则

Codex 必须优先保护用户已有工作。

除用户明确要求外，禁止：

```text
git reset --hard
git clean -fd
git restore 覆盖用户修改
git checkout 覆盖用户修改
git stash
git add -A
git add .
强制 push
改写远程历史
```

允许只读检查：

```text
git status
git diff
git log
git ls-files
```

如果用户要求提交，只能选择性暂存本次明确产生的文件，不得把脏工作区整体加入提交。

`.gitignore` 不会自动取消历史已跟踪文件。涉及 `git rm --cached`、取消跟踪、删除或历史整理时必须获得用户明确授权。

---

# 十一、缓存与清理规则

可以在没有科研任务运行且用户授权清理时删除普通运行缓存：

```text
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.DS_Store
```

但绝对不得把科研 cache 当作普通运行缓存删除，例如：

```text
results/**/_cache/
work_logs/**/candidate_score_cache/
work_logs/**/checkpoint/
work_logs/**/screening/
work_logs/**/runtime/
distance_matrices/
fingerprints/
科学 .npy/.npz/.json/.jsonl
```

删除任何不确定目录前，先判断其是否承担 checkpoint/resume、科学结果或 provenance 职责。

---

# 十二、完成前检查

## 新任务

至少确认：

1. module 正确；
2. scenario scope 正确；
3. Retrieval route（如适用）正确；
4. task name 语义明确；
5. 没有无必要创建新任务；
6. 没有 task-to-task 依赖；
7. 公共能力已优先复用 `shared/`；
8. 结果和日志写入正确 scope；
9. 未未经授权修改 raw；
10. Ruff 通过；
11. 相关测试/smoke test 通过或失败原因已明确记录；
12. `codex_handoff` 已追加摘要。

## 架构维护

还必须确认：

```text
Modules(scripts) = Modules(results) = Modules(work_logs)
```

以及：

- 一级模块恰好为固定 10 个；
- 普通科研 task 位于 scenario scope 下；
- Retrieval task 位于 route/scenario scope 下；
- 模块级 `shared/` 唯一；
- results/work_logs 不存在模块公共 shared；
- data/raw 与 data/processed 的 scenario 结构合法；
- 活动源码不存在旧路径依赖；
- 科研 result/checkpoint/cache 未因架构维护被重算或覆盖；
- 原始数据内容完整；
- Git 没有被破坏性操作。

## 测试分层与历史 artifact

工程测试应区分：

```text
pure unit / path / import tests
artifact-dependent regression tests
full scientific execution tests
```

普通工程维护默认优先运行前一类以及当前修改直接影响的测试。

如果历史 regression test（回归测试）依赖迁移前本来就不存在的科研产物，例如某个 CSV/JSON/model artifact：

- 应明确记录缺失原因；
- 不得为了让测试变绿而重新运行无关科研任务；
- 不得从其他任务复制结果；
- 不得创建伪造 artifact；
- 不得修改科研判定逻辑来掩盖缺失。

完整 `pytest` 是否作为硬门槛，应根据本次任务范围决定；静态检查、import smoke test（导入冒烟测试）、layout validator 和直接受影响测试可以作为工程维护的主要门槛。

---

# 十三、核心原则

Codex 在本工程中始终遵守：

1. 先判断工作模式；
2. 先确定模块，再确定场景，再确定任务；
3. Retrieval 任务额外先确定 route；
4. 固定 10 模块，未经批准不新增；
5. 普通科研架构使用 `Module -> Scenario -> Task`；
6. Retrieval 使用 `Module -> Route -> Scenario -> Task`；
7. 数据使用 `Data -> Scenario -> Experiment`；
8. 公共代码始终位于模块级 `shared/`；
9. 新 task 名原则上不重复场景前缀，历史 task 名不强制改名；
10. 任务按主要科研目标归属，不按 import 关系归属；
11. `behavior_modeling` 与 `retrieval_oriented_model_selection` 严格分离；
12. `cross_scenario` 只用于真正跨两个场景的科研任务；
13. 原始数据默认只读；
14. 可复用处理中间数据进入 `data/processed/<scenario>/`；
15. 工作日志中可能包含重要科研 checkpoint/cache，不能机械清理；
16. 不创建无意义任务级空目录；
17. 不覆盖用户已有工作；
18. 不擅自扩大任务范围；
19. 正式 CPU-heavy 搜索遵守 10 spawn worker × 1 BLAS thread 的默认执行约束；
20. 保持工程简单、清晰、可复现、可恢复、可验证。
