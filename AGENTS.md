# Codex 工程执行规则

## 适用范围

本文件用于由 `PROJECT_TEMPLATE.txt` 创建或整理的标准四目录 Python 工程。

当工程根目录中存在以下业务目录时，Codex 每次执行任务都必须遵守本规则：

```text
data/
results/
scripts/
work_logs/
```

如果本文件位于“Python工程模板配置”目录中，则它是供新工程复制的规则源文件，不要求模板配置目录本身建立四个业务目录。

本规则只定义两种工作模式：

1. **新任务模式**
2. **持续维护模式**

Codex 每次收到请求后，必须先判断本次工作属于哪一种模式，再执行后续操作。

---

## 一、工作模式判定

### 1. 核心判定原则

模式判定的主要依据是：

> **本次工作是否产生新的、可以独立解释、独立保存和独立复现的分析、实验或研究结果。**

模式判定不以以下事项作为主要依据：

- 是否修改了已有代码；
- 是否运行了 Python；
- 是否新建了文件；
- 是否生成了临时数据；
- 是否修改了配置；
- 是否新增了测试文件；
- 是否新增了公共模块；
- 是否重新执行了已有脚本。

### 2. 用户明确指定优先

如果用户明确指定：

- “这是一个新任务”；
- “按新任务模式处理”；
- “这只是维护”；
- “不要新建任务目录”；
- “按持续维护模式处理”；

则优先遵循用户指定。

如果用户指定的模式与工程完整性存在明显冲突，应说明原因，并尽量按照用户意图执行。

### 3. 默认判定示例

| 工作内容 | 默认模式 |
|---|---|
| 新的数据分析 | 新任务模式 |
| 新实验 | 新任务模式 |
| 新模型训练或拟合 | 新任务模式 |
| 新参数扫描 | 新任务模式 |
| 新评价指标分析 | 新任务模式 |
| 针对新的科研问题生成图形 | 新任务模式 |
| 基于已有数据回答一个新的独立分析问题 | 新任务模式 |
| 修复已有任务脚本 Bug | 持续维护模式 |
| 修改已有任务配置文件 | 持续维护模式 |
| 修改已有固定任务脚本 | 持续维护模式 |
| 优化已有任务脚本实现但不改变研究任务 | 持续维护模式 |
| 修改已有公共 `_core` 模块 | 持续维护模式 |
| 为已有任务增加单元测试 | 持续维护模式 |
| 增加工程工具或配置辅助代码 | 持续维护模式 |
| 持续整理已有任务相关文档 | 持续维护模式 |

---

## 二、新任务模式

### 1. 适用范围

新任务模式适用于新的分析问题、实验、数据处理、模型拟合、参数扫描、绘图、评价或其他会形成独立研究结果的工作。

新任务**不使用任务编号**。

每个新任务使用一个简短、明确、具有实际语义的小写英文 `snake_case` 名称：

```text
task_name
```

例如：

```text
fit_behavior_model
cross_bandwidth_consistency
lut_retrieval_evaluation
load_mismatch_classification
generate_paper_figures
```

不得使用缺乏语义的名称，例如：

```text
task1
task2
test
test1
new_task
tmp
temp
run1
```

### 2. 使用脚本的新任务

如果本次任务需要创建或使用独立任务脚本，应建立：

```text
scripts/task_name/
results/task_name/
work_logs/task_name/
```

例如：

```text
scripts/fit_behavior_model/
results/fit_behavior_model/
work_logs/fit_behavior_model/
```

同一个任务实际存在的任务子目录，其 `task_name` 必须完全一致。

### 3. 不使用脚本的新任务

如果本次任务不需要脚本，但会产生人工结果、仪器结果、第三方软件结果或其他独立任务产出，则只建立：

```text
results/task_name/
work_logs/task_name/
```

不得为了保持形式完整而建立空的：

```text
scripts/task_name/
```

### 4. 已存在同名任务时的处理

准备创建新任务目录前，必须检查是否已经存在同名任务。

如果：

- 本次工作属于原任务的继续执行、补充分析、结果完善或同一研究问题的后续处理，则直接复用原任务目录；
- 本次工作属于新的独立研究问题，则重新选择能够清楚区分两者的语义化任务名称；
- 不得覆盖、清空或重置已有任务目录中的结果与日志；
- 不得仅通过增加无意义后缀规避命名冲突，例如 `task_new`、`task_new2`、`task_final2`。

如果确实需要区分相近任务，应让名称体现研究差异，例如：

```text
behavior_model_off
behavior_model_stale_dpd
behavior_model_bandwidth_ablation
```

而不是：

```text
behavior_model_1
behavior_model_2
behavior_model_new
```

### 5. 新任务模式执行过程中修改已有固定文件

新任务可能依赖已有配置、共享模块或固定文件。

如果为了完成新任务，需要修改已有固定文件：

- 仅修改完成该新任务确实必要的部分；
- 不复制固定文件到 `results` 作为修改历史；
- 对与本任务直接相关的修改记录在本任务 `work_logs/task_name/execution_log.txt` 中；
- 不得借新任务之名进行无关的大范围工程重构。

---

## 三、持续维护模式

### 1. 适用范围

持续维护模式适用于对**已有任务**进行反复修改、修复、完善或整理，并且本次工作不形成新的独立研究结果。

典型情况包括：

- 修复已有任务脚本 Bug；
- 修改已有任务配置；
- 调整已有算法实现但研究目标不变；
- 完善已有任务图形或输出格式；
- 增加已有任务测试；
- 更新已有任务依赖的公共模块；
- 对已有任务文档进行持续整理。

如果用户没有明确要求创建新任务，而请求只是继续修改已经指定的已有任务，默认使用持续维护模式。

### 2. 持续维护模式规则

持续维护模式必须遵守：

1. 不创建新的 `scripts/task_name/`、`results/task_name/` 或 `work_logs/task_name/`；
2. 继续复用原任务已有目录；
3. 固定文件直接在其原始路径修改；
4. 不为了保存修改历史而复制固定文件到 `results`；
5. 不为每次修改建立时间戳副本；
6. 不创建无实际用途的占位目录、占位脚本或占位结果文件；
7. 所有与该任务相关的维护记录，直接追加到原任务的：

```text
work_logs/task_name/execution_log.txt
```

8. 每次维护完成后，在：

```text
work_logs/codex_handoff/codex_handoff.txt
```

中追加简短摘要。

### 3. 持续维护记录原则

持续维护不单独建立维护日志目录。

一个任务从首次建立到后续维护的完整历史，都保存在同一个：

```text
work_logs/task_name/execution_log.txt
```

中。

例如：

```text
scripts/fit_behavior_model/
results/fit_behavior_model/
work_logs/fit_behavior_model/execution_log.txt
```

后续如果修复 `fit_behavior_model` 的 Bug、调整参数、完善图形或修改结果输出，仍继续向：

```text
work_logs/fit_behavior_model/execution_log.txt
```

追加记录。

这样可以保证：

> **一个任务的首次执行、补充分析、修复、调整和结果完善全部在同一个日志中连续追踪。**

### 4. 持续维护过程中允许新增的文件

以下文件即使是新建，也仍然可以属于持续维护模式：

- 已有任务需要的测试文件；
- 已有任务需要的辅助脚本；
- 已有任务需要的配置文件；
- 跨任务共享模块；
- `scripts/_core/` 下的公共代码；
- 工程检查脚本；
- 路径管理工具；
- 日志工具；
- 通用数据结构；
- 开发辅助工具。

这些文件的共同特点是：

> 它们用于维护已有任务或工程公共能力，本身不构成新的独立研究结果。

### 5. 公共模块维护记录归属

如果修改 `scripts/_core/` 等公共模块是为了完成某个明确任务的维护，则维护记录写入该任务的：

```text
work_logs/task_name/execution_log.txt
```

如果同一次公共模块修改同时服务多个已有任务：

- 在主要受影响任务的 `execution_log.txt` 中记录详细过程；
- 在其他受影响任务日志中追加简短关联说明；
- 在 `codex_handoff.txt` 中说明该公共修改影响的任务范围。

### 6. 工程级文件维护

`AGENTS.md`、`README.md`、`environment.yml`、`.gitignore` 等工程级文件通常不属于某个具体研究任务。

如果工程级文件的修改明显由某个已有任务触发并服务于该任务，则记录到对应：

```text
work_logs/task_name/execution_log.txt
```

如果工程级维护无法合理归属任何具体任务，则：

- 不新建虚构任务目录；
- 直接在 `work_logs/codex_handoff/codex_handoff.txt` 中记录本次维护内容、修改原因和验证结果。

### 7. 从持续维护切换到新任务模式

如果持续维护过程中发现本次工作已经演变为新的独立分析、实验、模型、参数扫描或结果生成任务，则从该部分开始切换到新任务模式。

切换时：

1. 已完成的原任务维护继续保留在原任务 `execution_log.txt` 中；
2. 新的独立研究工作使用新的语义化 `task_name`；
3. 从产生独立任务脚本、分析过程或结果的时点开始建立对应任务目录；
4. 不重复复制之前已经修改的固定文件；
5. 在新任务日志中注明其依赖的已有任务或公共模块；
6. 新增公共模块、测试代码、工程工具或配置辅助文件本身不触发模式切换。

---

## 四、目录职责

### 1. `scripts`

任务级脚本必须写入：

```text
scripts/task_name/
```

任务脚本不得直接散落在 `scripts` 根目录。

推荐结构：

```text
scripts/
├── _core/
├── fit_behavior_model/
├── cross_bandwidth_consistency/
└── lut_retrieval_evaluation/
```

### 2. `scripts/_core`

`scripts/_core/` 用于跨任务复用的稳定公共模块。

适合放置：

- 配置读取；
- 数据加载；
- 信号处理公共函数；
- 行为模型公共实现；
- 距离计算；
- 绘图公共函数；
- 日志工具；
- 路径工具；
- 通用数据结构；
- 多个任务共同使用的算法基础模块。

不得把一次性实验代码、单个任务专用入口脚本或临时调试代码放入 `_core`。

任务级入口脚本仍应保留在对应：

```text
scripts/task_name/
```

中。

### 3. `results`

新任务产生的最终结果必须写入：

```text
results/task_name/
```

结果包括但不限于：

- 表格；
- CSV；
- JSON；
- MAT；
- NPZ；
- 模型文件；
- 指标结果；
- 图形；
- PDF；
- 人工整理后的最终文件；
- 仪器导出结果；
- 第三方软件导出结果。

结果不得直接散落在 `results` 根目录，也不得写入其他无关任务的结果目录。

持续维护已有任务时，如果重新生成或更新任务结果，仍写入原任务的：

```text
results/task_name/
```

不得仅因为维护而创建新的结果目录。

### 4. `work_logs`

每个任务的完整执行历史写入：

```text
work_logs/task_name/execution_log.txt
```

该日志同时包含：

- 首次执行记录；
- 后续补充分析；
- Bug 修复；
- 参数调整；
- 结果完善；
- 其他与该任务直接相关的持续维护记录。

总体工程交接和任务摘要写入：

```text
work_logs/codex_handoff/codex_handoff.txt
```

不再设置独立的：

```text
work_logs/ongoing_file_maintenance/
```

---

## 五、数据目录规则

### 1. `data/raw`

`data/raw/` 用于保存原始输入数据、原始实验数据、原始采集数据和不可替代的参考输入。

默认规则：

> **`data/raw/` 视为只读目录。**

除非用户明确要求，否则不得：

- 修改；
- 覆盖；
- 删除；
- 重命名；
- 移动；

`data/raw/` 中的任何原始文件。

如果需要清洗、转换、重采样、裁剪、对齐、滤波或重新组织原始数据，应将新数据写入：

```text
data/processed/
```

或对应任务的结果目录，而不是覆盖原始数据。

### 2. `data/processed`

`data/processed/` 用于保存：

> 可以重复生成、并且可能被多个后续任务复用的中间数据或标准化数据。

例如：

- 统一格式后的数据；
- 对齐后的公共数据集；
- 重采样后的公共数据；
- 清洗后的标准数据；
- 公共特征数据；
- 后续多个任务都会复用的预处理输出。

### 3. `results/task_name`

`results/task_name/` 用于保存：

> 与当前具体研究任务直接对应的最终分析结果或任务专属中间结果。

如果某个中间数据只服务于当前任务、不需要被其他任务复用，可以直接保存在：

```text
results/task_name/
```

中。

---

## 六、结果与文件创建原则

Codex 不得为了满足目录形式而创建没有实际内容的文件或目录。

禁止：

- 创建空 `scripts/task_name/` 仅为了和其他目录对应；
- 创建 `placeholder.py`；
- 创建无内容的结果文件；
- 创建没有实际用途的 README 作为占位；
- 为同一个固定文件建立大量时间戳副本；
- 为了“保险”而重复复制整个任务目录。

只有在文件或目录具有明确用途时才创建。

---

## 七、执行记录规则

### 1. 任务详细执行记录

每个任务的：

```text
work_logs/task_name/execution_log.txt
```

记录该任务从首次执行到后续持续维护的完整历史。

#### 首次执行至少记录

- 时间；
- 工作目标；
- 任务名称；
- 任务类型；
- 实际执行内容；
- 新建、移动、修改或删除的文件；
- 关键输入与输出路径；
- 验证结果；
- 当前状态。

#### 后续持续维护至少记录

- 时间；
- 维护目的；
- 维护对象；
- 实际修改；
- 验证结果；
- 当前状态。

#### 按需补充

根据任务实际情况补充：

- 关键命令；
- 参数设置；
- 数据来源；
- 数据规模；
- 模型配置；
- 评价指标；
- 非脚本结果的生成方式；
- 使用的第三方工具；
- 可复现性说明；
- 遇到的问题；
- 问题处理方式；
- 未完成事项；
- 下一步建议。

不得为了填满日志模板而写大量“无”“无问题”“不适用”等无意义内容。

### 2. 总体执行记录

`work_logs/codex_handoff/codex_handoff.txt` 用于持续汇总工程状态。

新任务完成后，追加：

- 任务名称；
- 任务状态；
- 主要操作；
- 主要成果；
- 关键验证结论；
- 未完成事项；
- 后续建议。

已有任务完成一次持续维护后，追加：

- 任务名称；
- 本次维护内容；
- 主要修改；
- 验证结果；
- 是否还有未完成事项。

无法归属具体任务的工程级维护，也记录在这里，并明确标记为工程级维护。

总体记录只写摘要。

详细过程原则上保留在对应任务的 `execution_log.txt` 中。

`codex_handoff.txt` 不记录任何：

- 任务编号；
- 下一个任务序号；
- 已使用编号；
- 编号缺口。

---

## 八、Git 安全规则

如果工程启用了 Git，Codex 必须优先保护用户已有工作。

### 允许

- 查看状态；
- 查看 diff；
- 按任务或维护内容进行小批次提交；
- 提交自己明确产生的修改；
- 在必要时创建便于回滚的普通提交。

### 禁止

除非用户明确要求，否则不得：

- `git reset --hard`；
- `git clean -fd`；
- 强制 checkout 覆盖用户未提交修改；
- 删除用户未跟踪文件；
- 覆盖用户修改；
- 强制推送；
- 改写远程历史；
- 大范围恢复到旧版本；
- 为了让工作区“干净”而删除无法确认来源的文件。

如果发现工作区存在用户已有未提交修改：

1. 先检查差异；
2. 避免覆盖；
3. 只修改本次任务必要的文件；
4. 在日志中记录相关情况；
5. 如果无法安全区分用户修改与当前任务修改，应停止高风险 Git 操作，但可以继续进行不破坏现有内容的工作。

---

## 九、任务复用与边界控制

### 1. 优先复用已有公共能力

如果已有：

```text
scripts/_core/
```

中的模块能够完成需求，应优先复用，不得在不同任务目录中重复复制相同实现。

### 2. 不随意扩大任务范围

执行新任务或持续维护时，不得自行顺带进行与用户请求无关的大规模：

- 重构；
- 重命名；
- 目录迁移；
- 格式统一；
- API 改写；
- 依赖升级。

确有必要时，应将变更限制在完成当前目标所需的最小范围。

### 3. 临时调试文件

临时调试文件应尽量避免。

如确需创建，应：

- 使用明确的临时名称；
- 不写入 `data/raw/`；
- 完成验证后删除无保留价值的临时文件；
- 不把临时文件作为正式任务结果。

---

## 十、完成前检查

Codex 完成每项工作前必须进行检查。

### 新任务模式

确认：

1. `task_name` 简短、明确、具有实际语义；
2. 如果任务使用脚本，脚本位于：

```text
scripts/task_name/
```

3. 如果任务不使用脚本，没有为了形式完整创建空脚本目录；
4. 最终结果位于：

```text
results/task_name/
```

5. 详细记录位于：

```text
work_logs/task_name/execution_log.txt
```

6. 同一任务实际存在的 `scripts`、`results`、`work_logs` 子目录名称一致；
7. 没有覆盖其他任务已有结果；
8. 没有修改或覆盖 `data/raw/` 原始数据；
9. 总体摘要已经追加到：

```text
work_logs/codex_handoff/codex_handoff.txt
```

10. 如果启用了 Git，已确认没有破坏用户已有修改。

### 持续维护模式

确认：

1. 没有无必要创建新的任务目录；
2. 继续复用了正确的原任务目录；
3. 固定文件在原路径完成修改；
4. 更新后的任务结果仍位于原任务 `results/task_name/`；
5. 本次维护记录已经追加到原任务：

```text
work_logs/task_name/execution_log.txt
```

6. 日志包含本次维护目的、实际修改和验证结果；
7. 总体维护摘要已经追加到：

```text
work_logs/codex_handoff/codex_handoff.txt
```

8. 没有修改或覆盖 `data/raw/` 原始数据；
9. 如果启用了 Git，已确认没有破坏用户已有修改。

---

## 十一、核心原则

Codex 在本工程中始终遵守以下原则：

1. **先判断模式，再执行工作；**
2. **新任务看“是否产生独立研究结果”，不看是否新建文件；**
3. **持续维护直接归入原任务，不单独建立维护目录；**
4. **一个任务的首次执行与后续维护共享同一个 `execution_log.txt`；**
5. **任务使用语义化名称，不使用任务编号；**
6. **公共能力放入 `_core`，任务代码留在任务目录；**
7. **`data/raw/` 默认只读；**
8. **可复用中间数据进入 `data/processed/`；**
9. **任务最终结果进入对应 `results/task_name/`；**
10. **详细过程写入任务执行日志，总体状态写入 handoff；**
11. **不创建无意义占位文件；**
12. **不覆盖用户已有工作；**
13. **在满足用户目标的前提下，保持工程结构简单、清晰、可复现、可回滚。**

---

## 十二、Python 环境与依赖管理

### 1. 环境配置的唯一事实来源

工程根目录中的：

```text
environment.yml
```

是本工程 Python 运行环境的**唯一事实来源（Single Source of Truth）**。

环境名称、Python 版本、Conda channel、直接依赖以及 BLAS/LAPACK 等数值计算后端约束，均以 `environment.yml` 当前内容为准。

`AGENTS.md` 和 `README.md` 可以说明环境管理原则和使用方法，但不得维护一份与 `environment.yml` 平行的依赖清单。

Codex 在创建环境、安装依赖或修改环境前，必须先读取当前 `environment.yml`，不得根据历史记录、旧电脑路径或已有终端状态猜测环境配置。

### 2. 受控工程环境

正式科研计算必须在本工程受控环境中执行。

当前环境名称由 `environment.yml` 声明。除非用户明确要求，不得：

- 使用 Conda `base` 环境执行正式任务；
- 使用系统 Python 或用户级全局 Python 执行正式任务；
- 为工程额外创建 `.venv/`、`venv/`、`env/` 等第二套 Python 环境；
- 使用其他工程的 Conda 环境；
- 使用来源未确认的 `python`、`pip`、`pytest`、`ruff` 等命令；
- 通过复制 `site-packages` 或手工移动解释器文件的方式迁移环境。

本工程不在规则文件中硬编码 Windows、macOS 或 Linux 的绝对解释器路径。

不得因为更换电脑或操作系统而直接复制旧机器的 Conda 环境目录。应依据 `environment.yml` 在目标机器重新创建环境。

### 3. 创建与同步环境

首次建立环境时使用：

```bash
conda env create -f environment.yml
```

已有环境需要与 `environment.yml` 同步时使用：

```bash
conda env update -f environment.yml
```

不得默认使用 `--prune` 删除环境中的现有包。只有在明确需要清理环境、已经检查删除计划且不会破坏工程时，才可以使用 `--prune`。

不得为了方便执行：

```bash
conda update --all
```

不得无依据地升级 Python、NumPy、SciPy、pandas、Matplotlib 或其他核心依赖。

### 4. 新增依赖

新增依赖属于环境配置变更，必须遵循以下流程：

1. 确认该依赖确实是当前任务所需；
2. 优先确认 `conda-forge` 是否提供兼容版本；
3. 对可能影响 Python、NumPy、SciPy、BLAS/LAPACK 或大量传递依赖的包，先检查 Conda 求解计划；
4. 只把工程直接依赖写入 `environment.yml`；
5. 使用 `conda env update -f environment.yml` 同步工程环境；
6. 完成依赖、代码和工程级验证；
7. 按本章记录规则更新任务日志或 handoff。

当前环境名称必须从 `environment.yml` 的 `name` 字段读取。

以下命令中的 `<environment-name>` 是文档占位符，执行时必须替换为 `environment.yml` 当前声明的环境名称，不得沿用历史名称或凭经验猜测。

需要检查单个 Conda 包的求解计划时，可执行类似：

```bash
conda install --dry-run -n <environment-name> -c conda-forge --strict-channel-priority <package-name>
```

如果环境名称发生修改，所有后续命令必须立即以 `environment.yml` 中的新名称为准。

如果求解计划出现以下情况，应停止安装并报告：

- Python 主版本或次版本发生非预期变化；
- NumPy、SciPy 或其他核心科学计算包发生与当前任务无关的大幅升级或降级；
- `environment.yml` 中声明的数值计算后端被替换；
- 大量现有包被删除；
- 引入未经批准的 channel；
- 出现明显的平台或二进制架构冲突；
- 现有工程代码可能因此失去兼容性。

### 5. 数值计算后端

AGENTS 不单独规定必须使用某一种 BLAS/LAPACK 实现。

如果 `environment.yml` 当前声明了 OpenBLAS、MKL、Accelerate 相关约束或其他数值计算后端，则该约束属于当前工程环境契约。

除非用户明确要求或任务确有必要，Codex 不得擅自更换该后端。

环境变更后应检查：

```bash
conda list -n <environment-name>
```

并在需要时检查 NumPy 实际数值后端信息：

```bash
conda run --no-capture-output -n <environment-name> \
  python -c "import numpy; numpy.__config__.show()"
```

这里的目标不是强制某个平台或 CPU 架构，而是确认实际运行环境与当前 `environment.yml` 一致。

### 6. pip 仅作为受控例外

默认优先通过 Conda/conda-forge 管理依赖。

只有满足以下至少一种情况时，才允许使用 pip：

- conda-forge 不提供该包；
- conda-forge 没有与当前 Python 版本兼容的版本；
- 软件官方明确要求通过 pip 安装；
- 所需功能只存在于官方 pip 发行版；
- 经核验 Conda 包不能满足当前任务需求。

使用 pip 前必须说明：

1. 为什么 Conda/conda-forge 无法满足需求；
2. 准备安装的准确包名和必要版本约束；
3. 是否可能影响现有核心依赖；
4. 安装后的验证方式。

pip 依赖必须写入 `environment.yml` 的 `pip:` 子项，使环境配置仍然可追溯。

如果需要直接执行 pip，必须通过受控工程环境中的 Python 调用：

```bash
conda run --no-capture-output -n <environment-name> \
  python -m pip install <package-name>
```

禁止直接使用来源未确认的：

```bash
pip install <package-name>
```

### 7. 环境验证

创建环境、同步环境或新增依赖后，应按影响范围执行验证。

#### 基本解释器检查

```bash
conda run --no-capture-output -n <environment-name> \
  python -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.system()); print(platform.machine())"
```

操作系统和 CPU 架构用于诊断，不作为工程规则中的硬性平台断言。

#### 核心科学计算包

```bash
conda run --no-capture-output -n <environment-name> \
  python -c "import numpy, scipy; print(numpy.__version__); print(scipy.__version__); numpy.__config__.show()"
```

#### Python 依赖一致性

```bash
conda run --no-capture-output -n <environment-name> \
  python -m pip check
```

`pip check` 只用于检查 Python 包依赖元数据，不替代工程测试。

#### 代码质量与任务验证

环境变化影响工程代码时，还必须：

- 运行 Ruff；
- 运行受影响任务已有的测试入口；
- 运行必要的最小功能 smoke test；
- 确认没有修改 `data/raw/`；
- 确认没有破坏已有任务结果；
- 检查 Git 状态。

如果验证失败，不得把当前环境视为可用于正式科研结果生成的已验证环境。

### 8. 最小变更原则

环境管理必须服务于当前任务的实际需求。

禁止：

- 提前安装“以后可能会用到”的依赖；
- 为方便一次性安装大型综合环境；
- 无依据地升级核心科学计算栈；
- 为同一工程维护多套无明确用途的 Python 环境；
- 修改 Conda `base` 环境来满足本工程；
- 把其他工程中的包作为隐式依赖；
- 在环境未验证的情况下继续生成正式科研结果。

### 9. 环境变更记录

新增依赖或环境变更本身通常属于工程配置维护，不自动构成新的科研任务。

如果变更服务于某个明确任务：

- 在该任务 `work_logs/task_name/execution_log.txt` 中记录原因、变更和验证结果；
- 在 `work_logs/codex_handoff/codex_handoff.txt` 中追加简短摘要。

如果无法合理归属于具体任务：

- 不创建虚构任务目录；
- 直接在 `codex_handoff.txt` 中记录工程级维护；
- 说明修改的环境配置和验证结果。

---

## 十三、Git 提交规则

### 1. 仓库职责

Git 仓库用于保存：

- 源代码和测试代码；
- 工程规则和说明；
- 环境与工具配置；
- 文本形式的任务执行日志和工程交接记录；
- 其他明确属于可维护源文件的内容。

运行产生的数据、模型、图形、表格、缓存和分析结果默认不属于源代码仓库。

### 2. 默认提交边界

默认允许提交：

```text
scripts/
work_logs/ 中的文本日志和交接文档
AGENTS.md
README.md
environment.yml
pyproject.toml
.gitignore
其他明确的手写配置文件
```

默认不提交：

```text
data/
results/
运行生成的模型、图形、表格、缓存和分析产物
本地 Python/Conda 环境
密钥、凭据和机器相关配置
```

文件是否属于可提交内容应按用途判断，而不是仅按扩展名判断。

`.gitignore` 是生成物和本地文件排除规则的主要技术实现；`AGENTS.md` 负责规定 Agent 的安全边界。

### 3. 提交前检查

提交前必须检查：

```bash
git status --short
git diff --cached --name-only
git diff --cached --stat
```

不得使用：

```bash
git add .
git add -A
```

将整个工程无差别加入暂存区。

应按本次任务明确修改的路径选择性暂存。

如果暂存区出现 `data/`、`results/` 或其他运行产物，应先确认其用途并取消不应提交的内容。

### 4. 已跟踪生成物

`.gitignore` 不会自动取消已经被 Git 跟踪的文件。

如果发现历史上已经跟踪的数据、结果或生成物：

- 不得静默删除本地文件；
- 不得擅自执行 `git rm --cached`；
- 不得擅自改写 Git 历史；
- 必须先检查文件用途和影响；
- 涉及取消跟踪、删除或历史整理时，需要用户明确授权。

### 5. 提交原则

Git 提交应：

- 与当前任务或维护内容一致；
- 尽量保持小而清晰；
- 不混入无关修改；
- 不覆盖用户已有未提交工作；
- 不通过强制推送或历史重写来“整理”仓库。

本章只约束新的 Git 操作，不自动清理已有历史或本地结果。
