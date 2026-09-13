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

## 十二、Conda 环境与扩展包管理

### 1. 固定 Conda 环境

本工程必须使用以下独立 Conda 环境：

```text
D:\Project_Files\python_project\conda_envs\project_20260821
```

本工程固定 Python 解释器为：

```text
D:\Project_Files\python_project\conda_envs\project_20260821\python.exe
```

Conda 管理程序固定使用：

```text
D:\Program_Files\anaconda3\Scripts\conda.exe
```

执行任何 Python 命令、安装依赖、运行测试、运行 Ruff 或执行任务脚本时，必须确认实际解释器属于上述工程环境。

不得为本工程另行创建：

```text
.venv/
venv/
env/
```

不得使用：

- Anaconda `base` 环境；
- 系统 Python；
- 用户级全局 Python；
- 其他工程的 Conda 环境；
- 未确认解释器来源的 `python`、`pip`、`pytest` 或 `ruff` 命令。

### 2. 新扩展包默认通过 Conda 安装

每次任务需要新的 Python 扩展包时，默认必须通过 Conda 安装到本工程固定环境。

安装时必须：

1. 使用完整 `--prefix` 指定本工程环境；
2. 使用 `conda-forge`；
3. 使用 `--override-channels`；
4. 使用 `--strict-channel-priority`；
5. 保持本工程既有 OpenBLAS 约束；
6. 不得把扩展包安装到 `base` 或其他工程环境。

标准安装形式为：

```powershell
$projectConda = "D:\Program_Files\anaconda3\Scripts\conda.exe"
$projectEnv = "D:\Project_Files\python_project\conda_envs\project_20260821"

& $projectConda install --yes `
  --prefix $projectEnv `
  --override-channels `
  --channel conda-forge `
  --strict-channel-priority `
  <package-name> `
  "libblas=*=*_openblas" `
  "liblapack=*=*_openblas" `
  "libcblas=*=*_openblas"
```

不得仅依赖当前终端的激活状态执行：

```powershell
conda install <package-name>
```

因为该命令可能把扩展包安装到错误环境。

### 3. 安装前检查求解计划

正式安装新扩展包前，应先检查 Conda 的求解结果。

对于可能影响 NumPy、SciPy、BLAS、编译运行时或大量传递依赖的扩展包，应先执行 `--dry-run`：

```powershell
$projectConda = "D:\Program_Files\anaconda3\Scripts\conda.exe"
$projectEnv = "D:\Project_Files\python_project\conda_envs\project_20260821"

& $projectConda install --dry-run `
  --prefix $projectEnv `
  --override-channels `
  --channel conda-forge `
  --strict-channel-priority `
  <package-name> `
  "libblas=*=*_openblas" `
  "liblapack=*=*_openblas" `
  "libcblas=*=*_openblas"
```

如果求解计划出现以下情况，应停止安装并报告：

- Python 主版本或次版本发生变化；
- NumPy、SciPy 或其他核心科学计算包发生非必要的大幅升级或降级；
- OpenBLAS 被替换为 MKL、BLIS 或其他 BLAS 实现；
- 大量现有包被删除；
- 出现明显与当前任务无关的依赖变更；
- 求解器需要混用未经批准的 channel；
- 现有工程代码可能因此失去兼容性。

不得为了安装一个扩展包而默认执行：

```powershell
conda update --all
```

不得顺带升级与当前任务无关的依赖。

### 4. OpenBLAS 不变量

本工程的 `environment.yml` 已固定：

```yaml
- "libblas=*=*_openblas"
- "liblapack=*=*_openblas"
- "libcblas=*=*_openblas"
```

新增或更新扩展包时，必须保持上述 OpenBLAS 约束。

不得静默将本工程切换到 MKL。

安装完成后，应检查：

```powershell
$projectConda = "D:\Program_Files\anaconda3\Scripts\conda.exe"
$projectEnv = "D:\Project_Files\python_project\conda_envs\project_20260821"

& $projectConda list --prefix $projectEnv |
  Select-String -Pattern "libblas|liblapack|libcblas|openblas|mkl"
```

如果发现 MKL 被引入、OpenBLAS 被替换，或者 BLAS 依赖状态与 `environment.yml` 不一致，应停止后续科研计算并处理环境问题。

### 5. 同步更新 environment.yml

成功安装任务实际需要的扩展包后，必须同步更新工程根目录中的：

```text
environment.yml
```

只将本工程直接依赖的包加入 `dependencies`。

不得用完整的自动导出结果覆盖现有 `environment.yml`，避免写入大量平台相关、构建相关或传递依赖。

例如新增 `openpyxl` 时，应在现有依赖中增加：

```yaml
dependencies:
  - python=3.11
  - numpy
  - pandas
  - scipy
  - matplotlib
  - seaborn
  - openpyxl
  - ruff
  - pip
  - "libblas=*=*_openblas"
  - "liblapack=*=*_openblas"
  - "libcblas=*=*_openblas"
```

如果任务要求固定特定版本，应在任务依据充分的情况下明确记录版本约束，不得无依据地随意锁定版本。

### 6. pip 仅作为受控例外

只有满足以下至少一种情况时，才允许使用 pip：

- conda-forge 不提供该包；
- conda-forge 没有与本工程 Python 版本兼容的版本；
- 软件官方明确要求通过 pip 安装；
- 所需功能只存在于官方 pip wheel；
- Conda 包经过核验不能满足当前任务需求。

使用 pip 前必须先说明：

1. 为什么 Conda/conda-forge 无法满足需求；
2. 准备安装的准确包名和版本；
3. 是否会影响现有核心依赖；
4. 安装后的验证方法。

使用 pip 时，必须通过本工程固定解释器调用：

```powershell
$projectPython = "D:\Project_Files\python_project\conda_envs\project_20260821\python.exe"

& $projectPython -m pip install <package-name>
```

禁止使用：

```powershell
pip install <package-name>
```

禁止使用其他 Python 解释器调用 pip。

如果使用 pip，应在 `environment.yml` 的 `pip:` 子项中记录直接依赖，并在任务日志或工程 handoff 中记录使用 pip 的原因。

### 7. 安装后的强制验证

扩展包安装完成后，至少执行以下验证。

#### 解释器路径

```powershell
$projectPython = "D:\Project_Files\python_project\conda_envs\project_20260821\python.exe"

& $projectPython -c "import sys; print(sys.executable); print(sys.version)"
```

必须确认：

```text
sys.executable =
D:\Project_Files\python_project\conda_envs\project_20260821\python.exe
```

#### 扩展包导入

```powershell
& $projectPython -c "import <package_name>; print(<package_name>.__version__)"
```

如果该包没有 `__version__`，应采用其官方支持的版本查询方式。

#### 依赖一致性

```powershell
& $projectPython -m pip check
```

必须得到：

```text
No broken requirements found.
```

#### 工程验证

如果新增扩展包影响工程代码，还必须：

- 运行相关 pytest；
- 运行 Ruff；
- 运行与该扩展包直接相关的最小功能检查；
- 确认没有修改 `data/raw/`；
- 确认没有破坏已有任务结果；
- 检查 Git 状态，区分用户既有修改和本次修改。

如果验证失败，不得继续生成正式科研结果，不得把失败的环境状态视为任务完成。

### 8. 任务模式与记录归属

新增依赖本身通常属于工程配置维护，不自动构成新的科研任务。

如果新增依赖服务于某个明确的新任务或已有任务：

- 将安装原因、包名、版本、安装方式和验证结果记录到该任务的 `execution_log.txt`；
- 在总体 `codex_handoff.txt` 中记录简短摘要；
- 不为依赖安装单独创建没有研究意义的任务目录。

如果依赖变更无法合理归属于具体任务，则按工程级维护处理：

- 不创建虚构任务目录；
- 直接在 `work_logs/codex_handoff/codex_handoff.txt` 中记录；
- 说明修改的 `environment.yml`、安装包、版本、验证结果和未完成事项。

### 9. 最小变更原则

每次扩展包安装必须只服务于当前任务的实际需求。

禁止：

- 提前安装“以后可能会用到”的包；
- 为了方便一次性安装大型综合环境；
- 无依据地升级 Python；
- 无依据地升级 NumPy、SciPy、pandas 或 Matplotlib；
- 创建第二套工程解释器；
- 修改 Anaconda `base` 环境；
- 把其他工程环境中的包作为本工程的隐式依赖；
- 通过复制 `site-packages` 的方式安装扩展包；
- 在没有验证的情况下继续运行正式分析。

核心原则是：

> 本工程使用一套固定、可追溯、可复现的 Conda 环境；新增依赖默认通过 conda-forge 安装，pip 仅作为有记录、有验证的受控例外。

## 十三、Git 提交规则：提交代码、配置、工作日志和交接文档

本工程的 GitHub 仓库保存可维护的源代码、测试代码、配置型文件、工作日志、交接文档和必要工程规则。代码运行产生的结果、数据和过程产物不进入新的 Git 提交。

### 1. 允许提交的内容

- `scripts/` 下的 Python、JavaScript、MJS、MATLAB 等源代码和测试代码；
- 运行代码所必需的配置型文件，例如 `environment.yml`、`pyproject.toml`、`.gitignore`、`*.yaml`、`*.yml`、`*.toml`、`*.ini`、`*.cfg` 和手写的配置型 `*.json`；
- `work_logs/` 下的工作日志、任务 `execution_log.txt` 和工程 `codex_handoff.txt` 交接文档；
- 维护工程所必需的规则和说明，例如 `AGENTS.md`、`README.md`；
- 小型、明确属于源代码契约的固定配置，不包括脚本运行后生成的分析数据。

### 2. 禁止提交的内容

- `results/` 下的分析结果、图形、表格、模型、指纹、缓存和导出文件；
- `data/raw/`、`data/processed/` 下的数据；
- `work_logs/` 下的二进制结果、图形、模型、缓存或其他运行产物；工作日志和交接文档不属于此项；
- 代码运行产生的 `.npz`、`.npy`、`.mat`、`.pkl`、`.joblib`、`.xlsx`、`.xls`、`.parquet`、`.h5`、`.hdf5`、`.ndjson`、`.json`、`.csv`、`.txt`、图片和 PDF；
- Python 缓存、测试缓存、Ruff 缓存、临时文件、本地环境和密钥。

文件是否允许提交按用途判断，而不只按扩展名判断：手写并作为程序输入的配置可以提交；脚本运行生成的 `validation.json`、`retrieval_results.csv`、`final_result_summary.txt` 等文件属于结果，即使扩展名是配置或文本格式，也不得提交。

### 3. 提交前检查

提交前必须确认暂存区只包含源代码、配置型文件、工作日志、交接文档和必要工程规则：

```powershell
git status --short
git diff --cached --name-only
git diff --cached --stat
```

不得直接使用 `git add .` 或 `git add -A` 将整个工程加入暂存区。应按明确的代码或配置路径选择性暂存，例如：

```powershell
git add scripts/
git add environment.yml pyproject.toml .gitignore AGENTS.md README.md
```

暂存区中出现 `results/`、`data/` 或脚本生成的结果文件时，必须先取消暂存，不能通过提交这些文件来保存分析结果。`work_logs/` 下的工作日志和交接文档可以保留在暂存区。

如果发现 `results/`、`work_logs/` 或其他生成物已经被 Git 跟踪，`.gitignore` 不会自动取消其跟踪。取消跟踪或重写尚未推送的历史必须先检查文件用途，并获得明确授权；不得静默删除本地结果、修改历史或强制推送。

### 4. 规则边界

本规则只约束新的 Git 提交，不删除已有本地结果，也不自动清理历史。结果仍按任务规则写入本地 `results/` 目录；工作日志和交接文档写入 `work_logs/`，用于复现和交接，并允许作为工程记录发布到 GitHub。
