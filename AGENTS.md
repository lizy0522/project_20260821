# Codex 工程执行规则

## 适用范围

本文件用于由 `PROJECT_TEMPLATE.txt` 创建或整理的标准四目录 Python 工程。

当工程根目录中存在 `data`、`results`、`scripts` 和 `work_logs` 时，Codex 每次执行任务都必须遵守本规则。
如果本文件位于“Python工程模板配置”目录中，则它是供新工程复制的规则源文件，不要求模板配置目录本身建立四个业务目录。

## 全局任务编号与对应子目录

Codex 每次执行新任务时，必须先分配一个全局唯一的三位任务编号。任务编号属于任务本身，
不单独属于 `scripts`，并按照任务下发顺序递增。

对于使用或创建脚本的任务，必须同时建立以下三个完全同名的任务子目录：

```text
scripts/NNN_task_name/
results/NNN_task_name/
work_logs/NNN_task_name/
```

对于不使用脚本、但会产生人工结果、仪器结果或第三方软件结果的任务，只建立：

```text
results/NNN_task_name/
work_logs/NNN_task_name/
```

此类任务不得为了补齐编号而建立空的 `scripts/NNN_task_name/`。

其中：

- `NNN` 是三位递增任务序号，例如 `001`、`002`、`003`。
- `task_name` 使用简短、明确的小写英文 `snake_case`。
- 同一任务实际存在的任务子目录，其编号和名称必须完全相同。
- 已使用的任务序号不得重排、重复或复用。
- `scripts` 中允许出现由非脚本任务造成的编号缺口，该缺口是正常现象，后续不得补用。

例如：

```text
scripts/003_fit_lut_model/
results/003_fit_lut_model/
work_logs/003_fit_lut_model/

results/004_manual_lut_adjustment/
work_logs/004_manual_lut_adjustment/

scripts/005_generate_figures/
results/005_generate_figures/
work_logs/005_generate_figures/
```

上例中的任务 `004` 是非脚本结果任务，因此没有 `scripts/004_manual_lut_adjustment/`；
下一个任务仍使用 `005`，不得复用 `004`。

## 目录职责

### scripts

如果本次任务使用或创建脚本，脚本必须写入：

```text
scripts/NNN_task_name/
```

脚本不得直接散落在 `scripts` 根目录。非脚本结果任务不创建空的脚本目录。

### results

本次任务产生的全部结果，无论是否由脚本生成，都必须写入：

```text
results/NNN_task_name/
```

结果包括脚本输出、人工制作或修改的文件、仪器导出文件以及第三方软件导出文件。
结果不得直接散落在 `results` 根目录，也不得写入其他任务的结果目录。

原始输入、参考资料和待处理文件不属于结果，应放入 `data/raw/`；清洗或转换后的中间数据
应放入 `data/processed/`。

### work_logs

本次任务的详细执行记录必须写入：

```text
work_logs/NNN_task_name/execution_log.txt
```

工程的总体执行记录必须追加写入：

```text
work_logs/codex_handoff/codex_handoff.txt
```

## 两级执行记录

### 详细执行记录

`work_logs/NNN_task_name/execution_log.txt` 至少记录：

- 工作目标；
- 任务类型：脚本任务或非脚本结果任务；
- 实际操作；
- 新建、移动、修改或删除的文件；
- 关键路径；
- 关键命令；
- 验证结果；
- 对非脚本结果记录其来源、生成方式、使用的工具和可复现性；
- 遇到的问题及处理方式；
- 未完成事项；
- 下一步建议。

### 总体执行记录

`work_logs/codex_handoff/codex_handoff.txt` 用于持续汇总：

- 任务编号和任务名称；
- 任务状态；
- 主要操作和成果；
- 关键验证结论；
- 未完成事项；
- 下一个任务序号。

总体记录只写任务摘要，详细过程保留在对应任务的 `execution_log.txt` 中。

## 完成前检查

Codex 完成每项任务前必须确认：

1. 同一任务实际存在的 `scripts`、`results` 和 `work_logs` 任务子目录完全同名；
2. 脚本任务的脚本位于对应的 `scripts/NNN_task_name/`；非脚本结果任务没有为补齐编号而创建空脚本目录；
3. 结果位于对应的 `results/NNN_task_name/`；
4. 详细记录位于对应的 `work_logs/NNN_task_name/execution_log.txt`；
5. 总体摘要已追加到 `work_logs/codex_handoff/codex_handoff.txt`；
6. `codex_handoff.txt` 中的下一个任务序号已更新。
