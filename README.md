# project_20260821

标准四目录 Python 数据分析与科研工程。

本工程用于组织原始数据、处理中间数据、分析脚本、任务结果和执行记录。

- **环境配置**：以根目录 `environment.yml` 为唯一事实来源。
- **Agent 执行规则**：以根目录 `AGENTS.md` 为准。
- **Python 工具配置**：以 `pyproject.toml` 为准。
- **Git 排除规则**：以 `.gitignore` 为主要技术实现。

---

## Quick Start

### 1. 创建工程环境

首次在一台电脑上使用本工程：

```bash
conda env create -f environment.yml
```

`environment.yml` 当前声明的环境名称为（该名称仅用于当前使用示例，唯一事实来源仍是 `environment.yml`）：

```text
project_20260821
```

### 2. 激活环境

```bash
conda activate project_20260821
```

除 `conda env create`、`conda env update` 等 Conda 环境管理命令外，本文后续出现的 `python`、`pip`、`ruff` 以及任务测试命令，均默认在 `environment.yml` 声明的工程环境已经正确激活后执行。

如果环境名称以后发生变化，应以 `environment.yml` 的 `name` 字段为准，并同步更新这里用于人工操作的激活示例。

### 3. 已有环境同步配置

当 `environment.yml` 更新后：

```bash
conda env update -f environment.yml
```

默认不使用 `--prune`。只有在明确需要清理环境并检查删除影响后，才使用该选项。

### 4. 基本验证

```bash
python -c "import sys, platform; print(sys.executable); print(sys.version); print(platform.system()); print(platform.machine())"
python -c "import numpy, scipy; print(numpy.__version__); print(scipy.__version__); numpy.__config__.show()"
python -m pip check
ruff check --no-cache scripts
```

正式科研计算前，还应运行与当前任务直接相关的已有测试入口或最小功能 smoke test。

---

## 目录结构

```text
project_20260821/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── _core/
│   ├── signal_segmentation/
│   └── task_name/
├── results/
│   └── task_name/
├── work_logs/
│   ├── task_name/
│   └── codex_handoff/
├── AGENTS.md
├── README.md
├── environment.yml
└── pyproject.toml
```

目录职责：

- `data/raw/`：原始输入和实验数据，原则上保持只读。
- `data/processed/`：可重复生成、可被多个任务复用的中间数据。
- `scripts/_core/`：跨任务复用的公共模块。
- `scripts/signal_segmentation/`：固定 A/B/C 样点所有权及相关规范信号处理。
- `scripts/task_name/`：具体任务脚本。
- `results/task_name/`：具体任务的分析结果和任务专属输出。
- `work_logs/task_name/`：任务首次执行及后续维护的连续执行记录。
- `work_logs/codex_handoff/`：工程总体交接和状态摘要。

任务目录使用具有实际语义的小写英文 `snake_case` 名称，不使用数字任务编号。

完整任务判定、维护、数据保护和日志规则见 `AGENTS.md`。

---

## Python 环境

### 唯一事实来源

工程环境由：

```text
environment.yml
```

定义。

其中包括：

- 环境名称；
- Conda channel；
- Python 版本；
- 直接 Python 依赖；
- 当前数值计算后端约束。

README 不重复维护完整依赖列表，以避免环境定义在多个文件之间漂移。

当前 `environment.yml` 如需修改，应优先修改该文件，再同步 Conda 环境。

### 跨平台原则

工程代码和规则不绑定 Windows、macOS 或 Linux 的用户绝对路径。

不同电脑或操作系统之间迁移工程时：

1. 同步源代码和配置；
2. 同步必要的数据与本地结果；
3. 根据 `environment.yml` 重新创建 Conda 环境；
4. 不直接复制旧电脑的 Conda 环境目录；
5. 重新执行基本环境和任务验证。

操作系统和 CPU 架构可以作为诊断信息，但不是 README 中的硬性运行平台限制。

---

## 脚本路径约定

任务脚本应通过脚本自身位置推导工程根目录，不硬编码工程绝对路径。

推荐：

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_NAME = Path(__file__).resolve().parent.name

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results" / TASK_NAME
WORK_LOG_DIR = PROJECT_ROOT / "work_logs" / TASK_NAME
```

任务结果原则上写入：

```text
results/task_name/
```

任务执行记录写入：

```text
work_logs/task_name/execution_log.txt
```

---

## 环境与依赖变更

新增 Python 依赖时：

1. 优先使用 `conda-forge`；
2. 只把工程直接依赖写入 `environment.yml`；
3. 必要时先检查 Conda 求解计划；
4. 修改 `environment.yml` 后使用：

```bash
conda env update -f environment.yml
```

5. 完成依赖一致性、Ruff 和受影响任务验证。

pip 只作为 Conda 无法满足需求时的受控例外。详细要求见 `AGENTS.md`。

当前数值计算后端的实际约束以 `environment.yml` 为准；不得仅根据 README 或旧机器环境推断。

---

## 代码检查与验证

基础代码检查：

```bash
ruff check --no-cache scripts
```

基础依赖检查：

```bash
python -m pip check
```

科学计算环境检查：

```bash
python -c "import numpy, scipy; print(numpy.__version__); print(scipy.__version__); numpy.__config__.show()"
```

对于具体科研任务，还应运行该任务已有测试或最小 smoke test。基础环境检查不能替代任务级数值验证。

---

## Git

当前远程仓库：

```text
https://github.com/lizy0522/project_20260821.git
```

常用命令：

```bash
git status --short --branch
git pull --ff-only
git push
```

仓库默认保存：

- 源代码和测试；
- 工程配置；
- `AGENTS.md`、`README.md`；
- 文本形式的工作日志和交接记录。

`data/`、`results/` 以及运行生成的数据、模型、图形、表格和缓存默认不进入新的 Git 提交。

提交和 Git 安全规则以 `AGENTS.md` 为准；文件排除规则以 `.gitignore` 为主要技术实现。

---

## Agent 执行规则

Codex 或其他 Agent 在本工程中工作时，应先读取：

```text
AGENTS.md
```

其中定义：

- 新任务模式与持续维护模式；
- 任务命名和目录边界；
- 公共模块复用；
- `data/raw/` 保护；
- 结果与日志管理；
- Python 环境与依赖管理；
- Git 安全；
- 完成前检查。

README 负责说明工程如何使用，不重复维护 Agent 的完整执行规则。
