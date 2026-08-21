# project_20260821

标准四目录 Python 数据分析与科研工程。

本工程用于组织原始数据、处理中间数据、分析脚本、任务结果和执行记录。

Codex 或其他 Agent 在本工程中执行任务时，应遵守工程根目录中的 `AGENTS.md`。

---

## 目录结构

```text
project_20260821/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── _core/
│   └── task_name/
├── results/
│   └── task_name/
├── work_logs/
│   ├── task_name/
│   └── codex_handoff/
├── AGENTS.md
├── README.md
└── environment.yml
```

各目录职责如下：

- `data/raw/`：保存原始输入数据和原始实验数据，原则上保持只读，默认不进入 Git。
- `data/processed/`：保存清洗、转换、对齐、重采样或其他可复用的中间数据。
- `scripts/_core/`：保存跨任务复用的公共模块。
- `scripts/task_name/`：保存某个具体任务使用的脚本。
- `results/task_name/`：保存对应任务的分析结果、图形、表格、模型或其他输出。
- `work_logs/task_name/`：保存对应任务从首次执行到后续持续维护的完整执行记录。
- `work_logs/codex_handoff/`：保存工程总体交接和执行摘要。

任务目录不使用数字编号，统一采用简短、明确的小写英文 `snake_case` 名称，例如：

```text
scripts/fit_behavior_model/
results/fit_behavior_model/
work_logs/fit_behavior_model/
```

如果某项工作属于已有任务的持续维护，应继续复用原任务目录，并把维护记录追加到原任务的：

```text
work_logs/task_name/execution_log.txt
```

具体任务判定、持续维护、结果保存、日志和 Git 安全规则，以 `AGENTS.md` 为准。

---

## Conda 环境

环境名称：

```text
project_20260821
```

环境路径：

```text
D:\Project_Files\python_project\conda_envs\project_20260821
```

创建环境时只使用 `conda-forge`，并显式启用严格频道优先级和 OpenBLAS：

```powershell
& "D:\Program_Files\anaconda3\Scripts\conda.exe" create --yes `
  --prefix "D:\Project_Files\python_project\conda_envs\project_20260821" `
  --override-channels `
  --strict-channel-priority `
  -c conda-forge `
  python=3.11 `
  numpy `
  pandas `
  scipy `
  matplotlib `
  seaborn `
  ruff `
  pip `
  "libblas=*=*_openblas" `
  "liblapack=*=*_openblas" `
  "libcblas=*=*_openblas"
```

`environment.yml` 应保存与工程环境一致的依赖约束。

---

## 环境验证

Conda 环境创建完成后，先验证 NumPy、SciPy 和 BLAS 配置：

```powershell
& "D:\Program_Files\anaconda3\Scripts\conda.exe" run `
  --no-capture-output `
  --prefix "D:\Project_Files\python_project\conda_envs\project_20260821" `
  python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__); numpy.__config__.show()"
```

验证结果应满足：

- NumPy 可以正常导入；
- SciPy 可以正常导入；
- BLAS 后端显示为 OpenBLAS。

如果出现 NumPy 导入崩溃、异常 MKL 组合或科学计算环境异常，应先解决环境问题，再执行后续分析。

---

## 激活环境

如果已经完成 Conda 的 PowerShell 初始化，可使用：

```powershell
conda activate "D:\Project_Files\python_project\conda_envs\project_20260821"
```

---

## 脚本路径约定

任务脚本应通过脚本自身位置推导工程根目录，不硬编码项目绝对路径。

推荐写法：

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

任务输出原则上写入：

```text
results/task_name/
```

详细执行记录写入：

```text
work_logs/task_name/execution_log.txt
```

对已有任务进行持续维护时，继续向同一个 `execution_log.txt` 追加记录，不另建维护日志目录。

---

## 代码检查

使用 Ruff 检查 `scripts`：

```powershell
& "D:\Program_Files\anaconda3\Scripts\conda.exe" run `
  --no-capture-output `
  --prefix "D:\Project_Files\python_project\conda_envs\project_20260821" `
  ruff check --no-cache scripts
```

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

涉及自动修改、提交、文件删除和历史操作时，应遵守 `AGENTS.md` 中的 Git 安全规则。

---

## Agent 执行规则

工程中的以下规则统一定义在：

```text
AGENTS.md
```

包括：

- 新任务模式与持续维护模式；
- 任务目录命名；
- 脚本、结果和日志保存位置；
- 公共模块复用；
- 原始数据保护；
- 持续维护记录；
- Git 安全；
- 完成前检查。

README 只负责说明工程结构、运行环境和基本使用方法，不重复维护 Agent 的具体执行规则。
