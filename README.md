# project_20260821

标准四目录 Python 数据分析工程。

## 目录

    data       数据
    results    结果
    scripts    脚本
    work_logs  工作日志

- data/raw：保存原始输入数据，默认不进入 Git。
- data/processed：保存清洗、转换或计算得到的中间数据。
- results：保存各任务对应的结果。
- scripts：保存各任务对应的 Python 脚本。
- work_logs：保存详细执行记录和总体交接记录。

任务目录必须使用三位递增编号，例如：

    scripts/001_task_name/
    results/001_task_name/
    work_logs/001_task_name/

本工程的初始化任务是非脚本结果任务，因此保留：

    results/001_initialize_project/
    work_logs/001_initialize_project/

## Conda 环境

环境名称：project_20260821

环境路径：

    D:\Project_Files\python_project\conda_envs\project_20260821

创建环境时只使用 conda-forge，并显式启用严格频道优先级和 OpenBLAS：

    & "D:\Program_Files\anaconda3\Scripts\conda.exe" create --yes --prefix "D:\Project_Files\python_project\conda_envs\project_20260821" --override-channels --strict-channel-priority -c conda-forge python=3.11 numpy pandas scipy matplotlib seaborn ruff pip "libblas=*=*_openblas" "liblapack=*=*_openblas" "libcblas=*=*_openblas"

environment.yml 保存相同的依赖约束。Conda 环境完成后，先验证：

    & "D:\Program_Files\anaconda3\Scripts\conda.exe" run --no-capture-output --prefix "D:\Project_Files\python_project\conda_envs\project_20260821" python -c "import numpy, scipy; print(numpy.__version__, scipy.__version__); numpy.__config__.show()"

输出必须确认 NumPy 和 SciPy 能正常导入，并显示 OpenBLAS；若出现未预期的 MKL 2026 组合或 NumPy 导入崩溃，应停止后续科学计算。

激活环境（需要先完成 Conda 的 PowerShell 初始化）：

    conda activate "D:\Project_Files\python_project\conda_envs\project_20260821"

## 脚本与结果

脚本通过文件位置推导工程根目录，不硬编码本机项目路径：

    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    TASK_NAME = Path(__file__).resolve().parent.name
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results" / TASK_NAME

脚本输出只能写入对应的 results/NNN_task_name；详细过程写入对应的 work_logs/NNN_task_name/execution_log.txt。

## 检查

    & "D:\Program_Files\anaconda3\Scripts\conda.exe" run --no-capture-output --prefix "D:\Project_Files\python_project\conda_envs\project_20260821" ruff check --no-cache scripts

## Git

当前远程仓库：

    https://github.com/lizy0522/project_20260821.git

常用命令：

    git status --short --branch
    git pull --ff-only
    git push
