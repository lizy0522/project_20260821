"""Frozen configuration for the Hard-20 strict-Volterra experiment."""

from __future__ import annotations

import math
import os
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent

EXPERIMENT_NAME = "scenario_2_hard20_volterra_basis_selection_5B"
RESULT_PARENT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2"
PRIMARY_RESULT_ROOT = RESULT_PARENT / EXPERIMENT_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /EXPERIMENT_NAME / "execution_log.txt"  # noqa: E501
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"

STATE_COUNT = 425
FULL_LENGTH = 24_576
TRAIN_START = 0
TRAIN_END = 16_384
TEST_START = 16_384
TEST_END = 24_576
TRAIN_LENGTH = TRAIN_END - TRAIN_START
TEST_LENGTH = TEST_END - TEST_START

ORDERS = (1, 3, 5)
DMAX = 2
CANDIDATE_COUNT = 81
MANDATORY_BASIS_ID = "V1_u0"
HARD_STATE_COUNT = 20
TARGET_NMSE_DB = -40.0
INNER_CV_FOLDS = 3

K_MAX = 30
BACKWARD_TOLERANCE_DB = 0.02
PLATEAU_STEPS = 3
PLATEAU_WORST_IMPROVEMENT_DB = 0.05
PAIR_TOP_L = 24
PAIR_MAX_ROUNDS = 2
PAIR_WORST_IMPROVEMENT_DB = 0.10
PAIR_MEAN_IMPROVEMENT_DB = 0.05
SWAP_TOP_L = 12
SWAP_MAX_ACCEPTS = 3
SWAP_WORST_IMPROVEMENT_DB = 0.05
PARETO_MAX_SUPPORTS = 6

RIDGE_LAMBDA_GRID = (
    0.0,
    1e-12,
    1e-11,
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
)

WORKER_BENCHMARK_COUNTS = (12, 14, 16)
WORKER_NEAR_TIE_FRACTION = 0.03
PREPROCESS_WORKERS = 12
BLAS_THREADS_PER_WORKER = 1
FINE_ALIGN_SUBTIME = 256
OLS_CONDITION_HARD_LIMIT = 1e10

EXPECTED_OUTPUT_FILENAMES = (
    "01_hard20_state_ranking.csv",
    "02_volterra_candidate_dictionary.csv",
    "03_selection_path.csv",
    "04_selected_models.csv",
    "05_hard20_inner_cv_metrics.csv",
    "06_hard20_final_train_test_metrics.csv",
    "07_final_result_summary.txt",
)


def logical_cpu_count() -> int:
    """Return the detected logical CPU count."""

    return int(os.cpu_count() or 1)


def available_benchmark_workers() -> tuple[int, ...]:
    """Return benchmark worker counts bounded by the current host."""

    logical = logical_cpu_count()
    values = tuple(value for value in WORKER_BENCHMARK_COUNTS if value <= logical)
    return values or (max(1, math.floor(0.7 * logical)),)


__all__ = [name for name in globals() if name.isupper()] + [
    "available_benchmark_workers",
    "logical_cpu_count",
]
