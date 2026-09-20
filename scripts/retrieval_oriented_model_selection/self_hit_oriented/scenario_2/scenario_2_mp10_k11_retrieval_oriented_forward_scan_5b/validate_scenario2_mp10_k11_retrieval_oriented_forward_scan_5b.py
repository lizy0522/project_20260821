# ruff: noqa: E402,E501,I001

"""Independent read-only validation for the completed K11 forward scan."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.shared.metrics import cnmse  # noqa: E402
from data_management.shared import load_by_id  # noqa: E402
from signal_segmentation.shared import build_off_segments, build_partition_from_xin  # noqa: E402
from behavior_fingerprint_retrieval.shared.envelope18_commonB_full_lut import top1_retrieval  # noqa: E402
from behavior_fingerprint_retrieval.shared.envelope19_c2endshared_commonB_full_lut import raw_manifest_gate  # noqa: E402

TASK_NAME = "scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
PREVIOUS_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2_mp10_full_lut_retrieval_5b"
REAL_B_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
STATE_COUNT = 425


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    summary = pd.read_csv(RESULT_ROOT / "08_candidate_retrieval_summary.csv")
    query = pd.read_csv(RESULT_ROOT / "09_candidate_query_metrics_long.csv")
    model = pd.read_csv(RESULT_ROOT / "06_candidate_model_quality_long.csv")
    delta = pd.read_csv(RESULT_ROOT / "10_forward_addition_delta.csv")
    folds = pd.read_csv(RESULT_ROOT / "13_query_state_cv_folds.csv")
    cv = pd.read_csv(RESULT_ROOT / "14_query_state_cv_results.csv")
    stability = pd.read_csv(RESULT_ROOT / "15_query_state_cv_selection_stability.csv")
    overlap = pd.read_csv(RESULT_ROOT / "02_mp10_envelope75_overlap_regression.csv")
    with np.load(RESULT_ROOT / "07_candidate_coefficients.npz", allow_pickle=False) as data:
        theta_a = np.asarray(data["theta_Aend_padded"])
        theta_c = np.asarray(data["theta_C2_padded"])

    _assert(summary.shape[0] == 66, "candidate summary must have 66 rows")
    _assert(summary["K"].eq(10).sum() == 1 and summary["K"].eq(11).sum() == 65, "K10/K11 counts failed")
    _assert(model.shape[0] == 66 * STATE_COUNT, "model quality long row count failed")
    _assert(query.shape[0] == 66 * STATE_COUNT, "query metrics long row count failed")
    _assert(delta.shape[0] == 65, "forward delta row count failed")
    _assert(theta_a.shape == (66, STATE_COUNT, 11) and theta_c.shape == theta_a.shape, "coefficient shape failed")
    _assert(float(overlap["max_abs_delta"].max()) <= 1e-12, "MP10/Envelope75 overlap regression failed")

    matrix_dir = RESULT_ROOT / "distance_matrices"
    matrix_checks = 0
    for row in summary.sort_values("model_id").itertuples(index=False):
        distance = np.load(matrix_dir / f"{row.model_name}.npy")
        _assert(distance.shape == (STATE_COUNT, STATE_COUNT), f"{row.model_name} distance shape failed")
        _assert(not np.isnan(distance).any() and not np.isposinf(distance).any(), f"{row.model_name} distance finite gate failed")
        selected, _, _, _ = top1_retrieval(distance)
        current = query.loc[query["candidate_id"].eq(int(row.model_id))].sort_values("State_R")
        _assert(np.array_equal(selected, current["State_Q"].to_numpy(dtype=int)), f"{row.model_name} State_Q mismatch")
        for state_id in (0, 42, 187, 325, 424):
            _assert(int(np.lexsort((np.arange(STATE_COUNT), distance[state_id]))[0]) == int(selected[state_id]), f"{row.model_name} argmin check failed")
            matrix_checks += 1

    previous = pd.read_csv(PREVIOUS_ROOT / "09_retrieval_results_all425.csv").sort_values("State_n_R").reset_index(drop=True)
    baseline = query.loc[query["candidate_id"].eq(0)].sort_values("State_R").reset_index(drop=True)
    _assert(np.array_equal(baseline["State_Q"].to_numpy(dtype=int), previous["State_n_Q"].to_numpy(dtype=int)), "baseline State_Q changed")
    _assert(int(baseline["realB_pass"].sum()) == 411, "baseline Real-B pass changed")
    _assert(int(baseline["exact_hit"].sum()) == 218, "baseline exact count changed")

    _assert(folds.shape[0] == STATE_COUNT and folds["fold"].value_counts().sort_index().to_dict() == {0: 85, 1: 85, 2: 85, 3: 85, 4: 85}, "CV fold balance failed")
    _assert(cv.shape[0] == 66 * 5 and cv.groupby("fold").size().eq(66).all(), "CV result shape failed")
    _assert(cv.groupby("fold")["train_is_winner"].sum().eq(1).all(), "CV winner-per-fold failed")
    _assert(stability.shape[0] == 66, "CV stability row count failed")

    real_b = np.load(REAL_B_ROOT / "04_realB_ground_truth_cnmse_matrix.npy")
    _assert(np.all(np.isneginf(np.diag(real_b))), "previous Real-B diagonal changed")
    rng = np.random.default_rng(20260916)
    cache: dict[int, np.ndarray] = {}
    maximum_error = 0.0
    for real_id, query_id in rng.integers(0, STATE_COUNT, size=(20, 2)):
        for state_id in (int(real_id), int(query_id)):
            if state_id not in cache:
                data = load_by_id(state_id)
                partition = build_partition_from_xin(np.asarray(data["xin"]))
                cache[state_id] = np.asarray(build_off_segments(data, partition)["B"].output[2:], dtype=np.complex128)
        direct = cnmse(cache[int(real_id)], cache[int(query_id)])
        stored = float(real_b[int(real_id), int(query_id)])
        if np.isneginf(direct) and np.isneginf(stored):
            continue
        maximum_error = max(maximum_error, abs(direct - stored))
    _assert(maximum_error <= 1e-9, "previous Real-B random-pair check failed")
    checks = {
        "task_name": TASK_NAME,
        "raw_manifest": raw_manifest_gate(),
        "candidate_count": 66,
        "k11_count": 65,
        "model_quality_rows": int(model.shape[0]),
        "query_metric_rows": int(query.shape[0]),
        "distance_matrix_argmin_checks": matrix_checks,
        "overlap_max_abs_delta": float(overlap["max_abs_delta"].max()),
        "real_B_random_pair_count": 20,
        "real_B_random_pair_max_abs_error_dB": maximum_error,
        "cv_rows": int(cv.shape[0]),
        "cv_winner_count_per_fold": 1,
        "pass": True,
    }
    (RESULT_ROOT / "23_validation_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
