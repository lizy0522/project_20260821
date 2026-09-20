# ruff: noqa: E402,E501,I001

"""Independent read-only validation for the completed K12 forward scan."""

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
from behavior_fingerprint_retrieval.shared.envelope18_commonB_full_lut import top1_retrieval  # noqa: E402
from behavior_fingerprint_retrieval.shared.envelope19_c2endshared_commonB_full_lut import raw_manifest_gate  # noqa: E402
from data_management.shared import load_by_id  # noqa: E402
from signal_segmentation.shared import build_off_segments, build_partition_from_xin  # noqa: E402

TASK_NAME = "scenario_2_k12_retrieval_oriented_forward_scan_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
K11_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b"
K9_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
STATE_COUNT = 425


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _real_b_check(matrix: np.ndarray) -> float:
    _assert(matrix.shape == (STATE_COUNT, STATE_COUNT), "Real-B matrix shape failed")
    _assert(np.all(np.isneginf(np.diag(matrix))), "Real-B diagonal is not -Inf")
    rng = np.random.default_rng(20260916)
    cache: dict[int, np.ndarray] = {}
    maximum = 0.0
    for real_id, query_id in rng.integers(0, STATE_COUNT, size=(20, 2)):
        for state_id in (int(real_id), int(query_id)):
            if state_id not in cache:
                data = load_by_id(state_id)
                partition = build_partition_from_xin(np.asarray(data["xin"]))
                cache[state_id] = np.asarray(build_off_segments(data, partition)["B"].output[2:], dtype=np.complex128)
        direct = cnmse(cache[int(real_id)], cache[int(query_id)])
        stored = float(matrix[int(real_id), int(query_id)])
        if np.isneginf(direct) and np.isneginf(stored):
            continue
        maximum = max(maximum, abs(direct - stored))
    _assert(maximum <= 1e-9, f"Real-B direct pair error too large: {maximum}")
    return maximum


def main() -> None:
    summary = pd.read_csv(RESULT_ROOT / "10_candidate_retrieval_summary.csv")
    query = pd.read_csv(RESULT_ROOT / "11_candidate_query_metrics_long.csv")
    model = pd.read_csv(RESULT_ROOT / "08_candidate_model_quality_long.csv")
    delta = pd.read_csv(RESULT_ROOT / "12_forward_addition_delta.csv")
    recovered = pd.read_csv(RESULT_ROOT / "13_recovered_regressed_states.csv")
    failure9 = pd.read_csv(RESULT_ROOT / "14_failure9_detailed_analysis.csv")
    hard = pd.read_csv(RESULT_ROOT / "15_hard_pair_correction_long.csv")
    folds = pd.read_csv(RESULT_ROOT / "16_query_state_cv_folds.csv")
    cv = pd.read_csv(RESULT_ROOT / "17_query_state_cv_results.csv")
    stability = pd.read_csv(RESULT_ROOT / "18_query_state_cv_selection_stability.csv")
    regression = json.loads((RESULT_ROOT / "07_baseline_regression_check.json").read_text())
    with np.load(RESULT_ROOT / "09_candidate_coefficients.npz", allow_pickle=False) as data:
        theta_a = np.asarray(data["theta_Aend_padded"])
        theta_c = np.asarray(data["theta_C2_padded"])

    _assert(summary.shape[0] == 65, "summary must contain 65 models")
    _assert(summary["K"].eq(11).sum() == 1 and summary["K"].eq(12).sum() == 64, "K11/K12 count failed")
    _assert(model.shape[0] == 65 * STATE_COUNT, "model quality row count failed")
    _assert(query.shape[0] == 65 * STATE_COUNT, "query metric row count failed")
    _assert(delta.shape[0] == 64, "forward delta row count failed")
    _assert(recovered.shape[0] == 64 * STATE_COUNT, "recovered/regressed row count failed")
    _assert(failure9.shape[0] == 64 * 9 and failure9["State_R"].nunique() == 9, "failure9 row count failed")
    _assert(theta_a.shape == (65, STATE_COUNT, 12) and theta_c.shape == theta_a.shape, "coefficient shape failed")
    _assert(regression["pass"] is True, "K11 baseline regression failed")

    matrix_checks = 0
    for row in summary.sort_values("candidate_id").itertuples(index=False):
        distance = np.load(RESULT_ROOT / "distance_matrices" / f"{row.candidate_name}.npy")
        _assert(distance.shape == (STATE_COUNT, STATE_COUNT), f"{row.candidate_name} distance shape failed")
        _assert(not np.isnan(distance).any() and not np.isposinf(distance).any(), f"{row.candidate_name} distance finite gate failed")
        selected, _, _, _ = top1_retrieval(distance)
        current = query.loc[query["candidate_id"].eq(int(row.candidate_id))].sort_values("State_R")
        _assert(np.array_equal(selected, current["State_Q"].to_numpy(dtype=int)), f"{row.candidate_name} State_Q mismatch")
        for state_id in (0, 42, 187, 325, 424):
            order = np.lexsort((np.arange(STATE_COUNT), distance[state_id]))
            _assert(int(order[0]) == int(selected[state_id]), f"{row.candidate_name} argmin failed at {state_id}")
            matrix_checks += 1

    real_b = np.load(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy")
    shareability = np.load(K9_ROOT / "05_realB_shareability_mask.npy")
    _assert(shareability.shape == real_b.shape and shareability.dtype == bool, "Real-B mask failed")
    _assert(shareability.diagonal().all(), "Real-B shareability diagonal failed")
    real_b_error = _real_b_check(real_b)

    k11_baseline = query.loc[query["candidate_id"].eq(0)].sort_values("State_R").reset_index(drop=True)
    _assert(int(k11_baseline["realB_pass"].sum()) == 416 and int(k11_baseline["exact_hit"].sum()) == 218, "K11 baseline counts failed")
    _assert(summary.loc[summary["added_basis_id"].eq("ENV_p04_m0_q1"), "top1_pass_count"].iloc[0] == 419, "best K12 count failed")
    _assert(len(pd.read_csv(RESULT_ROOT / "05_k11_hard_pairs.csv")) == 7, "hard-pair count failed")
    _assert(hard.shape[0] == 64 * 7, "hard-pair correction row count failed")

    k11_folds = pd.read_csv(K11_ROOT / "13_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    _assert(np.array_equal(folds[["state_id", "fold"]].to_numpy(), k11_folds[["state_id", "fold"]].to_numpy()), "K12 folds differ from K11")
    _assert(folds["fold"].value_counts().sort_index().to_dict() == {0: 85, 1: 85, 2: 85, 3: 85, 4: 85}, "CV fold balance failed")
    _assert(cv.shape[0] == 65 * 5 and cv.groupby("fold").size().eq(65).all(), "CV result rows failed")
    _assert(cv.groupby("fold")["train_is_winner"].sum().eq(1).all(), "CV winner-per-fold failed")
    _assert(stability.shape[0] == 65, "CV stability rows failed")

    checks = {
        "task_name": TASK_NAME,
        "raw_manifest": raw_manifest_gate(),
        "candidate_count": 65,
        "k12_count": 64,
        "model_quality_rows": int(model.shape[0]),
        "query_metric_rows": int(query.shape[0]),
        "distance_matrix_argmin_checks": matrix_checks,
        "real_B_random_pair_count": 20,
        "real_B_random_pair_max_abs_error_dB": real_b_error,
        "k11_failure_count": 9,
        "hard_pair_count": 7,
        "hard_pair_correction_rows": int(hard.shape[0]),
        "failure9_rows": int(failure9.shape[0]),
        "cv_rows": int(cv.shape[0]),
        "cv_winner_count_per_fold": 1,
        "baseline_regression_pass": True,
        "pass": True,
    }
    (RESULT_ROOT / "28_validation_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
