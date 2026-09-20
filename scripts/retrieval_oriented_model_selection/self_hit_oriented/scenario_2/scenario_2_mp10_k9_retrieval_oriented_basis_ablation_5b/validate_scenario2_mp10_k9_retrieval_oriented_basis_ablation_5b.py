# ruff: noqa: E402,E501,I001

"""Independent read-only validation of the completed MP10 K9 ablation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
from behavior_fingerprint_retrieval.shared.envelope19_c2endshared_commonB_full_lut import (  # noqa: E402
    raw_manifest_gate,
)
from core.shared.metrics import cnmse  # noqa: E402
from behavior_fingerprint_retrieval.shared.envelope18_commonB_full_lut import (  # noqa: E402
    top1_retrieval,
)
from data_management.shared import load_by_id  # noqa: E402
from signal_segmentation.shared import build_off_segments, build_partition_from_xin  # noqa: E402

TASK_NAME = "scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
STATE_COUNT = 425
THRESHOLD_DB = -40.0


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _check_real_b(root: Path) -> dict[str, Any]:
    distance = np.load(root / "04_realB_ground_truth_cnmse_matrix.npy")
    mask = np.load(root / "05_realB_shareability_mask.npy")
    _assert(distance.shape == (STATE_COUNT, STATE_COUNT), "Real-B matrix shape failed")
    _assert(distance.dtype == np.float64, "Real-B matrix dtype failed")
    _assert(mask.shape == distance.shape and mask.dtype == bool, "Real-B mask shape/dtype failed")
    _assert(np.all(np.isneginf(np.diag(distance))), "Real-B diagonal is not -Inf")
    _assert(bool(mask.diagonal().all()), "Real-B mask diagonal is not True")
    rng = np.random.default_rng(20260916)
    pairs = rng.integers(0, STATE_COUNT, size=(20, 2))
    waveform_cache: dict[int, np.ndarray] = {}
    maximum_error = 0.0
    for real_id, query_id in pairs:
        for state_id in (int(real_id), int(query_id)):
            if state_id not in waveform_cache:
                data = load_by_id(state_id)
                partition = build_partition_from_xin(np.asarray(data["xin"]))
                waveform_cache[state_id] = np.asarray(
                    build_off_segments(data, partition)["B"].output[2:],
                    dtype=np.complex128,
                )
        direct = cnmse(waveform_cache[int(real_id)], waveform_cache[int(query_id)])
        stored = float(distance[int(real_id), int(query_id)])
        if np.isneginf(direct) and np.isneginf(stored):
            continue
        maximum_error = max(maximum_error, abs(direct - stored))
    _assert(maximum_error <= 1e-9, f"Real-B random-pair error too large: {maximum_error}")
    return {
        "shape": list(distance.shape),
        "diagonal_negative_infinity": True,
        "mask_diagonal_true": True,
        "random_pair_count": 20,
        "random_pair_max_abs_error_dB": maximum_error,
        "pass": True,
    }


def _ranked(row: np.ndarray) -> np.ndarray:
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    return np.lexsort((state_ids, row))


def _check_candidate_matrices(root: Path) -> dict[str, Any]:
    summary = pd.read_csv(root / "08_candidate_retrieval_summary.csv")
    query_metrics = pd.read_csv(root / "09_candidate_query_metrics_long.csv")
    shareability = np.load(root / "05_realB_shareability_mask.npy")
    matrix_dir = root / "distance_matrices"
    rng = np.random.default_rng(20260916)
    candidate_count = int(summary.shape[0])
    _assert(candidate_count == 11, "candidate summary does not contain 11 candidates")
    matrix_checks = 0
    margin_checks = 0
    topk_checks = 0
    maximum_margin_error = 0.0
    for candidate in summary.sort_values("candidate_id").itertuples(index=False):
        distance = np.load(matrix_dir / f"{candidate.candidate_name}.npy")
        _assert(distance.shape == (STATE_COUNT, STATE_COUNT), f"{candidate.candidate_name} matrix shape failed")
        _assert(not np.isnan(distance).any() and not np.isposinf(distance).any(), f"{candidate.candidate_name} matrix finite gate failed")
        selected, _, _, _ = top1_retrieval(distance)
        frame = query_metrics.loc[query_metrics["candidate_id"].eq(int(candidate.candidate_id))].sort_values("State_R")
        _assert(frame.shape[0] == STATE_COUNT, f"{candidate.candidate_name} query metric row count failed")
        _assert(np.array_equal(frame["State_Q"].to_numpy(dtype=int), selected), f"{candidate.candidate_name} State_Q mismatch")
        for state_id in (0, 42, 187, 325, 424):
            order = _ranked(distance[state_id])
            _assert(int(order[0]) == int(selected[state_id]), f"{candidate.candidate_name} argmin mismatch at {state_id}")
            matrix_checks += 1
        for _ in range(2):
            state_id = int(rng.integers(0, STATE_COUNT))
            row = distance[state_id]
            share_idx = np.flatnonzero(shareability[state_id])
            nonshare_idx = np.flatnonzero(~shareability[state_id])
            _assert(share_idx.size and nonshare_idx.size, "shareable/nonshareable set is empty")
            share_order = share_idx[np.lexsort((share_idx, row[share_idx]))]
            nonshare_order = nonshare_idx[np.lexsort((nonshare_idx, row[nonshare_idx]))]
            expected_margin = float(row[nonshare_order[0]] - row[share_order[0]])
            stored = frame.loc[frame["State_R"].eq(state_id)].iloc[0]
            error = abs(expected_margin - float(stored["shareability_margin_dB"]))
            maximum_margin_error = max(maximum_margin_error, error)
            _assert(error <= 1e-12, f"{candidate.candidate_name} shareability margin mismatch")
            margin_checks += 1
            order = _ranked(row)
            for k, column in ((2, "Top2_has_shareable"), (3, "Top3_has_shareable"), (5, "Top5_has_shareable"), (10, "Top10_has_shareable")):
                expected = bool(shareability[state_id, order[:k]].any())
                _assert(expected == bool(stored[column]), f"{candidate.candidate_name} {column} mismatch")
                topk_checks += 1
    return {
        "candidate_count": candidate_count,
        "matrix_count": candidate_count,
        "argmin_checks": matrix_checks,
        "shareability_margin_checks": margin_checks,
        "topk_oracle_checks": topk_checks,
        "maximum_margin_abs_error": maximum_margin_error,
        "pass": True,
    }


def _check_cv(root: Path) -> dict[str, Any]:
    folds = pd.read_csv(root / "12_query_state_cv_folds.csv")
    cv = pd.read_csv(root / "13_query_state_cv_results.csv")
    stability = pd.read_csv(root / "14_query_state_cv_selection_stability.csv")
    _assert(folds.shape[0] == STATE_COUNT, "CV fold table row count failed")
    _assert(folds["fold"].value_counts().sort_index().to_dict() == {0: 85, 1: 85, 2: 85, 3: 85, 4: 85}, "CV folds are not balanced")
    _assert(cv.shape[0] == 55 and cv.groupby("fold").size().eq(11).all(), "CV result table shape failed")
    _assert(stability.shape[0] == 11, "CV stability table row count failed")
    _assert(cv.groupby("fold")["train_is_winner"].sum().eq(1).all(), "CV winner count per fold failed")
    return {
        "fold_count": 5,
        "fold_size": 85,
        "train_query_size": 340,
        "candidate_fold_rows": int(cv.shape[0]),
        "winner_count_per_fold": 1,
        "pass": True,
    }


def main() -> None:
    raw = raw_manifest_gate()
    real_b = _check_real_b(RESULT_ROOT)
    candidate = _check_candidate_matrices(RESULT_ROOT)
    cv = _check_cv(RESULT_ROOT)
    checks = {
        "task_name": TASK_NAME,
        "raw_manifest": raw,
        "real_B_ground_truth": real_b,
        "candidate_matrices": candidate,
        "query_state_cv": cv,
        "pass": True,
    }
    (RESULT_ROOT / "20_validation_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(checks, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
