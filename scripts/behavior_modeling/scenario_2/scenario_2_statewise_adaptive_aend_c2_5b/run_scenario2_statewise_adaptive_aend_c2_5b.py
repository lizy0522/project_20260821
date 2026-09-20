"""Run the statewise-adaptive Scenario 2 Aend/C2 5B model search."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
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
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from retrieval_oriented_model_selection.shared.scenario2_retrieval_oriented_model_scan import (  # noqa: E402
    BASELINE_CANDIDATE as EXISTING_BASELINE_CANDIDATE,
)
from retrieval_oriented_model_selection.shared.scenario2_retrieval_oriented_model_scan import (  # noqa: E402
    compute_candidate_distance_matrix,
)

from behavior_modeling.shared.basis import build_mp_basis  # noqa: E402
from behavior_modeling.shared.statewise_adaptive_model_search import (  # noqa: E402
    C2_STAGE,
    DPD_SHAREABLE_THRESHOLD_DB,
    EXISTING_CANDIDATE_COUNT,
    FORMAL_A_LENGTH,
    FORMAL_B_LENGTH,
    FORMAL_C_LENGTH,
    STATE_COUNT,
    WAVEFORM_LENGTH,
    AdaptiveCandidate,
    CandidateFit,
    PreparedStatePairs,
    candidate_grid,
    expand_candidate_shell,
    fit_candidates_for_state,
    initial_candidate_bank,
    native_feasible,
    prepare_state_pairs,
    selected_model_statistics,
)

TASK_NAME = "scenario_2_statewise_adaptive_aend_c2_5b"
RETRIEVAL_MODULE_NAME = "behavior_fingerprint_retrieval"
RETRIEVAL_TASK_NAME = "scenario_2_C2_to_Aend_retrieval"
MODEL_RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /TASK_NAME
    / "scenario_2_statewise_adaptive_Aend_C2_5B"
)
BASELINE_ROOT = (
    PROJECT_ROOT
    / "results"
    / RETRIEVAL_MODULE_NAME
    / RETRIEVAL_TASK_NAME
    / "scenario_2_C2_to_Aend"
)
EXISTING_SCAN_ROOT = (
    PROJECT_ROOT
    / "results"
    / "retrieval_oriented_model_selection"
    / "scenario_2_C2_to_Aend_retrieval_oriented_model_scan"
)
ALL_ILC_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_all_ilc_analysis"
    / "scenario_2"
)
MODEL_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"
RETRIEVAL_LOG = (
    PROJECT_ROOT
    / "work_logs"
    / RETRIEVAL_MODULE_NAME
    / RETRIEVAL_TASK_NAME
    / "execution_log.txt"
)
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
SEARCH_PROGRESS_PATH = MODEL_RESULT_ROOT / "search_progress.json"
SEARCH_AEND_PATH = MODEL_RESULT_ROOT / "candidate_search_Aend.csv.gz"
SEARCH_C2_PATH = MODEL_RESULT_ROOT / "candidate_search_C2.csv.gz"
ROUND0_CACHE_PATH = EXISTING_SCAN_ROOT / "candidate_fit_cache.npz"

PROTECTED_RESULT_DIRS = {
    "scenario_2_5B_baseline": BASELINE_ROOT,
    "scenario_2_1B": PROJECT_ROOT
    / "results"
    / RETRIEVAL_MODULE_NAME
    / "scenario_2_C2_to_Aend_1b"
    / "scenario_2_C2_to_Aend_1B",
    "scenario_2_0p5B": PROJECT_ROOT
    / "results"
    / RETRIEVAL_MODULE_NAME
    / "scenario_2_C2_to_Aend_0p5b"
    / "scenario_2_C2_to_Aend_0p5B",
    "scenario_2_equal_ABC": PROJECT_ROOT
    / "results"
    / RETRIEVAL_MODULE_NAME
    / "scenario_2_C2_to_Aend_equal_ABC",
    "scenario_2_unified_model_capacity": PROJECT_ROOT
    / "results"
    / RETRIEVAL_MODULE_NAME
    / "scenario_2_C2_to_Aend_retrieval"
    / "scenario_2_C2_to_Aend_unified_model_capacity",
    "scenario_2_retrieval_oriented": EXISTING_SCAN_ROOT,
    "scenario_2_all_ilc": ALL_ILC_ROOT,
    "low_bandwidth_operator_bank": PROJECT_ROOT
    / "results"
    / "low_bandwidth_behavior_analysis"
    / "scenario_2" /"low_bandwidth_observation"
    / "sample_rate_operator_bank",
}
PROTECTED_SCRIPT_DIRS = {
    "behavior_modeling": PROJECT_ROOT / "scripts" / "behavior_modeling",
    "signal_segmentation": PROJECT_ROOT / "scripts" / "signal_segmentation",
    "retrieval": PROJECT_ROOT / "scripts" / "retrieval_oriented_model_selection",
}


def _tree_digest(path: Path) -> str | None:
    path = Path(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*"), key=lambda item: str(item).lower()):
        if (
            not file.is_file()
            or "__pycache__" in file.parts
            or file.suffix.lower() == ".pyc"
            or file.name.startswith("~$")
        ):
            continue
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        content = file.read_bytes()
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _file_snapshot(path: Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_dir():
        return {}
    result: dict[str, str] = {}
    for file in sorted(path.rglob("*"), key=lambda item: str(item).lower()):
        if (
            not file.is_file()
            or "__pycache__" in file.parts
            or file.suffix.lower() == ".pyc"
            or file.name.startswith("~$")
        ):
            continue
        result[file.relative_to(path).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    return result


def _raw_manifest() -> dict[str, Any]:
    raw_root = PROJECT_ROOT / "data" / "raw"
    digest = hashlib.sha256()
    files = sorted(
        (file for file in raw_root.rglob("*") if file.is_file()),
        key=lambda item: str(item).lower(),
    )
    total_bytes = 0
    for file in files:
        relative = str(file.relative_to(raw_root)).replace("\\", "/")
        size = int(file.stat().st_size)
        digest.update(relative.encode() + b"\0")
        digest.update(size.to_bytes(8, "little"))
        total_bytes += size
    return {"sha256": digest.hexdigest(), "file_count": len(files), "bytes": total_bytes}


def protection_snapshot() -> dict[str, Any]:
    return {
        "data/raw": _raw_manifest(),
        "result_dirs": {
            name: {"path": str(path), "exists": path.is_dir(), "sha256": _tree_digest(path)}
            for name, path in PROTECTED_RESULT_DIRS.items()
        },
        "script_files": {
            name: _file_snapshot(path) for name, path in PROTECTED_SCRIPT_DIRS.items()
        },
    }


def verify_protection(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"data/raw": {}, "result_dirs": {}, "script_files": {}}
    result["data/raw"] = {
        "sha256_unchanged": before["data/raw"]["sha256"] == after["data/raw"]["sha256"],
        "file_count_unchanged": before["data/raw"]["file_count"] == after["data/raw"]["file_count"],
        "bytes_unchanged": before["data/raw"]["bytes"] == after["data/raw"]["bytes"],
    }
    for name, item in before["result_dirs"].items():
        after_item = after["result_dirs"][name]
        result["result_dirs"][name] = {
            "sha256_unchanged": item["sha256"] == after_item["sha256"],
            "before": item["sha256"],
            "after": after_item["sha256"],
        }
    for name, before_files in before["script_files"].items():
        after_files = after["script_files"][name]
        result["script_files"][name] = {
            relative: after_files.get(relative) == digest
            for relative, digest in before_files.items()
        }
    result["all_protected_unchanged"] = bool(
        all(result["data/raw"].values())
        and all(item["sha256_unchanged"] for item in result["result_dirs"].values())
        and all(all(values.values()) for values in result["script_files"].values())
    )
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _json_safe(value.item())
    if isinstance(value, (float,)):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip", encoding="utf-8-sig", float_format="%.17g")


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _append_log(path: Path, title: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] {title}\n")
        handle.write(text.rstrip() + "\n")


def _load_round0_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not ROUND0_CACHE_PATH.is_file():
        raise FileNotFoundError(f"missing existing candidate fit cache: {ROUND0_CACHE_PATH}")
    with np.load(ROUND0_CACHE_PATH, allow_pickle=False) as data:
        required = {"candidate_ids", "state_ids", "theta_Aend", "theta_C2", "metrics", "ilc_A_end"}
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(f"round0 cache missing fields: {missing}")
        candidate_ids = np.asarray(data["candidate_ids"], dtype=np.int64)
        state_ids = np.asarray(data["state_ids"], dtype=np.int64)
        theta_a = np.asarray(data["theta_Aend"], dtype=np.complex128)
        theta_c = np.asarray(data["theta_C2"], dtype=np.complex128)
        metrics = np.asarray(data["metrics"], dtype=np.float64)
        ilc_counts = np.asarray(data["ilc_A_end"], dtype=np.int64)
    if not np.array_equal(candidate_ids, np.arange(1, EXISTING_CANDIDATE_COUNT + 1)):
        raise RuntimeError("round0 candidate IDs are not 1...200")
    if not np.array_equal(state_ids, np.arange(STATE_COUNT)):
        raise RuntimeError("round0 state IDs are not 0...424")
    if theta_a.shape != theta_c.shape or theta_a.shape[0:2] != (
        EXISTING_CANDIDATE_COUNT,
        STATE_COUNT,
    ):
        raise RuntimeError(f"round0 theta shape mismatch: {theta_a.shape}/{theta_c.shape}")
    if metrics.shape != (EXISTING_CANDIDATE_COUNT, STATE_COUNT, 4):
        raise RuntimeError(f"round0 metric shape mismatch: {metrics.shape}")
    if ilc_counts.shape != (STATE_COUNT,) or np.any(ilc_counts < 2) or np.any(ilc_counts > 5):
        raise RuntimeError("round0 Aend stage map is not 2...5 for all states")
    if not np.all(np.isfinite(theta_a)) or not np.all(np.isfinite(theta_c)):
        raise RuntimeError("round0 theta contains NaN/Inf")
    if not np.all(np.isfinite(metrics)):
        raise RuntimeError("round0 metrics contain NaN/Inf")
    return theta_a, theta_c, metrics, ilc_counts, candidate_ids


def _formal_common_b_input() -> np.ndarray:
    with np.load(BASELINE_ROOT / "lut_fingerprints_Aend.npz", allow_pickle=False) as data:
        if "common_B_input" not in data.files:
            raise RuntimeError("formal baseline lacks common_B_input")
        common_b = np.asarray(data["common_B_input"], dtype=np.complex128)
    if common_b.shape != (FORMAL_B_LENGTH,) or not np.all(np.isfinite(common_b)):
        raise RuntimeError("formal common_B_input must be finite complex length 4915")
    return common_b


def _load_baseline_retrieval() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    results = pd.read_csv(BASELINE_ROOT / "retrieval_results.csv")
    diagnostics = pd.read_csv(BASELINE_ROOT / "retrieved_real_B_cnmse.csv")
    if results.shape[0] != STATE_COUNT or diagnostics.shape[0] != STATE_COUNT:
        raise RuntimeError("formal baseline retrieval must contain 425 rows")
    state_q = results.sort_values("State_n_R")["State_n_Q"].to_numpy(dtype=np.int64)
    real_b = diagnostics.sort_values("State_n_R")["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    return results, state_q, real_b


def _baseline_regression(
    theta_a: np.ndarray,
    theta_c: np.ndarray,
    metrics: np.ndarray,
    ilc_counts: np.ndarray,
    common_b_input: np.ndarray,
) -> dict[str, Any]:
    """Rebuild the frozen P9/M0/lambda baseline through this public path."""

    candidate = initial_candidate_bank()[EXISTING_BASELINE_CANDIDATE.candidate_id - 1]
    if (
        not candidate.is_baseline
        or candidate.candidate_id != EXISTING_BASELINE_CANDIDATE.candidate_id
    ):
        raise RuntimeError("baseline candidate anchor mismatch")
    phi = build_mp_basis(common_b_input, candidate.orders, candidate.memory_definition)
    fp_a = (phi @ theta_a[candidate.candidate_id - 1, :, : candidate.coefficient_count].T).T
    fp_c = (phi @ theta_c[candidate.candidate_id - 1, :, : candidate.coefficient_count].T).T
    old_lut = np.load(BASELINE_ROOT / "lut_fingerprints_Aend.npz", allow_pickle=False)
    old_query = np.load(BASELINE_ROOT / "query_fingerprints_C2.npz", allow_pickle=False)
    old_lut_fp = np.asarray(old_lut["Y_Aend_fingerprints"], dtype=np.complex128)
    old_query_fp = np.asarray(old_query["Q_C2"], dtype=np.complex128)
    lut_error = float(np.max(np.abs(fp_a - old_lut_fp)))
    query_error = float(np.max(np.abs(fp_c - old_query_fp)))
    new_distance = compute_candidate_distance_matrix(fp_c, fp_a)
    old_distance = np.asarray(
        np.load(BASELINE_ROOT / "retrieval_distance_matrix.npz", allow_pickle=False)["D_C2_Aend"],
        dtype=np.float64,
    )
    finite = np.isfinite(new_distance) & np.isfinite(old_distance)
    distance_error = float(np.max(np.abs(new_distance[finite] - old_distance[finite])))
    distance_pattern_equal = bool(
        np.array_equal(np.isneginf(new_distance), np.isneginf(old_distance))
    )
    old_results, old_state_q, old_real_b = _load_baseline_retrieval()
    # Candidate-cache metrics use the same native support as the formal all-ILC metrics.
    all_metrics = pd.read_csv(ALL_ILC_ROOT / "all_ilc_model_metrics.csv")
    metric_error = 0.0
    for state_id, ilc_count in enumerate(ilc_counts.tolist()):
        row_a = all_metrics.loc[
            (all_metrics["state_id"] == state_id)
            & (all_metrics["actual_ilc_n"] == int(ilc_count))
            & (all_metrics["model_role"] == "Y-A")
        ]
        row_c = all_metrics.loc[
            (all_metrics["state_id"] == state_id)
            & (all_metrics["actual_ilc_n"] == C2_STAGE)
            & (all_metrics["model_role"] == "Y-C")
        ]
        if row_a.shape[0] != 1 or row_c.shape[0] != 1:
            raise RuntimeError(f"all-ILC baseline metric row mismatch at state {state_id}")
        expected = np.array(
            [
                float(row_a.iloc[0]["train_NMSE_dB"]),
                float(row_a.iloc[0]["B_generalization_NMSE_dB"]),
                float(row_c.iloc[0]["train_NMSE_dB"]),
                float(row_c.iloc[0]["B_generalization_NMSE_dB"]),
            ]
        )
        observed = metrics[candidate.candidate_id - 1, state_id]
        metric_error = max(metric_error, float(np.max(np.abs(expected - observed))))
    retrieved_real_b = old_real_b
    exact = old_results.sort_values("State_n_R")["exact_hit"].to_numpy(dtype=bool)
    shareable = np.isneginf(retrieved_real_b) | (retrieved_real_b < DPD_SHAREABLE_THRESHOLD_DB)
    failure_ids = np.flatnonzero(~shareable).tolist()
    state_ids_grid = np.broadcast_to(np.arange(STATE_COUNT, dtype=np.int64), new_distance.shape)
    rebuilt_state_q = np.lexsort((state_ids_grid, new_distance), axis=1)[:, 0]
    pass_value = bool(
        lut_error <= 1e-12
        and query_error <= 1e-12
        and distance_pattern_equal
        and distance_error <= 1e-12
        and metric_error <= 1e-12
        and np.array_equal(old_state_q, rebuilt_state_q)
        and int(exact.sum()) == 218
        and int(shareable.sum()) == 411
        and len(failure_ids) == 14
    )
    return {
        "pass": pass_value,
        "candidate_id": candidate.candidate_id,
        "orders": list(candidate.orders),
        "memory_definition": candidate.memory_definition,
        "lambda": candidate.ridge_lambda,
        "fingerprint_max_abs_error": {"Aend": lut_error, "C2": query_error},
        "distance_max_abs_error": distance_error,
        "distance_nonfinite_pattern_equal": distance_pattern_equal,
        "model_metric_max_abs_error": metric_error,
        "state_Q_equal": bool(
            np.array_equal(
                old_state_q,
                rebuilt_state_q,
            )
        ),
        "observed_exact": int(exact.sum()),
        "observed_shareable": int(shareable.sum()),
        "observed_failure": int((~shareable).sum()),
        "failure_ids": failure_ids,
        "expected_failure_ids": [
            187,
            189,
            195,
            196,
            199,
            206,
            323,
            327,
            330,
            335,
            340,
            344,
            346,
            354,
        ],
    }


def _record_from_round0(
    candidates: Sequence[AdaptiveCandidate],
    theta_a: np.ndarray,
    theta_c: np.ndarray,
    metrics: np.ndarray,
    ilc_counts: np.ndarray,
) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[tuple[str, int, int], np.ndarray]]:
    records: dict[tuple[str, int, int], dict[str, Any]] = {}
    theta_cache: dict[tuple[str, int, int], np.ndarray] = {}
    for candidate in candidates:
        index = candidate.candidate_id - 1
        for state_id in range(STATE_COUNT):
            for side, metric_offset, theta_source in (
                ("Aend", 0, theta_a),
                ("C2", 2, theta_c),
            ):
                row = {
                    "state_id": state_id,
                    "side": side,
                    "candidate_id": candidate.candidate_id,
                    "search_round": candidate.search_round,
                    "source": candidate.source,
                    "order_profile": candidate.order_profile,
                    "orders": json.dumps(list(candidate.orders), separators=(",", ":")),
                    "memory_profile": candidate.memory_profile,
                    "memory_definition": json.dumps(
                        candidate.memory_definition, sort_keys=True, separators=(",", ":")
                    ),
                    "lambda": candidate.ridge_lambda,
                    "max_delay": candidate.max_delay,
                    "coefficient_count": candidate.coefficient_count,
                    "structure_id": candidate.structure_id,
                    "native_train_NMSE_dB": float(metrics[index, state_id, metric_offset]),
                    "native_B_NMSE_dB": float(metrics[index, state_id, metric_offset + 1]),
                    "rank": candidate.coefficient_count,
                    "condition_number_phi": np.nan,
                    "condition_number_augmented": np.nan,
                    "n_train_samples": (FORMAL_A_LENGTH if side == "Aend" else FORMAL_C_LENGTH)
                    - candidate.max_delay,
                    "theta_l2_norm": float(
                        np.linalg.norm(theta_source[index, state_id, : candidate.coefficient_count])
                    ),
                    "finite": True,
                    "valid": True,
                    "failure_reason": "",
                    "ilc_A_end": int(ilc_counts[state_id]),
                }
                records[(side, state_id, candidate.candidate_id)] = row
                theta_cache[(side, state_id, candidate.candidate_id)] = np.asarray(
                    theta_source[index, state_id, : candidate.coefficient_count],
                    dtype=np.complex128,
                )
    return records, theta_cache


def _index_records(
    records: Mapping[tuple[str, int, int], dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    result: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in records.values():
        result.setdefault((str(row["side"]), int(row["state_id"])), []).append(row)
    for rows in result.values():
        rows.sort(key=lambda item: int(item["candidate_id"]))
    return result


def _native_selection(
    record_index: Mapping[tuple[str, int], Sequence[dict[str, Any]]],
) -> tuple[dict[tuple[str, int], int], dict[str, list[int]]]:
    selected: dict[tuple[str, int], int] = {}
    unresolved = {"Aend": [], "C2": []}
    for side in ("Aend", "C2"):
        for state_id in range(STATE_COUNT):
            options = [
                row for row in record_index.get((side, state_id), ()) if native_feasible(row)
            ]
            if not options:
                unresolved[side].append(state_id)
                continue
            best = min(
                options,
                key=lambda row: (
                    int(row["coefficient_count"]),
                    int(row["max_delay"]),
                    -float(row["lambda"]),
                    max(float(row["native_train_NMSE_dB"]), float(row["native_B_NMSE_dB"])),
                    int(row["candidate_id"]),
                ),
            )
            selected[(side, state_id)] = int(best["candidate_id"])
    return selected, unresolved


def _theta_getter(
    theta_round0: Mapping[tuple[str, int, int], np.ndarray],
    theta_expanded: Mapping[tuple[str, int, int], np.ndarray],
    side: str,
    state_id: int,
    candidate_id: int,
) -> np.ndarray | None:
    key = (side, int(state_id), int(candidate_id))
    if key in theta_expanded:
        return theta_expanded[key]
    return theta_round0.get(key)


def _common_metrics_for_state(
    prepared: PreparedStatePairs,
    side: str,
    candidate: AdaptiveCandidate,
    theta: np.ndarray,
    common_delay: int,
) -> tuple[float, float]:
    canonical = prepared.a_end if side == "Aend" else prepared.c2
    train_name = "A" if side == "Aend" else "C"
    train = canonical[train_name]
    b = canonical["B"]
    phi_train = build_mp_basis(train.input, candidate.orders, candidate.memory_definition)
    phi_b = build_mp_basis(b.input, candidate.orders, candidate.memory_definition)
    native_train = phi_train @ theta
    native_b = phi_b @ theta
    offset = common_delay - candidate.max_delay
    if offset < 0:
        raise ValueError("common support precedes native support")
    train_nmse = calculate_nmse(
        np.asarray(train.output[common_delay:], dtype=np.complex128), native_train[offset:]
    )
    b_nmse = calculate_nmse(
        np.asarray(b.output[common_delay:], dtype=np.complex128), native_b[offset:]
    )
    if not np.isfinite(train_nmse) or not np.isfinite(b_nmse):
        raise ValueError("common NMSE is non-finite")
    return float(train_nmse), float(b_nmse)


def calculate_nmse(y_ref: np.ndarray, y_pred: np.ndarray) -> float:
    from behavior_modeling.shared.evaluation import calculate_nmse as _calculate_nmse

    return _calculate_nmse(y_ref, y_pred)


def _common_select_pass(
    records: Mapping[tuple[str, int, int], dict[str, Any]],
    record_index: Mapping[tuple[str, int], Sequence[dict[str, Any]]],
    candidates_by_id: Mapping[int, AdaptiveCandidate],
    theta_round0: Mapping[tuple[str, int, int], np.ndarray],
    theta_expanded: Mapping[tuple[str, int, int], np.ndarray],
    selected_native: Mapping[tuple[str, int], int],
    common_delay: int,
) -> tuple[
    dict[tuple[str, int], int], dict[str, list[int]], dict[tuple[str, int], tuple[float, float]]
]:
    selected_common: dict[tuple[str, int], int] = {}
    unresolved = {"Aend": [], "C2": []}
    metric_by_key: dict[tuple[str, int], tuple[float, float]] = {}
    for state_id in range(STATE_COUNT):
        prepared = prepare_state_pairs(state_id)
        for side in ("Aend", "C2"):
            options = [
                row for row in record_index.get((side, state_id), ()) if native_feasible(row)
            ]
            candidates_pass: list[tuple[dict[str, Any], float, float]] = []
            for row in options:
                candidate = candidates_by_id[int(row["candidate_id"])]
                theta = _theta_getter(
                    theta_round0, theta_expanded, side, state_id, candidate.candidate_id
                )
                if theta is None:
                    continue
                try:
                    train_nmse, b_nmse = _common_metrics_for_state(
                        prepared, side, candidate, theta, common_delay
                    )
                except Exception:
                    continue
                if train_nmse < DPD_SHAREABLE_THRESHOLD_DB and b_nmse < DPD_SHAREABLE_THRESHOLD_DB:
                    candidates_pass.append((row, train_nmse, b_nmse))
            if not candidates_pass:
                unresolved[side].append(state_id)
                continue
            best_row, best_train, best_b = min(
                candidates_pass,
                key=lambda item: (
                    int(item[0]["coefficient_count"]),
                    int(item[0]["max_delay"]),
                    -float(item[0]["lambda"]),
                    max(item[1], item[2]),
                    int(item[0]["candidate_id"]),
                ),
            )
            selected_common[(side, state_id)] = int(best_row["candidate_id"])
            metric_by_key[(side, state_id)] = (best_train, best_b)
    return selected_common, unresolved, metric_by_key


def _fit_progressive_target_state(
    prepared: PreparedStatePairs,
    candidates: list[AdaptiveCandidate],
    target_sides: set[str],
) -> tuple[list[CandidateFit], dict[tuple[str, int], np.ndarray]]:
    """Fit structure groups in selection order and stop after target sides resolve.

    Each group contains one order/memory structure and every lambda in the
    current shell.  All lambdas for a structure are evaluated together, while
    structures with larger coefficient count/delay are skipped once every
    requested side has a feasible structure.  This preserves the documented
    complexity-first tie rule without fitting irrelevant high-complexity shells.
    """

    groups: dict[tuple[Any, ...], list[AdaptiveCandidate]] = {}
    for candidate in candidates:
        structure_key = (
            candidate.memory_profile,
            candidate.orders,
            tuple(sorted(candidate.memory_definition.items())),
        )
        groups.setdefault(structure_key, []).append(candidate)
    ordered_groups = sorted(
        groups.values(),
        key=lambda group: (
            min(candidate.coefficient_count for candidate in group),
            min(candidate.max_delay for candidate in group),
            min(candidate.candidate_id for candidate in group),
        ),
    )
    all_fits: list[CandidateFit] = []
    theta_store: dict[tuple[str, int], np.ndarray] = {}
    best_structure: dict[str, tuple[int, int] | None] = {side: None for side in target_sides}
    for group in ordered_groups:
        structure_complexity = (
            group[0].coefficient_count,
            group[0].max_delay,
        )
        if all(value is not None for value in best_structure.values()):
            largest_selected_structure = max(
                value for value in best_structure.values() if value is not None
            )
            if structure_complexity > largest_selected_structure:
                break
        fits, group_theta = fit_candidates_for_state(prepared, group)
        all_fits.extend(fits)
        theta_store.update(group_theta)
        for side in target_sides:
            feasible = [
                fit.row for fit in fits if fit.row["side"] == side and native_feasible(fit.row)
            ]
            if feasible:
                current = min(
                    (int(row["coefficient_count"]), int(row["max_delay"])) for row in feasible
                )
                if best_structure[side] is None or current < best_structure[side]:
                    best_structure[side] = current
    return all_fits, theta_store


def _selected_frame(
    selected: Mapping[tuple[str, int], int],
    common_metrics: Mapping[tuple[str, int], tuple[float, float]],
    record_index: Mapping[tuple[str, int], Sequence[dict[str, Any]]],
    candidates_by_id: Mapping[int, AdaptiveCandidate],
    ilc_counts: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for side in ("Aend", "C2"):
        for state_id in range(STATE_COUNT):
            candidate_id = selected.get((side, state_id))
            if candidate_id is None:
                continue
            match = [
                row
                for row in record_index[(side, state_id)]
                if int(row["candidate_id"]) == candidate_id
            ]
            if len(match) != 1:
                raise RuntimeError("selected candidate missing from cache")
            row = dict(match[0])
            candidate = candidates_by_id[candidate_id]
            common_train, common_b = common_metrics.get((side, state_id), (np.nan, np.nan))
            row.update(
                {
                    "common_train_NMSE_dB": common_train,
                    "common_B_NMSE_dB": common_b,
                    "candidate_count_evaluated": len(record_index[(side, state_id)]),
                    "ilc_A_end": int(ilc_counts[state_id]),
                    "ilc_C_stage": 2,
                    "basis_terms": json.dumps(candidate.basis_terms, separators=(",", ":")),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["side", "state_id"]).reset_index(drop=True)


def _save_selected_coefficients(
    selected: Mapping[tuple[str, int], int],
    candidates_by_id: Mapping[int, AdaptiveCandidate],
    theta_round0: Mapping[tuple[str, int, int], np.ndarray],
    theta_expanded: Mapping[tuple[str, int, int], np.ndarray],
) -> None:
    arrays_a: dict[str, np.ndarray] = {}
    arrays_c: dict[str, np.ndarray] = {}
    descriptors: dict[str, list[dict[str, Any]]] = {"Aend": [], "C2": []}
    for side in ("Aend", "C2"):
        for state_id in range(STATE_COUNT):
            candidate_id = selected.get((side, state_id))
            if candidate_id is None:
                continue
            candidate = candidates_by_id[candidate_id]
            theta = _theta_getter(theta_round0, theta_expanded, side, state_id, candidate_id)
            if theta is None:
                raise RuntimeError("selected theta is missing")
            key = f"state_{state_id:03d}"
            (arrays_a if side == "Aend" else arrays_c)[key] = np.asarray(theta, dtype=np.complex128)
            descriptors[side].append(
                {
                    "state_id": state_id,
                    "candidate_id": candidate_id,
                    "orders": list(candidate.orders),
                    "basis_terms": [list(term) for term in candidate.basis_terms],
                    "memory": candidate.memory_definition,
                    "max_delay": candidate.max_delay,
                    "coefficient_count": candidate.coefficient_count,
                    "lambda": candidate.ridge_lambda,
                    "search_round": candidate.search_round,
                }
            )
    _write_npz(MODEL_RESULT_ROOT / "selected_Aend_coefficients.npz", arrays_a)
    _write_npz(MODEL_RESULT_ROOT / "selected_C2_coefficients.npz", arrays_c)
    _write_json(MODEL_RESULT_ROOT / "selected_model_descriptors.json", descriptors)


def main() -> None:
    print("Scenario 2 statewise-adaptive Aend/C2 5B model search started.", flush=True)
    MODEL_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    before = protection_snapshot()
    common_b_input = _formal_common_b_input()
    theta_a, theta_c, metrics, ilc_counts, _ = _load_round0_cache()
    baseline_regression = _baseline_regression(
        theta_a, theta_c, metrics, ilc_counts, common_b_input
    )
    _write_json(MODEL_RESULT_ROOT / "baseline_regression.json", baseline_regression)
    if not baseline_regression["pass"]:
        raise RuntimeError(f"baseline regression failed: {baseline_regression}")
    print("Baseline guard passed: Exact=218, Shareable=411, Failure=14.", flush=True)

    candidates = list(initial_candidate_bank())
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    records, theta_round0 = _record_from_round0(candidates, theta_a, theta_c, metrics, ilc_counts)
    theta_expanded: dict[tuple[str, int, int], np.ndarray] = {}
    record_index = _index_records(records)
    selected_native, native_unresolved = _native_selection(record_index)
    progress: dict[str, Any] = {
        "experiment": "scenario_2_statewise_adaptive_Aend_C2_5B",
        "state_count": STATE_COUNT,
        "observation_bandwidth": "5B",
        "low_bandwidth_operator_used": False,
        "ABC_changed": False,
        "initial_candidate_count": len(candidates),
        "rounds": [],
        "native_unresolved": native_unresolved,
        "common_support_history": [],
        "search_stop_reason": None,
    }
    print(
        f"Round 0 native feasible: Aend={STATE_COUNT - len(native_unresolved['Aend'])}/425, "
        f"C2={STATE_COUNT - len(native_unresolved['C2'])}/425.",
        flush=True,
    )

    selected_final: dict[tuple[str, int], int] = {}
    common_metrics_final: dict[tuple[str, int], tuple[float, float]] = {}
    common_delay_final: int | None = None
    unresolved_final = {
        "Aend": list(native_unresolved["Aend"]),
        "C2": list(native_unresolved["C2"]),
    }
    search_round = 0
    while True:
        record_index = _index_records(records)
        selected_native, native_unresolved = _native_selection(record_index)
        if native_unresolved["Aend"] or native_unresolved["C2"]:
            unresolved_targets = {
                (side, state_id) for side in ("Aend", "C2") for state_id in native_unresolved[side]
            }
            reason = "native_unresolved"
        else:
            candidate_delay = max(
                candidates_by_id[candidate_id].max_delay
                for candidate_id in selected_native.values()
            )
            if common_delay_final is not None and candidate_delay < common_delay_final:
                candidate_delay = common_delay_final
            common_selected, common_unresolved, common_metrics = _common_select_pass(
                records=records,
                record_index=record_index,
                candidates_by_id=candidates_by_id,
                theta_round0=theta_round0,
                theta_expanded=theta_expanded,
                selected_native=selected_native,
                common_delay=candidate_delay,
            )
            progress["common_support_history"].append(
                {
                    "search_round": search_round,
                    "M_common": candidate_delay,
                    "common_A_length": FORMAL_A_LENGTH - candidate_delay,
                    "common_B_length": FORMAL_B_LENGTH - candidate_delay,
                    "common_C_length": FORMAL_C_LENGTH - candidate_delay,
                    "unresolved_Aend": common_unresolved["Aend"],
                    "unresolved_C2": common_unresolved["C2"],
                }
            )
            if not common_unresolved["Aend"] and not common_unresolved["C2"]:
                selected_final = common_selected
                common_metrics_final = common_metrics
                common_delay_final = candidate_delay
                unresolved_final = {"Aend": [], "C2": []}
                progress["search_stop_reason"] = "all_850_models_pass_common_support"
                break
            selected_native = common_selected
            unresolved_targets = {
                (side, state_id) for side in ("Aend", "C2") for state_id in common_unresolved[side]
            }
            reason = "common_support_unresolved"

        search_round += 1
        new_candidates = expand_candidate_shell(candidates, search_round)
        usable = [
            candidate
            for candidate in new_candidates
            if candidate.coefficient_count < FORMAL_B_LENGTH
        ]
        progress["rounds"].append(
            {
                "search_round": search_round,
                "reason": reason,
                "target_count": len(unresolved_targets),
                "new_candidate_count": len(new_candidates),
                "usable_new_candidate_count": len(usable),
                "candidate_count_before": len(candidates),
            }
        )
        if not usable:
            progress["search_stop_reason"] = "basis_support_blocker_no_usable_new_candidate_shell"
            unresolved_final = {
                "Aend": sorted(state_id for side, state_id in unresolved_targets if side == "Aend"),
                "C2": sorted(state_id for side, state_id in unresolved_targets if side == "C2"),
            }
            break
        candidates.extend(new_candidates)
        candidates_by_id.update({candidate.candidate_id: candidate for candidate in new_candidates})
        target_states = sorted({state_id for _, state_id in unresolved_targets})
        for position, state_id in enumerate(target_states, start=1):
            prepared = prepare_state_pairs(state_id)
            if not np.array_equal(prepared.common_b_input, common_b_input):
                raise RuntimeError(f"state={state_id} B probe differs from formal common probe")
            target_sides = {side for side, _ in unresolved_targets}
            fits, theta_new = _fit_progressive_target_state(
                prepared,
                usable,
                target_sides,
            )
            for fit in fits:
                row = dict(fit.row)
                row["ilc_A_end"] = prepared.ilc_A_end
                key = (str(row["side"]), int(state_id), int(row["candidate_id"]))
                records[key] = row
                if fit.theta is not None:
                    theta_expanded[key] = fit.theta
            if position % 5 == 0 or position == len(target_states):
                _write_json(
                    SEARCH_PROGRESS_PATH,
                    {
                        **progress,
                        "active_search_round": search_round,
                        "completed_target_states": position,
                        "target_states": len(target_states),
                        "candidate_count": len(candidates),
                    },
                )
                print(
                    f"Round {search_round}: fitted {position}/{len(target_states)} target states; "
                    f"candidate bank={len(candidates)}.",
                    flush=True,
                )
        record_index = _index_records(records)
        rows_frame = pd.DataFrame(list(records.values())).sort_values(
            ["side", "state_id", "candidate_id"]
        )
        _write_frame(rows_frame.loc[rows_frame["side"] == "Aend"], SEARCH_AEND_PATH)
        _write_frame(rows_frame.loc[rows_frame["side"] == "C2"], SEARCH_C2_PATH)
        _write_json(
            SEARCH_PROGRESS_PATH,
            {
                **progress,
                "active_search_round": search_round,
                "candidate_count": len(candidates),
                "record_count": len(records),
            },
        )

    # Persist final candidate grid and the selected bank even when the search
    # is blocked; this makes partial diagnostics reproducible and explicit.
    _write_frame(candidate_grid(candidates), MODEL_RESULT_ROOT / "candidate_model_grid.csv.gz")
    record_index = _index_records(records)
    if selected_final:
        selected_frame = _selected_frame(
            selected_final,
            common_metrics_final,
            record_index,
            candidates_by_id,
            ilc_counts,
        )
        _write_frame(selected_frame, MODEL_RESULT_ROOT / "selected_models_statewise.csv.gz")
        _save_selected_coefficients(selected_final, candidates_by_id, theta_round0, theta_expanded)
        aend_frame = selected_frame.loc[selected_frame["side"] == "Aend"].copy()
        c2_frame = selected_frame.loc[selected_frame["side"] == "C2"].copy()
        _write_frame(aend_frame, MODEL_RESULT_ROOT / "selected_Aend_models.csv.gz")
        _write_frame(c2_frame, MODEL_RESULT_ROOT / "selected_C2_models.csv.gz")
        _write_json(
            MODEL_RESULT_ROOT / "model_complexity_statistics.json",
            selected_model_statistics(selected_frame),
        )
    else:
        selected_frame = pd.DataFrame()

    stage_distribution = {
        str(int(key)): int(value)
        for key, value in pd.Series(ilc_counts).value_counts().sort_index().items()
    }
    formal_complete = bool(
        selected_frame.shape[0] == 2 * STATE_COUNT
        and common_delay_final is not None
        and np.all(
            selected_frame["common_train_NMSE_dB"].to_numpy(dtype=float)
            < DPD_SHAREABLE_THRESHOLD_DB
        )
        and np.all(
            selected_frame["common_B_NMSE_dB"].to_numpy(dtype=float) < DPD_SHAREABLE_THRESHOLD_DB
        )
    )
    if common_delay_final is None and progress["common_support_history"]:
        common_delay_final = int(progress["common_support_history"][-1]["M_common"])
    model_validation = {
        "experiment": "scenario_2_C2_to_Aend_statewise_adaptive_model_5B",
        "observation_bandwidth": "5B",
        "observation_sampling_rate_Hz": 100_000_000,
        "low_bandwidth_operator_used": False,
        "state_count": STATE_COUNT,
        "ABC_changed": False,
        "ABC_ownership": {
            "A": [0, FORMAL_A_LENGTH],
            "B": [FORMAL_A_LENGTH, FORMAL_A_LENGTH + FORMAL_B_LENGTH],
            "C": [FORMAL_A_LENGTH + FORMAL_B_LENGTH, WAVEFORM_LENGTH],
        },
        "statewise_model_selection": True,
        "Aend_and_C2_can_use_different_models": True,
        "required_metrics_strict": {
            "Aend_train_NMSE_dB": "< -40",
            "Aend_B_NMSE_dB": "< -40",
            "C2_train_NMSE_dB": "< -40",
            "C2_B_NMSE_dB": "< -40",
        },
        "Aend_model_pass_count": int(
            np.count_nonzero(
                (selected_frame["side"] == "Aend")
                & (selected_frame["common_train_NMSE_dB"] < DPD_SHAREABLE_THRESHOLD_DB)
                & (selected_frame["common_B_NMSE_dB"] < DPD_SHAREABLE_THRESHOLD_DB)
            )
            if not selected_frame.empty
            else 0
        ),
        "C2_model_pass_count": int(
            np.count_nonzero(
                (selected_frame["side"] == "C2")
                & (selected_frame["common_train_NMSE_dB"] < DPD_SHAREABLE_THRESHOLD_DB)
                & (selected_frame["common_B_NMSE_dB"] < DPD_SHAREABLE_THRESHOLD_DB)
            )
            if not selected_frame.empty
            else 0
        ),
        "formal_model_bank_complete": formal_complete,
        "unresolved_Aend_state_ids": unresolved_final["Aend"],
        "unresolved_C2_state_ids": unresolved_final["C2"],
        "global_common_max_delay": common_delay_final,
        "common_support": (
            None
            if common_delay_final is None
            else {
                "A": f"A[{common_delay_final}:]",
                "B": f"B[{common_delay_final}:]",
                "C": f"C[{common_delay_final}:]",
                "A_length": FORMAL_A_LENGTH - common_delay_final,
                "B_length": FORMAL_B_LENGTH - common_delay_final,
                "C_length": FORMAL_C_LENGTH - common_delay_final,
                "retention_ratio": {
                    "A": (FORMAL_A_LENGTH - common_delay_final) / FORMAL_A_LENGTH,
                    "B": (FORMAL_B_LENGTH - common_delay_final) / FORMAL_B_LENGTH,
                    "C": (FORMAL_C_LENGTH - common_delay_final) / FORMAL_C_LENGTH,
                },
            }
        ),
        "ilc_A_end_stage_distribution": stage_distribution,
        "C2_available_count": STATE_COUNT,
        "candidate_count_total": len(candidates),
        "candidate_search_rounds": progress["rounds"],
        "search_stop_reason": progress["search_stop_reason"],
        "native_unresolved_final": native_unresolved,
        "baseline_regression": baseline_regression,
        "model_selection_uses_Y_Aend_B_generalization": True,
        "model_selection_uses_Y_C2_B_generalization": True,
        "OFF_RealB_used_for_model_selection": False,
        "raw_data_modified": False,
        "output_files": {
            path.name: str(path) for path in sorted(MODEL_RESULT_ROOT.iterdir()) if path.is_file()
        },
    }
    after = protection_snapshot()
    protection = verify_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw or protected existing result/script changed")
    model_validation["protection_before"] = before
    model_validation["protection_after"] = after
    model_validation["protection_verification"] = protection
    _write_json(MODEL_RESULT_ROOT / "validation.json", model_validation)
    _write_json(SEARCH_PROGRESS_PATH, {**progress, "final_validation": model_validation})

    log_lines = [
        "研究类型：Scenario 2 statewise-adaptive MP model search；只使用完整5B。",
        "Baseline regression="
        + json.dumps(baseline_regression, ensure_ascii=False, separators=(",", ":")),
        f"Aend stage distribution={stage_distribution}；C2 available={STATE_COUNT}/425。",
        f"candidate_count_total={len(candidates)}；formal_model_bank_complete={formal_complete}；"
        f"M_common={common_delay_final}；search_stop_reason={progress['search_stop_reason']}。",
        f"unresolved_Aend={unresolved_final['Aend']}；unresolved_C2={unresolved_final['C2']}。",
        f"raw/protected unchanged={protection['all_protected_unchanged']}。",
        f"model result root={MODEL_RESULT_ROOT}。",
    ]
    _append_log(
        MODEL_LOG, "Scenario 2 statewise-adaptive Aend/C2 5B model search", "\n".join(log_lines)
    )
    _append_log(
        RETRIEVAL_LOG,
        "Scenario 2 statewise-adaptive Aend/C2 5B model search dependency",
        "\n".join(log_lines),
    )
    _append_log(
        HANDOFF_LOG,
        "Scenario 2 statewise-adaptive Aend/C2 5B模型搜索",
        "；".join(log_lines),
    )
    print("Statewise-adaptive model search completed.", flush=True)
    print(f"Formal model bank complete: {formal_complete}", flush=True)
    print(f"Results: {MODEL_RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
