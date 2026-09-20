"""Run and persist the Scenario 2 unified model-capacity experiment."""

from __future__ import annotations

import hashlib
import json
import sys
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

from behavior_fingerprint_ranking_consistency.shared.scenario2_all_ilc_analysis import (  # noqa: E402
    load_frozen_scenario2_artifacts,
)
from data_management.shared import build_state_table  # noqa: E402

from behavior_modeling.scenario_2.scenario_2_unified_model_capacity_scan.plot_scenario2_unified_model_capacity_scan import (  # noqa: E402,E501
    plot_candidate_scan,
    plot_gain_correlations,
    plot_pa_quartile_gains,
    plot_retrieval_comparison,
    plot_state_model_metrics,
    plot_transition_classes,
)
from behavior_modeling.shared.scenario2_unified_model_capacity_scan import (  # noqa: E402
    BASELINE_LAMBDA,
    BASELINE_MEMORY,
    BASELINE_ORDERS,
    C2_STAGE,
    CANDIDATES,
    DEVELOPMENT_COUNT,
    DPD_SHAREABLE_THRESHOLD_DB,
    LAMBDA_GRID,
    MAX_DELAY,
    MEMORY_PROFILES,
    NUMERIC_GAIN_TOLERANCE_DB,
    ORDER_PROFILES,
    SPLIT_SEED,
    STATE_COUNT,
    VALID_LENGTHS,
    aggregate_development_scan,
    baseline_regression_report,
    build_model_candidate_grid,
    build_selected_retrieval,
    build_statewise_comparison,
    candidate_from_payload,
    evaluate_single_candidate,
    fit_selected_model_all_states,
    full_retrieval_summary,
    load_baseline_state_metrics,
    load_fixed_split,
    load_pa_nonlinearity_values,
    load_real_b_reference,
    retrieval_validation_summary,
    scan_development,
    select_best_candidate,
)

TASK_NAME = "scenario_2_unified_model_capacity_scan"
REFERENCE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_ranking_consistency"
    / "scenario_2"
)
ALL_ILC_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_all_ilc_analysis"
    / "scenario_2"
)
BASELINE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_retrieval"
    / "scenario_2_C2_to_Aend"
)
REAL_B_ROOT = BASELINE_ROOT / "real_B_distance_matrix.npz"
SPLIT_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_y_lut_weighted_fusion_analysis"
    / "scenario_2"
    / "split_definition.csv"
)
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /TASK_NAME
    / "scenario_2_C2_to_Aend_unified_model_capacity"
)
LOG_ROOT = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME
EXECUTION_LOG = LOG_ROOT / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"


PROTECTED_RESULT_DIRS = {
    "scenario_2": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    /"scenario_2" / "scenario_2",
    "scenario_2_all_ilc": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario_2_all_ilc",
    "scenario_2_y_lut_fusion": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario_2_y_lut_fusion",
    "scenario_2_y_lut_fusion_A123": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario_2_y_lut_fusion_A123",
    "scenario_2_y_lut_weighted_fusion": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario_2_y_lut_weighted_fusion",
    "scenario_2_C3_to_A2": PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /TASK_NAME
    / "scenario_2_C3_to_A2",
    "scenario_2_C2_to_Aend": BASELINE_ROOT,
    "scenario_2_ridge_analysis": PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /"scenario_2_ridge_analysis",
    "scenario_2_xy_analysis": PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /"scenario_2_xy_analysis",
}
PROTECTED_SCRIPT_DIRS = {
    "behavior_fingerprint_ranking_consistency": PROJECT_ROOT
    / "scripts"
    / "behavior_fingerprint_ranking_consistency",
    "behavior_modeling": PROJECT_ROOT / "scripts" / "behavior_modeling",
    "signal_segmentation": PROJECT_ROOT / "scripts" / "signal_segmentation",
}
EXPECTED_PROTECTED_HASHES = {
    "data/raw": "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0",
    "scenario_2": "7a7e30b8fe5901c254d4b4815c07fabcfe612e9aafa2f7df45a38ca3e4fe9abe",
    "scenario_2_all_ilc": "513fc7a5f0982ca29b7a00ae90accfc2b041396821e46d3fdfa312ae25b667a1",
    "scenario_2_y_lut_fusion": "146369951909106e315fcc0272079b89603bb4f67d61c1442e750a17333d412b",
    "scenario_2_y_lut_fusion_A123": (
        "68f405fe4ea6e27bdb3524d9556e76922635370570b98d949889c3978feb742a"
    ),
    "scenario_2_y_lut_weighted_fusion": (
        "4203fccd4751f486ebcc9091f3c89b0928c01e28f2f1763b4961b5491e902c93"
    ),
    "scenario_2_C3_to_A2": "a6c9e289c5771003f3d9681ecae9b1278a50b4b81c5f2b4acb11d30a296b41f1",
    "scenario_2_ridge_analysis": "67d89dc336e439022093b126ab94c4984a2d0241db0033945d4b0d6fa5141188",
    "scenario_2_xy_analysis": "715e1436707e13d23c0e6634dec5ab58b348f8d6485dcd81b60c26f83f2cd48e",
    "behavior_fingerprint_ranking_consistency": (
        "9fad9fcd47e398928bf0937ead01467076a060e4a40f61ab7859e3776d95c510"
    ),
    "behavior_modeling": "f503577b541220588069434d364b1171a51487a01e9a8231e86fd37d6cbd5127",
    "signal_segmentation": "bf804cb139a68fdb336865fff514831b14b0f56b178f5840e5dedef92e1311ef",
}


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    files = (
        file
        for file in path.rglob("*")
        if file.is_file()
        and "__pycache__" not in file.parts
        and file.suffix.lower() != ".pyc"
        and not file.name.startswith("~$")
    )
    for file in sorted(files, key=lambda item: str(item).lower()):
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest_digest() -> tuple[str, int, int]:
    raw_root = PROJECT_ROOT / "data" / "raw"
    digest = hashlib.sha256()
    files = sorted(
        (file for file in raw_root.rglob("*") if file.is_file()), key=lambda item: str(item).lower()
    )
    for file in files:
        digest.update(str(file.relative_to(raw_root)).replace("\\", "/").encode() + b"\0")
        digest.update(file.stat().st_size.to_bytes(8, "little"))
    return digest.hexdigest(), len(files), sum(file.stat().st_size for file in files)


def _file_snapshot(path: Path) -> dict[str, str]:
    path = Path(path)
    return {
        file.relative_to(path).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.rglob("*"), key=lambda item: str(item).lower())
        if file.is_file()
        and "__pycache__" not in file.parts
        and file.suffix.lower() != ".pyc"
        and not file.name.startswith("~$")
    }


def _protection_snapshot() -> dict[str, Any]:
    raw_hash, raw_count, raw_bytes = _raw_manifest_digest()
    return {
        "data/raw": {"sha256": raw_hash, "file_count": raw_count, "bytes": raw_bytes},
        "result_dirs": {
            name: {
                "path": str(path),
                "exists": path.is_dir(),
                "sha256": _tree_digest(path) if path.is_dir() else None,
            }
            for name, path in PROTECTED_RESULT_DIRS.items()
        },
        "script_dirs": {
            name: {
                "path": str(path),
                "exists": path.is_dir(),
                "sha256": _tree_digest(path) if path.is_dir() else None,
            }
            for name, path in PROTECTED_SCRIPT_DIRS.items()
        },
        "retrieval_script_files": _file_snapshot(PROJECT_ROOT / "scripts" / TASK_NAME),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法JSON序列化类型：{type(value)!r}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int, np.bool_, bool)):
        return value.item() if isinstance(value, np.generic) else value
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
            allow_nan=False,
        )
        handle.write("\n")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NaN", float_format="%.17g")


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _append_log(text: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with EXECUTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] Scenario 2 unified Y-Aend/Y-C2 model capacity\n")
        handle.write(text.rstrip() + "\n")


def _append_handoff(text: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{timestamp} | {TASK_NAME}持续维护：统一Y-Aend/Y-C2模型容量\n")
        handle.write(text.rstrip() + "\n")


def _verify_protection(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "data/raw": {},
        "result_dirs": {},
        "script_dirs": {},
        "retrieval_script_files": {},
    }
    raw_before = before["data/raw"]
    raw_after = after["data/raw"]
    result["data/raw"] = {
        "sha256_unchanged": raw_before["sha256"] == raw_after["sha256"],
        "file_count_unchanged": raw_before["file_count"] == raw_after["file_count"],
        "bytes_unchanged": raw_before["bytes"] == raw_after["bytes"],
    }
    for group in ("result_dirs", "script_dirs"):
        for name, item in before[group].items():
            after_item = after[group][name]
            result[group][name] = {
                "sha256_unchanged": item["sha256"] == after_item["sha256"],
                "before": item["sha256"],
                "after": after_item["sha256"],
            }
    before_files = before["retrieval_script_files"]
    after_files = after["retrieval_script_files"]
    for name, digest in before_files.items():
        result["retrieval_script_files"][name] = {
            "sha256_unchanged": after_files.get(name) == digest,
            "before": digest,
            "after": after_files.get(name),
        }
    all_unchanged = all(result["data/raw"].values()) and all(
        item["sha256_unchanged"]
        for group in ("result_dirs", "script_dirs", "retrieval_script_files")
        for item in result[group].values()
    )
    result["all_protected_unchanged"] = bool(all_unchanged)
    return result


def _baseline_retrieval_frame() -> pd.DataFrame:
    results = pd.read_csv(BASELINE_ROOT / "retrieval_results.csv")
    diagnostics = pd.read_csv(BASELINE_ROOT / "retrieved_real_B_cnmse.csv")
    merged = results.loc[:, ["State_n_R", "State_n_Q", "exact_hit"]].merge(
        diagnostics.loc[:, ["State_n_R", "retrieved_real_B_CNMSE_dB", "dpd_shareable", "failure"]],
        on="State_n_R",
        how="left",
        validate="one_to_one",
    )
    if merged.shape[0] != STATE_COUNT:
        raise RuntimeError("现有C2→Aend Baseline检索结果必须有425行")
    return merged


def _summary_comparison_rows(
    dev_summary: pd.DataFrame,
    selected_payload: dict[str, Any],
    validation_summary: dict[str, Any],
    full_summary: dict[str, Any],
    transition: dict[str, Any],
) -> pd.DataFrame:
    baseline = dev_summary.loc[dev_summary["is_baseline"].astype(bool)].iloc[0]
    selected = dev_summary.loc[
        dev_summary["candidate_id"] == int(selected_payload["candidate_id"])
    ].iloc[0]
    rows = [
        {
            "group": "development_model",
            "metric": "Aend_B_median_dB",
            "baseline": baseline["Aend_B_median_dB"],
            "selected": selected["Aend_B_median_dB"],
        },
        {
            "group": "development_model",
            "metric": "C2_B_median_dB",
            "baseline": baseline["C2_B_median_dB"],
            "selected": selected["C2_B_median_dB"],
        },
        {
            "group": "development_model",
            "metric": "worst_side_B_median_dB",
            "baseline": baseline["worst_side_B_median_dB"],
            "selected": selected["worst_side_B_median_dB"],
        },
        {
            "group": "development_model",
            "metric": "worst_side_B_q95_dB",
            "baseline": baseline["worst_side_B_q95_dB"],
            "selected": selected["worst_side_B_q95_dB"],
        },
        {
            "group": "validation_retrieval",
            "metric": "shareable_count",
            "baseline": validation_summary["baseline_shareable_count"],
            "selected": validation_summary["selected_shareable_count"],
        },
        {
            "group": "validation_retrieval",
            "metric": "shareable_rate",
            "baseline": validation_summary["baseline_shareable_rate"],
            "selected": validation_summary["selected_shareable_rate"],
        },
        {
            "group": "full425_retrieval",
            "metric": "shareable_count",
            "baseline": full_summary["baseline"]["shareable_count"],
            "selected": full_summary["selected"]["shareable_count"],
        },
        {
            "group": "full425_retrieval",
            "metric": "shareable_rate",
            "baseline": full_summary["baseline"]["shareable_rate"],
            "selected": full_summary["selected"]["shareable_rate"],
        },
        {
            "group": "full425_retrieval",
            "metric": "failure_count",
            "baseline": full_summary["baseline"]["failure_count"],
            "selected": full_summary["selected"]["failure_count"],
        },
        {
            "group": "transition",
            "metric": "failure_to_success",
            "baseline": np.nan,
            "selected": transition["comparison"]["transition_counts"].get("failure_to_success", 0),
        },
        {
            "group": "transition",
            "metric": "failure_to_failure",
            "baseline": np.nan,
            "selected": transition["comparison"]["transition_counts"].get("failure_to_failure", 0),
        },
        {
            "group": "transition",
            "metric": "success_to_failure",
            "baseline": np.nan,
            "selected": transition["comparison"]["transition_counts"].get("success_to_failure", 0),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["delta"] = frame["selected"] - frame["baseline"]
    return frame


def main() -> None:
    print("Scenario 2 unified Y-Aend/Y-C2 model-capacity study started.", flush=True)
    print(f"Project root: {PROJECT_ROOT}", flush=True)
    before = _protection_snapshot()
    observed = {"data/raw": before["data/raw"]["sha256"]}
    observed.update(
        {
            name: item["sha256"]
            for name, item in before["result_dirs"].items()
            if name != "scenario_2_C2_to_Aend"
        }
    )
    observed.update({name: item["sha256"] for name, item in before["script_dirs"].items()})
    mismatch = [
        f"{name}: observed={observed.get(name)} expected={expected}"
        for name, expected in EXPECTED_PROTECTED_HASHES.items()
        if observed.get(name) != expected
    ]
    if mismatch:
        raise RuntimeError("保护输入SHA与当前基线不一致：" + "; ".join(mismatch))

    split_frame = load_fixed_split(SPLIT_SOURCE)
    candidate_grid = build_model_candidate_grid()
    baseline_candidate = next(candidate for candidate in CANDIDATES if candidate.is_baseline)
    frozen = load_frozen_scenario2_artifacts(REFERENCE_ROOT)
    state_frame = pd.DataFrame(build_state_table()).sort_values("state_id").reset_index(drop=True)
    baseline_retrieval = _baseline_retrieval_frame()
    baseline_shareable = int(baseline_retrieval["dpd_shareable"].sum())
    baseline_exact = int(baseline_retrieval["exact_hit"].sum())
    baseline_failure = int(baseline_retrieval["failure"].sum())
    if (baseline_exact, baseline_shareable, baseline_failure) != (218, 411, 14):
        raise RuntimeError(
            "现有C2→Aend Baseline回归失败："
            f"exact/shareable/failure={baseline_exact}/{baseline_shareable}/{baseline_failure}"
        )

    real_b_distance, real_b_ranking = load_real_b_reference(
        REAL_B_ROOT,
        REFERENCE_ROOT / "distance_matrices.npz",
        REFERENCE_ROOT / "ranking_matrices.npz",
    )
    baseline_metrics = load_baseline_state_metrics(
        ALL_ILC_ROOT / "all_ilc_model_metrics.csv",
        ALL_ILC_ROOT / "ilc_availability.csv",
    )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_frame(split_frame, RESULT_ROOT / "split_definition.csv")
    _write_frame(candidate_grid, RESULT_ROOT / "model_candidate_grid.csv")
    _append_log(
        "\n".join(
            [
                "研究目标：在完全统一的Y-Aend/Y-C2 MP结构、memory和Ridge lambda下，"
                "以Development泛化指标选择模型，并检验是否转化为C2→Aend检索提升。",
                f"candidate grid={candidate_grid.shape[0]}；"
                f"order_profiles={sorted(candidate_grid['order_profile'].unique())}；"
                f"memory_profiles={sorted(candidate_grid['memory_profile'].unique())}；"
                f"lambda_count={candidate_grid['lambda'].nunique()}。",
                "Y-Aend/Y-C2共享candidate结构和lambda，系数独立拟合；所有candidate max_delay=2，"
                "有效长度A/B/C=12286/4913/7371。",
                "Baseline retrieval regression "
                f"exact/shareable/failure={baseline_exact}/{baseline_shareable}/{baseline_failure}；"
                "Real-B仅作为冻结参考，未进入candidate score。",
                f"固定split source={SPLIT_SOURCE}；Development={DEVELOPMENT_COUNT}，"
                f"Validation={split_frame['split'].eq('Validation').sum()}，seed={SPLIT_SEED}。",
            ]
        )
    )

    dev_ids = split_frame.loc[split_frame["split"] == "Development", "state_id"].to_numpy(
        dtype=np.int64
    )
    print("Step 1/8: strict Development Baseline regression", flush=True)
    baseline_observed = evaluate_single_candidate(dev_ids, baseline_candidate, progress_interval=25)
    baseline_regression = baseline_regression_report(
        baseline_observed, baseline_metrics.loc[baseline_metrics["state_id"].isin(dev_ids)]
    )
    _write_json(RESULT_ROOT / "baseline_regression.json", baseline_regression)
    _append_log(f"VALIDATION_LOCKED；Baseline P9/M0/lambda=1e-8 regression={baseline_regression}。")
    if not baseline_regression["pass"]:
        _append_log(
            "Baseline regression failed; 120-candidate scan stopped before model selection."
        )
        raise RuntimeError("Baseline回归未通过，按冻结流程停止120 candidate扫描")

    print("Step 2/8: Development-only 120-candidate scan", flush=True)
    development_state_metrics = scan_development(dev_ids, progress_interval=10)
    development_summary = aggregate_development_scan(development_state_metrics, candidate_grid)
    _write_frame(development_state_metrics, RESULT_ROOT / "development_state_metrics.csv")
    _write_frame(
        development_summary.sort_values("candidate_id"), RESULT_ROOT / "development_model_scan.csv"
    )
    _write_frame(development_summary, RESULT_ROOT / "development_model_ranking.csv")
    selected_payload = select_best_candidate(development_summary)
    selected_candidate = candidate_from_payload(selected_payload)
    selected_payload["candidate_count"] = int(candidate_grid.shape[0])
    selected_payload["all_candidates_max_delay"] = bool(
        np.all(candidate_grid["max_delay"].to_numpy(dtype=int) == MAX_DELAY)
    )
    selected_payload["validation_locked_during_selection"] = True
    selected_payload["real_B_used_during_selection"] = False
    selected_payload["failure_labels_used_during_selection"] = False
    _write_json(RESULT_ROOT / "selected_model.json", selected_payload)
    selected_payload = json.loads((RESULT_ROOT / "selected_model.json").read_text(encoding="utf-8"))
    selected_candidate = candidate_from_payload(selected_payload)
    if selected_candidate.ridge_lambda in (1e-12, 1e-4):
        _append_log(
            "WARNING: selected lambda lies at scan boundary; "
            "result is flagged for a Development-only decade extension review."
        )
    _append_log(
        f"UNIFIED_MODEL_HYPERPARAMETERS_FROZEN；selected candidate={selected_payload}；"
        "Development scoring未读取Validation/Real-B/retrieval/failure labels。"
    )

    print("Step 3/8: fit frozen selected model on all 425 states", flush=True)
    _append_log(
        "VALIDATION_UNLOCKED_AFTER_MODEL_FREEZE；selected_model.json已冻结，开始冻结模型验证和Full-425拟合。"
    )
    selected_metrics, theta_a, theta_c, fp_a, fp_c = fit_selected_model_all_states(
        selected_candidate,
        range(STATE_COUNT),
        frozen.common_B_input,
        split_frame=split_frame,
        progress_interval=10,
    )
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    selected_metrics = selected_metrics.merge(
        state_frame, on="state_id", how="left", validate="one_to_one"
    )
    _write_frame(
        selected_metrics.loc[selected_metrics["split"] == "Development"],
        RESULT_ROOT / "development_selected_state_metrics.csv",
    )
    _write_frame(
        selected_metrics.loc[selected_metrics["split"] == "Validation"],
        RESULT_ROOT / "validation_selected_state_metrics.csv",
    )
    _write_frame(selected_metrics, RESULT_ROOT / "selected_model_state_metrics_all425.csv")
    memory_keys = np.asarray(sorted(selected_candidate.memory_definition), dtype=np.int64)
    memory_values = np.asarray(
        [selected_candidate.memory_definition[key] for key in memory_keys], dtype=np.int64
    )
    _write_npz(
        RESULT_ROOT / "selected_model_coefficients.npz",
        {
            "state_ids": state_ids,
            "theta_Aend": theta_a,
            "theta_C2": theta_c,
            "orders": np.asarray(selected_candidate.orders, dtype=np.int64),
            "memory_orders": memory_keys,
            "memory_taps": memory_values,
            "ridge_lambda": np.asarray([selected_candidate.ridge_lambda], dtype=np.float64),
            "Aend_stage": selected_metrics["ilc_A_end"].to_numpy(dtype=np.int64),
        },
    )
    _write_npz(
        RESULT_ROOT / "selected_lut_fingerprints_Aend.npz",
        {
            "state_ids": state_ids,
            "common_B_input": frozen.common_B_input,
            "Y_Aend_fingerprints": fp_a,
            "Aend_stage": selected_metrics["ilc_A_end"].to_numpy(dtype=np.int64),
        },
    )
    _write_npz(
        RESULT_ROOT / "selected_query_fingerprints_C2.npz",
        {
            "state_ids": state_ids,
            "common_B_input": frozen.common_B_input,
            "Q_C2": fp_c,
            "effective_C_stage": np.full(STATE_COUNT, C2_STAGE, dtype=np.int64),
        },
    )

    validation_metrics = selected_metrics.loc[selected_metrics["split"] == "Validation"]
    selected_validation_model_summary = {
        "validation_count": int(validation_metrics.shape[0]),
        "Aend_train_median_dB": float(validation_metrics["Y_Aend_train_NMSE_dB"].median()),
        "Aend_B_median_dB": float(validation_metrics["Y_Aend_B_NMSE_dB"].median()),
        "C2_train_median_dB": float(validation_metrics["Y_C2_train_NMSE_dB"].median()),
        "C2_B_median_dB": float(validation_metrics["Y_C2_B_NMSE_dB"].median()),
    }
    _write_json(
        RESULT_ROOT / "validation_selected_model_summary.json", selected_validation_model_summary
    )

    print("Step 4/8: selected C2 -> Aend retrieval with frozen Real-B matrix", flush=True)
    retrieval_state_frame = state_frame.copy()
    retrieval_state_frame["ilc_A_end"] = (
        selected_metrics.set_index("state_id")
        .loc[retrieval_state_frame["state_id"], "ilc_A_end"]
        .to_numpy(dtype=np.int64)
    )
    retrieval = build_selected_retrieval(
        fp_c,
        fp_a,
        real_b_distance,
        retrieval_state_frame,
    )
    _write_npz(
        RESULT_ROOT / "selected_retrieval_distance_matrix.npz",
        {"state_ids": state_ids, "D_C2_Aend": retrieval["distance"]},
    )
    _write_npz(
        RESULT_ROOT / "selected_retrieval_ranking_matrix.npz",
        {"state_ids": state_ids, "R_C2_Aend": retrieval["ranking"]},
    )
    _write_frame(retrieval["results"], RESULT_ROOT / "selected_retrieval_results.csv")
    _write_frame(retrieval["diagnostics"], RESULT_ROOT / "selected_retrieved_real_B_cnmse.csv")
    _write_frame(
        retrieval["results"]
        .loc[retrieval["results"]["failure"]]
        .sort_values(["retrieved_real_B_CNMSE_dB", "State_n_R"], ascending=[False, True]),
        RESULT_ROOT / "selected_failed_retrieval_states.csv",
    )
    _write_npz(
        RESULT_ROOT / "real_B_distance_matrix_reused.npz",
        {"state_ids": state_ids, "D_B": real_b_distance, "R_B": real_b_ranking},
    )

    print("Step 5/8: statewise gains, quartiles and transitions", flush=True)
    nmse_withoutdpd = load_pa_nonlinearity_values(range(STATE_COUNT), progress_interval=50)
    baseline_metrics = baseline_metrics.merge(
        state_frame, on="state_id", how="left", validate="one_to_one", suffixes=("", "_state")
    )
    statewise, quartile_summary, baseline_failure_transition, transition = (
        build_statewise_comparison(
            baseline_metrics,
            selected_metrics,
            baseline_retrieval,
            retrieval["results"],
            nmse_withoutdpd,
            split_frame,
        )
    )
    _write_frame(statewise, RESULT_ROOT / "statewise_baseline_vs_selected.csv")
    _write_frame(baseline_failure_transition, RESULT_ROOT / "baseline_failure_transition.csv")
    _write_frame(
        statewise.loc[statewise["retrieval_gain_finite_finite"]],
        RESULT_ROOT / "model_gain_vs_retrieval_gain.csv",
    )
    _write_frame(quartile_summary, RESULT_ROOT / "pa_nonlinearity_quartile_gain_summary.csv")
    _write_frame(transition["transition_summary"], RESULT_ROOT / "transition_summary.csv")

    validation_retrieval = retrieval_validation_summary(
        baseline_retrieval,
        retrieval["results"],
        split_frame.loc[split_frame["split"] == "Validation", "state_id"],
    )
    full_summary = full_retrieval_summary(baseline_retrieval, retrieval["results"])
    comparison_rows = _summary_comparison_rows(
        development_summary,
        selected_payload,
        validation_retrieval,
        full_summary,
        transition,
    )
    _write_frame(comparison_rows, RESULT_ROOT / "comparison_vs_baseline_summary.csv")

    print(
        "Step 6/8: generate six evidence figures (Figure 6 has three independent panels)",
        flush=True,
    )
    figure_details = {
        "figure1": plot_candidate_scan(
            development_summary, selected_payload, RESULT_ROOT / "figure1_model_candidate_scan.png"
        ),
        "figure2": plot_pa_quartile_gains(
            quartile_summary, RESULT_ROOT / "figure2_pa_nonlinearity_quartile_gain.png"
        ),
        "figure3": plot_state_model_metrics(
            statewise, RESULT_ROOT / "figure3_baseline_vs_selected_model_metrics.png"
        ),
        "figure4": plot_retrieval_comparison(
            statewise, RESULT_ROOT / "figure4_baseline_vs_selected_retrieval.png"
        ),
        "figure5": plot_transition_classes(
            statewise, RESULT_ROOT / "figure5_failure_success_transitions.png"
        ),
        "figure6": plot_gain_correlations(statewise, RESULT_ROOT),
    }

    print("Step 7/8: write validation and protection results", flush=True)
    after = _protection_snapshot()
    protection = _verify_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw或旧正式结果/脚本发生变化，拒绝完成统一模型容量结果")
    baseline_dev_selected = development_summary.loc[
        development_summary["is_baseline"].astype(bool)
    ].iloc[0]
    selected_dev = development_summary.loc[
        development_summary["candidate_id"] == int(selected_payload["candidate_id"])
    ].iloc[0]
    validation_payload: dict[str, Any] = {
        "study": "unified Y-Aend/Y-C2 model capacity versus LUT retrieval performance",
        "scenario": 2,
        "Aend_and_C2_share_same_model": True,
        "baseline_orders": list(BASELINE_ORDERS),
        "baseline_memory": BASELINE_MEMORY,
        "baseline_lambda": BASELINE_LAMBDA,
        "order_profiles": {name: list(values) for name, values in ORDER_PROFILES.items()},
        "memory_profiles": MEMORY_PROFILES,
        "lambda_grid": list(LAMBDA_GRID),
        "candidate_count": int(candidate_grid.shape[0]),
        "baseline_candidate_count": int(candidate_grid["is_baseline"].sum()),
        "all_candidates_max_delay": int(candidate_grid["max_delay"].max()),
        "valid_lengths": VALID_LENGTHS,
        "development_count": DEVELOPMENT_COUNT,
        "validation_count": int(split_frame["split"].eq("Validation").sum()),
        "split_seed": SPLIT_SEED,
        "model_selection_uses_retrieval": False,
        "model_selection_uses_real_B": False,
        "model_selection_uses_failure_labels": False,
        "selection_rule": selected_payload["selection_rule"],
        "selected_model": selected_payload,
        "selected_development_metrics": {
            "Aend_B_median_dB": float(selected_dev["Aend_B_median_dB"]),
            "C2_B_median_dB": float(selected_dev["C2_B_median_dB"]),
            "worst_side_B_median_dB": float(selected_dev["worst_side_B_median_dB"]),
            "worst_side_B_q95_dB": float(selected_dev["worst_side_B_q95_dB"]),
        },
        "baseline_development_metrics": {
            "Aend_B_median_dB": float(baseline_dev_selected["Aend_B_median_dB"]),
            "C2_B_median_dB": float(baseline_dev_selected["C2_B_median_dB"]),
            "worst_side_B_median_dB": float(baseline_dev_selected["worst_side_B_median_dB"]),
            "worst_side_B_q95_dB": float(baseline_dev_selected["worst_side_B_q95_dB"]),
        },
        "baseline_regression": baseline_regression,
        "model_frozen_before_retrieval": True,
        "validation_selected_model_summary": selected_validation_model_summary,
        "query": "Y-C2",
        "lut": "Y-Aend",
        "selected_fingerprint_shapes": {"Aend": list(fp_a.shape), "C2": list(fp_c.shape)},
        "selected_fingerprint_dtype": {"Aend": str(fp_a.dtype), "C2": str(fp_c.dtype)},
        "selected_fingerprints_all_finite": bool(
            np.all(np.isfinite(fp_a)) and np.all(np.isfinite(fp_c))
        ),
        "selected_retrieval_distance_shape": list(retrieval["distance"].shape),
        "selected_retrieval_ranking_shape": list(retrieval["ranking"].shape),
        "candidate_count_per_query": STATE_COUNT,
        "self_match": True,
        "DPD_shareable_threshold_dB": DPD_SHAREABLE_THRESHOLD_DB,
        "threshold_rule": "Real-B CNMSE < -40 dB",
        "threshold_is_strict": True,
        "numeric_gain_tolerance_dB": NUMERIC_GAIN_TOLERANCE_DB,
        "baseline_full425": full_summary["baseline"],
        "selected_full425": full_summary["selected"],
        "validation_retrieval": validation_retrieval,
        "validation_retrieval_improvement": validation_retrieval[
            "validation_retrieval_improvement"
        ],
        "transition_analysis": transition["comparison"],
        "correlations": transition["comparison"]["correlations"],
        "pa_quartile_summary": quartile_summary.to_dict("records"),
        "raw_data_modified": False,
        "real_B_matrix_reused": True,
        "real_B_matrix_source": str(REAL_B_ROOT),
        "protection_before": before,
        "protection_after": after,
        "protection_verification": protection,
        "figures": figure_details,
        "output_files": {
            path.name: str(path) for path in sorted(RESULT_ROOT.iterdir()) if path.is_file()
        },
    }
    _write_json(RESULT_ROOT / "validation.json", validation_payload)

    print("Step 8/8: append execution logs", flush=True)
    _append_log(
        "\n".join(
            [
                "120 candidate Development scan completed；"
                f"selected={selected_payload['candidate_id']} "
                f"{selected_payload['order_profile']}/{selected_payload['memory_profile']}/"
                f"lambda={selected_payload['lambda']}。",
                "Baseline regression "
                f"max_abs_error_dB={baseline_regression['max_abs_error_dB']} "
                f"pass={baseline_regression['pass']}。",
                "Validation retrieval baseline/selected shareable="
                f"{validation_retrieval['baseline_shareable_count']}/"
                f"{validation_retrieval['selected_shareable_count']} "
                f"({validation_retrieval['baseline_shareable_rate']:.9g}/"
                f"{validation_retrieval['selected_shareable_rate']:.9g}); "
                f"strict improvement={validation_retrieval['validation_retrieval_improvement']}。",
                "Full425 baseline/selected shareable="
                f"{full_summary['baseline']['shareable_count']}/"
                f"{full_summary['selected']['shareable_count']}; "
                f"failure={full_summary['baseline']['failure_count']}/{full_summary['selected']['failure_count']}。",
                f"Transitions={transition['comparison']['transition_counts']}；"
                f"old failure→success={transition['comparison']['old_failure_to_success_count']}；"
                f"new success→failure={transition['comparison']['new_success_to_failure_count']}。",
                f"Finite/finite gain correlations={transition['comparison']['correlations']}。",
                "All selected fingerprints finite "
                f"shape={fp_a.shape}/{fp_c.shape} dtype={fp_a.dtype}/{fp_c.dtype}；"
                "retrieval distance/ranking="
                f"{retrieval['distance'].shape}/{retrieval['ranking'].shape}。",
                "Protection "
                f"all_protected_unchanged={protection['all_protected_unchanged']}；"
                "raw_data_modified=False；"
                f"result_root={RESULT_ROOT}。",
            ]
        )
    )
    _append_handoff(
        "完成Scenario 2统一Y-Aend/Y-C2模型容量研究："
        "120 candidates在Development 340状态上按双侧B泛化指标选择，"
        f"selected={selected_payload['order_profile']}/{selected_payload['memory_profile']}/lambda={selected_payload['lambda']}，"
        "Validation retrieval shareable baseline/selected="
        f"{validation_retrieval['baseline_shareable_count']}/"
        f"{validation_retrieval['selected_shareable_count']}，严格提升={validation_retrieval['validation_retrieval_improvement']}；"
        f"Full425 selected shareable={full_summary['selected']['shareable_count']}/{STATE_COUNT}，"
        f"failure={full_summary['selected']['failure_count']}；raw及旧正式结果保护通过，结果写入{RESULT_ROOT}。"
    )
    print("Scenario 2 unified Y-Aend/Y-C2 model-capacity study completed.", flush=True)
    print(f"Results: {RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
