"""Run and persist the Scenario 2 equal-length ABC ablation."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_ranking_consistency.ranking import distance_to_ranks  # noqa: E402
from behavior_model.basis import build_mp_basis  # noqa: E402
from data_manager import build_state_table  # noqa: E402

from behavior_fingerprint_lut_retrieval.plot_scenario2_c2_to_aend_equal_abc import (  # noqa: E402
    plot_equal_abc_state_metrics,
    plot_nonexact_failure_detail,
)
from behavior_fingerprint_lut_retrieval.scenario2_c2_to_aend_equal_abc import (  # noqa: E402
    DPD_SHAREABLE_THRESHOLD_DB,
    FINGERPRINT_LENGTH,
    MEMORY,
    N_COMPLEX_COEFFICIENTS,
    ORDERS,
    RIDGE_LAMBDA,
    SEGMENT_LENGTH,
    STATE_COUNT,
    VALID_LENGTH,
    VALID_LENGTHS,
    build_correlation_summary,
    build_equal_abc_partition,
    build_equal_retrieval_results,
    build_generalization_gap_summary,
    build_quartile_summary,
    build_retrieval_class_summary,
    compute_equal_cnmse_distance_matrix,
    fit_equal_abc_state,
    load_raw_scalar_metrics,
    prepare_equal_abc_state,
    segment_definition,
)
from behavior_fingerprint_lut_retrieval.scenario2_unified_model_capacity_scan import (  # noqa: E402
    load_baseline_state_metrics,
)

TASK_NAME = "behavior_fingerprint_lut_retrieval"
ALL_ILC_ROOT = (
    PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2_all_ilc"
)
BASELINE_ROOT = PROJECT_ROOT / "results" / TASK_NAME / "scenario_2_C2_to_Aend"
CAPACITY_ROOT = (
    PROJECT_ROOT / "results" / TASK_NAME / "scenario_2_C2_to_Aend_unified_model_capacity"
)
RESULT_ROOT = PROJECT_ROOT / "results" / TASK_NAME / "scenario_2_C2_to_Aend_equal_ABC"
LOG_ROOT = PROJECT_ROOT / "work_logs" / TASK_NAME
EXECUTION_LOG = LOG_ROOT / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"


PROTECTED_RESULT_DIRS = {
    "scenario_2": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2",
    "scenario_2_all_ilc": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_all_ilc",
    "scenario_2_y_lut_fusion": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_y_lut_fusion",
    "scenario_2_y_lut_fusion_A123": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_y_lut_fusion_A123",
    "scenario_2_y_lut_weighted_fusion": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_y_lut_weighted_fusion",
    "scenario_2_C3_to_A2": PROJECT_ROOT / "results" / TASK_NAME / "scenario_2_C3_to_A2",
    "scenario_2_C2_to_Aend": BASELINE_ROOT,
    "scenario_2_C2_to_Aend_unified_model_capacity": CAPACITY_ROOT,
    "scenario_2_ridge_analysis": PROJECT_ROOT
    / "results"
    / "behavior_model"
    / "scenario_2_ridge_analysis",
    "scenario_2_xy_analysis": PROJECT_ROOT
    / "results"
    / "behavior_model"
    / "scenario_2_xy_analysis",
}
PROTECTED_SCRIPT_DIRS = {
    "behavior_fingerprint_ranking_consistency": PROJECT_ROOT
    / "scripts"
    / "behavior_fingerprint_ranking_consistency",
    "behavior_model": PROJECT_ROOT / "scripts" / "behavior_model",
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
    "behavior_model": "f503577b541220588069434d364b1171a51487a01e9a8231e86fd37d6cbd5127",
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


def _raw_manifest() -> dict[str, Any]:
    raw_root = Path(r"\\?\{}".format(PROJECT_ROOT / "data" / "raw"))
    digest = hashlib.sha256()
    files = sorted(
        (file for file in raw_root.rglob("*") if file.is_file()), key=lambda item: str(item).lower()
    )
    for file in files:
        digest.update(str(file.relative_to(raw_root)).replace("\\", "/").encode() + b"\0")
        digest.update(file.stat().st_size.to_bytes(8, "little"))
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "bytes": sum(file.stat().st_size for file in files),
    }


def _file_snapshot(path: Path) -> dict[str, str]:
    return {
        file.relative_to(path).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.rglob("*"), key=lambda item: str(item).lower())
        if file.is_file()
        and "__pycache__" not in file.parts
        and file.suffix.lower() != ".pyc"
        and not file.name.startswith("~$")
    }


def _protection_snapshot() -> dict[str, Any]:
    raw = _raw_manifest()
    return {
        "data/raw": raw,
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


def _verify_protection(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "data/raw": {},
        "result_dirs": {},
        "script_dirs": {},
        "retrieval_script_files": {},
    }
    result["data/raw"] = {
        "sha256_unchanged": before["data/raw"]["sha256"] == after["data/raw"]["sha256"],
        "file_count_unchanged": before["data/raw"]["file_count"] == after["data/raw"]["file_count"],
        "bytes_unchanged": before["data/raw"]["bytes"] == after["data/raw"]["bytes"],
    }
    for group in ("result_dirs", "script_dirs"):
        for name, item in before[group].items():
            result[group][name] = {
                "sha256_unchanged": item["sha256"] == after[group][name]["sha256"],
                "before": item["sha256"],
                "after": after[group][name]["sha256"],
            }
    for name, digest in before["retrieval_script_files"].items():
        result["retrieval_script_files"][name] = {
            "sha256_unchanged": after["retrieval_script_files"].get(name) == digest,
            "before": digest,
            "after": after["retrieval_script_files"].get(name),
        }
    result["all_protected_unchanged"] = bool(
        all(result["data/raw"].values())
        and all(
            item["sha256_unchanged"]
            for group in ("result_dirs", "script_dirs", "retrieval_script_files")
            for item in result[group].values()
        )
    )
    return result


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
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
        handle.write(f"\n\n[{timestamp}] Scenario 2 equal-length ABC ablation\n{text.rstrip()}\n")


def _append_handoff(text: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n{timestamp} | {TASK_NAME}持续维护：Scenario 2 equal ABC 消融\n{text.rstrip()}\n"
        )


def _load_old_retrieval() -> pd.DataFrame:
    results = pd.read_csv(BASELINE_ROOT / "retrieval_results.csv")
    diagnostics = pd.read_csv(BASELINE_ROOT / "retrieved_real_B_cnmse.csv")
    merged = results.loc[:, ["State_n_R", "State_n_Q", "exact_hit"]].merge(
        diagnostics.loc[:, ["State_n_R", "retrieved_real_B_CNMSE_dB", "dpd_shareable", "failure"]],
        on="State_n_R",
        how="left",
        validate="one_to_one",
    )
    if merged.shape[0] != STATE_COUNT:
        raise RuntimeError("旧正式 C2→Aend retrieval 必须包含425个状态")
    return merged


def _physical_mismatch(code: Any) -> str:
    value = int(round(float(code)))
    if value == 0:
        return "0"
    return f"{value / 100:.2f}".rstrip("0").rstrip(".")


def _old_new_comparison(
    old_metrics: pd.DataFrame,
    new_metrics: pd.DataFrame,
    old_retrieval: pd.DataFrame,
    new_retrieval: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    old = old_metrics.rename(
        columns={
            "Y_Aend_train_NMSE_dB": "old_Y_Aend_train_NMSE_dB",
            "Y_Aend_B_NMSE_dB": "old_Y_Aend_B_NMSE_dB",
            "Y_C2_train_NMSE_dB": "old_Y_C2_train_NMSE_dB",
            "Y_C2_B_NMSE_dB": "old_Y_C2_B_NMSE_dB",
        }
    )
    new = new_metrics.loc[
        :,
        [
            "state_id",
            "Y_Aend_train_NMSE_dB",
            "Y_Aend_B_NMSE_dB",
            "Y_C2_train_NMSE_dB",
            "Y_C2_B_NMSE_dB",
        ],
    ].rename(
        columns={
            "Y_Aend_train_NMSE_dB": "new_Y_Aend_train_NMSE_dB",
            "Y_Aend_B_NMSE_dB": "new_Y_Aend_B_NMSE_dB",
            "Y_C2_train_NMSE_dB": "new_Y_C2_train_NMSE_dB",
            "Y_C2_B_NMSE_dB": "new_Y_C2_B_NMSE_dB",
        }
    )
    old_ret = old_retrieval.rename(
        columns={
            "State_n_R": "state_id",
            "State_n_Q": "old_state_Q",
            "exact_hit": "old_exact",
            "retrieved_real_B_CNMSE_dB": "old_real_B_CNMSE",
            "dpd_shareable": "old_shareable",
            "failure": "old_failure",
        }
    )
    new_ret = new_retrieval.rename(
        columns={
            "State_n_R": "state_id",
            "State_n_Q": "new_state_Q",
            "exact_hit": "new_exact",
            "retrieved_real_B_CNMSE_dB": "new_real_B_CNMSE",
            "dpd_shareable": "new_shareable",
            "failure": "new_failure",
        }
    )
    statewise = (
        old.merge(new, on="state_id", how="inner", validate="one_to_one")
        .merge(
            old_ret.loc[
                :,
                [
                    "state_id",
                    "old_state_Q",
                    "old_exact",
                    "old_real_B_CNMSE",
                    "old_shareable",
                    "old_failure",
                ],
            ],
            on="state_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            new_ret.loc[
                :,
                [
                    "state_id",
                    "new_state_Q",
                    "new_exact",
                    "new_real_B_CNMSE",
                    "new_shareable",
                    "new_failure",
                ],
            ],
            on="state_id",
            how="inner",
            validate="one_to_one",
        )
    )
    statewise["state_Q_changed"] = statewise["old_state_Q"] != statewise["new_state_Q"]
    statewise["old_new_real_B_window_note"] = (
        "old B=4915 ownership samples versus equal ABC B=8192 ownership samples; "
        "this is an end-to-end segmentation comparison, not a fixed-window gain"
    )
    statewise["transition_class"] = np.select(
        [
            statewise["old_failure"] & ~statewise["new_failure"],
            statewise["old_failure"] & statewise["new_failure"],
            ~statewise["old_failure"] & ~statewise["new_failure"],
            ~statewise["old_failure"] & statewise["new_failure"],
        ],
        ["failure_to_success", "failure_to_failure", "success_to_success", "success_to_failure"],
        default="unknown",
    )
    statewise["exact_transition_class"] = np.select(
        [
            statewise["old_exact"] & statewise["new_exact"],
            ~statewise["old_exact"] & statewise["new_exact"],
            statewise["old_exact"] & ~statewise["new_exact"],
        ],
        ["exact_to_exact", "nonexact_to_exact", "exact_to_nonexact"],
        default="nonexact_to_nonexact",
    )
    rows: list[dict[str, Any]] = []
    model_pairs = (
        ("Y_Aend_train_NMSE_dB", "old_Y_Aend_train_NMSE_dB", "new_Y_Aend_train_NMSE_dB"),
        ("Y_Aend_B_NMSE_dB", "old_Y_Aend_B_NMSE_dB", "new_Y_Aend_B_NMSE_dB"),
        ("Y_C2_train_NMSE_dB", "old_Y_C2_train_NMSE_dB", "new_Y_C2_train_NMSE_dB"),
        ("Y_C2_B_NMSE_dB", "old_Y_C2_B_NMSE_dB", "new_Y_C2_B_NMSE_dB"),
    )
    for metric, old_column, new_column in model_pairs:
        old_values = statewise[old_column].to_numpy(dtype=float)
        new_values = statewise[new_column].to_numpy(dtype=float)
        rows.extend(
            [
                {
                    "metric": f"{metric}_median",
                    "old_ABC": float(np.median(old_values)),
                    "equal_ABC": float(np.median(new_values)),
                },
                {
                    "metric": f"{metric}_q95",
                    "old_ABC": float(np.quantile(old_values, 0.95)),
                    "equal_ABC": float(np.quantile(new_values, 0.95)),
                },
            ]
        )
    gap_pairs = (
        (
            "Gap_A_dB",
            "old_Y_Aend_B_NMSE_dB",
            "old_Y_Aend_train_NMSE_dB",
            "new_Y_Aend_B_NMSE_dB",
            "new_Y_Aend_train_NMSE_dB",
        ),
        (
            "Gap_C_dB",
            "old_Y_C2_B_NMSE_dB",
            "old_Y_C2_train_NMSE_dB",
            "new_Y_C2_B_NMSE_dB",
            "new_Y_C2_train_NMSE_dB",
        ),
    )
    for metric, old_b, old_train, new_b, new_train in gap_pairs:
        old_gap = statewise[old_b].to_numpy(dtype=float) - statewise[old_train].to_numpy(
            dtype=float
        )
        new_gap = statewise[new_b].to_numpy(dtype=float) - statewise[new_train].to_numpy(
            dtype=float
        )
        rows.extend(
            [
                {
                    "metric": f"{metric}_median",
                    "old_ABC": float(np.median(old_gap)),
                    "equal_ABC": float(np.median(new_gap)),
                },
                {
                    "metric": f"{metric}_q95",
                    "old_ABC": float(np.quantile(old_gap, 0.95)),
                    "equal_ABC": float(np.quantile(new_gap, 0.95)),
                },
            ]
        )
    state_q_changed = statewise["state_Q_changed"].sum()
    for metric, old_values, new_values in (
        ("Exact hit count", statewise["old_exact"].sum(), statewise["new_exact"].sum()),
        ("Exact hit rate", statewise["old_exact"].mean(), statewise["new_exact"].mean()),
        ("DPD-shareable count", statewise["old_shareable"].sum(), statewise["new_shareable"].sum()),
        (
            "DPD-shareable rate",
            statewise["old_shareable"].mean(),
            statewise["new_shareable"].mean(),
        ),
        ("Failure count", statewise["old_failure"].sum(), statewise["new_failure"].sum()),
        ("State_Q changed count", 0, state_q_changed),
    ):
        rows.append(
            {"metric": metric, "old_ABC": float(old_values), "equal_ABC": float(new_values)}
        )
    old_finite = statewise.loc[
        (~statewise["old_exact"]) & np.isfinite(statewise["old_real_B_CNMSE"]), "old_real_B_CNMSE"
    ]
    new_finite = statewise.loc[
        (~statewise["new_exact"]) & np.isfinite(statewise["new_real_B_CNMSE"]), "new_real_B_CNMSE"
    ]
    for metric, old_values, new_values in (
        ("Non-exact Real-B median", old_finite, new_finite),
        ("Non-exact Real-B q95", old_finite, new_finite),
        ("Non-exact Real-B worst", old_finite, new_finite),
    ):
        quantile = 0.95 if "q95" in metric else None
        old_value = (
            np.quantile(old_values, quantile)
            if quantile
            else (np.max(old_values) if "worst" in metric else np.median(old_values))
        )
        new_value = (
            np.quantile(new_values, quantile)
            if quantile
            else (np.max(new_values) if "worst" in metric else np.median(new_values))
        )
        rows.append({"metric": metric, "old_ABC": float(old_value), "equal_ABC": float(new_value)})
    comparison = pd.DataFrame(rows)
    comparison["difference_equal_minus_old"] = comparison["equal_ABC"] - comparison["old_ABC"]
    transition = statewise.loc[
        :,
        [
            "state_id",
            "old_state_Q",
            "new_state_Q",
            "old_exact",
            "new_exact",
            "old_real_B_CNMSE",
            "new_real_B_CNMSE",
            "old_shareable",
            "new_shareable",
            "transition_class",
            "exact_transition_class",
            "state_Q_changed",
            "old_new_real_B_window_note",
        ],
    ].copy()
    detail = {
        "old_failure_count": int(statewise["old_failure"].sum()),
        "new_failure_count": int(statewise["new_failure"].sum()),
        "old_failure_to_success": int(
            (statewise["transition_class"] == "failure_to_success").sum()
        ),
        "old_failure_to_failure": int(
            (statewise["transition_class"] == "failure_to_failure").sum()
        ),
        "old_success_to_failure": int(
            (statewise["transition_class"] == "success_to_failure").sum()
        ),
        "state_Q_changed_count": int(statewise["state_Q_changed"].sum()),
        "old_new_real_B_window_note": (
            "old B=4915 ownership samples versus equal ABC B=8192 ownership samples; "
            "old/new Real-B difference is an end-to-end segmentation comparison"
        ),
    }
    return comparison, transition, detail


def main() -> None:
    print("Scenario 2 equal-length ABC ablation started.", flush=True)
    before = _protection_snapshot()
    observed = {"data/raw": before["data/raw"]["sha256"]}
    observed.update(
        {
            name: item["sha256"]
            for name, item in before["result_dirs"].items()
            if name not in {"scenario_2_C2_to_Aend", "scenario_2_C2_to_Aend_unified_model_capacity"}
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

    partition = build_equal_abc_partition()
    state_frame = pd.DataFrame(build_state_table()).sort_values("state_id").reset_index(drop=True)
    old_stage = (
        pd.read_csv(BASELINE_ROOT / "a_end_stage_map.csv")
        .sort_values("state_id")
        .reset_index(drop=True)
    )
    if not np.array_equal(old_stage["state_id"].to_numpy(dtype=np.int64), np.arange(STATE_COUNT)):
        raise RuntimeError("旧Aend stage map state_id不完整")
    if not old_stage["C2_available"].astype(bool).all():
        raise RuntimeError("旧正式结果显示存在C2不可用状态")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(RESULT_ROOT / "segment_definition.json", segment_definition())
    _append_log(
        "\n".join(
            [
                "研究目标：只改变ABC ownership为等长8192/8192/8192，"
                "重新评价统一P9-M0-lambda=1e-8行为模型、指纹、C2→Aend检索和Real-B。",
                "旧不等长ABC=12288/4915/7373；本轮不修改signal_segmentation默认配置，"
                "不复用旧processed cache或旧Real-B矩阵。",
                f"equal partition={partition.as_dict()}；"
                f"segment-local valid lengths={VALID_LENGTHS}；"
                f"orders={list(ORDERS)}；memory={MEMORY}；lambda={RIDGE_LAMBDA}。",
                "Y-Aend=state-specific actual last ILC column；Y-C2=actual ILC2；"
                "Aend/C2结构完全统一。",
            ]
        )
    )

    print("Step 1/6: rebuild equal-ABC canonical pairs and fit 425 states", flush=True)
    theta_a = np.empty((STATE_COUNT, N_COMPLEX_COEFFICIENTS), dtype=np.complex128)
    theta_c = np.empty_like(theta_a)
    real_b_waveforms = np.empty((STATE_COUNT, VALID_LENGTH), dtype=np.complex128)
    model_rows: list[dict[str, Any]] = []
    common_b_input: np.ndarray | None = None
    for position, state_id in enumerate(range(STATE_COUNT), start=1):
        prepared = prepare_equal_abc_state(state_id)
        if common_b_input is None:
            common_b_input = prepared.common_b_input.copy()
        elif not np.array_equal(common_b_input, prepared.common_b_input):
            raise RuntimeError(f"state={state_id} common-B input与state0不一致")
        row, theta_a_row, theta_c_row, real_b = fit_equal_abc_state(prepared)
        theta_a[state_id] = theta_a_row
        theta_c[state_id] = theta_c_row
        real_b_waveforms[state_id] = real_b
        model_rows.append(row)
        if position % 10 == 0 or position == STATE_COUNT:
            print(f"equal ABC model: processed {position} / {STATE_COUNT} states", flush=True)
    if common_b_input is None:
        raise RuntimeError("common-B input未生成")
    model_metrics = pd.DataFrame(model_rows)
    if model_metrics.shape[0] != STATE_COUNT:
        raise RuntimeError("equal_ABC_state_model_metrics状态数错误")

    state_frame["ilc_A_end"] = (
        model_metrics.set_index("state_id")
        .loc[state_frame["state_id"], "ilc_A_end"]
        .to_numpy(dtype=np.int64)
    )
    state_frame["C2_available"] = True
    state_frame["N_ilc_available"] = state_frame["ilc_A_end"]
    if not np.array_equal(
        state_frame["ilc_A_end"].to_numpy(dtype=np.int64),
        old_stage["ilc_A_end"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("等长分段不应改变Aend stage map")
    stage_map = state_frame.loc[
        :,
        [
            "state_id",
            "funMng",
            "funAng",
            "secMng",
            "secAng",
            "Vm",
            "Pin",
            "N_ilc_available",
            "ilc_A_end",
            "C2_available",
        ],
    ].copy()
    _write_frame(stage_map, RESULT_ROOT / "a_end_stage_map.csv")

    raw_scalars = load_raw_scalar_metrics(progress_interval=50)
    metrics = state_frame.merge(
        model_metrics,
        on=["state_id", "ilc_A_end"],
        how="left",
        validate="one_to_one",
    )
    metrics = metrics.merge(raw_scalars, on="state_id", how="left", validate="one_to_one")
    metrics = metrics.rename(columns={"nmse_withoutdpd": "nmse_withoutdpd_dB"})
    metrics["load_config"] = [
        f"funMng={_physical_mismatch(row.funMng)}, funAng={int(row.funAng)}°, "
        f"secMng={_physical_mismatch(row.secMng)}, secAng={int(row.secAng)}°"
        for row in metrics.itertuples(index=False)
    ]
    metrics["Gap_A_dB"] = metrics["Y_Aend_B_NMSE_dB"] - metrics["Y_Aend_train_NMSE_dB"]
    metrics["Gap_C_dB"] = metrics["Y_C2_B_NMSE_dB"] - metrics["Y_C2_train_NMSE_dB"]
    metrics["Delta_AC_B_dB"] = metrics["Y_C2_B_NMSE_dB"] - metrics["Y_Aend_B_NMSE_dB"]
    if metrics.shape[0] != STATE_COUNT or not np.all(
        np.isfinite(
            metrics[
                [
                    "Y_Aend_train_NMSE_dB",
                    "Y_Aend_B_NMSE_dB",
                    "Y_C2_train_NMSE_dB",
                    "Y_C2_B_NMSE_dB",
                    "nmse_withoutdpd_dB",
                    "acpr_withoutdpd_avg_dBc",
                ]
            ].to_numpy(dtype=float)
        )
    ):
        raise RuntimeError("equal ABC state model metrics存在缺失或非finite")
    _write_frame(metrics, RESULT_ROOT / "equal_ABC_state_model_metrics.csv")

    phi_common = build_mp_basis(common_b_input, ORDERS, MEMORY)
    if phi_common.shape != (FINGERPRINT_LENGTH, N_COMPLEX_COEFFICIENTS):
        raise RuntimeError(f"new common-B Phi shape错误：{phi_common.shape}")
    fingerprints_a = (phi_common @ theta_a.T).T.astype(np.complex128, copy=False)
    fingerprints_c = (phi_common @ theta_c.T).T.astype(np.complex128, copy=False)
    if (
        fingerprints_a.shape != (STATE_COUNT, FINGERPRINT_LENGTH)
        or fingerprints_c.shape != fingerprints_a.shape
    ):
        raise RuntimeError("equal ABC fingerprint shape错误")
    if not np.all(np.isfinite(fingerprints_a)) or not np.all(np.isfinite(fingerprints_c)):
        raise RuntimeError("equal ABC fingerprint包含非finite")
    _write_npz(
        RESULT_ROOT / "equal_ABC_model_coefficients.npz",
        {
            "state_ids": np.arange(STATE_COUNT, dtype=np.int64),
            "theta_Aend": theta_a,
            "theta_C2": theta_c,
            "Aend_stage_map": metrics["ilc_A_end"].to_numpy(dtype=np.int64),
            "orders": np.asarray(ORDERS, dtype=np.int64),
            "memory_orders": np.asarray(sorted(MEMORY), dtype=np.int64),
            "memory_taps": np.asarray([MEMORY[key] for key in sorted(MEMORY)], dtype=np.int64),
            "ridge_lambda": np.asarray([RIDGE_LAMBDA], dtype=np.float64),
            "segment_lengths": np.asarray([SEGMENT_LENGTH] * 3, dtype=np.int64),
            "valid_lengths": np.asarray([VALID_LENGTH] * 3, dtype=np.int64),
        },
    )
    _write_npz(
        RESULT_ROOT / "equal_ABC_lut_fingerprints_Aend.npz",
        {
            "state_ids": np.arange(STATE_COUNT, dtype=np.int64),
            "common_B_input": common_b_input,
            "Y_Aend_fingerprints": fingerprints_a,
        },
    )
    _write_npz(
        RESULT_ROOT / "equal_ABC_query_fingerprints_C2.npz",
        {
            "state_ids": np.arange(STATE_COUNT, dtype=np.int64),
            "common_B_input": common_b_input,
            "Q_C2": fingerprints_c,
        },
    )

    print("Step 2/6: rebuild equal-ABC Real-B matrix and C2->Aend retrieval", flush=True)
    real_b_distance = compute_equal_cnmse_distance_matrix(real_b_waveforms, real_b_waveforms)
    real_b_ranking = distance_to_ranks(real_b_distance)
    retrieval_distance = compute_equal_cnmse_distance_matrix(fingerprints_c, fingerprints_a)
    retrieval_ranking = distance_to_ranks(retrieval_distance)
    retrieval_results, diagnostics, retrieval_summary = build_equal_retrieval_results(
        retrieval_distance, real_b_distance, stage_map
    )
    _write_npz(
        RESULT_ROOT / "equal_ABC_retrieval_distance_matrix.npz",
        {"state_ids": np.arange(STATE_COUNT, dtype=np.int64), "D_C2_Aend": retrieval_distance},
    )
    _write_npz(
        RESULT_ROOT / "equal_ABC_retrieval_ranking_matrix.npz",
        {"state_ids": np.arange(STATE_COUNT, dtype=np.int64), "R_C2_Aend": retrieval_ranking},
    )
    _write_npz(
        RESULT_ROOT / "equal_ABC_real_B_distance_matrix.npz",
        {
            "state_ids": np.arange(STATE_COUNT, dtype=np.int64),
            "D_B": real_b_distance,
            "R_B": real_b_ranking,
        },
    )
    _write_frame(retrieval_results, RESULT_ROOT / "retrieval_results.csv")
    _write_frame(diagnostics, RESULT_ROOT / "retrieved_real_B_cnmse.csv")
    _write_frame(pd.DataFrame([retrieval_summary]), RESULT_ROOT / "retrieval_summary.csv")
    nonexact = (
        retrieval_results.loc[retrieval_results["State_n_R"] != retrieval_results["State_n_Q"]]
        .sort_values("State_n_R")
        .reset_index(drop=True)
    )
    failed = (
        retrieval_results.loc[retrieval_results["failure"]]
        .sort_values(["retrieved_real_B_CNMSE_dB", "State_n_R"], ascending=[False, True])
        .reset_index(drop=True)
    )
    failed = failed.merge(
        metrics.loc[
            :,
            [
                "state_id",
                "load_config",
                "nmse_withoutdpd_dB",
                "acpr_withoutdpd_avg_dBc",
                "Y_Aend_train_NMSE_dB",
                "Y_Aend_B_NMSE_dB",
                "Y_C2_train_NMSE_dB",
                "Y_C2_B_NMSE_dB",
            ],
        ],
        left_on="State_n_R",
        right_on="state_id",
        how="left",
        validate="one_to_one",
    ).drop(columns=["state_id"])
    failed["distance_above_threshold_dB"] = (
        failed["retrieved_real_B_CNMSE_dB"] - DPD_SHAREABLE_THRESHOLD_DB
    )
    _write_frame(nonexact, RESULT_ROOT / "nonexact_retrieval_states.csv")
    _write_frame(failed, RESULT_ROOT / "failed_retrieval_states.csv")

    print("Step 3/6: correlations, quartiles, class and gap summaries", flush=True)
    statewise = metrics.merge(
        retrieval_results.loc[
            :,
            [
                "State_n_R",
                "State_n_Q",
                "exact_hit",
                "true_state_rank",
                "retrieved_real_B_CNMSE_dB",
                "dpd_shareable",
                "failure",
            ],
        ],
        left_on="state_id",
        right_on="State_n_R",
        how="inner",
        validate="one_to_one",
    ).rename(columns={"State_n_Q": "state_id_Q", "State_n_R": "state_id_R"})
    correlations = build_correlation_summary(statewise)
    quartiles = build_quartile_summary(statewise)
    classes = build_retrieval_class_summary(statewise)
    gaps = build_generalization_gap_summary(statewise)
    _write_frame(correlations, RESULT_ROOT / "analysis_correlation_summary.csv")
    _write_frame(quartiles, RESULT_ROOT / "pa_nonlinearity_quartile_summary.csv")
    _write_frame(classes, RESULT_ROOT / "retrieval_class_model_summary.csv")
    _write_frame(gaps, RESULT_ROOT / "generalization_gap_summary.csv")

    print("Step 4/6: compare old unequal-ABC and new equal-ABC end-to-end results", flush=True)
    old_metrics = load_baseline_state_metrics(
        ALL_ILC_ROOT / "all_ilc_model_metrics.csv", ALL_ILC_ROOT / "ilc_availability.csv"
    )
    old_retrieval = _load_old_retrieval()
    comparison, transition, transition_detail = _old_new_comparison(
        old_metrics, metrics, old_retrieval, retrieval_results
    )
    _write_frame(comparison, RESULT_ROOT / "comparison_vs_original_ABC.csv")
    _write_frame(transition, RESULT_ROOT / "original_vs_equal_ABC_state_transition.csv")

    print("Step 5/6: generate Python/matplotlib figures", flush=True)
    figure_main = plot_equal_abc_state_metrics(
        statewise, RESULT_ROOT / "figure_state_metrics_and_retrieved_real_B_equal_ABC.png"
    )
    figure_detail = plot_nonexact_failure_detail(
        statewise, RESULT_ROOT / "figure_nonexact_and_failure_detail_equal_ABC.png"
    )

    old_values = old_retrieval["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    old_exact = old_retrieval["exact_hit"].to_numpy(dtype=bool)
    old_finite = old_values[(~old_exact) & np.isfinite(old_values)]
    old_summary = {
        "exact_hit_count": int(old_exact.sum()),
        "exact_hit_rate": float(old_exact.mean()),
        "nonexact_count": int((~old_exact).sum()),
        "shareable_count": int(old_retrieval["dpd_shareable"].sum()),
        "shareable_rate": float(old_retrieval["dpd_shareable"].mean()),
        "failure_count": int(old_retrieval["failure"].sum()),
        "nonexact_real_B_median_dB": float(np.median(old_finite)),
        "nonexact_real_B_q95_dB": float(np.quantile(old_finite, 0.95)),
        "nonexact_real_B_worst_dB": float(np.max(old_finite)),
    }
    validation: dict[str, Any] = {
        "study": "equal-length ABC segmentation ablation for Scenario 2 C2->Aend retrieval",
        "scenario": 2,
        "segment_definition": segment_definition(),
        "model_orders": list(ORDERS),
        "memory": MEMORY,
        "lambda": RIDGE_LAMBDA,
        "n_complex_coefficients": N_COMPLEX_COEFFICIENTS,
        "Aend_and_C2_share_same_model": True,
        "Aend_definition": "actual state-specific last ILC column",
        "C2_definition": "actual ILC column 2",
        "state_count": STATE_COUNT,
        "state_id_range": [0, STATE_COUNT - 1],
        "C2_available_count": STATE_COUNT,
        "Aend_stage_map_unchanged_vs_original": True,
        "common_B_probe_rebuilt_for_equal_B": True,
        "common_B_input_shape": list(common_b_input.shape),
        "common_B_valid_length": VALID_LENGTH,
        "lut_fingerprint_shape": list(fingerprints_a.shape),
        "query_fingerprint_shape": list(fingerprints_c.shape),
        "fingerprint_dtype": str(fingerprints_a.dtype),
        "fingerprints_all_finite": bool(
            np.all(np.isfinite(fingerprints_a)) and np.all(np.isfinite(fingerprints_c))
        ),
        "retrieval_matrix_shape": list(retrieval_distance.shape),
        "ranking_matrix_shape": list(retrieval_ranking.shape),
        "candidate_count_per_query": STATE_COUNT,
        "self_match": True,
        "minimum_tie_count_rows": retrieval_summary["minimum_tie_count_rows"],
        "real_B_matrix_reused": False,
        "real_B_reference": "equal-ABC canonical OFF.B.output[2:]",
        "real_B_valid_waveform_shape": list(real_b_waveforms.shape),
        "real_B_distance_shape": list(real_b_distance.shape),
        "real_B_diagonal_all_negative_infinity": bool(
            np.all(np.isneginf(np.diag(real_b_distance)))
        ),
        "DPD_shareable_threshold_dB": DPD_SHAREABLE_THRESHOLD_DB,
        "threshold_rule": "Real-B CNMSE < -40 dB",
        "threshold_is_strict": True,
        "retrieval_summary": retrieval_summary,
        "old_unequal_ABC_lengths": {"A": 12288, "B": 4915, "C": 7373},
        "new_equal_ABC_lengths": {"A": 8192, "B": 8192, "C": 8192},
        "old_retrieval_summary": old_summary,
        "old_new_real_B_window_note": transition_detail["old_new_real_B_window_note"],
        "old_vs_equal_ABC_transition": transition_detail,
        "old_vs_equal_ABC_comparison_metrics": comparison.to_dict("records"),
        "correlation_rows": int(correlations.shape[0]),
        "quartile_summary": quartiles.to_dict("records"),
        "generalization_gap_summary": gaps.to_dict("records"),
        "raw_data_modified": False,
        "protected_old_results_unchanged": None,
        "figures": {"main": figure_main, "detail": figure_detail},
        "output_files": {
            path.name: str(path) for path in sorted(RESULT_ROOT.iterdir()) if path.is_file()
        },
    }
    after = _protection_snapshot()
    protection = _verify_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("equal ABC执行期间raw或旧结果/脚本发生变化")
    validation["protected_old_results_unchanged"] = True
    validation["protection_before"] = before
    validation["protection_after"] = after
    validation["protection_verification"] = protection
    _write_json(RESULT_ROOT / "validation.json", validation)

    _append_log(
        "\n".join(
            [
                f"equal ABC canonical/model completed for {STATE_COUNT} states；"
                "A/B/C ownership=8192/8192/8192；valid=8190/8190/8190。",
                f"Aend stage map unchanged={validation['Aend_stage_map_unchanged_vs_original']}；"
                f"C2 available={validation['C2_available_count']}/{STATE_COUNT}。",
                f"theta shapes={theta_a.shape}/{theta_c.shape}；"
                f"fingerprints={fingerprints_a.shape}/{fingerprints_c.shape}/{fingerprints_a.dtype}；"
                f"retrieval={retrieval_distance.shape}；new Real-B matrix={real_b_distance.shape} "
                "diagonal=-Inf。",
                f"Equal ABC retrieval exact/shareable/failure="
                f"{retrieval_summary['exact_hit_count']}/{retrieval_summary['DPD_shareable_count']}/"
                f"{retrieval_summary['failure_count']}；",
                f"nonexact Real-B median/q95/worst="
                f"{retrieval_summary['nonexact_B_CNMSE_median']}/"
                f"{retrieval_summary['nonexact_B_CNMSE_q95']}/"
                f"{retrieval_summary['nonexact_B_CNMSE_max']} dB。",
                f"Old unequal ABC exact/shareable/failure="
                f"{old_summary['exact_hit_count']}/{old_summary['shareable_count']}/"
                f"{old_summary['failure_count']}；",
                f"State_Q changed={transition_detail['state_Q_changed_count']}；"
                f"transitions={transition['transition_class'].value_counts().to_dict()}。",
                "Old/new Real-B uses different B windows; comparison is end-to-end segmentation "
                "comparison, not fixed-window fingerprint gain.",
                f"Protection all_protected_unchanged={protection['all_protected_unchanged']}；"
                f"raw_data_modified=False；result_root={RESULT_ROOT}。",
            ]
        )
    )
    _append_handoff(
        "完成Scenario 2 ABC三等分消融：新A/B/C=8192/8192/8192，"
        "segment-local有效长度=8190/8190/8190，冻结P9-M0-lambda=1e-8，"
        "Aend/C2结构统一；equal-ABC检索exact/shareable/failure="
        f"{retrieval_summary['exact_hit_count']}/{retrieval_summary['DPD_shareable_count']}/"
        f"{retrieval_summary['failure_count']}，旧14 failure中failure→success="
        f"{transition_detail['old_failure_to_success']}，"
        f"state_Q changed={transition_detail['state_Q_changed_count']}；"
        "新Real-B矩阵重新计算，旧结果和raw保护通过，结果写入"
        f"{RESULT_ROOT}。"
    )
    print("Scenario 2 equal-length ABC ablation completed.", flush=True)
    print(f"Results: {RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
