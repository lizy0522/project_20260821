"""Run and persist the Scenario 2 Y-C3 to Y-A2 LUT retrieval study."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_retrieval.scenario_2.scenario_2_C3_to_A2_retrieval.plot_scenario2_c3_to_a2_retrieval import (  # noqa: E402,E501
    plot_retrieved_real_b_cnmse_by_state,
    plot_retrieved_real_b_cnmse_distribution,
    plot_state_retrieval_mapping,
)
from behavior_fingerprint_retrieval.shared.scenario2_c3_to_a2_retrieval import (  # noqa: E402
    run_retrieval_analysis,
)

TASK_NAME = "scenario_2_C3_to_A2_retrieval"
REFERENCE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_ranking_consistency"
    / "scenario_2"
)
A123_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_y_lut_fusion_a123_analysis"
    / "scenario_2"
)
ALL_ILC_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_all_ilc_analysis"
    / "scenario_2"
)
RESULT_ROOT = (
    PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /TASK_NAME / "scenario_2_C3_to_A2"  # noqa: E501
)
LOG_ROOT = PROJECT_ROOT / "work_logs" / "behavior_fingerprint_retrieval" / "scenario_2" /TASK_NAME
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
    "scenario_2_y_lut_fusion_A123": A123_ROOT,
    "scenario_2_y_lut_weighted_fusion": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario_2_y_lut_weighted_fusion",
    "scenario_2_ridge_analysis": PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /"scenario_2_ridge_analysis",
    "scenario_2_xy_analysis": PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /"scenario_2_xy_analysis",
}
PROTECTED_TREE_DIRS = {
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
        if file.is_file() and "__pycache__" not in file.parts and file.suffix.lower() != ".pyc"
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
        (file for file in raw_root.rglob("*") if file.is_file()),
        key=lambda item: str(item).lower(),
    )
    for file in files:
        digest.update(str(file.relative_to(raw_root)).replace("\\", "/").encode() + b"\0")
        digest.update(file.stat().st_size.to_bytes(8, "little"))
    return digest.hexdigest(), len(files), sum(file.stat().st_size for file in files)


def _protection_snapshot() -> dict[str, Any]:
    raw_hash, raw_count, raw_bytes = _raw_manifest_digest()
    snapshot: dict[str, Any] = {
        "data/raw": {"sha256": raw_hash, "file_count": raw_count, "bytes": raw_bytes},
        "result_dirs": {},
        "script_dirs": {},
    }
    for name, path in PROTECTED_RESULT_DIRS.items():
        snapshot["result_dirs"][name] = {
            "path": str(path),
            "exists": path.is_dir(),
            "sha256": _tree_digest(path) if path.is_dir() else None,
        }
    for name, path in PROTECTED_TREE_DIRS.items():
        snapshot["script_dirs"][name] = {
            "path": str(path),
            "exists": path.is_dir(),
            "sha256": _tree_digest(path) if path.is_dir() else None,
        }
    return snapshot


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")


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


def _write_frame(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        na_rep="NaN",
        float_format="%.17g",
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _append_log(text: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with EXECUTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] Scenario 2 - Y-C3 to Y-A2 LUT retrieval\n")
        handle.write(text.rstrip() + "\n")


def _append_handoff(text: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{timestamp} | {TASK_NAME}新任务完成\n")
        handle.write(text.rstrip() + "\n")


def _assert_protection(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    unchanged: dict[str, Any] = {"data/raw": {}, "result_dirs": {}, "script_dirs": {}}
    raw_before = before["data/raw"]
    raw_after = after["data/raw"]
    unchanged["data/raw"] = {
        "sha256_unchanged": raw_before["sha256"] == raw_after["sha256"],
        "file_count_unchanged": raw_before["file_count"] == raw_after["file_count"],
        "bytes_unchanged": raw_before["bytes"] == raw_after["bytes"],
    }
    for group in ("result_dirs", "script_dirs"):
        for name, item in before[group].items():
            after_item = after[group][name]
            unchanged[group][name] = {
                "sha256_unchanged": item["sha256"] == after_item["sha256"],
                "before": item["sha256"],
                "after": after_item["sha256"],
            }
    all_unchanged = all(unchanged["data/raw"].values())
    all_unchanged = all_unchanged and all(
        item["sha256_unchanged"]
        for group in ("result_dirs", "script_dirs")
        for item in unchanged[group].values()
    )
    unchanged["all_protected_unchanged"] = bool(all_unchanged)
    return unchanged


def main() -> None:
    print("Scenario 2 Y-C3 to Y-A2 LUT retrieval started.")
    print(f"Project root: {PROJECT_ROOT}")
    before = _protection_snapshot()
    expected_mismatches = []
    observed_expected = {
        "data/raw": before["data/raw"]["sha256"],
        **{name: item["sha256"] for name, item in before["result_dirs"].items()},
        **{name: item["sha256"] for name, item in before["script_dirs"].items()},
    }
    for name, expected in EXPECTED_PROTECTED_HASHES.items():
        if observed_expected.get(name) != expected:
            expected_mismatches.append(
                f"{name}: observed={observed_expected.get(name)} expected={expected}"
            )
    if expected_mismatches:
        raise RuntimeError("保护输入SHA与当前基线不一致：" + "; ".join(expected_mismatches))
    print("Step 1/4: load frozen A123, Scenario 2 Real-B and effective-stage metadata")
    result = run_retrieval_analysis(A123_ROOT, REFERENCE_ROOT, ALL_ILC_ROOT)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Step 2/4: write matrices, retrieval tables and validation")
    _write_npz(
        RESULT_ROOT / "retrieval_distance_matrix.npz",
        {
            "state_ids": result.inputs.state_ids,
            "D_QA2": result.query_to_a2_distance,
            "R_QA2": result.query_to_a2_ranking,
        },
    )
    _write_npz(
        RESULT_ROOT / "real_B_distance_matrix.npz",
        {
            "state_ids": result.inputs.state_ids,
            "D_B": result.real_b_distance,
            "R_B": result.real_b_ranking,
        },
    )
    _write_frame(result.retrieval_results, RESULT_ROOT / "retrieval_results.csv")
    _write_frame(result.real_b_diagnostics, RESULT_ROOT / "retrieved_real_B_cnmse.csv")
    _write_frame(result.retrieval_summary, RESULT_ROOT / "retrieval_summary.csv")
    _write_frame(result.threshold_sweep, RESULT_ROOT / "threshold_sweep.csv")
    _write_frame(result.worst_retrieval_states, RESULT_ROOT / "worst_retrieval_states.csv")

    print("Step 3/4: generate three Python/matplotlib figures")
    figure1 = plot_state_retrieval_mapping(
        result.retrieval_results,
        RESULT_ROOT / "figure1_state_retrieval_mapping.png",
    )
    figure2 = plot_retrieved_real_b_cnmse_by_state(
        result.retrieval_results,
        result.real_b_diagnostics,
        RESULT_ROOT / "figure2_retrieved_real_B_cnmse_by_state.png",
    )
    figure3 = plot_retrieved_real_b_cnmse_distribution(
        result.real_b_diagnostics,
        result.retrieval_results,
        RESULT_ROOT / "figure3_retrieved_real_B_cnmse_distribution.png",
    )
    after = _protection_snapshot()
    protection = _assert_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("保护输入或旧结果SHA发生变化，拒绝完成正式输出")

    print("Step 4/4: write validation and task logs")
    result.validation["figures"] = {"figure1": figure1, "figure2": figure2, "figure3": figure3}
    result.validation["output_files"] = {
        "retrieval_distance_matrix": str(RESULT_ROOT / "retrieval_distance_matrix.npz"),
        "retrieval_results": str(RESULT_ROOT / "retrieval_results.csv"),
        "real_B_distance_matrix": str(RESULT_ROOT / "real_B_distance_matrix.npz"),
        "retrieved_real_B_cnmse": str(RESULT_ROOT / "retrieved_real_B_cnmse.csv"),
        "retrieval_summary": str(RESULT_ROOT / "retrieval_summary.csv"),
        "threshold_sweep": str(RESULT_ROOT / "threshold_sweep.csv"),
        "worst_retrieval_states": str(RESULT_ROOT / "worst_retrieval_states.csv"),
        "figure1": str(RESULT_ROOT / "figure1_state_retrieval_mapping.png"),
        "figure2": str(RESULT_ROOT / "figure2_retrieved_real_B_cnmse_by_state.png"),
        "figure3": str(RESULT_ROOT / "figure3_retrieved_real_B_cnmse_distribution.png"),
        "validation": str(RESULT_ROOT / "validation.json"),
    }
    result.validation["protection_before"] = before
    result.validation["protection_after"] = after
    result.validation["protection_verification"] = protection
    _write_json(RESULT_ROOT / "validation.json", result.validation)

    summary = result.retrieval_summary.iloc[0]
    nonexact_stats = "; ".join(
        f"{key}={summary[key]:.9g}"
        for key in (
            "non_exact_B_CNMSE_mean",
            "non_exact_B_CNMSE_median",
            "non_exact_B_CNMSE_q05",
            "non_exact_B_CNMSE_q25",
            "non_exact_B_CNMSE_q75",
            "non_exact_B_CNMSE_q95",
            "non_exact_B_CNMSE_min",
            "non_exact_B_CNMSE_max",
        )
    )
    _append_log(
        "\n".join(
            [
                "研究问题：固定Y-C3在线行为指纹检索Y-A2 LUT，并用真实OFF-B行为距离验证检索状态。",
                "输入均来自冻结A123/Scenario 2/all-ILC结果；未重新读取raw、"
                "训练模型、扫描lambda、改变MP或canonical。",
                f"Y-C3 Query={result.inputs.query_c3.shape}/{result.inputs.query_c3.dtype}；"
                f"Y-A2 LUT={result.inputs.lut_a2.shape}/{result.inputs.lut_a2.dtype}；"
                f"candidate_count={summary['candidate_count_per_query']:.0f}；self_match_included=True。",
                f"D_QA2 shape={result.query_to_a2_distance.shape}；"
                "C3→A2 ranking regression="
                f"{result.validation['c3_to_a2_ranking_regression']['pass']}。",
                f"完成425个State_R Top-1检索；exact={summary['exact_hit_count']:.0f}/"
                f"{summary['exact_hit_rate']:.9g}；non-exact={summary['non_exact_count']:.0f}；"
                f"Top3={summary['top3_hit_rate']:.9g}；Top5={summary['top5_hit_rate']:.9g}；"
                "true-rank median/max="
                f"{summary['true_state_rank_median']:.9g}/"
                f"{summary['true_state_rank_max']:.0f}。",
                f"D_B shape={result.real_b_distance.shape}；Real-B ranking regression="
                f"{result.validation['real_B_ranking_regression']['pass']}；"
                "exact diagonal -inf="
                f"{result.validation['real_B_exact_diagonal_all_negative_infinity']}。",
                f"Non-exact有限Real-B统计：{nonexact_stats}；"
                f"regret mean/median/max={summary['non_exact_regret_mean']:.9g}/"
                f"{summary['non_exact_regret_median']:.9g}/{summary['non_exact_regret_max']:.9g}。",
                "threshold sweep仅为描述性统计，未冻结behavioral success threshold。",
                f"三幅图均由Python/matplotlib生成；Figure2有限y范围="
                f"[{figure2['y_min']:.15g},{figure2['y_max']:.15g}]。",
                f"保护复核all_protected_unchanged={protection['all_protected_unchanged']}；"
                "未执行破坏性Git操作；结果目录："
                f"{RESULT_ROOT}。",
            ]
        )
    )
    _append_handoff(
        f"完成Scenario 2 C3→A2实际LUT检索：425个Query、425个候选；"
        f"exact={summary['exact_hit_count']:.0f}（{summary['exact_hit_rate']:.6g}），"
        f"Top3/Top5={summary['top3_hit_rate']:.6g}/{summary['top5_hit_rate']:.6g}；"
        f"C3→A2与Real-B→Real-B排名回归均通过；生成检索矩阵、行为诊断、threshold sweep和三幅图，"
        f"保护复核通过，结果位于{RESULT_ROOT}。"
    )
    print("Scenario 2 Y-C3 to Y-A2 LUT retrieval completed.")
    print(f"Results: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
