"""Run and persist the formal Scenario 2 Y-C2 -> Y-A_end retrieval study."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_lut_retrieval.plot_scenario2_c2_to_aend_retrieval import (  # noqa: E402
    plot_dpd_shareable_success_by_state,
    plot_real_b_cnmse_by_state,
    plot_real_b_cnmse_distribution,
    plot_state_retrieval_mapping,
)
from behavior_fingerprint_lut_retrieval.scenario2_c2_to_aend_retrieval import (  # noqa: E402
    run_c2_aend_retrieval_analysis,
)

TASK_NAME = "behavior_fingerprint_lut_retrieval"
REFERENCE_ROOT = (
    PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2"
)
ALL_ILC_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_all_ilc"
)
A123_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_y_lut_fusion_A123"
)
PREVIOUS_C3A2_ROOT = (
    PROJECT_ROOT / "results" / TASK_NAME / "scenario_2_C3_to_A2"
)
RESULT_ROOT = PROJECT_ROOT / "results" / TASK_NAME / "scenario_2_C2_to_Aend"
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
    "scenario_2_y_lut_fusion_A123": A123_ROOT,
    "scenario_2_y_lut_weighted_fusion": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_y_lut_weighted_fusion",
    "scenario_2_C3_to_A2": PREVIOUS_C3A2_ROOT,
    "scenario_2_ridge_analysis": PROJECT_ROOT
    / "results" / "behavior_model" / "scenario_2_ridge_analysis",
    "scenario_2_xy_analysis": PROJECT_ROOT
    / "results" / "behavior_model" / "scenario_2_xy_analysis",
}
PROTECTED_SCRIPT_DIRS = {
    "behavior_fingerprint_ranking_consistency": PROJECT_ROOT
    / "scripts" / "behavior_fingerprint_ranking_consistency",
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
        if file.is_file() and "__pycache__" not in file.parts and file.suffix.lower() != ".pyc"
    )
    for file in sorted(files, key=lambda item: str(item).lower()):
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest_digest() -> tuple[str, int, int]:
    raw_root = Path(r"\\?\{}".format(PROJECT_ROOT / "data" / "raw"))
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
    for name, path in PROTECTED_SCRIPT_DIRS.items():
        snapshot["script_dirs"][name] = {
            "path": str(path),
            "exists": path.is_dir(),
            "sha256": _tree_digest(path) if path.is_dir() else None,
        }
    return snapshot


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")


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
        handle.write(f"\n\n[{timestamp}] Scenario 2 - Y-C2 to Y-Aend retrieval\n")
        handle.write(text.rstrip() + "\n")


def _append_handoff(text: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{timestamp} | {TASK_NAME}持续维护：Scenario 2 C2→Aend检索\n")
        handle.write(text.rstrip() + "\n")


def _protection_verification(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {"data/raw": {}, "result_dirs": {}, "script_dirs": {}}
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
    all_unchanged = all(result["data/raw"].values()) and all(
        item["sha256_unchanged"]
        for group in ("result_dirs", "script_dirs")
        for item in result[group].values()
    )
    result["all_protected_unchanged"] = bool(all_unchanged)
    return result


def main() -> None:
    print("Scenario 2 Y-C2 to Y-Aend retrieval started.")
    print(f"Project root: {PROJECT_ROOT}")
    before = _protection_snapshot()
    observed = {
        "data/raw": before["data/raw"]["sha256"],
        **{name: item["sha256"] for name, item in before["result_dirs"].items()},
        **{name: item["sha256"] for name, item in before["script_dirs"].items()},
    }
    mismatch = [
        f"{name}: observed={observed.get(name)} expected={expected}"
        for name, expected in EXPECTED_PROTECTED_HASHES.items()
        if observed.get(name) != expected
    ]
    if mismatch:
        raise RuntimeError("保护输入SHA与当前基线不一致：" + "; ".join(mismatch))

    print("Step 1/4: load frozen all-ILC theta, C2/Aend references and old C3/A2 result")
    result = run_c2_aend_retrieval_analysis(
        ALL_ILC_ROOT, A123_ROOT, PREVIOUS_C3A2_ROOT, REFERENCE_ROOT
    )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Step 2/4: write Aend/C2 arrays, retrieval tables and summaries")
    _write_frame(result.a_end_stage_map, RESULT_ROOT / "a_end_stage_map.csv")
    _write_npz(
        RESULT_ROOT / "lut_fingerprints_Aend.npz",
        {
            "state_ids": result.inputs.state_ids,
            "common_B_input": result.inputs.common_b_input,
            "Y_Aend_fingerprints": result.a_end_fingerprints,
            "Aend_stage": result.a_end_stage_map["ilc_A_end"].to_numpy(dtype=np.int64),
            "N_ilc_available": result.inputs.ilc_column_counts,
        },
    )
    _write_npz(
        RESULT_ROOT / "query_fingerprints_C2.npz",
        {
            "state_ids": result.inputs.state_ids,
            "common_B_input": result.inputs.common_b_input,
            "Q_C2": result.c2_query_fingerprint,
            "effective_C_stage": np.full(425, 2, dtype=np.int64),
        },
    )
    _write_npz(
        RESULT_ROOT / "retrieval_distance_matrix.npz",
        {"state_ids": result.inputs.state_ids, "D_C2_Aend": result.retrieval_distance},
    )
    _write_npz(
        RESULT_ROOT / "retrieval_ranking_matrix.npz",
        {"state_ids": result.inputs.state_ids, "R_C2_Aend": result.retrieval_ranking},
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
    _write_frame(result.dpd_shareable_summary, RESULT_ROOT / "dpd_shareable_summary.csv")
    _write_frame(result.failed_retrieval_states, RESULT_ROOT / "failed_retrieval_states.csv")
    _write_frame(result.condition_region_summary, RESULT_ROOT / "condition_region_summary.csv")
    _write_frame(result.oracle_shareable_rank, RESULT_ROOT / "oracle_shareable_rank.csv")
    _write_frame(result.comparison_vs_c3_a2, RESULT_ROOT / "comparison_vs_C3_to_A2.csv")

    print("Step 3/4: generate four Python/matplotlib figures")
    figure1 = plot_state_retrieval_mapping(
        result.retrieval_results,
        result.real_b_diagnostics,
        RESULT_ROOT / "figure1_state_retrieval_mapping.png",
    )
    figure2 = plot_real_b_cnmse_by_state(
        result.retrieval_results,
        result.real_b_diagnostics,
        RESULT_ROOT / "figure2_real_B_cnmse_by_state.png",
    )
    figure3 = plot_real_b_cnmse_distribution(
        result.retrieval_results,
        result.real_b_diagnostics,
        RESULT_ROOT / "figure3_real_B_cnmse_distribution.png",
    )
    figure4 = plot_dpd_shareable_success_by_state(
        result.retrieval_results,
        result.real_b_diagnostics,
        RESULT_ROOT / "figure4_dpd_shareable_success_by_state.png",
    )
    after = _protection_snapshot()
    protection = _protection_verification(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("保护输入或旧结果SHA发生变化，拒绝完成正式输出")

    result.validation["figures"] = {
        "figure1": figure1,
        "figure2": figure2,
        "figure3": figure3,
        "figure4": figure4,
    }
    result.validation["output_files"] = {
        "a_end_stage_map": str(RESULT_ROOT / "a_end_stage_map.csv"),
        "lut_fingerprints_Aend": str(RESULT_ROOT / "lut_fingerprints_Aend.npz"),
        "query_fingerprints_C2": str(RESULT_ROOT / "query_fingerprints_C2.npz"),
        "retrieval_distance_matrix": str(RESULT_ROOT / "retrieval_distance_matrix.npz"),
        "retrieval_ranking_matrix": str(RESULT_ROOT / "retrieval_ranking_matrix.npz"),
        "retrieval_results": str(RESULT_ROOT / "retrieval_results.csv"),
        "real_B_distance_matrix": str(RESULT_ROOT / "real_B_distance_matrix.npz"),
        "retrieved_real_B_cnmse": str(RESULT_ROOT / "retrieved_real_B_cnmse.csv"),
        "retrieval_summary": str(RESULT_ROOT / "retrieval_summary.csv"),
        "dpd_shareable_summary": str(RESULT_ROOT / "dpd_shareable_summary.csv"),
        "failed_retrieval_states": str(RESULT_ROOT / "failed_retrieval_states.csv"),
        "condition_region_summary": str(RESULT_ROOT / "condition_region_summary.csv"),
        "oracle_shareable_rank": str(RESULT_ROOT / "oracle_shareable_rank.csv"),
        "comparison_vs_C3_to_A2": str(RESULT_ROOT / "comparison_vs_C3_to_A2.csv"),
        "figure1": str(RESULT_ROOT / "figure1_state_retrieval_mapping.png"),
        "figure2": str(RESULT_ROOT / "figure2_real_B_cnmse_by_state.png"),
        "figure3": str(RESULT_ROOT / "figure3_real_B_cnmse_distribution.png"),
        "figure4": str(RESULT_ROOT / "figure4_dpd_shareable_success_by_state.png"),
        "validation": str(RESULT_ROOT / "validation.json"),
    }
    result.validation["protection_before"] = before
    result.validation["protection_after"] = after
    result.validation["protection_verification"] = protection
    _write_json(RESULT_ROOT / "validation.json", result.validation)

    summary = result.retrieval_summary.iloc[0]
    _append_log(
        "\n".join(
            [
                "研究问题：固定Y-C2 Query检索按每个状态实际最后ILC列构造的Y-Aend LUT，"
                "并以Real-B CNMSE执行严格<-40 dB的DPD-shareable判定。",
                "输入全部来自冻结all-ILC/A123/Scenario 2/C3→A2结果；未重新读取raw、"
                "训练模型、扫描lambda、改变MP、ABC、canonical或common-B probe。",
                f"C2_available={int(summary['C2_available_count']):.0f}/425；"
                f"N_ilc distribution={result.validation['N_ilc_distribution']}；"
                f"Aend distribution={result.validation['A_end_stage_distribution']}。",
                f"Aend LUT={result.a_end_fingerprints.shape}/{result.a_end_fingerprints.dtype}；"
                f"C2 Query={result.c2_query_fingerprint.shape}/"
                f"{result.c2_query_fingerprint.dtype}；"
                f"Aend/C2 regression={result.validation['a_end_fingerprint_regression']['pass']}/"
                f"{result.validation['c2_fingerprint_regression']['pass']}。",
                f"D_C2_Aend={result.retrieval_distance.shape}；R_C2_Aend="
                f"{result.retrieval_ranking.shape}；candidate=425；self_match=True；"
                f"minimum_tie_rows={summary['minimum_tie_count_rows']:.0f}。",
                f"Exact={summary['exact_hit_count']:.0f}/{summary['exact_hit_rate']:.9g}；"
                f"Non-exact={summary['non_exact_count']:.0f}；"
                f"true-rank median/max={summary['true_state_rank_median']:.9g}/"
                f"{summary['true_state_rank_max']:.0f}；"
                f"Top3/Top5={summary['top3_true_state_hit_rate']:.9g}/"
                f"{summary['top5_true_state_hit_rate']:.9g}。",
                f"DPD-shareable strict<-40 dB={summary['DPD_shareable_count']:.0f}/"
                f"{summary['DPD_shareable_rate']:.9g}；failure={summary['failure_count']:.0f}；"
                f"non-exact shareable={summary['non_exact_shareable_count']:.0f}/"
                f"{summary['non_exact_shareable_rate']:.9g}。",
                "Non-exact Real-B CNMSE median/q95/max="
                f"{summary['non_exact_B_CNMSE_median']:.9g}/"
                f"{summary['non_exact_B_CNMSE_q95']:.9g}/"
                f"{summary['non_exact_B_CNMSE_max']:.9g} dB；"
                "regret median/max="
                f"{summary['regret_median']:.9g}/{summary['regret_max']:.9g} dB。",
                f"Oracle shareable coverage Top3/Top5/Top10="
                f"{summary['top3_shareable_oracle_coverage']:.9g}/"
                f"{summary['top5_shareable_oracle_coverage']:.9g}/"
                f"{summary['top10_shareable_oracle_coverage']:.9g}。",
                "与旧C3→A2只做固定指标对比，不据此修改新方法；四幅图由Python/matplotlib生成。",
                f"保护复核all_protected_unchanged={protection['all_protected_unchanged']}；"
                f"结果目录：{RESULT_ROOT}。",
            ]
        )
    )
    _append_handoff(
        f"完成Scenario 2正式C2→Aend检索：425个状态全部真实存在C2，Aend按状态最后可用ILC列选择；"
        f"Exact={summary['exact_hit_count']:.0f}/{summary['exact_hit_rate']:.6g}，"
        f"DPD-shareable(strict<-40 dB)={summary['DPD_shareable_count']:.0f}/"
        f"{summary['DPD_shareable_rate']:.6g}，failure={summary['failure_count']:.0f}；"
        "Aend/C2与冻结响应回归、Real-B/R_RR回归和保护复核通过，结果及四幅图已写入"
        f"{RESULT_ROOT}。"
    )
    print("Scenario 2 Y-C2 to Y-Aend retrieval completed.")
    print(f"Results: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
