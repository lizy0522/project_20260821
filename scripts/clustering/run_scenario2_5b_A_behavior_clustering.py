"""Add deterministic cluster representatives to the validated 5B-A clusters.

This maintenance runner reuses the existing pairwise matrix and Complete-Link
implementation, first reproduces the current -40 dB member sets, and only then
adds minimax-medoid representative state references.  It never reads or copies
fingerprint or DPD payloads.
"""

# ruff: noqa: E402,E501,I001

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import UTC, datetime
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

for _thread_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_name] = "1"

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _core.behavior_indexed_dpd.clustering import (
    cluster_behavior_states,
    summarize_clusters,
)
from _core.behavior_indexed_dpd.clustering.representative_selection import (
    select_cluster_representative,
)
from clustering import run_scenario2_5b_A_yout_withoutdpd_ori_complete_link as legacy_runner


TASK_NAME = "scenario_2_yout_withoutdpd_ori_5B_A_complete_link_clustering"
RESULT_ROOT = PROJECT_ROOT / "results" / "clustering" / TASK_NAME
MATRIX_PATH = RESULT_ROOT / "pairwise_cnmse_5B_A_preprocessed.npz"
THRESHOLD_DIR = RESULT_ROOT / "threshold_m40p0dB"
OLD_ASSIGNMENTS = THRESHOLD_DIR / "cluster_assignments.csv"
OLD_SUMMARY = THRESHOLD_DIR / "cluster_summary.csv"
STATE_COUNT = 425
DEFAULT_THRESHOLD = -40.0
REPRO_TOLERANCE_DB = 1e-9


def _synthetic_representative_gates() -> dict[str, bool]:
    matrix = np.asarray(
        [
            [-np.inf, -50.0, -48.0, -40.5],
            [-50.0, -np.inf, -45.0, -43.0],
            [-48.0, -45.0, -np.inf, -44.0],
            [-40.5, -43.0, -44.0, -np.inf],
        ],
        dtype=np.float64,
    )
    representative = select_cluster_representative((0, 1, 2, 3), matrix, cluster_id=0)
    minimax_gate = representative.representative_state_id == 2

    tie_median_matrix = np.full((4, 4), -np.inf, dtype=np.float64)
    tie_median_matrix[0, 1:] = [-40.0, -42.0, -43.0]
    tie_median_matrix[1, 0] = -40.0
    tie_median_matrix[1, 2:] = [-45.0, -44.0]
    tie_median_matrix[2, 0] = -42.0
    tie_median_matrix[2, 1] = -45.0
    tie_median_matrix[2, 3] = -39.0
    tie_median_matrix[3, :3] = [-43.0, -44.0, -39.0]
    median_choice = select_cluster_representative((0, 1, 2, 3), tie_median_matrix)
    median_gate = median_choice.representative_state_id == 1

    tie_mean_matrix = np.full((5, 5), -np.inf, dtype=np.float64)
    tie_mean_matrix[0, 1:] = [-40.0, -42.0, -43.0, -45.0]
    tie_mean_matrix[1, 0] = -40.0
    tie_mean_matrix[1, 2:] = [-42.0, -43.0, -47.0]
    tie_mean_matrix[2, 0] = -42.0
    tie_mean_matrix[2, 1] = -42.0
    tie_mean_matrix[2, 3:] = [-39.0, -39.0]
    tie_mean_matrix[3, 0] = -43.0
    tie_mean_matrix[3, 1] = -43.0
    tie_mean_matrix[3, 2] = -39.0
    tie_mean_matrix[3, 4] = -39.0
    tie_mean_matrix[4, :4] = [-45.0, -47.0, -39.0, -39.0]
    mean_choice = select_cluster_representative((0, 1, 2, 3, 4), tie_mean_matrix)
    mean_gate = mean_choice.representative_state_id == 1

    tie_state_matrix = np.full((2, 2), -np.inf, dtype=np.float64)
    tie_state_matrix[0, 1] = tie_state_matrix[1, 0] = -43.0
    state_choice = select_cluster_representative((0, 1), tie_state_matrix)
    state_gate = state_choice.representative_state_id == 0

    singleton_matrix = np.full((38, 38), -50.0, dtype=np.float64)
    np.fill_diagonal(singleton_matrix, -np.inf)
    singleton = select_cluster_representative((37,), singleton_matrix, cluster_id=0)
    singleton_gate = singleton.representative_state_id == 37 and np.isneginf(singleton.worst_cnmse_db)
    return {
        "minimax_gate": minimax_gate,
        "median_tie_gate": median_gate,
        "mean_tie_gate": mean_gate,
        "state_id_tie_gate": state_gate,
        "singleton_gate": singleton_gate,
    }


def _member_sets(assignments: pd.DataFrame) -> list[tuple[int, ...]]:
    groups = [tuple(sorted(group["state_id"].astype(int).tolist())) for _, group in assignments.groupby("cluster_id", sort=True)]
    return sorted(groups, key=lambda members: (members[0], members))


def _state_metadata(state_frame: pd.DataFrame, state_id: int) -> dict[str, Any]:
    row = state_frame.loc[state_frame["state_id"] == int(state_id)]
    if row.empty:
        raise ValueError(f"state metadata missing for state_id={state_id}")
    item = row.iloc[0]
    return {"funMng": int(item["funMng"]), "funAng": int(item["funAng"]), "secMng": int(item["secMng"]), "secAng": int(item["secAng"]), "v_carrier": float(item["v_carrier"]), "inputPower": float(item["inputPower"])}


def _write_updated_results(
    result: Any,
    matrix: np.ndarray,
    state_frame: pd.DataFrame,
    matrix_metadata: dict[str, Any],
    threshold: float,
    matrix_action: str,
    reproduction: dict[str, Any],
    synthetic: dict[str, bool],
) -> dict[str, Any]:
    representatives = tuple(result.representatives)
    summary_rows = summarize_clusters(matrix, result.clusters)
    summary_frame = pd.DataFrame(summary_rows).sort_values("cluster_id").reset_index(drop=True)
    for column in ("representative_state_id", "representative_fingerprint_state_id", "representative_dpd_state_id", "representative_worst_cnmse_db", "representative_median_cnmse_db", "representative_mean_cnmse_db"):
        summary_frame[column] = np.nan
    for item in representatives:
        mask = summary_frame["cluster_id"] == item.cluster_id
        summary_frame.loc[mask, "representative_state_id"] = item.representative_state_id
        summary_frame.loc[mask, "representative_fingerprint_state_id"] = item.representative_state_id
        summary_frame.loc[mask, "representative_dpd_state_id"] = item.representative_state_id
        summary_frame.loc[mask, "representative_worst_cnmse_db"] = item.worst_cnmse_db
        summary_frame.loc[mask, "representative_median_cnmse_db"] = item.median_cnmse_db
        summary_frame.loc[mask, "representative_mean_cnmse_db"] = item.mean_cnmse_db
    for column in ("representative_state_id", "representative_fingerprint_state_id", "representative_dpd_state_id"):
        summary_frame[column] = summary_frame[column].astype("Int64")

    assignment_rows: list[dict[str, Any]] = []
    for state_id in range(STATE_COUNT):
        cluster_id = int(result.state_to_cluster[state_id])
        representative_state_id = int(result.representative_state_ids[cluster_id])
        assignment_rows.append({
            "state_id": state_id,
            "cluster_id": cluster_id,
            "cluster_size": len(result.clusters[cluster_id]),
            **_state_metadata(state_frame, state_id),
            "representative_state_id": representative_state_id,
            "representative_fingerprint_state_id": representative_state_id,
            "representative_dpd_state_id": representative_state_id,
            "is_representative": int(state_id == representative_state_id),
        })
    assignment_frame = pd.DataFrame(assignment_rows).sort_values("state_id").reset_index(drop=True)

    representative_rows: list[dict[str, Any]] = []
    for item in representatives:
        summary_row = summary_frame.loc[summary_frame["cluster_id"] == item.cluster_id].iloc[0]
        representative_rows.append({
            "cluster_id": item.cluster_id,
            "cluster_size": item.cluster_size,
            "representative_state_id": item.representative_state_id,
            "representative_fingerprint_state_id": item.representative_state_id,
            "representative_dpd_state_id": item.representative_state_id,
            "representative_worst_cnmse_db": item.worst_cnmse_db,
            "representative_median_cnmse_db": item.median_cnmse_db,
            "representative_mean_cnmse_db": item.mean_cnmse_db,
            "cluster_worst_pair_cnmse_db": float(summary_row["worst_pair_cnmse_db"]),
            **_state_metadata(state_frame, item.representative_state_id),
            "member_state_ids": summary_row["member_state_ids"],
        })
    representative_frame = pd.DataFrame(representative_rows).sort_values("cluster_id").reset_index(drop=True)
    representative_ids_equal = bool(
        (
            representative_frame["representative_state_id"]
            == representative_frame["representative_fingerprint_state_id"]
        ).all()
        and (
            representative_frame["representative_state_id"]
            == representative_frame["representative_dpd_state_id"]
        ).all()
    )

    THRESHOLD_DIR.mkdir(parents=True, exist_ok=True)
    assignment_frame.to_csv(THRESHOLD_DIR / "cluster_assignments.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    summary_frame.to_csv(THRESHOLD_DIR / "cluster_summary.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    representative_frame.to_csv(THRESHOLD_DIR / "cluster_representatives.csv", index=False, encoding="utf-8-sig", float_format="%.17g")

    non_singleton = summary_frame.loc[summary_frame["pair_count"] > 0]
    global_best = float(non_singleton["best_pair_cnmse_db"].min()) if not non_singleton.empty else float("-inf")
    global_worst = float(summary_frame["worst_pair_cnmse_db"].max()) if not summary_frame.empty else float("-inf")
    top_representatives = representative_frame.head(10)
    summary_lines = [
        f"Experiment: {TASK_NAME}",
        "Scenario: 2",
        f"State count: {STATE_COUNT}",
        "Signal used to form clusters: 5B A-segment yout_withoutdpd_ori after full-record xin-referenced alignment, fixed A split, and A-only complex-gain adjustment.",
        f"CNMSE threshold: {threshold:.17g} dB",
        "Threshold rule: strict D_ij < threshold",
        f"Pairwise matrix: {STATE_COUNT} x {STATE_COUNT}; action={matrix_action}; symmetry={matrix_metadata.get('cnmse_symmetry_mode', 'unknown')}",
        "Clustering: deterministic global complete-link threshold clustering",
        f"-40 dB member-set reproduction: {reproduction['pass']}",
        f"Cluster count: {len(result.clusters)}",
        f"Largest cluster size: {max((len(cluster) for cluster in result.clusters), default=0)}",
        f"Median cluster size: {float(np.median([len(cluster) for cluster in result.clusters])) if result.clusters else 0.0:.6g}",
        f"Singleton cluster count: {int((summary_frame['cluster_size'] == 1).sum()) if not summary_frame.empty else 0}",
        f"Global worst within-cluster CNMSE: {global_worst:.9f} dB",
        f"Global best non-singleton within-cluster CNMSE: {global_best:.9f} dB",
        f"All within-cluster pairs satisfy threshold: {reproduction['new_validation']['within_cluster_pass']}",
        f"All states assigned exactly once: {reproduction['new_validation']['coverage_pass']}",
        f"Any two final clusters still mergeable: {not reproduction['new_validation']['non_mergeability_pass']}",
        "",
        "Representative selection",
        "Method: minimax medoid",
        "Primary criterion: minimum worst CNMSE to other members",
        "Tie-breaking: worst -> median -> mean -> state_id, ascending",
        f"Representative count: {len(representatives)}",
        f"One representative per cluster: {len(representatives) == len(result.clusters)}",
        f"Representative belongs to own cluster: {all(item.representative_state_id in result.clusters[item.cluster_id] for item in representatives)}",
        f"representative_state_id == representative_fingerprint_state_id == representative_dpd_state_id: {representative_ids_equal}",
        "Only state ID references are exported; fingerprint and DPD payloads are not copied or loaded.",
        "",
        "Representative preview (first 10 clusters by cluster_id)",
        "cluster_id | size | representative_state_id | representative_worst_cnmse_db",
    ]
    for _, row in top_representatives.iterrows():
        summary_lines.append(f"{int(row['cluster_id'])} | {int(row['cluster_size'])} | {int(row['representative_state_id'])} | {float(row['representative_worst_cnmse_db']):.9f}")
    summary_lines.extend([
        "",
        "Gates",
        f"Synthetic minimax/median/mean/state-id/singleton gates: {all(synthetic.values())}",
        f"All clustering gates: {reproduction['new_validation']['all_gates_pass']}",
        f"Raw/protected historical results unchanged: {reproduction['protection']['all_protected_unchanged']}",
        "No representative DPD validation, LUT retrieval, Real-B shareability, or behavior-model fitting was performed.",
    ])
    (THRESHOLD_DIR / "final_result_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {"assignment_frame": assignment_frame, "summary_frame": summary_frame, "representative_frame": representative_frame, "global_best": global_best, "global_worst": global_worst}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnmse-threshold-db", type=float, default=DEFAULT_THRESHOLD, dest="threshold")
    parser.add_argument("--force-recompute-matrix", action="store_true")
    return parser.parse_args()


def main() -> None:
    freeze_support()
    args = _parse_args()
    if not np.isfinite(args.threshold):
        raise ValueError("--cnmse-threshold-db must be finite")
    if abs(float(args.threshold) - DEFAULT_THRESHOLD) > 1e-12:
        raise ValueError("This maintenance/reproduction pass is intentionally frozen to -40 dB")
    synthetic = _synthetic_representative_gates()
    if not all(synthetic.values()):
        raise RuntimeError(f"representative selection synthetic gate failed: {synthetic}")
    if not MATRIX_PATH.is_file():
        raise FileNotFoundError(f"pairwise matrix missing: {MATRIX_PATH}")
    if not OLD_ASSIGNMENTS.is_file() or not OLD_SUMMARY.is_file():
        raise FileNotFoundError("validated -40 dB clustering outputs are required for reproduction")
    before = legacy_runner._uniform_snapshot()
    logical_cpu_count = int(os.cpu_count() or 1)
    worker_count = max(1, int(math.floor(legacy_runner.CPU_TARGET * logical_cpu_count)))
    matrix_exists_before = MATRIX_PATH.is_file()
    matrix, state_frame, matrix_metadata, _diagnostics = legacy_runner._load_or_build_matrix(MATRIX_PATH, worker_count, args.force_recompute_matrix)
    matrix_action = "recomputed" if args.force_recompute_matrix or not matrix_exists_before else "reused"
    result = cluster_behavior_states(matrix, float(args.threshold))
    old_assignments = pd.read_csv(OLD_ASSIGNMENTS)
    old_sets = _member_sets(old_assignments)
    new_sets = sorted(result.clusters, key=lambda members: (members[0], members))
    old_summary = pd.read_csv(OLD_SUMMARY)
    new_summary = pd.DataFrame(summarize_clusters(matrix, result.clusters))
    old_worst = float(old_summary["worst_pair_cnmse_db"].max())
    new_worst = float(new_summary["worst_pair_cnmse_db"].max())
    new_validation = legacy_runner.validate_clusters(matrix, result.clusters, float(args.threshold))
    reproduction = {
        "pass": bool(old_sets == new_sets and len(old_sets) == 77 and max(map(len, new_sets)) == 12 and float(np.median([len(item) for item in new_sets])) == 6.0 and sum(len(item) == 1 for item in new_sets) == 4 and abs(old_worst - new_worst) <= REPRO_TOLERANCE_DB),
        "old_cluster_count": len(old_sets),
        "new_cluster_count": len(new_sets),
        "member_sets_equal": old_sets == new_sets,
        "old_global_worst": old_worst,
        "new_global_worst": new_worst,
        "new_validation": new_validation,
    }
    if not reproduction["pass"]:
        raise RuntimeError(f"-40 dB clustering reproduction failed: {reproduction}")
    if len(result.representatives) != len(result.clusters):
        raise RuntimeError("representative count does not equal cluster count")
    if not all(item.representative_state_id in result.clusters[item.cluster_id] for item in result.representatives):
        raise RuntimeError("representative is not a member of its cluster")
    if not all(item.representative_state_id == int(result.representative_state_ids[item.cluster_id]) for item in result.representatives):
        raise RuntimeError("representative state map mismatch")
    after_reproduction = legacy_runner._uniform_snapshot()
    reproduction["protection"] = legacy_runner._uniform_protection_check(before, after_reproduction)
    if not reproduction["protection"]["all_protected_unchanged"]:
        raise RuntimeError("raw or protected historical result changed before representative output")
    _write_updated_results(result, matrix, state_frame, matrix_metadata, float(args.threshold), matrix_action, reproduction, synthetic)
    after = legacy_runner._uniform_snapshot()
    protection = legacy_runner._uniform_protection_check(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw or protected historical result changed")
    timestamp = datetime.now(UTC).isoformat()
    log = (
        f"{TASK_NAME} representative-selection maintenance completed: threshold=-40 dB, clusters={len(result.clusters)}, representatives={len(result.representatives)}, "
        f"matrix={matrix_action}, reproduction={reproduction['pass']}, all_cluster_gates={new_validation['all_gates_pass']}, "
        f"representative_gates={all(synthetic.values())}, raw/protected unchanged={protection['all_protected_unchanged']}. "
        f"Updated cluster assignments/summary and added cluster_representatives.csv; no fingerprint/DPD payload copied."
    )
    WORK_LOG = PROJECT_ROOT / "work_logs" / "clustering" / "execution_log.txt"
    HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WORK_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] {TASK_NAME} representative selection\n{log}\n")
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n{timestamp} | clustering 持续维护：Complete-Link 代表状态选择\n{log}\n")
    print("Pairwise matrix: reused" if matrix_action == "reused" else "Pairwise matrix: recomputed", flush=True)
    print("CNMSE threshold: -40 dB", flush=True)
    print(f"Clustering: {len(result.clusters)} clusters", flush=True)
    print(f"Representative selection: minimax_medoid; {len(result.representatives)}/{len(result.clusters)} selected", flush=True)
    print(f"-40 dB reproduction: {reproduction['pass']}", flush=True)
    print(f"Result path: {THRESHOLD_DIR}", flush=True)


if __name__ == "__main__":
    main()
