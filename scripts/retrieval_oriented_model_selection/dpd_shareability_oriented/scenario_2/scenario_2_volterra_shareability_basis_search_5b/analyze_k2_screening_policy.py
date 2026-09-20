"""Offline K2 screening-policy analysis from completed exact/screening artifacts.

This module deliberately performs no model fitting, exact evaluation, worker
launch, or screening recomputation. It only joins the completed 38,334-row
ground truth with the completed screening partitions.
"""

# Exact analysis field names and report payloads are intentionally verbose.
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from behavior_modeling.shared.volterra_terms import VolterraTermSpec
from scipy.stats import kendalltau, spearmanr

from retrieval_oriented_model_selection.shared.shareability_persistence import (
    load_screening_artifact,
)
from retrieval_oriented_model_selection.shared.shareability_screening import (
    ParentScreeningScores,
)
from retrieval_oriented_model_selection.shared.shareability_types import CandidateScreeningSpec

from .run_search import validated_science_fingerprint, validated_search_policy_fingerprint
from .task_config import DICTIONARY, LOG_ROOT, SCREENING, SEED_TERM

ANALYSIS_ROOT = LOG_ROOT / "pre_search_validation" / "screening_policy_analysis"
EXACT_TRACE = LOG_ROOT / "pre_search_validation" / "k2_exhaustive_results.jsonl"
PARTITION_ROOT = LOG_ROOT / "pre_search_validation" / "screening_partitions"
QUALITY_JSON = LOG_ROOT / "optimization" / "quality-validation_k2_n1000_w10_b64_ea02db99bd11.json"

SHORTLIST_SIZES = (
    500,
    750,
    1000,
    1500,
    2000,
    2500,
    3000,
    4000,
    5000,
    6000,
    7500,
    10000,
    15000,
    20000,
)
MIXED_SIZES = (500, 1000, 2000, 3000, 5000, 7500, 10000)
EXACT_TOP_KS = (1, 3, 10, 20, 50, 100)
EXPLORATION_RATIOS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_exact() -> list[dict[str, Any]]:
    if not EXACT_TRACE.is_file():
        raise FileNotFoundError(EXACT_TRACE)
    rows = [
        json.loads(line)
        for line in EXACT_TRACE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 38334:
        raise RuntimeError(f"exact row count mismatch: {len(rows)}")
    basis_ids = [str(row["basis_id"]) for row in rows]
    if len(set(basis_ids)) != len(basis_ids) or SEED_TERM.basis_id in basis_ids:
        raise RuntimeError("exact candidate IDs are duplicated or contain the fixed seed")
    required = {"N_shareable", "N_self", "N_top3", "N_top5", "N_top10"}
    if any(not required <= set(row) for row in rows):
        raise RuntimeError("exact trace is missing a required score field")
    return rows


def _load_screening() -> tuple[tuple[str, ...], dict[str, Any]]:
    rows: list[tuple[str, float, float, float, str]] = []
    metadata_rows = sorted(PARTITION_ROOT.glob("partition_*.json"))
    if len(metadata_rows) != 40:
        raise RuntimeError(f"screening partition metadata count mismatch: {len(metadata_rows)}")
    exact_fingerprint = validated_science_fingerprint()["sha256"]
    policy_fingerprint = validated_search_policy_fingerprint("screened")[
        "search_policy_fingerprint"
    ]
    for metadata_path in metadata_rows:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["metadata"]
        index = int(metadata["partition_index"])
        if metadata["exact_evaluation_fingerprint"] != exact_fingerprint:
            raise RuntimeError(f"screening exact fingerprint mismatch in partition {index}")
        if metadata["search_policy_fingerprint"] != policy_fingerprint:
            raise RuntimeError(f"screening policy fingerprint mismatch in partition {index}")
        arrays = load_screening_artifact(
            PARTITION_ROOT,
            f"partition_{index:04d}",
            {
                "stage": "screening_k2_validation",
                "partition_index": index,
                "candidate_pool_hash": metadata["candidate_pool_hash"],
                "exact_evaluation_fingerprint": exact_fingerprint,
                "search_policy_fingerprint": policy_fingerprint,
                "parent_support_id": metadata["parent_support_id"],
            },
        )
        rows.extend(
            (str(basis), float(score), float(aend), float(c2), str(stratum))
            for basis, score, aend, c2, stratum in zip(
                arrays["basis_ids"],
                arrays["scores"],
                arrays["aend_scores"],
                arrays["c2_scores"],
                arrays["strata"],
                strict=True,
            )
        )
    if len(rows) != 38334 or len({row[0] for row in rows}) != 38334:
        raise RuntimeError(f"screening row count/uniqueness mismatch: {len(rows)}")
    if any(
        not np.isfinite(row[1]) or not np.isfinite(row[2]) or not np.isfinite(row[3])
        for row in rows
    ):
        raise RuntimeError("screening contains NaN or Inf")
    rows.sort(key=lambda row: row[0])
    return tuple(row[0] for row in rows), {
        "scores": np.asarray([row[1] for row in rows], dtype=np.float64),
        "aend_scores": np.asarray([row[2] for row in rows], dtype=np.float64),
        "c2_scores": np.asarray([row[3] for row in rows], dtype=np.float64),
        "strata": tuple(row[4] for row in rows),
    }


def _spec(total: int, explore: int) -> CandidateScreeningSpec:
    return CandidateScreeningSpec(
        shortlist_size=total,
        exploit_size=total - explore,
        explore_size=explore,
        enabled=True,
        method=SCREENING.method,
        aend_weight=SCREENING.aend_weight,
        c2_weight=SCREENING.c2_weight,
        state_aggregation=SCREENING.state_aggregation,
        deterministic=True,
        stratification_version=SCREENING.stratification_version,
    )


def _recall(chosen: set[str], exact_ranked: tuple[str, ...]) -> dict[str, int | bool]:
    return {
        f"top{k}": (
            exact_ranked[0] in chosen
            if k == 1
            else int(sum(item in chosen for item in exact_ranked[:k]))
        )
        for k in EXACT_TOP_KS
    }


def _rank_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def _descriptor(term: VolterraTermSpec) -> dict[str, object]:
    return {
        "basis_id": term.basis_id,
        "nonlinear_order": term.nonlinear_order,
        "nonconjugate_delays": list(term.nonconjugate_delays),
        "conjugate_delays": list(term.conjugate_delays),
        "interaction_type": "diagonal" if term.is_diagonal else "cross_memory",
        "max_delay": term.max_delay,
    }


def run_analysis() -> dict[str, object]:
    exact_rows = _load_exact()
    exact_by_id = {str(row["basis_id"]): row for row in exact_rows}
    exact_ranked = tuple(
        row["basis_id"]
        for row in sorted(exact_rows, key=lambda row: (-int(row["N_shareable"]), row["basis_id"]))
    )
    screening_ids, screening_arrays = _load_screening()
    exact_ids = set(exact_by_id)
    if set(screening_ids) != exact_ids or len(exact_ids) != 38334:
        raise RuntimeError("exact/screening candidate join mismatch")
    screen_record = ParentScreeningScores(
        (SEED_TERM.basis_id,),
        screening_ids,
        screening_arrays["scores"],
        screening_arrays["aend_scores"],
        screening_arrays["c2_scores"],
        screening_arrays["strata"],
        {},
    )
    current = screen_record.shortlist(SCREENING, DICTIONARY)
    pure_top500 = screen_record.shortlist(_spec(500, 0), DICTIONARY)
    screen_rank = {
        entry.basis_id: entry.rank for entry in screen_record.shortlist(_spec(38334, 0), DICTIONARY)
    }
    exact_rank = {basis: index + 1 for index, basis in enumerate(exact_ranked)}
    current_source = {entry.basis_id: entry.proposal_source for entry in current}
    master = {
        "basis_id": np.asarray(screening_ids),
        "exact_N_shareable": np.asarray(
            [exact_by_id[b]["N_shareable"] for b in screening_ids], dtype=np.int64
        ),
        "exact_N_self": np.asarray(
            [exact_by_id[b]["N_self"] for b in screening_ids], dtype=np.int64
        ),
        "exact_N_top3": np.asarray(
            [exact_by_id[b]["N_top3"] for b in screening_ids], dtype=np.int64
        ),
        "exact_N_top5": np.asarray(
            [exact_by_id[b]["N_top5"] for b in screening_ids], dtype=np.int64
        ),
        "exact_N_top10": np.asarray(
            [exact_by_id[b]["N_top10"] for b in screening_ids], dtype=np.int64
        ),
        "exact_rank": np.asarray([exact_rank[b] for b in screening_ids], dtype=np.int64),
        "somp_score": screening_arrays["scores"],
        "somp_aend_score": screening_arrays["aend_scores"],
        "somp_c2_score": screening_arrays["c2_scores"],
        "somp_rank": np.asarray([screen_rank[b] for b in screening_ids], dtype=np.int64),
        "current_400plus100_selected": np.asarray(
            [b in current_source for b in screening_ids], dtype=bool
        ),
        "current_proposal_source": np.asarray(
            [current_source.get(b, "not_selected") for b in screening_ids]
        ),
        "stratum": np.asarray(screening_arrays["strata"]),
        "nonlinear_order": np.asarray(
            [DICTIONARY[b].nonlinear_order for b in screening_ids], dtype=np.int64
        ),
        "interaction_type": np.asarray(
            ["diagonal" if DICTIONARY[b].is_diagonal else "cross_memory" for b in screening_ids]
        ),
        "max_delay": np.asarray([DICTIONARY[b].max_delay for b in screening_ids], dtype=np.int64),
    }
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ANALYSIS_ROOT / "k2_screening_policy_master.npz", **master)
    current_recall = _recall(set(entry.basis_id for entry in current), exact_ranked)
    pure_recall = _recall(set(entry.basis_id for entry in pure_top500), exact_ranked)
    expected_current = {"top1": True, "top3": 2, "top10": 4, "top50": 10, "top100": 19}
    if any(current_recall[key] != value for key, value in expected_current.items()):
        raise RuntimeError(f"current 400+100 reproduction mismatch: {current_recall}")
    expected_pure = {"top1": False, "top3": 0, "top10": 1, "top50": 2, "top100": 5}
    if any(pure_recall[key] != value for key, value in expected_pure.items()):
        raise RuntimeError(f"pure Top500 reproduction mismatch: {pure_recall}")

    pure_sweep = []
    for total in SHORTLIST_SIZES:
        chosen = {entry.basis_id for entry in screen_record.shortlist(_spec(total, 0), DICTIONARY)}
        pure_sweep.append({"shortlist": total, **_recall(chosen, exact_ranked)})
    minimum_shortlist = {
        str(k): max(screen_rank[basis] for basis in exact_ranked[:k]) for k in EXACT_TOP_KS
    }
    mixed_grid = []
    for total in MIXED_SIZES:
        for ratio in EXPLORATION_RATIOS:
            explore = int(round(ratio * total))
            chosen_entries = screen_record.shortlist(_spec(total, explore), DICTIONARY)
            chosen = {entry.basis_id for entry in chosen_entries}
            exact_good = set(exact_ranked[:100])
            exploit_set = {
                entry.basis_id for entry in screen_record.shortlist(_spec(total, 0), DICTIONARY)
            }
            mixed_grid.append(
                {
                    "shortlist": total,
                    "exploration_ratio": ratio,
                    "exploit": total - explore,
                    "exploration": explore,
                    **_recall(chosen, exact_ranked),
                    "exploration_added_exact_top100": len((chosen - exploit_set) & exact_good),
                    "exploit_exact_top100_displaced": len((exploit_set - chosen) & exact_good),
                }
            )
    exact_rank_array = np.asarray([exact_rank[b] for b in screening_ids], dtype=np.float64)
    screen_rank_array = np.asarray([screen_rank[b] for b in screening_ids], dtype=np.float64)
    exact_score_array = np.asarray(
        [exact_by_id[b]["N_shareable"] for b in screening_ids], dtype=np.float64
    )
    alignment = {
        "spearman_screen_rank_vs_exact_rank": float(
            spearmanr(screen_rank_array, exact_rank_array).statistic
        ),
        "spearman_somp_score_vs_exact_N_shareable": float(
            spearmanr(screening_arrays["scores"], exact_score_array).statistic
        ),
        "kendall_screen_rank_vs_exact_rank": float(
            kendalltau(screen_rank_array, exact_rank_array).statistic
        ),
        "rank_displacement_exact_top100": _rank_stats(
            np.asarray(
                [screen_rank[b] - exact_rank[b] for b in exact_ranked[:100]], dtype=np.float64
            )
        ),
        "rank_displacement_exact_top10": _rank_stats(
            np.asarray(
                [screen_rank[b] - exact_rank[b] for b in exact_ranked[:10]], dtype=np.float64
            )
        ),
    }
    strata = {}
    for order in (1, 3, 5, 7, 9, 11):
        for interaction in ("diagonal", "cross_memory"):
            for delay in range(5):
                selected = [
                    basis
                    for basis in exact_ranked[:100]
                    if DICTIONARY[basis].nonlinear_order == order
                    and ("diagonal" if DICTIONARY[basis].is_diagonal else "cross_memory")
                    == interaction
                    and DICTIONARY[basis].max_delay == delay
                ]
                if selected:
                    key = f"p{order}|{interaction}|d{delay}"
                    strata[key] = {
                        "exact_top100_count": len(selected),
                        "median_somp_rank": float(np.median([screen_rank[b] for b in selected])),
                        "min_somp_rank": min(screen_rank[b] for b in selected),
                        "max_somp_rank": max(screen_rank[b] for b in selected),
                    }
    measured_exact_rate = 21.221
    measured_full_rate = 18.834657300639634
    runtime_tradeoff = []
    for total in SHORTLIST_SIZES:
        rate = measured_full_rate if total == 38334 else measured_exact_rate
        runtime_tradeoff.append(
            {
                "shortlist": total,
                "exact_evaluation_time_seconds_estimated": total / rate,
                "source": "measured_full_k2_rate"
                if total == 38334
                else "linear_estimate_from_measured_1000_candidate_exact_rate",
                "compression_vs_full_exact": 38334 / total,
            }
        )
    recommendation = {
        "status": "not_frozen",
        "current_500_policy": "rejected by exact Top3/Top10 recall gate",
        "pure_somp_minimum_shortlist_for_full_recall": minimum_shortlist,
        "interpretation": "Use the completed offline table to choose between a widened SOMP shortlist and a second proxy; do not modify task_config or start formal search in this analysis pass.",
        "current_data_driven_observation": "The exact leaders are not reliably ordered by SOMP; policy scans and minimum ranks must be reviewed before a new policy is frozen.",
    }
    _json_write(
        ANALYSIS_ROOT / "01_master_table_validation.json",
        {
            "rows": len(screening_ids),
            "unique_basis_id": len(set(screening_ids)),
            "missing_exact": 0,
            "missing_somp": 0,
            "nan_somp": 0,
            "inf_somp": 0,
            "seed_in_candidate_pool": SEED_TERM.basis_id in set(screening_ids),
            "exact_fingerprint": validated_science_fingerprint()["sha256"],
            "screening_policy_fingerprint": validated_search_policy_fingerprint("screened")[
                "search_policy_fingerprint"
            ],
        },
    )
    _json_write(ANALYSIS_ROOT / "02_pure_somp_shortlist_sweep.json", {"sweep": pure_sweep})
    _json_write(
        ANALYSIS_ROOT / "03_minimum_shortlist_for_full_recall.json",
        {"minimum_somp_rank": minimum_shortlist},
    )
    _json_write(ANALYSIS_ROOT / "04_mixed_policy_grid.json", {"grid": mixed_grid})
    _json_write(ANALYSIS_ROOT / "05_rank_alignment_analysis.json", alignment)
    _json_write(ANALYSIS_ROOT / "06_stratum_bias_analysis.json", {"exact_top100_strata": strata})
    _json_write(ANALYSIS_ROOT / "07_runtime_tradeoff.json", {"tradeoff": runtime_tradeoff})
    _json_write(
        ANALYSIS_ROOT / "08_policy_pareto_frontier.json",
        {
            "note": "Policy frontier is retained as full grid for user review; no policy was auto-frozen.",
            "grid": mixed_grid,
        },
    )
    _json_write(ANALYSIS_ROOT / "09_screening_policy_recommendation.json", recommendation)
    return {
        "master_rows": len(screening_ids),
        "current_recall": current_recall,
        "pure_top500_recall": pure_recall,
        "minimum_shortlist": minimum_shortlist,
        "alignment": alignment,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    print(json.dumps(run_analysis(), ensure_ascii=False, indent=2))
