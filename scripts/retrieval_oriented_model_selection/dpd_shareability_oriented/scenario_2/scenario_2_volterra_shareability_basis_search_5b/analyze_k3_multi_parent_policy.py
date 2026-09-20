"""Offline K3 exact ranking/provenance analysis; no model or worker execution."""

# Report field names are intentionally verbose.
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

from .validate_k3_multi_parent_policy import K3_ROOT, _load_k2_parents


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run() -> dict[str, object]:
    manifest = json.loads((K3_ROOT / "04_k3_candidate_pool_manifest.json").read_text(encoding="utf-8"))
    trace_path = K3_ROOT / "k3_exhaustive_results.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 114996 or len({row["support_id"] for row in rows}) != 114996:
        raise RuntimeError("K3 exact trace is incomplete or duplicated")
    by_id = {row["support_id"]: row for row in rows}
    parents = _load_k2_parents()
    provenance = manifest["provenance"]
    sid_to_parents: dict[str, set[int]] = {
        support_id: {int(parent_index)}
        for support_id, parent_index in zip(
            manifest["support_ids"], manifest["primary_parent_index"], strict=True
        )
    }
    for support_id, origins in provenance.items():
        sid_to_parents.setdefault(support_id, set()).update(
            int(origin["parent_index"]) for origin in origins
        )
    per_parent: dict[str, object] = {}
    for parent_index, parent in enumerate(parents):
        support_ids = [
            support_id for support_id, parent_indices in sid_to_parents.items()
            if parent_index in parent_indices
        ]
        ranked = sorted(
            (by_id[support_id] for support_id in support_ids),
            key=lambda row: (-int(row["N_shareable"]), row["support_id"]),
        )
        per_parent[str(parent_index)] = {
            "parent_support": list(parent),
            "child_count": len(ranked),
            "top1": ranked[:1],
            "top3": ranked[:3],
            "top10": ranked[:10],
            "top20": ranked[:20],
            "top50": ranked[:50],
        }
    global_ranked = sorted(rows, key=lambda row: (-int(row["N_shareable"]), row["support_id"]))
    global_payload = {
        "unique_supports": len(global_ranked),
        "top3": global_ranked[:3],
        "top10": global_ranked[:10],
        "top20": global_ranked[:20],
        "top50": global_ranked[:50],
        "boundary_N_shareable": int(global_ranked[2]["N_shareable"]),
        "boundary_tie_count": sum(int(row["N_shareable"]) == int(global_ranked[2]["N_shareable"]) for row in global_ranked),
    }
    summary = {
        "stage": "k3_exact_ranking_analysis",
        "parents": [list(parent) for parent in parents],
        "raw_proposals": manifest["raw_proposals"],
        "unique_supports": manifest["unique_supports"],
        "per_parent": per_parent,
        "global": global_payload,
        "next": "screening_policy_recall_only; no evaluator rerun",
    }
    _write(K3_ROOT / "05_k3_exhaustive_exact_summary.json", {"unique_supports": len(rows), "trace": str(trace_path), "cache_expected": len(rows)})
    _write(K3_ROOT / "06_k3_per_parent_exact_rankings.json", {"per_parent": per_parent})
    _write(K3_ROOT / "07_k3_global_exact_ranking.json", global_payload)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
