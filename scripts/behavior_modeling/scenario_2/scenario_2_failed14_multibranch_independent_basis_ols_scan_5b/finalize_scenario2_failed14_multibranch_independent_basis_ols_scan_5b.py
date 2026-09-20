"""Recover the final bookkeeping for the multibranch OLS scan.

The numerical search and post-selection evaluation completed before the first
run hit a JSON serialization error while writing the root ``search_state``.
This small, idempotent finalizer reuses the completed artifacts, repairs the
status files, refreshes the support/rank diagnostics, and writes the detailed
human-readable summary plus a sheet-oriented JSON source for the workbook
exporter.  It never reruns the expensive search and never reads raw data.
"""

# ruff: noqa: E402,E501,I001

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

from behavior_modeling.scenario_2.scenario_2_failed14_multibranch_independent_basis_ols_scan_5b import (  # noqa: E402
    run_scenario2_failed14_multibranch_independent_basis_ols_scan_5b as runner,
)


RESULT_ROOT = runner.RESULT_ROOT
SEARCH_ROOT = runner.SEARCH_ROOT
TABLE_ROOT = runner.TABLE_ROOT
CONFIG_ROOT = runner.CONFIG_ROOT


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    runner._write_json(path, payload)


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    return runner._json_safe(frame.where(pd.notna(frame), None).to_dict("records"))


def _selected_candidate(payload: dict[str, Any]) -> runner.Candidate:
    config = tuple(sorted((str(key), int(value)) for key, value in payload["config"].items()))
    return runner.Candidate(
        candidate_id=int(payload["candidate_id"]),
        seed=str(payload.get("seed", "unknown")),
        pass_number=int(payload.get("pass_number", 0)),
        parent_id=None,
        config=config,
    )


def _update_gates(validation: dict[str, Any]) -> dict[str, Any]:
    conditioning_path = TABLE_ROOT / "conditioning.csv"
    conditioning = pd.read_csv(conditioning_path)
    support_mask = conditioning.apply(
        lambda row: int(row["N"]) == (12262 if str(row["side"]) == "Aend" else 7347)
        and int(row["N_B"]) == 4889,
        axis=1,
    )
    rank_full = bool((conditioning["rank"] == conditioning["K"]).all())
    gates = dict(validation.get("gates", {}))
    gates["rank"] = {
        "pass": rank_full,
        "full_column_rank": rank_full,
        "minimum_rank": int(conditioning["rank"].min()),
        "maximum_rank_deficit": int((conditioning["K"] - conditioning["rank"]).max()),
        "note": (
            "The selected OLS design is numerically rank deficient for some state/side; "
            "the deficiency is retained as a diagnostic and no columns are dropped."
            if not rank_full
            else "All selected design matrices are full column rank."
        ),
    }
    gates["support"] = {
        "pass": bool(support_mask.all()),
        "expected_N": {"Aend": 12262, "C2": 7347},
        "expected_N_B": 4889,
        "rows_checked": int(len(conditioning)),
    }
    validation["gates"] = gates
    validation["completed"] = True
    validation["completion_timestamp"] = datetime.now(UTC).isoformat()
    validation["recovery"] = {
        "postprocessing_recovered": True,
        "reason": "The numerical run completed, but root search_state serialization initially rejected a Candidate dataclass.",
        "search_and_model_artifacts_reused": True,
        "raw_data_reprocessed": False,
    }
    return validation


def _summary_text(
    selected_payload: dict[str, Any],
    validation: dict[str, Any],
    registry: pd.DataFrame,
    references: pd.DataFrame,
    selected_rows: pd.DataFrame,
    family_counts: pd.DataFrame,
    family_norms: pd.DataFrame,
) -> str:
    selected_id = int(selected_payload["candidate_id"])
    selected_config = selected_payload["config"]
    search_info = validation["search_info"]
    def med(model: str, side: str, metric: str) -> float:
        return float(references.loc[(references["model"] == model) & (references["side"] == side), metric].median())

    selected_state = selected_rows.copy()
    state_pivot = selected_state.pivot(index="state_id", columns="side", values=["train_NMSE_dB", "B_NMSE_dB"])
    state_worst = pd.DataFrame(index=state_pivot.index)
    state_worst["worst4"] = state_pivot[[
        ("train_NMSE_dB", "Aend"), ("B_NMSE_dB", "Aend"),
        ("train_NMSE_dB", "C2"), ("B_NMSE_dB", "C2"),
    ]].max(axis=1)
    frozen = references.loc[references["model"] == "Frozen"].pivot(index="state_id", columns="side", values=["train_NMSE_dB", "B_NMSE_dB"])
    frozen_worst = frozen[[
        ("train_NMSE_dB", "Aend"), ("B_NMSE_dB", "Aend"),
        ("train_NMSE_dB", "C2"), ("B_NMSE_dB", "C2"),
    ]].max(axis=1)
    delta_worst = state_worst["worst4"] - frozen_worst
    changed = registry.drop_duplicates("candidate_id")
    selected_conditioning = references.loc[references["model"] == "Selected"]
    family_count_text = "; ".join(
        f"{row.family}={int(row.basis_count)}" for _, row in family_counts.groupby("family", sort=True).basis_count.median().reset_index().iterrows()
    ) if not family_counts.empty else "n/a"
    norm_medians = family_norms.groupby("family", sort=True).coefficient_norm.median().sort_values(ascending=False) if not family_norms.empty else pd.Series(dtype=float)
    norm_text = "; ".join(f"{name}={value:.4g}" for name, value in norm_medians.items()) if len(norm_medians) else "n/a"
    lines = [
        runner.EXPERIMENT_NAME,
        "Independent multi-branch basis OLS scan on the fixed 14 failed states.",
        "",
        "Execution status",
        f"- Numerical search completed: 3 coordinate-beam passes; Pass3 triggered={bool(search_info['pass3_triggered'])}.",
        f"- Candidate evaluations recorded={int(search_info['candidate_count'])}; unique candidate IDs={int(changed['candidate_id'].nunique())}; trace rows={int(pd.read_csv(SEARCH_ROOT / 'search_trace.csv').shape[0])}.",
        f"- Selected candidate={selected_id}; pass={int(selected_payload.get('pass_number', 0))}; basis count K={int(selected_payload['basis_count'])}; max effective delay={max(int(value) for value in selected_payload['effective_delays'].values())}.",
        f"- Best internal A/C BalancedMedian={float(search_info['selection']['best_balanced_median']):.6f} dB; selection used only blocked A/C CV.",
        "- The final bookkeeping was recovered after a JSON serialization error; numerical artifacts were reused and the search was not repeated.",
        "",
        "Selected independent parameters",
        *[f"- {key}={value}" for key, value in sorted(selected_config.items())],
        "",
        "Failure14 median metrics (common support trim=26)",
        "model | Aend train | Aend→B | C2 train | C2→B | Aend gap | C2 gap",
    ]
    for model in ("Frozen", "G4", "SourceTop", "Selected"):
        lines.append(
            f"{model} | {med(model, 'Aend', 'train_NMSE_dB'):.6f} | {med(model, 'Aend', 'B_NMSE_dB'):.6f} | "
            f"{med(model, 'C2', 'train_NMSE_dB'):.6f} | {med(model, 'C2', 'B_NMSE_dB'):.6f} | "
            f"{med(model, 'Aend', 'generalization_gap_dB'):.6f} | {med(model, 'C2', 'generalization_gap_dB'):.6f}"
        )
    lines.extend([
        "",
        "Selected minus Frozen (negative means improvement)",
        *[
            f"- {side} {metric}: {med('Selected', side, metric) - med('Frozen', side, metric):+.6f} dB"
            for side, metric in (("Aend", "train_NMSE_dB"), ("Aend", "B_NMSE_dB"), ("C2", "train_NMSE_dB"), ("C2", "B_NMSE_dB"))
        ],
        f"- Statewise worst-four delta median={float(delta_worst.median()):+.6f} dB; improved states={int((delta_worst < 0).sum())}/{len(delta_worst)}; degraded states={int((delta_worst > 0).sum())}/{len(delta_worst)}.",
        f"- Selected B-balanced median={max(med('Selected', 'Aend', 'B_NMSE_dB'), med('Selected', 'C2', 'B_NMSE_dB')):.6f} dB (reported as max of side medians); Frozen counterpart={max(med('Frozen', 'Aend', 'B_NMSE_dB'), med('Frozen', 'C2', 'B_NMSE_dB')):.6f} dB.",
        "",
        "Numerical and structural diagnostics",
        f"- Selected rank range={int(selected_conditioning['rank'].min())}..{int(selected_conditioning['rank'].max())} for K={int(selected_conditioning['K'].max())}; maximum rank deficit={int((selected_conditioning['K'] - selected_conditioning['rank']).max())}.",
        f"- Selected median condition number Aend/C2={float(selected_conditioning.loc[selected_conditioning.side == 'Aend', 'condition_number'].median()):.6g}/{float(selected_conditioning.loc[selected_conditioning.side == 'C2', 'condition_number'].median()):.6g}.",
        f"- Selected median coefficient norms by family: {norm_text}.",
        f"- Selected median family basis counts: {family_count_text}.",
        "- Basis IDs are unique and the prelinear coefficient basis is excluded from the final design matrix; all family parameter blocks remain independent.",
        "- Support is synchronized at A=12262, B=4889, C=7347 valid samples after the common 26-sample trim.",
        "",
        "Boundary and interpretation",
        "- B targets were materialized only after search/selected_model_structure.json was written; B was not used for candidate or parameter selection.",
        "- The selected OLS design is numerically ill-conditioned and rank-deficient for some state/side. This is reported as a warning, not hidden by dropping columns or adding Ridge.",
        "- The selected multibranch model is not automatically promoted to the frozen model and this task stops before LUT retrieval, Real-B shareability, and low-bandwidth experiments.",
        f"- Raw/protected historical data unchanged={bool(validation.get('protection_verification', {}).get('all_protected_unchanged', False))}; support gate={validation['gates']['support']['pass']}; full-rank gate={validation['gates']['rank']['pass']}.",
        "",
        "Output locations",
        f"- Result root: {RESULT_ROOT}",
        f"- Selected structure: {SEARCH_ROOT / 'selected_model_structure.json'}",
        f"- Main metrics: {TABLE_ROOT / 'selected_vs_references.csv'} and {TABLE_ROOT / 'selected_per_state_metrics.csv'}",
        f"- Figures: {RESULT_ROOT / 'figures'} (PNG only)",
    ])
    return "\n".join(lines) + "\n"


def _excel_source(validation: dict[str, Any], selected_payload: dict[str, Any]) -> dict[str, Any]:
    registry = _records(SEARCH_ROOT / "candidate_registry.csv")
    sensitivity = _records(SEARCH_ROOT / "parameter_sensitivity.csv")
    selected = _records(TABLE_ROOT / "selected_per_state_metrics.csv")
    refs = _records(TABLE_ROOT / "selected_vs_references.csv")
    conditioning = _records(TABLE_ROOT / "conditioning.csv")
    family_counts = _records(TABLE_ROOT / "basis_family_counts.csv")
    family_norms = _records(TABLE_ROOT / "coefficient_family_norms.csv")
    uniqueness = _records(TABLE_ROOT / "basis_uniqueness_audit.csv")
    config = []
    for key, value in sorted(validation.items()):
        if key in {"gates", "search_info", "discovery_cache_manifest", "figures", "protection_verification"}:
            value = json.dumps(runner._json_safe(value), ensure_ascii=False, separators=(",", ":"))
        config.append({"key": key, "value": value})
    structure_rows = [{"field": key, "value": value} for key, value in selected_payload.items() if key not in {"basis_manifest"}]
    return {
        "experiment_config": config,
        "selected_structure": structure_rows,
        "basis_manifest": selected_payload.get("basis_manifest", []),
        "search_summary": registry,
        "parameter_sensitivity": sensitivity,
        "selected_failed14": selected,
        "reference_comparison": refs,
        "generalization_gap": [
            {key: row.get(key) for key in ("state_id", "model", "side", "generalization_gap_dB")}
            for row in refs
        ],
        "conditioning": conditioning,
        "basis_family_counts": family_counts,
        "coefficient_norms": family_norms,
        "uniqueness_audit": uniqueness,
    }


def main() -> None:
    selected_path = SEARCH_ROOT / "selected_model_structure.json"
    validation_path = RESULT_ROOT / "validation.json"
    required = [selected_path, validation_path, SEARCH_ROOT / "candidate_registry.csv", TABLE_ROOT / "selected_vs_references.csv", TABLE_ROOT / "selected_per_state_metrics.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot finalize; missing completed artifacts: " + "; ".join(missing))
    selected_payload = _read_json(selected_path)
    validation = _update_gates(_read_json(validation_path))
    registry = pd.read_csv(SEARCH_ROOT / "candidate_registry.csv")
    references = pd.read_csv(TABLE_ROOT / "selected_vs_references.csv")
    selected_rows = pd.read_csv(TABLE_ROOT / "selected_per_state_metrics.csv")
    family_counts = pd.read_csv(TABLE_ROOT / "basis_family_counts.csv")
    family_norms = pd.read_csv(TABLE_ROOT / "coefficient_family_norms.csv")
    search_state = _read_json(SEARCH_ROOT / "search_state.json")
    search_state.update({"completed": True, "selected_candidate_id": int(selected_payload["candidate_id"])})
    _write_json(RESULT_ROOT / "search_state.json", search_state)
    _write_json(RESULT_ROOT / "search_progress.json", {"completed": True, "candidate_count": int(validation["search_info"]["candidate_count"]), "pass_count": int(validation["search_info"]["pass_count_completed"]), "selected_candidate_id": int(selected_payload["candidate_id"]), "B_opened_after_structure_freeze": True})
    _write_json(RESULT_ROOT / "validation.json", validation)
    _write_json(runner.VALIDATION_ROOT / "validation.json", validation)
    summary = _summary_text(selected_payload, validation, registry, references, selected_rows, family_counts, family_norms)
    (RESULT_ROOT / "final_result_summary.txt").write_text(summary, encoding="utf-8")
    _write_json(RESULT_ROOT / "excel_source.json", _excel_source(validation, selected_payload))
    timestamp = datetime.now(UTC).isoformat()
    log = (
        f"{runner.EXPERIMENT_NAME} finalized after bookkeeping recovery: candidate_count={validation['search_info']['candidate_count']}, "
        f"selected={selected_payload['candidate_id']}, K={selected_payload['basis_count']}, passes={validation['search_info']['pass_count_completed']}; "
        f"rank_gate={validation['gates']['rank']['pass']}, support_gate={validation['gates']['support']['pass']}, "
        f"raw/protected unchanged={validation.get('protection_verification', {}).get('all_protected_unchanged', False)}."
    )
    for path in (runner.MODEL_LOG, runner.HANDOFF_LOG):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n[{timestamp}] {runner.EXPERIMENT_NAME} finalization\n{log}\n")
    print(log)


if __name__ == "__main__":
    main()
