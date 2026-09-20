# ruff: noqa: E501

"""Discover and normalize historical support/lambda retrieval candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .self_first_metrics import canonical_support_hash

MP_CANONICAL_IDS = (
    "LIN_d0",
    "LIN_d1",
    "LIN_d2",
    "ENV_p02_m0_q0",
    "ENV_p02_m1_q1",
    "ENV_p03_m0_q0",
    "ENV_p03_m1_q1",
    "ENV_p05_m0_q0",
    "ENV_p07_m0_q0",
    "ENV_p09_m0_q0",
)
MP_SHORT_TO_CANONICAL = {
    short: basis_id
    for short, basis_id in zip(
        ("p1_m0", "p1_m1", "p1_m2", "p2_m0", "p2_m1", "p3_m0", "p3_m1", "p5_m0", "p7_m0", "p9_m0"),
        MP_CANONICAL_IDS,
    )
}


def _parse_list(value: Any) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return ()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in text.split(";") if item.strip()]
    if not isinstance(parsed, (list, tuple)):
        return (str(parsed),)
    return tuple(str(item) for item in parsed)


def _canonicalize(ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(MP_SHORT_TO_CANONICAL.get(item, item) for item in ids)


def _first_value(*values: Any, default: float | None = None) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return default


def _find_distance(root: Path, name: str, distance_file: str | None = None) -> Path | None:
    if distance_file and distance_file not in {"", "nan", "None"}:
        direct = root / distance_file
        if direct.is_file():
            return direct
        matches = list(root.rglob(Path(distance_file).name))
        if matches:
            return matches[0]
    direct = root / f"{name}.npy"
    if direct.is_file():
        return direct
    matches = list(root.rglob(f"{name}.npy"))
    return matches[0] if matches else None


def _find_related_npz(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.npz") if "coefficient" in path.name.lower())


def _ensure_entry(entries: dict[tuple[str, float], dict[str, Any]], ids: tuple[str, ...], lambda_value: float) -> dict[str, Any]:
    if not ids:
        raise ValueError("cannot register an empty support")
    digest = canonical_support_hash(ids)
    key = (digest, float(lambda_value))
    if key not in entries:
        entries[key] = {
            "support_hash": digest,
            "basis_ids": ids,
            "K": len(ids),
            "lambda": float(lambda_value),
            "source_tasks": set(),
            "source_model_names": set(),
            "source_model_ids": set(),
            "distance_paths": set(),
            "coefficient_paths": set(),
            "fingerprint_paths": set(),
            "query_metrics_paths": set(),
        }
    return entries[key]


def _add(
    entries: dict[tuple[str, float], dict[str, Any]],
    *,
    ids: tuple[str, ...],
    lambda_value: float,
    source_task: str,
    source_model_name: str,
    source_model_id: Any,
    distance_path: Path | None,
    root: Path,
    query_metrics_path: Path | None = None,
) -> None:
    entry = _ensure_entry(entries, _canonicalize(ids), float(lambda_value))
    entry["source_tasks"].add(str(source_task))
    entry["source_model_names"].add(str(source_model_name))
    if source_model_id is not None and str(source_model_id).lower() != "nan":
        entry["source_model_ids"].add(str(source_model_id))
    if distance_path is not None and distance_path.is_file():
        entry["distance_paths"].add(str(distance_path))
    for path in _find_related_npz(root):
        entry["coefficient_paths"].add(str(path))
    if query_metrics_path is not None and query_metrics_path.is_file():
        entry["query_metrics_paths"].add(str(query_metrics_path))


def _add_summary(
    entries: dict[tuple[str, float], dict[str, Any]],
    root: Path,
    summary_path: Path,
    support_path: Path | None = None,
    default_lambda: float | None = None,
) -> int:
    if not summary_path.is_file():
        return 0
    summary = pd.read_csv(summary_path)
    support = pd.read_csv(support_path) if support_path is not None and support_path.is_file() else None
    added = 0
    for _, row in summary.iterrows():
        support_row = None
        if support is not None:
            if "candidate_id" in row and "model_id" in support:
                matches = support.loc[support["model_id"].astype(str) == str(row.get("candidate_id"))]
                if matches.shape[0] == 1:
                    support_row = matches.iloc[0]
            if support_row is None and "candidate_name" in row and "model_name" in support:
                matches = support.loc[support["model_name"].astype(str) == str(row.get("candidate_name"))]
                if matches.shape[0] == 1:
                    support_row = matches.iloc[0]
        ids = ()
        for column in ("support_basis_ids", "basis_ids", "retained_basis_ids", "support_ids"):
            if column in row:
                ids = _parse_list(row.get(column))
                if ids:
                    break
            if support_row is not None and column in support_row:
                ids = _parse_list(support_row.get(column))
                if ids:
                    break
        ids = _canonicalize(ids)
        if not ids:
            continue
        lambda_value = _first_value(
            row.get("lambda") if "lambda" in row else None,
            row.get("ridge_lambda") if "ridge_lambda" in row else None,
            row.get("lambda_value") if "lambda_value" in row else None,
            support_row.get("lambda") if support_row is not None and "lambda" in support_row else None,
            support_row.get("ridge_lambda") if support_row is not None and "ridge_lambda" in support_row else None,
            default=default_lambda,
        )
        if lambda_value is None:
            continue
        name = str(row.get("candidate_name", row.get("model_name", "candidate")))
        distance_file = str(row.get("distance_file", "")) if "distance_file" in row else None
        distance_path = _find_distance(root, name, distance_file)
        _add(
            entries,
            ids=ids,
            lambda_value=lambda_value,
            source_task=root.name,
            source_model_name=name,
            source_model_id=row.get("candidate_id", row.get("model_id")),
            distance_path=distance_path,
            root=root,
            query_metrics_path=summary_path,
        )
        added += 1
    return added


def _add_definition(entries: dict[tuple[str, float], dict[str, Any]], path: Path, default_lambda: float | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return False
    ids: tuple[str, ...] = ()
    if "basis_id" in frame:
        ordered = frame.sort_values("term_index") if "term_index" in frame else frame
        ids = tuple(str(item) for item in ordered["basis_id"].dropna())
    else:
        for column in ("basis_ids", "support_basis_ids"):
            if column in frame:
                ids = _parse_list(frame.iloc[0][column])
                if ids:
                    break
    ids = _canonicalize(ids)
    if not ids:
        return False
    row = frame.iloc[0]
    lambda_value = _first_value(
        row.get("lambda") if "lambda" in row else None,
        row.get("ridge_lambda") if "ridge_lambda" in row else None,
        default=default_lambda,
    )
    if lambda_value is None:
        return False
    source_task = path.parent.name
    name = str(row.get("model_name", path.stem))
    _add(
        entries,
        ids=ids,
        lambda_value=lambda_value,
        source_task=source_task,
        source_model_name=name,
        source_model_id="definition",
        distance_path=None,
        root=path.parent,
    )
    return True


def _add_npz_definition(entries: dict[tuple[str, float], dict[str, Any]], path: Path, source_task: str, default_lambda: float | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if "basis_ids" not in data:
                return False
            ids = tuple(str(item) for item in np.asarray(data["basis_ids"]).reshape(-1))
            lambda_value = _first_value(data["lambda_value"] if "lambda_value" in data else None, data["selected_lambda"] if "selected_lambda" in data else None, default=default_lambda)
    except (OSError, ValueError):
        return False
    if lambda_value is None or not ids:
        return False
    _add(entries, ids=ids, lambda_value=lambda_value, source_task=source_task, source_model_name=path.stem, source_model_id="npz_definition", distance_path=None, root=path.parent)
    return True


def build_registry(project_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the current historical registry without touching raw data."""

    results = project_root / "results"
    entries: dict[tuple[str, float], dict[str, Any]] = {}
    audit: dict[str, Any] = {"result_roots_seen": [], "summary_rows_seen": 0, "unavailable_notes": []}

    def add_summary(root_name: str, summary_name: str, support_name: str | None = None, default_lambda: float | None = None) -> None:
        root = results / root_name
        audit["result_roots_seen"].append(root_name)
        summary_path = root / summary_name
        before = len(entries)
        _add_summary(entries, root, summary_path, root / support_name if support_name else None, default_lambda)
        if summary_path.is_file():
            try:
                audit["summary_rows_seen"] += max(0, sum(1 for _ in summary_path.open(encoding="utf-8")) - 1)
            except OSError:
                pass
        if len(entries) == before and not summary_path.is_file():
            audit["unavailable_notes"].append(f"missing {summary_path}")

    add_summary("scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b", "08_candidate_retrieval_summary.csv", "03_candidate_supports.csv", 1e-8)
    add_summary("scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b", "08_candidate_retrieval_summary.csv", "04_candidate_supports.csv", 1e-8)
    add_summary("scenario_2_k12_retrieval_oriented_forward_scan_5b", "10_candidate_retrieval_summary.csv", "06_candidate_supports.csv", 1e-8)
    add_summary("scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b", "12_retrieval_summary.csv", "07_unique_k13_supports.csv", 1e-8)
    add_summary("scenario_2_beam3_k13_floating_backward_retrieval_ablation_5b", "11_retrieval_summary.csv", "05_unique_k12_children.csv", 1e-8)
    add_summary("scenario_2_updated_beam3_iteration2_retrieval_oriented_forward_expansion_5b", "13_retrieval_summary.csv", "07_unique_children.csv", 1e-8)

    mp_root = results / "scenario_2_mp10_full_lut_retrieval_5b"
    protocol_path = mp_root / "02_protocol_contract.json"
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        lam = float(protocol["model"]["ridge_lambda"])
        _add(entries, ids=MP_CANONICAL_IDS, lambda_value=lam, source_task=mp_root.name, source_model_name="MP10_full", source_model_id=0, distance_path=mp_root / "08_mp10_fingerprint_cnmse_matrix.npy", root=mp_root, query_metrics_path=mp_root / "09_retrieval_results_all425.csv")

    e18_root = results / "scenario_2_envelope18_commonB_full_lut_retrieval_5B"
    try:
        from behavior_modeling.shared.basis_function_selection.all425_envelope18_evaluation import (
            FINAL_SUPPORT_IDS,
        )

        _add(entries, ids=tuple(FINAL_SUPPORT_IDS), lambda_value=0.0, source_task=e18_root.name, source_model_name="Envelope18", source_model_id="E18", distance_path=e18_root / "05_fingerprint_cnmse_matrix.npy", root=e18_root, query_metrics_path=e18_root / "03_all425_retrieval_metrics.csv")
    except (ImportError, AttributeError):
        audit["unavailable_notes"].append("Envelope18 support constants unavailable")

    for root_name, npz_name, matrix_name, model_name in (
        ("scenario_2_envelope19_c2endshared_commonB_full_lut_retrieval_5B", "04_envelope19_aend_c2_coefficients.npz", "05_envelope19_fingerprint_cnmse_matrix.npy", "Envelope19-C2EndShared"),
        ("scenario_2_envelope23_ilcend_commonB_full_lut_retrieval_5B", "04_envelope23_aend_c2_coefficients.npz", "05_envelope23_fingerprint_cnmse_matrix.npy", "Envelope23-ilcEnd"),
    ):
        root = results / root_name
        npz_path = root / npz_name
        if not npz_path.is_file():
            audit["unavailable_notes"].append(f"missing {npz_path}")
            continue
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                ids = tuple(str(item) for item in np.asarray(data["basis_ids"]).reshape(-1))
                lam = float(np.asarray(data["lambda_value"]).reshape(-1)[0])
            _add(entries, ids=ids, lambda_value=lam, source_task=root.name, source_model_name=model_name, source_model_id="Full-LUT", distance_path=root / matrix_name, root=root, query_metrics_path=root / "07_retrieval_diagnostics_all425.csv")
        except (OSError, ValueError, KeyError):
            audit["unavailable_notes"].append(f"invalid {npz_path}")

    for path in sorted(results.rglob("*model_definition*.csv")):
        _add_definition(entries, path)
    for root_name, npz_name in (
        ("scenario_2_all425_ilcend_envelope23_coverage_evaluation_5B", "04_all425_train_coefficients.npz"),
        ("scenario_2_all425_c2_envelope23_ridge_generalization_5B", "06_final_ridge_coefficients.npz"),
        ("scenario_2_all425_c2_ilcend_envelope19_shared_coverage_5B", "06_final_dual_behavior_coefficients.npz"),
    ):
        root = results / root_name
        _add_npz_definition(entries, root / npz_name, root_name)

    rows: list[dict[str, Any]] = []
    for candidate_id, key in enumerate(sorted(entries, key=lambda item: (len(entries[item]["basis_ids"]), item[0], item[1]))):
        entry = entries[key]
        distance_paths = sorted(entry["distance_paths"])
        coefficient_paths = sorted(entry["coefficient_paths"])
        query_paths = sorted(entry["query_metrics_paths"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "support_id": f"S{candidate_id:04d}",
                "support_hash": entry["support_hash"],
                "K": entry["K"],
                "basis_ids": json.dumps(list(entry["basis_ids"]), ensure_ascii=False),
                "lambda": entry["lambda"],
                "source_task": ";".join(sorted(entry["source_tasks"])),
                "source_model_name": ";".join(sorted(entry["source_model_names"])),
                "source_model_id": ";".join(sorted(entry["source_model_ids"])),
                "distance_matrix_path": distance_paths[0] if distance_paths else "",
                "coefficients_path": coefficient_paths[0] if coefficient_paths else "",
                "query_metrics_path": query_paths[0] if query_paths else "",
                "coefficients_available": bool(coefficient_paths),
                "fingerprints_available": bool(distance_paths),
                "distance_matrix_available": bool(distance_paths),
                "query_metrics_available": bool(query_paths),
            }
        )
    audit["candidate_count"] = len(rows)
    audit["distance_candidate_count"] = int(sum(bool(row["distance_matrix_available"]) for row in rows))
    audit["support_count"] = len({row["support_hash"] for row in rows})
    return pd.DataFrame(rows), audit


__all__ = ["MP_CANONICAL_IDS", "build_registry"]
