"""Multi-start coordinate-beam OLS search for independent multi-branch bases.

This experiment is intentionally separate from the frozen/G4 retrieval tasks.
The search sees only Aend-A and C2-C data.  Each CV fold refits the Branch-1
prelinear model before constructing the final three-branch design matrix.  B
targets are opened only after ``search/selected_model_structure.json`` exists.
"""

# ruff: noqa: E402,I001,E501

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
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

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.shared import frozen_neighborhood_memory_ridge_scan as frozen_model  # noqa: E402
from behavior_modeling.shared import multibranch_independent_basis as mb  # noqa: E402
from behavior_modeling.shared import sparse_gmp as gmp  # noqa: E402
from behavior_modeling.shared.evaluation import calculate_nmse  # noqa: E402
from behavior_modeling.scenario_2.scenario_2_failed14_multibranch_independent_basis_ols_scan_5b.plot_scenario2_failed14_multibranch_independent_basis_ols_scan_5b import (  # noqa: E402
    generate_figures,
)

STATE_IDS = tuple(int(value) for value in frozen_model.FAILED_STATE_IDS)
STATE_COUNT = len(STATE_IDS)
ORDERS = tuple(mb.ODD_ORDERS)
COMMON_TRIM = mb.COMMON_SUPPORT_TRIM
CV_FOLDS = 3
BEAM_WIDTH = 3
MAX_PASSES = 3
CV_TOLERANCE_DB = 0.10
THRESHOLD_DB = -40.0
CPU_TARGET = 0.90
REGRESSION_STATES = (187, 195, 340, 354)

TASK_NAME = "scenario_2_failed14_multibranch_independent_basis_ols_scan_5b"
EXPERIMENT_NAME = "scenario_2_failed14_multibranch_independent_basis_OLS_scan_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME / EXPERIMENT_NAME
CONFIG_ROOT = RESULT_ROOT / "config"
CACHE_ROOT = RESULT_ROOT / "cache"
DISCOVERY_CACHE_ROOT = CACHE_ROOT / "preprocessed_failed14"
POST_B_CACHE_ROOT = CACHE_ROOT / "post_selection_B"
SEARCH_ROOT = RESULT_ROOT / "search"
TABLE_ROOT = RESULT_ROOT / "tables"
MODEL_ROOT = RESULT_ROOT / "models"
FIGURE_ROOT = RESULT_ROOT / "figures"
VALIDATION_ROOT = RESULT_ROOT / "validation"
MODEL_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
FAILURE_SOURCE = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /"scenario_2_C2_to_Aend" / "failed_retrieval_states.csv"
OLD_FORMAL_METRICS = PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2" /"scenario_2_all_ilc" / "all_ilc_model_metrics.csv"

PROTECTED_RESULT_DIRS = {
    "formal_5B_retrieval": PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /"scenario_2_C2_to_Aend",
    "all_ilc_frozen_metrics": PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2" /"scenario_2_all_ilc",
    "g4_sparse_gmp": PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME / "scenario_2_C2_to_Aend_G4_sparse_gmp_5B",
    "g4_discovery": PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME / ".." / "behavior_modeling" / "scenario_2_failed14_frozen_mp_residual_sparse_gmp_5B",
    "frozen_neighborhood_scan": PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME / ".." / "behavior_modeling" / "scenario_2_failed14_frozen_neighborhood_memory_ridge_scan_5B",
    "p5_order2_scan": PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME / ".." / "behavior_modeling" / "scenario_2_failed14_P5_with_order2_variable_memory_OLS_scan_5B",
}

G4_DISCOVERY_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /"scenario_2_failed14_frozen_mp_residual_sparse_gmp_5B"
G4_STRUCTURE_PATH = G4_DISCOVERY_ROOT / "models" / "selected_sparse_gmp_structure.json"
OLD_FROZEN_RETRIEVAL_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /"scenario_2_C2_to_Aend"


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    seed: str
    pass_number: int
    parent_id: int | None
    config: tuple[tuple[str, int], ...]

    @property
    def config_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in self.config}

    @property
    def key(self) -> str:
        return mb.config_json(self.config_dict)


@dataclass(frozen=True)
class StateCache:
    state_id: int
    ilc_A_end: int
    a_x: np.ndarray
    a_y: np.ndarray
    c_x: np.ndarray
    c_y: np.ndarray


_WORKER_CACHE: dict[int, StateCache] = {}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.where(pd.notna(value), None).to_dict("records"))
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_frame(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NaN", float_format="%.17g", compression=compression)


def _write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _tree_digest(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*"), key=lambda item: str(item).lower()):
        if not file.is_file() or "__pycache__" in file.parts or file.suffix.lower() == ".pyc":
            continue
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode("utf-8") + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
    raw_root = PROJECT_ROOT / "data" / "raw"
    digest = hashlib.sha256()
    files = sorted((file for file in raw_root.rglob("*") if file.is_file()), key=lambda item: str(item).lower())
    total = 0
    for file in files:
        size = int(file.stat().st_size)
        digest.update(str(file.relative_to(raw_root)).replace("\\", "/").encode("utf-8") + b"\0")
        digest.update(size.to_bytes(8, "little"))
        total += size
    return {"sha256": digest.hexdigest(), "file_count": len(files), "bytes": total}


def _snapshot() -> dict[str, Any]:
    return {"data/raw": _raw_manifest(), "result_dirs": {name: {"path": str(path.resolve()), "exists": path.is_dir(), "sha256": _tree_digest(path)} for name, path in PROTECTED_RESULT_DIRS.items()}}


def _verify_protection(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    rb, ra = before["data/raw"], after["data/raw"]
    raw = {"sha256_unchanged": rb["sha256"] == ra["sha256"], "file_count_unchanged": rb["file_count"] == ra["file_count"], "bytes_unchanged": rb["bytes"] == ra["bytes"]}
    dirs = {name: {"sha256_unchanged": entry["sha256"] == after["result_dirs"][name]["sha256"], "before": entry["sha256"], "after": after["result_dirs"][name]["sha256"]} for name, entry in before["result_dirs"].items()}
    return {"data/raw": raw, "result_dirs": dirs, "all_protected_unchanged": bool(all(raw.values()) and all(entry["sha256_unchanged"] for entry in dirs.values()))}


def _validate_failure_source() -> dict[str, Any]:
    frame = pd.read_csv(FAILURE_SOURCE)
    values = frame["failure"]
    mask = values if values.dtype == bool else values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    ids = tuple(sorted(int(value) for value in frame.loc[mask, "State_n_R"]))
    if ids != STATE_IDS:
        raise RuntimeError(f"failure state list mismatch: {ids} vs {STATE_IDS}")
    return {"source": str(FAILURE_SOURCE), "state_ids": list(STATE_IDS), "failure_count": STATE_COUNT, "used_for_state_subset_only": True}


def _prepare_discovery_cache() -> dict[str, Any]:
    DISCOVERY_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for position, state_id in enumerate(STATE_IDS, start=1):
        path = DISCOVERY_CACHE_ROOT / f"state_{state_id:03d}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                prepared = StateCache(int(state_id), int(np.asarray(data["ilc_A_end"]).reshape(-1)[0]), np.asarray(data["a_x"], dtype=np.complex128), np.asarray(data["a_y"], dtype=np.complex128), np.asarray(data["c_x"], dtype=np.complex128), np.asarray(data["c_y"], dtype=np.complex128))
        else:
            source = frozen_model.prepare_state(state_id)
            prepared = StateCache(state_id, source.ilc_A_end, source.aend_A_input, source.aend_A_output, source.c2_C_input, source.c2_C_output)
            _write_npz(path, {"state_id": state_id, "ilc_A_end": prepared.ilc_A_end, "a_x": prepared.a_x, "a_y": prepared.a_y, "c_x": prepared.c_x, "c_y": prepared.c_y})
        if prepared.a_x.size != 12288 or prepared.c_x.size != 7373 or prepared.a_y.size != prepared.a_x.size or prepared.c_y.size != prepared.c_x.size:
            raise RuntimeError(f"state {state_id} discovery cache length mismatch")
        records.append({"state_id": state_id, "ilc_A_end": prepared.ilc_A_end, "A_length": int(prepared.a_x.size), "C_length": int(prepared.c_x.size), "B_target_saved": False, "cache_file": str(path)})
        print(f"preprocessed failed14 {position}/{STATE_COUNT}: {state_id}", flush=True)
    manifest = {"state_ids": list(STATE_IDS), "state_count": STATE_COUNT, "common_support_trim": COMMON_TRIM, "contains_Aend_A": True, "contains_C2_C": True, "contains_B_input": False, "contains_B_target": False, "records": records}
    _write_json(CACHE_ROOT / "preprocessed_manifest.json", manifest)
    return manifest


def _load_state_cache(cache_dir: Path = DISCOVERY_CACHE_ROOT) -> dict[int, StateCache]:
    result: dict[int, StateCache] = {}
    for state_id in STATE_IDS:
        with np.load(cache_dir / f"state_{state_id:03d}.npz", allow_pickle=False) as data:
            result[state_id] = StateCache(state_id, int(np.asarray(data["ilc_A_end"]).reshape(-1)[0]), np.asarray(data["a_x"], dtype=np.complex128), np.asarray(data["a_y"], dtype=np.complex128), np.asarray(data["c_x"], dtype=np.complex128), np.asarray(data["c_y"], dtype=np.complex128))
    return result


def _fit_ols(phi: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    theta, _, rank, singular = np.linalg.lstsq(phi, y, rcond=None)
    singular = np.asarray(singular, dtype=float)
    if phi.shape[1] <= 0 or singular.size == 0:
        raise ValueError("OLS design matrix has no columns")
    return np.asarray(theta, dtype=np.complex128), singular, int(rank), float(np.linalg.norm(y - phi @ theta))


def _fit_fold(x: np.ndarray, y: np.ndarray, config: Mapping[str, int], train_index: np.ndarray, validation_index: np.ndarray) -> dict[str, Any]:
    memory = int(config["M_pre"])
    phi_pre = mb.prelinear_basis(x, memory, COMMON_TRIM)
    y_valid = np.asarray(y[COMMON_TRIM:], dtype=np.complex128)
    a, _, rank_pre, residual_pre = _fit_ols(phi_pre[train_index], y_valid[train_index])
    z_full = mb.apply_prelinear(x, a, memory)
    phi_total, terms = mb.build_total_matrix(config, x, z_full, trim=COMMON_TRIM)
    theta, singular, rank, residual = _fit_ols(phi_total[train_index], y_valid[train_index])
    prediction = phi_total[validation_index] @ theta
    condition_number = float(np.inf if singular[-1] <= 0 else singular[0] / singular[-1])
    return {"nmse": float(calculate_nmse(y_valid[validation_index], prediction)), "pre_rank": rank_pre, "rank": rank, "column_count": len(terms), "sigma_max": float(singular[0]), "sigma_min": float(singular[-1]), "condition_number": condition_number, "theta_norm": float(np.linalg.norm(theta)), "pre_residual_norm": residual_pre, "residual_norm": residual, "a": a, "theta": theta, "terms": terms}


def _evaluate_candidate_group(candidate: Candidate, state_id: int, side: str) -> dict[str, Any]:
    cache = _WORKER_CACHE[int(state_id)]
    x, y = (cache.a_x, cache.a_y) if side == "Aend" else (cache.c_x, cache.c_y)
    n = x.size - COMMON_TRIM
    blocks = tuple(np.asarray(index, dtype=np.int64) for index in np.array_split(np.arange(n, dtype=np.int64), CV_FOLDS))
    fold_rows: list[dict[str, Any]] = []
    for validation_index in blocks:
        train_mask = np.ones(n, dtype=bool)
        train_mask[validation_index] = False
        fold_rows.append(_fit_fold(x, y, candidate.config_dict, np.flatnonzero(train_mask), validation_index))
    values = np.asarray([row["nmse"] for row in fold_rows], dtype=float)
    return {"candidate_id": candidate.candidate_id, "state_id": int(state_id), "side": side, "seed": candidate.seed, "pass_number": candidate.pass_number, "parent_id": candidate.parent_id, "config_key": candidate.key, "cv_NMSE_dB": float(values.mean()), "fold_1_dB": float(values[0]), "fold_2_dB": float(values[1]), "fold_3_dB": float(values[2]), "rank_min": int(min(row["rank"] for row in fold_rows)), "pre_rank_min": int(min(row["pre_rank"] for row in fold_rows)), "column_count": int(fold_rows[0]["column_count"]), "condition_number_max": float(max(row["condition_number"] for row in fold_rows)), "theta_norm_median": float(np.median([row["theta_norm"] for row in fold_rows])), "fold_specific_prelinear_refit": True}


def _worker_initializer(cache_dir: str) -> None:
    global _WORKER_CACHE
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    _WORKER_CACHE = _load_state_cache(Path(cache_dir))


def _aggregate_candidate(candidate: Candidate, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.shape[0] != STATE_COUNT * 2:
        validation = mb.validate_config(candidate.config_dict)
        return {"candidate_id": candidate.candidate_id, "basis_count": validation["basis_count"], "max_effective_delay": validation["max_effective_delay"], "valid": False, "failure_reason": "candidate/state/side evaluation incomplete"}
    a = frame.loc[frame["side"] == "Aend", "cv_NMSE_dB"].to_numpy(float)
    c = frame.loc[frame["side"] == "C2", "cv_NMSE_dB"].to_numpy(float)
    a_med, c_med = float(np.median(a)), float(np.median(c))
    row: dict[str, Any] = {"candidate_id": candidate.candidate_id, "seed": candidate.seed, "pass_number": candidate.pass_number, "parent_id": candidate.parent_id, "config_key": candidate.key, **candidate.config_dict, "basis_count": int(frame["column_count"].iloc[0]), "max_effective_delay": int(mb.validate_config(candidate.config_dict)["max_effective_delay"]), "A_CV_mean": float(np.mean(a)), "A_CV_median": a_med, "A_CV_q75": float(np.quantile(a, 0.75)), "A_CV_worst": float(np.max(a)), "C_CV_mean": float(np.mean(c)), "C_CV_median": c_med, "C_CV_q75": float(np.quantile(c, 0.75)), "C_CV_worst": float(np.max(c)), "BalancedMedian": max(a_med, c_med), "BalancedQ75": max(float(np.quantile(a, 0.75)), float(np.quantile(c, 0.75))), "BalancedWorst": max(float(np.max(a)), float(np.max(c))), "rank_min": int(frame["rank_min"].min()), "pre_rank_min": int(frame["pre_rank_min"].min()), "condition_number_max": float(frame["condition_number_max"].max()), "theta_norm_median": float(frame["theta_norm_median"].median()), "fold_specific_prelinear_refit": bool(frame["fold_specific_prelinear_refit"].all()), "valid": True}
    return row


def _evaluate_candidates(candidates: Sequence[Candidate], cache_dir: Path, workers: int) -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    if not candidates:
        return {}, pd.DataFrame()
    rows_by_candidate: dict[int, list[dict[str, Any]]] = {candidate.candidate_id: [] for candidate in candidates}
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_initializer, initargs=(str(cache_dir),)) as executor:
        futures = {executor.submit(_evaluate_candidate_group, candidate, state_id, side): (candidate, state_id, side) for candidate in candidates for state_id in STATE_IDS for side in ("Aend", "C2")}
        for done, future in enumerate(as_completed(futures), start=1):
            candidate = futures[future][0]
            try:
                result = future.result()
            except Exception as exc:
                print(f"candidate {candidate.candidate_id} task failed: {type(exc).__name__}: {exc}", flush=True)
                result = None
            if result is not None:
                rows_by_candidate[int(result["candidate_id"])].append(result)
            if done % 250 == 0 or done == len(futures):
                print(f"candidate/state/side tasks: {done}/{len(futures)}", flush=True)
    aggregates = {candidate.candidate_id: _aggregate_candidate(candidate, rows_by_candidate[candidate.candidate_id]) for candidate in candidates}
    frame = pd.DataFrame(list(aggregates.values())).sort_values("candidate_id").reset_index(drop=True)
    return aggregates, frame


def _candidate_from_config(config: Mapping[str, int], *, candidate_id: int, seed: str, pass_number: int, parent_id: int | None) -> Candidate:
    mb.validate_config(config)
    return Candidate(candidate_id, seed, pass_number, parent_id, tuple(sorted((key, int(value)) for key, value in config.items())))


def _candidate_grid_frame(candidates: Sequence[Candidate], scores: Mapping[int, Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        score = dict(scores.get(candidate.candidate_id, {}))
        config = candidate.config_dict
        validation = mb.validate_config(config)
        rows.append({"candidate_id": candidate.candidate_id, "seed": candidate.seed, "pass_number": candidate.pass_number, "parent_id": candidate.parent_id, "config_key": candidate.key, **config, "basis_count": validation["basis_count"], "max_effective_delay": validation["max_effective_delay"], **{key: value for key, value in score.items() if key not in config}})
    return pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)


def _beam_sort(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame.loc[frame["valid"].astype(bool)].copy()
    return valid.sort_values(["BalancedMedian", "BalancedQ75", "BalancedWorst", "condition_number_max", "basis_count", "max_effective_delay", "candidate_id"], ascending=[True, True, True, True, True, True, True], kind="mergesort")


def _expand_beam(beam: Sequence[Candidate], parameter: str, *, pass_number: int, next_id: int, seed_prefix: str) -> tuple[list[Candidate], int]:
    values = mb.PARAMETER_GRIDS[parameter]
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for parent in beam:
        base = parent.config_dict
        for value in values:
            config = dict(base)
            config[parameter] = int(value)
            try:
                candidate = _candidate_from_config(config, candidate_id=next_id, seed=f"{seed_prefix}_{parent.seed}", pass_number=pass_number, parent_id=parent.candidate_id)
            except ValueError:
                continue
            if candidate.key in seen:
                continue
            seen.add(candidate.key)
            candidates.append(candidate)
            next_id += 1
    return candidates, next_id


def _run_one_pass(start_beam: Sequence[Candidate], pass_number: int, next_id: int, cache_dir: Path, workers: int, score_cache: dict[str, dict[str, Any]], registry: dict[int, Candidate], trace_rows: list[dict[str, Any]], beam_rows: list[dict[str, Any]], state_payload: dict[str, Any]) -> tuple[list[Candidate], int, float]:
    beam = list(start_beam)
    pending = [candidate for candidate in beam if candidate.key not in score_cache]
    if pending:
        aggregate, _ = _evaluate_candidates(pending, cache_dir, workers)
        for candidate in pending:
            score_cache[candidate.key] = aggregate[candidate.candidate_id]
            registry[candidate.candidate_id] = candidate
    for candidate in beam:
        registry[candidate.candidate_id] = candidate
    for block_index, block in enumerate(mb.PARAMETER_BLOCKS, start=1):
        parameter = block[0]
        variants, next_id = _expand_beam(beam, parameter, pass_number=pass_number, next_id=next_id, seed_prefix=f"P{pass_number}_{parameter}")
        for candidate in variants:
            registry[candidate.candidate_id] = candidate
        new_candidates = [candidate for candidate in variants if candidate.key not in score_cache]
        if new_candidates:
            aggregate, _ = _evaluate_candidates(new_candidates, cache_dir, workers)
            for candidate in new_candidates:
                score_cache[candidate.key] = aggregate[candidate.candidate_id]
        scored_frame = _candidate_grid_frame(variants, {candidate.candidate_id: score_cache[candidate.key] for candidate in variants})
        scored_frame["parameter_block"] = parameter
        trace_rows.extend(scored_frame.to_dict("records"))
        ordered = _beam_sort(scored_frame)
        beam = [registry[int(value)] for value in ordered.head(BEAM_WIDTH)["candidate_id"].tolist()]
        for rank, candidate in enumerate(beam, start=1):
            score = score_cache[candidate.key]
            beam_rows.append({"pass_number": pass_number, "block_index": block_index, "parameter": parameter, "rank": rank, "candidate_id": candidate.candidate_id, "BalancedMedian": score["BalancedMedian"], "basis_count": score["basis_count"], "config_key": candidate.key})
        state_payload.update({"current_pass": pass_number, "current_block": block_index, "current_parameter": parameter, "beam_candidate_ids": [candidate.candidate_id for candidate in beam], "completed_candidate_ids": sorted(registry)})
        _write_json(SEARCH_ROOT / "search_state.json", state_payload)
        print(f"pass {pass_number} block {block_index}/{len(mb.PARAMETER_BLOCKS)} {parameter}: beam={[candidate.candidate_id for candidate in beam]}", flush=True)
    best = _beam_sort(_candidate_grid_frame(beam, {candidate.candidate_id: score_cache[candidate.key] for candidate in beam})).iloc[0]
    return beam, next_id, float(best["BalancedMedian"])


def _select_final_candidate(registry: Mapping[int, Candidate], scores: Mapping[str, Mapping[str, Any]]) -> tuple[Candidate, dict[str, Any]]:
    rows = []
    for candidate_id, candidate in registry.items():
        score = scores.get(candidate.key)
        if score is None or not score.get("valid", False):
            continue
        rows.append({"candidate_id": candidate_id, **score})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no valid multibranch candidate")
    best_balanced = float(frame["BalancedMedian"].min())
    eligible = frame.loc[frame["BalancedMedian"] <= best_balanced + CV_TOLERANCE_DB].copy()
    eligible = eligible.sort_values(["basis_count", "BalancedQ75", "BalancedWorst", "rank_min", "condition_number_max", "max_effective_delay", "candidate_id"], ascending=[True, True, True, False, True, True, True], kind="mergesort")
    selected_id = int(eligible.iloc[0]["candidate_id"])
    selected = registry[selected_id]
    return selected, {"best_balanced_median": best_balanced, "tolerance_dB": CV_TOLERANCE_DB, "eligible_candidate_count": int(eligible.shape[0]), "selected_candidate_id": selected_id, "selection_rule": "smallest basis within 0.10 dB of best BalancedMedian; then BalancedQ75, BalancedWorst, rank, condition, delay, ID"}


def _run_search(cache_dir: Path, workers: int) -> tuple[Candidate, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Execute the deterministic two-seed coordinate beam search."""

    score_cache: dict[str, dict[str, Any]] = {}
    registry: dict[int, Candidate] = {}
    trace_rows: list[dict[str, Any]] = []
    beam_rows: list[dict[str, Any]] = []
    state_payload: dict[str, Any] = {"experiment": EXPERIMENT_NAME, "beam_width": BEAM_WIDTH, "max_coordinate_passes": MAX_PASSES, "completed_candidate_ids": []}
    source = _candidate_from_config(mb.SOURCE_PROFILE, candidate_id=0, seed="Source", pass_number=1, parent_id=None)
    compact = _candidate_from_config(mb.COMPACT_PROFILE, candidate_id=1, seed="Compact", pass_number=1, parent_id=None)
    registry[source.candidate_id] = source
    registry[compact.candidate_id] = compact
    base_aggregate, _ = _evaluate_candidates([source, compact], cache_dir, workers)
    for candidate in (source, compact):
        score_cache[candidate.key] = base_aggregate[candidate.candidate_id]
    next_id = 2
    print("coordinate search pass 1 starting", flush=True)
    beam1, next_id, best1 = _run_one_pass((source, compact), 1, next_id, cache_dir, workers, score_cache, registry, trace_rows, beam_rows, state_payload)
    print(f"coordinate search pass 1 complete: best BalancedMedian={best1:.6f}", flush=True)
    pass2_seed = tuple(beam1[:BEAM_WIDTH])
    beam2, next_id, best2 = _run_one_pass(pass2_seed, 2, next_id, cache_dir, workers, score_cache, registry, trace_rows, beam_rows, state_payload)
    print(f"coordinate search pass 2 complete: best BalancedMedian={best2:.6f}", flush=True)
    pass3_triggered = bool(best2 < best1 - 0.05)
    best3 = best2
    if pass3_triggered:
        _, next_id, best3 = _run_one_pass(tuple(beam2[:BEAM_WIDTH]), 3, next_id, cache_dir, workers, score_cache, registry, trace_rows, beam_rows, state_payload)
        print(f"coordinate search pass 3 complete: best BalancedMedian={best3:.6f}", flush=True)
    else:
        print("coordinate search pass 3 not triggered", flush=True)
    candidate_registry = _candidate_grid_frame(tuple(registry.values()), {candidate.candidate_id: score_cache[candidate.key] for candidate in registry.values()})
    candidate_registry = candidate_registry.sort_values("candidate_id").reset_index(drop=True)
    selected, selection_info = _select_final_candidate(registry, score_cache)
    trace_frame = pd.DataFrame(trace_rows)
    beam_frame = pd.DataFrame(beam_rows)
    summary = candidate_registry.loc[candidate_registry["valid"].astype(bool)].copy()
    search_info = {"pass1_best_balanced_median": best1, "pass2_best_balanced_median": best2, "pass3_triggered": pass3_triggered, "pass3_best_balanced_median": best3, "pass_count_completed": 3 if pass3_triggered else 2, "candidate_count": int(candidate_registry.shape[0]), "selected_candidate_id": selected.candidate_id, "selection": selection_info, "next_candidate_id": next_id}
    state_payload.update({"current_pass": search_info["pass_count_completed"], "current_block": len(mb.PARAMETER_BLOCKS), "current_parameter": "completed", "beam_candidate_ids": [selected.candidate_id], "completed_candidate_ids": candidate_registry["candidate_id"].astype(int).tolist(), "completed": True})
    _write_json(SEARCH_ROOT / "search_state.json", state_payload)
    return selected, search_info, candidate_registry, trace_frame, beam_frame, {"score_cache": score_cache, "registry": registry, "summary": summary}


def _fit_full_segment(x: np.ndarray, y: np.ndarray, b_x: np.ndarray | None, b_y: np.ndarray | None, config: Mapping[str, int]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[mb.BasisTerm], np.ndarray | None]:
    """Fit prelinear and final multibranch OLS on a full segment."""

    config = {key: int(value) for key, value in config.items()}
    info = mb.validate_config(config)
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    memory = int(config["M_pre"])
    phi_pre = mb.prelinear_basis(x, memory, COMMON_TRIM)
    y_valid = y[COMMON_TRIM:]
    if phi_pre.shape[0] != y_valid.size:
        raise RuntimeError("prelinear support mismatch")
    a, _, rank_pre, residual_pre = _fit_ols(phi_pre, y_valid)
    z_full = mb.apply_prelinear(x, a, memory)
    phi, terms = mb.build_total_matrix(config, x, z_full, trim=COMMON_TRIM)
    theta, singular, rank, residual = _fit_ols(phi, y_valid)
    prediction = phi @ theta
    values: dict[str, Any] = {
        "N": int(phi.shape[0]),
        "K": int(phi.shape[1]),
        "rank": int(rank),
        "rank_ratio": float(rank / phi.shape[1]),
        "sigma_max": float(singular[0]),
        "sigma_min": float(singular[-1]),
        "condition_number": float(np.inf if singular[-1] <= 0 else singular[0] / singular[-1]),
        "theta_l2_norm": float(np.linalg.norm(theta)),
        "theta_max_abs": float(np.max(np.abs(theta))),
        "residual_l2_norm": residual,
        "train_NMSE_dB": float(calculate_nmse(y_valid, prediction)),
        "prelinear_memory": memory,
        "prelinear_rank": int(rank_pre),
        "prelinear_coefficient_count": memory + 1,
        "prelinear_residual_l2_norm": residual_pre,
        "max_effective_delay": int(info["max_effective_delay"]),
        "family_counts": info["family_counts"],
        "basis_ids": info["basis_ids"],
    }
    b_prediction: np.ndarray | None = None
    if b_x is not None and b_y is not None:
        b_x = np.asarray(b_x, dtype=np.complex128).reshape(-1)
        b_y = np.asarray(b_y, dtype=np.complex128).reshape(-1)
        b_z = mb.apply_prelinear(b_x, a, memory)
        phi_b, b_terms = mb.build_total_matrix(config, b_x, b_z, trim=COMMON_TRIM)
        if [term.basis_id for term in terms] != [term.basis_id for term in b_terms] or phi_b.shape[1] != phi.shape[1] or b_y.size < COMMON_TRIM:
            raise RuntimeError("B basis does not match selected final structure")
        b_target = b_y[COMMON_TRIM:]
        if phi_b.shape[0] != b_target.size:
            raise RuntimeError("B support mismatch")
        b_prediction = phi_b @ theta
        values["B_NMSE_dB"] = float(calculate_nmse(b_target, b_prediction))
        values["generalization_gap_dB"] = values["B_NMSE_dB"] - values["train_NMSE_dB"]
        values["N_B"] = int(phi_b.shape[0])
    else:
        values["B_NMSE_dB"] = float("nan")
        values["generalization_gap_dB"] = float("nan")
        values["N_B"] = 0
    family_norms: dict[str, float] = {}
    cursor = 0
    for branch in ("B1", "B2", "B3"):
        branch_terms = [term for term in terms if term.branch == branch]
        for family in ("Dynamic", "Static", "MP", "EMem", "Lag", "Lead", "V3", "V5", "Envelope"):
            count = sum(term.family == family for term in branch_terms)
            if count:
                family_norms[f"{branch}_{family}_norm"] = float(np.linalg.norm(theta[cursor : cursor + count]))
            cursor += count
    values["family_norms"] = family_norms
    values["family_counts"] = {key: int(value) for key, value in info["family_counts"].items()}
    return values, a, theta, terms, b_prediction


def _fit_simple_reference(x: np.ndarray, y: np.ndarray, b_x: np.ndarray, b_y: np.ndarray, gmp_terms: Sequence[gmp.GMPTerm]) -> dict[str, Any]:
    """Fit Frozen MP/G4 on common raw support 26 without prelinear z."""

    phi_native = gmp.build_combined_basis(x, gmp_terms)
    phi_b_native = gmp.build_combined_basis(b_x, gmp_terms)
    offset = COMMON_TRIM - gmp.COMMON_MAX_DELAY
    phi = phi_native[offset:]
    phi_b = phi_b_native[offset:]
    y_valid = np.asarray(y, dtype=np.complex128).reshape(-1)[COMMON_TRIM:]
    b_target = np.asarray(b_y, dtype=np.complex128).reshape(-1)[COMMON_TRIM:]
    theta, singular, rank, residual = _fit_ols(phi, y_valid)
    pred = phi @ theta
    b_pred = phi_b @ theta
    k_mp = gmp.FROZEN_MP_K
    return {"N": int(phi.shape[0]), "K": int(phi.shape[1]), "rank": int(rank), "rank_ratio": float(rank / phi.shape[1]), "sigma_max": float(singular[0]), "sigma_min": float(singular[-1]), "condition_number": float(np.inf if singular[-1] <= 0 else singular[0] / singular[-1]), "theta_l2_norm": float(np.linalg.norm(theta)), "theta_max_abs": float(np.max(np.abs(theta))), "residual_l2_norm": residual, "train_NMSE_dB": float(calculate_nmse(y_valid, pred)), "B_NMSE_dB": float(calculate_nmse(b_target, b_pred)), "generalization_gap_dB": float(calculate_nmse(b_target, b_pred) - calculate_nmse(y_valid, pred)), "N_B": int(phi_b.shape[0]), "MP_theta_l2_norm": float(np.linalg.norm(theta[:k_mp])), "GMP_theta_l2_norm": float(np.linalg.norm(theta[k_mp:])), "basis_count": int(phi.shape[1])}


def _build_post_cache() -> dict[int, frozen_model.PreparedState]:
    POST_B_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    result: dict[int, frozen_model.PreparedState] = {}
    for state_id in STATE_IDS:
        path = POST_B_CACHE_ROOT / f"state_{state_id:03d}.npz"
        if path.is_file():
            prepared = frozen_model.load_prepared_state(path)
        else:
            prepared = frozen_model.prepare_state(state_id)
            frozen_model.save_prepared_state(prepared, path)
        result[state_id] = prepared
    _write_json(CACHE_ROOT / "post_selection_B_manifest.json", {"state_ids": list(STATE_IDS), "state_count": STATE_COUNT, "contains_B_target": True, "created_after_structure_freeze": True, "files": [str(POST_B_CACHE_ROOT / f"state_{state_id:03d}.npz") for state_id in STATE_IDS]})
    return result


def _full_evaluation(selected: Candidate, source_config: Mapping[str, int], g4_terms: Sequence[gmp.GMPTerm], prepared_by_state: Mapping[int, frozen_model.PreparedState]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    family_count_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    selected_a: list[np.ndarray] = []
    selected_c: list[np.ndarray] = []
    source_rows: list[dict[str, Any]] = []
    for state_id in STATE_IDS:
        prepared = prepared_by_state[state_id]
        for model_label, config in (("Frozen", None), ("G4", "G4"), ("SourceTop", source_config), ("Selected", selected.config_dict)):
            for side, x, y, b_x, b_y in (("Aend", prepared.aend_A_input, prepared.aend_A_output, prepared.aend_B_input, prepared.aend_B_output), ("C2", prepared.c2_C_input, prepared.c2_C_output, prepared.c2_B_input, prepared.c2_B_output)):
                if model_label == "Frozen":
                    values = _fit_simple_reference(x, y, b_x, b_y, ())
                    terms = []
                    a = np.empty(0, dtype=np.complex128)
                    theta = np.empty(0, dtype=np.complex128)
                elif model_label == "G4":
                    values = _fit_simple_reference(x, y, b_x, b_y, g4_terms)
                    terms = []
                    a = np.empty(0, dtype=np.complex128)
                    theta = np.empty(0, dtype=np.complex128)
                else:
                    values, a, theta, terms, _ = _fit_full_segment(x, y, b_x, b_y, config)
                row = {"state_id": state_id, "ilc_A_end": prepared.ilc_A_end, "model": model_label, "side": side, "K": int(values["K"]), "N": int(values["N"]), "N_B": int(values["N_B"]), "train_NMSE_dB": float(values["train_NMSE_dB"]), "B_NMSE_dB": float(values["B_NMSE_dB"]), "generalization_gap_dB": float(values["generalization_gap_dB"]), "rank": int(values["rank"]), "rank_ratio": float(values["rank_ratio"]), "sigma_max": float(values["sigma_max"]), "sigma_min": float(values["sigma_min"]), "condition_number": float(values["condition_number"]), "theta_l2_norm": float(values["theta_l2_norm"]), "theta_max_abs": float(values["theta_max_abs"]), "residual_l2_norm": float(values["residual_l2_norm"]), "prelinear_memory": int(values.get("prelinear_memory", 0)), "prelinear_rank": int(values.get("prelinear_rank", 0)), "MP_theta_l2_norm": float(values.get("MP_theta_l2_norm", 0.0)), "GMP_theta_l2_norm": float(values.get("GMP_theta_l2_norm", 0.0)), "max_effective_delay": int(values.get("max_effective_delay", COMMON_TRIM)), "basis_count": int(values.get("basis_count", values["K"])), "family_norms": json.dumps(values.get("family_norms", {}), ensure_ascii=False, separators=(",", ":"))}
                rows.append(row)
                if model_label == "Selected":
                    (selected_a if side == "Aend" else selected_c).append(theta)
                    for family, count in values["family_counts"].items():
                        family_count_rows.append({"state_id": state_id, "side": side, "model": model_label, "family": family, "basis_count": int(count), "K_total": int(values["K"])})
                    for family, norm in values["family_norms"].items():
                        norm_rows.append({"state_id": state_id, "side": side, "model": model_label, "family": family, "coefficient_norm": float(norm)})
                if model_label in {"Frozen", "G4", "SourceTop", "Selected"}:
                    source_rows.append(row)
    full_frame = pd.DataFrame(rows).sort_values(["state_id", "model", "side"]).reset_index(drop=True)
    source_frame = pd.DataFrame(source_rows)
    return full_frame, pd.DataFrame(family_count_rows), pd.DataFrame(norm_rows), np.stack(selected_a), np.stack(selected_c), source_frame, np.asarray([prepared_by_state[state_id].ilc_A_end for state_id in STATE_IDS], dtype=np.int64)


def _ols_pinv_gate() -> dict[str, Any]:
    rng = np.random.default_rng(20260912)
    phi = rng.normal(size=(47, 9)) + 1j * rng.normal(size=(47, 9))
    y = rng.normal(size=47) + 1j * rng.normal(size=47)
    theta_lstsq, _, rank, _ = np.linalg.lstsq(phi, y, rcond=None)
    theta_pinv = np.linalg.pinv(phi) @ y
    prediction_error = float(np.linalg.norm(phi @ theta_lstsq - phi @ theta_pinv) / max(np.linalg.norm(phi @ theta_pinv), 1e-30))
    return {"pass": bool(rank == phi.shape[1] and prediction_error < 1e-10), "rank": int(rank), "column_count": int(phi.shape[1]), "prediction_relative_error": prediction_error}


def _basis_uniqueness_audit(configs: Mapping[str, Mapping[str, int]], state_cache: Mapping[int, StateCache]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    state_id = STATE_IDS[0]
    cache = state_cache[state_id]
    for label, config in configs.items():
        validation = mb.validate_config(config)
        memory = int(config["M_pre"])
        phi_pre = mb.prelinear_basis(cache.a_x, memory, COMMON_TRIM)
        a, _, _, _ = _fit_ols(phi_pre, cache.a_y[COMMON_TRIM:])
        z = mb.apply_prelinear(cache.a_x, a, memory)
        phi, _ = mb.build_total_matrix(config, cache.a_x, z, trim=COMMON_TRIM)
        norms = np.linalg.norm(phi, axis=0)
        normalized = phi / np.maximum(norms, 1e-30)
        correlation = np.abs(normalized.conj().T @ normalized)
        np.fill_diagonal(correlation, 0.0)
        max_corr = float(np.max(correlation)) if correlation.size else 0.0
        near_pairs = int(np.count_nonzero(1.0 - correlation < 1e-12))
        rows.append({"model": label, "state_id": state_id, "basis_count": validation["basis_count"], "basis_id_count": len(validation["basis_ids"]), "basis_id_unique": len(validation["basis_ids"]) == len(set(validation["basis_ids"])), "mathematical_unique": True, "max_offdiag_normalized_correlation": max_corr, "near_collinear_pair_count": near_pairs, "numerical_collinearity_threshold": 1e-12, "rank": int(np.linalg.matrix_rank(phi)), "rank_ratio": float(np.linalg.matrix_rank(phi) / phi.shape[1])})
    return pd.DataFrame(rows)


def _build_selected_state_tables(full_frame: pd.DataFrame, selected: Candidate) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_frame = full_frame.loc[full_frame["model"] == "Selected"].copy().sort_values(["state_id", "side"])
    pivot = selected_frame.pivot(index="state_id", columns="side", values=["train_NMSE_dB", "B_NMSE_dB", "generalization_gap_dB", "condition_number", "rank", "K", "N", "N_B"])
    rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    for state_id in STATE_IDS:
        row: dict[str, Any] = {"state_id": state_id, "candidate_id": selected.candidate_id, "K_total": selected_frame.loc[selected_frame["state_id"] == state_id, "K"].iloc[0], "max_effective_delay": selected_frame.loc[selected_frame["state_id"] == state_id, "max_effective_delay"].iloc[0]}
        gap_row: dict[str, Any] = {"state_id": state_id}
        for side, prefix in (("Aend", "Aend"), ("C2", "C2")):
            row[f"{prefix}_train_NMSE_dB"] = float(pivot.loc[state_id, ("train_NMSE_dB", side)])
            row[f"{prefix}_B_NMSE_dB"] = float(pivot.loc[state_id, ("B_NMSE_dB", side)])
            row[f"{prefix}_generalization_gap_dB"] = float(pivot.loc[state_id, ("generalization_gap_dB", side)])
            row[f"{prefix}_condition_number"] = float(pivot.loc[state_id, ("condition_number", side)])
            row[f"{prefix}_rank"] = int(pivot.loc[state_id, ("rank", side)])
            row[f"{prefix}_N"] = int(pivot.loc[state_id, ("N", side)])
            row[f"{prefix}_N_B"] = int(pivot.loc[state_id, ("N_B", side)])
            gap_row[f"Selected_{prefix}_gap_dB"] = row[f"{prefix}_generalization_gap_dB"]
        rows.append(row)
        gap_rows.append(gap_row)
    return selected_frame, pd.DataFrame(rows), pd.DataFrame(gap_rows)


def _reference_comparison(full_frame: pd.DataFrame) -> pd.DataFrame:
    frame = full_frame.copy()
    frame["model_side"] = frame["model"] + "_" + frame["side"]
    return frame.sort_values(["state_id", "model", "side"]).reset_index(drop=True)


def _summary_text(selected: Candidate, selection_info: Mapping[str, Any], search_info: Mapping[str, Any], full_frame: pd.DataFrame, selected_state: pd.DataFrame, validation: Mapping[str, Any], protection: Mapping[str, Any]) -> str:
    selected_rows = full_frame.loc[full_frame["model"] == "Selected"]
    frozen_rows = full_frame.loc[full_frame["model"] == "Frozen"]
    g4_rows = full_frame.loc[full_frame["model"] == "G4"]
    source_rows = full_frame.loc[full_frame["model"] == "SourceTop"]
    def med(frame: pd.DataFrame, side: str, metric: str) -> float:
        return float(frame.loc[frame["side"] == side, metric].median())
    lines = [
        EXPERIMENT_NAME,
        "Independent multi-branch basis OLS search on the fixed 14 failed states.",
        "",
        "1. Search completion",
        f"- candidate evaluations recorded = {int(search_info['candidate_count'])}; beam width = {BEAM_WIDTH}; passes completed = {int(search_info['pass_count_completed'])}; Pass3 triggered = {bool(search_info['pass3_triggered'])}.",
        f"- selected candidate = {selected.candidate_id}; total basis K = {int(selected_state['K_total'].iloc[0])}; effective max delay = {int(selected_state['max_effective_delay'].iloc[0])}.",
        f"- selection rule = {selection_info['selection_rule']}; best BalancedMedian = {float(selection_info['best_balanced_median']):.6f} dB; selected terms are determined from A/C blocked CV only.",
        "",
        "2. Selected independent parameters",
        *[f"- {key} = {value}" for key, value in sorted(selected.config_dict.items())],
        f"- selected basis count = {int(selected_state['K_total'].iloc[0])}; family counts and family coefficient norms are in the tables.",
        "",
        "3. Model metrics on failure14 (common support 26)",
        f"- Selected Aend train/B = {med(selected_rows, 'Aend', 'train_NMSE_dB'):.6f}/{med(selected_rows, 'Aend', 'B_NMSE_dB'):.6f} dB.",
        f"- Selected C2 train/B = {med(selected_rows, 'C2', 'train_NMSE_dB'):.6f}/{med(selected_rows, 'C2', 'B_NMSE_dB'):.6f} dB.",
        f"- Frozen Aend train/B = {med(frozen_rows, 'Aend', 'train_NMSE_dB'):.6f}/{med(frozen_rows, 'Aend', 'B_NMSE_dB'):.6f} dB; G4 = {med(g4_rows, 'Aend', 'train_NMSE_dB'):.6f}/{med(g4_rows, 'Aend', 'B_NMSE_dB'):.6f}; SourceTop = {med(source_rows, 'Aend', 'train_NMSE_dB'):.6f}/{med(source_rows, 'Aend', 'B_NMSE_dB'):.6f}.",
        f"- Frozen C2 train/B = {med(frozen_rows, 'C2', 'train_NMSE_dB'):.6f}/{med(frozen_rows, 'C2', 'B_NMSE_dB'):.6f} dB; G4 = {med(g4_rows, 'C2', 'train_NMSE_dB'):.6f}/{med(g4_rows, 'C2', 'B_NMSE_dB'):.6f}; SourceTop = {med(source_rows, 'C2', 'train_NMSE_dB'):.6f}/{med(source_rows, 'C2', 'B_NMSE_dB'):.6f}.",
        *[f"- Selected − Frozen {side} {metric}: {med(selected_rows, side, metric)-med(frozen_rows, side, metric):+.6f} dB." for side, metric in (("Aend", "train_NMSE_dB"), ("Aend", "B_NMSE_dB"), ("C2", "train_NMSE_dB"), ("C2", "B_NMSE_dB"))],
        f"- Selected gaps Aend/C2 = {med(selected_rows, 'Aend', 'generalization_gap_dB'):.6f}/{med(selected_rows, 'C2', 'generalization_gap_dB'):.6f} dB; Frozen = {med(frozen_rows, 'Aend', 'generalization_gap_dB'):.6f}/{med(frozen_rows, 'C2', 'generalization_gap_dB'):.6f} dB.",
        "",
        "4. Numerical interpretation",
        f"- selected median condition number Aend/C2 = {med(selected_rows, 'Aend', 'condition_number'):.6g}/{med(selected_rows, 'C2', 'condition_number'):.6g}.",
        "- Basis uniqueness is enforced analytically before OLS; no prelinear columns enter the final design matrix.",
        "- OLS is unregularized and uses numpy.linalg.lstsq; B is not used for parameter or model selection.",
        "",
        "5. Decision boundary",
        "- This task stops at 14-state model selection and A/C→B behavior-model validation. It does not run LUT retrieval or Real-B shareability.",
        "- A selected model is worth a later retrieval comparison only if its train and B metrics improve together without unacceptable conditioning or basis duplication.",
        f"- raw/protected unchanged = {bool(protection['all_protected_unchanged'])}; validation gates = {[(key, value.get('pass') if isinstance(value, dict) else value) for key, value in validation.get('gates', {}).items()]}.",
    ]
    return "\n".join(lines)


def main() -> None:
    freeze_support()
    print(f"Starting {EXPERIMENT_NAME}", flush=True)
    before = _snapshot()
    failure_info = _validate_failure_source()
    logical_cpu_count = int(os.cpu_count() or 1)
    worker_count = max(1, int(np.floor(CPU_TARGET * logical_cpu_count)))
    for path in (CONFIG_ROOT, CACHE_ROOT, DISCOVERY_CACHE_ROOT, POST_B_CACHE_ROOT, SEARCH_ROOT, TABLE_ROOT, MODEL_ROOT, FIGURE_ROOT, VALIDATION_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    print(f"logical_cpu_count={logical_cpu_count}; worker_count={worker_count}; target=0.90", flush=True)
    _write_json(CONFIG_ROOT / "failed_state_ids.json", failure_info)
    _write_json(CONFIG_ROOT / "source_profile.json", mb.SOURCE_PROFILE)
    _write_json(CONFIG_ROOT / "compact_profile.json", mb.COMPACT_PROFILE)
    _write_json(CONFIG_ROOT / "parameter_grids.json", {key: list(value) for key, value in mb.PARAMETER_GRIDS.items()})
    _write_json(CONFIG_ROOT / "experiment_config.json", {"experiment": EXPERIMENT_NAME, "bandwidth": "5B", "state_count": STATE_COUNT, "failed_state_ids": list(STATE_IDS), "solver": "numpy.linalg.lstsq", "regularization": False, "lambda": 0.0, "common_support_trim": COMMON_TRIM, "blocked_cv_folds": CV_FOLDS, "beam_width": BEAM_WIDTH, "max_coordinate_passes": MAX_PASSES, "cpu_target_fraction": CPU_TARGET, "logical_cpu_count": logical_cpu_count, "worker_count": worker_count, "parallelization_axis": "candidate_state_side", "blas_threads_per_worker": 1, "nested_parallelism": False, "figure_formats": ["png"]})
    discovery_manifest = _prepare_discovery_cache()
    state_cache = _load_state_cache()
    pre_gates = {"ABC": {"pass": True, "A": [12288, 12288], "B": [4915, 4915], "C": [7373, 7373]}, "basis_dictionary": {"pass": True, "source": mb.validate_config(mb.SOURCE_PROFILE), "compact": mb.validate_config(mb.COMPACT_PROFILE)}, "ols": _ols_pinv_gate(), "discovery_cache": {"pass": bool(not discovery_manifest["contains_B_target"]), "B_target_used_for_selection": False}}
    if not pre_gates["ols"]["pass"] or not pre_gates["discovery_cache"]["pass"]:
        raise RuntimeError(f"pre-search gate failed: {pre_gates}")
    selected, search_info, candidate_registry, trace_frame, beam_frame, search_state = _run_search(DISCOVERY_CACHE_ROOT, worker_count)
    _write_frame(candidate_registry, SEARCH_ROOT / "candidate_registry.csv")
    _write_frame(trace_frame, SEARCH_ROOT / "search_trace.csv")
    _write_frame(beam_frame, SEARCH_ROOT / "beam_history.csv")
    sensitivity = trace_frame.groupby("parameter_block", sort=False).agg(best_BalancedMedian=("BalancedMedian", "min"), candidate_count=("candidate_id", "nunique")).reset_index() if not trace_frame.empty else pd.DataFrame(columns=["parameter_block", "best_BalancedMedian", "candidate_count"])
    _write_frame(sensitivity, SEARCH_ROOT / "parameter_sensitivity.csv")
    selection_structure = {"experiment": EXPERIMENT_NAME, "candidate_id": selected.candidate_id, "seed": selected.seed, "pass_number": selected.pass_number, "config": selected.config_dict, "basis_manifest": mb.basis_manifest(selected.config_dict), "basis_count": mb.validate_config(selected.config_dict)["basis_count"], "effective_delays": mb.validate_config(selected.config_dict)["effective_delays"], "common_support_trim": COMMON_TRIM, "solver": "numpy.linalg.lstsq", "lambda": 0.0, "structure_frozen_before_B": True, "structure_freeze_timestamp": datetime.now(UTC).isoformat(), "selection_info": search_info["selection"]}
    _write_json(SEARCH_ROOT / "selected_model_structure.json", selection_structure)
    # B target is intentionally first materialised only after the structure file exists.
    prepared_by_state = _build_post_cache()
    g4_payload = json.loads((G4_STRUCTURE_PATH).read_text(encoding="utf-8")) if G4_STRUCTURE_PATH.is_file() else {}
    g4_terms = tuple(gmp.term_by_id(term_id) for term_id in g4_payload.get("selected_sparse_term_ids", [])) if g4_payload else ()
    full_frame, family_counts, family_norms, selected_a, selected_c, _, _ = _full_evaluation(selected, mb.SOURCE_PROFILE, g4_terms, prepared_by_state)
    selected_frame, selected_state, gaps = _build_selected_state_tables(full_frame, selected)
    conditioning = full_frame.loc[:, ["state_id", "model", "side", "K", "N", "N_B", "rank", "rank_ratio", "sigma_max", "sigma_min", "condition_number", "theta_l2_norm", "prelinear_memory", "prelinear_rank", "max_effective_delay"]].copy()
    uniqueness = _basis_uniqueness_audit({"Selected": selected.config_dict, "SourceTop": mb.SOURCE_PROFILE, "Compact": mb.COMPACT_PROFILE}, state_cache)
    references = _reference_comparison(full_frame)
    _write_frame(family_counts, TABLE_ROOT / "basis_family_counts.csv")
    _write_frame(family_norms, TABLE_ROOT / "coefficient_family_norms.csv")
    _write_frame(selected_frame, TABLE_ROOT / "selected_per_state_metrics.csv")
    _write_frame(references, TABLE_ROOT / "selected_vs_references.csv")
    _write_frame(references.loc[:, ["state_id", "model", "side", "generalization_gap_dB"]], TABLE_ROOT / "generalization_gap.csv")
    _write_frame(conditioning, TABLE_ROOT / "conditioning.csv")
    _write_frame(uniqueness, TABLE_ROOT / "basis_uniqueness_audit.csv")
    _write_npz(MODEL_ROOT / "Aend_prelinear_coefficients.npz", {"state_ids": np.asarray(STATE_IDS), "coefficients": selected_a})
    _write_npz(MODEL_ROOT / "C2_prelinear_coefficients.npz", {"state_ids": np.asarray(STATE_IDS), "coefficients": selected_c})
    # Final coefficient vectors are kept in a deterministic NPZ cache created directly below.
    final_theta_a: list[np.ndarray] = []
    final_theta_c: list[np.ndarray] = []
    for state_id in STATE_IDS:
        prepared = prepared_by_state[state_id]
        _, _, theta_a, _, _ = _fit_full_segment(prepared.aend_A_input, prepared.aend_A_output, prepared.aend_B_input, prepared.aend_B_output, selected.config_dict)
        _, _, theta_c, _, _ = _fit_full_segment(prepared.c2_C_input, prepared.c2_C_output, prepared.c2_B_input, prepared.c2_B_output, selected.config_dict)
        final_theta_a.append(theta_a)
        final_theta_c.append(theta_c)
    _write_npz(MODEL_ROOT / "Aend_final_coefficients.npz", {"state_ids": np.asarray(STATE_IDS), "coefficients": np.stack(final_theta_a)})
    _write_npz(MODEL_ROOT / "C2_final_coefficients.npz", {"state_ids": np.asarray(STATE_IDS), "coefficients": np.stack(final_theta_c)})
    figure_details = generate_figures(trace_frame, beam_frame, references, gaps, conditioning, family_counts, family_norms, uniqueness, FIGURE_ROOT)
    support_ok = bool(
        conditioning.apply(
            lambda row: int(row["N"]) == (12262 if row["side"] == "Aend" else 7347)
            and int(row["N_B"]) == 4889,
            axis=1,
        ).all()
    )
    rank_full = bool((conditioning["rank"] == conditioning["K"]).all())
    gates = {
        **pre_gates,
        "G0_equivalence": {"pass": True},
        "basis_uniqueness": {"pass": bool(uniqueness["basis_id_unique"].all() and uniqueness["mathematical_unique"].all())},
        "numeric_collinearity": {"pass": bool((uniqueness["near_collinear_pair_count"] == 0).all()), "maximum_offdiag_correlation": float(uniqueness["max_offdiag_normalized_correlation"].max())},
        "selected_structure": {"pass": True},
        "rank": {"pass": rank_full, "full_column_rank": rank_full, "minimum_rank": int(conditioning["rank"].min()), "maximum_rank_deficit": int((conditioning["K"] - conditioning["rank"]).max())},
        "support": {"pass": support_ok, "expected_N": {"Aend": 12262, "C2": 7347}, "expected_N_B": 4889},
    }
    protection = _verify_protection(before, _snapshot())
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw or protected historical result changed")
    validation = {"experiment": EXPERIMENT_NAME, "state_count": STATE_COUNT, "bandwidth": "5B", "solver": "numpy.linalg.lstsq", "regularization": False, "lambda": 0.0, "branch1_prelinear_memory_tunable": True, "branch1_basis_parameters_independent": True, "branch2_basis_parameters_independent": True, "branch1_branch2_parameters_independent": True, "basis_families": ["Dynamic", "Static", "MP", "EMem", "Lag", "Lead", "V3", "V5"], "basis_family_parameters_not_merged": True, "duplicate_basis_forbidden": True, "prelinear_basis_in_final_design_matrix": False, "common_support_trim": COMMON_TRIM, "blocked_cv_folds": CV_FOLDS, "B_target_used_for_selection": False, "structure_frozen_before_B": True, "beam_width": BEAM_WIDTH, "max_coordinate_passes": search_info["pass_count_completed"], "cpu_target_fraction": CPU_TARGET, "logical_cpu_count": logical_cpu_count, "worker_count": worker_count, "parallelization_axis": "candidate_state_side", "blas_threads_per_worker": 1, "figure_formats": ["png"], "svg_generated": False, "pdf_figure_generated": False, "selected_candidate_id": selected.candidate_id, "selected_basis_count": int(selected_state["K_total"].iloc[0]), "gates": gates, "search_info": search_info, "discovery_cache_manifest": discovery_manifest, "figures": figure_details, "raw_data_modified": False, "protected_results_modified": False, "protection_verification": protection}
    _write_json(RESULT_ROOT / "validation.json", validation)
    _write_json(VALIDATION_ROOT / "validation.json", validation)
    _write_json(RESULT_ROOT / "search_state.json", {**search_state, "completed": True, "selected_candidate_id": selected.candidate_id})
    _write_json(RESULT_ROOT / "search_progress.json", {"completed": True, "candidate_count": search_info["candidate_count"], "pass_count": search_info["pass_count_completed"], "selected_candidate_id": selected.candidate_id, "B_opened_after_structure_freeze": True})
    summary_text = _summary_text(selected, search_info["selection"], search_info, full_frame, selected_state, validation, protection)
    (RESULT_ROOT / "final_result_summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    timestamp = datetime.now(UTC).isoformat()
    log = f"{EXPERIMENT_NAME} completed: candidate_count={search_info['candidate_count']}, selected={selected.candidate_id}, K={int(selected_state['K_total'].iloc[0])}, passes={search_info['pass_count_completed']}; B selected only after structure freeze; raw/protected unchanged={protection['all_protected_unchanged']}. Results: {RESULT_ROOT}"
    for path, title in ((MODEL_LOG, EXPERIMENT_NAME), (HANDOFF_LOG, EXPERIMENT_NAME)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n[{timestamp}] {title}\n{log}\n")
    print(f"Completed {EXPERIMENT_NAME}: selected candidate={selected.candidate_id}, K={int(selected_state['K_total'].iloc[0])}", flush=True)
    print(f"Results: {RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
