"""Run the formal Scenario 2 / 5B shareability-oriented support search.

The task writes numerical runtime/checkpoint material only under work_logs.
The final results directory is reserved for PNG/XLSX artifacts.
"""

# Task paths and audit messages intentionally contain long exact strings.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from behavior_fingerprint_ranking_consistency.shared.distance_matrix import (
    compute_cnmse_distance_matrix,
)
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ridge
from behavior_modeling.shared.volterra_terms import build_basis_columns, build_candidate_dictionary
from data_management.shared import get_state_info, load_by_id
from signal_segmentation.shared import build_partition_from_xin, get_ilc_pair, preprocess_full_pair

from retrieval_oriented_model_selection.shared.shareability_parallel import (
    CandidateEvaluation,
    SpawnEvaluator,
    WorkerReferences,
    get_worker_context,
)
from retrieval_oriented_model_selection.shared.shareability_persistence import (
    load_screening_artifact,
    save_screening_artifact,
)
from retrieval_oriented_model_selection.shared.shareability_screening import (
    ParentScreeningScores,
    candidate_pool_fingerprint,
    screening_policy_fingerprint,
)
from retrieval_oriented_model_selection.shared.shareability_search import dictionary_fingerprint
from retrieval_oriented_model_selection.shared.shareability_structure import ScreenedChildProposal
from retrieval_oriented_model_selection.shared.shareability_sufficient_stats import support_id
from retrieval_oriented_model_selection.shared.shareability_types import CandidateModelSpec

from .task_config import (
    BEAM_WIDTH,
    CHECKPOINT_ROOT,
    COMMON_B_SOURCE,
    DICTIONARY,
    DMAX_GLOBAL,
    K_MAX,
    LOG_ROOT,
    OBSERVATION,
    PARALLEL,
    RAW_MANIFEST,
    REAL_B_SOURCE,
    RIDGE_GRID,
    RUNTIME_ROOT,
    SCREENING,
    SCREENING_ROOT,
    SEED_TERM,
    STATE_COUNT,
    STRUCTURE,
    SWAP_ROUNDS,
    TASK_NAME,
    WAVEFORM_LENGTH,
)

REAL_B_THRESHOLD_DB = -40.0

TASK_CONTEXT_PATH = RUNTIME_ROOT / "worker_context.json"
K1_CHECKPOINT = CHECKPOINT_ROOT / "k1_search_checkpoint.json"
SEARCH_TRACE_PATH = LOG_ROOT / "search_trace.jsonl"
EXECUTION_LOG = LOG_ROOT / "execution_log.txt"
DATASET_FILES = {
    "x_aend": RUNTIME_ROOT / "x_aend_5b.npy",
    "y_aend": RUNTIME_ROOT / "y_aend_5b.npy",
    "x_c2": RUNTIME_ROOT / "x_c2_5b.npy",
    "y_c2": RUNTIME_ROOT / "y_c2_5b.npy",
    "common_b": RUNTIME_ROOT / "common_b_5b.npy",
}
REAL_B_SHA = ""
_WORKER_DATA: dict[str, Any] | None = None


def validated_science_fingerprint() -> dict[str, Any]:
    """Return the exact-evaluation fingerprint, independent of search policy."""
    if _raw_manifest() != RAW_MANIFEST:
        raise RuntimeError("raw manifest changed; refusing cached runtime")
    context = json.loads(TASK_CONTEXT_PATH.read_text(encoding="utf-8"))
    if (
        context.get("state_count") != STATE_COUNT
        or context.get("dmax_global") != 4
        or context.get("observation") != OBSERVATION.to_dict()
        or context.get("real_b_sha256") != _sha256(REAL_B_SOURCE)
    ):
        raise RuntimeError("runtime context scientific inputs have changed")
    expected = {
        "x_aend": (STATE_COUNT, 12288),
        "y_aend": (STATE_COUNT, 12288),
        "x_c2": (STATE_COUNT, 7373),
        "y_c2": (STATE_COUNT, 7373),
        "common_b": (4915,),
    }
    for name, shape in expected.items():
        path = DATASET_FILES[name]
        if (
            context["dataset_files"].get(name) != str(path)
            or np.load(path, mmap_mode="r").shape != shape
        ):
            raise RuntimeError(f"cached segment shape/path mismatch: {name}")
    if json.loads((RUNTIME_ROOT / "raw_manifest.json").read_text(encoding="utf-8")) != RAW_MANIFEST:
        raise RuntimeError("runtime raw manifest mismatch")
    sources = [
        Path(__file__).resolve().parents[2] / "shared" / "shareability_sufficient_stats.py",
        Path(__file__).resolve().parents[2] / "shared" / "shareability_optimized_evaluator.py",
        Path(__file__).resolve().parents[3] / "behavior_modeling" / "shared" / "volterra_terms.py",
        Path(__file__).resolve().parents[3]
        / "behavior_modeling"
        / "shared"
        / "basis_function_selection"
        / "model_solver.py",
        Path(__file__).resolve().parents[3]
        / "behavior_fingerprint_ranking_consistency"
        / "shared"
        / "distance_matrix.py",
        Path(__file__).resolve().parents[3] / "signal_segmentation" / "shared" / "__init__.py",
    ]
    fingerprint_fields = {
        "version": "fixed-seed-exact-evaluation-v2",
        "raw_manifest": RAW_MANIFEST,
        "scenario": 2,
        "bandwidth": OBSERVATION.to_dict(),
        "boundaries": {"A": [0, 12288], "B": [12288, 17203], "C": [17203, 24576]},
        "dmax": 4,
        "segment_local_history": True,
        "effective_rows": {"A": 12284, "B": 4911, "C": 7369},
        "state_map_sha256": _sha256(Path(context["state_map_path"])),
        "common_b_source_sha256": _sha256(COMMON_B_SOURCE),
        "runtime_common_b_sha256": _sha256(DATASET_FILES["common_b"]),
        "real_b_sha256": _sha256(REAL_B_SOURCE),
        "oracle_threshold_db": REAL_B_THRESHOLD_DB,
        "oracle_strict": True,
        "dictionary": context["candidate_dictionary"],
        "dictionary_sha256": dictionary_fingerprint(
            sorted(build_candidate_dictionary(orders=(1, 3, 5, 7, 9, 11), max_delay=4))
        ),
        "seed": {
            "id": SEED_TERM.basis_id,
            "order": SEED_TERM.nonlinear_order,
            "u": SEED_TERM.nonconjugate_delays,
            "c": SEED_TERM.conjugate_delays,
        },
        "structure_ridge_lambda": STRUCTURE.ridge_lambda,
        "observation_sources": {
            str(p.relative_to(Path(__file__).resolve().parents[3])): _sha256(p) for p in sources
        },
        "segment_definitions": {"lut": "Y-Aend", "query": "Y-C2", "probe": "common-B"},
    }
    digest = hashlib.sha256(json.dumps(fingerprint_fields, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "fields": fingerprint_fields}


def validated_search_policy_fingerprint(search_mode: str = "screened") -> dict[str, Any]:
    """Fingerprint checkpoint/screening state without poisoning exact score reuse."""
    exact = validated_science_fingerprint()
    screening_source = Path(__file__).resolve().parents[2] / "shared" / "shareability_screening.py"
    implementation_sha = _sha256(screening_source)
    policy = screening_policy_fingerprint(
        exact["sha256"],
        SCREENING,
        search_mode=search_mode,
        implementation_fingerprint=implementation_sha,
    )
    return {
        "exact_evaluation_fingerprint": exact["sha256"],
        "search_policy_fingerprint": policy,
        "search_mode": search_mode,
        "screening": SCREENING.to_dict() if search_mode == "screened" else None,
        "screening_implementation_sha256": implementation_sha,
    }


def persist_screening_scores(
    scores: ParentScreeningScores,
    *,
    search_mode: str = "screened",
    replace: bool = False,
) -> tuple[Path, Path]:
    """Persist proposal data only; it is not a CandidateScore or scientific result."""
    policy = validated_search_policy_fingerprint(search_mode)
    entries = scores.shortlist(SCREENING, DICTIONARY)
    valid = np.isfinite(scores.scores)
    ranked = sorted(
        (
            (basis, float(score))
            for basis, score in zip(scores.candidate_ids, scores.scores, strict=True)
            if np.isfinite(score)
        ),
        key=lambda item: (-item[1], item[0]),
    )
    ranks = {basis: index + 1 for index, (basis, _) in enumerate(ranked)}
    source = {entry.basis_id: entry.proposal_source for entry in entries}
    pool_fingerprint = candidate_pool_fingerprint(
        [basis for basis, include in zip(scores.candidate_ids, valid, strict=True) if include]
    )
    metadata = {
        **policy,
        "parent_support": list(scores.parent_support),
        "parent_support_id": support_id(scores.parent_support),
        "candidate_pool_fingerprint": pool_fingerprint,
        "screening_spec_fingerprint": SCREENING.fingerprint,
        "candidate_count": int(np.count_nonzero(valid)),
        "artifact_kind": "ScreeningScore",
    }
    stem = (
        f"k{len(scores.parent_support) + 1:02d}_parent_{support_id(scores.parent_support)}"
        f"_pool_{pool_fingerprint[:12]}_policy_{policy['search_policy_fingerprint'][:12]}_scores"
    )
    arrays = {
        "basis_ids": np.asarray(
            [basis for basis, include in zip(scores.candidate_ids, valid, strict=True) if include]
        ),
        "scores": scores.scores[valid],
        "aend_scores": scores.aend_scores[valid],
        "c2_scores": scores.c2_scores[valid],
        "screening_ranks": np.asarray(
            [
                ranks[basis]
                for basis, include in zip(scores.candidate_ids, valid, strict=True)
                if include
            ],
            dtype=np.int64,
        ),
        "stratum_codes": np.asarray(
            [stratum for stratum, include in zip(scores.strata, valid, strict=True) if include]
        ),
        "proposal_source": np.asarray(
            [
                source.get(basis, "not_selected")
                for basis, include in zip(scores.candidate_ids, valid, strict=True)
                if include
            ]
        ),
        "selected_exploit": np.asarray(
            [entry.basis_id for entry in entries if entry.proposal_source == "exploit"]
        ),
        "selected_explore": np.asarray(
            [entry.basis_id for entry in entries if entry.proposal_source == "exploration"]
        ),
    }
    data_path = SCREENING_ROOT / f"{stem}.npz"
    metadata_path = SCREENING_ROOT / f"{stem}.json"
    if data_path.exists() or metadata_path.exists():
        if replace:
            return save_screening_artifact(SCREENING_ROOT, stem, metadata, arrays, replace=True)
        load_screening_artifact(SCREENING_ROOT, stem, metadata)
        return data_path, metadata_path
    return save_screening_artifact(SCREENING_ROOT, stem, metadata, arrays)


def _log(message: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with EXECUTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message.rstrip()}\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
    raw = Path(__file__).resolve().parents[4] / "data" / "raw"
    files = sorted(p for p in raw.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    total = 0
    mats = 0
    for path in files:
        size = path.stat().st_size
        digest.update(path.relative_to(raw).as_posix().encode() + b"\0")
        digest.update(size.to_bytes(8, "little"))
        total += size
        mats += path.suffix.lower() == ".mat"
    return {
        "file_count": len(files),
        "mat_count": mats,
        "bytes": total,
        "sha256": digest.hexdigest(),
    }


def _load_common_b() -> np.ndarray:
    with np.load(COMMON_B_SOURCE, allow_pickle=False) as data:
        value = np.asarray(data["common_B"], dtype=np.complex128)
    if value.shape != (4915,) or not np.all(np.isfinite(value)):
        raise RuntimeError("Common-B source contract failed")
    return value


def _prepare_runtime() -> None:
    raise RuntimeError("historical runtime preparation is sealed; validate existing cache instead")
    manifest = _raw_manifest()
    if manifest != RAW_MANIFEST:
        raise RuntimeError(f"raw manifest mismatch: {manifest}")
    if not REAL_B_SOURCE.is_file() or not COMMON_B_SOURCE.is_file():
        raise FileNotFoundError("canonical Common-B or Real-B source missing")
    runtime = RUNTIME_ROOT
    runtime.mkdir(parents=True, exist_ok=True)
    shapes = {
        "x_aend": (STATE_COUNT, 12288),
        "y_aend": (STATE_COUNT, 12288),
        "x_c2": (STATE_COUNT, 7373),
        "y_c2": (STATE_COUNT, 7373),
    }
    arrays = {
        name: np.lib.format.open_memmap(path, mode="w+", dtype=np.complex128, shape=shape)
        for name, shape in shapes.items()
        for path in [DATASET_FILES[name]]
    }
    state_map = []
    for state_id in range(STATE_COUNT):
        data = load_by_id(state_id)
        xin = np.asarray(data["xin"])
        partition = build_partition_from_xin(xin)
        input_history = np.asarray(data["xin_pd_ori_ilc"])
        output_history = np.asarray(data["yout_withdpd_ori_ilc"])
        if input_history.shape != output_history.shape or input_history.shape[1] < 2:
            raise RuntimeError(f"state {state_id} ILC shape/C2 contract failed")
        a_pair = get_ilc_pair(data, input_history.shape[1] - 1)
        c_pair = get_ilc_pair(data, 1)
        canonical_a = preprocess_full_pair(
            a_pair.input_full,
            a_pair.output_raw_full,
            partition,
            pair_type="ILC",
            iteration_index=input_history.shape[1] - 1,
            input_peak_normalization_factor=a_pair.input_peak_normalization_factor,
        )
        canonical_c = preprocess_full_pair(
            c_pair.input_full,
            c_pair.output_raw_full,
            partition,
            pair_type="ILC",
            iteration_index=1,
            input_peak_normalization_factor=c_pair.input_peak_normalization_factor,
        )
        arrays["x_aend"][state_id] = canonical_a["A"].input
        arrays["y_aend"][state_id] = canonical_a["A"].output
        arrays["x_c2"][state_id] = canonical_c["C"].input
        arrays["y_c2"][state_id] = canonical_c["C"].output
        state_map.append(
            {
                "state_id": state_id,
                **get_state_info(state_id),
                "ilc_A_end": int(input_history.shape[1]),
            }
        )
        if (state_id + 1) % 25 == 0:
            _log(f"runtime preparation: {state_id + 1}/{STATE_COUNT} states")
    for array in arrays.values():
        array.flush()
    common_b = _load_common_b()
    np.save(DATASET_FILES["common_b"], common_b, allow_pickle=False)
    real_b = np.load(REAL_B_SOURCE, mmap_mode="r")
    if (
        real_b.shape != (STATE_COUNT, STATE_COUNT)
        or np.isnan(real_b).any()
        or np.isposinf(real_b).any()
    ):
        raise RuntimeError("canonical Real-B oracle shape/finite contract failed")
    global REAL_B_SHA
    REAL_B_SHA = _sha256(REAL_B_SOURCE)
    context = {
        "task": TASK_NAME,
        "state_count": STATE_COUNT,
        "waveform_length": WAVEFORM_LENGTH,
        "dmax_global": DMAX_GLOBAL,
        "observation": OBSERVATION.to_dict(),
        "dataset_files": {name: str(path) for name, path in DATASET_FILES.items()},
        "real_b_path": str(REAL_B_SOURCE),
        "real_b_sha256": REAL_B_SHA,
        "candidate_dictionary": {"orders": [1, 3, 5, 7, 9, 11], "max_delay": 4},
        "state_map_path": str(runtime / "state_map.json"),
    }
    (runtime / "state_map.json").write_text(
        json.dumps(state_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    TASK_CONTEXT_PATH.write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (runtime / "raw_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _log(f"runtime preparation complete; real_B_sha256={REAL_B_SHA}; state_count={STATE_COUNT}")


def _worker_arrays() -> dict[str, Any]:
    global _WORKER_DATA
    if _WORKER_DATA is not None:
        return _WORKER_DATA
    context = get_worker_context()
    files = context["dataset_files"]
    _WORKER_DATA = {
        "x_aend": np.load(files["x_aend"], mmap_mode="r"),
        "y_aend": np.load(files["y_aend"], mmap_mode="r"),
        "x_c2": np.load(files["x_c2"], mmap_mode="r"),
        "y_c2": np.load(files["y_c2"], mmap_mode="r"),
        "common_b": np.load(files["common_b"], mmap_mode="r"),
        "real_b": np.load(context["real_b_path"], mmap_mode="r"),
        "dictionary": build_candidate_dictionary(orders=(1, 3, 5, 7, 9, 11), max_delay=4),
    }
    return _WORKER_DATA


def evaluate_candidate(candidate: CandidateModelSpec) -> dict[str, Any]:
    data = _worker_arrays()
    dictionary = data["dictionary"]
    basis_ids = candidate.basis_ids
    n = STATE_COUNT
    if candidate.k_selected == 1:
        return _evaluate_single_basis(candidate, data, dictionary)
    theta_a = np.empty((n, candidate.k_selected), dtype=np.complex128)
    theta_c = np.empty_like(theta_a)
    for state_id in range(n):
        phi_a = build_basis_columns(
            data["x_aend"][state_id], basis_ids, dictionary, dmax=DMAX_GLOBAL
        )
        phi_c = build_basis_columns(data["x_c2"][state_id], basis_ids, dictionary, dmax=DMAX_GLOBAL)
        theta_a[state_id] = fit_ridge(
            phi_a, data["y_aend"][state_id, DMAX_GLOBAL:], candidate.ridge_lambda
        ).theta
        theta_c[state_id] = fit_ridge(
            phi_c, data["y_c2"][state_id, DMAX_GLOBAL:], candidate.ridge_lambda
        ).theta
    probe = build_basis_columns(data["common_b"], basis_ids, dictionary, dmax=DMAX_GLOBAL)
    lut = (probe @ theta_a.T).T
    query = (probe @ theta_c.T).T
    distance = compute_cnmse_distance_matrix(query, lut)
    state_ids = np.arange(n, dtype=np.int64)
    ranking = np.vstack([np.lexsort((state_ids, row)) for row in distance])
    top1 = ranking[:, 0]
    real_b = data["real_b"][state_ids, top1]
    share = real_b < REAL_B_THRESHOLD_DB
    true = state_ids

    def count(k: int) -> int:
        return int(np.count_nonzero(np.any(ranking[:, :k] == true[:, None], axis=1)))

    margin = -40.0 - real_b[np.isfinite(real_b)]
    return {
        "candidate_id": candidate.candidate_id,
        "N_shareable": int(share.sum()),
        "shareable_rate": float(share.mean()),
        "N_self": int((top1 == true).sum()),
        "N_top3": count(3),
        "N_top5": count(5),
        "N_top10": count(10),
        "mean_margin_dB": float(np.mean(margin)) if margin.size else float("nan"),
        "median_margin_dB": float(np.median(margin)) if margin.size else float("nan"),
        "p05_margin_dB": float(np.quantile(margin, 0.05)) if margin.size else float("nan"),
        "K": candidate.k_selected,
        "ridge_lambda": candidate.ridge_lambda,
        "basis_ids": list(candidate.basis_ids),
    }


def _evaluate_single_basis(
    candidate: CandidateModelSpec,
    data: dict[str, Any],
    dictionary: dict[str, Any],
) -> dict[str, Any]:
    """Equivalent scalar Ridge path for K=1, avoiding 850 SVDs and 4911-wide products."""

    basis_id = candidate.basis_ids[0]
    theta_a = np.empty(STATE_COUNT, dtype=np.complex128)
    theta_c = np.empty(STATE_COUNT, dtype=np.complex128)
    for state_id in range(STATE_COUNT):
        phi_a = build_basis_columns(
            data["x_aend"][state_id], (basis_id,), dictionary, dmax=DMAX_GLOBAL
        )[:, 0]
        phi_c = build_basis_columns(
            data["x_c2"][state_id], (basis_id,), dictionary, dmax=DMAX_GLOBAL
        )[:, 0]
        target_a = data["y_aend"][state_id, DMAX_GLOBAL:]
        target_c = data["y_c2"][state_id, DMAX_GLOBAL:]
        theta_a[state_id] = np.vdot(phi_a, target_a) / (
            np.vdot(phi_a, phi_a).real + phi_a.size * candidate.ridge_lambda
        )
        theta_c[state_id] = np.vdot(phi_c, target_c) / (
            np.vdot(phi_c, phi_c).real + phi_c.size * candidate.ridge_lambda
        )
    probe = build_basis_columns(data["common_b"], (basis_id,), dictionary, dmax=DMAX_GLOBAL)[:, 0]
    probe_energy = float(np.vdot(probe, probe).real)
    query = theta_c[:, None]
    lut = theta_a[None, :]
    query_energy = np.abs(theta_c) ** 2
    squared = np.abs(query - lut) ** 2 * probe_energy
    distance = np.full((STATE_COUNT, STATE_COUNT), np.inf, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = 10.0 * np.log10(squared / (query_energy[:, None] * probe_energy))
    distance[squared == 0.0] = -np.inf
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    ranking = np.vstack([np.lexsort((state_ids, row)) for row in distance])
    top1 = ranking[:, 0]
    real_b = data["real_b"][state_ids, top1]
    share = real_b < REAL_B_THRESHOLD_DB
    true = state_ids

    def count(k: int) -> int:
        return int(np.count_nonzero(np.any(ranking[:, :k] == true[:, None], axis=1)))

    margin = -40.0 - real_b[np.isfinite(real_b)]
    return {
        "candidate_id": candidate.candidate_id,
        "N_shareable": int(share.sum()),
        "shareable_rate": float(share.mean()),
        "N_self": int((top1 == true).sum()),
        "N_top3": count(3),
        "N_top5": count(5),
        "N_top10": count(10),
        "mean_margin_dB": float(np.mean(margin)) if margin.size else float("nan"),
        "median_margin_dB": float(np.median(margin)) if margin.size else float("nan"),
        "p05_margin_dB": float(np.quantile(margin, 0.05)) if margin.size else float("nan"),
        "K": 1,
        "ridge_lambda": candidate.ridge_lambda,
        "basis_ids": [basis_id],
        "fast_path": True,
    }


def _candidate_row(
    candidate: CandidateModelSpec,
    result: CandidateEvaluation | dict[str, Any],
    proposal: ScreenedChildProposal | None = None,
) -> dict[str, Any]:
    payload = result.metrics if isinstance(result, CandidateEvaluation) else result
    row = {"candidate_id": candidate.candidate_id, **payload}
    if proposal is not None:
        if proposal.model.candidate_id != candidate.candidate_id:
            raise ValueError("screening proposal does not match trace candidate")
        row.update(proposal.trace_fields())
    return row


def run_k1_search() -> None:
    raise RuntimeError(
        "old exhaustive K1 and multi-Ridge search is incompatible with fixed x[n] seed"
    )
    dictionary = build_candidate_dictionary(orders=(1, 3, 5, 7, 9, 11), max_delay=4)
    basis_ids = sorted(dictionary)
    candidates = [
        CandidateModelSpec(OBSERVATION, (basis_id,), ridge)
        for basis_id in basis_ids
        for ridge in RIDGE_GRID.values
    ]
    checkpoint = {}
    if K1_CHECKPOINT.exists():
        checkpoint = json.loads(K1_CHECKPOINT.read_text(encoding="utf-8"))
    start = int(checkpoint.get("next_index", 0))
    trace_mode = "a" if start else "w"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with SEARCH_TRACE_PATH.open(trace_mode, encoding="utf-8") as trace:
        refs = WorkerReferences(context_path=str(TASK_CONTEXT_PATH), oracle_path=str(REAL_B_SOURCE))
        with SpawnEvaluator(
            evaluate_candidate, parallel_spec=PARALLEL, references=refs
        ) as executor:
            batch_size = 10
            for begin in range(start, len(candidates), batch_size):
                batch = candidates[begin : begin + batch_size]
                scores = executor.map_scored(batch)
                for candidate, score in zip(batch, scores, strict=True):
                    trace.write(
                        json.dumps(_candidate_row(candidate, score), ensure_ascii=False) + "\n"
                    )
                trace.flush()
                checkpoint = {
                    "task": TASK_NAME,
                    "stage": "seed_k1",
                    "next_index": begin + len(batch),
                    "total_candidates": len(candidates),
                    "candidate_dictionary_hash": dictionary_fingerprint(basis_ids),
                    "ridge_grid": list(RIDGE_GRID.values),
                    "parallel_execution": executor.runtime_metadata,
                    "scientific_search_started": True,
                    "formal_425_state_search": True,
                }
                K1_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
                K1_CHECKPOINT.write_text(
                    json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                _log(
                    f"K1 seed progress {begin + len(batch)}/{len(candidates)}; effective_workers={len({s.worker_pid for s in scores})}"
                )
    _log("K1 seed stage complete; forward beam stages are pending implementation/continuation.")


def _read_trace_rows() -> list[dict[str, Any]]:
    if not SEARCH_TRACE_PATH.is_file():
        raise FileNotFoundError(SEARCH_TRACE_PATH)
    rows = []
    with SEARCH_TRACE_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _model_from_row(row: dict[str, Any]) -> CandidateModelSpec:
    return CandidateModelSpec(OBSERVATION, tuple(row["basis_ids"]), float(row["ridge_lambda"]))


def _candidate_neighbors(
    models: list[CandidateModelSpec],
    dictionary_ids: list[str],
    stage: str,
    ridge_values: tuple[float, ...],
) -> list[CandidateModelSpec]:
    generated: dict[str, CandidateModelSpec] = {}
    for model in models:
        support = set(model.basis_ids)
        if stage == "forward":
            supports = [
                tuple(sorted(support | {basis})) for basis in dictionary_ids if basis not in support
            ]
        elif stage == "backward":
            supports = [
                tuple(sorted(support - {basis})) for basis in sorted(support) if len(support) > 1
            ]
        elif stage == "swap":
            supports = [
                tuple(sorted((support - {removed}) | {added}))
                for removed in sorted(support)
                for added in dictionary_ids
                if added not in support
            ]
        elif stage == "ridge":
            supports = [tuple(sorted(support))]
        else:
            raise ValueError(f"unknown continuation stage: {stage}")
        for candidate_support in supports:
            for ridge_lambda in ridge_values:
                candidate = CandidateModelSpec(OBSERVATION, candidate_support, ridge_lambda)
                if candidate.k_selected <= K_MAX:
                    generated[candidate.candidate_id] = candidate
    return [generated[key] for key in sorted(generated)]


def _record_rows(
    trace: Any,
    batch: list[CandidateModelSpec],
    scores: list[CandidateEvaluation],
    stage: str,
    round_index: int,
) -> list[dict[str, Any]]:
    rows = []
    for candidate, score in zip(batch, scores, strict=True):
        row = _candidate_row(candidate, score)
        row.update(
            {
                "stage": stage,
                "search_round": round_index,
                "status": score.status,
                "worker_pid": score.worker_pid,
                "elapsed_seconds": score.elapsed_seconds,
            }
        )
        trace.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
    trace.flush()
    return rows


def _select_frontier(rows: list[dict[str, Any]], width: int = 3) -> list[CandidateModelSpec]:
    valid = [row for row in rows if row.get("status", "success") == "success"]
    valid.sort(
        key=lambda row: (
            -int(row["N_shareable"]),
            -float(row.get("median_margin_dB", float("-inf"))),
            str(row["candidate_id"]),
        )
    )
    return [_model_from_row(row) for row in valid[:width]]


def run_continuation() -> None:
    raise RuntimeError("old checkpoint topology cannot be resumed under the fixed x[n] seed")
    rows = _read_trace_rows()
    if len(rows) < 3:
        raise RuntimeError("K=1 seed trace is not ready for continuation")
    dictionary = build_candidate_dictionary(orders=(1, 3, 5, 7, 9, 11), max_delay=4)
    dictionary_ids = sorted(dictionary)
    frontier = _select_frontier(rows, BEAM_WIDTH)
    checkpoint_path = CHECKPOINT_ROOT / "formal_search_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    start_stage = checkpoint.get("stage", "forward")
    start_round = int(checkpoint.get("round", 0))
    if checkpoint.get("frontier"):
        frontier = [_model_from_row(item) for item in checkpoint["frontier"]]
    stage_order = ["forward", "backward", "swap", "ridge"]
    start_index = stage_order.index(start_stage) if start_stage in stage_order else 0
    trace_mode = "a"
    with SEARCH_TRACE_PATH.open(trace_mode, encoding="utf-8") as trace:
        refs = WorkerReferences(context_path=str(TASK_CONTEXT_PATH), oracle_path=str(REAL_B_SOURCE))
        with SpawnEvaluator(
            evaluate_candidate, parallel_spec=PARALLEL, references=refs
        ) as executor:
            for stage_index in range(start_index, len(stage_order)):
                stage = stage_order[stage_index]
                if stage == "forward":
                    current_k = max(model.k_selected for model in frontier)
                    targets = range(current_k + 1, K_MAX + 1)
                elif stage == "backward":
                    targets = (0,)
                elif stage == "swap":
                    targets = range(start_round + 1, SWAP_ROUNDS + 1)
                else:
                    targets = (0,)
                for round_index in targets:
                    if stage == "forward":
                        next_models = _candidate_neighbors(
                            frontier, dictionary_ids, stage, tuple(RIDGE_GRID.values)
                        )
                    elif stage == "backward":
                        next_models = _candidate_neighbors(
                            frontier, dictionary_ids, stage, tuple(RIDGE_GRID.values)
                        )
                    elif stage == "swap":
                        next_models = _candidate_neighbors(
                            frontier, dictionary_ids, stage, tuple(RIDGE_GRID.values)
                        )
                    else:
                        next_models = _candidate_neighbors(
                            frontier, dictionary_ids, stage, tuple(RIDGE_GRID.values)
                        )
                    if not next_models:
                        break
                    round_rows = []
                    for begin in range(0, len(next_models), 10):
                        batch = next_models[begin : begin + 10]
                        scores = executor.map_scored(batch)
                        round_rows.extend(_record_rows(trace, batch, scores, stage, round_index))
                    frontier = _select_frontier(round_rows, BEAM_WIDTH)
                    checkpoint = {
                        "task": TASK_NAME,
                        "stage": stage,
                        "round": round_index,
                        "frontier": [model.to_dict() for model in frontier],
                        "parallel_execution": executor.runtime_metadata,
                        "candidate_dictionary_hash": dictionary_fingerprint(dictionary_ids),
                    }
                    checkpoint_path.write_text(
                        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    _log(
                        f"continuation stage={stage} round={round_index} evaluated={len(round_rows)} frontier={len(frontier)}"
                    )
                    if stage == "backward" or stage == "swap" or stage == "ridge":
                        break
    _log("formal continuation complete; render_results.py and validate_results.py remain to be run")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--search-k1", action="store_true")
    parser.add_argument("--continue-search", action="store_true")
    parser.add_argument("--validate-runtime", action="store_true")
    args = parser.parse_args()
    if args.search_k1 or args.continue_search:
        parser.error(
            "historical exhaustive/multi-Ridge search is incompatible with the fixed x[n] seed; old trace and checkpoint are read-only"
        )
    if args.prepare:
        parser.error(
            "runtime preparation is disabled in this maintenance pass to preserve existing mmap/cache; use --validate-runtime"
        )
    if args.validate_runtime:
        print(json.dumps(validated_science_fingerprint(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
