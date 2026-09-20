"""Frozen configuration for the first formal 5B shareability search."""

# Task configuration intentionally keeps exact scientific path names together.
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

from behavior_modeling.shared.volterra_terms import build_candidate_dictionary
from low_bandwidth_behavior_analysis.shared import SampleRateSpec

from retrieval_oriented_model_selection.shared.shareability_types import (
    CandidateScreeningSpec,
    ObservationBandwidthSpec,
    ParallelExecutionSpec,
    RidgeGridSpec,
    StructureSearchSpec,
)

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
MODULE = "retrieval_oriented_model_selection"
ROUTE = "dpd_shareability_oriented"
TASK_NAME = "scenario_2_volterra_shareability_basis_search_5b"
TASK_ROOT = PROJECT_ROOT / "scripts" / MODULE / ROUTE / TASK_NAME
RESULT_ROOT = PROJECT_ROOT / "results" / MODULE / ROUTE / TASK_NAME
LOG_ROOT = PROJECT_ROOT / "work_logs" / MODULE / ROUTE / TASK_NAME
RUNTIME_ROOT = LOG_ROOT / "runtime"
CHECKPOINT_ROOT = LOG_ROOT / "checkpoint"
SCREENING_ROOT = LOG_ROOT / "screening"
STATE_COUNT = 425
WAVEFORM_LENGTH = 24576
P_MAX = 11
M_MAX = 4
DMAX_GLOBAL = 4
K_MAX = 20
BEAM_WIDTH = 3
SWAP_ROUNDS = 3
REAL_B_THRESHOLD_DB = -40.0
COMMON_B_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2_k18_aend_c2_full_lut_retrieval_5b"
    / "05_lut_fingerprints.npz"
)
REAL_B_SOURCE = (
    PROJECT_ROOT
    / "results"
    / MODULE
    / "self_hit_oriented"
    / "scenario_2"
    / "scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
    / "04_realB_ground_truth_cnmse_matrix.npy"
)
RAW_MANIFEST = {
    "file_count": 429,
    "mat_count": 427,
    "bytes": 2258448137,
    "sha256": "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0",
}
OBSERVATION = ObservationBandwidthSpec("5B", SampleRateSpec(50))
PARALLEL = ParallelExecutionSpec()
# Historical self-hit search uses 0 + 1e-14 ... 1e2. This task deliberately
# freezes five points spanning that range and retaining the historical 1e-8.
RIDGE_GRID = RidgeGridSpec((0.0, 1e-14, 1e-8, 1e-2, 1e2))

# RIDGE_GRID describes the preserved historical trace only. It must never
# drive the new structure stage. Dense Ridge values are intentionally unset.
DICTIONARY = build_candidate_dictionary(orders=(1, 3, 5, 7, 9, 11), max_delay=DMAX_GLOBAL)
if len(DICTIONARY) != 38335:
    raise RuntimeError(
        f"canonical Volterra dictionary has {len(DICTIONARY)} entries, expected 38335"
    )
SEED_MATCHES = [
    term
    for term in DICTIONARY.values()
    if term.nonlinear_order == 1 and term.nonconjugate_delays == (0,) and not term.conjugate_delays
]
if len(SEED_MATCHES) != 1:
    raise RuntimeError("x[n] must have exactly one canonical descriptor")
SEED_TERM = SEED_MATCHES[0]
STRUCTURE = StructureSearchSpec(
    SEED_TERM.basis_id, max_support_size=K_MAX, beam_width=BEAM_WIDTH, global_dmax=DMAX_GLOBAL
)
DENSE_RIDGE_VALUES: tuple[float, ...] = ()
SCREENING = CandidateScreeningSpec(
    shortlist_size=500,
    exploit_size=400,
    explore_size=100,
    aend_weight=0.5,
    c2_weight=0.5,
)
SCREENING_BLOCK_SIZE = 64
SEARCH_MODE_DEFAULT = "screened"

__all__ = [
    "PROJECT_ROOT",
    "TASK_NAME",
    "TASK_ROOT",
    "RESULT_ROOT",
    "LOG_ROOT",
    "RUNTIME_ROOT",
    "CHECKPOINT_ROOT",
    "SCREENING_ROOT",
    "STATE_COUNT",
    "WAVEFORM_LENGTH",
    "P_MAX",
    "M_MAX",
    "DMAX_GLOBAL",
    "K_MAX",
    "BEAM_WIDTH",
    "SWAP_ROUNDS",
    "REAL_B_THRESHOLD_DB",
    "COMMON_B_SOURCE",
    "REAL_B_SOURCE",
    "RAW_MANIFEST",
    "OBSERVATION",
    "PARALLEL",
    "RIDGE_GRID",
    "DICTIONARY",
    "SEED_TERM",
    "STRUCTURE",
    "DENSE_RIDGE_VALUES",
    "SCREENING",
    "SCREENING_BLOCK_SIZE",
    "SEARCH_MODE_DEFAULT",
]
