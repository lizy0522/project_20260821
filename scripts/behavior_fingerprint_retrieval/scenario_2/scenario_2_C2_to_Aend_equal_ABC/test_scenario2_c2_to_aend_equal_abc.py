"""Focused regression tests for the equal-length ABC ablation."""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

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

from behavior_fingerprint_retrieval.scenario_2.scenario_2_C2_to_Aend_equal_ABC.scenario2_c2_to_aend_equal_abc import (  # noqa: E402,E501
    C2_STAGE,
    DPD_SHAREABLE_THRESHOLD_DB,
    FINGERPRINT_LENGTH,
    MAX_DELAY,
    MEMORY,
    N_COMPLEX_COEFFICIENTS,
    ORDERS,
    RIDGE_LAMBDA,
    SEGMENT_LENGTH,
    STATE_COUNT,
    VALID_LENGTH,
    build_equal_abc_partition,
    compute_equal_cnmse_distance_matrix,
    segment_definition,
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_equal_ABC"
)
OLD_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_retrieval"
    / "scenario_2_C2_to_Aend"
)


def _assert_raises(exception_type: type[BaseException], function) -> None:
    try:
        function()
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def test_equal_partition_boundaries() -> None:
    partition = build_equal_abc_partition()
    assert partition.total_length == 24576
    assert (partition.a_slice.start, partition.a_slice.stop) == (0, 8192)
    assert (partition.b_slice.start, partition.b_slice.stop) == (8192, 16384)
    assert (partition.c_slice.start, partition.c_slice.stop) == (16384, 24576)
    x = np.arange(24576)
    assert x[partition.a_slice][0] == 0 and x[partition.a_slice][-1] == 8191
    assert x[partition.b_slice][0] == 8192 and x[partition.b_slice][-1] == 16383
    assert x[partition.c_slice][0] == 16384 and x[partition.c_slice][-1] == 24575


def test_segment_local_basis_does_not_borrow_history() -> None:
    partition = build_equal_abc_partition()
    full = (np.arange(24576, dtype=float) + 1j * np.arange(24576, dtype=float)).astype(
        np.complex128
    )
    a = full[partition.a_slice]
    b = full[partition.b_slice]
    c = full[partition.c_slice]
    from behavior_modeling.shared.basis import build_mp_basis

    basis_b = build_mp_basis(b, ORDERS, MEMORY)
    basis_c = build_mp_basis(c, ORDERS, MEMORY)
    assert basis_b.shape == (8190, 10)
    assert basis_c.shape == (8190, 10)
    # First valid row uses B[2], B[1], B[0], never A[-2:] or B[-2:] for C.
    assert basis_b[0, 0] == b[2]
    assert basis_b[0, 1] == b[1]
    assert basis_b[0, 2] == b[0]
    assert basis_c[0, 0] == c[2]
    assert basis_c[0, 1] == c[1]
    assert basis_c[0, 2] == c[0]
    assert a[-1] != b[0] and b[-1] != c[0]


def test_frozen_model_contract() -> None:
    assert ORDERS == (1, 2, 3, 5, 7, 9)
    assert MEMORY == {1: 3, 2: 2, 3: 2, 5: 1, 7: 1, 9: 1}
    assert RIDGE_LAMBDA == 1e-8
    assert N_COMPLEX_COEFFICIENTS == 10
    assert MAX_DELAY == 2
    assert VALID_LENGTH == 8190
    assert FINGERPRINT_LENGTH == 8190
    assert C2_STAGE == 2


def test_threshold_contract() -> None:
    values = np.asarray([-40.000001, -40.0, -39.999999, -np.inf])
    assert (values < DPD_SHAREABLE_THRESHOLD_DB).tolist() == [True, False, False, True]


def test_equal_cnmse_exact_and_shapes() -> None:
    x = np.arange(3 * VALID_LENGTH, dtype=float).reshape(3, VALID_LENGTH).astype(np.complex128)
    distance = compute_equal_cnmse_distance_matrix(x, x)
    assert distance.shape == (3, 3)
    assert distance.dtype == np.float64
    assert np.all(np.isneginf(np.diag(distance)))
    _assert_raises(ValueError, lambda: compute_equal_cnmse_distance_matrix(x[:, :10], x))


def test_segment_definition_json() -> None:
    definition = segment_definition()
    assert definition["A_length"] == SEGMENT_LENGTH
    assert definition["B_length"] == SEGMENT_LENGTH
    assert definition["C_length"] == SEGMENT_LENGTH
    assert definition["basis_is_segment_local"] is True
    assert definition["A_valid_length"] == VALID_LENGTH


def test_result_model_and_fingerprint_shapes() -> None:
    metrics = pd.read_csv(RESULT_ROOT / "equal_ABC_state_model_metrics.csv")
    assert metrics.shape[0] == STATE_COUNT
    assert metrics["state_id"].to_numpy().tolist() == list(range(STATE_COUNT))
    assert metrics["A_valid_length"].eq(8190).all()
    assert metrics["B_valid_length_Aend"].eq(8190).all()
    assert metrics["C_valid_length_C2"].eq(8190).all()
    assert metrics["real_B_valid_length"].eq(8190).all()
    with np.load(RESULT_ROOT / "equal_ABC_model_coefficients.npz", allow_pickle=False) as data:
        assert np.asarray(data["theta_Aend"]).shape == (425, 10)
        assert np.asarray(data["theta_C2"]).shape == (425, 10)
        assert np.asarray(data["theta_Aend"]).dtype == np.complex128
    with np.load(RESULT_ROOT / "equal_ABC_lut_fingerprints_Aend.npz", allow_pickle=False) as data:
        lut = np.asarray(data["Y_Aend_fingerprints"])
    with np.load(RESULT_ROOT / "equal_ABC_query_fingerprints_C2.npz", allow_pickle=False) as data:
        query = np.asarray(data["Q_C2"])
    assert lut.shape == (425, 8190) and query.shape == (425, 8190)
    assert lut.dtype == np.complex128 and query.dtype == np.complex128
    assert np.all(np.isfinite(lut)) and np.all(np.isfinite(query))


def test_result_retrieval_and_real_b_contract() -> None:
    retrieval = pd.read_csv(RESULT_ROOT / "retrieval_results.csv")
    diagnostics = pd.read_csv(RESULT_ROOT / "retrieved_real_B_cnmse.csv")
    assert retrieval.shape[0] == STATE_COUNT
    assert diagnostics.shape[0] == STATE_COUNT
    assert retrieval["State_n_R"].to_numpy().tolist() == list(range(STATE_COUNT))
    assert np.array_equal(retrieval["State_n_Q"], diagnostics["State_n_Q"])
    assert np.array_equal(retrieval["exact_hit"], diagnostics["exact_hit"])
    with np.load(RESULT_ROOT / "equal_ABC_real_B_distance_matrix.npz", allow_pickle=False) as data:
        distance = np.asarray(data["D_B"])
    with np.load(
        RESULT_ROOT / "equal_ABC_retrieval_distance_matrix.npz", allow_pickle=False
    ) as data:
        retrieval_distance = np.asarray(data["D_C2_Aend"])
    with np.load(
        RESULT_ROOT / "equal_ABC_retrieval_ranking_matrix.npz", allow_pickle=False
    ) as data:
        ranking = np.asarray(data["R_C2_Aend"])
    assert distance.shape == (425, 425)
    assert retrieval_distance.shape == (425, 425)
    assert ranking.shape == (425, 425)
    assert np.all(np.isneginf(np.diag(distance)))
    assert int(retrieval["exact_hit"].sum()) == int(diagnostics["exact_hit"].sum())
    assert int(diagnostics["dpd_shareable"].sum()) + int(diagnostics["failure"].sum()) == 425
    assert not (RESULT_ROOT / "real_B_distance_matrix_reused.npz").exists()


def test_nonexact_and_failure_outputs() -> None:
    retrieval = pd.read_csv(RESULT_ROOT / "retrieval_results.csv")
    nonexact = pd.read_csv(RESULT_ROOT / "nonexact_retrieval_states.csv")
    failures = pd.read_csv(RESULT_ROOT / "failed_retrieval_states.csv")
    assert nonexact.shape[0] == int((retrieval["State_n_R"] != retrieval["State_n_Q"]).sum())
    assert (nonexact["State_n_R"] != nonexact["State_n_Q"]).all()
    assert failures.shape[0] == int(retrieval["failure"].sum())
    assert (failures["retrieved_real_B_CNMSE_dB"] >= -40.0).all()
    assert np.isfinite(failures["retrieved_real_B_CNMSE_dB"]).all()


def test_old_stage_map_and_validation_flags() -> None:
    old_stage = pd.read_csv(OLD_ROOT / "a_end_stage_map.csv").sort_values("state_id")
    new_stage = pd.read_csv(RESULT_ROOT / "a_end_stage_map.csv").sort_values("state_id")
    assert np.array_equal(old_stage["state_id"], new_stage["state_id"])
    assert np.array_equal(old_stage["ilc_A_end"], new_stage["ilc_A_end"])
    payload = json.loads((RESULT_ROOT / "validation.json").read_text(encoding="utf-8"))
    assert payload["real_B_matrix_reused"] is False
    assert payload["Aend_stage_map_unchanged_vs_original"] is True
    assert payload["C2_available_count"] == 425
    assert payload["protected_old_results_unchanged"] is True
    assert payload["protection_verification"]["all_protected_unchanged"] is True
    assert payload["raw_data_modified"] is False


def test_excel_shape_and_exact_text() -> None:
    path = RESULT_ROOT / "scenario_2_C2_to_Aend_equal_ABC_state_summary.xlsx"
    if not path.is_file():
        return
    with zipfile.ZipFile(path) as archive:
        sheet_xml = archive.read("xl/worksheets/sheet1.xml").decode("utf-8", errors="ignore")
        sheet = ElementTree.fromstring(sheet_xml)
        rows = sheet.findall(".//{*}row")
        assert len(rows) == 426
        exact_cells = re.findall(
            r't="str"><x:v>-Inf</x:v>',
            sheet_xml,
        )
        expected_exact = int(
            pd.read_csv(RESULT_ROOT / "retrieval_summary.csv").iloc[0]["exact_hit_count"]
        )
        assert len(exact_cells) == expected_exact


def main() -> None:
    tests = [
        test_equal_partition_boundaries,
        test_segment_local_basis_does_not_borrow_history,
        test_frozen_model_contract,
        test_threshold_contract,
        test_equal_cnmse_exact_and_shapes,
        test_segment_definition_json,
        test_result_model_and_fingerprint_shapes,
        test_result_retrieval_and_real_b_contract,
        test_nonexact_and_failure_outputs,
        test_old_stage_map_and_validation_flags,
        test_excel_shape_and_exact_text,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All equal-ABC tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
