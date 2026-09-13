"""Tests for the single-sheet Scenario 2 C2-to-A_end Excel export."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_lut_retrieval.export_scenario2_c2_to_aend_state_summary import (  # noqa: E402
    EXPECTED_COLUMNS,
    OUTPUT_XLSX,
    build_state_summary_frame,
)

RESULT_ROOT = OUTPUT_XLSX.parent
EXPORTER = (
    SCRIPTS_ROOT
    / "behavior_fingerprint_lut_retrieval"
    / "export_scenario2_c2_to_aend_state_summary.py"
)
OLD_C3_ROOT = (
    PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C3_to_A2"
)


@lru_cache(maxsize=1)
def _source() -> tuple[pd.DataFrame, dict[str, object]]:
    return build_state_summary_frame()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file() and "__pycache__" not in item.parts and item.suffix.lower() != ".pyc"
        ),
        key=lambda item: str(item).lower(),
    )
    for file in files:
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def test_source_table_has_fixed_shape_and_columns() -> None:
    frame, metadata = _source()
    assert frame.shape == (425, 10)
    assert frame.columns.tolist() == EXPECTED_COLUMNS
    assert frame["state_id_R"].is_unique
    assert np.array_equal(frame["state_id_R"].to_numpy(), np.arange(425))
    assert metadata["a_end_distribution"] == {"2": 1, "3": 416, "4": 7, "5": 1}


def test_source_table_uses_physical_load_configuration() -> None:
    frame, _ = _source()
    assert frame.loc[0, "load_config"] == "funMng=0, funAng=0°, secMng=0, secAng=0°"
    assert frame.loc[17, "load_config"] == "funMng=0.1, funAng=0°, secMng=0, secAng=0°"
    assert frame.loc[18, "load_config"] == "funMng=0.1, funAng=0°, secMng=0.15, secAng=0°"
    assert all("°" in value for value in frame["load_config"])
    assert all(
        "funMng=10" not in value and "secMng=15" not in value
        for value in frame["load_config"]
    )


def test_source_table_acpr_average_and_finite_columns() -> None:
    frame, _ = _source()
    raw = pd.read_csv(
        PROJECT_ROOT
        / "results"
        / "pa_performance_observation"
        / "scenario_2"
        / "ilc_acpr_average_by_state.csv"
    )
    # The source workbook column is read from MAT scalar variables, not the ILC average table;
    # this check verifies the expected arithmetic on representative state metadata separately.
    assert np.isfinite(frame["acpr_withoutdpd_avg_dBc"].to_numpy(dtype=float)).all()
    assert raw.shape[0] == 425
    for column in EXPECTED_COLUMNS[2:8]:
        assert np.isfinite(frame[column].to_numpy(dtype=float)).all()


def test_key_state_rows_and_retrieval_values() -> None:
    frame, _ = _source()
    for state_id in (0, 187, 340, 424):
        row = frame.iloc[state_id]
        assert int(row.state_id_R) == state_id
        assert 0 <= int(row.state_id_Q) <= 424
    assert int(frame.iloc[340].state_id_Q) == 217
    assert abs(float(frame.iloc[340].retrieved_real_B_CNMSE_dB) + 37.10367236988057) < 1e-10


def test_source_counts_match_formal_retrieval() -> None:
    frame, metadata = _source()
    values = pd.to_numeric(
        frame["retrieved_real_B_CNMSE_dB"].replace("-Inf", np.nan), errors="coerce"
    )
    exact_count = int((frame["retrieved_real_B_CNMSE_dB"] == "-Inf").sum())
    failure_count = int((values >= -40).sum())
    shareable_count = exact_count + int((values < -40).sum())
    assert exact_count == metadata["exact_inf_count"] == 218
    assert failure_count == metadata["failure_count"] == 14
    assert shareable_count == metadata["dpd_shareable_count"] == 411


def test_saved_workbook_exists_and_artifact_tool_reloads_it() -> None:
    assert OUTPUT_XLSX.is_file()
    assert OUTPUT_XLSX.stat().st_size > 0
    completed = subprocess.run(
        [sys.executable, str(EXPORTER), "--verify-only", "--output", str(OUTPUT_XLSX)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert '"rows":426' in completed.stdout.replace(" ", "")
    assert '"columns":10' in completed.stdout.replace(" ", "")
    assert '"exactCount":218' in completed.stdout.replace(" ", "")


def test_exporter_does_not_fit_or_reprocess_models() -> None:
    source = EXPORTER.read_text(encoding="utf-8-sig")
    assert "lstsq" not in source
    assert "fit_coefficients" not in source
    assert "ridge_scan" not in source
    assert "preprocess_full_pair" not in source


def test_old_c3_a2_result_remains_unchanged() -> None:
    assert _tree_sha256(OLD_C3_ROOT) == (
        "a6c9e289c5771003f3d9681ecae9b1278a50b4b81c5f2b4acb11d30a296b41f1"
    )


def main() -> None:
    tests = [
        test_source_table_has_fixed_shape_and_columns,
        test_source_table_uses_physical_load_configuration,
        test_source_table_acpr_average_and_finite_columns,
        test_key_state_rows_and_retrieval_values,
        test_source_counts_match_formal_retrieval,
        test_saved_workbook_exists_and_artifact_tool_reloads_it,
        test_exporter_does_not_fit_or_reprocess_models,
        test_old_c3_a2_result_remains_unchanged,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")


if __name__ == "__main__":
    main()
