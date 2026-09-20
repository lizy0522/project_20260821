"""Tests for the Excel-backed six-curve C2-to-A_end state figure."""

from __future__ import annotations

import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_retrieval.scenario_2.scenario_2_C2_to_Aend_retrieval.plot_scenario2_c2_to_aend_state_metrics import (  # noqa: E402,E501
    CURVE_COLUMNS,
    INPUT_XLSX,
    OUTPUT_FIGURE,
    PREVIOUS_FIGURES,
    _read_excel_with_bundled_python,
    plot_state_metrics_and_retrieval,
    validate_summary_frame,
)

PROJECT_ROOT = SCRIPTS_ROOT.parent
PLOTTER = Path(__file__).with_name("plot_scenario2_c2_to_aend_state_metrics.py")


@lru_cache(maxsize=1)
def _validated() -> dict[str, object]:
    return validate_summary_frame(_read_excel_with_bundled_python(INPUT_XLSX))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_excel_input_contract() -> None:
    metadata = _validated()
    frame = metadata["frame"]
    assert isinstance(frame, pd.DataFrame)
    assert frame.shape[0] == 425
    assert list(frame["state_id_R"]) == list(range(425))
    assert tuple(CURVE_COLUMNS) == (
        "nmse_withoutdpd_dB",
        "Y_Aend_train_NMSE_dB",
        "Y_Aend_B_NMSE_dB",
        "Y_C2_train_NMSE_dB",
        "Y_C2_B_NMSE_dB",
        "retrieved_real_B_CNMSE_dB",
    )


def test_all_five_model_curves_are_finite() -> None:
    metadata = _validated()
    curve_values = metadata["curve_values"]
    for column in CURVE_COLUMNS[:-1]:
        assert np.isfinite(curve_values[column]).all()
        assert curve_values[column].shape == (425,)


def test_retrieved_curve_exact_and_failure_masks() -> None:
    metadata = _validated()
    assert int(metadata["exact_count"]) == 218
    assert int(metadata["finite_retrieved_count"]) == 207
    assert int(metadata["failure_count"]) == 14
    assert int(metadata["shareable_count"]) == 411
    assert int(metadata["query_states"][340]) == 217
    assert abs(float(metadata["retrieved"][340]) + 37.10367236988057) < 1e-10


def test_plot_metadata_has_six_curves_and_all_q_annotations() -> None:
    metadata = _validated()
    output = OUTPUT_FIGURE.with_name("_test_state_metrics_and_retrieval_cnmse.png")
    try:
        plotted = plot_state_metrics_and_retrieval(metadata["frame"], output)
        assert plotted["curve_count"] == 6
        assert plotted["q_annotation_count"] == 207
        assert plotted["exact_count"] == 218
        assert plotted["ylim"][0] < plotted["finite_y_min"]
        assert plotted["finite_y_min"] <= -40.0 <= plotted["finite_y_max"]
        assert output.is_file() and output.stat().st_size > 0
    finally:
        if output.exists():
            output.unlink()


def test_formal_threshold_is_only_for_retrieved_real_b() -> None:
    source = PLOTTER.read_text(encoding="utf-8-sig")
    assert "-40.0" in source
    assert "threshold_scope" in source
    assert "retrieved_real_B_CNMSE_dB" in source
    assert "data/raw" not in source


def test_no_smoothing_or_state_resorting() -> None:
    source = PLOTTER.read_text(encoding="utf-8-sig")
    for forbidden in ("savgol", "moving_average", "interpolate", "rolling"):
        assert forbidden not in source.lower()
    assert "sort_values(\"state_id_R\")" in source


def test_output_and_previous_figures_are_readable() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert OUTPUT_FIGURE.is_file()
    for path in (*PREVIOUS_FIGURES, OUTPUT_FIGURE):
        image = plt.imread(path)
        assert image.size > 0
        assert image.ndim in (2, 3)
    plt.close("all")


def test_input_and_previous_figure_sha_can_be_snapshotted() -> None:
    before = {
        "input": _file_sha256(INPUT_XLSX),
        "figures": {path.name: _file_sha256(path) for path in PREVIOUS_FIGURES},
    }
    assert before["input"]
    assert len(before["figures"]) == 4


def main() -> None:
    tests = [
        test_excel_input_contract,
        test_all_five_model_curves_are_finite,
        test_retrieved_curve_exact_and_failure_masks,
        test_plot_metadata_has_six_curves_and_all_q_annotations,
        test_formal_threshold_is_only_for_retrieved_real_b,
        test_no_smoothing_or_state_resorting,
        test_output_and_previous_figures_are_readable,
        test_input_and_previous_figure_sha_can_be_snapshotted,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")


if __name__ == "__main__":
    main()
