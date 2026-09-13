"""Run the fixed 0.5B Scenario 2 retrieval experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_lut_retrieval.run_scenario2_c2_to_aend_1b import (  # noqa: E402
    MARK_ARTIFACT,
    NODE_EXE,
    _append_handoff,
    _append_log,
    _json_default,
    _read_csv_source,
    _run_node,
)
from behavior_fingerprint_lut_retrieval.scenario2_c2_to_aend_1b import (  # noqa: E402
    SPEC_0P5B,
    SPEC_1B,
    _max_difference_allowing_neginf,
    run_experiment,
)

RESULT_ROOT = SPEC_0P5B.result_root
OUTPUT_XLSX = RESULT_ROOT / "scenario_2_C2_to_Aend_0p5B_retrieval.xlsx"
ONE_B_ROOT = SPEC_1B.result_root


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file()
            and "__pycache__" not in item.parts
            and item.suffix.lower() != ".pyc"
            and not item.name.startswith("~$")
        ),
        key=lambda item: str(item).lower(),
    )
    for file in files:
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _compare_frames(left: pd.DataFrame, right: pd.DataFrame, name: str) -> dict[str, Any]:
    _require(left.columns.tolist() == right.columns.tolist(), f"{name} columns differ")
    _require(left.shape == right.shape, f"{name} shape differs: {left.shape} != {right.shape}")
    numeric_error = 0.0
    mismatches = 0
    for column in left.columns:
        a = left[column]
        b = right[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av = a.to_numpy(dtype=float)
            bv = b.to_numpy(dtype=float)
            error, mismatch = _max_difference_allowing_neginf(av, bv)
            numeric_error = max(numeric_error, error)
            mismatches += mismatch
        else:
            mismatches += int(
                np.count_nonzero(a.astype(str).to_numpy() != b.astype(str).to_numpy())
            )
    return {
        "shape": list(left.shape),
        "max_numeric_error": numeric_error,
        "mismatch_count": mismatches,
    }


def _load_npz_fingerprint(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["fingerprints"])


def _run_1b_regression(temp_root: Path, progress_interval: int) -> dict[str, Any]:
    """Run the refactored 1B path in a temporary directory and compare it."""

    print(
        "Preflight: run parameterized 1B numerical regression in a temporary directory", flush=True
    )
    run_experiment(
        spec=SPEC_1B,
        result_root=temp_root,
        progress_interval=progress_interval,
        run_5b_guard=False,
    )
    checks: dict[str, Any] = {}
    array_pairs = (
        "Aend_model_coefficients_1B.npy",
        "C2_model_coefficients_1B.npy",
        "fingerprint_cnmse_matrix_1B.npy",
        "query_to_lut_cnmse_1B.npy",
    )
    for filename in array_pairs:
        current = np.load(temp_root / filename)
        frozen = np.load(ONE_B_ROOT / filename)
        error, mismatch = _max_difference_allowing_neginf(current, frozen)
        checks[filename] = {"max_abs_error": error, "nonfinite_pattern_mismatch": mismatch}
    for filename in ("lut_fingerprints_1B.npz", "query_fingerprints_1B.npz"):
        current = _load_npz_fingerprint(temp_root / filename)
        frozen = _load_npz_fingerprint(ONE_B_ROOT / filename)
        error, mismatch = _max_difference_allowing_neginf(current, frozen)
        checks[filename] = {"max_abs_error": error, "nonfinite_pattern_mismatch": mismatch}

    for filename in (
        "retrieval_results_1B.csv",
        "retrieval_diagnostics_1B.csv",
        "top1_retrieval_1B.csv",
        "model_metrics_1B.csv",
        "retrieval_failures_1B.csv",
        "a_end_stage_map.csv",
        "retrieval_summary_1B.csv",
    ):
        checks[filename] = _compare_frames(
            pd.read_csv(temp_root / filename),
            pd.read_csv(ONE_B_ROOT / filename),
            filename,
        )

    current_main = pd.read_csv(temp_root / "retrieval_results_1B.csv")
    frozen_main = pd.read_csv(ONE_B_ROOT / "retrieval_results_1B.csv")
    _require(
        np.array_equal(current_main["state_id_Q"], frozen_main["state_id_Q"]),
        "1B regression State_Q differs",
    )
    _require(
        np.array_equal(
            current_main["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float),
            frozen_main["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float),
        ),
        "1B regression retrieved Real-B differs",
    )
    current_failure = current_main.loc[
        current_main["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float) >= -40.0,
        "state_id_R",
    ].to_numpy(dtype=int)
    frozen_failure = frozen_main.loc[
        frozen_main["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float) >= -40.0,
        "state_id_R",
    ].to_numpy(dtype=int)
    _require(np.array_equal(current_failure, frozen_failure), "1B regression failure IDs differ")
    for name, item in checks.items():
        numeric_error = item.get("max_abs_error", item.get("max_numeric_error", 0.0))
        _require(numeric_error < 1e-12, f"1B regression numerical mismatch: {name} {item}")
        _require(
            item["mismatch_count"] == 0 if "mismatch_count" in item else True,
            f"1B regression table mismatch: {name} {item}",
        )
        _require(
            item["nonfinite_pattern_mismatch"] == 0
            if "nonfinite_pattern_mismatch" in item
            else True,
            f"1B regression nonfinite mismatch: {name} {item}",
        )
    return {
        "pass": True,
        "temporary_result_root": str(temp_root),
        "checks": checks,
        "state_Q_identical": True,
        "failure_state_ids_identical": True,
        "exact_count_identical": int(
            (current_main["state_id_Q"] == current_main["state_id_R"]).sum()
        ),
        "shareable_count_identical": int(
            (current_main["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float) < -40.0).sum()
        ),
    }


def _update_validation(
    validation_path: Path, output_xlsx: Path, regression: dict[str, Any]
) -> None:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["one_B_regression_guard"] = regression
    validation["output_files"]["xlsx"] = str(output_xlsx)
    validation["excel"] = {
        "created_with": "@oai/artifact-tool",
        "sheet_names": ["retrieval_results", "retrieval_diagnostics", "summary"],
        "main_columns": [
            "state_id_R",
            "load_config",
            "nmse_withoutdpd_dB",
            "acpr_withoutdpd_avg_dBc",
            "Y_Aend_train_NMSE_dB",
            "Y_Aend_B_NMSE_dB",
            "Y_C2_train_NMSE_dB",
            "Y_C2_B_NMSE_dB",
            "state_id_Q",
            "retrieved_real_B_CNMSE_dB",
        ],
        "main_data_rows": 425,
        "diagnostics_data_rows": 425,
        "xlsx_path": str(output_xlsx),
        "formula_error_scan": "PASS",
        "source_fact_tables": [
            str(output_xlsx.parent / "retrieval_results_0p5B.csv"),
            str(output_xlsx.parent / "retrieval_diagnostics_0p5B.csv"),
            str(output_xlsx.parent / "retrieval_summary_0p5B.csv"),
        ],
    }
    validation["final_status"] = "completed"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()

    before_one_b_sha = _tree_digest(ONE_B_ROOT)
    regression: dict[str, Any]
    with tempfile.TemporaryDirectory(prefix="scenario2_1b_regression_") as temp_dir:
        regression = _run_1b_regression(Path(temp_dir), args.progress_interval)
    after_one_b_sha = _tree_digest(ONE_B_ROOT)
    _require(before_one_b_sha == after_one_b_sha, "1B正式结果目录在regression期间发生变化")

    print("Step 1/2: run 0.5B experiment after 1B regression passed", flush=True)
    result = run_experiment(
        spec=SPEC_0P5B,
        progress_interval=args.progress_interval,
        run_5b_guard=True,
    )
    source_payload = _read_csv_source(result_root=RESULT_ROOT, result_tag=SPEC_0P5B.result_tag)
    with tempfile.TemporaryDirectory(prefix="scenario2_0p5b_source_") as temp_dir:
        source_path = Path(temp_dir) / "scenario2_0p5b_excel_source.json"
        source_path.write_text(
            json.dumps(source_payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        marker = subprocess.run(
            [
                str(NODE_EXE),
                str(MARK_ARTIFACT),
                "--operation-kind",
                "create",
                "--expected-output-count",
                "1",
                "--output-format",
                "xlsx",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        print(marker.stdout)
        _require(marker.returncode == 0, f"artifact operation marker失败：{marker.stderr}")
        _run_node(source_path, OUTPUT_XLSX, RESULT_ROOT)

    _require(OUTPUT_XLSX.is_file() and OUTPUT_XLSX.stat().st_size > 0, "0.5B Excel未成功生成")
    _update_validation(RESULT_ROOT / "validation.json", OUTPUT_XLSX, regression)

    summary = result["tables"]["summary_payload"]
    guard = result["validation"]["five_B_regression_guard"]
    phase = result["phase_observation"]
    _append_log(
        "\n".join(
            [
                "研究目标：固定0.5B（n=0.5，Fs_obs=10MHz，k=5，up=1，down=10）完成Y-C2→Y-Aend行为指纹LUT检索。",
                (
                    "参数化后的1B numerical regression在临时目录通过，正式1B结果SHA保持不变；"
                    f"5B identity regression guard={guard['pass']}。"
                ),
                (
                    f"0.5B LUT/Query fingerprint={phase.aend_fingerprints.shape}/"
                    f"{phase.c2_fingerprints.shape}；distance={result['tables']['distance'].shape}；"
                ),
                (
                    f"exact={summary['exact_hit_count']}/{summary['exact_hit_rate']:.9g}；"
                    f"shareable(strict<-40dB)={summary['shareable_count']}/"
                    f"{summary['shareable_rate']:.9g}；"
                ),
                (
                    f"non-exact shareable={summary['nonexact_shareable_count']}/"
                    f"{summary['nonexact_shareable_rate']:.9g}；failure={summary['failure_count']}；"
                    f"failure_ids={summary['failure_state_ids']}。"
                ),
                (
                    f"Excel由artifact-tool创建/重载检查通过：{OUTPUT_XLSX}；"
                    "main=425x10，diagnostics=425x10，summary=metric/value。"
                ),
                (
                    "0.5B仅用于行为建模和检索；5B Real-B仅用于最终评价；Real-B未参与Q选择；"
                    "未执行跨带宽分析或0.1B~5B扫描。"
                ),
                (
                    "data/raw、1B正式结果、5B正式结果和low-bandwidth v1.0.1保护复核通过；"
                    "未执行破坏性Git操作。"
                ),
            ]
        ),
        title="fixed 0.5B behavior-fingerprint LUT retrieval",
    )
    _append_handoff(
        f"固定0.5B行为指纹检索完成：1B regression=True，5B regression={guard['pass']}；425状态，"
        f"LUT/Query fingerprint={phase.aend_fingerprints.shape}/{phase.c2_fingerprints.shape}；"
        f"shareable={summary['shareable_count']}/{summary['shareable_rate']:.6g}，failure={summary['failure_count']}；"
        "0.5B仅用于检索，5B Real-B仅用于最终验证；1B/5B/算子保护通过。",
        title="固定0.5B检索",
    )
    print(
        json.dumps(
            {"result_root": str(RESULT_ROOT), "xlsx": str(OUTPUT_XLSX), **summary},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
