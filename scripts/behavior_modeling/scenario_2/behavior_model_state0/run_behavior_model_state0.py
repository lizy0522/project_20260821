"""
功能说明：读取state_id=0，提取1个X类和全部ILC迭代Y类正向MP行为模型。
输入：data_manager提供的state0 MAT变量和冻结10基函数配置。
输出：theta NPY、每模型evaluation JSON及state_000/summary.csv。
用途：仅完成state0验证；不执行多状态、Behavior Index、LUT、检索、聚类或A/B/C划分。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.shared.signal import preprocess_pair  # noqa: E402
from data_management.shared import get_state_info, load_by_id  # noqa: E402

from behavior_modeling.shared import (  # noqa: E402
    BASIS_TERMS,
    MAX_DELAY,
    MP_CONFIG,
    NUM_COEFFICIENTS,
    MemoryPolynomialModel,
    calculate_nmse,
)

STATE_ID = 0
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /"behavior_model_state0"


def _as_complex_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or not np.iscomplexobj(array):
        raise ValueError(f"{name}必须是一维复数向量或(N,1)复数列，实际shape={array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含NaN或Inf")
    return array


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _basis_manifest() -> list[dict[str, int]]:
    return [{"order": order, "memory": memory} for order, memory in BASIS_TERMS]


def _evaluation_payload(
    model: MemoryPolynomialModel,
    nmse_db: float,
    model_type: str,
    iteration: int | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    if model.theta is None or model.fit_diagnostics is None:
        raise RuntimeError("不能为未拟合模型生成评价")
    return {
        "state_id": STATE_ID,
        "state_info": get_state_info(STATE_ID),
        "model_type": model_type,
        "iteration": iteration,
        "nmse_db": nmse_db,
        "theta_shape": list(model.theta.shape),
        "theta_dtype": str(model.theta.dtype),
        "orders": list(MP_CONFIG["orders"]),
        "memory_depth": MP_CONFIG["memory_depth"],
        "basis_terms": _basis_manifest(),
        "max_delay": MAX_DELAY,
        "least_squares": model.fit_diagnostics.as_dict(),
        **extra,
    }


def _fit_and_save(
    x: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    theta_filename: str,
    model_type: str,
    iteration: int | None,
    extra: dict[str, Any],
) -> dict[str, str | int | float]:
    model = MemoryPolynomialModel().fit(x, y)
    if model.theta is None or model.theta.shape != (NUM_COEFFICIENTS,):
        raise RuntimeError("模型未生成10个系数")
    y_prediction = model.predict(x)
    y_valid = y[MAX_DELAY:]
    nmse_db = calculate_nmse(y_valid, y_prediction)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / theta_filename, model.theta)
    _write_json(
        output_dir / "evaluation.json",
        _evaluation_payload(model, nmse_db, model_type, iteration, extra),
    )
    return {
        "model_type": model_type,
        "iteration": "" if iteration is None else iteration,
        "nmse_db": nmse_db,
    }


def main() -> None:
    data = load_by_id(STATE_ID)
    x_x = _as_complex_vector(data["xin"], "xin")
    y_x = _as_complex_vector(data["yout_withoutdpd"], "yout_withoutdpd")
    x_ilc = np.asarray(data["xin_pd_ori_ilc"])
    y_ilc_raw = np.asarray(data["yout_withdpd_ori_ilc"])
    if x_ilc.ndim != 2 or y_ilc_raw.shape != x_ilc.shape:
        raise ValueError(
            f"ILC输入输出历史shape必须一致且为二维，实际为{x_ilc.shape}和{y_ilc_raw.shape}"
        )
    ilc_iterations = x_ilc.shape[1]
    saved_iterations = int(np.asarray(data["nth_inter"]).reshape(-1)[0])
    if ilc_iterations != saved_iterations:
        raise ValueError(f"ILC列数{ilc_iterations}与nth_inter={saved_iterations}不一致")

    rows: list[dict[str, str | int | float]] = []
    rows.append(
        _fit_and_save(
            x_x,
            y_x,
            RESULT_ROOT / "X",
            "theta_X.npy",
            "X",
            None,
            {
                "input_variable": "xin",
                "output_variable": "yout_withoutdpd",
                "preprocessing": "saved_processed_output_no_additional_processing",
            },
        )
    )
    print(f"X model: NMSE={rows[-1]['nmse_db']:.12f} dB")

    for column in range(ilc_iterations):
        iteration = column + 1
        x_unscaled = _as_complex_vector(x_ilc[:, column], f"xin_pd_ori_ilc[:,{column}]")
        y_raw = _as_complex_vector(
            y_ilc_raw[:, column],
            f"yout_withdpd_ori_ilc[:,{column}]",
        )
        peak_scale = float(np.max(np.abs(x_unscaled)))
        if peak_scale == 0:
            raise ValueError(f"第{iteration}次ILC输入峰值为零")
        x_actual = x_unscaled / peak_scale
        y_processed = preprocess_pair(x_actual, y_raw, subtime=256)

        row = _fit_and_save(
            x_actual,
            y_processed,
            RESULT_ROOT / "Y" / f"iter_{iteration:03d}",
            "theta_Y.npy",
            "Y",
            iteration,
            {
                "input_variable": "xin_pd_ori_ilc",
                "output_variable": "yout_withdpd_ori_ilc",
                "input_peak_normalization_factor": peak_scale,
                "preprocessing": "peak_normalize_input_then_rough_fine_complex_gain_output",
                "fine_align_subtime": 256,
            },
        )
        rows.append(row)
        print(f"Y iteration {iteration}: NMSE={row['nmse_db']:.12f} dB")

    summary_path = RESULT_ROOT / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_type", "iteration", "nmse_db"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Completed state_id={STATE_ID}: 1 X model + {ilc_iterations} Y models")
    print(f"Results: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
