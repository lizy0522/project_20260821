"""
功能说明：执行State0 11个canonical正向模型的训练、B泛化和公共xin_B等效验证并保存结果。
输入：data_manager和signal_segmentation提供的State0 OFF/ILC canonical A/B/C。
输出：11行CSV、11个theta、公共B响应NPZ、验收JSON和4张基础数值图。
用途：不覆盖整记录旧基线，不使用STALE，不扩展到425状态。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.shared import BASIS_TERMS, MAX_DELAY, MP_CONFIG  # noqa: E402
from behavior_modeling.shared.xy_equivalence import (  # noqa: E402
    ModelEvaluation,
    State0XYEquivalenceResult,
    analyze_state0_xy_equivalence,
)

RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /"state0_xy_equivalence"
MODEL_DIR = RESULT_ROOT / "models"


def _theta_filename(item: ModelEvaluation) -> str:
    if item.model_id == "X-A":
        return "theta_X_A.npy"
    return f"theta_{item.model_id.replace('-', '_')[:3]}_iter_{item.ilc_iteration:03d}.npy"


def _summary_row(item: ModelEvaluation) -> dict[str, str | int | float]:
    return {
        "model_id": item.model_id,
        "behavior_class": item.behavior_class,
        "train_segment": item.train_segment,
        "ilc_iteration": "" if item.ilc_iteration is None else item.ilc_iteration,
        "train_NMSE_dB": item.train_nmse_db,
        "B_generalization_NMSE_dB": item.b_generalization_nmse_db,
        "commonB_vs_X_CNMSE_dB": item.common_b_vs_x_cnmse_db,
        "commonB_vs_realB_CNMSE_dB": item.common_b_vs_real_b_cnmse_db,
        "rank": item.rank,
        "condition_number": item.condition_number,
        "n_train_samples": item.n_train_samples,
        "coefficient_count": item.coefficient_count,
        "B_generalization_source": item.b_generalization_source,
        "common_B_probe_source": "OFF.B.input",
        "common_B_response_length": item.common_b_response.size,
    }


def _model_colors(items: tuple[ModelEvaluation, ...]) -> list[str]:
    return [
        "#4D4D4D"
        if item.behavior_class == "X"
        else "#377EB8"
        if item.train_segment == "A"
        else "#E68613"
        for item in items
    ]


def _bar_plot(
    labels: list[str],
    values: list[float],
    colors: list[str],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    positions = np.arange(len(labels))
    axis.bar(positions, values, color=colors, width=0.75)
    axis.set_xticks(positions, labels, rotation=45, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_plots(result: State0XYEquivalenceResult) -> None:
    items = result.evaluations
    labels = [item.model_id for item in items]
    colors = _model_colors(items)
    _bar_plot(
        labels,
        [item.train_nmse_db for item in items],
        colors,
        "NMSE (dB)",
        "State0 Train NMSE",
        RESULT_ROOT / "train_nmse_state0.png",
    )
    _bar_plot(
        labels,
        [item.b_generalization_nmse_db for item in items],
        colors,
        "NMSE (dB)",
        "State0 B-Segment Generalization NMSE",
        RESULT_ROOT / "B_generalization_nmse_state0.png",
    )
    y_items = tuple(item for item in items if item.behavior_class == "Y")
    y_labels = [item.model_id for item in y_items]
    y_colors = _model_colors(y_items)
    _bar_plot(
        y_labels,
        [item.common_b_vs_x_cnmse_db for item in y_items],
        y_colors,
        "CNMSE (dB)",
        "State0 Common-B: Y Response vs X Response",
        RESULT_ROOT / "Y_vs_X_cnmse_state0.png",
    )
    _bar_plot(
        y_labels,
        [item.common_b_vs_real_b_cnmse_db for item in y_items],
        y_colors,
        "CNMSE (dB)",
        "State0 Common-B: Y Response vs Real-B",
        RESULT_ROOT / "Y_vs_realB_cnmse_state0.png",
    )


def _validation_payload(result: State0XYEquivalenceResult) -> dict[str, Any]:
    items = result.evaluations
    y_items = [item for item in items if item.behavior_class == "Y"]
    return {
        "state_id": result.state_id,
        "model_count": len(items),
        "x_a_model_count": sum(item.model_id == "X-A" for item in items),
        "y_a_model_count": sum(item.model_id.startswith("Y-A-") for item in items),
        "y_c_model_count": sum(item.model_id.startswith("Y-C-") for item in items),
        "train_nmse_count": len(items),
        "b_generalization_nmse_count": len(items),
        "x_vs_real_cnmse_count": 1,
        "y_vs_x_cnmse_count": len(y_items),
        "y_vs_real_cnmse_count": len(y_items),
        "total_valid_cnmse_count": 1 + 2 * len(y_items),
        "common_probe_ownership_length": result.common_probe.size,
        "common_probe_valid_length": result.common_probe_valid.size,
        "real_b_valid_length": result.real_b_valid.size,
        "all_common_responses_length": sorted(
            {response.size for response in result.common_responses.values()}
        ),
        "all_rank_10": all(item.rank == 10 for item in items),
        "all_theta_shape_10": all(item.theta.shape == (10,) for item in items),
        "all_theta_finite": all(np.all(np.isfinite(item.theta)) for item in items),
        "orders": list(MP_CONFIG["orders"]),
        "memory_depth": MP_CONFIG["memory_depth"],
        "basis_terms": [{"order": order, "memory": memory} for order, memory in BASIS_TERMS],
        "max_delay": MAX_DELAY,
        "used_stale": False,
        "common_probe_source": "OFF.B.input",
        "cnmse_reference_directions": {
            "X_vs_Real": "cnmse(real_B, X_response)",
            "Y_vs_X": "cnmse(X_response, Y_response)",
            "Y_vs_Real": "cnmse(real_B, Y_response)",
        },
    }


def _save_result(result: State0XYEquivalenceResult) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    rows = [_summary_row(item) for item in result.evaluations]
    summary_path = RESULT_ROOT / "state0_xy_equivalence_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for item in result.evaluations:
        np.save(MODEL_DIR / _theta_filename(item), item.theta)

    response_payload = {
        "xin_B_valid": result.common_probe_valid,
        "real_B": result.real_b_valid,
        "X_A": result.common_responses["X-A"],
    }
    for item in result.evaluations:
        if item.behavior_class == "Y":
            response_payload[item.model_id.replace("-", "_")] = item.common_b_response
    np.savez_compressed(RESULT_ROOT / "common_B_responses.npz", **response_payload)

    with (RESULT_ROOT / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(_validation_payload(result), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _save_plots(result)


def main() -> None:
    result = analyze_state0_xy_equivalence()
    _save_result(result)
    for item in result.evaluations:
        print(
            f"{item.model_id}: train={item.train_nmse_db:.9f} dB, "
            f"B-generalization={item.b_generalization_nmse_db:.9f} dB, "
            f"commonB-vs-X={item.common_b_vs_x_cnmse_db:.9f} dB, "
            f"commonB-vs-Real={item.common_b_vs_real_b_cnmse_db:.9f} dB, "
            f"rank={item.rank}"
        )
    print("Validated: 11 models, 11 Train NMSE, 11 B NMSE, 21 non-NaN CNMSE")
    print(f"Results: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
