"""Plot the six frozen Scenario 2 state metrics on one wide matplotlib axis."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_retrieval"
    / "scenario_2_C2_to_Aend"
)
INPUT_XLSX = RESULT_ROOT / "scenario_2_C2_to_Aend_state_summary.xlsx"
OUTPUT_FIGURE = RESULT_ROOT / "figure5_state_metrics_and_retrieval_cnmse.png"
WORK_LOG = (
    PROJECT_ROOT
    / "work_logs"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_retrieval"
    / "execution_log.txt"
)
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
BUNDLED_PYTHON = Path(
    r"C:\Users\lizy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)
PREVIOUS_FIGURES = tuple(
    RESULT_ROOT / f"figure{index}_{suffix}.png"
    for index, suffix in (
        (1, "state_retrieval_mapping"),
        (2, "real_B_cnmse_by_state"),
        (3, "real_B_cnmse_distribution"),
        (4, "dpd_shareable_success_by_state"),
    )
)
REQUIRED_COLUMNS = (
    "state_id_R",
    "nmse_withoutdpd_dB",
    "Y_Aend_train_NMSE_dB",
    "Y_Aend_B_NMSE_dB",
    "Y_C2_train_NMSE_dB",
    "Y_C2_B_NMSE_dB",
    "state_id_Q",
    "retrieved_real_B_CNMSE_dB",
)
CURVE_COLUMNS = (
    "nmse_withoutdpd_dB",
    "Y_Aend_train_NMSE_dB",
    "Y_Aend_B_NMSE_dB",
    "Y_C2_train_NMSE_dB",
    "Y_C2_B_NMSE_dB",
    "retrieved_real_B_CNMSE_dB",
)
CURVE_LABELS = {column: column for column in CURVE_COLUMNS}
CURVE_LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)))
CURVE_MARKERS = ("o", "s", "^", "v", "D", "x")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_excel_with_bundled_python(path: Path) -> pd.DataFrame:
    """Read the workbook with bundled openpyxl without using it for plotting."""

    _require(BUNDLED_PYTHON.is_file(), f"缺少bundled Python：{BUNDLED_PYTHON}")
    code = (
        "import json, sys, pandas as pd; "
        "frame = pd.read_excel(sys.argv[1], sheet_name=sys.argv[2]); "
        "print(json.dumps({'columns': frame.columns.tolist(), 'rows': "
        "frame.to_dict(orient='records')}, ensure_ascii=False, allow_nan=True))"
    )
    completed = subprocess.run(
        [str(BUNDLED_PYTHON), "-c", code, str(path), "state_summary"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Excel读取失败：{completed.stdout}\n{completed.stderr}")
    payload = json.loads(completed.stdout)
    frame = pd.DataFrame(payload["rows"])
    if list(payload["columns"]) != list(frame.columns):
        raise RuntimeError("Excel列元数据与读取表不一致")
    return frame


def _parse_retrieved_values(series: pd.Series) -> np.ndarray:
    values: list[float] = []
    for index, value in enumerate(series):
        if isinstance(value, str) and value.strip().lower() in {"-inf", "-infinity"}:
            values.append(-np.inf)
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"retrieved Real-B CNMSE 第{index + 2}行无法解析：{value!r}"
            ) from exc
        values.append(parsed)
    return np.asarray(values, dtype=np.float64)


def validate_summary_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate the frozen Excel contract and return plotting arrays/metadata."""

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    _require(not missing, f"Excel缺少字段：{missing}")
    _require(frame.shape[0] == 425, f"Excel数据行数必须为425，实际{frame.shape[0]}")
    ordered = frame.sort_values("state_id_R").reset_index(drop=True)
    state_ids = ordered["state_id_R"].to_numpy(dtype=np.int64)
    _require(np.array_equal(state_ids, np.arange(425)), "state_id_R必须完整为0...424")
    _require(ordered["state_id_R"].is_unique, "state_id_R存在重复")
    curve_values: dict[str, np.ndarray] = {}
    for column in CURVE_COLUMNS[:-1]:
        values = ordered[column].to_numpy(dtype=float)
        _require(np.all(np.isfinite(values)), f"{column}必须全部finite")
        curve_values[column] = values
    query_states = ordered["state_id_Q"].to_numpy(dtype=float)
    _require(np.all(np.isfinite(query_states)), "state_id_Q存在非finite")
    _require(np.all(query_states == np.floor(query_states)), "state_id_Q必须为整数")
    _require(np.all((query_states >= 0) & (query_states <= 424)), "state_id_Q越界")
    retrieved = _parse_retrieved_values(ordered["retrieved_real_B_CNMSE_dB"])
    exact_mask = np.isneginf(retrieved)
    finite_mask = np.isfinite(retrieved)
    _require(np.count_nonzero(exact_mask) == 218, "retrieved Real-B CNMSE的-Inf数量必须为218")
    _require(np.count_nonzero(finite_mask) == 207, "retrieved Real-B CNMSE的finite数量必须为207")
    exact_from_ids = query_states.astype(np.int64) == state_ids
    _require(np.array_equal(exact_mask, exact_from_ids), "-Inf与state_id_Q==state_id_R不一致")
    failure_mask = finite_mask & (retrieved >= -40.0)
    shareable_mask = exact_mask | (finite_mask & (retrieved < -40.0))
    _require(np.count_nonzero(failure_mask) == 14, "正式failure数量必须为14")
    _require(np.count_nonzero(shareable_mask) == 411, "正式shareable数量必须为411")
    state_340 = int(np.flatnonzero(state_ids == 340)[0])
    _require(int(query_states[state_340]) == 217, "State 340 的state_id_Q必须为217")
    _require(abs(float(retrieved[state_340]) + 37.10367236988057) < 1e-10, "State 340 CNMSE不一致")
    curve_values["retrieved_real_B_CNMSE_dB"] = retrieved
    finite_for_ylim = np.concatenate(
        [values for values in curve_values.values() if values is not retrieved]
        + [retrieved[finite_mask]]
    )
    return {
        "frame": ordered,
        "state_ids": state_ids,
        "query_states": query_states.astype(np.int64),
        "curve_values": curve_values,
        "retrieved": retrieved,
        "finite_mask": finite_mask,
        "exact_mask": exact_mask,
        "failure_mask": failure_mask,
        "shareable_mask": shareable_mask,
        "finite_y_min": float(np.min(finite_for_ylim)),
        "finite_y_max": float(np.max(finite_for_ylim)),
        "exact_count": int(exact_mask.sum()),
        "finite_retrieved_count": int(finite_mask.sum()),
        "failure_count": int(failure_mask.sum()),
        "shareable_count": int(shareable_mask.sum()),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_snapshot() -> dict[str, Any]:
    _require(INPUT_XLSX.is_file(), f"缺少输入Excel：{INPUT_XLSX}")
    for path in PREVIOUS_FIGURES:
        _require(path.is_file(), f"缺少既有Figure：{path}")
    return {
        "input_excel_sha256": _file_sha256(INPUT_XLSX),
        "previous_figure_sha256": {
            path.name: _file_sha256(path) for path in PREVIOUS_FIGURES
        },
    }


def _verify_protected(before: dict[str, Any], after: dict[str, Any]) -> None:
    _require(before["input_excel_sha256"] == after["input_excel_sha256"], "输入Excel SHA发生变化")
    _require(
        before["previous_figure_sha256"] == after["previous_figure_sha256"],
        "既有Figure 1~4 SHA发生变化",
    )


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_state_metrics_and_retrieval(
    frame: pd.DataFrame, output_path: Path = OUTPUT_FIGURE
) -> dict[str, Any]:
    """Create the single-Axes six-curve state summary figure."""

    metadata = validate_summary_frame(frame)
    _configure_matplotlib()
    x = metadata["state_ids"]
    curve_values = metadata["curve_values"]
    retrieved = metadata["retrieved"]
    finite_mask = metadata["finite_mask"]
    exact_mask = metadata["exact_mask"]
    figure, axis = plt.subplots(figsize=(32, 10))
    for index, column in enumerate(CURVE_COLUMNS[:-1]):
        axis.plot(
            x,
            curve_values[column],
            label=CURVE_LABELS[column],
            linestyle=CURVE_LINESTYLES[index],
            marker=CURVE_MARKERS[index],
            markevery=10,
            markersize=3.0,
            linewidth=1.05,
            alpha=0.9,
            zorder=2,
        )
    retrieved_plot = retrieved.copy()
    retrieved_plot[~finite_mask] = np.nan
    axis.plot(
        x,
        retrieved_plot,
        label=CURVE_LABELS["retrieved_real_B_CNMSE_dB"],
        linestyle=CURVE_LINESTYLES[-1],
        marker=CURVE_MARKERS[-1],
        markevery=1,
        markersize=4.2,
        linewidth=1.35,
        alpha=0.95,
        zorder=5,
    )
    display_y = metadata["finite_y_min"] - 1.5
    axis.scatter(
        x[exact_mask],
        np.full(int(exact_mask.sum()), display_y),
        marker="|",
        s=90,
        linewidths=1.15,
        label="Exact Hit (Q = R, Real-B CNMSE = -Inf)",
        zorder=7,
    )
    axis.axhline(
        -40.0,
        linestyle="--",
        linewidth=1.0,
        label="DPD-shareable threshold (-40 dB)",
        zorder=1,
    )
    finite_indices = np.flatnonzero(finite_mask)
    offsets = (5, -10, 10, -5)
    for label_index, row_index in enumerate(finite_indices):
        axis.annotate(
            f"Q={int(metadata['query_states'][row_index])}",
            xy=(int(x[row_index]), float(retrieved[row_index])),
            xytext=(0, offsets[label_index % len(offsets)]),
            textcoords="offset points",
            ha="center",
            va="bottom" if offsets[label_index % len(offsets)] > 0 else "top",
            fontsize=5,
            rotation=90,
            color="black",
            zorder=8,
        )
    axis.set_xlim(0, 424)
    axis.set_ylim(display_y, metadata["finite_y_max"])
    axis.set_xticks(np.arange(0, 425, 20))
    axis.set_xlabel("State ID")
    axis.set_ylabel("Metric (dB)")
    axis.set_title("Scenario 2 C2-to-Aend State Metrics and Retrieval Results")
    axis.grid(True, alpha=0.2, linewidth=0.5)
    axis.legend(loc="upper left", ncol=3, fontsize=8, handlelength=2.8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    metadata.update(
        {
            "output_path": str(output_path),
            "display_y": float(display_y),
            "curve_count": 6,
            "q_annotation_count": int(finite_mask.sum()),
            "annotation_rule": (
                "all finite retrieved markers labeled Q=state_id_Q; "
                "exact hits shown in bottom strip"
            ),
            "threshold_scope": "-40 dB applies only to retrieved_real_B_CNMSE_dB",
            "ylim": [float(display_y), metadata["finite_y_max"]],
        }
    )
    return metadata


def _append_log(metadata: dict[str, Any], before: dict[str, Any], after: dict[str, Any]) -> None:
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with WORK_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] Scenario 2 state metrics Figure 5\n")
        handle.write(
            "纯绘图维护：只读取state_summary Excel，未读取raw、未重算模型/检索/指标；"
            f"figure={metadata['output_path']}；curves=6；"
            f"Q annotations={metadata['q_annotation_count']}。\n"
        )
        handle.write(
            f"state_count=425；exact=-Inf={metadata['exact_count']}；finite retrieved="
            f"{metadata['finite_retrieved_count']}；failure={metadata['failure_count']}；"
            f"shareable={metadata['shareable_count']}；ylim={metadata['ylim']}；"
            "threshold scope=Real-B CNMSE only。\n"
        )
        handle.write(
            "保护复核：input Excel SHA unchanged="
            f"{before['input_excel_sha256'] == after['input_excel_sha256']}；"
            "Figure1~4 SHA unchanged="
            f"{before['previous_figure_sha256'] == after['previous_figure_sha256']}；"
            "Exact -Inf仅用于底部显示strip，未改写正式值；未执行破坏性Git操作。\n"
        )


def _append_handoff(metadata: dict[str, Any]) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n{timestamp} | behavior_fingerprint_lut_retrieval持续维护："
            "Figure5状态指标汇总图\n"
        )
        handle.write(
            f"基于scenario_2_C2_to_Aend_state_summary.xlsx生成单Axes六曲线图，"
            "207个finite retrieved marker均标注Q，Exact -Inf底部strip，"
            "-40 dB仅作Real-B CNMSE阈值；"
            f"输出{metadata['output_path']}，输入Excel和Figure1~4保护不变。\n"
        )


def main() -> None:
    before = _protected_snapshot()
    frame = _read_excel_with_bundled_python(INPUT_XLSX)
    metadata = plot_state_metrics_and_retrieval(frame, OUTPUT_FIGURE)
    after = _protected_snapshot()
    _verify_protected(before, after)
    _append_log(metadata, before, after)
    _append_handoff(metadata)
    report = {
        key: metadata[key]
        for key in (
            "output_path",
            "curve_count",
            "q_annotation_count",
            "exact_count",
            "finite_retrieved_count",
            "failure_count",
            "shareable_count",
            "finite_y_min",
            "finite_y_max",
            "display_y",
            "ylim",
        )
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
