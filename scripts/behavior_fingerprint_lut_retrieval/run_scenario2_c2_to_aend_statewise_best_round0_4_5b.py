"""Run frozen Round 0--4 statewise-best 5B retrieval and evidence outputs."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_model.select_statewise_best_round0_4_models import (  # noqa: E402
    RESULT_ROOT as MODEL_SELECTION_ROOT,
)

from behavior_fingerprint_lut_retrieval.scenario2_c2_to_aend_statewise_best_round0_4_5b import (  # noqa: E402  # noqa: E402
    BASELINE_ROOT,
    DPD_SHAREABLE_THRESHOLD_DB,
    FORMAL_B_LENGTH,
    RESULT_ROOT,
    STATE_COUNT,
    STATE_IDS,
    _load_selected_inputs,
    build_common_support_diagnostics,
    build_fingerprints,
    build_retrieval_tables,
    compute_generic_cnmse_distance_matrix,
    load_canonical_real_b,
    stable_top1,
)
from behavior_fingerprint_lut_retrieval.scenario2_retrieval_oriented_model_scan import (  # noqa: E402
    load_raw_scalar_metrics,
)

MODEL_SOURCE_ROOT = (
    PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_statewise_adaptive_Aend_C2_5B"
)
MODEL_SELECTION_VALIDATION = MODEL_SELECTION_ROOT / "selection_validation.json"
EXCEL_SOURCE_PATH = RESULT_ROOT / "excel_source.json"
EXCEL_OUTPUT_PATH = RESULT_ROOT / "scenario_2_C2_to_Aend_statewise_best_round0_4_5B_retrieval.xlsx"


PROTECTED_RESULT_DIRS = {
    "scenario_2_C2_to_Aend": BASELINE_ROOT,
    "scenario_2_C2_to_Aend_1B": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_1B",
    "scenario_2_C2_to_Aend_0p5B": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_0p5B",
    "scenario_2_C2_to_Aend_equal_ABC": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_equal_ABC",
    "scenario_2_C2_to_Aend_unified_model_capacity": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_unified_model_capacity",
    "scenario_2_C2_to_Aend_retrieval_oriented_model_scan": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_retrieval_oriented_model_scan",
    "scenario_2_all_ilc": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_all_ilc",
    "low_bandwidth_operator_bank": PROJECT_ROOT
    / "results"
    / "low_bandwidth_observation"
    / "sample_rate_operator_bank",
}


def _tree_digest(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*"), key=lambda item: str(item).lower()):
        if not file.is_file() or "__pycache__" in file.parts or file.suffix.lower() == ".pyc":
            continue
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
    root = Path(r"\\?\{}".format(PROJECT_ROOT / "data" / "raw"))
    digest = hashlib.sha256()
    files = sorted(
        (file for file in root.rglob("*") if file.is_file()), key=lambda item: str(item).lower()
    )
    total = 0
    for file in files:
        size = int(file.stat().st_size)
        digest.update(str(file.relative_to(root)).replace("\\", "/").encode() + b"\0")
        digest.update(size.to_bytes(8, "little"))
        total += size
    return {"sha256": digest.hexdigest(), "file_count": len(files), "bytes": total}


def _snapshot() -> dict[str, Any]:
    return {
        "data/raw": _raw_manifest(),
        "result_dirs": {
            name: {"path": str(path), "exists": path.is_dir(), "sha256": _tree_digest(path)}
            for name, path in PROTECTED_RESULT_DIRS.items()
        },
    }


def _verify(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    raw = {
        "sha256_unchanged": before["data/raw"]["sha256"] == after["data/raw"]["sha256"],
        "file_count_unchanged": before["data/raw"]["file_count"] == after["data/raw"]["file_count"],
        "bytes_unchanged": before["data/raw"]["bytes"] == after["data/raw"]["bytes"],
    }
    result_dirs = {
        name: {
            "sha256_unchanged": item["sha256"] == after["result_dirs"][name]["sha256"],
            "before": item["sha256"],
            "after": after["result_dirs"][name]["sha256"],
        }
        for name, item in before["result_dirs"].items()
    }
    return {
        "data/raw": raw,
        "result_dirs": result_dirs,
        "all_protected_unchanged": bool(
            all(raw.values()) and all(item["sha256_unchanged"] for item in result_dirs.values())
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NaN", float_format="%.17g")


def _plot_base(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": stem.with_suffix(".png"),
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
    }
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(path) for key, path in outputs.items()}


def _plot_main(result: pd.DataFrame, stem: Path, common_delay: int) -> dict[str, Any]:
    """Nature-style quantitative statewise grid/trend figure, Python-only."""

    columns = (
        "nmse_withoutdpd_dB",
        "Y_Aend_train_NMSE_dB",
        "Y_Aend_B_NMSE_dB",
        "Y_C2_train_NMSE_dB",
        "Y_C2_B_NMSE_dB",
        "retrieved_real_B_CNMSE_dB",
    )
    labels = (
        "NMSE without DPD",
        "Y-Aend train",
        "Y-Aend B generalization",
        "Y-C2 train",
        "Y-C2 B generalization",
        "Retrieved 5B Real-B CNMSE",
    )
    colors = ("#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#B279A2")
    markers = ("o", "s", "^", "D", "P", "X")
    linestyles = ("-", "-", "--", "-", "--", "-")
    x = result["state_id_R"].to_numpy(dtype=float)
    finite_values = np.concatenate(
        [
            result[column].to_numpy(dtype=float)[np.isfinite(result[column].to_numpy(dtype=float))]
            for column in columns
        ]
    )
    floor = float(np.min(finite_values) - 2.0)
    ceiling = float(np.max(finite_values) + 2.0)
    plot_values = {column: result[column].to_numpy(dtype=float).copy() for column in columns}
    exact = np.isneginf(plot_values["retrieved_real_B_CNMSE_dB"])
    plot_values["retrieved_real_B_CNMSE_dB"][exact] = floor
    fig, ax = plt.subplots(figsize=(30, 12))
    for column, label, color, marker, linestyle in zip(
        columns, labels, colors, markers, linestyles, strict=True
    ):
        ax.plot(
            x,
            plot_values[column],
            color=color,
            linewidth=1.0,
            linestyle=linestyle,
            marker=marker,
            markersize=2.8,
            markevery=(1 if column == "retrieved_real_B_CNMSE_dB" else 12),
            alpha=0.9,
            label=label,
        )
    q_values = result["state_id_Q"].to_numpy(dtype=np.int64)
    retrieved_values = plot_values["retrieved_real_B_CNMSE_dB"]
    for xi, yi, q in zip(x, retrieved_values, q_values, strict=True):
        ax.annotate(
            f"Q={int(q)}",
            xy=(xi, yi),
            xytext=(0, 2),
            textcoords="offset points",
            rotation=90,
            fontsize=3.0,
            color=colors[-1],
            ha="center",
            va="bottom",
            alpha=0.72,
        )
    ax.axhline(-40.0, color="#666666", linestyle=":", linewidth=1.0, label="-40 dB threshold")
    ax.set_xlim(-2, STATE_COUNT + 1)
    ax.set_ylim(floor - 1.0, ceiling)
    ax.set_xlabel("State ID (State_R)")
    ax.set_ylabel("Metric (dB)")
    ax.set_title("Statewise-Best-Available MP Model and 5B Retrieval Metrics")
    ax.text(
        0.995,
        0.02,
        f"Round 0–4 frozen candidate pool; M_common={common_delay}; "
        "Exact self (-Inf) shown at plotting floor only",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=8)
    ax.grid(False)
    fig.tight_layout()
    return {
        "core_conclusion": (
            "Statewise-best-available Aend/C2 models determine the full-rate 5B retrieval outcome."
        ),
        "archetype": "quantitative grid/trend",
        "backend": "Python/matplotlib",
        "panel_map": {"single_axes": "six statewise dB curves and the -40 dB threshold"},
        "q_labels": "all 425 retrieved markers labeled Q=state_id_Q",
        "exact_plot_rule": "-Inf retained in data and mapped to plotting floor only",
        "y_floor": floor,
        "y_ceiling": ceiling,
        "outputs": _plot_base(fig, stem),
    }


def _plot_complexity(frame: pd.DataFrame, side: str, stem: Path) -> dict[str, Any]:
    selected = frame.loc[frame["side"] == side].sort_values("state_id").reset_index(drop=True)
    if selected.shape[0] != STATE_COUNT:
        raise RuntimeError(f"{side} complexity plot requires 425 rows")
    x = selected["state_id"].to_numpy(dtype=float)
    lambdas = selected["lambda"].to_numpy(dtype=float)
    positive = lambdas[lambdas > 0]
    floor = float(np.min(positive) / 10.0) if positive.size else 1e-15
    log_lambda = np.log10(np.where(lambdas > 0, lambdas, floor))
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    axes[0].plot(
        x, selected["coefficient_count"], color="#4C78A8", marker="o", markersize=2.0, linewidth=0.8
    )
    axes[0].set_ylabel("Complex coefficients")
    axes[1].plot(
        x, selected["max_delay"], color="#F58518", marker="s", markersize=2.0, linewidth=0.8
    )
    axes[1].set_ylabel("max_delay")
    axes[2].plot(x, log_lambda, color="#54A24B", marker="^", markersize=2.0, linewidth=0.8)
    axes[2].set_ylabel("log10(lambda)")
    axes[2].set_xlabel("State ID")
    fig.suptitle(f"Selected {side} Model Complexity")
    fig.tight_layout()
    return {"side": side, "outputs": _plot_base(fig, stem)}


def _build_summary(
    result: pd.DataFrame,
    diagnostics: pd.DataFrame,
    aend: pd.DataFrame,
    c2: pd.DataFrame,
    common_delay: int,
    provenance: dict[str, Any],
    baseline_counts: dict[str, int],
) -> dict[str, Any]:
    real_b = result["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    exact = result["state_id_Q"].to_numpy(dtype=np.int64) == result["state_id_R"].to_numpy(
        dtype=np.int64
    )
    shareable = np.isneginf(real_b) | (real_b < DPD_SHAREABLE_THRESHOLD_DB)
    nonexact = ~exact
    nonexact_shareable = nonexact & shareable
    finite_nonexact = real_b[nonexact & np.isfinite(real_b)]
    return {
        "experiment": "scenario_2_C2_to_Aend_statewise_best_round0_4_5B",
        "bandwidth": "5B",
        "low_bandwidth_operator_used": False,
        "ABC_changed": False,
        "candidate_pool_source": str(MODEL_SOURCE_ROOT),
        "persisted_round_max": int(provenance["persisted_search_round_max"]),
        "candidate_count": int(provenance["persisted_candidate_count"]),
        "selected_Aend_count": int(aend.shape[0]),
        "selected_C2_count": int(c2.shape[0]),
        "native_feasible_Aend_count": int(aend["joint_feasible_native"].sum()),
        "native_feasible_C2_count": int(c2["joint_feasible_native"].sum()),
        "M_common": int(common_delay),
        "fingerprint_length": int(FORMAL_B_LENGTH - common_delay),
        "common_A_length": int(12288 - common_delay),
        "common_C_length": int(7373 - common_delay),
        "Exact": int(exact.sum()),
        "Exact_rate": float(exact.mean()),
        "Shareable": int(shareable.sum()),
        "Shareable_rate": float(shareable.mean()),
        "Nonexact": int(nonexact.sum()),
        "Nonexact_shareable": int(nonexact_shareable.sum()),
        "Nonexact_shareable_rate": float(nonexact_shareable.sum() / max(1, nonexact.sum())),
        "Failure": int((~shareable).sum()),
        "failure_state_ids": result.loc[~shareable, "state_id_R"].astype(int).tolist(),
        "baseline_Exact": baseline_counts["Exact"],
        "baseline_Shareable": baseline_counts["Shareable"],
        "baseline_Failure": baseline_counts["Failure"],
        "delta_Exact": int(exact.sum() - baseline_counts["Exact"]),
        "delta_Shareable": int(shareable.sum() - baseline_counts["Shareable"]),
        "delta_Failure": int((~shareable).sum() - baseline_counts["Failure"]),
        "nonexact_real_B_median_dB": float(np.median(finite_nonexact))
        if finite_nonexact.size
        else None,
        "nonexact_real_B_q95_dB": float(np.quantile(finite_nonexact, 0.95))
        if finite_nonexact.size
        else None,
        "common_support_validation_completed": bool(diagnostics.shape[0] == STATE_COUNT),
        "common_support_used_to_trigger_new_search": False,
    }


def _build_excel_source(
    result: pd.DataFrame,
    selected_models: pd.DataFrame,
    common_diag: pd.DataFrame,
    retrieval_diag: pd.DataFrame,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    main_columns = [
        "状态序号",
        "负载配置",
        "NMSE(NMSE_WITHOUTDPD)",
        "上下边带ACPR平均值withoutdpd",
        "A段建模精度",
        "AB段泛化精度",
        "C段建模精度",
        "CB段泛化精度",
        "命中的序号",
        "命中的CNMSE",
    ]
    main_rows: list[list[Any]] = []
    for row in result.to_dict("records"):
        value = row["retrieved_real_B_CNMSE_dB"]
        main_rows.append(
            [
                int(row["state_id_R"]),
                row["load_config"],
                float(row["nmse_withoutdpd_dB"]),
                float(row["acpr_withoutdpd_avg_dBc"]),
                float(row["Y_Aend_train_NMSE_dB"]),
                float(row["Y_Aend_B_NMSE_dB"]),
                float(row["Y_C2_train_NMSE_dB"]),
                float(row["Y_C2_B_NMSE_dB"]),
                int(row["state_id_Q"]),
                "-Inf" if np.isneginf(float(value)) else float(value),
            ]
        )
    summary_rows = [
        [
            str(key),
            json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value,
        ]
        for key, value in summary.items()
    ]
    return {
        "sheets": {
            "retrieval_results": {"columns": main_columns, "rows": main_rows},
            "selected_models": {
                "columns": selected_models.columns.tolist(),
                "rows": selected_models.where(pd.notna(selected_models), None).to_dict("records"),
            },
            "common_support_diagnostics": {
                "columns": common_diag.columns.tolist(),
                "rows": common_diag.where(pd.notna(common_diag), None).to_dict("records"),
            },
            "retrieval_diagnostics": {
                "columns": retrieval_diag.columns.tolist(),
                "rows": retrieval_diag.assign(
                    top1_fingerprint_CNMSE_dB=retrieval_diag["top1_fingerprint_CNMSE_dB"].map(
                        lambda value: "-Inf" if np.isneginf(float(value)) else float(value)
                    ),
                    retrieved_real_B_CNMSE_dB=retrieval_diag["retrieved_real_B_CNMSE_dB"].map(
                        lambda value: "-Inf" if np.isneginf(float(value)) else float(value)
                    ),
                )
                .where(pd.notna(retrieval_diag), None)
                .to_dict("records"),
            },
            "summary": {"columns": ["metric", "value"], "rows": summary_rows},
        },
        "metadata": {
            "main_columns": main_columns,
            "state_count": STATE_COUNT,
            "selected_metrics_are_native_from_one_actual_candidate": True,
            "independent_metric_minima_used_for_model_selection": False,
            "round_5_used": False,
            "additional_model_search_performed": False,
        },
    }


def _spot_checks(distance: np.ndarray, state_q: np.ndarray, real_b: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(20260908)
    sampled = np.sort(rng.choice(STATE_COUNT, size=20, replace=False)).astype(np.int64)
    top1_ok = True
    real_b_ok = True
    rows: list[dict[str, Any]] = []
    for state_id in sampled:
        order = np.lexsort((np.arange(STATE_COUNT, dtype=np.int64), distance[int(state_id)]))
        expected_q = int(order[0])
        top1_ok &= expected_q == int(state_q[int(state_id)])
        expected_real_b = float(load_canonical_real_b()[int(state_id), expected_q])
        observed_real_b = float(real_b[int(state_id)])
        if np.isneginf(expected_real_b) and np.isneginf(observed_real_b):
            rb_match = True
        else:
            rb_match = bool(abs(expected_real_b - observed_real_b) <= 1e-12)
        real_b_ok &= rb_match
        rows.append(
            {
                "state_id_R": int(state_id),
                "expected_Q": expected_q,
                "observed_Q": int(state_q[int(state_id)]),
                "real_B_match": rb_match,
            }
        )
    return {
        "sample_count": 20,
        "sample_state_ids": sampled.tolist(),
        "top1_recomputed_equal": bool(top1_ok),
        "real_B_lookup_equal": bool(real_b_ok),
        "rows": rows,
    }


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    before = _snapshot()
    if not MODEL_SELECTION_VALIDATION.is_file():
        raise FileNotFoundError("selected model validation is missing")
    selection_validation = json.loads(MODEL_SELECTION_VALIDATION.read_text(encoding="utf-8"))
    if selection_validation.get("all_850_refit") is not True:
        raise RuntimeError("selected model refit validation is not complete")
    if (
        selection_validation.get("persisted_search_round_max") != 4
        or selection_validation.get("persisted_candidate_count") != 1064
    ):
        raise RuntimeError("selected model source is not the frozen Round 0--4 / 1064 pool")
    aend, c2, theta_a, theta_c, common_b, provenance = _load_selected_inputs()
    common_delay = int(max(aend["max_delay"].max(), c2["max_delay"].max()))
    common_diag = build_common_support_diagnostics(aend, c2, theta_a, theta_c, common_delay)
    lut_fp = build_fingerprints(aend, theta_a, common_b, common_delay)
    query_fp = build_fingerprints(c2, theta_c, common_b, common_delay)
    distance = compute_generic_cnmse_distance_matrix(query_fp, lut_fp)
    state_q, top1_distance, tie_count, ranking_order = stable_top1(distance)
    real_b_matrix = load_canonical_real_b()
    retrieved_real_b = real_b_matrix[STATE_IDS, state_q]
    raw_scalars = load_raw_scalar_metrics(range(STATE_COUNT))
    retrieval_result, retrieval_diag = build_retrieval_tables(
        aend,
        c2,
        common_diag,
        state_q,
        top1_distance,
        retrieved_real_b,
        raw_scalars,
    )
    retrieval_result["exact_hit"] = retrieval_diag["exact_hit"].to_numpy(dtype=bool)
    retrieval_result["dpd_shareable"] = retrieval_diag["dpd_shareable"].to_numpy(dtype=bool)
    selected_models = (
        pd.concat([aend, c2], ignore_index=True)
        .sort_values(["side", "state_id"])
        .reset_index(drop=True)
    )
    selected_model_diagnostics = selected_models.merge(
        common_diag,
        on="state_id",
        how="left",
        validate="many_to_one",
    )
    _write_frame(
        retrieval_result.drop(columns=["exact_hit", "dpd_shareable"]),
        RESULT_ROOT / "retrieval_results_statewise_best_round0_4_5B.csv",
    )
    _write_frame(retrieval_diag, RESULT_ROOT / "top1_fingerprint_retrieval.csv")
    _write_frame(common_diag, RESULT_ROOT / "common_support_diagnostics.csv")
    _write_frame(selected_model_diagnostics, RESULT_ROOT / "selected_model_diagnostics.csv")
    _write_frame(
        pd.DataFrame(
            {
                "state_id": STATE_IDS,
                "Aend_native_feasible": aend["joint_feasible_native"].to_numpy(dtype=bool),
                "C2_native_feasible": c2["joint_feasible_native"].to_numpy(dtype=bool),
                "both_native_feasible": aend["joint_feasible_native"].to_numpy(dtype=bool)
                & c2["joint_feasible_native"].to_numpy(dtype=bool),
            }
        ),
        RESULT_ROOT / "selected_model_feasibility.csv",
    )

    def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
        np.savez_compressed(path, **arrays)

    write_npz(
        RESULT_ROOT / "lut_fingerprints_statewise_best_5B.npz",
        {
            "state_ids": STATE_IDS,
            "common_B_input": common_b,
            "Y_Aend_fingerprints": lut_fp,
            "M_common": np.asarray([common_delay], dtype=np.int64),
        },
    )
    write_npz(
        RESULT_ROOT / "query_fingerprints_statewise_best_5B.npz",
        {
            "state_ids": STATE_IDS,
            "common_B_input": common_b,
            "Q_C2": query_fp,
            "M_common": np.asarray([common_delay], dtype=np.int64),
        },
    )
    write_npz(
        RESULT_ROOT / "fingerprint_cnmse_matrix.npz",
        {
            "state_ids": STATE_IDS,
            "D_C2_Aend": distance,
            "R_C2_Aend": np.asarray(ranking_order, dtype=np.int64),
            "tie_count": tie_count,
        },
    )
    write_npz(
        RESULT_ROOT / "real_B_distance_matrix_reused.npz",
        {"state_ids": STATE_IDS, "D_B": real_b_matrix},
    )
    _write_json(
        RESULT_ROOT / "fingerprint_metadata.json",
        {
            "M_common": common_delay,
            "fingerprint_length": FORMAL_B_LENGTH - common_delay,
            "fingerprint_support": f"B[{common_delay}:]",
            "probe_source": str(BASELINE_ROOT / "lut_fingerprints_Aend.npz"),
            "selected_model_source": str(MODEL_SELECTION_ROOT),
            "candidate_pool_round_max": 4,
            "candidate_pool_count": 1064,
        },
    )
    baseline_results = pd.read_csv(BASELINE_ROOT / "retrieval_results.csv")
    baseline_diag = pd.read_csv(BASELINE_ROOT / "retrieved_real_B_cnmse.csv")
    baseline_real = baseline_diag.sort_values("State_n_R")["retrieved_real_B_CNMSE_dB"].to_numpy(
        dtype=float
    )
    baseline_counts = {
        "Exact": int(baseline_results["exact_hit"].sum()),
        "Shareable": int(
            (np.isneginf(baseline_real) | (baseline_real < DPD_SHAREABLE_THRESHOLD_DB)).sum()
        ),
        "Failure": int(
            (~(np.isneginf(baseline_real) | (baseline_real < DPD_SHAREABLE_THRESHOLD_DB))).sum()
        ),
    }
    summary = _build_summary(
        retrieval_result, common_diag, aend, c2, common_delay, provenance, baseline_counts
    )
    _write_json(RESULT_ROOT / "retrieval_summary.json", summary)
    _write_json(RESULT_ROOT / "spot_checks.json", _spot_checks(distance, state_q, retrieved_real_b))
    figure_details = {
        "main": _plot_main(
            retrieval_result,
            RESULT_ROOT / "figure_statewise_best_round0_4_5B_model_and_retrieval_metrics",
            common_delay,
        ),
        "Aend_complexity": _plot_complexity(
            aend, "Aend", RESULT_ROOT / "figure_selected_Aend_model_complexity"
        ),
        "C2_complexity": _plot_complexity(
            c2, "C2", RESULT_ROOT / "figure_selected_C2_model_complexity"
        ),
    }
    excel_source = _build_excel_source(
        retrieval_result,
        selected_models,
        common_diag,
        retrieval_diag,
        summary,
    )
    excel_source["metadata"].update(
        {
            "candidate_pool_frozen": True,
            "round_5_used": False,
            "additional_model_search_performed": False,
            "real_B_used_only_after_State_Q_fixed": True,
            "figure_backend": "Python/matplotlib",
        }
    )
    _write_json(EXCEL_SOURCE_PATH, excel_source)
    after = _snapshot()
    protection = _verify(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw or protected old results changed")
    validation = {
        **summary,
        "experiment": "scenario_2_C2_to_Aend_statewise_best_round0_4_5B",
        "statewise_model_selection": True,
        "Aend_and_C2_selected_independently": True,
        "one_actual_candidate_per_state_side": True,
        "independent_metric_minima_used_for_model_selection": False,
        "feasible_selection_rule": "minimum_complexity_among_joint_lt_minus40",
        "fallback_selection_rule": "minimum_worst_native_NMSE",
        "candidate_pool_frozen": True,
        "round_5_used": False,
        "additional_model_search_performed": False,
        "global_common_max_delay": common_delay,
        "fingerprint_support": f"B[{common_delay}:]",
        "common_support_used_for_fingerprint_comparison": True,
        "common_support_used_to_trigger_new_search": False,
        "OFF_RealB_used_for_model_selection": False,
        "OFF_RealB_used_for_retrieval_selection": False,
        "OFF_RealB_used_only_after_State_Q_fixed": True,
        "top1_selection_source": "fingerprint CNMSE only",
        "real_B_matrix_reused": True,
        "real_B_matrix_source": str(BASELINE_ROOT / "real_B_distance_matrix.npz"),
        "selected_model_refit_validation": json.loads(
            (MODEL_SELECTION_ROOT / "selection_validation.json").read_text(encoding="utf-8")
        ),
        "spot_checks": json.loads((RESULT_ROOT / "spot_checks.json").read_text(encoding="utf-8")),
        "figures": figure_details,
        "protection_before": before,
        "protection_after": after,
        "protection_verification": protection,
        "output_files": {
            path.name: str(path) for path in sorted(RESULT_ROOT.iterdir()) if path.is_file()
        },
    }
    _write_json(RESULT_ROOT / "validation.json", validation)
    print(
        json.dumps(
            {
                "M_common": common_delay,
                "fingerprint_length": FORMAL_B_LENGTH - common_delay,
                "Exact": summary["Exact"],
                "Shareable": summary["Shareable"],
                "Failure": summary["Failure"],
                "baseline": baseline_counts,
                "selected_Aend": 425,
                "selected_C2": 425,
                "protection": protection["all_protected_unchanged"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
