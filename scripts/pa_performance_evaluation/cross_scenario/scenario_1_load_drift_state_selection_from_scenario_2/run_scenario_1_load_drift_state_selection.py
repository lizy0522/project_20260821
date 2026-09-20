"""Select representative Scenario 1 load-drift states from Scenario 2 data."""

# ruff: noqa: E402,E501

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir())
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.shared.metrics import cnmse  # noqa: E402
from data_management.shared import build_state_table, get_state_id, load_by_id  # noqa: E402
from signal_segmentation.shared import build_off_segments, build_partition_from_xin  # noqa: E402

from pa_performance_evaluation.cross_scenario.scenario_1_load_drift_state_selection_from_scenario_2.plot_load_drift_selection import (  # noqa: E402
    write_plots,
)

TASK_NAME = "scenario_1_load_drift_state_selection_from_scenario_2"
RESULT_ROOT = PROJECT_ROOT / "results" / "pa_performance_evaluation" / "cross_scenario" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "pa_performance_evaluation" / "cross_scenario" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
STATE_COUNT = 425
PHASES = (0, 45, 90, 135, 180, 225, 270, 315)
FUN_LEVELS = (0, 10, 20)
SEC_MNG = 0
SEC_ANG = 0
PROTECTION_OUTPUT_POWER_DBM = 37.0
OWN_ACPR_TARGET_DB = -50.0
EXPECTED_RAW = {"sha256": "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0", "file_count": 429, "mat_count": 427, "bytes": 2_258_448_137}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON serialize {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _append_log(message: str) -> None:
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WORK_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} | {message.rstrip()}\n")


def _append_handoff(message: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{_now()} | 新任务：{TASK_NAME}\n{message.rstrip()}\n")


def _write_checkpoint(phase: str, **payload: object) -> None:
    _write_json(RESULT_ROOT / "09_checkpoint.json", {"task_name": TASK_NAME, "phase": phase, **payload})


def _raw_manifest() -> dict[str, object]:
    root = PROJECT_ROOT / "data" / "raw"
    digest = hashlib.sha256()
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: str(p).lower())
    total = 0
    mats = 0
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        size = path.stat().st_size
        digest.update(int(size).to_bytes(8, "little"))
        total += size
        mats += path.suffix.lower() == ".mat"
    return {"sha256": digest.hexdigest(), "file_count": len(files), "mat_count": mats, "bytes": total}


def _raw_gate() -> dict[str, object]:
    value = _raw_manifest()
    if value != EXPECTED_RAW:
        raise RuntimeError(f"raw manifest changed: {value}")
    return value


def _scalar(data: dict[str, Any], name: str) -> float:
    value = np.asarray(data[name]).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]):
        raise ValueError(f"{name} must be one finite scalar")
    return float(value[0])


def _acpr_avg(data: dict[str, Any], prefix: str) -> float:
    return (_scalar(data, f"acpr_low_{prefix}") + _scalar(data, f"acpr_upper_{prefix}")) / 2.0


def _candidate_id(fun_mng: int, fun_ang: int) -> str:
    if fun_mng == 0:
        return "L0"
    return f"L{fun_mng // 10:02d}_{fun_ang:03d}"


def _load_label(fun_mng: int, fun_ang: int) -> str:
    if fun_mng == 0:
        return "matched"
    return f"{fun_mng / 100:.1f}∠{fun_ang}°"


def _level_label(fun_mng: int) -> str:
    return {0: "nominal", 10: "light", 20: "moderate"}[fun_mng]


def _candidate_specs() -> list[dict[str, object]]:
    table = build_state_table()
    specs: list[dict[str, object]] = []
    for fun_mng in FUN_LEVELS:
        for fun_ang in ((0,) if fun_mng == 0 else PHASES):
            state_id = get_state_id(fun_mng, fun_ang, SEC_MNG, SEC_ANG, 2.30, -21)
            row = table[state_id]
            if int(row["state_id"]) != state_id:
                raise RuntimeError("canonical state table mismatch")
            specs.append({"Candidate_ID": _candidate_id(fun_mng, fun_ang), "State": state_id, "funMng": fun_mng, "funAng_deg": fun_ang, "secMng": SEC_MNG, "secAng_deg": SEC_ANG, "load_label": _load_label(fun_mng, fun_ang), "load_level": _level_label(fun_mng)})
    if len(specs) != 17:
        raise RuntimeError(f"candidate count changed: {len(specs)}")
    if [int(item["State"]) for item in specs] != [0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255, 272]:
        raise RuntimeError("candidate canonical State mapping differs from frozen expected mapping")
    return specs


def _canonical_real_b(state_id: int) -> np.ndarray:
    data = load_by_id(state_id)
    partition = build_partition_from_xin(np.asarray(data["xin"]))
    off = build_off_segments(data, partition)
    value = np.asarray(off["B"].output[2:], dtype=np.complex128)
    if value.shape != (4913,) or not np.all(np.isfinite(value)):
        raise RuntimeError(f"state {state_id} canonical Real-B shape/finite failure")
    return value


def _quantile_level(value: float, q33: float, q66: float) -> str:
    if value <= q33:
        return "small"
    if value <= q66:
        return "medium"
    return "large"


def _selection_note(row: pd.Series) -> str:
    if row["load_level"] == "nominal":
        return "nominal reference state"
    return f"{row['load_level']} mismatch; stale-DPD {row['stale_mismatch_level']} degradation; own DPD recovered"


def _safe_records(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record in dataframe.to_dict(orient="records"):
        safe: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, (float, np.floating)):
                if np.isneginf(value):
                    safe[key] = "-Inf"
                elif np.isposinf(value):
                    safe[key] = "Inf"
                elif np.isnan(value):
                    safe[key] = None
                else:
                    safe[key] = float(value)
            elif isinstance(value, np.generic):
                safe[key] = value.item()
            elif pd.isna(value):
                safe[key] = None
            else:
                safe[key] = value
        records.append(safe)
    return records


def _build_candidate_frame(specs: list[dict[str, object]], nominal_b: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    b_signals: dict[int, np.ndarray] = {int(spec["State"]): _canonical_real_b(int(spec["State"])) for spec in specs}
    for spec in specs:
        state_id = int(spec["State"])
        data = load_by_id(state_id)
        own_low = _scalar(data, "acpr_low_withdpd")
        own_upper = _scalar(data, "acpr_upper_withdpd")
        stale_low = _scalar(data, "acpr_low_withdpd_stale")
        stale_upper = _scalar(data, "acpr_upper_withdpd_stale")
        raw = {**spec, "NMSE_withoutDPD_dB": _scalar(data, "nmse_withoutdpd"), "ACPR_low_withoutDPD_dBc": _scalar(data, "acpr_low_withoutdpd"), "ACPR_upper_withoutDPD_dBc": _scalar(data, "acpr_upper_withoutdpd"), "ACPR_avg_withoutDPD_dBc": _acpr_avg(data, "withoutdpd"), "outputPower_withoutDPD_dBm": _scalar(data, "outputPower_withoutdpd"), "i_withoutDPD_A": _scalar(data, "i_withoutdpd"), "dcPower_withoutDPD_W": _scalar(data, "dcPower_withoutdpd"), "efficiency_withoutDPD_pct": _scalar(data, "efficiency_withoutdpd"), "NMSE_stale_dB": _scalar(data, "nmse_withdpd_stale"), "ACPR_low_stale_dBc": stale_low, "ACPR_upper_stale_dBc": stale_upper, "ACPR_avg_stale_dBc": (stale_low + stale_upper) / 2.0, "outputPower_stale_dBm": _scalar(data, "outputPower_withdpd_stale"), "i_stale_A": _scalar(data, "i_withdpd_stale"), "efficiency_stale_pct": _scalar(data, "efficiency_withdpd_stale"), "NMSE_ownDPD_dB": _scalar(data, "nmse_withdpd"), "ACPR_low_ownDPD_dBc": own_low, "ACPR_upper_ownDPD_dBc": own_upper, "ACPR_avg_ownDPD_dBc": (own_low + own_upper) / 2.0, "ILC_iterations": int(_scalar(data, "nth_inter")), "own_DPD_target_reached": bool(own_low <= OWN_ACPR_TARGET_DB and own_upper <= OWN_ACPR_TARGET_DB), "outputPower_ownDPD_dBm": _scalar(data, "outputPower_withdpd"), "i_ownDPD_A": _scalar(data, "i_withdpd"), "efficiency_ownDPD_pct": _scalar(data, "efficiency_withdpd"), "CNMSE_B_to_nominal_dB": float(cnmse(nominal_b, b_signals[state_id]))}
        rows.append(raw)
    frame = pd.DataFrame(rows).sort_values("State").reset_index(drop=True)
    nominal_stale_nmse = float(frame.loc[frame["load_level"] == "nominal", "NMSE_stale_dB"].iloc[0])
    nominal_stale_acpr = float(frame.loc[frame["load_level"] == "nominal", "ACPR_avg_stale_dBc"].iloc[0])
    frame["Delta_NMSE_stale_vs_nominal_dB"] = frame["NMSE_stale_dB"] - nominal_stale_nmse
    frame["Delta_ACPR_stale_vs_nominal_dB"] = frame["ACPR_avg_stale_dBc"] - nominal_stale_acpr
    nonnominal = frame.loc[frame["load_level"] != "nominal"]
    behavior_values = nonnominal["CNMSE_B_to_nominal_dB"].to_numpy(dtype=float)
    q_behavior = np.quantile(behavior_values[np.isfinite(behavior_values)], [1 / 3, 2 / 3])
    nmse_values = nonnominal["Delta_NMSE_stale_vs_nominal_dB"].to_numpy(dtype=float)
    acpr_values = nonnominal["Delta_ACPR_stale_vs_nominal_dB"].to_numpy(dtype=float)
    q_nmse = np.quantile(nmse_values, [1 / 3, 2 / 3])
    q_acpr = np.quantile(acpr_values, [1 / 3, 2 / 3])
    frame["behavior_shift_level"] = ["nominal" if level == "nominal" else _quantile_level(float(value), *q_behavior) for level, value in zip(frame["load_level"], frame["CNMSE_B_to_nominal_dB"], strict=True)]
    stale_levels: list[str] = []
    for level, nmse_value, acpr_value in zip(frame["load_level"], frame["Delta_NMSE_stale_vs_nominal_dB"], frame["Delta_ACPR_stale_vs_nominal_dB"], strict=True):
        if level == "nominal":
            stale_levels.append("nominal")
        else:
            parts = {_quantile_level(float(nmse_value), *q_nmse), _quantile_level(float(acpr_value), *q_acpr)}
            stale_levels.append(max(parts, key={"small": 0, "medium": 1, "large": 2}.get))
    frame["stale_mismatch_level"] = stale_levels
    frame["design_margin_pass"] = (
        frame[["outputPower_withoutDPD_dBm", "outputPower_stale_dBm", "outputPower_ownDPD_dBm"]].max(axis=1) <= PROTECTION_OUTPUT_POWER_DBM
    ) & frame["own_DPD_target_reached"]
    frame["selection_note"] = frame.apply(_selection_note, axis=1)
    return frame


def _pairwise(frame: pd.DataFrame, signals: dict[int, np.ndarray]) -> np.ndarray:
    ids = frame["State"].to_numpy(dtype=int)
    matrix = np.empty((len(ids), len(ids)), dtype=float)
    for i, state_i in enumerate(ids):
        for j, state_j in enumerate(ids):
            matrix[i, j] = float("-inf") if i == j else cnmse(signals[state_i], signals[state_j])
    return matrix


def _phase_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for phase in PHASES:
        a = frame.loc[(frame["funMng"] == 10) & (frame["funAng_deg"] == phase)].iloc[0]
        b = frame.loc[(frame["funMng"] == 20) & (frame["funAng_deg"] == phase)].iloc[0]
        row: dict[str, object] = {"phase_deg": phase}
        for suffix, item in (("01", a), ("02", b)):
            row.update({f"State_{suffix}": int(item["State"]), f"NMSE_withoutDPD_{suffix}": item["NMSE_withoutDPD_dB"], f"ACPR_avg_withoutDPD_{suffix}": item["ACPR_avg_withoutDPD_dBc"], f"NMSE_stale_{suffix}": item["NMSE_stale_dB"], f"ACPR_avg_stale_{suffix}": item["ACPR_avg_stale_dBc"], f"Delta_NMSE_stale_{suffix}": item["Delta_NMSE_stale_vs_nominal_dB"], f"Delta_ACPR_stale_{suffix}": item["Delta_ACPR_stale_vs_nominal_dB"], f"CNMSE_to_nominal_{suffix}": item["CNMSE_B_to_nominal_dB"], f"own_DPD_ACPR_avg_{suffix}": item["ACPR_avg_ownDPD_dBc"], f"own_DPD_target_reached_{suffix}": bool(item["own_DPD_target_reached"])})
        rows.append(row)
    return pd.DataFrame(rows)


def _circular_min_sep(phases: tuple[int, int, int]) -> int:
    ordered = sorted(phases)
    gaps = [ordered[1] - ordered[0], ordered[2] - ordered[1], 360 + ordered[0] - ordered[2]]
    return int(min(gaps))


def _triplets(frame: pd.DataFrame, pairwise: np.ndarray) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    by_state = frame.set_index("State")
    candidate_ids = frame["State"].to_numpy(dtype=int).tolist()
    state_to_index = {state: index for index, state in enumerate(candidate_ids)}
    rows: list[dict[str, object]] = []
    for triplet_id, phases in enumerate(itertools.combinations(PHASES, 3), start=1):
        states = [0] + [int(by_state.loc[(10, phase), "State"]) if False else int(frame.loc[(frame["funMng"] == 10) & (frame["funAng_deg"] == phase), "State"].iloc[0]) for phase in phases] + [int(frame.loc[(frame["funMng"] == 20) & (frame["funAng_deg"] == phase), "State"].iloc[0]) for phase in phases]
        selected = frame[frame["State"].isin(states)]
        stale_acpr = selected.loc[selected["load_level"] != "nominal", "Delta_ACPR_stale_vs_nominal_dB"].to_numpy(dtype=float)
        stale_nmse = selected.loc[selected["load_level"] != "nominal", "Delta_NMSE_stale_vs_nominal_dB"].to_numpy(dtype=float)
        cnmse_values = selected.loc[selected["load_level"] != "nominal", "CNMSE_B_to_nominal_dB"].to_numpy(dtype=float)
        pair_values = []
        selected_indices = [state_to_index[state] for state in states[1:]]
        for i, left in enumerate(selected_indices):
            for right in selected_indices[i + 1 :]:
                if np.isfinite(pairwise[left, right]):
                    pair_values.append(float(pairwise[left, right]))
        behavior_coverage = sorted(set(selected.loc[selected["load_level"] != "nominal", "behavior_shift_level"].astype(str)))
        stale_coverage = sorted(set(selected.loc[selected["load_level"] != "nominal", "stale_mismatch_level"].astype(str)))
        rows.append({"triplet_id": triplet_id, "phase_1": phases[0], "phase_2": phases[1], "phase_3": phases[2], "state_01_phase1": states[1], "state_01_phase2": states[2], "state_01_phase3": states[3], "state_02_phase1": states[4], "state_02_phase2": states[5], "state_02_phase3": states[6], "all_design_margin_pass": bool(selected["design_margin_pass"].all()), "all_own_DPD_target_reached": bool(selected["own_DPD_target_reached"].all()), "mean_stale_ACPR_degradation_dB": float(np.mean(stale_acpr)), "min_stale_ACPR_degradation_dB": float(np.min(stale_acpr)), "max_stale_ACPR_degradation_dB": float(np.max(stale_acpr)), "range_stale_ACPR_degradation_dB": float(np.ptp(stale_acpr)), "mean_stale_NMSE_degradation_dB": float(np.mean(stale_nmse)), "range_stale_NMSE_degradation_dB": float(np.ptp(stale_nmse)), "mean_CNMSE_to_nominal_dB": float(np.mean(cnmse_values)), "min_CNMSE_to_nominal_dB": float(np.min(cnmse_values)), "max_CNMSE_to_nominal_dB": float(np.max(cnmse_values)), "pairwise_CNMSE_mean_dB": float(np.mean(pair_values)), "pairwise_CNMSE_median_dB": float(np.median(pair_values)), "pairwise_CNMSE_max_dB": float(np.max(pair_values)), "pairwise_CNMSE_min_finite_dB": float(np.min(pair_values)), "minimum_phase_separation_deg": _circular_min_sep(phases), "behavior_levels_covered": json.dumps(behavior_coverage, ensure_ascii=False), "stale_levels_covered": json.dumps(stale_coverage, ensure_ascii=False), "behavior_coverage_note": f"behavior levels={','.join(behavior_coverage)}; stale levels={','.join(stale_coverage)}"})
    table = pd.DataFrame(rows)
    feasible = table[table["all_design_margin_pass"] & table["all_own_DPD_target_reached"]].copy()
    if feasible.empty:
        raise RuntimeError("no feasible phase triplet")
    feasible["behavior_coverage_count"] = feasible["behavior_levels_covered"].map(lambda value: len(json.loads(value)))
    feasible["stale_coverage_count"] = feasible["stale_levels_covered"].map(lambda value: len(json.loads(value)))
    ranked = feasible.sort_values(["behavior_coverage_count", "stale_coverage_count", "minimum_phase_separation_deg", "pairwise_CNMSE_min_finite_dB", "pairwise_CNMSE_median_dB"], ascending=[False, False, False, False, False], kind="stable")
    recommendations = [{"triplet_id": int(row.triplet_id), "phases": [int(row.phase_1), int(row.phase_2), int(row.phase_3)]} for row in ranked.head(3).itertuples(index=False)]
    return table, recommendations


def run(*, allow_existing: bool = False) -> dict[str, object]:
    if RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir()) and not allow_existing:
        raise RuntimeError(f"result directory is not empty: {RESULT_ROOT}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    _append_log("START Scenario 1 load-drift state selection")
    raw_before = _raw_gate()
    specs = _candidate_specs()
    signals = {int(spec["State"]): _canonical_real_b(int(spec["State"])) for spec in specs}
    nominal_b = signals[0]
    frame = _build_candidate_frame(specs, nominal_b)
    pairwise = _pairwise(frame, signals)
    phase = _phase_comparison(frame)
    triplets, recommendations = _triplets(frame, pairwise)
    primary = recommendations[0]
    primary_row = triplets.loc[triplets["triplet_id"] == primary["triplet_id"]].iloc[0]
    primary_phases = [int(primary_row["phase_1"]), int(primary_row["phase_2"]), int(primary_row["phase_3"])]
    recommended_states = [0] + [int(frame.loc[(frame["funMng"] == 10) & (frame["funAng_deg"] == phase_value), "State"].iloc[0]) for phase_value in primary_phases] + [int(frame.loc[(frame["funMng"] == 20) & (frame["funAng_deg"] == phase_value), "State"].iloc[0]) for phase_value in primary_phases]
    role_map = {0: "nominal", 10: "light-drift", 20: "moderate-drift"}
    recommended_rows: list[dict[str, object]] = []
    for state_id in recommended_states:
        item = frame.loc[frame["State"] == state_id].iloc[0]
        if int(item["funMng"]) == 0:
            role = "nominal"
        else:
            phase_rank = primary_phases.index(int(item["funAng_deg"]))
            role = f"{role_map[int(item['funMng'])]} direction {chr(ord('A') + phase_rank)}"
        recommended_rows.append({"Scenario1_Load_ID": len(recommended_rows), "Scenario2_State": int(item["State"]), "funMng": int(item["funMng"]), "funAng": int(item["funAng_deg"]), "secMng": int(item["secMng"]), "secAng": int(item["secAng_deg"]), "load_label": item["load_label"], "NMSE_withoutDPD_dB": item["NMSE_withoutDPD_dB"], "ACPR_avg_withoutDPD_dBc": item["ACPR_avg_withoutDPD_dBc"], "NMSE_stale_dB": item["NMSE_stale_dB"], "ACPR_avg_stale_dBc": item["ACPR_avg_stale_dBc"], "Delta_NMSE_stale_vs_nominal_dB": item["Delta_NMSE_stale_vs_nominal_dB"], "Delta_ACPR_stale_vs_nominal_dB": item["Delta_ACPR_stale_vs_nominal_dB"], "CNMSE_B_to_nominal_dB": item["CNMSE_B_to_nominal_dB"], "NMSE_ownDPD_dB": item["NMSE_ownDPD_dB"], "ACPR_avg_ownDPD_dBc": item["ACPR_avg_ownDPD_dBc"], "own_DPD_target_reached": bool(item["own_DPD_target_reached"]), "selection_role": role, "selection_reason": item["selection_note"]})
    recommended_frame = pd.DataFrame(recommended_rows)
    frame.to_csv(RESULT_ROOT / "01_candidate_summary.csv", index=False)
    phase.to_csv(RESULT_ROOT / "02_phase_comparison.csv", index=False)
    triplets.to_csv(RESULT_ROOT / "03_phase_triplet_evaluation.csv", index=False)
    recommended_frame.to_csv(RESULT_ROOT / "04_recommended_7.csv", index=False)
    pd.DataFrame(pairwise, index=frame["load_label"], columns=frame["load_label"]).to_csv(RESULT_ROOT / "05_pairwise_B_CNMSE.csv")
    np.save(RESULT_ROOT / "05_pairwise_B_CNMSE.npy", pairwise)
    alt = recommendations[1:]
    nominal = frame.loc[frame["load_level"] == "nominal"].iloc[0]
    summary = {"status": "NUMERICAL_COMPLETE", "task_name": TASK_NAME, "module": "pa_performance_evaluation", "source_scenario": 2, "source_state_count": STATE_COUNT, "candidate_count": len(frame), "candidate_rule": {"funMng": [0, 10, 20], "funAng": list(PHASES), "secMng": 0, "secAng": 0}, "nominal_state": int(nominal["State"]), "nominal_load": nominal["load_label"], "nominal_NMSE_withoutDPD_dB": float(nominal["NMSE_withoutDPD_dB"]), "nominal_NMSE_stale_dB": float(nominal["NMSE_stale_dB"]), "nominal_ACPR_avg_stale_dBc": float(nominal["ACPR_avg_stale_dBc"]), "candidate_states": frame["State"].astype(int).tolist(), "valid_candidate_count": int(frame["design_margin_pass"].sum()), "rejected_candidate_list": frame.loc[~frame["design_margin_pass"], "State"].astype(int).tolist(), "rejection_reason": "outputPower protection threshold or own-DPD target not reached", "candidate_quantiles": {"behavior_shift_CNMMSE_non_nominal_q33_q66": np.quantile(frame.loc[frame["load_level"] != "nominal", "CNMSE_B_to_nominal_dB"].to_numpy(dtype=float), [1 / 3, 2 / 3]).tolist(), "stale_delta_NMSE_q33_q66": np.quantile(frame.loc[frame["load_level"] != "nominal", "Delta_NMSE_stale_vs_nominal_dB"].to_numpy(dtype=float), [1 / 3, 2 / 3]).tolist(), "stale_delta_ACPR_q33_q66": np.quantile(frame.loc[frame["load_level"] != "nominal", "Delta_ACPR_stale_vs_nominal_dB"].to_numpy(dtype=float), [1 / 3, 2 / 3]).tolist()}, "primary_recommended_phase_triplet": primary_phases, "primary_recommended_7_states": recommended_states, "alternative_triplets": alt, "initial_hypothesis_45_180_270_supported": primary_phases == [45, 180, 270], "protection_output_power_threshold_dBm": PROTECTION_OUTPUT_POWER_DBM, "own_DPD_target_ACPR_dBc": OWN_ACPR_TARGET_DB, "raw_manifest_before": raw_before, "raw_manifest_after": _raw_manifest(), "raw_unchanged": raw_before == _raw_manifest(), "pairwise_matrix_shape": list(pairwise.shape), "selection_rule": "feasible first; maximize behavior/stale level coverage; then circular phase separation; then closest-pair and median pairwise CNMSE diversity; no weighted black-box score", "outputs": {"xlsx": str(RESULT_ROOT / "scenario_1_load_drift_candidate_selection.xlsx"), "candidate_summary": str(RESULT_ROOT / "01_candidate_summary.csv"), "phase_comparison": str(RESULT_ROOT / "02_phase_comparison.csv"), "triplets": str(RESULT_ROOT / "03_phase_triplet_evaluation.csv"), "recommended_7": str(RESULT_ROOT / "04_recommended_7.csv"), "pairwise": str(RESULT_ROOT / "05_pairwise_B_CNMSE.csv")}}
    _write_json(RESULT_ROOT / "06_selection_summary.json", summary)
    xlsx_source = {"candidate_summary": {"headers": list(frame.columns), "rows": _safe_records(frame)}, "pairwise_B_CNMSE": {"headers": ["load_label"] + frame["load_label"].tolist(), "rows": [[label] + ["-Inf" if np.isneginf(value) else float(value) for value in row] for label, row in zip(frame["load_label"], pairwise, strict=True)]}, "phase_comparison": {"headers": phase.columns.tolist(), "rows": _safe_records(phase)}, "phase_triplet_evaluation": {"headers": triplets.columns.tolist(), "rows": _safe_records(triplets)}, "recommended_7": {"headers": recommended_frame.columns.tolist(), "rows": _safe_records(recommended_frame)}}
    _write_json(RESULT_ROOT / "00_xlsx_source.json", xlsx_source)
    figure_meta = write_plots()
    summary["figure_metadata"] = figure_meta
    _write_json(RESULT_ROOT / "06_selection_summary.json", summary)
    (RESULT_ROOT / "07_final_result_summary.txt").write_text("\n".join([f"Task: {TASK_NAME}", "Purpose: select 7 Scenario 1 load-drift states from Scenario 2 real PA measurements.", "Candidate set: matched + fundamental-only |Gamma|=0.1/0.2; secMng=secAng=0.", f"Primary phase triplet: {primary_phases}", f"Primary states: {recommended_states}", f"Initial hypothesis [45,180,270] supported: {summary['initial_hypothesis_45_180_270_supported']}", f"Valid candidate count: {summary['valid_candidate_count']}/17", f"Candidate quantiles: {json.dumps(summary['candidate_quantiles'], ensure_ascii=False)}", "Selection rule: feasibility, level coverage, phase separation, then pairwise CNMSE diversity; no weighted black-box score.", "See candidate_summary, phase_comparison, phase_triplet_evaluation and recommended_7 outputs for detailed statewise evidence."]) + "\n", encoding="utf-8")
    _write_checkpoint(phase="numerical_outputs_complete", raw_unchanged=summary["raw_unchanged"], primary_phases=primary_phases, recommended_states=recommended_states, validation_status="pending_artifact_tool_and_validator")
    _append_log(f"NUMERICAL COMPLETE candidates=17 triplets=56 primary_phases={primary_phases} recommended_states={recommended_states} raw_unchanged={summary['raw_unchanged']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(allow_existing=args.allow_existing), ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
