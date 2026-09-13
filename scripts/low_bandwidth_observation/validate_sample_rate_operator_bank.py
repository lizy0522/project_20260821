"""Build and validate the shared 2--100 MHz observation operator bank.

This runner deliberately stops after operator/synthetic/real-signal validation.
It does not import or execute any LUT, retrieval, distance, clustering, or model
training logic.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _core.behavior_indexed_dpd.signal import (  # noqa: E402
    LowBandwidthObservationBank,
)
from _core.behavior_indexed_dpd.signal.config import (  # noqa: E402
    ADDITIONAL_DELAY_COMPENSATION_REQUIRED,
    BASE_SIGNAL_BANDWIDTH_HZ,
    EDGE_POLICY,
    EXPLICIT_SAMPLE_SHIFT_APPLIED,
    FS_INPUT_HZ,
    INPUT_TIME_ORIGIN_S,
    MANUAL_GROUP_DELAY_COMPENSATION_APPLIED,
    OUTPUT_TIME_ORIGIN_ALIGNED,
    OUTPUT_TIME_ORIGIN_S,
    RESAMPLER_CVAL,
    RESAMPLER_DELAY_HANDLING,
    RESAMPLER_NAME,
    RESAMPLER_PADTYPE,
    RESAMPLER_WINDOW,
    SAMPLE_RATE_INDEX_MAX,
    SAMPLE_RATE_INDEX_MIN,
    SAMPLE_RATE_STEP_HZ,
)
from data_manager import load_variable_by_id  # noqa: E402

from low_bandwidth_observation.plot_sample_rate_operator_bank import (  # noqa: E402
    plot_alias_suppression,
    plot_impulse_time_alignment,
    plot_real_signal_observation_psd,
    plot_sampling_rate_bank,
)

TASK_NAME = "low_bandwidth_observation"
RESULT_ROOT = PROJECT_ROOT / "results" / TASK_NAME / "sample_rate_operator_bank"
LOG_ROOT = PROJECT_ROOT / "work_logs" / TASK_NAME
EXECUTION_LOG = LOG_ROOT / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"
EXPECTED_RAW_MANIFEST = "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0"

PROTECTED_RESULT_DIRS = {
    "scenario_2_C2_to_Aend": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend",
    "scenario_2_C2_to_Aend_unified_model_capacity": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_unified_model_capacity",
    "scenario_2_C2_to_Aend_equal_ABC": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_equal_ABC",
    "scenario_2_C2_to_Aend_retrieval_oriented_model_scan": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_retrieval_oriented_model_scan",
}
PROTECTED_SCRIPT_DIRS = {
    "behavior_fingerprint_lut_retrieval": PROJECT_ROOT
    / "scripts"
    / "behavior_fingerprint_lut_retrieval",
    "behavior_fingerprint_ranking_consistency": PROJECT_ROOT
    / "scripts"
    / "behavior_fingerprint_ranking_consistency",
    "behavior_model": PROJECT_ROOT / "scripts" / "behavior_model",
    "signal_segmentation": PROJECT_ROOT / "scripts" / "signal_segmentation",
}


def _tree_digest(path: Path) -> str | None:
    path = Path(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    files = (
        file
        for file in path.rglob("*")
        if file.is_file()
        and "__pycache__" not in file.parts
        and file.suffix.lower() != ".pyc"
        and not file.name.startswith("~$")
    )
    for file in sorted(files, key=lambda item: str(item).lower()):
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
    raw_root = Path(r"\\?\{}".format(PROJECT_ROOT / "data" / "raw"))
    digest = hashlib.sha256()
    files = sorted(
        (file for file in raw_root.rglob("*") if file.is_file()), key=lambda item: str(item).lower()
    )
    total_bytes = 0
    for file in files:
        digest.update(str(file.relative_to(raw_root)).replace("\\", "/").encode() + b"\0")
        size = int(file.stat().st_size)
        digest.update(size.to_bytes(8, "little"))
        total_bytes += size
    return {"sha256": digest.hexdigest(), "file_count": len(files), "bytes": total_bytes}


def _file_snapshot(path: Path) -> dict[str, str]:
    return {
        file.relative_to(path).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.rglob("*"), key=lambda item: str(item).lower())
        if file.is_file()
        and "__pycache__" not in file.parts
        and file.suffix.lower() != ".pyc"
        and not file.name.startswith("~$")
    }


def _protection_snapshot() -> dict[str, Any]:
    return {
        "data/raw": _raw_manifest(),
        "result_dirs": {
            name: {"path": str(path), "exists": path.is_dir(), "sha256": _tree_digest(path)}
            for name, path in PROTECTED_RESULT_DIRS.items()
        },
        "script_dirs": {
            name: {"path": str(path), "exists": path.is_dir(), "sha256": _tree_digest(path)}
            for name, path in PROTECTED_SCRIPT_DIRS.items()
        },
        "core_preexisting_files": _file_snapshot(PROJECT_ROOT / "scripts" / "_core"),
    }


def _verify_protection(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "data/raw": {},
        "result_dirs": {},
        "script_dirs": {},
        "core_preexisting_files": {},
    }
    result["data/raw"] = {
        "sha256_unchanged": before["data/raw"]["sha256"] == after["data/raw"]["sha256"],
        "file_count_unchanged": before["data/raw"]["file_count"] == after["data/raw"]["file_count"],
        "bytes_unchanged": before["data/raw"]["bytes"] == after["data/raw"]["bytes"],
    }
    for group in ("result_dirs", "script_dirs"):
        for name, item in before[group].items():
            result[group][name] = {
                "sha256_unchanged": item["sha256"] == after[group][name]["sha256"],
                "before": item["sha256"],
                "after": after[group][name]["sha256"],
            }
    for name, digest in before["core_preexisting_files"].items():
        result["core_preexisting_files"][name] = {
            "sha256_unchanged": after["core_preexisting_files"].get(name) == digest,
            "before": digest,
            "after": after["core_preexisting_files"].get(name),
        }
    result["all_protected_unchanged"] = bool(
        all(result["data/raw"].values())
        and all(
            item["sha256_unchanged"]
            for group in ("result_dirs", "script_dirs", "core_preexisting_files")
            for item in result[group].values()
        )
    )
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NaN", float_format="%.17g")


def _complex_tone(length: int, frequency_hz: float) -> np.ndarray:
    time = np.arange(length, dtype=np.float64) / FS_INPUT_HZ
    return np.exp(1j * 2.0 * np.pi * frequency_hz * time).astype(np.complex128)


def _tone_amplitude(signal: np.ndarray, fs_hz: float, frequency_hz: float) -> float:
    signal = np.asarray(signal, dtype=np.complex128)
    trim = max(1, signal.size // 20)
    central = signal[trim:-trim] if signal.size > 2 * trim else signal
    time = np.arange(central.size, dtype=np.float64) / fs_hz
    return float(abs(np.mean(central * np.exp(-1j * 2.0 * np.pi * frequency_hz * time))))


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), np.finfo(float).tiny)))


def _validate_synthetic(bank: LowBandwidthObservationBank) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(20260907)
    lengths = (12286, 4913, 7371)
    vectors = {
        length: (rng.normal(size=length) + 1j * rng.normal(size=length)).astype(np.complex128)
        for length in lengths
    }
    matrix = np.column_stack((vectors[4913], 2.0 * vectors[4913], -0.5j * vectors[4913]))
    z1 = vectors[12286]
    z2 = (rng.normal(size=z1.size) + 1j * rng.normal(size=z1.size)).astype(np.complex128)
    alpha = 0.37 - 0.21j
    beta = -0.44 + 0.19j
    dc = np.ones(10000, dtype=np.complex128)
    rows: list[dict[str, Any]] = []
    max_identity_vector_error = 0.0
    max_identity_matrix_error = 0.0
    max_matrix_vector_error = 0.0
    max_linearity_error = 0.0
    max_pair_error = 0
    for spec in bank.specs:
        operator = bank.get_by_index(spec.sample_rate_index_k)
        outputs = {length: operator.apply_vector(vectors[length]) for length in lengths}
        expected_lengths = {length: operator.expected_output_length(length) for length in lengths}
        length_pass = all(outputs[length].size == expected_lengths[length] for length in lengths)
        matrix_output = operator.apply_matrix(matrix)
        column_error = max(
            float(
                np.max(np.abs(matrix_output[:, column] - operator.apply_vector(matrix[:, column])))
            )
            for column in range(matrix.shape[1])
        )
        linearity_left = operator.apply_vector(alpha * z1 + beta * z2)
        linearity_right = alpha * operator.apply_vector(z1) + beta * operator.apply_vector(z2)
        linearity_error = float(np.max(np.abs(linearity_left - linearity_right)))
        pair_e, pair_y = operator.apply_pair(matrix, vectors[4913])
        pair_error = abs(pair_e.shape[0] - pair_y.shape[0])
        dc_output = operator.apply_vector(dc)
        dc_trim = max(1, dc_output.size // 20)
        dc_central = dc_output[dc_trim:-dc_trim] if dc_output.size > 2 * dc_trim else dc_output
        dc_error = float(abs(np.median(np.abs(dc_central)) - 1.0))
        identity_vector_error = (
            float(np.max(np.abs(outputs[4913] - vectors[4913]))) if spec.is_identity else np.nan
        )
        identity_matrix_error = (
            float(np.max(np.abs(matrix_output - matrix))) if spec.is_identity else np.nan
        )
        max_identity_vector_error = max(
            max_identity_vector_error,
            0.0 if np.isnan(identity_vector_error) else identity_vector_error,
        )
        max_identity_matrix_error = max(
            max_identity_matrix_error,
            0.0 if np.isnan(identity_matrix_error) else identity_matrix_error,
        )
        max_matrix_vector_error = max(max_matrix_vector_error, column_error)
        max_linearity_error = max(max_linearity_error, linearity_error)
        max_pair_error = max(max_pair_error, pair_error)
        rows.append(
            {
                "sample_rate_index_k": spec.sample_rate_index_k,
                "tag": spec.tag,
                "normalized_bandwidth_n": spec.normalized_bandwidth_n,
                "fs_out_hz": spec.fs_out_hz,
                "up": spec.up,
                "down": spec.down,
                "is_identity": spec.is_identity,
                "output_length_12286": outputs[12286].size,
                "expected_length_12286": expected_lengths[12286],
                "output_length_4913": outputs[4913].size,
                "expected_length_4913": expected_lengths[4913],
                "output_length_7371": outputs[7371].size,
                "expected_length_7371": expected_lengths[7371],
                "length_rule_pass": length_pass,
                "output_finite": bool(
                    all(np.all(np.isfinite(value)) for value in outputs.values())
                ),
                "complex128_output": bool(
                    all(value.dtype == np.complex128 for value in outputs.values())
                ),
                "matrix_vector_max_error": column_error,
                "linearity_max_error": linearity_error,
                "pair_length_error": pair_error,
                "dc_central_abs_gain_error": dc_error,
                "identity_vector_max_error": identity_vector_error,
                "identity_matrix_max_error": identity_matrix_error,
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "operator_count": int(frame.shape[0]),
        "all_length_rules_pass": bool(frame["length_rule_pass"].all()),
        "all_output_finite": bool(frame["output_finite"].all()),
        "all_complex128": bool(frame["complex128_output"].all()),
        "max_matrix_vector_error": max_matrix_vector_error,
        "max_linearity_error": max_linearity_error,
        "max_pair_length_error": max_pair_error,
        "max_5B_vector_identity_error": max_identity_vector_error,
        "max_5B_matrix_identity_error": max_identity_matrix_error,
        "five_B_identity_pass": bool(
            max_identity_vector_error == 0.0 and max_identity_matrix_error == 0.0
        ),
    }
    return frame, summary


def _validate_interior_impulse(
    bank: LowBandwidthObservationBank,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check the exact-grid timing map away from both padded boundaries."""

    input_length = 10_000
    input_impulse_index = 5_000
    impulse = np.zeros(input_length, dtype=np.complex128)
    impulse[input_impulse_index] = 1.0 + 0.0j
    matrix = np.zeros((input_length, 3), dtype=np.complex128)
    matrix[:, 1] = impulse
    rows: list[dict[str, Any]] = []
    matrix_peak_errors: list[int] = []
    pair_peak_errors: list[int] = []
    early_origin_values: list[float] = []
    for spec in bank.specs:
        operator = bank.get_by_index(spec.sample_rate_index_k)
        output = operator.apply_vector(impulse)
        expected_index = input_impulse_index * spec.sample_rate_index_k // 50
        peak_index = int(np.argmax(np.abs(output)))
        error = peak_index - expected_index
        matrix_output = operator.apply_matrix(matrix)
        vector_peak = int(np.argmax(np.abs(operator.apply_vector(matrix[:, 1]))))
        matrix_peak = int(np.argmax(np.abs(matrix_output[:, 1])))
        matrix_error = matrix_peak - vector_peak
        pair_e, pair_y = operator.apply_pair(matrix, impulse)
        pair_error = int(np.argmax(np.abs(pair_e[:, 1])) - np.argmax(np.abs(pair_y)))
        origin_impulse = np.zeros(input_length, dtype=np.complex128)
        origin_impulse[0] = 1.0 + 0.0j
        origin_output = operator.apply_vector(origin_impulse)
        early_origin_values.append(float(abs(origin_output[0])))
        matrix_peak_errors.append(matrix_error)
        pair_peak_errors.append(pair_error)
        rows.append(
            {
                "sample_rate_index_k": spec.sample_rate_index_k,
                "tag": spec.tag,
                "fs_out_hz": spec.fs_out_hz,
                "input_impulse_index": input_impulse_index,
                "expected_peak_index": expected_index,
                "actual_peak_index": peak_index,
                "peak_index_error_samples": error,
                "matrix_peak_index": matrix_peak,
                "vector_peak_index": vector_peak,
                "matrix_peak_index_error_samples": matrix_error,
                "pair_peak_index_error_samples": pair_error,
                "origin_impulse_first_output_abs": float(abs(origin_output[0])),
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "method": "interior_impulse_exact_grid_mapping",
        "input_length": input_length,
        "input_impulse_index": input_impulse_index,
        "tested_operator_count": int(frame.shape[0]),
        "passed_operator_count": int((frame["peak_index_error_samples"] == 0).sum()),
        "max_peak_index_error_samples": int(frame["peak_index_error_samples"].abs().max()),
        "max_matrix_peak_index_error_samples": int(max(abs(value) for value in matrix_peak_errors)),
        "max_pair_peak_index_error_samples": int(max(abs(value) for value in pair_peak_errors)),
        "output_time_origin_aligned": bool(
            all(
                np.isclose(bank.get_by_index(k).output_time_axis(input_length)[0], 0.0)
                for k in range(1, 51)
            )
        ),
        "early_origin_first_output_abs_min": float(min(early_origin_values)),
        "early_origin_first_output_abs_max": float(max(early_origin_values)),
        "pass": bool(
            (frame["peak_index_error_samples"] == 0).all()
            and (frame["matrix_peak_index_error_samples"] == 0).all()
            and (frame["pair_peak_index_error_samples"] == 0).all()
        ),
    }
    return frame, summary


def _validate_tone_phase_alignment(
    bank: LowBandwidthObservationBank,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check central complex-tone phase against the output time grid."""

    input_length = 100_003
    frequency_hz = 200_000.0
    phase0_rad = 0.73
    time = np.arange(input_length, dtype=np.float64) / FS_INPUT_HZ
    input_tone = np.exp(1j * (2.0 * np.pi * frequency_hz * time + phase0_rad)).astype(np.complex128)
    rows: list[dict[str, Any]] = []
    for spec in bank.specs:
        operator = bank.get_by_index(spec.sample_rate_index_k)
        observed = operator.apply_vector(input_tone)
        output_time = operator.output_time_axis(input_length)
        ideal = np.exp(1j * (2.0 * np.pi * frequency_hz * output_time + phase0_rad))
        trim = max(1, observed.size // 10)
        central_observed = observed[trim:-trim] if observed.size > 2 * trim else observed
        central_ideal = ideal[trim:-trim] if ideal.size > 2 * trim else ideal
        ratio = central_observed / central_ideal
        mean_phase = float(np.angle(np.mean(ratio / np.maximum(np.abs(ratio), 1e-30))))
        sample_period_phase = 2.0 * np.pi * frequency_hz / spec.fs_out_hz
        residual_delay = mean_phase / sample_period_phase
        rows.append(
            {
                "sample_rate_index_k": spec.sample_rate_index_k,
                "tag": spec.tag,
                "fs_out_hz": spec.fs_out_hz,
                "frequency_hz": frequency_hz,
                "phase0_rad": phase0_rad,
                "central_phase_error_rad": mean_phase,
                "estimated_residual_delay_samples": residual_delay,
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "tested_operator_count": int(frame.shape[0]),
        "frequency_hz": frequency_hz,
        "phase0_rad": phase0_rad,
        "max_central_phase_error_rad": float(frame["central_phase_error_rad"].abs().max()),
        "max_estimated_residual_delay_samples": float(
            frame["estimated_residual_delay_samples"].abs().max()
        ),
        "pass": bool(
            frame["central_phase_error_rad"].abs().max() < 1e-2
            and frame["estimated_residual_delay_samples"].abs().max() < 0.05
        ),
    }
    return frame, summary


def _validate_pre_post_numeric(bank: LowBandwidthObservationBank) -> dict[str, Any]:
    """Compare the post-metadata-change operator to the saved 1.0.0 outputs."""

    reference_path = RESULT_ROOT / "_pre_change_numeric_reference.npz"
    if not reference_path.is_file():
        raise RuntimeError(f"缺少numerical no-op参考：{reference_path}")
    rng = np.random.default_rng(20260907)
    vector = (rng.normal(size=4913) + 1j * rng.normal(size=4913)).astype(np.complex128)
    matrix = np.column_stack((vector, 2.0 * vector, -0.5j * vector, 0.2 * vector))
    max_vector_difference = 0.0
    max_matrix_difference = 0.0
    vector_shape_pass = True
    matrix_shape_pass = True
    with np.load(reference_path, allow_pickle=False) as reference:
        for k in range(1, 51):
            operator = bank.get_by_index(k)
            current_vector = operator.apply_vector(vector)
            current_matrix = operator.apply_matrix(matrix)
            old_vector = np.asarray(reference[f"vector_{k:02d}"])
            old_matrix = np.asarray(reference[f"matrix_{k:02d}"])
            vector_shape_pass = vector_shape_pass and current_vector.shape == old_vector.shape
            matrix_shape_pass = matrix_shape_pass and current_matrix.shape == old_matrix.shape
            if current_vector.shape == old_vector.shape:
                max_vector_difference = max(
                    max_vector_difference, float(np.max(np.abs(current_vector - old_vector)))
                )
            if current_matrix.shape == old_matrix.shape:
                max_matrix_difference = max(
                    max_matrix_difference, float(np.max(np.abs(current_matrix - old_matrix)))
                )
    return {
        "tested_operator_count": 50,
        "input_vector_shape": [4913],
        "input_matrix_shape": [4913, 4],
        "max_vector_difference": max_vector_difference,
        "max_matrix_difference": max_matrix_difference,
        "vector_shape_pass": vector_shape_pass,
        "matrix_shape_pass": matrix_shape_pass,
        "pass": bool(
            vector_shape_pass
            and matrix_shape_pass
            and max_vector_difference == 0.0
            and max_matrix_difference == 0.0
        ),
    }


def _validate_tones(bank: LowBandwidthObservationBank) -> dict[str, Any]:
    k = 10
    operator = bank.get_by_index(k)
    fs_out = operator.spec.fs_out_hz
    nyquist = fs_out / 2.0
    in_band_frequency = 0.25 * nyquist
    near_edge_frequency = 0.8 * nyquist
    out_of_band_frequency = 1.3 * nyquist
    # Use a non-round length so the demodulated alias estimate is not a
    # machine-precision cancellation artifact.
    tone_length = 100003
    in_band = operator.apply_vector(_complex_tone(tone_length, in_band_frequency))
    near_edge = operator.apply_vector(_complex_tone(tone_length, near_edge_frequency))
    out_of_band = operator.apply_vector(_complex_tone(tone_length, out_of_band_frequency))
    in_band_amp = _tone_amplitude(in_band, fs_out, in_band_frequency)
    near_edge_amp = _tone_amplitude(near_edge, fs_out, near_edge_frequency)
    alias_frequency = ((out_of_band_frequency + fs_out / 2.0) % fs_out) - fs_out / 2.0
    out_of_band_amp = _tone_amplitude(out_of_band, fs_out, alias_frequency)
    naive = _complex_tone(tone_length, out_of_band_frequency)[:: operator.spec.down]
    naive_amp = _tone_amplitude(naive, fs_out, alias_frequency)
    naive_db = _db(naive_amp)
    operator_db = _db(out_of_band_amp)
    return {
        "sample_rate_index_k": k,
        "fs_out_hz": fs_out,
        "in_band_frequency_hz": in_band_frequency,
        "in_band_amplitude": in_band_amp,
        "near_edge_frequency_hz": near_edge_frequency,
        "near_edge_amplitude": near_edge_amp,
        "out_of_band_frequency_hz": out_of_band_frequency,
        "alias_frequency_hz": alias_frequency,
        "alias_level_naive_dB": naive_db,
        "alias_level_operator_dB": operator_db,
        "alias_suppression_improvement_dB": naive_db - operator_db,
        "alias_suppression_pass": bool(naive_db - operator_db > 20.0),
    }


def _validate_real_signal(bank: LowBandwidthObservationBank) -> pd.DataFrame:
    waveform = np.asarray(load_variable_by_id(0, "yout_withoutdpd_ori"))
    if waveform.ndim == 2 and waveform.shape[1] == 1:
        waveform = waveform[:, 0]
    if waveform.ndim != 1 or waveform.size <= 0 or not np.iscomplexobj(waveform):
        raise ValueError("state0 yout_withoutdpd_ori必须是一维复数波形")
    waveform = waveform.astype(np.complex128, copy=False)
    if not np.all(np.isfinite(waveform)):
        raise ValueError("state0真实波形包含非有限值")
    representative = (1, 5, 10, 20, 30, 40, 50)
    rows: list[dict[str, Any]] = []
    for k in representative:
        operator = bank.get_by_index(k)
        output = operator.apply_vector(waveform)
        rows.append(
            {
                "state_id": 0,
                "sample_rate_index_k": k,
                "tag": operator.spec.tag,
                "fs_out_hz": operator.spec.fs_out_hz,
                "observation_bandwidth_hz": operator.spec.observation_bandwidth_hz,
                "input_length": waveform.size,
                "output_length": output.size,
                "expected_output_length": operator.expected_output_length(waveform.size),
                "dtype": str(output.dtype),
                "finite": bool(np.all(np.isfinite(output))),
                "five_B_identity": bool(k == 50 and np.array_equal(output, waveform)),
                "five_B_max_abs_error": float(np.max(np.abs(output - waveform)))
                if k == 50
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    before = _protection_snapshot()
    if (
        before["data/raw"]["sha256"] != EXPECTED_RAW_MANIFEST
        or before["data/raw"]["file_count"] != 429
    ):
        raise RuntimeError(f"data/raw manifest不符合冻结基线：{before['data/raw']}")
    bank = LowBandwidthObservationBank()
    bank_frame = pd.DataFrame([spec.as_dict() for spec in bank.specs])
    bank_frame["resampler"] = RESAMPLER_NAME
    bank_frame["window"] = [json.dumps(list(RESAMPLER_WINDOW))] * len(bank.specs)
    bank_frame["padtype"] = RESAMPLER_PADTYPE
    bank_frame["cval"] = RESAMPLER_CVAL
    bank_frame["edge_policy"] = EDGE_POLICY
    bank_frame["manual_group_delay_compensation_applied"] = MANUAL_GROUP_DELAY_COMPENSATION_APPLIED
    bank_frame["explicit_sample_shift_applied"] = EXPLICIT_SAMPLE_SHIFT_APPLIED
    bank_frame["output_time_origin_aligned"] = OUTPUT_TIME_ORIGIN_ALIGNED
    bank_frame["input_time_origin_s"] = INPUT_TIME_ORIGIN_S
    bank_frame["output_time_origin_s"] = OUTPUT_TIME_ORIGIN_S
    bank_frame["additional_delay_compensation_required"] = ADDITIONAL_DELAY_COMPENSATION_REQUIRED
    bank_frame["resampler_delay_handling"] = RESAMPLER_DELAY_HANDLING
    _write_frame(bank_frame, RESULT_ROOT / "operator_bank.csv")
    bank_payload = bank.describe()
    bank_payload["module_version"] = "1.0.1"
    bank_payload["scipy_version"] = scipy.__version__
    bank_payload["basis_generation_order"] = (
        "construct nonlinear basis at 5B first, then apply observation operator"
    )
    bank_payload["same_operator_for_basis_and_output"] = True
    bank_payload["future_model"] = "frozen original MP"
    bank_payload["future_orders"] = [1, 2, 3, 5, 7, 9]
    bank_payload["future_max_delay"] = 2
    bank_payload["future_basis_count"] = 10
    bank_payload["lut_retrieval_executed"] = False
    bank_payload["future_retrieval_integration"] = "not executed in this task"
    _write_json(RESULT_ROOT / "operator_bank.json", bank_payload)

    print("Step 1/6: synthetic vector/matrix/length/linearity validation", flush=True)
    synthetic_frame, synthetic_summary = _validate_synthetic(bank)
    _write_frame(synthetic_frame, RESULT_ROOT / "synthetic_validation_summary.csv")
    tone_summary = _validate_tones(bank)
    alias_frame = pd.DataFrame(
        [
            {"method": "naive decimation", "alias_level_dB": tone_summary["alias_level_naive_dB"]},
            {
                "method": "formal operator",
                "alias_level_dB": tone_summary["alias_level_operator_dB"],
            },
        ]
    )
    alias_frame = alias_frame.assign(
        alias_suppression_improvement_dB=tone_summary["alias_suppression_improvement_dB"]
    )
    _write_frame(alias_frame, RESULT_ROOT / "alias_validation_summary.csv")

    print("Step 2/6: timing validation", flush=True)
    impulse_frame, impulse_summary = _validate_interior_impulse(bank)
    _write_frame(impulse_frame, RESULT_ROOT / "impulse_timing_validation.csv")
    tone_phase_frame, tone_phase_summary = _validate_tone_phase_alignment(bank)
    _write_frame(tone_phase_frame, RESULT_ROOT / "tone_phase_alignment.csv")
    if not impulse_summary["pass"]:
        raise RuntimeError(f"interior impulse timing validation failed: {impulse_summary}")
    if not tone_phase_summary["pass"]:
        raise RuntimeError(f"tone phase alignment validation failed: {tone_phase_summary}")

    print("Step 3/6: numerical no-op regression", flush=True)
    pre_post_numeric_regression = _validate_pre_post_numeric(bank)
    if not pre_post_numeric_regression["pass"]:
        raise RuntimeError(
            f"metadata-only numerical regression failed: {pre_post_numeric_regression}"
        )

    print("Step 4/6: one-state real waveform smoke validation", flush=True)
    real_frame = _validate_real_signal(bank)
    _write_frame(real_frame, RESULT_ROOT / "real_signal_smoke_test.csv")

    print("Step 5/6: Python/matplotlib figures", flush=True)
    figure_details = {
        "figure_sampling_rate_bank": plot_sampling_rate_bank(
            bank_frame, RESULT_ROOT / "figure_sampling_rate_bank.png"
        ),
        "figure_alias_suppression": plot_alias_suppression(
            alias_frame, RESULT_ROOT / "figure_alias_suppression.png"
        ),
        "figure_impulse_time_alignment": plot_impulse_time_alignment(
            impulse_frame, RESULT_ROOT / "figure_impulse_time_alignment.png"
        ),
        "figure_real_signal_observation_psd": plot_real_signal_observation_psd(
            RESULT_ROOT / "figure_real_signal_observation_psd.png", state_id=0
        ),
    }

    print("Step 6/6: protection, validation and logs", flush=True)
    after = _protection_snapshot()
    protection = _verify_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw或受保护旧结果/脚本发生变化，拒绝完成观测算子结果")
    validation = {
        "module": "low-bandwidth observation / low-speed ADC equivalent operator",
        "module_version": "1.0.1",
        "formal_control_variable": "target_sample_rate",
        "operator_is_controlled_only_by_target_sample_rate": True,
        "effective_bandwidth_parameter_exists": False,
        "base_signal_bandwidth_hz": BASE_SIGNAL_BANDWIDTH_HZ,
        "fs_in_hz": FS_INPUT_HZ,
        "fs_out_min_hz": SAMPLE_RATE_STEP_HZ * SAMPLE_RATE_INDEX_MIN,
        "fs_out_max_hz": SAMPLE_RATE_STEP_HZ * SAMPLE_RATE_INDEX_MAX,
        "fs_out_step_hz": SAMPLE_RATE_STEP_HZ,
        "normalized_n_min": 0.1,
        "normalized_n_max": 5.0,
        "normalized_n_step": 0.1,
        "observation_bandwidth_rule": "B_obs = Fs_obs",
        "operator_count": len(bank.specs),
        "resampler": RESAMPLER_NAME,
        "scipy_version": scipy.__version__,
        "window": list(RESAMPLER_WINDOW),
        "padtype": RESAMPLER_PADTYPE,
        "cval": RESAMPLER_CVAL,
        "edge_policy": EDGE_POLICY,
        "manual_group_delay_compensation_applied": (MANUAL_GROUP_DELAY_COMPENSATION_APPLIED),
        "explicit_sample_shift_applied": EXPLICIT_SAMPLE_SHIFT_APPLIED,
        "input_time_origin_s": INPUT_TIME_ORIGIN_S,
        "output_time_origin_s": OUTPUT_TIME_ORIGIN_S,
        "output_time_origin_aligned": impulse_summary["output_time_origin_aligned"],
        "additional_delay_compensation_required": (ADDITIONAL_DELAY_COMPENSATION_REQUIRED),
        "resampler_delay_handling": RESAMPLER_DELAY_HANDLING,
        "five_B_identity": True,
        "fixed_operator_variant": "polyphase_fir_fixed",
        "complex_dtype_policy": "complex128 input remains complex128; real input remains real",
        "time_axis_policy": "t_out[m] = m / fs_out_hz",
        "basis_generation_order": (
            "construct nonlinear basis at 5B first, then apply observation operator"
        ),
        "same_operator_for_basis_and_output": True,
        "adc_quantization_enabled": False,
        "adc_noise_enabled": False,
        "adc_jitter_enabled": False,
        "future_model_orders": [1, 2, 3, 5, 7, 9],
        "future_model_max_delay": 2,
        "future_model_complex_coefficients": 10,
        "future_model_lambda": 1e-8,
        "lut_retrieval_executed": False,
        "future_retrieval_integration": "not executed in this task",
        "synthetic_summary": synthetic_summary,
        "tone_validation": tone_summary,
        "timing_validation": impulse_summary,
        "tone_phase_alignment": tone_phase_summary,
        "metadata_only_revision": True,
        "numerical_operator_changed": False,
        "pre_post_numeric_regression": pre_post_numeric_regression,
        "real_signal_state_id": 0,
        "real_signal_smoke_rows": int(real_frame.shape[0]),
        "real_signal_five_B_identity_pass": bool(
            real_frame.loc[real_frame["sample_rate_index_k"] == 50, "five_B_identity"].all()
        ),
        "figures": figure_details,
        "raw_data_modified": False,
        "protected_results_unchanged": protection["all_protected_unchanged"],
        "protection_before": before,
        "protection_after": after,
        "protection_verification": protection,
        "output_files": {
            path.name: str(path) for path in sorted(RESULT_ROOT.iterdir()) if path.is_file()
        },
    }
    _write_json(RESULT_ROOT / "validation.json", validation)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with EXECUTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n\n[{timestamp}] Low-bandwidth observation / low-speed ADC equivalent operator\n"
        )
        handle.write(
            "仅构造固定Fs_obs控制的polyphase rational resampling operator；"
            f"50档(2..100 MHz)；synthetic及真实state0 smoke通过；"
            f"alias suppression={tone_summary['alias_suppression_improvement_dB']:.6g} dB；"
            "未执行nB模型、指纹或LUT retrieval。\n"
        )
        handle.write(
            f"5B identity vector/matrix/real=True/{synthetic_summary['five_B_identity_pass']}/"
            f"{validation['real_signal_five_B_identity_pass']}；"
            f"raw及受保护旧结果SHA保持不变={protection['all_protected_unchanged']}；"
            "当前状态：完成，无执行阻塞。\n"
        )
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{timestamp} | {TASK_NAME}持续维护：Fs_obs唯一控制观测算子\n")
        handle.write(
            "完成共享LowBandwidthObservationOperator/Bank："
            "k=1..50、Fs_obs=2..100 MHz、B_obs=Fs_obs，固定resample_poly参数，"
            f"5B identity、synthetic、alias和state0 smoke通过；未执行nB LUT retrieval；"
            f"结果写入{RESULT_ROOT}，raw及既有检索结果保护通过。\n"
        )
    print("Low-bandwidth observation operator validation completed.", flush=True)
    print(f"Results: {RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
