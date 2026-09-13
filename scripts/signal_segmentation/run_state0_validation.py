"""
功能说明：通过data_manager对state0 OFF、5列ILC和STALE执行真实canonical验收。
输入：state_id=0完整46变量字典，不使用MAT绝对路径。
输出：控制台摘要和results/signal_segmentation/state0_validation_summary.csv。
用途：只保存轻量验收元数据，不保存canonical波形、不生成科研图。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from data_manager import load_by_id  # noqa: E402

from signal_segmentation import (  # noqa: E402
    CanonicalSegments,
    build_ilc_segments,
    build_off_segments,
    build_partition_from_xin,
    build_stale_segments,
    validate_current_experiment_partition,
)

STATE_ID = 0
OUTPUT_PATH = PROJECT_ROOT / "results" / "signal_segmentation" / "state0_validation_summary.csv"


def _input_peak(canonical: CanonicalSegments) -> float:
    return max(float(np.max(np.abs(canonical[name].input))) for name in ("A", "B", "C"))


def _all_finite(canonical: CanonicalSegments) -> bool:
    return all(
        np.all(np.isfinite(canonical[name].input)) and np.all(np.isfinite(canonical[name].output))
        for name in ("A", "B", "C")
    )


def _row(canonical: CanonicalSegments) -> dict[str, str | int | float | bool]:
    return {
        "state_id": STATE_ID,
        "pair_type": canonical.pair_type,
        "iteration": ("" if canonical.iteration_index is None else canonical.iteration_index + 1),
        "full_length": canonical.partition.total_length,
        "a_start": canonical.partition.a_slice.start,
        "a_stop": canonical.partition.a_slice.stop,
        "n_a": canonical.partition.n_a,
        "b_start": canonical.partition.b_slice.start,
        "b_stop": canonical.partition.b_slice.stop,
        "n_b": canonical.partition.n_b,
        "c_start": canonical.partition.c_slice.start,
        "c_stop": canonical.partition.c_slice.stop,
        "n_c": canonical.partition.n_c,
        "raw_input_peak": (
            _input_peak(canonical)
            if canonical.input_peak_normalization_factor is None
            else canonical.input_peak_normalization_factor
        ),
        "canonical_input_peak": _input_peak(canonical),
        "rough_delay": canonical.rough_delay,
        "fraction_delay": canonical.fraction_delay,
        "all_finite": _all_finite(canonical),
    }


def main() -> None:
    state = load_by_id(STATE_ID)
    partition = build_partition_from_xin(np.asarray(state["xin"]))
    validate_current_experiment_partition(partition)
    off = build_off_segments(state, partition)
    ilc = build_ilc_segments(state, partition)
    stale = build_stale_segments(state, partition)
    canonical_sets = (off, *ilc, stale)
    rows = [_row(canonical) for canonical in canonical_sets]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        "Partition: "
        f"A=[0,{partition.a_slice.stop})/{partition.n_a}, "
        f"B=[{partition.b_slice.start},{partition.b_slice.stop})/{partition.n_b}, "
        f"C=[{partition.c_slice.start},{partition.c_slice.stop})/{partition.n_c}"
    )
    for row in rows:
        iteration = "" if row["iteration"] == "" else f" iter={row['iteration']}"
        print(
            f"{row['pair_type']}{iteration}: "
            f"raw_peak={row['raw_input_peak']:.15g}, "
            f"canonical_peak={row['canonical_input_peak']:.15g}, "
            f"rough={row['rough_delay']}, fine={row['fraction_delay']}, "
            f"finite={row['all_finite']}"
        )
    print(f"Validation summary: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
