"""
Scenario 2 统一 Ridge 开发/验证流程的契约测试。

这些测试覆盖确定性 split、Validation 泄漏保护、gain 符号、统一 lambda、ILC1
边界、候选选择和冻结前 payload；真实全425状态 lambda=0 门槛由正式 runner 执行。
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_model.scenario2_ridge_analysis import (  # noqa: E402
    COARSE_LAMBDAS,
    MODEL_IDS,
    add_paired_gain_columns,
    build_scan_summary,
    build_selection_payload,
    build_validation_result,
    create_state_split,
    evaluate_prepared_state,
    load_or_create_state_split,
    make_fine_lambda_grid,
    prepare_state_ridge_data,
    run_ols_vs_ridge_comparison,
    select_lambda,
    validate_development_state_ids,
)


def _assert_raises(exception_type: type[BaseException], message: str, function) -> None:
    try:
        function()
    except exception_type as error:
        assert message in str(error)
    else:
        raise AssertionError(f"应抛出{exception_type.__name__}")


def test_split_is_deterministic_and_has_fixed_counts() -> None:
    first = create_state_split()
    second = create_state_split()
    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (425, 10)
    assert first["split"].value_counts().to_dict() == {
        "development": 340,
        "validation": 85,
    }
    assert first.loc[first["state_id"] == 0, "split"].item() == "development"
    assert first.loc[first["state_id"] != 0, "stratum_key"].nunique() == 11
    development = set(first.loc[first["split"] == "development", "state_id"])
    validation = set(first.loc[first["split"] == "validation", "state_id"])
    assert development.isdisjoint(validation)
    assert development | validation == set(range(425))


def test_manifest_is_read_back_without_repartitioning() -> None:
    with TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "state_split.csv"
        created = load_or_create_state_split(manifest_path)
        loaded = load_or_create_state_split(manifest_path)
        pd.testing.assert_frame_equal(created, loaded)
        assert manifest_path.is_file()


def test_validation_leakage_guard() -> None:
    development = (0, 2, 4)
    validation = (1, 3)
    assert validate_development_state_ids(
        development,
        development,
        validation,
    ) == development
    _assert_raises(
        ValueError,
        "只能接收development_state_ids",
        lambda: validate_development_state_ids((0, 1), development, validation),
    )
    _assert_raises(
        ValueError,
        "不能相交",
        lambda: validate_development_state_ids(development, (0, 1), (1, 3)),
    )


def test_fine_grid_is_centered_and_contains_zero() -> None:
    grid = make_fine_lambda_grid(1e-8)
    assert grid.shape == (18,)
    assert grid[0] == 0.0
    np.testing.assert_allclose(grid[1], 1e-9)
    np.testing.assert_allclose(grid[-1], 1e-7)
    assert np.all(np.diff(grid) > 0)


def _synthetic_scan_rows(*, degraded_x: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ridge_lambda in (0.0, 1e-8):
        for model_id in MODEL_IDS:
            if model_id == "X-A":
                train = -40.0
                b_gen = -39.0 if (ridge_lambda and degraded_x) else -40.0
                vs_x = np.nan
                vs_real = b_gen
                theta_norm = 10.0 if ridge_lambda == 0 else 9.0
            elif model_id == "Y-A-1":
                train = -40.0
                b_gen = -40.0
                vs_x = -35.0
                vs_real = -34.0
                theta_norm = 12.0 if ridge_lambda == 0 else 11.0
            else:
                train = -40.0 if ridge_lambda == 0 else -39.0
                b_gen = -36.0 if ridge_lambda == 0 else -40.0
                vs_x = -37.0 if ridge_lambda == 0 else -39.0
                vs_real = -35.0 if ridge_lambda == 0 else -37.0
                theta_norm = 15.0 if ridge_lambda == 0 else 10.0
            rows.append(
                {
                    "state_id": 0,
                    "ridge_lambda": ridge_lambda,
                    "model_id": model_id,
                    "train_nmse_db": train,
                    "B_generalization_nmse_db": b_gen,
                    "commonB_vs_X_cnmse_db": vs_x,
                    "commonB_vs_realB_cnmse_db": vs_real,
                    "theta_l2_norm": theta_norm,
                    "rank": 10,
                }
            )
    return pd.DataFrame(rows)


def test_gain_and_degradation_signs() -> None:
    paired = add_paired_gain_columns(_synthetic_scan_rows())
    ridge = paired[paired["ridge_lambda"] == 1e-8].set_index("model_id")
    assert ridge.loc["Y-C-1", "gain_B_generalization_nmse_db"] == 4.0
    assert ridge.loc["Y-C-1", "degradation_train_nmse_db"] == 1.0
    assert ridge.loc["Y-C-1", "gain_commonB_vs_X_cnmse_db"] == 2.0
    assert ridge.loc["Y-C-1", "gain_commonB_vs_realB_cnmse_db"] == 2.0


def test_uniform_lambda_and_explicit_ilc1_state0() -> None:
    prepared = prepare_state_ridge_data(0)
    result = evaluate_prepared_state(prepared, 1e-8)
    assert tuple(row["model_id"] for row in result.rows) == MODEL_IDS
    assert {row["ridge_lambda"] for row in result.rows} == {1e-8}
    assert [row["ilc_iteration_used"] for row in result.rows] == [None, 1, 1]
    assert [row["train_source"] for row in result.rows] == ["OFF.A", "ILC1.A", "ILC1.C"]
    assert [row["B_generalization_source"] for row in result.rows] == [
        "OFF.B",
        "ILC1.B",
        "ILC1.B",
    ]
    assert result.theta.shape == (3, 10)


def test_scan_summary_applies_protection_constraints() -> None:
    _, feasible_summary = build_scan_summary(_synthetic_scan_rows())
    candidate = feasible_summary.loc[feasible_summary["ridge_lambda"] == 1e-8].iloc[0]
    assert bool(candidate["feasible"])
    _, rejected_summary = build_scan_summary(_synthetic_scan_rows(degraded_x=True))
    rejected = rejected_summary.loc[rejected_summary["ridge_lambda"] == 1e-8].iloc[0]
    assert not bool(rejected["feasible"])
    assert rejected["median_X_B_gen_degradation_dB"] == 1.0


def test_lambda_selection_uses_lexicographic_priority() -> None:
    summary = pd.DataFrame(
        [
            {
                "ridge_lambda": 0.0,
                "feasible": False,
                "median_YC_B_gen_gain_dB": 0.0,
                "median_YC_vs_X_gain_dB": 0.0,
                "median_YC_vs_real_gain_dB": 0.0,
            },
            {
                "ridge_lambda": 1e-8,
                "feasible": True,
                "median_YC_B_gen_gain_dB": 2.0,
                "median_YC_vs_X_gain_dB": 0.5,
                "median_YC_vs_real_gain_dB": 0.5,
            },
            {
                "ridge_lambda": 1e-7,
                "feasible": True,
                "median_YC_B_gen_gain_dB": 2.0,
                "median_YC_vs_X_gain_dB": 0.6,
                "median_YC_vs_real_gain_dB": 0.1,
            },
            {
                "ridge_lambda": 1e-6,
                "feasible": False,
                "median_YC_B_gen_gain_dB": 5.0,
                "median_YC_vs_X_gain_dB": 5.0,
                "median_YC_vs_real_gain_dB": 5.0,
            },
        ]
    )
    selection = select_lambda(summary, stage="test")
    assert selection["status"] == "SELECTED"
    assert selection["selected_lambda"] == 1e-7


def test_selection_payload_freezes_before_validation() -> None:
    split = create_state_split()
    coarse = select_lambda(
        pd.DataFrame(
            [
                {
                    "ridge_lambda": 0.0,
                    "feasible": False,
                    "median_YC_B_gen_gain_dB": 0.0,
                    "median_YC_vs_X_gain_dB": 0.0,
                    "median_YC_vs_real_gain_dB": 0.0,
                },
                {
                    "ridge_lambda": 1e-8,
                    "feasible": True,
                    "median_YC_B_gen_gain_dB": 1.0,
                    "median_YC_vs_X_gain_dB": 1.0,
                    "median_YC_vs_real_gain_dB": 1.0,
                },
            ]
        ),
        stage="coarse",
    )
    payload = build_selection_payload(
        baseline_check={"lambda_zero_regression_passed": True},
        split_frame=split,
        coarse_lambdas=COARSE_LAMBDAS,
        coarse_selection=coarse,
        fine_lambdas=None,
        fine_selection=None,
    )
    assert payload["selected_lambda"] == 1e-8
    assert payload["selection_stage"] == "coarse"
    assert payload["status"] == "SELECTED"


@lru_cache(maxsize=1)
def _state0_comparison():
    return run_ols_vs_ridge_comparison((0,), 1e-8, progress_interval=0)


def test_single_state_comparison_and_validation_shape() -> None:
    comparison = _state0_comparison()
    assert comparison.ols_metrics.shape[0] == 3
    assert comparison.ridge_metrics.shape[0] == 3
    assert comparison.theta_ols.shape == (1, 3, 10)
    assert comparison.theta_ridge.shape == (1, 3, 10)
    assert comparison.state_summary.shape[0] == 1
    validation = build_validation_result(
        comparison.state_summary,
        selected_lambda=1e-8,
        validation_state_count=1,
    )
    assert validation["selected_lambda"] == 1e-8
    assert set(validation["improved_state_counts"]) == {
        "Y_C_B_gen",
        "Y_C_vs_X",
        "Y_C_vs_real",
    }


def test_full_run_artifacts_and_frozen_lambda() -> None:
    result_root = PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_ridge_analysis"
    split = pd.read_csv(result_root / "state_split.csv")
    assert split.shape == (425, 10)
    assert split["split"].value_counts().to_dict() == {
        "development": 340,
        "validation": 85,
    }
    selection = json.loads(
        (result_root / "development" / "lambda_selection.json").read_text(
            encoding="utf-8-sig"
        )
    )
    validation = json.loads(
        (result_root / "validation" / "validation_result.json").read_text(
            encoding="utf-8-sig"
        )
    )
    final_validation = json.loads(
        (result_root / "final" / "scenario2_ridge_validation.json").read_text(
            encoding="utf-8-sig"
        )
    )
    assert selection["status"] == "SELECTED"
    assert selection["lambda_zero_regression"]["lambda_zero_regression_passed"] is True
    assert validation["pass"] is True
    assert selection["selected_lambda"] == validation["selected_lambda"]
    assert final_validation["selected_lambda"] == validation["selected_lambda"]
    metrics = pd.read_csv(result_root / "final" / "scenario2_ridge_model_metrics.csv")
    state_summary = pd.read_csv(result_root / "final" / "scenario2_ridge_state_summary.csv")
    assert metrics.shape[0] == 1275
    assert state_summary.shape[0] == 425
    assert metrics["model_id"].value_counts().to_dict() == {
        "X-A": 425,
        "Y-A-1": 425,
        "Y-C-1": 425,
    }
    assert metrics["ridge_lambda"].nunique() == 1
    theta = np.load(result_root / "final" / "scenario2_ridge_theta.npz")
    assert theta["theta"].shape == (425, 3, 10)
    assert np.all(np.isfinite(theta["theta"]))


def _tree_digest(path: Path, *, include_cache: bool = False) -> str:
    digest = hashlib.sha256()
    files = (
        file
        for file in path.rglob("*")
        if file.is_file()
        and (include_cache or "__pycache__" not in file.parts)
        and (include_cache or file.suffix.lower() != ".pyc")
    )
    for file in sorted(files, key=lambda item: str(item).lower()):
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest_digest() -> str:
    raw_root = Path(
        r"\\?\D:\Project_Files\python_project\project_20260821\data\raw"
    )
    digest = hashlib.sha256()
    for file in sorted(
        (item for item in raw_root.rglob("*") if item.is_file()),
        key=lambda item: str(item).lower(),
    ):
        digest.update(
            str(file.relative_to(raw_root)).replace("\\", "/").encode() + b"\0"
        )
        digest.update(file.stat().st_size.to_bytes(8, "little"))
    return digest.hexdigest()


def test_protected_sources_and_raw_manifest_unchanged() -> None:
    protected = {
        PROJECT_ROOT / "results" / "behavior_model" / "state_000" / "xy_equivalence":
            "b319b05051394bcb7f05c998e4ae22fa338ad130483ecb43af30c4d17d7e4e20",
        PROJECT_ROOT / "results" / "behavior_model" / "state_000" / "ridge_scan":
            "f9ce27eeba0ff2da0b05501826c78e20b53f520991b421e461e6df4b32995aca",
        PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_xy_analysis":
            "715e1436707e13d23c0e6634dec5ab58b348f8d6485dcd81b60c26f83f2cd48e",
        PROJECT_ROOT / "scripts" / "signal_segmentation":
            "bf804cb139a68fdb336865fff514831b14b0f56b178f5840e5dedef92e1311ef",
    }
    for path, expected in protected.items():
        assert _tree_digest(path) == expected
    assert _raw_manifest_digest() == (
        "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0"
    )


def main() -> None:
    tests = [
        test_split_is_deterministic_and_has_fixed_counts,
        test_manifest_is_read_back_without_repartitioning,
        test_validation_leakage_guard,
        test_fine_grid_is_centered_and_contains_zero,
        test_gain_and_degradation_signs,
        test_uniform_lambda_and_explicit_ilc1_state0,
        test_scan_summary_applies_protection_constraints,
        test_lambda_selection_uses_lexicographic_priority,
        test_selection_payload_freezes_before_validation,
        test_single_state_comparison_and_validation_shape,
        test_full_run_artifacts_and_frozen_lambda,
        test_protected_sources_and_raw_manifest_unchanged,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")


if __name__ == "__main__":
    main()
