"""
功能说明：验证Ridge参数防护、增广复数LS、OLS复现、theta范数和State0 16×11统计口径。
输入：小型复数合成问题、现有OLS xy_equivalence结果和State0 canonical数据。
输出：Ridge solver、扫描数量、公共probe、Y真实B泛化和基线保护测试PASS信息。
用途：测试oracle可用正规方程solve，但生产Ridge路径只允许增广lstsq。
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

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

from data_management.shared import load_by_id  # noqa: E402
from signal_segmentation.shared import build_partition_from_xin, get_common_probe  # noqa: E402

from behavior_modeling.scenario_2.state0_ridge_scan.ridge_scan import (  # noqa: E402
    RIDGE_LAMBDAS,
    RidgeScanResult,
    run_state0_ridge_scan,
    validate_ols_reproduction,
)
from behavior_modeling.shared.coefficient import _solve_basis_ols  # noqa: E402
from behavior_modeling.shared.ridge import (  # noqa: E402
    fit_coefficients_ridge,
    validate_ridge_lambda,
)


def _synthetic_system() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260826)
    phi = rng.normal(size=(128, 10)) + 1j * rng.normal(size=(128, 10))
    theta = rng.normal(size=10) + 1j * rng.normal(size=10)
    noise = 0.01 * (rng.normal(size=128) + 1j * rng.normal(size=128))
    return phi.astype(np.complex128), (phi @ theta + noise).astype(np.complex128)


@lru_cache(maxsize=1)
def _scan() -> RidgeScanResult:
    return run_state0_ridge_scan()


def test_invalid_lambda_guards() -> None:
    for value in (-1.0, np.nan, np.inf, -np.inf):
        try:
            validate_ridge_lambda(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法lambda必须报错：{value}")


def test_lambda_zero_uses_exact_ols_path() -> None:
    phi, y = _synthetic_system()
    theta_ridge, diagnostics = fit_coefficients_ridge(phi, y, 0.0)
    theta_ols, _, rank, _ = _solve_basis_ols(phi, y)
    np.testing.assert_array_equal(theta_ridge, theta_ols)
    assert diagnostics.solver_path == "existing_ols"
    assert diagnostics.rank_phi == rank


def test_augmented_ridge_matches_normal_equation_oracle() -> None:
    phi, y = _synthetic_system()
    ridge_lambda = 1e-2
    theta, diagnostics = fit_coefficients_ridge(phi, y, ridge_lambda)
    n_samples = phi.shape[0]
    gram = phi.conj().T @ phi / n_samples
    rhs = phi.conj().T @ y / n_samples
    oracle = np.linalg.solve(gram + ridge_lambda * np.eye(phi.shape[1]), rhs)
    np.testing.assert_allclose(theta, oracle, rtol=1e-11, atol=1e-11)
    assert diagnostics.solver_path == "augmented_lstsq"


def test_theta_norm_decreases_for_large_lambda() -> None:
    phi, y = _synthetic_system()
    theta_ols, _ = fit_coefficients_ridge(phi, y, 0.0)
    theta_large, _ = fit_coefficients_ridge(phi, y, 100.0)
    assert np.linalg.norm(theta_large) < np.linalg.norm(theta_ols)


def test_state0_scan_counts_and_uniform_lambda() -> None:
    result = _scan()
    assert result.lambdas.shape == (16,)
    assert len(result.model_ids) == 11
    assert result.theta.shape == (16, 11, 10)
    assert len(result.model_rows) == 176
    assert len(result.summary_rows) == 16
    for ridge_lambda in RIDGE_LAMBDAS:
        rows = [row for row in result.model_rows if row["ridge_lambda"] == ridge_lambda]
        assert len(rows) == 11
        assert {row["ridge_lambda"] for row in rows} == {ridge_lambda}
        assert sum(not np.isnan(row["commonB_vs_X_CNMSE_dB"]) for row in rows) == 10
        assert sum(not np.isnan(row["commonB_vs_realB_CNMSE_dB"]) for row in rows) == 11


def test_lambda_zero_reproduces_existing_ols() -> None:
    baseline_root = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /"state0_xy_equivalence"  # noqa: E501
    validation = validate_ols_reproduction(_scan(), baseline_root, atol=1e-10)
    assert validation["ols_baseline_reproduced"] is True


def test_common_probe_and_y_generalization_sources_remain_frozen() -> None:
    result = _scan()
    state = load_by_id(0)
    partition = build_partition_from_xin(np.asarray(state["xin"]))
    expected_probe = get_common_probe(state, partition)
    np.testing.assert_array_equal(result.common_probe, expected_probe)
    for row in result.model_rows:
        if row["behavior_class"] == "Y":
            assert row["B_generalization_source"] == f"ILC[{row['ilc_iteration']}].B"
        assert row["common_B_probe_source"] == "OFF.B.input"


def test_all_theta_finite_rank_and_norm_behavior() -> None:
    result = _scan()
    assert np.all(np.isfinite(result.theta))
    assert all(row["rank"] == 10 for row in result.model_rows)
    assert result.diagnostic["all_theta_norm_nonincreasing_by_model"] is True


def main() -> None:
    tests = [
        ("invalid lambda guard test", test_invalid_lambda_guards),
        ("lambda zero OLS path test", test_lambda_zero_uses_exact_ols_path),
        ("augmented ridge oracle test", test_augmented_ridge_matches_normal_equation_oracle),
        ("theta norm shrinkage test", test_theta_norm_decreases_for_large_lambda),
        ("State0 scan count test", test_state0_scan_counts_and_uniform_lambda),
        ("lambda zero baseline reproduction test", test_lambda_zero_reproduces_existing_ols),
        ("probe and Y-B source test", test_common_probe_and_y_generalization_sources_remain_frozen),
        ("theta finite rank norm test", test_all_theta_finite_rank_and_norm_behavior),
    ]
    for label, test in tests:
        test()
        print(f"{label}: PASS")


if __name__ == "__main__":
    main()
