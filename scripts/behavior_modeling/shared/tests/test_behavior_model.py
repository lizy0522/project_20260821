"""
功能说明：直接执行或由pytest收集的冻结10基函数正向行为模型测试。
输入：确定性合成复信号及data_manager只读加载的state0 X数据。
输出：配置、基函数、LS恢复、模型封装、评价和state0拟合测试PASS信息。
用途：不写任务结果，仅验证数学实现、接口和真实数据数值秩。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from data_management.shared import load_by_id  # noqa: E402

from behavior_modeling.shared import (  # noqa: E402
    BASIS_TERMS,
    MAX_DELAY,
    MP_CONFIG,
    NUM_COEFFICIENTS,
    MemoryPolynomialModel,
    build_mp_basis,
    calculate_cnmse,
    calculate_nmse,
    extract_behavior_coefficient,
)


def _complex_signal(sample_count: int = 512) -> np.ndarray:
    rng = np.random.default_rng(20260825)
    x = rng.normal(size=sample_count) + 1j * rng.normal(size=sample_count)
    return x / np.max(np.abs(x))


def test_frozen_config_and_basis_shape() -> None:
    x = _complex_signal()
    basis = build_mp_basis(x, MP_CONFIG["orders"], MP_CONFIG["memory_depth"])
    assert NUM_COEFFICIENTS == 10
    assert len(BASIS_TERMS) == 10
    assert MAX_DELAY == 2
    assert basis.shape == (x.size - 2, 10)
    assert BASIS_TERMS == (
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (5, 0),
        (7, 0),
        (9, 0),
    )


def test_complex_lstsq_recovers_known_coefficients() -> None:
    x = _complex_signal()
    basis = build_mp_basis(x, MP_CONFIG["orders"], MP_CONFIG["memory_depth"])
    rng = np.random.default_rng(1234)
    theta_true = rng.normal(size=10) + 1j * rng.normal(size=10)
    y = np.zeros_like(x)
    y[MAX_DELAY:] = basis @ theta_true
    theta = extract_behavior_coefficient(x, y, MP_CONFIG)
    np.testing.assert_allclose(theta, theta_true, rtol=1e-10, atol=1e-10)


def test_model_fit_predict_and_metrics() -> None:
    x = _complex_signal()
    basis = build_mp_basis(x, MP_CONFIG["orders"], MP_CONFIG["memory_depth"])
    theta_true = np.arange(1, 11) * (0.02 + 0.01j)
    y = np.zeros_like(x)
    y[MAX_DELAY:] = basis @ theta_true
    model = MemoryPolynomialModel().fit(x, y)
    prediction = model.predict(x)
    assert model.theta is not None
    assert model.fit_diagnostics is not None
    assert model.fit_diagnostics.rank == 10
    np.testing.assert_allclose(model.theta, theta_true, rtol=1e-10, atol=1e-10)
    assert calculate_nmse(y[MAX_DELAY:], prediction) < -250.0
    assert np.isneginf(calculate_cnmse(prediction, prediction))


def test_state0_x_model_is_finite_and_full_rank() -> None:
    data = load_by_id(0)
    x = np.asarray(data["xin"])[:, 0]
    y = np.asarray(data["yout_withoutdpd"])[:, 0]
    model = MemoryPolynomialModel().fit(x, y)
    prediction = model.predict(x)
    assert model.theta is not None
    assert model.theta.shape == (10,)
    assert np.all(np.isfinite(model.theta))
    assert model.fit_diagnostics is not None
    assert model.fit_diagnostics.rank == 10
    assert np.isfinite(calculate_nmse(y[MAX_DELAY:], prediction))


def test_state0_y_inputs_require_peak_normalization() -> None:
    data = load_by_id(0)
    x_ilc = np.asarray(data["xin_pd_ori_ilc"])
    assert x_ilc.shape == (24576, 5)
    for column in range(x_ilc.shape[1]):
        peak_scale = np.max(np.abs(x_ilc[:, column]))
        assert peak_scale > 0
        normalized = x_ilc[:, column] / peak_scale
        np.testing.assert_allclose(np.max(np.abs(normalized)), 1.0, rtol=0, atol=1e-14)


def main() -> None:
    tests = [
        ("frozen config and basis test", test_frozen_config_and_basis_shape),
        ("complex lstsq recovery test", test_complex_lstsq_recovers_known_coefficients),
        ("model predict and metrics test", test_model_fit_predict_and_metrics),
        ("state0 X full-rank test", test_state0_x_model_is_finite_and_full_rank),
        ("state0 Y normalization test", test_state0_y_inputs_require_peak_normalization),
    ]
    for label, test in tests:
        test()
        print(f"{label}: PASS")


if __name__ == "__main__":
    main()
