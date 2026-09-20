"""Leakage-gated final Hard-20 Train/Test evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from core.shared.metrics import nmse
from core.shared.signal import adjust_complex_gain
from data_management.shared import get_state_info

from .config import DMAX, TARGET_NMSE_DB
from .data_preparation import HardCacheSpec, load_hard_cache_arrays, prepare_aligned_state
from .frozen_centered_dictionary import FrozenCenteredBasis, build_frozen_centered_bank
from .model_solver import fit_ols, fit_ridge
from .volterra_dictionary import VolterraBasis, build_basis_bank


def assert_test_access_allowed(*, model_frozen: bool) -> None:
    """Hard fail when Test processing is requested before model freeze."""

    if not model_frozen:
        raise AssertionError("Test data access is forbidden before support/lambda freeze")


def evaluate_final_model(
    spec: HardCacheSpec,
    terms: Sequence[VolterraBasis],
    support: Sequence[int],
    ridge_lambda: float,
    ranking_by_state: Mapping[int, Mapping[str, object]],
    *,
    model_frozen: bool,
) -> list[dict[str, object]]:
    """Fit each state on full Train and predict independently constructed Test."""

    assert_test_access_allowed(model_frozen=model_frozen)
    arrays = load_hard_cache_arrays(spec)
    support_tuple = tuple(sorted(int(index) for index in support))
    rows: list[dict[str, object]] = []
    for state_index, state_id in enumerate(spec.state_ids):
        train_phi = np.asarray(
            arrays["full_bank"][state_index][:, support_tuple], dtype=np.complex128
        )
        train_target = np.asarray(arrays["full_target"][state_index], dtype=np.complex128)
        fit = (
            fit_ols(train_phi, train_target, scale_columns=True)
            if ridge_lambda == 0.0
            else fit_ridge(train_phi, train_target, ridge_lambda)
        )
        x_test = np.asarray(arrays["x_test"][state_index], dtype=np.complex128)
        y_test_aligned = np.asarray(arrays["y_test_aligned"][state_index], dtype=np.complex128)
        y_test_adjusted, test_gain = adjust_complex_gain(x_test, y_test_aligned)
        test_bank = build_basis_bank(x_test, terms)
        test_target = np.asarray(y_test_adjusted[DMAX:], dtype=np.complex128)
        test_prediction = np.asarray(test_bank[:, support_tuple] @ fit.theta, dtype=np.complex128)
        test_nmse_db = float(nmse(test_target, test_prediction))
        train_pass = bool(fit.nmse_db < TARGET_NMSE_DB)
        test_pass = bool(test_nmse_db < TARGET_NMSE_DB)
        if train_pass and test_pass:
            failure_type = "A_complete_success"
        elif train_pass:
            failure_type = "B_train_success_test_generalization_failure"
        else:
            failure_type = "C_train_modeling_failure"
        state_info = get_state_info(state_id)
        rank_row = ranking_by_state[int(state_id)]
        rows.append(
            {
                "state_id": int(state_id),
                "funMng": int(state_info["funMng"]),
                "funAng": int(state_info["funAng"]),
                "secMng": int(state_info["secMng"]),
                "secAng": int(state_info["secAng"]),
                "Vm": float(state_info["Vm"]),
                "Pin": float(state_info["Pin"]),
                "original_train_noDPD_NMSE_dB": float(rank_row["train_noDPD_NMSE_dB"]),
                "final_train_nmse_db": float(fit.nmse_db),
                "final_test_nmse_db": test_nmse_db,
                "pass_train40": train_pass,
                "pass_test40": test_pass,
                "pass_both40": bool(train_pass and test_pass),
                "failure_type": failure_type,
                "test_gain_real": float(test_gain.real),
                "test_gain_imag": float(test_gain.imag),
                "train_rank": int(fit.rank),
                "train_condition_number": float(fit.condition_number),
            }
        )
        print(f"[FINAL TEST] {state_index + 1}/20", flush=True)
    return rows


def evaluate_frozen_centered_final_model(
    state_ids: Sequence[int],
    terms: Sequence[FrozenCenteredBasis],
    support: Sequence[int],
    ridge_lambda: float,
    hard20_rank_by_state: Mapping[int, int],
    inner_w_by_state: Mapping[int, float],
    *,
    model_frozen: bool,
) -> list[dict[str, object]]:
    """Run the one-time Train fit and segment-local Test evaluation."""

    assert_test_access_allowed(model_frozen=model_frozen)
    support_tuple = tuple(sorted(int(index) for index in support))
    rows = []
    for done, state_id in enumerate(state_ids, start=1):
        prepared = prepare_aligned_state(int(state_id))
        train_bank = build_frozen_centered_bank(prepared.x_train, terms, support_tuple)
        train_target = prepared.y_train_adjusted[DMAX:]
        fit = (
            fit_ols(train_bank, train_target, scale_columns=True)
            if ridge_lambda == 0.0
            else fit_ridge(train_bank, train_target, ridge_lambda)
        )
        y_test_adjusted, test_gain = adjust_complex_gain(
            prepared.x_test,
            prepared.y_test_time_aligned,
        )
        test_bank = build_frozen_centered_bank(prepared.x_test, terms, support_tuple)
        test_target = y_test_adjusted[DMAX:]
        test_prediction = test_bank @ fit.theta
        test_nmse = float(nmse(test_target, test_prediction))
        train_pass = bool(fit.nmse_db < TARGET_NMSE_DB)
        test_pass = bool(test_nmse < TARGET_NMSE_DB)
        if train_pass and test_pass:
            failure_type = "A_complete_success"
        elif train_pass:
            failure_type = "B_train_success_test_generalization_failure"
        else:
            failure_type = "C_train_modeling_failure"
        state_info = get_state_info(int(state_id))
        inner_w = float(inner_w_by_state[int(state_id)])
        rows.append(
            {
                "state_id": int(state_id),
                "hard20_rank": int(hard20_rank_by_state[int(state_id)]),
                "funMng": int(state_info["funMng"]),
                "funAng": int(state_info["funAng"]),
                "secMng": int(state_info["secMng"]),
                "secAng": int(state_info["secAng"]),
                "Vm": float(state_info["Vm"]),
                "Pin": float(state_info["Pin"]),
                "train_nmse_db": float(fit.nmse_db),
                "test_nmse_db": test_nmse,
                "pass_train40": train_pass,
                "pass_test40": test_pass,
                "pass_both40": bool(train_pass and test_pass),
                "inner_W_db": inner_w,
                "pass_inner_W40": bool(inner_w < TARGET_NMSE_DB),
                "failure_type": failure_type,
                "test_gain_real": float(test_gain.real),
                "test_gain_imag": float(test_gain.imag),
            }
        )
        print(f"[FINAL TEST] {done}/{len(state_ids)}", flush=True)
    return rows


__all__ = [
    "assert_test_access_allowed",
    "evaluate_final_model",
    "evaluate_frozen_centered_final_model",
]
