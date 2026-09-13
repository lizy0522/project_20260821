"""Independent multi-branch basis construction for the 5B OLS scan.

The implementation follows the MATLAB ``PA_MP/code4.9/top.m`` family
definitions, while making family boundaries explicit so no aligned term can
be generated twice.  It does not load data or select a model.
"""

# ruff: noqa: E402,E501

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

COMMON_SUPPORT_TRIM = 26
ODD_ORDERS = (3, 5, 7, 9, 11)
FAMILIES = ("Dynamic", "Static", "MP", "EMem", "Lag", "Lead", "V3", "V5")


@dataclass(frozen=True)
class BasisTerm:
    branch: str
    family: str
    basis_id: str
    parameters: tuple[int, ...]
    effective_delay: int
    expression: str


SOURCE_PROFILE: dict[str, int] = {
    "M_pre": 15,
    "M_D1": 11, "K_S1": 9,
    "K_MP1": 9, "M_MP1_L": 8, "M_MP1_H": 2, "KTH_MP1": 3,
    "K_E1": 7, "M_E1_L": 4, "M_E1_H": 1, "KTH_E1": 3,
    "K_Lag1": 7, "M_Lag1_L": 2, "M_Lag1_H": 2, "KTH_Lag1": 3, "L_Lag1": 3,
    "K_Lead1": 7, "M_Lead1_L": 1, "M_Lead1_H": 1, "KTH_Lead1": 3, "L_Lead1": 1,
    "M_V3_1": 3, "M_V5_1": 2,
    "M_D2": 10, "K_S2": 11,
    "K_MP2": 9, "M_MP2_L": 9, "M_MP2_H": 2, "KTH_MP2": 5,
    "K_E2": 11, "M_E2_L": 9, "M_E2_H": 2, "KTH_E2": 5,
    "K_Lag2": 9, "M_Lag2_L": 5, "M_Lag2_H": 2, "KTH_Lag2": 3, "L_Lag2": 3,
    "K_Lead2": 7, "M_Lead2_L": 5, "M_Lead2_H": 2, "KTH_Lead2": 3, "L_Lead2": 3,
    "M_V3_2": 3, "M_V5_2": 2,
    "M_Env": 3,
}

COMPACT_PROFILE: dict[str, int] = {
    "M_pre": 4,
    "M_D1": 4, "K_S1": 7,
    "K_MP1": 7, "M_MP1_L": 3, "M_MP1_H": 1, "KTH_MP1": 3,
    "K_E1": 5, "M_E1_L": 2, "M_E1_H": 1, "KTH_E1": 3,
    "K_Lag1": 5, "M_Lag1_L": 2, "M_Lag1_H": 1, "KTH_Lag1": 3, "L_Lag1": 2,
    "K_Lead1": 5, "M_Lead1_L": 2, "M_Lead1_H": 1, "KTH_Lead1": 3, "L_Lead1": 2,
    "M_V3_1": 1, "M_V5_1": 1,
    "M_D2": 4, "K_S2": 7,
    "K_MP2": 7, "M_MP2_L": 3, "M_MP2_H": 1, "KTH_MP2": 3,
    "K_E2": 5, "M_E2_L": 2, "M_E2_H": 1, "KTH_E2": 3,
    "K_Lag2": 5, "M_Lag2_L": 2, "M_Lag2_H": 1, "KTH_Lag2": 3, "L_Lag2": 2,
    "K_Lead2": 5, "M_Lead2_L": 2, "M_Lead2_H": 1, "KTH_Lead2": 3, "L_Lead2": 2,
    "M_V3_2": 1, "M_V5_2": 1,
    "M_Env": 2,
}

PARAMETER_GRIDS: dict[str, tuple[int, ...]] = {
    "M_pre": (1, 2, 4, 6, 8, 10, 12, 15),
    "M_D1": (0, 1, 2, 4, 6, 8, 10, 11),
    "K_S1": (3, 5, 7, 9, 11),
    "K_MP1": (3, 5, 7, 9, 11), "M_MP1_L": (1, 2, 3, 4, 5, 6, 8, 9), "M_MP1_H": (1, 2, 3), "KTH_MP1": (3, 5, 7),
    "K_E1": (3, 5, 7, 9, 11), "M_E1_L": (1, 2, 3, 4, 5, 6, 8, 9), "M_E1_H": (1, 2, 3), "KTH_E1": (3, 5, 7),
    "K_Lag1": (3, 5, 7, 9), "M_Lag1_L": (1, 2, 3, 4, 5), "M_Lag1_H": (1, 2, 3), "KTH_Lag1": (3, 5, 7), "L_Lag1": (1, 2, 3),
    "K_Lead1": (3, 5, 7, 9), "M_Lead1_L": (1, 2, 3, 4, 5), "M_Lead1_H": (1, 2, 3), "KTH_Lead1": (3, 5, 7), "L_Lead1": (1, 2, 3),
    "M_V3_1": (1, 2, 3), "M_V5_1": (1, 2),
    "M_D2": (0, 1, 2, 4, 6, 8, 10, 11), "K_S2": (3, 5, 7, 9, 11),
    "K_MP2": (3, 5, 7, 9, 11), "M_MP2_L": (1, 2, 3, 4, 5, 6, 8, 9), "M_MP2_H": (1, 2, 3), "KTH_MP2": (3, 5, 7),
    "K_E2": (3, 5, 7, 9, 11), "M_E2_L": (1, 2, 3, 4, 5, 6, 8, 9), "M_E2_H": (1, 2, 3), "KTH_E2": (3, 5, 7),
    "K_Lag2": (3, 5, 7, 9), "M_Lag2_L": (1, 2, 3, 4, 5), "M_Lag2_H": (1, 2, 3), "KTH_Lag2": (3, 5, 7), "L_Lag2": (1, 2, 3),
    "K_Lead2": (3, 5, 7, 9), "M_Lead2_L": (1, 2, 3, 4, 5), "M_Lead2_H": (1, 2, 3), "KTH_Lead2": (3, 5, 7), "L_Lead2": (1, 2, 3),
    "M_V3_2": (1, 2, 3), "M_V5_2": (1, 2), "M_Env": (0, 1, 2, 3, 4),
}

PARAMETER_BLOCKS = (
    ("M_pre",), ("M_D1",), ("K_S1",),
    ("K_MP1",), ("M_MP1_L",), ("M_MP1_H",), ("KTH_MP1",),
    ("K_E1",), ("M_E1_L",), ("M_E1_H",), ("KTH_E1",),
    ("K_Lag1",), ("M_Lag1_L",), ("M_Lag1_H",), ("KTH_Lag1",), ("L_Lag1",),
    ("K_Lead1",), ("M_Lead1_L",), ("M_Lead1_H",), ("KTH_Lead1",), ("L_Lead1",),
    ("M_V3_1",), ("M_V5_1",),
    ("M_D2",), ("K_S2",),
    ("K_MP2",), ("M_MP2_L",), ("M_MP2_H",), ("KTH_MP2",),
    ("K_E2",), ("M_E2_L",), ("M_E2_H",), ("KTH_E2",),
    ("K_Lag2",), ("M_Lag2_L",), ("M_Lag2_H",), ("KTH_Lag2",), ("L_Lag2",),
    ("K_Lead2",), ("M_Lead2_L",), ("M_Lead2_H",), ("KTH_Lead2",), ("L_Lead2",),
    ("M_V3_2",), ("M_V5_2",), ("M_Env",),
)


def _odd_orders(k_max: int) -> tuple[int, ...]:
    return tuple(order for order in ODD_ORDERS if order <= int(k_max))


def _value(config: Mapping[str, int], name: str) -> int:
    return int(config[name])


def _term(branch: str, family: str, parameters: tuple[int, ...], delay: int, expression: str) -> BasisTerm:
    suffix = "_".join(str(value) for value in parameters)
    family_code = {"Dynamic": "DYN", "Static": "STATIC", "MP": "MP", "EMem": "EMEM", "Lag": "LAG", "Lead": "LEAD", "V3": "V3", "V5": "V5", "Envelope": "ENV_DYN"}[family]
    return BasisTerm(branch, family, f"{branch}_{family_code}_{suffix}", parameters, int(delay), expression)


def _poly_memory_terms(branch: str, family: str, config: Mapping[str, int], *, input_label: str) -> list[BasisTerm]:
    index = branch[-1]
    k_name, low_name, high_name, kth_name = {
        "MP": (f"K_MP{index}", f"M_MP{index}_L", f"M_MP{index}_H", f"KTH_MP{index}"),
        "EMem": (f"K_E{index}", f"M_E{index}_L", f"M_E{index}_H", f"KTH_E{index}"),
    }[family]
    terms: list[BasisTerm] = []
    for order in _odd_orders(_value(config, k_name)):
        depth = _value(config, low_name) if order <= _value(config, kth_name) else _value(config, high_name)
        for memory in range(1, depth + 1):
            if family == "MP":
                expr = f"{input_label}[n-{memory}]|{input_label}[n-{memory}]|^{order - 1}"
            else:
                expr = f"{input_label}[n]|{input_label}[n-{memory}]|^{order - 1}"
            terms.append(_term(branch, family, (order, memory), memory, expr))
    return terms


def build_branch_terms(config: Mapping[str, int], branch: str) -> list[BasisTerm]:
    """Return one branch's deterministic, mathematically unique family terms."""

    if branch not in ("B1", "B2"):
        raise ValueError("branch must be B1 or B2")
    index = branch[-1]
    input_label = "z" if branch == "B1" else "x"
    terms: list[BasisTerm] = []
    dynamic_depth = _value(config, f"M_D{index}")
    for memory in range(dynamic_depth + 1):
        terms.append(_term(branch, "Dynamic", (memory,), memory, f"{input_label}[n-{memory}]"))
    for order in _odd_orders(_value(config, f"K_S{index}")):
        terms.append(_term(branch, "Static", (order,), 0, f"{input_label}[n]|{input_label}[n]|^{order - 1}"))
    terms.extend(_poly_memory_terms(branch, "MP", config, input_label=input_label))
    terms.extend(_poly_memory_terms(branch, "EMem", config, input_label=input_label))

    for family in ("Lag", "Lead"):
        k_name = f"K_{family}1" if family == "Lag" and branch == "B1" else f"K_{family}{index}"
        low_name = f"M_{family}{index}_L"
        high_name = f"M_{family}{index}_H"
        kth_name = f"KTH_{family}{index}"
        l_name = f"L_{family}{index}"
        for order in _odd_orders(_value(config, k_name)):
            depth = _value(config, low_name) if order <= _value(config, kth_name) else _value(config, high_name)
            for memory in range(1, depth + 1):
                for length in range(1, _value(config, l_name) + 1):
                    if family == "Lead" and length > memory:
                        continue
                    envelope_delay = memory + length if family == "Lag" else memory - length
                    if family == "Lag":
                        expression = f"{input_label}[n-{memory}]|{input_label}[n-{envelope_delay}]|^{order - 1}"
                    else:
                        expression = f"{input_label}[n-{memory}]|{input_label}[n-{envelope_delay}]|^{order - 1}"
                    terms.append(_term(branch, family, (order, memory, length), max(memory, envelope_delay), expression))

    for memory_1, memory_2, memory_3 in itertools.combinations_with_replacement(range(_value(config, f"M_V3_{index}") + 1), 3):
        if memory_2 == memory_3:
            continue
        expression = f"{input_label}[n-{memory_1}]{input_label}[n-{memory_2}]{input_label}*[n-{memory_3}]"
        terms.append(_term(branch, "V3", (memory_1, memory_2, memory_3), max(memory_1, memory_2, memory_3), expression))
    for delays in itertools.combinations_with_replacement(range(_value(config, f"M_V5_{index}") + 1), 5):
        if delays[1] == delays[2] == delays[3] == delays[4]:
            continue
        expression = "".join(f"{input_label}{'*' if index >= 3 else ''}[n-{delay}]" for index, delay in enumerate(delays))
        terms.append(_term(branch, "V5", tuple(delays), max(delays), expression))
    return terms


def build_branch3_terms(config: Mapping[str, int]) -> list[BasisTerm]:
    depth = _value(config, "M_Env")
    return [_term("B3", "Envelope", (memory,), memory, f"|x[n-{memory}]|") for memory in range(depth + 1)]


def build_basis_terms(config: Mapping[str, int]) -> list[BasisTerm]:
    terms = build_branch_terms(config, "B1") + build_branch_terms(config, "B2") + build_branch3_terms(config)
    validate_basis_uniqueness(terms)
    return terms


def validate_basis_uniqueness(terms: Sequence[BasisTerm]) -> None:
    ids = [term.basis_id for term in terms]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise ValueError(f"duplicate canonical basis IDs: {duplicates}")
    mathematical_keys = [(term.branch, term.family, term.parameters) for term in terms]
    if len(mathematical_keys) != len(set(mathematical_keys)):
        raise ValueError("duplicate mathematical basis terms")
    if any(term.family in {"Static", "MP", "EMem", "Lag", "Lead"} and term.parameters[0] % 2 != 1 for term in terms):
        raise ValueError("non-odd polynomial order leaked into family basis")


def effective_delay_by_family(config: Mapping[str, int]) -> dict[str, int]:
    b1 = build_branch_terms(config, "B1")
    b2 = build_branch_terms(config, "B2")
    b3 = build_branch3_terms(config)
    return {
        "B1_Dynamic": _value(config, "M_pre") + max((term.effective_delay for term in b1 if term.family == "Dynamic"), default=0),
        "B1_Static": _value(config, "M_pre"),
        "B1_MP": _value(config, "M_pre") + max((term.effective_delay for term in b1 if term.family == "MP"), default=0),
        "B1_EMem": _value(config, "M_pre") + max((term.effective_delay for term in b1 if term.family == "EMem"), default=0),
        "B1_Lag": _value(config, "M_pre") + max((term.effective_delay for term in b1 if term.family == "Lag"), default=0),
        "B1_Lead": _value(config, "M_pre") + max((term.effective_delay for term in b1 if term.family == "Lead"), default=0),
        "B1_V3": _value(config, "M_pre") + max((term.effective_delay for term in b1 if term.family == "V3"), default=0),
        "B1_V5": _value(config, "M_pre") + max((term.effective_delay for term in b1 if term.family == "V5"), default=0),
        "B2": max((term.effective_delay for term in b2), default=0),
        "B3": max((term.effective_delay for term in b3), default=0),
    }


def validate_config(config: Mapping[str, int], *, max_delay: int = COMMON_SUPPORT_TRIM) -> dict[str, Any]:
    required = set(SOURCE_PROFILE)
    missing = required - set(config)
    if missing:
        raise ValueError(f"config missing parameters: {sorted(missing)}")
    values = {key: int(value) for key, value in config.items()}
    if values["M_pre"] < 0 or values["M_Env"] < 0:
        raise ValueError("memory must be non-negative")
    for suffix in ("1", "2"):
        for family in ("MP", "E", "Lag", "Lead"):
            low = values[f"M_{family}{suffix}_L"]
            high = values[f"M_{family}{suffix}_H"]
            if low < 0 or high < 0 or high > low:
                raise ValueError(f"high memory must not exceed low memory for {family}{suffix}")
            if values[f"KTH_{family}{suffix}"] > values[f"K_{family}{suffix}"]:
                raise ValueError(f"KTH must not exceed K for {family}{suffix}")
    delays = effective_delay_by_family(values)
    max_effective = max(delays.values())
    if max_effective > max_delay:
        raise ValueError(f"effective delay {max_effective} exceeds common support trim {max_delay}")
    terms = build_basis_terms(values)
    counts: dict[str, int] = {}
    for term in terms:
        key = f"{term.branch}_{term.family}"
        counts[key] = counts.get(key, 0) + 1
    return {"valid": True, "basis_count": len(terms), "family_counts": counts, "effective_delays": delays, "max_effective_delay": max_effective, "basis_ids": [term.basis_id for term in terms]}


def prelinear_basis(x: np.ndarray, memory: int, trim: int = COMMON_SUPPORT_TRIM) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    if trim < memory or x.size <= trim:
        raise ValueError("prelinear memory/support is invalid")
    return np.column_stack([x[trim - delay : x.size - delay] for delay in range(memory + 1)])


def apply_prelinear(x: np.ndarray, coefficients: np.ndarray, memory: int, *, output_length: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    coefficients = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
    if coefficients.size != memory + 1:
        raise ValueError("prelinear coefficient count mismatch")
    start = memory
    phi = np.column_stack([x[start - delay : x.size - delay] for delay in range(memory + 1)])
    z = phi @ coefficients
    if output_length is not None:
        if output_length > z.size:
            raise ValueError("requested prelinear output exceeds available samples")
        z = z[:output_length]
    z_full = np.full(x.size, np.nan + 0j, dtype=np.complex128)
    z_full[start:] = z
    return z_full


def _delayed(u: np.ndarray, delay: int, trim: int) -> np.ndarray:
    return u[trim - delay : u.size - delay]


def build_family_matrix(u: np.ndarray, terms: Sequence[BasisTerm], *, trim: int = COMMON_SUPPORT_TRIM) -> np.ndarray:
    """Build a matrix for a supplied term list on rows corresponding to raw ``trim:``."""

    u = np.asarray(u, dtype=np.complex128).reshape(-1)
    columns: list[np.ndarray] = []
    for term in terms:
        params = term.parameters
        if term.family in {"Dynamic", "Envelope"}:
            column = _delayed(u, params[0], trim)
        elif term.family == "Static":
            value = _delayed(u, 0, trim)
            column = value * np.abs(value) ** (params[0] - 1)
        elif term.family == "MP":
            value = _delayed(u, params[1], trim)
            column = value * np.abs(value) ** (params[0] - 1)
        elif term.family == "EMem":
            carrier = _delayed(u, 0, trim)
            envelope = _delayed(u, params[1], trim)
            column = carrier * np.abs(envelope) ** (params[0] - 1)
        elif term.family in {"Lag", "Lead"}:
            order, memory, length = params
            carrier = _delayed(u, memory, trim)
            envelope_delay = memory + length if term.family == "Lag" else memory - length
            envelope = _delayed(u, envelope_delay, trim)
            column = carrier * np.abs(envelope) ** (order - 1)
        elif term.family == "V3":
            m1, m2, m3 = params
            column = _delayed(u, m1, trim) * _delayed(u, m2, trim) * np.conj(_delayed(u, m3, trim))
        elif term.family == "V5":
            m1, m2, m3, m4, m5 = params
            column = _delayed(u, m1, trim) * _delayed(u, m2, trim) * _delayed(u, m3, trim) * np.conj(_delayed(u, m4, trim)) * np.conj(_delayed(u, m5, trim))
        else:
            raise ValueError(f"unknown basis family {term.family}")
        columns.append(np.asarray(column, dtype=np.complex128))
    if not columns:
        return np.empty((u.size - trim, 0), dtype=np.complex128)
    matrix = np.column_stack(columns).astype(np.complex128, copy=False)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("basis matrix contains NaN/Inf; effective delay/support mismatch")
    return matrix


def build_total_matrix(config: Mapping[str, int], x: np.ndarray, z_full: np.ndarray, *, trim: int = COMMON_SUPPORT_TRIM) -> tuple[np.ndarray, list[BasisTerm]]:
    terms = build_basis_terms(config)
    b1_terms = [term for term in terms if term.branch == "B1"]
    b2_terms = [term for term in terms if term.branch == "B2"]
    b3_terms = [term for term in terms if term.branch == "B3"]
    b1 = build_family_matrix(z_full, b1_terms, trim=trim)
    b2 = build_family_matrix(np.asarray(x, dtype=np.complex128), b2_terms, trim=trim)
    b3 = build_family_matrix(np.abs(np.asarray(x, dtype=np.complex128)), b3_terms, trim=trim)
    matrix = np.column_stack((b1, b2, b3))
    return matrix, terms


def basis_manifest(config: Mapping[str, int]) -> list[dict[str, Any]]:
    return [{"branch": term.branch, "family": term.family, "basis_id": term.basis_id, "parameters": list(term.parameters), "effective_delay": term.effective_delay, "expression": term.expression} for term in build_basis_terms(config)]


def config_json(config: Mapping[str, int]) -> str:
    return json.dumps({key: int(config[key]) for key in sorted(config)}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "BasisTerm", "COMMON_SUPPORT_TRIM", "COMPACT_PROFILE", "FAMILIES", "ODD_ORDERS", "PARAMETER_BLOCKS", "PARAMETER_GRIDS", "SOURCE_PROFILE",
    "apply_prelinear", "basis_manifest", "build_basis_terms", "build_branch3_terms", "build_branch_terms", "build_family_matrix", "build_total_matrix", "config_json", "effective_delay_by_family", "prelinear_basis", "validate_basis_uniqueness", "validate_config",
]
