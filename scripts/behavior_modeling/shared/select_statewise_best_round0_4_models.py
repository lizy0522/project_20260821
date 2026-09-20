"""Select and rebuild one real Round 0--4 candidate per state and side."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.shared.basis import build_mp_basis  # noqa: E402
from behavior_modeling.shared.evaluation import calculate_nmse  # noqa: E402
from behavior_modeling.shared.ridge import fit_coefficients_ridge  # noqa: E402
from behavior_modeling.shared.statewise_adaptive_model_search import (  # noqa: E402
    DPD_SHAREABLE_THRESHOLD_DB,
    STATE_COUNT,
    PreparedStatePairs,
    prepare_state_pairs,
)

SOURCE_ROOT = (
    PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /"scenario_2_statewise_adaptive_Aend_C2_5B"  # noqa: E501
)
RESULT_ROOT = (
    PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /"scenario_2_statewise_best_round0_4_5B"  # noqa: E501
)
GRID_PATH = SOURCE_ROOT / "candidate_model_grid.csv.gz"
AEND_LOG_PATH = SOURCE_ROOT / "candidate_search_Aend.csv.gz"
C2_LOG_PATH = SOURCE_ROOT / "candidate_search_C2.csv.gz"
PARTIAL_VALIDATION_PATH = SOURCE_ROOT / "partial_validation.json"
PROGRESS_PATH = SOURCE_ROOT / "search_progress.json"


@dataclass(frozen=True)
class SelectedModel:
    """A selected real candidate for one state/side."""

    state_id: int
    side: str
    candidate_id: int
    search_round: int
    orders: tuple[int, ...]
    memory_definition: dict[int, int]
    max_delay: int
    coefficient_count: int
    ridge_lambda: float
    native_train_nmse_db: float
    native_b_nmse_db: float
    joint_score_db: float
    joint_feasible_native: bool
    selection_mode: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "side": self.side,
            "candidate_id": self.candidate_id,
            "search_round": self.search_round,
            "orders": json.dumps(list(self.orders), separators=(",", ":")),
            "memory_definition": json.dumps(
                self.memory_definition, sort_keys=True, separators=(",", ":")
            ),
            "max_delay": self.max_delay,
            "coefficient_count": self.coefficient_count,
            "lambda": self.ridge_lambda,
            "native_train_NMSE_dB": self.native_train_nmse_db,
            "native_B_NMSE_dB": self.native_b_nmse_db,
            "joint_score_dB": self.joint_score_db,
            "joint_feasible_native": self.joint_feasible_native,
            "selection_mode": self.selection_mode,
        }


def _parse_json(value: Any, name: str) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    raise ValueError(f"{name} must be a JSON string")


def load_candidate_pool() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Load and validate only the persisted Round 0--4 candidate records."""

    if not GRID_PATH.is_file() or not AEND_LOG_PATH.is_file() or not C2_LOG_PATH.is_file():
        raise FileNotFoundError("persisted Round 0--4 candidate pool is incomplete")
    grid = pd.read_csv(GRID_PATH)
    required_grid = {
        "candidate_id",
        "search_round",
        "orders",
        "memory_definition",
        "lambda",
        "max_delay",
        "coefficient_count",
    }
    missing = sorted(required_grid - set(grid.columns))
    if missing:
        raise ValueError(f"candidate grid missing fields: {missing}")
    grid["candidate_id"] = grid["candidate_id"].astype(np.int64)
    grid["search_round"] = grid["search_round"].astype(np.int64)
    if not np.array_equal(grid["candidate_id"].to_numpy(), np.arange(1, 1065)):
        raise ValueError("candidate grid must contain IDs 1...1064")
    if int(grid["search_round"].max()) != 4:
        raise ValueError("candidate grid persisted round maximum must be 4")
    logs: dict[str, pd.DataFrame] = {}
    for side, path in (("Aend", AEND_LOG_PATH), ("C2", C2_LOG_PATH)):
        frame = pd.read_csv(path)
        required = {
            "state_id",
            "candidate_id",
            "search_round",
            "native_train_NMSE_dB",
            "native_B_NMSE_dB",
            "valid",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{side} log missing fields: {missing}")
        frame = frame.loc[frame["valid"].astype(bool)].copy()
        frame = frame.loc[frame["search_round"].astype(int) <= 4].copy()
        frame["state_id"] = frame["state_id"].astype(np.int64)
        frame["candidate_id"] = frame["candidate_id"].astype(np.int64)
        frame["search_round"] = frame["search_round"].astype(np.int64)
        for column in ("native_train_NMSE_dB", "native_B_NMSE_dB"):
            frame[column] = frame[column].astype(float)
            if not np.all(np.isfinite(frame[column].to_numpy(dtype=float))):
                raise ValueError(f"{side} log has non-finite {column}")
        if frame["state_id"].nunique() != STATE_COUNT:
            raise ValueError(f"{side} log does not cover all 425 states")
        unknown = set(frame["candidate_id"]) - set(grid["candidate_id"])
        if unknown:
            raise ValueError(f"{side} log contains unknown candidate IDs: {sorted(unknown)[:5]}")
        logs[side] = frame
    partial: dict[str, Any] = {}
    if PARTIAL_VALIDATION_PATH.is_file():
        partial = json.loads(PARTIAL_VALIDATION_PATH.read_text(encoding="utf-8"))
    progress: dict[str, Any] = {}
    if PROGRESS_PATH.is_file():
        progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    provenance = {
        "persisted_search_round_max": int(grid["search_round"].max()),
        "persisted_candidate_count": int(grid.shape[0]),
        "candidate_row_count_by_side": {side: int(frame.shape[0]) for side, frame in logs.items()},
        "partial_validation_joint_feasible": {
            side: int(partial.get(side, {}).get("joint_native_feasible_state_count", -1))
            for side in ("Aend", "C2")
        },
        "round_5_used": False,
        "round_5_partial_status": progress.get("partial_round_status"),
    }
    if provenance["round_5_partial_status"] != "round_5_generated_but_not_persisted":
        raise ValueError("Round 5 provenance is not explicitly marked non-persisted")
    return grid, logs, provenance


def _candidate_from_row(row: Mapping[str, Any], side: str) -> SelectedModel:
    orders = tuple(int(value) for value in _parse_json(row["orders"], "orders"))
    memory_raw = _parse_json(row["memory_definition"], "memory_definition")
    memory = {int(key): int(value) for key, value in memory_raw.items()}
    train = float(row["native_train_NMSE_dB"])
    b_nmse = float(row["native_B_NMSE_dB"])
    feasible = train < DPD_SHAREABLE_THRESHOLD_DB and b_nmse < DPD_SHAREABLE_THRESHOLD_DB
    return SelectedModel(
        state_id=int(row["state_id"]),
        side=side,
        candidate_id=int(row["candidate_id"]),
        search_round=int(row["search_round"]),
        orders=orders,
        memory_definition=memory,
        max_delay=int(row["max_delay"]),
        coefficient_count=int(row["coefficient_count"]),
        ridge_lambda=float(row["lambda"]),
        native_train_nmse_db=train,
        native_b_nmse_db=b_nmse,
        joint_score_db=max(train, b_nmse),
        joint_feasible_native=feasible,
        selection_mode=("feasible_min_complexity" if feasible else "best_available_minimax"),
    )


def select_side_models(frame: pd.DataFrame, grid: pd.DataFrame, side: str) -> list[SelectedModel]:
    """Select one actual candidate per state using the frozen feasible/fallback rules."""

    merged = frame.merge(
        grid.loc[
            :,
            [
                "candidate_id",
                "orders",
                "memory_definition",
                "max_delay",
                "coefficient_count",
                "lambda",
            ],
        ],
        on="candidate_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_grid"),
    )
    selected: list[SelectedModel] = []
    for state_id, group in merged.groupby("state_id", sort=True):
        if group.shape[0] == 0:
            raise RuntimeError(f"state {state_id} has no {side} candidate")
        train = group["native_train_NMSE_dB"].to_numpy(dtype=float)
        b_nmse = group["native_B_NMSE_dB"].to_numpy(dtype=float)
        feasible_mask = (train < DPD_SHAREABLE_THRESHOLD_DB) & (b_nmse < DPD_SHAREABLE_THRESHOLD_DB)
        candidates = group.loc[feasible_mask].copy()
        if candidates.empty:
            candidates = group.copy()
            selection_mode = "best_available_minimax"
            candidates["joint_score_dB"] = candidates[
                ["native_train_NMSE_dB", "native_B_NMSE_dB"]
            ].max(axis=1)
            candidates = candidates.sort_values(
                [
                    "joint_score_dB",
                    "coefficient_count",
                    "max_delay",
                    "lambda",
                    "candidate_id",
                ],
                ascending=[True, True, True, False, True],
                kind="mergesort",
            )
        else:
            selection_mode = "feasible_min_complexity"
            candidates["joint_score_dB"] = candidates[
                ["native_train_NMSE_dB", "native_B_NMSE_dB"]
            ].max(axis=1)
            candidates = candidates.sort_values(
                [
                    "coefficient_count",
                    "max_delay",
                    "lambda",
                    "joint_score_dB",
                    "candidate_id",
                ],
                ascending=[True, True, False, True, True],
                kind="mergesort",
            )
        chosen = candidates.iloc[0].copy()
        chosen_model = _candidate_from_row(chosen, side)
        selected.append(
            SelectedModel(
                **{
                    **chosen_model.__dict__,
                    "selection_mode": selection_mode,
                    "joint_feasible_native": bool(feasible_mask.any()),
                }
            )
        )
    if len(selected) != STATE_COUNT:
        raise RuntimeError(f"{side} selection count must be 425")
    return selected


def _parse_selected(value: SelectedModel) -> dict[str, Any]:
    return value.as_dict()


def selected_frames(
    grid: pd.DataFrame, logs: Mapping[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    aend = pd.DataFrame(
        [_parse_selected(item) for item in select_side_models(logs["Aend"], grid, "Aend")]
    )
    c2 = pd.DataFrame(
        [_parse_selected(item) for item in select_side_models(logs["C2"], grid, "C2")]
    )
    return aend.sort_values("state_id").reset_index(drop=True), c2.sort_values(
        "state_id"
    ).reset_index(drop=True)


def _fit_selected_one(
    prepared: PreparedStatePairs,
    selected: Mapping[str, Any],
    side: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    orders = tuple(int(value) for value in json.loads(selected["orders"]))
    memory_raw = json.loads(selected["memory_definition"])
    memory = {int(key): int(value) for key, value in memory_raw.items()}
    candidate_delay = int(selected["max_delay"])
    canonical = prepared.a_end if side == "Aend" else prepared.c2
    train_name = "A" if side == "Aend" else "C"
    train = canonical[train_name]
    b_segment = canonical["B"]
    phi_train = build_mp_basis(train.input, orders, memory)
    phi_b = build_mp_basis(b_segment.input, orders, memory)
    y_train = np.asarray(train.output[candidate_delay:], dtype=np.complex128)
    y_b = np.asarray(b_segment.output[candidate_delay:], dtype=np.complex128)
    if phi_train.shape[0] != y_train.size or phi_b.shape[0] != y_b.size:
        raise RuntimeError("selected candidate native support mismatch")
    theta, diagnostics = fit_coefficients_ridge(phi_train, y_train, float(selected["lambda"]))
    train_nmse = calculate_nmse(y_train, phi_train @ theta)
    b_nmse = calculate_nmse(y_b, phi_b @ theta)
    if not np.isfinite(train_nmse) or not np.isfinite(b_nmse):
        raise RuntimeError("selected candidate refit produced non-finite metrics")
    diff_train = abs(float(train_nmse) - float(selected["native_train_NMSE_dB"]))
    diff_b = abs(float(b_nmse) - float(selected["native_B_NMSE_dB"]))
    if diff_train > 1e-8 or diff_b > 1e-8:
        raise RuntimeError(
            f"selected candidate regression failed state={selected['state_id']} side={side} "
            f"candidate={selected['candidate_id']} diffs={diff_train}/{diff_b}"
        )
    diagnostic = {
        **dict(selected),
        "refit_train_NMSE_dB": float(train_nmse),
        "refit_B_NMSE_dB": float(b_nmse),
        "train_abs_diff_dB": diff_train,
        "B_abs_diff_dB": diff_b,
        "rank": int(diagnostics.rank_phi),
        "condition_number_phi": float(diagnostics.condition_number_phi),
        "condition_number_augmented": float(diagnostics.condition_number_augmented),
        "n_train_samples": int(diagnostics.n_train_samples),
        "theta_l2_norm": float(diagnostics.theta_l2_norm),
        "finite": True,
    }
    return np.asarray(theta, dtype=np.complex128), diagnostic


def refit_selected_models(
    aend: pd.DataFrame,
    c2: pd.DataFrame,
    *,
    result_root: Path = RESULT_ROOT,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    """Rebuild all 850 selected candidates and verify native log metrics."""

    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    aend_by_state = {int(row["state_id"]): row for row in aend.to_dict("records")}
    c2_by_state = {int(row["state_id"]): row for row in c2.to_dict("records")}
    theta_a: dict[str, np.ndarray] = {}
    theta_c: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, Any]] = []
    for state_id in range(STATE_COUNT):
        prepared = prepare_state_pairs(state_id)
        for side, selected in (("Aend", aend_by_state[state_id]), ("C2", c2_by_state[state_id])):
            theta, diagnostic = _fit_selected_one(prepared, selected, side)
            key = f"state_{state_id:03d}"
            if side == "Aend":
                theta_a[key] = theta
            else:
                theta_c[key] = theta
            diagnostic["state_id"] = state_id
            diagnostic["side"] = side
            diagnostic["ilc_A_end"] = int(prepared.ilc_A_end)
            diagnostics.append(diagnostic)
        if (state_id + 1) % 25 == 0 or state_id == STATE_COUNT - 1:
            print(f"selected refit: processed {state_id + 1}/{STATE_COUNT} states", flush=True)
    if len(theta_a) != STATE_COUNT or len(theta_c) != STATE_COUNT:
        raise RuntimeError("selected coefficients must contain 425 states per side")
    np.savez_compressed(result_root / "selected_Aend_coefficients.npz", **theta_a)
    np.savez_compressed(result_root / "selected_C2_coefficients.npz", **theta_c)
    diagnostics_frame = (
        pd.DataFrame(diagnostics).sort_values(["side", "state_id"]).reset_index(drop=True)
    )
    diagnostics_frame.to_csv(
        result_root / "selected_model_native_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.17g",
    )
    descriptors = {
        "Aend": aend.to_dict("records"),
        "C2": c2.to_dict("records"),
    }
    (result_root / "selected_model_descriptors.json").write_text(
        json.dumps(_json_safe(descriptors), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return theta_a, theta_c, diagnostics_frame


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def main() -> None:
    grid, logs, provenance = load_candidate_pool()
    aend, c2 = selected_frames(grid, logs)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    aend.to_csv(
        RESULT_ROOT / "selected_Aend_models.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.17g",
    )
    c2.to_csv(
        RESULT_ROOT / "selected_C2_models.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.17g",
    )
    selected = (
        pd.concat([aend, c2], ignore_index=True)
        .sort_values(["side", "state_id"])
        .reset_index(drop=True)
    )
    selected.to_csv(
        RESULT_ROOT / "selected_models.csv", index=False, encoding="utf-8-sig", float_format="%.17g"
    )
    feasibility = pd.DataFrame(
        {
            "state_id": np.arange(STATE_COUNT, dtype=np.int64),
            "Aend_native_feasible": aend["joint_feasible_native"].to_numpy(dtype=bool),
            "C2_native_feasible": c2["joint_feasible_native"].to_numpy(dtype=bool),
        }
    )
    feasibility["both_native_feasible"] = (
        feasibility["Aend_native_feasible"] & feasibility["C2_native_feasible"]
    )
    feasibility.to_csv(
        RESULT_ROOT / "selected_model_feasibility.csv", index=False, encoding="utf-8-sig"
    )
    theta_a, theta_c, diagnostics = refit_selected_models(aend, c2)
    payload = {
        "experiment": "scenario_2_C2_to_Aend_statewise_best_round0_4_5B",
        "candidate_pool_frozen": True,
        "persisted_search_round_max": provenance["persisted_search_round_max"],
        "persisted_candidate_count": provenance["persisted_candidate_count"],
        "round_5_used": False,
        "additional_model_search_performed": False,
        "selected_Aend_count": int(aend.shape[0]),
        "selected_C2_count": int(c2.shape[0]),
        "native_feasible_Aend_count": int(aend["joint_feasible_native"].sum()),
        "native_feasible_C2_count": int(c2["joint_feasible_native"].sum()),
        "fallback_Aend_count": int((~aend["joint_feasible_native"]).sum()),
        "fallback_C2_count": int((~c2["joint_feasible_native"]).sum()),
        "one_actual_candidate_per_state_side": True,
        "independent_metric_minima_used_for_model_selection": False,
        "feasible_selection_rule": "minimum_complexity_among_joint_lt_minus40",
        "fallback_selection_rule": "minimum_worst_native_NMSE",
        "selection_pool_provenance": provenance,
        "all_850_refit": bool(diagnostics.shape[0] == 2 * STATE_COUNT),
        "max_native_refit_train_abs_diff_dB": float(diagnostics["train_abs_diff_dB"].max()),
        "max_native_refit_B_abs_diff_dB": float(diagnostics["B_abs_diff_dB"].max()),
        "output_files": {
            path.name: str(path) for path in sorted(RESULT_ROOT.iterdir()) if path.is_file()
        },
    }
    (RESULT_ROOT / "selection_validation.json").write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_Aend": 425,
                "selected_C2": 425,
                "refit_rows": int(diagnostics.shape[0]),
                "Aend_native_feasible": payload["native_feasible_Aend_count"],
                "C2_native_feasible": payload["native_feasible_C2_count"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


__all__ = [
    "RESULT_ROOT",
    "SelectedModel",
    "load_candidate_pool",
    "refit_selected_models",
    "selected_frames",
    "select_side_models",
]


if __name__ == "__main__":
    main()
