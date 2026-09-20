"""Run the frozen Round 0--4 statewise-best model selection and refit."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.shared.select_statewise_best_round0_4_models import RESULT_ROOT  # noqa: E402
from behavior_modeling.shared.select_statewise_best_round0_4_models import (  # noqa: E402
    main as select_and_refit_main,
)

TASK_NAME = "scenario_2_select_statewise_best_round0_4_models"
RETRIEVAL_MODULE_NAME = "behavior_fingerprint_retrieval"
RETRIEVAL_TASK_NAME = "scenario_2_C2_to_Aend_retrieval"
SOURCE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /TASK_NAME
    / "scenario_2_statewise_adaptive_Aend_C2_5B"
)
BASELINE_ROOT = (
    PROJECT_ROOT
    / "results"
    / RETRIEVAL_MODULE_NAME
    / RETRIEVAL_TASK_NAME
    / "scenario_2_C2_to_Aend"
)
MODEL_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"  # noqa: E501
RETRIEVAL_LOG = (
    PROJECT_ROOT
    / "work_logs"
    / RETRIEVAL_MODULE_NAME
    / RETRIEVAL_TASK_NAME
    / "execution_log.txt"
)
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"

PROTECTED_RESULT_DIRS = {
    "scenario_2_5B_baseline": BASELINE_ROOT,
    "scenario_2_1B": PROJECT_ROOT
    / "results"
    / RETRIEVAL_MODULE_NAME
    / "scenario_2_C2_to_Aend_1b"
    / "scenario_2_C2_to_Aend_1B",
    "scenario_2_0p5B": PROJECT_ROOT
    / "results"
    / RETRIEVAL_MODULE_NAME
    / "scenario_2_C2_to_Aend_0p5b"
    / "scenario_2_C2_to_Aend_0p5B",
    "scenario_2_equal_ABC": PROJECT_ROOT
    / "results"
    / RETRIEVAL_MODULE_NAME
    / "scenario_2_C2_to_Aend_equal_ABC",
    "scenario_2_unified_model_capacity": PROJECT_ROOT
    / "results"
    / RETRIEVAL_MODULE_NAME
    / "scenario_2_C2_to_Aend_retrieval"
    / "scenario_2_C2_to_Aend_unified_model_capacity",
    "scenario_2_retrieval_oriented": PROJECT_ROOT
    / "results"
    / "retrieval_oriented_model_selection"
    / "scenario_2_C2_to_Aend_retrieval_oriented_model_scan",
    "scenario_2_all_ilc": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_all_ilc_analysis"
    / "scenario_2",
    "low_bandwidth_operator_bank": PROJECT_ROOT
    / "results"
    / "low_bandwidth_behavior_analysis"
    / "scenario_2" /"low_bandwidth_observation"
    / "sample_rate_operator_bank",
}
PROTECTED_SCRIPT_DIRS = {
    "behavior_modeling": PROJECT_ROOT / "scripts" / "behavior_modeling",
    "signal_segmentation": PROJECT_ROOT / "scripts" / "signal_segmentation",
    "retrieval": PROJECT_ROOT / "scripts" / "retrieval_oriented_model_selection",
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


def _file_snapshot(path: Path) -> dict[str, str]:
    if not path.is_dir():
        return {}
    return {
        file.relative_to(path).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.rglob("*"), key=lambda item: str(item).lower())
        if file.is_file() and "__pycache__" not in file.parts and file.suffix.lower() != ".pyc"
    }


def _raw_manifest() -> dict[str, Any]:
    root = PROJECT_ROOT / "data" / "raw"
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
        "script_files": {
            name: _file_snapshot(path) for name, path in PROTECTED_SCRIPT_DIRS.items()
        },
    }


def _verify(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    raw = {
        "sha256_unchanged": before["data/raw"]["sha256"] == after["data/raw"]["sha256"],
        "file_count_unchanged": before["data/raw"]["file_count"] == after["data/raw"]["file_count"],
        "bytes_unchanged": before["data/raw"]["bytes"] == after["data/raw"]["bytes"],
    }
    results = {
        name: {
            "sha256_unchanged": item["sha256"] == after["result_dirs"][name]["sha256"],
            "before": item["sha256"],
            "after": after["result_dirs"][name]["sha256"],
        }
        for name, item in before["result_dirs"].items()
    }
    scripts = {
        name: {
            relative: after["script_files"][name].get(relative) == digest
            for relative, digest in files.items()
        }
        for name, files in before["script_files"].items()
    }
    return {
        "data/raw": raw,
        "result_dirs": results,
        "script_files": scripts,
        "all_protected_unchanged": bool(
            all(raw.values())
            and all(item["sha256_unchanged"] for item in results.values())
            and all(all(values.values()) for values in scripts.values())
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _append(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{datetime.now(UTC).isoformat()}] {title}\n{body.rstrip()}\n")


def main() -> None:
    if not (SOURCE_ROOT / "partial_validation.json").is_file():
        raise FileNotFoundError("partial_validation.json is required")
    before = _snapshot()
    select_and_refit_main()
    validation_path = RESULT_ROOT / "selection_validation.json"
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        payload.get("persisted_search_round_max") != 4
        or payload.get("persisted_candidate_count") != 1064
    ):
        raise RuntimeError("selected model pool is not the frozen Round 0--4 / 1064 pool")
    if (
        payload.get("round_5_used") is not False
        or payload.get("additional_model_search_performed") is not False
    ):
        raise RuntimeError("Round 5 or additional model search was used")
    if payload.get("selected_Aend_count") != 425 or payload.get("selected_C2_count") != 425:
        raise RuntimeError("selected model count is not 425 per side")
    after = _snapshot()
    protection = _verify(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw or protected old results/scripts changed")
    payload["protection_before"] = before
    payload["protection_after"] = after
    payload["protection_verification"] = protection
    payload["output_files"] = {
        path.name: str(path) for path in sorted(RESULT_ROOT.iterdir()) if path.is_file()
    }
    validation_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    body = "\n".join(
        [
            "本次仅冻结并派生选择 Round 0--4 已落盘候选，不继续模型搜索。",
            f"selected Aend/C2={payload['selected_Aend_count']}/{payload['selected_C2_count']}；",
            "native feasible="
            f"{payload['native_feasible_Aend_count']}/{payload['native_feasible_C2_count']}；",
            f"850个selected模型重新拟合并与候选日志核对，最大误差="
            f"{payload['max_native_refit_train_abs_diff_dB']}"
            f"/{payload['max_native_refit_B_abs_diff_dB']} dB。",
            f"保护复核={protection['all_protected_unchanged']}；结果目录={RESULT_ROOT}。",
        ]
    )
    _append(MODEL_LOG, "Scenario 2 statewise-best Round 0--4 model selection/refit", body)
    _append(RETRIEVAL_LOG, "Scenario 2 statewise-best Round 0--4 model selection dependency", body)
    _append(HANDOFF_LOG, "Scenario 2 statewise-best Round 0--4 模型选择", body)
    print(
        json.dumps(
            {
                "selected_Aend": 425,
                "selected_C2": 425,
                "native_feasible_Aend": payload["native_feasible_Aend_count"],
                "native_feasible_C2": payload["native_feasible_C2_count"],
                "protection": protection["all_protected_unchanged"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
