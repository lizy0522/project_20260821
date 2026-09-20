"""Remove only non-scientific runtime caches after architecture validation."""

# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
LOG_ROOT = PROJECT_ROOT / "work_logs" / "core" / "final_architecture_consolidation_20260921"


def cleanup() -> dict[str, object]:
    targets: list[Path] = []
    scripts_root = PROJECT_ROOT / "scripts"
    targets.extend(path for path in scripts_root.rglob("__pycache__") if path.is_dir())
    targets.extend(path for path in scripts_root.rglob("*.pyc") if path.is_file())
    for name in (".pytest_cache", ".ruff_cache"):
        path = PROJECT_ROOT / name
        if path.exists():
            targets.append(path)
    # .DS_Store is non-scientific runtime metadata; data/raw is intentionally
    # excluded because its experiment-content manifest must remain untouched.
    targets.extend(
        path
        for tree in ("scripts", "results", "work_logs")
        for path in (PROJECT_ROOT / tree).rglob(".DS_Store")
        if path.is_file()
    )
    removed = []
    for path in sorted(set(targets), key=lambda item: len(item.parts), reverse=True):
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(str(path))
    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "removed_count": len(removed),
        "removed": removed,
        "data_raw_touched": False,
        "results_touched": False,
        "work_logs_touched": False,
        "scientific_cache_touched": False,
    }
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / "runtime_cache_cleanup.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(cleanup(), ensure_ascii=False, indent=2))
