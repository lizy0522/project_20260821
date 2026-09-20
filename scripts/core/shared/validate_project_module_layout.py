"""Validate the frozen ten-module alignment across project roots."""

from __future__ import annotations

import json
from pathlib import Path

from core.shared.project_layout import validate_project_layout

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
VALIDATION_ROOT = PROJECT_ROOT / "results" / "core" / "retrieval_model_selection_route_split"


def main() -> None:
    result = validate_project_layout(PROJECT_ROOT)
    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    (VALIDATION_ROOT / "14_project_layout_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("PASS" if result["pass"] else "FAIL")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
