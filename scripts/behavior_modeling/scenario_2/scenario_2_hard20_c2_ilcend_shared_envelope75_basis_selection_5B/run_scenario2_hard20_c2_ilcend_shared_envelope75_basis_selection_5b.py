"""Formal runner for the shared C2 plus ILC_END Envelope75 task."""

# ruff: noqa: E402,E501

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.scenario_2.scenario_2_hard20_c2_ilcend_shared_envelope75_basis_selection_5B.hard20_c2_ilcend_shared_envelope75_selection import (
    main,  # noqa: E402
)

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
