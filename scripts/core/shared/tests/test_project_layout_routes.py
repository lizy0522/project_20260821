"""Route-aware global layout validation on a disposable synthetic project."""

# ruff: noqa: E501

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.shared.module_registry import MODULES
from core.shared.project_layout import validate_project_layout


class RouteLayoutTests(unittest.TestCase):
    def test_scenario_route_layout_and_flat_task_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for tree in ("scripts", "results", "work_logs"):
                for module in MODULES:
                    base = root / tree / module
                    base.mkdir(parents=True)
                    if tree == "scripts":
                        (base / "shared").mkdir()
                        (base / "shared" / "__init__.py").touch()
                        (base / "__init__.py").touch()
                    if module == "retrieval_oriented_model_selection":
                        for route in ("self_hit_oriented", "dpd_shareability_oriented"):
                            (base / route).mkdir()
                            if tree == "scripts":
                                (base / route / "__init__.py").touch()
                            (base / route / "scenario_2").mkdir()
                            if tree == "scripts":
                                (base / route / "scenario_2" / "__init__.py").touch()
                    elif module not in {"core", "data_management"}:
                        (base / "scenario_2").mkdir()
                        if tree == "scripts":
                            (base / "scenario_2" / "__init__.py").touch()
            self.assertTrue(validate_project_layout(root)["pass"])
            (root / "scripts/retrieval_oriented_model_selection/self_hit_oriented/flat_task").mkdir()
            report = validate_project_layout(root)
            self.assertFalse(report["pass"])
            self.assertIn("flat_task",
                          {row["type"] for row in report["layout_violations"]})


if __name__ == "__main__":
    unittest.main()
