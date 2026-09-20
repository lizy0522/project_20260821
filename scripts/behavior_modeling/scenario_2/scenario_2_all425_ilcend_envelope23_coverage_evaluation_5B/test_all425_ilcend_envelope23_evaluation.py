"""Direct contracts for evaluation-only frozen Envelope23 coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.shared.basis_function_selection.all425_ilcend_envelope23_evaluation import (  # noqa: E402
    DMAX,
    PREVIOUS_HARD20_FINAL,
    RESULT_ROOT,
    STATE_COUNT,
    verify_frozen_ilcend_support,
)


def test_frozen_model_contract() -> None:
    terms, indices, digest = verify_frozen_ilcend_support()
    assert len(terms) == 75
    assert len(indices) == 23
    assert digest
    assert DMAX == 2


def test_all425_artifact_contract() -> None:
    metrics = pd.read_csv(RESULT_ROOT / "01_all425_ilcend_metrics.csv")
    failed = pd.read_csv(RESULT_ROOT / "02_failed_states.csv")
    with np.load(RESULT_ROOT / "04_all425_train_coefficients.npz", allow_pickle=False) as data:
        assert data["theta"].shape == (STATE_COUNT, 23)
        assert data["theta"].dtype == np.complex128
        assert data["state_ids"].tolist() == list(range(STATE_COUNT))
        assert data["K"].item() == 23
        assert data["lambda_value"].item() == 0.0
        assert data["dmax"].item() == 2
        assert data["coefficient_source"].item() == "Train only"
    checkpoint = json.loads((RESULT_ROOT / "09_checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["evaluation_only"] is True
    assert checkpoint["model_selection_performed"] is False
    assert checkpoint["model_frozen"] is True
    assert checkpoint["test_unlocked"] is True
    assert checkpoint["worker_count"] == 10
    assert metrics.shape[0] == STATE_COUNT
    assert metrics["State_ID"].tolist() == list(range(STATE_COUNT))
    assert metrics["Train_pass"].all()
    assert metrics["Test_pass"].all()
    assert metrics["Both_pass"].all()
    assert failed.empty
    previous = pd.read_csv(PREVIOUS_HARD20_FINAL)
    current = metrics.loc[metrics["is_Hard20"]]
    assert len(current) == 20
    assert set(previous["State_ID"]) == set(current["State_ID"])


def main() -> None:
    test_frozen_model_contract()
    test_all425_artifact_contract()
    print("All-425 ilc_end Envelope23 evaluation direct regression: PASS")


if __name__ == "__main__":
    main()
