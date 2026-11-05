"""scripts/hammond_solver_check.py: its input rebuild matches 08b's per_cell, and both checks pass on agreeing
inputs and stop on disagreeing ones."""
import sys

import numpy as np
import pandas as pd
import pytest

from sc2bulk import readout_secondary
from sc2bulk.config import readout as R
from sc2bulk.config import secondary as C
from sc2bulk.config import settings

sys.path.insert(0, str(settings.REPO / "scripts"))
import hammond_solver_check as H  # noqa: E402
from test_readout_secondary import cells_toy, per_cell_fixture  # noqa: E402

ALPHAS = [C.PER_CELL_CAP * a for a in R.RIDGE_ALPHAS]


def test_solver_check_passes_on_toys():
    for seed in range(3):
        X, y, folds, X_bulk = cells_toy(seed=seed)
        assert H.solver_check(X, y, folds, None, X_bulk, None, ALPHAS)
    X, y, folds, X_bulk = cells_toy()
    strata, strata_bulk = np.where(np.arange(len(y)) % 2, "kidney", "lung"), np.where(np.arange(len(X_bulk)) % 2, "kidney", "lung")
    assert H.solver_check(X, y, folds, strata, X_bulk, strata_bulk, ALPHAS)


def test_solver_check_stops_on_a_perturbed_curve(monkeypatch):
    cell_ridge = readout_secondary.cell_ridge

    def perturbed(*args):
        out = cell_ridge(*args)
        return {**out, "cv_mse": out["cv_mse"] * (1 + 1e-6)}

    monkeypatch.setattr(readout_secondary, "cell_ridge", perturbed)
    X, y, folds, X_bulk = cells_toy()
    with pytest.raises(AssertionError, match="CV-MSE curves differ"):
        H.solver_check(X, y, folds, None, X_bulk, None, ALPHAS)


def test_inputs_reproduce_per_cell_and_stored_check(monkeypatch):
    B = H.stage08b()
    monkeypatch.setattr(settings, "SMOKE", False)
    c, cells, X_bulk = per_cell_fixture(B, monkeypatch)
    tables = B.per_cell("tms", ("all", 0), "per_cell_classifier")
    predictions, heads = pd.concat(tables["predictions"]), pd.concat(tables["heads"])
    for r in range(2):
        x = H.per_cell_inputs(B, "tms", ("all", 0), r)
        out = H.sample_level_ridge(x)
        assert H.stored_check(predictions, heads, out, x["bulk_ids"], r)
    bad = predictions.assign(score=predictions["score"] + 1e-6)
    with pytest.raises(AssertionError, match="stored scores differ"):
        H.stored_check(bad, heads, out, x["bulk_ids"], 1)
    with pytest.raises(AssertionError, match="stored alpha"):
        H.stored_check(predictions, heads.assign(alpha=-1.0), out, x["bulk_ids"], 1)
    with pytest.raises(AssertionError, match="stored bulk rows differ from the scored bulk"):
        H.stored_check(predictions, heads, out, list(x["bulk_ids"])[1:], 1)
