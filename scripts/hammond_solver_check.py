"""The smoke Hammond solver check (due before the analysis-fixed tag). On the smoke run's stage 08b output for Hammond all_0 at
read-out seed 0, it rebuilds the per-cell classifier's inputs as pipeline/08b_secondary.per_cell builds them and checks:
  1. the closed-form solver (readout_secondary.cell_ridge) equals readout.run's ridge on the same cells with
     cell-level moments (the only moments readout.run takes): the fold-mean CV-MSE curve matches GridSearchCV's
     (rtol 1e-10); alpha is equal, or at a near-tie the two alphas' CV MSEs agree within 1e-10 relative; at an equal
     alpha, coef, intercept and bulk scores are allclose at 1e-8;
  2. 08b's stored per_cell_classifier file (made by 08b's current code) equals cell_ridge with the sample-level
     moments 08b uses: bulk scores (rtol 1e-10) and the chosen alpha.
Label-free, like 08b (single-cell groups and bulk expression). Reads the smoke outputs; writes nothing.

  SC2BULK_SMOKE=1 SC2BULK_OUTPUTS=<outputs root> python scripts/hammond_solver_check.py
"""
import importlib.util

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from sc2bulk import bags, determinism, paths, properties, readout, readout_secondary, training_inputs
from sc2bulk.config import readout as R
from sc2bulk.config import secondary as C
from sc2bulk.config import settings

PAIR, UNIT, SEED = "hammond", ("all", 0), 0


def stage08b():
    spec = importlib.util.spec_from_file_location("stage08b", settings.REPO / "pipeline" / "08b_secondary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def per_cell_inputs(B, pair, unit, r) -> dict:
    """The per-cell classifier's inputs at read-out seed r, built as 08b's per_cell builds them."""
    c = B.context(pair, unit)
    cells = training_inputs.unit_cells(pair, unit)
    test = bags.test_bags(cells.cell_id, cells.sample_id, c["train_ids"], C.PER_CELL_CAP, 1, pair, unit)
    row = pd.Series(np.arange(len(cells.cell_id)), index=cells.cell_id)
    take = np.concatenate([row[test[s][0]].to_numpy() for s in c["train_ids"]])
    position = np.repeat(np.arange(len(c["train_ids"])), [len(test[s][0]) for s in c["train_ids"]])
    X_cells = cells.X[take].toarray().astype(np.float64)
    cell_strata = None if c["strata_train"] is None else c["strata_train"][position]
    moments = readout.scaler(properties.mean_log_cells(X_cells, position, range(len(c["train_ids"]))), c["strata_train"])
    X_bulk = B.pseudobulk_bulk(pair, unit, c["bulk_ids"])
    return {"X_cells": X_cells, "y": c["y"][position], "folds": B.cv_folds(pair, unit, c["train"], c["train_ids"])[r][position],
            "cell_strata": cell_strata, "moments": moments, "X_bulk": X_bulk, "strata_bulk": c["strata_bulk"],
            "bulk_ids": c["bulk_ids"], "alphas": [C.PER_CELL_CAP * a for a in R.RIDGE_ALPHAS]}


def solver_check(X_cells, y, folds, cell_strata, X_bulk, strata_bulk, alphas) -> str:
    """Check 1 (asserts): cell_ridge with cell-level moments against GridSearchCV's curve and readout.run's ridge."""
    Z = readout.scale(X_cells, readout.scaler(X_cells, cell_strata), cell_strata)
    Z_bulk = readout.scale(X_bulk, readout.scaler(X_bulk, strata_bulk), strata_bulk)
    out = readout_secondary.cell_ridge(Z, y, folds, Z_bulk, alphas)
    search = GridSearchCV(Ridge(fit_intercept=R.READOUT_FIT_INTERCEPT), {"alpha": list(alphas)}, cv=PredefinedSplit(folds),
                          scoring="neg_mean_squared_error", error_score="raise").fit(Z, y)
    assert np.allclose(out["cv_mse"], -search.cv_results_["mean_test_score"], rtol=1e-10, atol=0), "CV-MSE curves differ"
    ref = readout.run(X_cells, y, folds, X_bulk, strata_train=cell_strata, strata_bulk=strata_bulk, alphas=alphas)
    if out["alpha"] != ref["alpha"]:   # a near-tie on the plateau may flip argmin between solvers
        curve = dict(zip(alphas, out["cv_mse"]))
        assert np.isclose(curve[out["alpha"]], curve[ref["alpha"]], rtol=1e-10, atol=0), \
            f"alpha {out['alpha']} vs readout.run's {ref['alpha']}, and not a near-tie"
        return f"near-tie: alpha {out['alpha']:g} vs readout.run's {ref['alpha']:g}, CV MSEs within 1e-10"
    assert out["alpha_at_boundary"] == ref["alpha_at_boundary"], "alpha_at_boundary differs"
    assert np.allclose(out["coef"], ref["coef"], rtol=1e-8, atol=1e-12), "coef differs"
    assert np.isclose(out["intercept"], ref["intercept"], rtol=1e-8), "intercept differs"
    assert np.allclose(out["score"], ref["score"], rtol=1e-8, atol=1e-12), "bulk scores differ"
    return (f"equal alpha {out['alpha']:g}; max |coef diff| {np.abs(out['coef'] - ref['coef']).max():.2e}, "
            f"max |score diff| {np.abs(out['score'] - ref['score']).max():.2e}")


def sample_level_ridge(x) -> dict:
    """cell_ridge on per_cell_inputs' output with the sample-level moments 08b uses (check 2's recompute)."""
    Z_cells = readout.scale(x["X_cells"], x["moments"], x["cell_strata"])
    Z_bulk = readout.scale(x["X_bulk"], readout.scaler(x["X_bulk"], x["strata_bulk"]), x["strata_bulk"])
    return readout_secondary.cell_ridge(Z_cells, x["y"], x["folds"], Z_bulk, x["alphas"])


def stored_check(predictions, heads, out, bulk_ids, r) -> str:
    """Check 2 (asserts): 08b's stored per_cell_classifier rows at read-out seed r against cell_ridge's output."""
    stored = predictions[(predictions["readout_seed"] == r) & (predictions["side"] == "bulk")].set_index("sample_id")
    assert stored.index.is_unique and set(stored.index) == set(bulk_ids), "stored bulk rows differ from the scored bulk"
    score = stored.loc[list(bulk_ids), "score"].to_numpy(dtype=float)
    assert np.allclose(score, out["score"], rtol=1e-10, atol=1e-12), "stored scores differ from cell_ridge's"
    alpha = set(heads.loc[heads["readout_seed"] == r, "alpha"])
    assert alpha == {out["alpha"]}, f"stored alpha {sorted(alpha)} vs cell_ridge's {out['alpha']}"
    return f"{len(score)} bulk scores, max |diff| {np.abs(score - out['score']).max():.2e}; alpha {out['alpha']:g}"


def main():
    assert settings.SMOKE, "the check runs on the smoke outputs (SC2BULK_SMOKE=1)"
    determinism.configure()
    B = stage08b()
    print(f"repo: {settings.REPO}; outputs: {settings.OUTPUTS}; {PAIR} all_0, read-out seed {SEED}", flush=True)
    x = per_cell_inputs(B, PAIR, UNIT, SEED)
    print(f"{len(x['y'])} cells x {x['X_cells'].shape[1]} genes, {len(x['bulk_ids'])} bulk samples", flush=True)
    print("1. cell_ridge = readout.run (cell-level moments):",
          solver_check(x["X_cells"], x["y"], x["folds"], x["cell_strata"], x["X_bulk"], x["strata_bulk"], x["alphas"]), flush=True)
    directory = paths.output("08b_secondary", PAIR, "reference_rows", UNIT)
    predictions = pd.read_parquet(B.checked(directory / "per_cell_classifier_predictions.parquet", "08b_secondary"))
    heads = pd.read_parquet(B.checked(directory / "per_cell_classifier_heads.parquet", "08b_secondary"))
    out = sample_level_ridge(x)
    print("2. stored per_cell_classifier = cell_ridge (sample-level moments):",
          stored_check(predictions, heads, out, x["bulk_ids"], SEED), flush=True)
    print("PASS: the smoke Hammond solver check", flush=True)


if __name__ == "__main__":
    main()
