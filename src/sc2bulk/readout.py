"""The read-out: one code path
for every configuration, stage 08's transfer read-outs, stage 09's aligned read-outs (scaling="stacked";
its "None" is stage 08's primary file), 08b and the released evaluate().

Per side, features are standardized on that side's own rows (per stratum: mouse organs within
organ; uncapped). Ridge (primary) or LASSO (secondary) of the 0/1 group is fit on the single-cell
rows, with alpha chosen by grouped CV on stage 03's read-out folds (MSE; ridge's ties go to the
smaller alpha, GridSearchCV's first best over an ascending grid; LassoCV's to the larger, as it scans
a descending path), and applied to the bulk rows (when given) and to scored single-cell rows (when
given: Fig 2B's encoder validation sides) with the single-cell side's scaler. Only the final fit's
convergence is recorded. Nothing reads a bulk label.

With scaling="stacked" (stage 09, after an alignment), one scaler per stratum is
fit on the training and bulk rows stacked and applied to both sides instead: per feature, ComBat is an
affine map within each side, so per-side scaling afterwards would undo it.

readout_folds() deals read-out CV folds for training rows stage 03 makes none for (the out-of-fold
row's encoder validation sides): a copy of stage 03's, kept here because
sc2bulk.splits is hashed into stages 03, 06, 08 and 11.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Ridge
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from sc2bulk.folds import deal
from sc2bulk.config import readout as R
from sc2bulk.config import settings

ZERO_SD = 1e-6    # an SD at or below ZERO_SD x max(1, |mean|) counts as zero variance (SD set to 1): above
                  # float32 resolution, so a collapsed encoder's float-noise dimensions aren't blown up


def scaler(X: np.ndarray, strata=None) -> dict:
    """{stratum: (mean, scale)} of X's rows (one stratum, "all", when strata is None)."""
    strata = np.full(len(X), "all") if strata is None else np.asarray(strata)
    out = {}
    for s in dict.fromkeys(strata):
        rows = X[strata == s]
        mean, sd = rows.mean(axis=0), rows.std(axis=0, ddof=R.SCALER_DDOF)
        out[s] = (mean, np.where(sd <= ZERO_SD * np.maximum(1.0, np.abs(mean)), 1.0, sd))
    return out


def scale(X: np.ndarray, moments: dict, strata=None) -> np.ndarray:
    strata = np.full(len(X), "all") if strata is None else np.asarray(strata)
    out = np.empty_like(X, dtype=np.float64)
    for s, (mean, sd) in moments.items():
        rows = strata == s
        out[rows] = (X[rows] - mean) / sd
    assert set(dict.fromkeys(strata)) <= set(moments), "a stratum without moments"
    return out


def readout_folds(samples: pd.DataFrame, seed: int) -> pd.Series:
    """Read-out fold per sample (sample_id, holdout_group, group): pipeline/03_splits.py's readout_folds,
    copied (each label's holdout groups shuffled with the seed and dealt to the READOUT_FOLDS folds in
    turn, from a seeded start). Stage 03's seed for read-out seed r: seeds.derived("readout_folds", pair,
    r, *unit)."""
    labels = samples.groupby("holdout_group")["group"].agg(["nunique", "first"])
    assert (labels["nunique"] == 1).all(), "a holdout group has samples of both groups"
    fold = samples["holdout_group"].map(deal(labels["first"], settings.READOUT_FOLDS, seed))
    assert fold.nunique() == settings.READOUT_FOLDS, "a read-out fold is empty"
    return fold


def run(X_train, y, folds, X_bulk, readout="ridge", strata_train=None, strata_bulk=None,
        alphas=None, X_scored=None, strata_scored=None, scaling="per_side") -> dict:
    """One read-out. X_train: single-cell rows (training samples) x features; y: their 0/1 groups;
    folds: each row's read-out CV fold (stage 03); X_bulk: the scored bulk rows, or None (a
    within-single-cell fit); scaling: "per_side" or "stacked" (see the module docstring). Returns the
    bulk scores (score; probability = score clipped to [0, 1]; when X_bulk is given), the scored
    single-cell rows' scores (scored_score, when X_scored is given), the chosen alpha and whether it sits
    at an edge of the grid, the head (coef, intercept) and the scalers (per side: single_cell and bulk;
    stacked: stacked)."""
    X_train = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert set(np.unique(y)) == {0.0, 1.0}
    assert scaling in ("per_side", "stacked")
    if scaling == "stacked":   # single_cell then holds the stacked moments, applied to both sides
        assert X_bulk is not None and X_scored is None, "the stacked scaler needs bulk rows and scores no single-cell rows"
        assert (strata_train is None) == (strata_bulk is None), "strata on both sides or on neither"
        strata = None if strata_train is None else np.concatenate([strata_train, strata_bulk])
        single_cell = scaler(np.vstack([X_train, np.asarray(X_bulk, dtype=np.float64)]), strata)
    else:
        single_cell = scaler(X_train, strata_train)
    Z_train = scale(X_train, single_cell, strata_train)
    split = PredefinedSplit(np.asarray(folds))
    alphas = tuple(R.RIDGE_ALPHAS if readout == "ridge" else R.LASSO_ALPHAS) if alphas is None else tuple(alphas)
    assert list(alphas) == sorted(alphas), "the alpha grid must be ascending"
    if readout == "ridge":
        search = GridSearchCV(Ridge(fit_intercept=R.READOUT_FIT_INTERCEPT), {"alpha": list(alphas)}, cv=split,
                              scoring="neg_mean_squared_error", error_score="raise").fit(Z_train, y)
        model, alpha, extra = search.best_estimator_, search.best_params_["alpha"], {}
    else:
        model = LassoCV(alphas=list(alphas), cv=split, max_iter=R.LASSO_MAX_ITER, tol=R.LASSO_TOL,
                        fit_intercept=R.READOUT_FIT_INTERCEPT).fit(Z_train, y)   # refit at alpha_ on all rows
        alpha = float(model.alpha_)
        extra = {"converged": bool(model.n_iter_ < R.LASSO_MAX_ITER), "n_iter": int(model.n_iter_),
                 "n_nonzero": int((model.coef_ != 0).sum())}
    if X_scored is not None:
        extra["scored_score"] = model.predict(scale(np.asarray(X_scored, dtype=np.float64), single_cell, strata_scored))
    out = {"alpha": float(alpha), "alpha_at_boundary": bool(alpha in (min(alphas), max(alphas))), "coef": model.coef_,
           "intercept": float(model.intercept_), "scalers": {"single_cell": single_cell}, **extra}
    if scaling == "stacked":
        out["scalers"] = {"stacked": single_cell}
        out["score"] = model.predict(scale(np.asarray(X_bulk, dtype=np.float64), single_cell, strata_bulk))
        out["probability"] = np.clip(out["score"], 0.0, 1.0)
    elif X_bulk is not None:
        X_bulk = np.asarray(X_bulk, dtype=np.float64)
        out["scalers"]["bulk"] = scaler(X_bulk, strata_bulk)
        out["score"] = model.predict(scale(X_bulk, out["scalers"]["bulk"], strata_bulk))
        out["probability"] = np.clip(out["score"], 0.0, 1.0)
    return out
