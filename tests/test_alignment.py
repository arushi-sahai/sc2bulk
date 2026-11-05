"""The read-out's stacked scaler, readout.run(..., scaling="stacked"). Stage 09's own tests are in test_alignment_stage09.py."""
import numpy as np
import pytest
from sklearn.linear_model import Lasso, Ridge

from sc2bulk import readout

ALPHAS = (0.1, 1.0, 10.0, 100.0)


def data(seed=0, n=40, p=6, shift=1.0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, p)) + shift * y[:, None] * (np.arange(p) < 2)
    folds = np.arange(n) % 5
    bulk = rng.normal(3.0, 2.0, size=(15, p))
    return X, y, folds, bulk


def test_stacked_moments_are_one_set_over_both_sides():
    X, y, folds, bulk = data()
    out = readout.run(X, y, folds, bulk, alphas=ALPHAS, scaling="stacked")
    assert set(out["scalers"]) == {"stacked"}
    (mean, sd), = out["scalers"]["stacked"].values()
    stacked = np.vstack([X, bulk])
    assert np.allclose(mean, stacked.mean(axis=0)) and np.allclose(sd, stacked.std(axis=0))   # ddof 0, as per side
    # the fit and the bulk scores use those moments on both sides
    Z, Z_bulk = (stacked - mean) / sd, (bulk - mean) / sd
    ridge = Ridge(alpha=out["alpha"]).fit(Z[:len(X)], y)
    assert np.allclose(out["coef"], ridge.coef_) and np.allclose(out["score"], ridge.predict(Z_bulk))
    assert np.allclose(out["probability"], np.clip(out["score"], 0, 1))


def test_stacked_per_stratum():
    X, y, folds, bulk = data(1)
    strata_train, strata_bulk = np.array(["a", "b"] * 20), np.array(["a"] * 7 + ["b"] * 8)
    out = readout.run(X, y, folds, bulk, alphas=ALPHAS, strata_train=strata_train, strata_bulk=strata_bulk, scaling="stacked")
    moments = out["scalers"]["stacked"]
    assert set(moments) == {"a", "b"}
    for s in ("a", "b"):
        rows = np.vstack([X[strata_train == s], bulk[strata_bulk == s]])
        assert np.allclose(moments[s][0], rows.mean(axis=0)) and np.allclose(moments[s][1], rows.std(axis=0))
    Z = readout.scale(X, moments, strata_train)
    Z_bulk = readout.scale(bulk, moments, strata_bulk)
    stacked = np.vstack([Z, Z_bulk])
    strata = np.concatenate([strata_train, strata_bulk])
    assert np.allclose(stacked[strata == "b"].mean(axis=0), 0) and np.allclose(stacked[strata == "b"].std(axis=0), 1)
    assert np.allclose(out["score"], Z_bulk @ out["coef"] + out["intercept"])


def test_stacked_lasso_matches_a_hand_fit():
    X, y, folds, bulk = data(2)
    out = readout.run(X, y, folds, bulk, readout="lasso", alphas=(1e-3, 1e-2, 1e-1), scaling="stacked")
    (mean, sd), = out["scalers"]["stacked"].values()
    stacked = (np.vstack([X, bulk]) - mean) / sd
    lasso = Lasso(alpha=out["alpha"], max_iter=200_000, tol=1e-4).fit(stacked[:len(X)], y)
    assert out["converged"] and np.allclose(out["coef"], lasso.coef_, atol=1e-6)
    assert np.allclose(out["score"], lasso.predict(stacked[len(X):]), atol=1e-6)


def test_stacked_is_uncapped():
    X, y, folds, _ = data(3)
    bulk = np.random.default_rng(3).normal(size=(200, X.shape[1]))
    bulk[0, 0] = 1e4
    out = readout.run(X, y, folds, bulk, alphas=ALPHAS, scaling="stacked")
    Z_bulk = readout.scale(bulk, out["scalers"]["stacked"])
    assert np.abs(Z_bulk).max() > 10
    assert np.allclose(out["score"], Z_bulk @ out["coef"] + out["intercept"])


def test_per_side_is_the_default_and_differs_from_stacked():
    X, y, folds, bulk = data(4)
    default, per_side = readout.run(X, y, folds, bulk, alphas=ALPHAS), readout.run(X, y, folds, bulk, alphas=ALPHAS, scaling="per_side")
    assert np.array_equal(default["score"], per_side["score"]) and set(default["scalers"]) == {"single_cell", "bulk"}
    stacked = readout.run(X, y, folds, bulk, alphas=ALPHAS, scaling="stacked")
    assert not np.allclose(stacked["score"], per_side["score"])   # bulk sits at mean 3: per-side scaling centers it away


def test_stacked_asserts():
    X, y, folds, bulk = data(5)
    with pytest.raises(AssertionError):
        readout.run(X, y, folds, None, scaling="stacked")                                    # needs bulk rows
    with pytest.raises(AssertionError):
        readout.run(X, y, folds, bulk, X_scored=X[:3], scaling="stacked")                    # scores no single-cell rows
    with pytest.raises(AssertionError):
        readout.run(X, y, folds, bulk, strata_train=np.array(["a", "b"] * 20), scaling="stacked")   # strata on one side only
    with pytest.raises(AssertionError):
        readout.run(X, y, folds, bulk, scaling="pooled")
