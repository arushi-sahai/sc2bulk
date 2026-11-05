"""Stage 08b's numerics and rules."""
import importlib.util

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sc2bulk import preprocess, readout, readout_secondary
from sc2bulk.config import readout as R
from sc2bulk.config import secondary as C
from sc2bulk.config import settings


def data(seed=0, n=40, p=6, shift=1.0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(size=(n, p)) + shift * y[:, None] * np.linspace(1, 0, p)
    X_bulk = rng.normal(size=(30, p)) * 2 + 5
    return X, y, np.arange(n) % 5, X_bulk


def stage08b():
    spec = importlib.util.spec_from_file_location("stage08b", settings.REPO / "pipeline" / "08b_secondary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_head_application_equals_the_primary_fit():
    X, y, folds, X_bulk = data()
    strata = np.where(np.arange(30) % 2, "kidney", "lung")
    strata_train = np.where(np.arange(40) % 2, "kidney", "lung")
    for name in ("ridge", "lasso"):
        out = readout.run(X, y, folds, X_bulk, readout=name, strata_train=strata_train, strata_bulk=strata)
        score = readout_secondary.apply_head(X_bulk, out["coef"], out["intercept"], strata)
        assert np.allclose(score, out["score"], rtol=1e-12, atol=1e-12)   # the head sees single-cell rows only


def test_head_application_family_reproduces_primary_predictions(monkeypatch):
    B = stage08b()
    X, y, folds, X_bulk = data()
    bulk = pd.DataFrame({"sample_id": [f"b{i}" for i in range(30)], "scoring_unit_id": [f"b{i}" for i in range(30)]})
    out = readout.run(X, y, folds, X_bulk)
    heads = pd.concat(B.R8["head_rows"]({**B.R8["key_columns"]("tms", ("all", 0), "geneformer_cells"), "encoder_seed": -1,
                                             "readout": "ridge", "readout_seed": 0}, out, [str(i) for i in range(6)]))
    heads = heads[heads["side"] == "single_cell"]
    monkeypatch.setattr(B, "context", lambda pair, unit: {"bulk": bulk, "bulk_ids": list(bulk["sample_id"]), "strata_bulk": None})
    monkeypatch.setattr(B, "primary_heads", lambda pair, unit, key: heads)
    table = B.by_heads("tms", ("all", 0), "geneformer_cells", lambda e: X_bulk, "thinned_pseudocells")["predictions"][0]
    assert np.allclose(table["score"], out["score"], rtol=1e-12) and (table["variant"] == "thinned_pseudocells").all()


def test_halves_and_outside_moments():
    units = np.repeat([f"d{i}" for i in range(12)], 2)              # two samples per donor
    strata = np.where(np.arange(24) < 12, "kidney", "lung")
    half = readout_secondary.halves(units, strata, seed=7)
    assert half == readout_secondary.halves(units[::-1], strata[::-1], seed=7)   # deterministic, order-free
    for s in ("kidney", "lung"):
        members = set(units[strata == s])
        assert {half[u] for u in members} == {0, 1}                  # each half non-empty in every stratum
    X = np.random.default_rng(1).normal(size=(24, 3)) + 3
    h = np.array([half[u] for u in units])
    scaled = readout_secondary.outside_moments(X, h, strata)
    for s in ("kidney", "lung"):
        rows, other = (h == 0) & (strata == s), (h == 1) & (strata == s)
        mean, sd = X[other].mean(axis=0), X[other].std(axis=0)
        assert np.allclose(scaled[rows], (X[rows] - mean) / sd)      # its moments exclude its own half


def test_alternative_readouts_match_direct_fits():
    X, y, folds, X_bulk = data(n=60, p=8)
    Z, Zb = readout.scale(X, readout.scaler(X)), readout.scale(X_bulk, readout.scaler(X_bulk))
    split = PredefinedSplit(folds)
    largest = readout_secondary.smallest_training_fold(folds)
    direct = {
        "pls": GridSearchCV(PLSRegression(scale=False), {"n_components": [c for c in C.PLS_COMPONENTS if c < largest and c <= 8]}, cv=split,
                            scoring="neg_mean_squared_error").fit(Z, y),
        "knn": GridSearchCV(KNeighborsRegressor(), {"n_neighbors": [k for k in C.KNN_NEIGHBOURS if k <= largest]}, cv=split,
                            scoring="neg_mean_squared_error").fit(Z, y),
        "mlp": GridSearchCV(MLPRegressor(hidden_layer_sizes=C.MLP_HIDDEN, max_iter=C.MLP_MAX_ITER, random_state=3), {"alpha": list(C.MLP_ALPHAS)},
                            cv=split, scoring="neg_mean_squared_error").fit(Z, y)}
    for name, search in direct.items():
        out = readout_secondary.alternative(X, y, folds, X_bulk, name, random_state=3)
        assert np.allclose(out["score"], np.ravel(search.best_estimator_.predict(Zb)))
    out = readout_secondary.alternative(X, y, folds, X_bulk, "elasticnet")
    net = ElasticNetCV(l1_ratio=list(C.ELASTICNET_L1_RATIOS), alphas=list(R.LASSO_ALPHAS), cv=split, max_iter=R.LASSO_MAX_ITER, tol=R.LASSO_TOL).fit(Z, y)
    assert np.allclose(out["score"], net.predict(Zb)) and out["hyper"]["l1_ratio"] == net.l1_ratio_
    again = readout_secondary.alternative(X, y, folds, X_bulk, "mlp", random_state=3)
    assert np.array_equal(again["score"], readout_secondary.alternative(X, y, folds, X_bulk, "mlp", random_state=3)["score"])


def test_knn_ties_go_to_the_smoother_model():
    assert list(C.KNN_NEIGHBOURS) == sorted(C.KNN_NEIGHBOURS, reverse=True)   # GridSearchCV takes the first best
    assert list(C.PLS_COMPONENTS) == sorted(C.PLS_COMPONENTS) and list(C.MLP_ALPHAS) == sorted(C.MLP_ALPHAS, reverse=True)


def test_ridge_continuous_matches_readout_run_on_a_binary_target():
    X, y, folds, X_bulk = data()
    out, ref = readout_secondary.ridge_continuous(X, y, folds, X_bulk), readout.run(X, y, folds, X_bulk)
    assert out["alpha"] == ref["alpha"] and np.allclose(out["score"], ref["score"])


def test_tmm():
    rng = np.random.default_rng(0)
    base = rng.poisson(rng.gamma(2, 50, size=500), size=(1, 500)).astype(float) + 1
    counts = np.vstack([base[0], 3 * base[0], rng.poisson(base[0] * 2)])
    f = readout_secondary.tmm_factors(counts)
    assert np.isclose(np.exp(np.mean(np.log(f))), 1.0)                   # geometric mean 1
    assert np.isclose(f[0], f[1])                                         # proportional libraries: the same factor
    assert readout_secondary.tmm_pair(base[0], base[0], base[0].sum(), base[0].sum()) == 1.0
    # composition, by hand: 10% of the genes 4x up in one sample take 1.3x its library, so its unchanged genes
    # look 1/1.3 down and its factor is 1/1.3 of the others'
    shifted = base[0].copy()
    shifted[:50] *= 4
    shifted *= base[0].sum() / (base[0].sum() + 3 * base[0][:50].sum()) * 1.3   # its library: 1.3 x the unchanged genes' share
    g = readout_secondary.tmm_factors(np.vstack([base[0], base[0], shifted]))
    share = base[0].sum() / (base[0].sum() + 3 * base[0][:50].sum())
    assert np.isclose(g[0], g[1]) and np.isclose(g[2] / g[0], share, rtol=0.02)
    # log-CPM: edgeR's prior scaling, by hand for the given factors
    library = counts.sum(axis=1) * f
    prior = 2 * library / library.mean()
    expected = np.log2((counts + prior[:, None]) / (library + 2 * prior)[:, None] * 1e6)
    assert np.allclose(readout_secondary.tmm_log_cpm(counts), expected)


def test_vst_monotone_finite_deterministic():
    rng = np.random.default_rng(1)
    mu = rng.gamma(1.0, 100, size=300)
    counts = rng.negative_binomial(5, 5 / (5 + mu), size=(20, 300)).astype(float)
    out = readout_secondary.vst(counts)
    assert np.isfinite(out).all() and np.array_equal(out, readout_secondary.vst(counts))
    q = counts / readout_secondary.size_factors(counts)[:, None]
    for g in range(0, 300, 37):
        order = np.argsort(q[:, g])
        assert (np.diff(out[order, g]) >= -1e-12).all()                 # monotone per gene
    big = out[q > 1e3] - np.log2(q[q > 1e3])
    assert big.size and np.ptp(big) < 0.1                                # log2-like at large counts


def test_bag_pseudobulk_is_the_mean_of_per_bag_profiles():
    rng = np.random.default_rng(2)
    counts = sp.csr_matrix(rng.poisson(1.0, size=(30, 12)) + 0.0)
    bags = [np.arange(0, 10), np.arange(5, 15), np.arange(20, 30)]
    columns = np.array([1, 3, 5, 7])
    expected = np.mean([np.log1p(np.asarray(counts[b].sum(axis=0)).ravel() / counts[b].sum() * settings.TARGET_SUM)[columns] for b in bags], axis=0)
    assert np.allclose(readout_secondary.bag_pseudobulk(counts, bags, columns), expected, rtol=1e-6)
    whole = readout_secondary.bag_pseudobulk(counts, [np.arange(30)], columns)
    assert np.allclose(whole, preprocess.log_normalize(preprocess.summed(counts)[None, :])[:, columns].toarray().ravel())


def test_families_never_request_attribution_seeds(monkeypatch):
    B = stage08b()
    monkeypatch.setattr(settings, "SMOKE", False)
    for key in B.TRAINED:   # rows A, B, J: seeds 10-19 outside human blood, aging never enter 08b
        seeds = B.headline_seeds("tms", key)
        assert seeds == list(range(B.N_ENCODER)) and max(seeds) < 10


def test_layout_and_kinds():
    B = stage08b()
    names = [p.name for p in B.outputs("tms", ("all", 0), "alternative_readouts", "pseudobulk/mlp")]
    assert names == ["pseudobulk__mlp_predictions.parquet"]                                          # no head for MLP
    assert B.kinds(("all", 0), "alternative_readouts", "pseudobulk/elasticnet") == ("predictions", "heads")
    assert B.kinds(("crossfit", 1), "one_type", "pseudobulk_within") == ("predictions",)              # held-out rows: stored heads
    assert B.kinds(("all", 0), "one_type", "pseudobulk_within") == ("predictions", "heads")
    for family in ("thinned", "observed_library", "outside_moments", "single_cell_moments", "native_head"):
        assert B.kinds(("all", 0), family, "x") == ("predictions",)
    path, = B.outputs("tms", ("all", 0), "single_cell_moments", "pseudobulk")                          # its own family directory
    assert path.relative_to(settings.OUTPUTS).parts == ("08b_secondary", "tms", "all_0", "single_cell_moments", "pseudobulk_predictions.parquet")


def stored_heads(B, X, y, folds, X_bulk, strata_train, strata_bulk):
    """Stage 08's stored heads of a toy fit (ridge and LASSO, read-out seeds 0-1; single-cell side), and the fits."""
    fits, heads = {}, []
    for name in ("ridge", "lasso"):
        for r in (0, 1):
            fits[name, r] = readout.run(X, y, (folds + r) % 5, X_bulk, readout=name, strata_train=strata_train, strata_bulk=strata_bulk)
            heads += B.R8["head_rows"]({**B.R8["key_columns"]("tms", ("all", 0), "geneformer_cells"), "encoder_seed": -1,
                                        "readout": name, "readout_seed": r}, fits[name, r], [str(i) for i in range(X.shape[1])])
    heads = pd.concat(heads)
    return heads[heads["side"] == "single_cell"], fits


def test_single_cell_moments_family(monkeypatch):
    """The stored heads on bulk standardized with the single-cell training moments, per organ; variant
    "primary". Per stratum, the ranking is that of the single-cell linear model on the unscaled bulk features."""
    B = stage08b()
    X, y, folds, X_bulk = data()
    strata_train = np.where(np.arange(40) % 2, "kidney", "lung")
    strata_bulk = np.where(np.arange(30) % 3, "kidney", "lung")
    heads, fits = stored_heads(B, X, y, folds, X_bulk, strata_train, strata_bulk)
    bulk = pd.DataFrame({"sample_id": [f"b{i}" for i in range(30)], "scoring_unit_id": [f"b{i}" for i in range(30)]})
    monkeypatch.setattr(B, "context", lambda pair, unit: {"bulk": bulk, "bulk_ids": list(bulk["sample_id"]), "strata_bulk": strata_bulk,
                                                          "train_ids": [], "strata_train": strata_train})
    monkeypatch.setattr(B, "primary_features", lambda pair, unit, key, a, b: [(-1, X, X_bulk)])
    monkeypatch.setattr(B, "primary_heads", lambda pair, unit, key: heads)
    table = pd.concat(B.single_cell_moments("tms", ("all", 0), "geneformer_cells")["predictions"])
    assert (table["variant"] == "primary").all() and (table["side"] == "bulk").all() and len(table) == 4 * 30
    for (name, r), out in fits.items():
        score = table[(table["readout"] == name) & (table["readout_seed"] == r)]["score"].to_numpy()
        moments = readout.scaler(X, strata_train)                                                     # the single-cell training moments
        expected = np.empty(30)
        for s, (mean, sd) in moments.items():
            rows = strata_bulk == s
            expected[rows] = (X_bulk[rows] - mean) / sd @ out["coef"] + out["intercept"]
            raw = X_bulk[rows] @ (out["coef"] / sd)                                                    # the single-cell model, unscaled bulk
            assert np.array_equal(np.argsort(score[rows], kind="stable"), np.argsort(raw, kind="stable"))
        assert np.allclose(score, expected, rtol=1e-12, atol=1e-12)
        assert not np.allclose(score, out["score"])                                                    # not the per-side bulk scaling


def test_outside_moments_family_unchanged(monkeypatch):
    """by_heads' scale_of now takes the head too; the outside-moments family's scores are unchanged."""
    B = stage08b()
    X, y, folds, X_bulk = data()
    heads, fits = stored_heads(B, X, y, folds, X_bulk, None, None)
    bulk = pd.DataFrame({"sample_id": [f"b{i}" for i in range(30)], "scoring_unit_id": [f"b{i}" for i in range(30)]})
    c = {"bulk": bulk, "bulk_ids": list(bulk["sample_id"]), "strata_bulk": None, "train_ids": [], "strata_train": None}
    monkeypatch.setattr(B, "context", lambda pair, unit: c)
    monkeypatch.setattr(B, "primary_features", lambda pair, unit, key, a, b: [(-1, X, X_bulk)])
    monkeypatch.setattr(B, "primary_heads", lambda pair, unit, key: heads)
    monkeypatch.setattr(B, "N_READOUT", 2)
    table = pd.concat(B.outside("tms", ("all", 0), "geneformer_cells")["predictions"])
    assert (table["variant"] == "outside_moments_scaling").all()
    for (name, r), out in fits.items():
        half = B.bulk_halves("tms", ("all", 0), c, r)
        expected = readout_secondary.outside_moments(X_bulk, half) @ out["coef"] + out["intercept"]
        assert np.allclose(table[(table["readout"] == name) & (table["readout_seed"] == r)]["score"], expected, rtol=1e-12, atol=1e-12)


def test_cv_folds_stage03_or_dealt(monkeypatch):
    from sc2bulk import seeds
    B = stage08b()
    samples = pd.DataFrame({"sample_id": [f"s{i}" for i in range(20)], "holdout_group": [f"s{i}" for i in range(20)],
                            "group": np.repeat([0, 1], 10)})
    monkeypatch.setitem(B.R8, "readout_folds", lambda pair, unit, ids, r, context: np.full(len(ids), r))
    monkeypatch.setattr(B, "N_READOUT", 2)
    full = B.cv_folds("tms", ("all", 0), samples, list(samples["sample_id"]))
    assert full[1].tolist() == [1] * 20                                                                # stage 03's
    part = B.cv_folds("tms", ("all", 0), samples.iloc[:16], list(samples["sample_id"]))
    expected = readout.readout_folds(samples.iloc[:16], seeds.derived("readout_folds", "tms", 1, "all", 0)).to_numpy()
    assert np.array_equal(part[1], expected)                                                          # dealt with stage 03's seed


def test_a_stratum_too_small_to_halve_fails():
    with pytest.raises(AssertionError):
        readout_secondary.halves(["a"], ["s"], 0)


def fake_context(B, monkeypatch, n=40, p=6):
    X, y, _, X_bulk = data(n=n, p=p)
    ids = [f"s{i}" for i in range(n)]
    train = pd.DataFrame({"sample_id": ids, "holdout_group": ids, "group": y, "label_value": np.linspace(20, 70, n)})
    bulk = pd.DataFrame({"sample_id": [f"b{i}" for i in range(30)], "scoring_unit_id": [f"b{i}" for i in range(30)]})
    c = {"train": train, "train_ids": ids, "y": y, "bulk": bulk, "bulk_ids": list(bulk["sample_id"]), "strata_train": None, "strata_bulk": None}
    monkeypatch.setattr(B, "context", lambda pair, unit: c)
    monkeypatch.setattr(B, "primary_features", lambda pair, unit, key, a, b: [(-1, X, X_bulk)])
    monkeypatch.setattr(B, "N_READOUT", 2)
    monkeypatch.setitem(B.R8, "readout_folds", lambda pair, unit, ids_, r, context="training": np.arange(len(ids_)) % 5)
    return c, X, X_bulk


def test_permuted_family(monkeypatch):
    from sc2bulk import seeds, training_inputs
    B = stage08b()
    c, X, X_bulk = fake_context(B, monkeypatch)
    flip = lambda j: np.roll(c["y"], j + 1)
    monkeypatch.setattr(training_inputs, "permuted_groups", lambda pair, unit, j: pd.DataFrame({"group": flip(j)}, index=c["train_ids"]))
    monkeypatch.setattr(training_inputs, "encoder_seeds", lambda pair, row: [0, 1, 2])
    table = pd.concat(B.permuted("lupus", ("all", 0), "geneformer_cells")["predictions"])
    assert set(table["permutation"]) == {0, 1, 2} and (table["encoder_seed"] == -1).all() and set(table["readout"]) == {"ridge"}
    samples = c["train"].assign(group=flip(1))                                            # y = row D's labels, folds dealt on them
    folds_ = readout.readout_folds(samples, seeds.derived("readout_folds", "lupus", 1, "all", 0)).to_numpy()
    out = readout.run(X, flip(1), folds_, X_bulk)
    one = table[(table["permutation"] == 1) & (table["readout_seed"] == 1)]
    assert np.allclose(one["score"], out["score"])


def test_learning_curve_family(monkeypatch):
    from sc2bulk import training_inputs
    B = stage08b()
    c, X, X_bulk = fake_context(B, monkeypatch)
    keep = {s: "train" for s in c["train_ids"][::2]}
    monkeypatch.setattr(training_inputs, "learning_curve_subset", lambda pair, unit, e, f: keep)
    monkeypatch.setattr(training_inputs, "encoder_seeds", lambda pair, row: [0, 1])
    tables = B.learning_curve("lupus", ("all", 0), "geneformer_cells")
    table = pd.concat(tables["predictions"])
    assert set(table["encoder_seed"]) == {0, 1}                                            # e for every configuration
    assert set(table["variant"]) == {"training_fraction_0.25", "training_fraction_0.5"}
    heads = pd.concat(tables["heads"])
    assert set(heads["readout"]) == {"ridge", "lasso"}


def test_numeric_age_family(monkeypatch):
    from sc2bulk import training_inputs
    B = stage08b()
    c, X, X_bulk = fake_context(B, monkeypatch)
    z = dict(zip(c["train_ids"], (c["train"]["label_value"] - 45) / 15))
    monkeypatch.setattr(training_inputs, "training_ages", lambda pair, unit, ids: z)
    table = pd.concat(B.numeric_age("hammond", ("all", 0), "geneformer_cells")["predictions"])
    out = readout_secondary.ridge_continuous(X, np.array([z[s] for s in c["train_ids"]]), np.arange(40) % 5, X_bulk)
    assert (table["variant"] == "numeric_age_readout").all()
    assert np.allclose(table[table["readout_seed"] == 0]["score"], out["score"])


# ---- Fig 2A's reference rows ----

def cells_toy(n_cells=400, p=8, seed=0):
    """Cells of 20 samples (20 cells each), a 0/1 group per sample, five CV folds by sample, and bulk rows."""
    rng = np.random.default_rng(seed)
    sample = np.repeat(np.arange(20), n_cells // 20)
    y = (sample % 2).astype(float)
    X = rng.normal(size=(n_cells, p)) + 0.8 * y[:, None] * np.linspace(1, 0, p) + 0.5 * rng.normal(size=(20, p))[sample]
    return X, y, sample % 5, rng.normal(size=(30, p)) * 2 + 3


def test_cell_ridge_equals_readout_run():
    """With cell-level moments (what readout.run takes), cell_ridge equals readout.run's ridge: the CV-MSE
    curve (rtol 1e-10), alpha (or a near-tie), and coef, intercept and scores (allclose 1e-8)."""
    from sklearn.linear_model import Ridge
    for seed in range(3):
        X, y, folds, X_bulk = cells_toy(seed=seed)
        alphas = [C.PER_CELL_CAP * a for a in R.RIDGE_ALPHAS]
        Z, Z_bulk = readout.scale(X, readout.scaler(X)), readout.scale(X_bulk, readout.scaler(X_bulk))
        out = readout_secondary.cell_ridge(Z, y, folds, Z_bulk, alphas)
        search = GridSearchCV(Ridge(fit_intercept=R.READOUT_FIT_INTERCEPT), {"alpha": alphas}, cv=PredefinedSplit(folds),
                              scoring="neg_mean_squared_error").fit(Z, y)
        assert np.allclose(out["cv_mse"], -search.cv_results_["mean_test_score"], rtol=1e-10, atol=0)
        ref = readout.run(X, y, folds, X_bulk, alphas=alphas)
        if out["alpha"] != ref["alpha"]:   # a near-tie on the plateau may flip argmin between solvers
            curve = dict(zip(alphas, out["cv_mse"]))
            assert np.isclose(curve[out["alpha"]], curve[ref["alpha"]], rtol=1e-10, atol=0)
            continue
        assert out["alpha_at_boundary"] == ref["alpha_at_boundary"]
        assert np.allclose(out["coef"], ref["coef"], rtol=1e-8, atol=1e-12) and np.isclose(out["intercept"], ref["intercept"], rtol=1e-8)
        assert np.allclose(out["score"], ref["score"], rtol=1e-8, atol=1e-12)


def test_cell_ridge_ties_go_to_the_smaller_alpha():
    X, y, folds, X_bulk = cells_toy()
    out = readout_secondary.cell_ridge(np.zeros_like(X), y, folds, np.zeros_like(X_bulk), [1.0, 10.0, 100.0])   # every alpha fits the mean
    assert out["alpha"] == 1.0 and out["alpha_at_boundary"] and np.allclose(out["coef"], 0)


def test_per_cell_cap_and_grid():
    assert C.PER_CELL_CAP == settings.MIN_CELLS_PER_SAMPLE   # every sample has exactly the cap outside the smoke run


def per_cell_fixture(B, monkeypatch, cells_per_sample=120):
    """A fake unit: 20 training samples with cells_per_sample cells each, 30 bulk rows."""
    import types
    from sc2bulk import training_inputs
    c, X, X_bulk = fake_context(B, monkeypatch, n=20, p=8)
    rng = np.random.default_rng(3)
    sample = np.repeat(c["train_ids"], cells_per_sample)
    position = np.repeat(np.arange(20), cells_per_sample)
    cell_X = np.abs(X[position] + rng.normal(size=(len(sample), 8)))
    cells = types.SimpleNamespace(X=sp.csr_matrix(cell_X), genes=[f"g{i}" for i in range(8)], cell_id=np.array([f"c{i}" for i in range(len(sample))]),
                                  sample_id=sample)
    monkeypatch.setattr(training_inputs, "unit_cells", lambda pair, unit: cells)
    monkeypatch.setattr(B, "pseudobulk_bulk", lambda pair, unit, ids: X_bulk)
    return c, cells, X_bulk


def test_per_cell_family(monkeypatch):
    from sc2bulk import bags
    B = stage08b()
    monkeypatch.setattr(settings, "SMOKE", False)
    c, cells, X_bulk = per_cell_fixture(B, monkeypatch)
    tables = B.per_cell("tms", ("all", 0), "per_cell_classifier")
    assert set(tables) == set(B.kinds(("all", 0), "reference_rows", "per_cell_classifier"))
    test = bags.test_bags(cells.cell_id, cells.sample_id, c["train_ids"], 100, 10, "tms", ("all", 0))   # Fig 4A's n = 100 bags: bag 0
    row = pd.Series(np.arange(len(cells.cell_id)), index=cells.cell_id)
    take = np.concatenate([row[test[s][0]].to_numpy() for s in c["train_ids"]])
    position = np.repeat(np.arange(20), 100)
    X_cells = cells.X[take].toarray()
    means = np.vstack([X_cells[position == i].mean(axis=0) for i in range(20)])
    moments = readout.scaler(means)                                                       # sample-level moments
    assert np.allclose(moments["all"][0], X_cells.mean(axis=0))                           # the mean is the cell-level mean
    assert not np.allclose(moments["all"][1], X_cells.std(axis=0))                        # the SD is the samples'
    for r in range(2):                                                                    # a cell's fold is its sample's
        out = readout_secondary.cell_ridge(readout.scale(X_cells, moments), c["y"][position], (np.arange(20) % 5)[position],
                                           readout.scale(X_bulk, readout.scaler(X_bulk)), [100 * a for a in R.RIDGE_ALPHAS])
        got = tables["predictions"][r]
        assert np.allclose(got["score"], out["score"], rtol=1e-12) and (got["readout_seed"] == r).all()
    table = pd.concat(tables["predictions"])
    assert set(table["readout"]) == {"ridge"} and set(table["n_cells"]) == {100} and set(table["row_kind"]) == {"reference"}
    heads = pd.concat(tables["heads"])
    assert set(heads["side"]) == {"single_cell", "bulk"} and set(heads["alpha"]) <= {100 * a for a in R.RIDGE_ALPHAS}
    schemas_check(table, "predictions"), schemas_check(heads, "heads")
    few = per_cell_fixture(B, monkeypatch, cells_per_sample=80)
    with pytest.raises(AssertionError, match="fewer than 100"):
        B.per_cell("tms", ("all", 0), "per_cell_classifier")


def schemas_check(table, kind):
    from sc2bulk import schemas
    return schemas.check(kind, schemas.with_neutral_keys(table.reset_index(drop=True)))


def test_pca64_family(monkeypatch):
    from sklearn.decomposition import PCA
    from sc2bulk import properties
    B = stage08b()
    c, X, X_bulk = fake_context(B, monkeypatch, n=20, p=30)
    calls = []
    real = properties.pca64
    monkeypatch.setattr(properties, "pca64", lambda *a: calls.append(a) or real(*a))   # the shared function, not a copy
    monkeypatch.setitem(B.R8, "features", lambda pair, unit, key, a, b: [(-1, X, X_bulk)])
    tables = B.pca64("tms", ("all", 0), "pseudobulk_pca64")
    assert len(calls) == 1 and set(tables) == set(B.kinds(("all", 0), "reference_rows", "pseudobulk_pca64"))
    assert np.allclose(calls[0][1], readout.scale(X_bulk, readout.scaler(X_bulk)))           # bulk standardized on its own rows first (D2)
    Z = readout.scale(X, readout.scaler(X))
    P_train, P_bulk, k = real(*calls[0])
    assert k == 19 and np.allclose(P_train, PCA(n_components=19, svd_solver="full").fit(Z).transform(Z), atol=1e-10)   # its re-standardization is the identity
    heads = pd.concat(tables["heads"])
    assert set(heads["gene_or_dim"]) == {f"pc{i}" for i in range(19)} and set(heads["readout"]) == {"ridge", "lasso"}
    order = np.random.default_rng(0).permutation(30)
    monkeypatch.setitem(B.R8, "features", lambda pair, unit, key, a, b: [(-1, X, X_bulk[order])])
    again = B.pca64("tms", ("all", 0), "pseudobulk_pca64")
    for before, after in zip(tables["predictions"], again["predictions"]):                 # no fit on bulk
        assert np.allclose(before["score"].to_numpy()[order], after["score"].to_numpy())
    schemas_check(pd.concat(tables["predictions"]), "predictions")


def test_mean_log_cells_family(monkeypatch):
    from sc2bulk import properties
    B = stage08b()
    c, cells, X_bulk = per_cell_fixture(B, monkeypatch)
    seen = []
    monkeypatch.setattr(B, "transfer", lambda c_, X_train, X_bulk_, cv, row, dims: seen.append((X_train, X_bulk_, row, dims)) or ([], []))
    assert set(B.mean_log("tms", ("all", 0), "mean_log_cells")) == set(B.kinds(("all", 0), "reference_rows", "mean_log_cells"))
    X_train, X_bulk_, row, dims = seen[0]
    assert np.array_equal(X_train, properties.mean_log_cells(cells.X, cells.sample_id, c["train_ids"]))
    assert X_bulk_ is X_bulk and dims == cells.genes and row["key"] == "mean_log_cells" and row["n_cells"] == 0 and row["encoder_seed"] == -1


def test_cytoset_untrained_family(tmp_path, monkeypatch):
    import json
    from sc2bulk import training_inputs
    B = stage08b()
    c, X, X_bulk = fake_context(B, monkeypatch)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"frozen": {"ncell": 500}}))
    monkeypatch.setattr(B, "untrained_selection", lambda pair, unit: selection)
    monkeypatch.setattr(training_inputs, "encoder_seeds", lambda pair, row: [0, 1] if row == "C" else 1 / 0)
    sides = {e: {s: "validation" if i % 5 == e else "train" for i, s in enumerate(c["train_ids"])} for e in (0, 1)}
    monkeypatch.setattr(training_inputs, "encoder_sides", lambda pair, unit, e: sides[e])
    read = []
    def embedding(path):
        e = int(path.stem[1:])
        read.append(path.parts[-5:])
        obs = pd.DataFrame({"sample_id": c["train_ids"] + c["bulk_ids"], "side": [f"single_cell_{sides[e][s]}" for s in c["train_ids"]] + ["bulk"] * 30})
        return obs.assign(row=np.arange(len(obs))), np.vstack([X, X_bulk]) + e
    monkeypatch.setattr(B, "embedding", embedding)
    tables = B.cytoset_untrained("tms", ("all", 0), "cytoset_untrained")
    assert set(tables) == set(B.kinds(("all", 0), "reference_rows", "cytoset_untrained"))
    assert [p[1] for p in read] == ["C", "C"] and [p[-1] for p in read] == ["s0.h5ad", "s1.h5ad"]   # row C, seeds 0-9 only
    table = pd.concat(tables["predictions"])
    assert set(table["encoder_seed"]) == {0, 1} and set(table["n_cells"]) == {500} and set(table["model"]) == {"cytoset"}
    sides[1] = {s: "train" for s in c["train_ids"]}                                             # stage 06's sides aren't split 1's
    monkeypatch.setattr(B, "embedding", lambda path: (pd.DataFrame({"sample_id": c["train_ids"], "side": ["single_cell_validation"] * 40,
                                                                     "row": np.arange(40)}), X))
    with pytest.raises(AssertionError, match="sides differ"):
        B.cytoset_untrained("tms", ("all", 0), "cytoset_untrained")


def fractions_toy(organs=("kidney", "lung")):
    """Long fractions: per organ its own types; single-cell estimated and true, bulk estimated."""
    rng = np.random.default_rng(4)
    parts = []
    for side, kind, prefix, n in (("single_cell", "estimated", "s", 12), ("single_cell", "true", "s", 12), ("bulk", "estimated", "b", 8)):
        for i in range(n):
            organ = organs[i % len(organs)]
            types_ = ["B", "T", "myeloid"] if organ == organs[0] else ["B", "epithelial"]
            f = rng.dirichlet(np.ones(len(types_)))
            parts.append(pd.DataFrame({"sample_id": f"{prefix}{i}", "type": types_, "fraction": f, "stratum": organ, "side": side, "kind": kind}))
    return pd.concat(parts, ignore_index=True)


def test_fraction_features():
    from sc2bulk import deconvolution
    table = fractions_toy()
    part = table[(table["side"] == "single_cell") & (table["kind"] == "estimated")]
    ids = [f"s{i}" for i in (3, 0, 1)]
    X, dims = readout_secondary.fraction_features(part, ids, "clr")
    assert dims == ["kidney:B", "kidney:T", "kidney:myeloid", "lung:B", "lung:epithelial"]   # organ-prefixed: types repeat across organs
    kidney = part[part["sample_id"] == "s0"].set_index("type")["fraction"][["B", "T", "myeloid"]]
    assert np.allclose(X[1, :3], deconvolution.clr(kidney.to_frame().T).to_numpy().ravel())   # deconvolution.clr per block
    assert np.array_equal(X[1, 3:], [0, 0]) and np.array_equal(X[0, :3], [0, 0, 0])          # zero outside the sample's organ
    raw, _ = readout_secondary.fraction_features(part, ids, "raw")
    assert np.allclose(raw[1, :3], kidney.to_numpy())
    one = fractions_toy(organs=("all",))
    X_one, dims_one = readout_secondary.fraction_features(one[(one["side"] == "bulk")], ["b0", "b1"], "clr")
    assert dims_one == ["all:B", "all:T", "all:myeloid"] and np.allclose(X_one.sum(axis=1), 0)
    with pytest.raises(AssertionError, match="without fractions"):
        readout_secondary.fraction_features(part, ["s0", "b0"], "clr")
    assert readout_secondary.type_list_sha256(part) == readout_secondary.type_list_sha256(part.iloc[::-1])   # order-free


def test_deconvolution_rows_and_review_gate(monkeypatch):
    B = stage08b()
    c, X, X_bulk = fake_context(B, monkeypatch)
    table = fractions_toy()
    c.update(train_ids=[f"s{i}" for i in range(12)], bulk_ids=[f"b{i}" for i in range(8)])
    monkeypatch.setattr(B, "read_fractions", lambda path: table)
    monkeypatch.setattr(B, "checked", lambda path, stage: path)
    monkeypatch.setattr(B.manifest, "content_hash", lambda path: "types-hash")
    monkeypatch.setattr(settings, "SMOKE", False)
    with pytest.raises(AssertionError, match="pinned"):                                    # no pin: the bulk read-out waits
        B.fractions("lupus", ("all", 0), "")
    monkeypatch.setitem(C.DECONVOLUTION_TYPES_SHA256, ("lupus", "all_0", "main"), "types-hash")
    monkeypatch.setitem(C.DECONVOLUTION_TYPES_SHA256, ("lupus", "all_0", "no_ifn"), readout_secondary.type_list_sha256(table))
    assert B.fractions("lupus", ("all", 0), "") is table and B.fractions("lupus", ("all", 0), "no_ifn_markers") is table
    seen = []
    monkeypatch.setattr(B, "transfer", lambda c_, X_train, X_bulk_, cv, row, dims: seen.append((X_train, X_bulk_, row, dims)) or ([], []))
    monkeypatch.setattr(B, "cv_folds", lambda *a: None)
    tables = B.deconvolution_row("lupus", ("all", 0), "deconvolution_true/raw_fractions")
    assert set(tables) == set(B.kinds(("all", 0), "reference_rows", "deconvolution_true/raw_fractions"))
    X_train, X_bulk_, row, dims = seen[-1]
    true = table[(table["kind"] == "true") & (table["side"] == "single_cell")]
    assert np.allclose(X_train, readout_secondary.fraction_features(true, c["train_ids"], "raw")[0])   # trained on annotated fractions
    bulk = table[table["side"] == "bulk"]
    assert np.allclose(X_bulk_, readout_secondary.fraction_features(bulk, c["bulk_ids"], "raw")[0])   # bulk: the estimated fractions
    assert row["key"] == "deconvolution_true" and row["variant"] == "raw_fractions" and row["row_kind"] == "reference"


def test_reference_jobs_per_pair(monkeypatch):
    from sc2bulk import training_inputs
    B = stage08b()
    monkeypatch.setattr(training_inputs, "encoder_seeds", lambda pair, row: list(range(10)))
    names = {pair: [n for n, _, _ in B.reference_jobs(pair, ("all", 0))] for pair in ("hammond", "tms", "lupus")}
    common = {"pseudobulk_pca64", "mean_log_cells", "per_cell_classifier", "cytoset_untrained", "known_signature", "known_signature_genes"}
    assert set(names["hammond"]) == common                                                      # mouse microglia: no deconvolution
    deconvolution_names = {"deconvolution_clr", "deconvolution_true", "deconvolution_clr_within", "deconvolution_true_within",
                           "deconvolution_clr/raw_fractions", "deconvolution_true/raw_fractions", "deconvolution_flags"}
    assert set(names["tms"]) == common | deconvolution_names | {"known_signature/pooled_z"}
    assert set(names["lupus"]) == common | deconvolution_names | {"known_signature/median_centered", "deconvolution_clr/no_ifn_markers",
                                                                  "deconvolution_true/no_ifn_markers"}
    assert names["tms"].index("deconvolution_clr") < names["tms"].index("deconvolution_clr_within")   # heads before the held-out rows
    tms = {n: extra for n, _, extra in B.reference_jobs("tms", ("all", 0))}
    assert any("ensembl" in str(p) for p in tms["known_signature"])                             # the ortholog tables are hashed inputs (F6)
    assert [p.name for p in tms["cytoset_untrained"]][:2] == ["s0.h5ad", "s1.h5ad"]
    assert B.kinds(("all", 0), "reference_rows", "known_signature/pooled_z") == ("predictions",)
    assert B.kinds(("all", 0), "reference_rows", "deconvolution_flags") == ("csv",)
    assert B.kinds(("crossfit", 0), "reference_rows", "deconvolution_clr_within") == ("predictions",)
    assert B.kinds(("all", 0), "reference_rows", "per_cell_classifier") == ("predictions", "heads")
    assert [p.name for p in B.outputs("tms", ("all", 0), "reference_rows", "known_signature_genes")] == ["known_signature_genes.csv"]
    assert [p.name for p in B.outputs("tms", ("all", 0), "reference_rows", "deconvolution_clr/raw_fractions")] == \
        ["deconvolution_clr__raw_fractions_predictions.parquet", "deconvolution_clr__raw_fractions_heads.parquet"]


def test_deconvolution_within(monkeypatch):
    """D6: per split e, a fit on its training side's fractions scoring its validation side; estimated fractions from
    split e's own reference (a training sample without fractions in split e is left out of that fit)."""
    from sc2bulk import training_inputs
    B = stage08b()
    c, X, X_bulk = fake_context(B, monkeypatch, n=20)
    ids = c["train_ids"]
    c["train"] = c["train"].assign(organ="all")
    rng = np.random.default_rng(5)
    sides = {e: {s: "validation" if i % 4 == e else "train" for i, s in enumerate(ids)} for e in (0, 1)}
    rows = [{"sample_id": s, "split": e, "side": sides[e][s], "stratum": "all", "type": t, "estimated": f}
            for e in (0, 1) for s in ids[(e == 1):] for t, f in zip(("B", "T", "NK"), rng.dirichlet(np.ones(3)))]
    splits = pd.DataFrame(rows)
    monkeypatch.setattr(B, "read_fractions", lambda path: splits)
    monkeypatch.setattr(B, "N_ENCODER", 2)
    monkeypatch.setattr(training_inputs, "encoder_sides", lambda pair, unit, e: sides[e])
    monkeypatch.setitem(B.R8, "scoring_units", lambda pair, table: table["sample_id"])
    monkeypatch.setitem(B.R8, "readout_folds", lambda pair, unit, ids_, r, context="training": np.arange(len(ids_)) % 5)   # split 0: stage 03's
    gated = []
    monkeypatch.setattr(B, "fractions", lambda pair, unit, variant: gated.append((pair, unit, variant)))
    out = B.deconvolution_within("lupus", ("all", 0), "deconvolution_clr_within")
    assert gated == [("lupus", ("all", 0), "")]                                            # the review gate runs first
    assert set(out) == set(B.kinds(("all", 0), "reference_rows", "deconvolution_clr_within"))   # its own heads, outside cross-fitting
    table = pd.concat(out["predictions"])
    assert set(table["side"]) == {"single_cell_validation"} and set(table["encoder_seed"]) == {0, 1}
    e = 1
    part = splits[splits["split"] == e].rename(columns={"estimated": "fraction"})
    present = [s for s in ids if s in set(part["sample_id"])]
    samples = c["train"][c["train"]["sample_id"].isin(present)]
    fit = (samples["sample_id"].map(sides[e]) == "train").to_numpy()
    Xf = readout_secondary.fraction_features(part, present, "clr")[0]
    from sc2bulk import seeds
    folds_ = readout.readout_folds(samples[fit], seeds.derived("readout_folds", "lupus", 0, "all", 0)).to_numpy()
    ref = readout.run(Xf[fit], samples[fit]["group"].to_numpy(), folds_, None, X_scored=Xf[~fit])
    got = table[(table["encoder_seed"] == e) & (table["readout"] == "ridge") & (table["readout_seed"] == 0)]
    assert list(got["sample_id"]) == list(samples[~fit]["sample_id"]) and np.allclose(got["score"], ref["scored_score"])


def test_deconvolution_within_unpinned_raises(monkeypatch):
    """An all_0 unit's within-single-cell row of the estimated fractions passes the review gate too (deconvolution
    V1-V4 review): without its pin it stops before reading fractions_splits.parquet."""
    B = stage08b()
    fake_context(B, monkeypatch)
    read = []
    monkeypatch.setattr(B, "read_fractions", lambda path: read.append(path.name) or fractions_toy())
    monkeypatch.setattr(B, "checked", lambda path, stage: path)
    monkeypatch.setattr(B.manifest, "content_hash", lambda path: "types-hash")
    monkeypatch.setitem(B.R8, "scoring_units", lambda pair, table: table["sample_id"])
    monkeypatch.setattr(settings, "SMOKE", False)
    monkeypatch.delitem(C.DECONVOLUTION_TYPES_SHA256, ("lupus", "all_0", "main"), raising=False)
    with pytest.raises(AssertionError, match="pinned"):
        B.deconvolution_within("lupus", ("all", 0), "deconvolution_clr_within")
    assert read == ["fractions.parquet"]


def test_known_signature_score_by_hand():
    cpm = np.array([[10, 0, 5, 100, 2], [20, 3, 5, 90, 2], [30, 0, 6, 80, 2], [15, 4, 7, 70, 2], [25, 0, 8, 60, 2], [35, 5, 9, 50, 2]], dtype=float)
    used = readout_secondary.signature_used(cpm)
    assert used.tolist() == [True, True, True, True, False]       # gene 5 is constant; gene 2 is >= 1 CPM in half the samples
    assert readout_secondary.signature_detected(cpm / 1000).tolist() == [False] * 5   # below 1 CPM everywhere
    L = np.log1p(cpm[:, :4])
    z = (L - L.mean(axis=0)) / L.std(axis=0)                                   # ddof 0 (SCALER_DDOF)
    up = np.array([True, True, False, False])
    assert np.allclose(readout_secondary.signature_score(cpm[:, :4], up), z[:, :2].mean(axis=1) - z[:, 2:].mean(axis=1))
    assert np.allclose(readout_secondary.signature_score(cpm[:, :2], [True, True]), z[:, :2].mean(axis=1))   # up-only: the up term
    organ = np.array(["kidney", "lung"] * 3)
    within = readout_secondary.signature_score(cpm[:, :4], up, organ)
    for o in ("kidney", "lung"):
        Lo = L[organ == o]
        sd = Lo.std(axis=0)
        zo = np.where(sd > 1e-6 * np.maximum(1, np.abs(Lo.mean(axis=0))), (Lo - Lo.mean(axis=0)) / np.where(sd > 0, sd, 1), 0.0)   # constant within organ: z 0
        assert np.allclose(within[organ == o], zo[:, :2].mean(axis=1) - zo[:, 2:].mean(axis=1))
    assert np.allclose(readout_secondary.median_centered(cpm[:, :3]), np.median(np.log1p(cpm[:, :3]) - np.log1p(cpm[:, :3]).mean(axis=0), axis=1))


def test_signature_matching(monkeypatch):
    from sc2bulk import orthologs, signatures
    B = stage08b()
    listed = pd.DataFrame({"gene": ["CD74", "APOE", "TREM2"], "direction": ["up", "up", "down"]})
    monkeypatch.setattr(signatures, "load", lambda pair: listed)
    var = pd.DataFrame({"ensembl": ["e1", "e2", "e3", "e4"]}, index=["Cd74", "Apoe", "Trem2", "Actb"])
    matched = B.signature_matches("hammond", ("all", 0), var)
    assert list(matched["gene"]) == ["Cd74", "Apoe", "Trem2"] and list(matched["direction"]) == ["up", "up", "down"]   # case-insensitive
    collide = pd.DataFrame({"ensembl": ["e1", "e2"]}, index=["Cd74", "CD74"])
    with pytest.raises(AssertionError, match="case"):
        B.signature_matches("hammond", ("all", 0), collide)
    assert list(B.signature_matches("lupus", ("all", 0), pd.DataFrame(index=["CD74", "TREM2.1", "APOE"]))["gene"]) == ["CD74", "APOE"]
    monkeypatch.setattr(orthologs, "one2one", lambda raw: "pairs")
    monkeypatch.setattr(orthologs, "human_names", lambda raw: "names")
    monkeypatch.setattr(orthologs, "map_genes", lambda genes, pairs, names: genes.assign(
        human_symbol=["CD74", "APOE", "", "TREM2"], status=["mapped", "mapped", "no one2one ortholog", "conflicting ids"]))
    assert list(B.signature_matches("tms", ("all", 0), var)["gene"]) == ["Cd74", "Apoe"]          # one-to-one orthologs only


def test_known_signature_family(monkeypatch):
    import types
    B = stage08b()
    bulk = pd.DataFrame({"sample_id": [f"b{i}" for i in range(6)], "scoring_unit_id": [f"b{i}" for i in range(6)]})
    c = {"bulk": bulk, "bulk_ids": list(bulk["sample_id"]), "strata_bulk": np.array(["kidney", "lung"] * 3)}
    rng = np.random.default_rng(6)
    cpm = rng.gamma(2.0, 20.0, size=(6, 4))
    matched = pd.DataFrame({"gene": list("abcd"), "direction": ["up", "up", "down", "up"]})
    monkeypatch.setattr(B, "signature_inputs", lambda pair, unit: (c, cpm, matched, np.array([True, True, True, False])))
    out = B.known_signature("tms", ("all", 0), "known_signature")["predictions"][0]
    assert np.allclose(out["score"], readout_secondary.signature_score(cpm[:, :3], [True, True, False], c["strata_bulk"]))   # within organ
    assert out["probability"].isna().all() and set(out["readout"]) == {"none"} and set(out["readout_seed"]) == {-1}
    assert set(out["encoder_seed"]) == {-1} and set(out["row_kind"]) == {"reference"}
    pooled = B.known_signature("tms", ("all", 0), "known_signature/pooled_z")["predictions"][0]
    assert np.allclose(pooled["score"], readout_secondary.signature_score(cpm[:, :3], [True, True, False])) and set(pooled["variant"]) == {"pooled_z"}
    median = B.known_signature("lupus", ("all", 0), "known_signature/median_centered")["predictions"][0]
    assert np.allclose(median["score"], readout_secondary.median_centered(cpm[:, :3]))
    for table in (out, pooled, median):
        schemas_check(table, "predictions")
    for name in ("known_signature", "known_signature/pooled_z", "known_signature/median_centered"):
        assert set(B.known_signature("tms", ("all", 0), name)) == set(B.kinds(("all", 0), "reference_rows", name))   # no heads


def test_signature_inputs_cpm_over_the_full_axis(monkeypatch):
    import anndata
    B = stage08b()
    bulk = pd.DataFrame({"sample_id": [f"b{i}" for i in range(6)], "scoring_unit_id": [f"b{i}" for i in range(6)]})
    c = {"bulk": bulk, "bulk_ids": list(bulk["sample_id"]), "strata_bulk": None}
    counts = np.random.default_rng(7).poisson(50, size=(6, 5)).astype(np.int64)
    data = anndata.AnnData(counts, obs=pd.DataFrame(index=bulk["sample_id"]), var=pd.DataFrame({"ensembl": [""] * 5}, index=list("vwxyz")))
    monkeypatch.setattr(B, "context", lambda pair, unit: c)
    monkeypatch.setattr(anndata, "read_h5ad", lambda path: data)
    monkeypatch.setattr(B, "signature_matches", lambda pair, unit, var: pd.DataFrame({"gene": ["x", "v"], "direction": ["up", "down"]}))
    from sc2bulk import signatures
    monkeypatch.setattr(signatures, "load", lambda pair: pd.DataFrame({"gene": ["X", "V", "Q"], "direction": ["up", "down", "up"]}))
    _, cpm, matched, used = B.signature_inputs("lupus", ("all", 0))
    assert np.allclose(cpm, counts[:, [2, 0]] / counts.sum(axis=1, keepdims=True) * 1e6) and used.all()
    tables = B.known_signature_genes("lupus", ("all", 0), "known_signature_genes")
    assert set(tables) == set(B.kinds(("all", 0), "reference_rows", "known_signature_genes"))
    genes = tables["csv"][0]
    assert genes.set_index("direction")[["listed", "matched", "used"]].to_dict("index") == {"down": {"listed": 1, "matched": 1, "used": 1},
                                                                                            "up": {"listed": 2, "matched": 1, "used": 1}}


def test_deconvolution_within_crossfit_uses_the_transfer_heads(monkeypatch):
    """Human blood, aging: the fold's held-out samples' fractions (04b's diagnostic files), CLR, scored by the family's
    own transfer heads; their types must be the transfer row's."""
    B = stage08b()
    table = fractions_toy(organs=("all",))
    held = pd.DataFrame({"sample_id": ["h0", "h1", "h2"], "scoring_unit_id": ["d0", "d0", "d1"]})
    wide = pd.DataFrame({"sample_id": held["sample_id"], "B": [0.2, 0.5, 0.3], "T": [0.7, 0.1, 0.3], "myeloid": [0.1, 0.4, 0.4], "residual": 0.1})
    monkeypatch.setattr(B, "context", lambda pair, unit: {})
    monkeypatch.setattr(B, "fractions", lambda pair, unit, variant: table)
    monkeypatch.setattr(B, "read_fractions", lambda path: wide)
    monkeypatch.setitem(B.R8, "heldout_samples", lambda pair, unit: held)
    seen = {}
    monkeypatch.setattr(B, "heldout_by_heads", lambda pair, unit, path, X_of, held_, keys: seen.update(path=path, X=X_of(0), keys=keys) or {"predictions": []})
    B.deconvolution_within("soundlife", ("crossfit", 1), "deconvolution_clr_within")
    long = wide.drop(columns="residual").melt(id_vars="sample_id", var_name="type", value_name="fraction").assign(stratum="all")
    assert np.allclose(seen["X"], readout_secondary.fraction_features(long, list(held["sample_id"]), "clr")[0])
    assert seen["path"].name == "deconvolution_clr_heads.parquet" and seen["keys"]["key"] == "deconvolution_clr"
    monkeypatch.setattr(B, "read_fractions", lambda path: wide.drop(columns="myeloid"))
    with pytest.raises(AssertionError, match="types differ"):
        B.deconvolution_within("soundlife", ("crossfit", 1), "deconvolution_clr_within")


def test_deconvolution_flags(monkeypatch):
    B = stage08b()
    v1 = pd.DataFrame({"type": ["B", "T", "NK"], "r": [0.9, 0.6, 0.8], "gate_passed": True, "stratum": "all"})
    v4 = pd.DataFrame({"stratum": ["all"], "single_cell_median_residual": [0.25], "single_cell_in_sample_median_residual": [0.2],
                       "bulk_median_residual": [0.7], "reference_mismatch": [False], "neutrophil_leak_spearman": [0.2]})
    v3 = {f"crossfit_{f}_v3.csv": pd.DataFrame({"r": [0.3 + 0.1 * f, 0.5, np.nan if f == 1 else 0.6]}) for f in range(4)}
    monkeypatch.setattr(B, "checked", lambda path, stage: path)
    monkeypatch.setattr(B.pd, "read_csv", lambda path, **k: v1 if path.name == "check_v1.csv" else v4 if path.name == "check_v4.csv" else v3[path.name])
    monkeypatch.setattr(B.folds, "units", lambda pair: [("crossfit", f) for f in range(4)] + [("split67", 0)])
    tables = B.deconvolution_flags("soundlife", ("crossfit", 2), "deconvolution_flags")
    assert set(tables) == set(B.kinds(("crossfit", 2), "reference_rows", "deconvolution_flags"))
    row = tables["csv"][0].iloc[0]
    medians = [np.median([0.3 + 0.1 * f, 0.5, 0.0 if f == 1 else 0.6]) for f in range(4)]       # an undefined r counts as 0
    assert row["v1_median_r"] == 0.8 and row["v1_min_r"] == 0.6 and not row["reference_mismatch"] and row["neutrophil_leak_spearman"] == 0.2
    assert np.isclose(row["v4_ratio"], 0.7 / 0.25) and np.isclose(row["v4_in_sample_ratio"], 0.7 / 0.2)
    assert np.isclose(row["v3_median_r"], medians[2]) and np.isclose(row["v3_mean_fold_median_r"], np.mean(medians))
    assert row["v3_flag"] == (np.mean(medians) < 0.5)


def test_bag_size_skips_age_loss_cytoset_all(monkeypatch):
    """No CytoSet row F "all" in a unit whose frozen CytoSet configuration has the age loss (stage 06
    trains none); its input list omits those files; other configurations and units keep the point."""
    from sc2bulk import training_inputs
    B = stage08b()
    c, X, X_bulk = fake_context(B, monkeypatch)
    monkeypatch.setattr(training_inputs, "encoder_seeds", lambda pair, row: [0, 1])
    monkeypatch.setattr(B, "age_loss_cytoset", lambda pair, unit: unit == ("crossfit", 1))
    monkeypatch.setattr(B, "has_per_sample_route", lambda pair, unit, encoder: False)
    monkeypatch.setattr(B, "stage06_rows", lambda pair, unit, row, model, point, e, ids, bulk_ids: (X, X_bulk))
    for unit in (("crossfit", 0), ("crossfit", 1)):
        extra = {name: inputs for family, name, _, inputs in B.jobs("soundlife", unit) if family == "bag_size"}
        for key in ("cytoset_cells", "set_transformer_cells"):
            has_all = not (key == "cytoset_cells" and unit == ("crossfit", 1))
            table = pd.concat(B.bag_size("soundlife", unit, key)["predictions"])
            assert ("all_available" in set(table["cell_subset"])) == has_all
            assert set(table.loc[table["cell_subset"] == "all", "n_cells"]) == {n for n in B.representations.BAG_SIZES if n != 500}
            assert any(p.parts[-3] == "n_all" for p in extra[key]) == has_all
            assert sum(p.parts[-3] == "n1000" for p in extra[key]) == 2


def one_type_jobs_fixture(B, monkeypatch, tmp_path):
    """fake_context's unit under a tmp tree, with the one-type store, row H's seeds (0-1) and their splits (split e: every
    fifth sample validation) given; returns (context, a function listing the unit's one_type job names)."""
    from sc2bulk import training_inputs
    c, _, _ = fake_context(B, monkeypatch)
    monkeypatch.setattr(settings, "OUTPUTS", tmp_path)
    monkeypatch.setattr(training_inputs, "encoder_seeds", lambda pair, row: [0, 1])
    sides = {e: {s: "validation" if i % 5 == e else "train" for i, s in enumerate(c["train_ids"])} for e in (0, 1)}
    monkeypatch.setattr(training_inputs, "encoder_sides", lambda pair, unit, e: sides[e])
    monkeypatch.setattr(B, "store_samples", lambda train: train.reset_index(drop=True))
    monkeypatch.setattr(B, "age_loss_cytoset", lambda pair, unit: False)
    monkeypatch.setattr(B, "has_per_sample_route", lambda pair, unit, encoder: False)
    return c, lambda unit: {name for family, name, *_ in B.jobs("soundlife", unit) if family == "one_type"}


def test_one_type_absent_under_smoke_only(tmp_path, monkeypatch):
    """A unit whose row H stage 06 skipped under smoke (06_train.smoke_marker; tms all_0 in the smoke run) has no
    configuration's one_type jobs, trained or fixed (pseudobulk and the FMs read the same short store), and 08b's
    one_type marker says why; another unit keeps them all and has no marker. The real run asserts there is no marker."""
    import json
    B = stage08b()
    _, one_type = one_type_jobs_fixture(B, monkeypatch, tmp_path)
    unit = ("crossfit", 0)
    everything = one_type(unit)
    assert everything == set(B.KEYS) | {f"{k}_within" for k in B.KEYS}
    marker = tmp_path / "06_train" / "soundlife" / "crossfit_0" / "H" / "smoke_skipped.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    monkeypatch.setattr(settings, "SMOKE", True)
    assert one_type(unit) == set()
    assert json.loads(B.one_type_marker("soundlife", unit).read_text()) == {"row H": ["stage 06 skipped it"]}
    assert one_type(("crossfit", 1)) == everything and not B.one_type_marker("soundlife", ("crossfit", 1)).exists()
    monkeypatch.setattr(settings, "SMOKE", False)
    with pytest.raises(AssertionError, match="smoke one_type marker"):
        B.jobs("soundlife", unit)
    B.one_type_marker("soundlife", unit).unlink()
    with pytest.raises(AssertionError, match="smoke row H marker"):
        B.jobs("soundlife", unit)


def test_one_type_short_store_under_smoke_only(tmp_path, monkeypatch):
    """Under smoke, a one-type store too short for 08b's fits (fewer than READOUT_FOLDS holdout groups, a label missing,
    or outside cross-fitting a split e side short) skips every configuration's one_type jobs with the reasons in 08b's
    marker; the count mirrors readout.readout_folds and readout.run without calling them. The real run lists them all
    and never counts (its asserts stand)."""
    import json
    B = stage08b()
    c, one_type = one_type_jobs_fixture(B, monkeypatch, tmp_path)
    everything = one_type(("crossfit", 0))
    monkeypatch.setattr(settings, "SMOKE", True)
    assert B.one_type_shortfall("soundlife", ("all", 0)) == {"transfer": [], "split 0": [], "split 1": []}
    assert one_type(("all", 0)) == everything and not B.one_type_marker("soundlife", ("all", 0)).exists()
    four = c["train"].iloc[[0, 1, 20, 21]]                                                        # four holdout groups, both labels
    monkeypatch.setattr(B, "store_samples", lambda train: four.reset_index(drop=True))
    assert B.one_type_shortfall("soundlife", ("crossfit", 0)) == {"transfer": ["4 holdout groups"]}
    with pytest.raises(AssertionError, match="a read-out fold is empty"):                         # what it stands in for
        readout.readout_folds(four, 0)
    assert one_type(("crossfit", 0)) == set()
    assert json.loads(B.one_type_marker("soundlife", ("crossfit", 0)).read_text()) == {"transfer": ["4 holdout groups"]}
    zeros = c["train"][c["y"] == 0]                                                               # 20 groups, one label
    monkeypatch.setattr(B, "store_samples", lambda train: zeros.reset_index(drop=True))
    assert B.one_type_shortfall("soundlife", ("crossfit", 0)) == {"transfer": ["no training samples of label 1"]}
    assert one_type(("crossfit", 0)) == set()
    split = c["train"].iloc[[0, 1, 2, 5, 10, 15, 20, 21, 22, 25, 30, 35]]                         # split 0's training side: 4 groups
    monkeypatch.setattr(B, "store_samples", lambda train: split.reset_index(drop=True))
    assert B.one_type_shortfall("soundlife", ("all", 0)) == {"transfer": [], "split 0": ["4 holdout groups"], "split 1": []}
    assert one_type(("all", 0)) == set() and one_type(("crossfit", 0)) == everything            # cross-fitting: no within fits
    assert not B.one_type_marker("soundlife", ("crossfit", 0)).exists()                            # cleared once the store suffices
    monkeypatch.setattr(settings, "SMOKE", False)
    B.one_type_marker("soundlife", ("all", 0)).unlink()
    monkeypatch.setattr(B, "one_type_shortfall", lambda pair, unit: 1 / 0)
    assert one_type(("all", 0)) == everything


def test_fit_shortfall_strata():
    """A stratum of the scored single-cell rows (a within fit's validation side) without training rows (readout.scale's
    assert) and an empty scored side are shortfalls too; the transfer fit passes no strata (bulk on its own moments)."""
    B = stage08b()
    fitted = pd.DataFrame({"holdout_group": [f"m{i}" for i in range(6)], "group": [0, 1] * 3, "organ": "kidney"})
    assert B.fit_shortfall("tms", fitted, ["kidney", "lung"], 3) == ["no training samples of stratum lung"]
    assert B.fit_shortfall("tms", fitted, [], 0) == ["no scored samples"]
    assert B.fit_shortfall("lupus", fitted, [], 3) == []
    with pytest.raises(AssertionError, match="a stratum without moments"):
        readout.scale(np.zeros((1, 2)), readout.scaler(np.ones((6, 2)), fitted["organ"].to_numpy()), np.array(["lung"]))


def test_transfer_fit_takes_no_bulk_strata(monkeypatch):
    """The transfer fit's bulk is scaled on its own moments, so a bulk organ without one-type training samples is no
    shortfall, and readout.run indeed runs."""
    B = stage08b()
    X, y, folds_, X_bulk = data()
    train = pd.DataFrame({"sample_id": [f"s{i}" for i in range(40)], "holdout_group": [f"s{i}" for i in range(40)], "group": y, "organ": "kidney"})
    bulk = pd.DataFrame({"sample_id": [f"b{i}" for i in range(30)]})
    strata_bulk = np.where(np.arange(30) % 2, "kidney", "lung")
    monkeypatch.setattr(B, "context", lambda pair, unit: {"train": train, "bulk": bulk, "strata_bulk": strata_bulk})
    monkeypatch.setattr(B, "store_samples", lambda t: t)
    assert B.one_type_shortfall("tms", ("crossfit", 0)) == {"transfer": []}
    readout.run(X, y, folds_, X_bulk, strata_train=train["organ"].to_numpy(), strata_bulk=strata_bulk)


def test_learning_curve_absent_under_smoke_only(tmp_path, monkeypatch):
    """A unit whose row E stage 06 skipped under smoke (06_train.smoke_marker) has no learning_curve jobs for any
    configuration (the subset itself falls short); other units and families keep theirs. The real run asserts there
    is no marker."""
    B = stage08b()
    one_type_jobs_fixture(B, monkeypatch, tmp_path)
    unit = ("crossfit", 0)
    families = lambda unit: [(family, name) for family, name, *_ in B.jobs("soundlife", unit)]
    everything = families(unit)
    assert {name for family, name in everything if family == "learning_curve"} == set(B.KEYS)
    marker = tmp_path / "06_train" / "soundlife" / "crossfit_0" / "E" / "smoke_skipped.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    monkeypatch.setattr(settings, "SMOKE", True)
    assert families(unit) == [j for j in everything if j[0] != "learning_curve"]
    assert families(("crossfit", 1)) == everything and not B.row_absent("soundlife", ("crossfit", 1), "E")
    monkeypatch.setattr(settings, "SMOKE", False)
    with pytest.raises(AssertionError, match="smoke row E marker"):
        B.jobs("soundlife", unit)


# ---- the fold leak (reference_rows_audit.md "The leak cases") ----

def fold_zero_tree(B, monkeypatch, root, seed):
    """Human blood, aging's crossfit_0 on a synthetic tree under root: 20 training donors and 4 held-out (scored)
    donors, one single-cell sample each, with their cells, pseudobulk profiles, row C embeddings and stage 04b
    fractions (the held-out donors' in 04b's diagnostic files); 24 genes (PCA-64 takes 19 components); 6 scored
    bulk samples. seed draws the held-out donors' single-cell data only. The readers serve the whole store (held-out donors included) wherever they take
    sample ids. Every reference row is read out; returns {file name: bytes}."""
    import types
    from sc2bulk import training_inputs
    pair, unit = "soundlife", ("crossfit", 0)
    rng, held_rng = np.random.default_rng(0), np.random.default_rng(seed)
    train_ids, held_ids, bulk_ids = [f"t{i}" for i in range(20)], [f"h{i}" for i in range(4)], [f"b{i}" for i in range(6)]
    y = np.array([0, 1] * 10)
    train = pd.DataFrame({"sample_id": train_ids, "donor_id": train_ids, "holdout_group": train_ids, "group": y, "label_value": 30 + 30 * y})
    bulk = pd.DataFrame({"sample_id": bulk_ids, "donor_id": ["h0", "h0", "h1", "h2", "h3", "h3"]}).assign(scoring_unit_id=lambda t: t["donor_id"])
    held = pd.DataFrame({"sample_id": held_ids, "donor_id": held_ids, "scoring_unit_id": held_ids})
    profiles = {**{s: rng.normal(size=24) + 0.8 * g for s, g in zip(train_ids, y)}, **{s: held_rng.normal(size=24) for s in held_ids}}
    X_bulk = pd.DataFrame(rng.normal(size=(6, 24)), index=bulk_ids)
    store = list(train_ids) + held_ids
    cell_X = np.vstack([np.abs((rng if s in train_ids else held_rng).normal(size=(120, 24)) + profiles[s]) for s in store])
    cells = types.SimpleNamespace(X=sp.csr_matrix(cell_X), genes=[f"g{i}" for i in range(24)], sample_id=np.repeat(store, 120),
                                  cell_id=np.array([f"{s}_c{i}" for s in store for i in range(120)]))
    sides = {e: {s: "validation" if i % 5 == e else "train" for i, s in enumerate(train_ids)} for e in (0, 1)}
    embeddings = {}
    for e in (0, 1):
        obs = pd.DataFrame({"sample_id": train_ids + held_ids + bulk_ids,
                            "side": [f"single_cell_{sides[e][s]}" for s in train_ids] + ["single_cell_heldout"] * 4 + ["bulk"] * 6})
        embeddings[e] = obs.assign(row=np.arange(30)), np.vstack([rng.normal(size=(20, 5)), held_rng.normal(size=(4, 5)), rng.normal(size=(6, 5))])
    long = [pd.DataFrame({"sample_id": s, "type": ["B", "NK", "T"], "fraction": rng.dirichlet(np.ones(3)), "stratum": "all", "side": side, "kind": kind})
            for side, kind, ids in (("single_cell", "estimated", train_ids), ("single_cell", "true", train_ids), ("bulk", "estimated", bulk_ids)) for s in ids]
    folder = root / "04b_deconvolve" / pair
    (folder / "crossfit_0").mkdir(parents=True)
    (folder / "diagnostic").mkdir()
    pd.concat(long, ignore_index=True).to_parquet(folder / "crossfit_0" / "fractions.parquet")
    pd.DataFrame({"stratum": "all", "type": ["B", "NK", "T"]}).to_csv(folder / "crossfit_0" / "types.csv", index=False)
    for which in ("estimated", "true"):
        wide = pd.DataFrame(held_rng.dirichlet(np.ones(4), size=4), columns=["B", "NK", "T", "residual"]).assign(sample_id=held_ids)
        wide.to_parquet(folder / "diagnostic" / f"crossfit_0_scored_{which}.parquet")
    (root / "05_search" / pair / "crossfit_0" / "cytoset").mkdir(parents=True)
    (root / "05_search" / pair / "crossfit_0" / "cytoset" / "selection.json").write_text('{"frozen": {"ncell": 500}}')

    monkeypatch.setattr(settings, "OUTPUTS", root)
    monkeypatch.setattr(settings, "SMOKE", True)                                                   # no type-list pin on a toy
    c = {"train": train, "train_ids": train_ids, "y": y, "bulk": bulk, "bulk_ids": bulk_ids, "strata_train": None, "strata_bulk": None}
    monkeypatch.setattr(B, "context", lambda pair_, unit_: c)
    monkeypatch.setattr(B, "N_READOUT", 2)
    monkeypatch.setitem(B.R8, "readout_folds", lambda pair_, unit_, ids, r, context="training": np.arange(len(ids)) % 5)
    monkeypatch.setitem(B.R8, "features", lambda pair_, unit_, key, t, b: [(-1, np.vstack([profiles[s] for s in t]), X_bulk.loc[b].to_numpy())])
    monkeypatch.setitem(B.R8, "heldout_samples", lambda pair_, unit_: held)
    monkeypatch.setattr(B, "pseudobulk_bulk", lambda pair_, unit_, ids: X_bulk.loc[ids].to_numpy())
    monkeypatch.setattr(training_inputs, "unit_cells", lambda pair_, unit_: cells)
    monkeypatch.setattr(training_inputs, "encoder_seeds", lambda pair_, row: [0, 1])
    monkeypatch.setattr(training_inputs, "encoder_sides", lambda pair_, unit_, e: sides[e])
    monkeypatch.setattr(B, "embedding", lambda path: embeddings[int(path.stem[1:])])
    monkeypatch.setattr(B, "checked", lambda path, stage: path)
    monkeypatch.setattr(B.manifest, "temporary", lambda path: path)
    monkeypatch.setattr(B.manifest, "finish", lambda *a, **k: None)
    for name, compute, _ in B.reference_jobs(pair, unit):
        if not name.startswith(("known_signature", "deconvolution_flags")):                        # bulk and 04b's checks only
            B.run(pair, unit, "reference_rows", name, lambda: compute(pair, unit, name), {}, {}, [])
    return {p.name: p.read_bytes() for p in (root / "08b_secondary" / pair / "crossfit_0" / "reference_rows").iterdir()}


def test_fold_zero_reference_rows_ignore_its_heldout_donors(tmp_path, monkeypatch):
    """Changing crossfit_0's held-out donors' single-cell data (cells, profiles, row C rows, 04b's diagnostic
    fractions) leaves every transfer row's files byte-identical, and changes the crossfit within files, which
    score exactly those donors (the positive control)."""
    B = stage08b()
    a = fold_zero_tree(B, monkeypatch, tmp_path / "a", seed=1)
    b = fold_zero_tree(B, monkeypatch, tmp_path / "b", seed=2)
    transfer = {f"{n}_{kind}.parquet" for n in ("pseudobulk_pca64", "mean_log_cells", "per_cell_classifier", "cytoset_untrained", "deconvolution_clr",
                                                "deconvolution_true", "deconvolution_clr__raw_fractions", "deconvolution_true__raw_fractions")
                for kind in ("predictions", "heads")}
    within = {"deconvolution_clr_within_predictions.parquet", "deconvolution_true_within_predictions.parquet"}
    assert set(a) == set(b) == transfer | within
    assert {n for n in a if a[n] != b[n]} == within


# ---- every family's tables are its file kinds (the smoke run's alternative_readouts/pls failure) ----

def every_family_fixture(B, monkeypatch, tmp_path):
    """A fake human blood, aging unit (no strata; every family but the reference rows, which fold_zero_tree runs
    through B.run): 40 training samples, 4 held-out ones, 30 bulk; every upstream read served from memory, the
    stored heads a toy fit's. Encoder seeds 0-1, one read-out seed, a short MLP."""
    import anndata
    from sc2bulk import training_inputs
    c, X, X_bulk = fake_context(B, monkeypatch)
    rng = np.random.default_rng(9)
    held = pd.DataFrame({"sample_id": [f"h{i}" for i in range(4)], "scoring_unit_id": [f"h{i}" for i in range(4)], "group": [0, 1, 0, 1]})
    rows = pd.DataFrame(np.vstack([X, rng.normal(size=(4, X.shape[1]))]), index=c["train_ids"] + list(held["sample_id"]))
    obs = pd.DataFrame({"sample_id": list(rows.index) + c["bulk_ids"], "side": ["single_cell_train"] * 40 + ["single_cell_heldout"] * 4 + ["bulk"] * 30})
    embedded = obs.assign(row=np.arange(len(obs))), np.vstack([rows.to_numpy(), X_bulk])
    sides = {e: {s: "validation" if i % 5 == e else "train" for i, s in enumerate(c["train_ids"])} for e in (0, 1)}
    heads, _ = stored_heads(B, X, c["y"], np.arange(40) % 5, X_bulk, None, None)
    heads = heads[heads["readout_seed"] == 0]                                               # one read-out seed
    genes = [f"g{i}" for i in range(300)]
    mu = rng.gamma(1.0, 100, size=300) + 1
    size = 1 / (0.1 + 2 / mu)                                                               # dispersion 0.1 + 2 / mean: VST's parametric trend
    counts = lambda n: rng.negative_binomial(size, size / (size + mu), size=(n, 300)).astype(np.float64)
    single, bulk = counts(40), counts(30)
    panel = pd.DataFrame({"gene": genes[:50], "rank": np.arange(50)})
    native = tmp_path / "native_head.csv"
    pd.DataFrame({"sample_id": c["bulk_ids"], "side": "bulk", "logit": rng.normal(size=30)}).to_csv(native, index=False)
    fm = lambda what: anndata.AnnData(X_bulk if what == "bulk" else X, obs=pd.DataFrame(index=c["bulk_ids"] if what == "bulk" else c["train_ids"]))
    read_csv = pd.read_csv

    monkeypatch.setattr(settings, "OUTPUTS", tmp_path)
    monkeypatch.setattr(B, "N_READOUT", 1)
    monkeypatch.setattr(C, "MLP_MAX_ITER", 20)
    monkeypatch.setattr(training_inputs, "encoder_seeds", lambda pair, row: [0, 1])
    monkeypatch.setattr(training_inputs, "encoder_sides", lambda pair, unit, e: sides[e])
    monkeypatch.setattr(training_inputs, "permuted_groups", lambda pair, unit, j: pd.DataFrame({"group": np.roll(c["y"], j + 1)}, index=c["train_ids"]))
    monkeypatch.setattr(training_inputs, "learning_curve_subset", lambda pair, unit, e, f: {s: "train" for s in c["train_ids"][::2]})
    monkeypatch.setattr(training_inputs, "training_ages", lambda pair, unit, ids: dict(zip(ids, np.linspace(-1, 1, len(ids)))))
    monkeypatch.setattr(training_inputs, "panel", lambda pair, unit: [str(i) for i in range(X.shape[1])])
    monkeypatch.setitem(B.R8, "scoring_units", lambda pair, table: table["sample_id"])
    monkeypatch.setitem(B.R8, "heldout_samples", lambda pair, unit: held)
    monkeypatch.setitem(B.R8, "trained_features", lambda pair, unit, key, e, ids, bulk_ids: (rows.loc[ids].to_numpy(), X_bulk))
    monkeypatch.setitem(B.R8, "trained_embedding", lambda pair, unit, key, e: embedded)
    monkeypatch.setitem(B.R8, "fm_file", lambda pair, unit, model, what: fm(what))
    monkeypatch.setattr(B.folds, "single_cell_samples", lambda pair: pd.concat([c["train"], held]))
    monkeypatch.setattr(B, "stage06_rows", lambda pair, unit, row, model, point, e, ids, bulk_ids, variant="primary": (rows.loc[ids].to_numpy(), X_bulk))
    monkeypatch.setattr(B, "embedding", lambda path, stage="06_train": embedded)
    monkeypatch.setattr(B, "store_samples", lambda train: train.reset_index(drop=True))
    monkeypatch.setattr(B, "one_type_features", lambda pair, unit, key, ids, bulk_ids, e=None, heldout=False: (rows.loc[ids].to_numpy(), X_bulk))
    monkeypatch.setattr(B, "primary_heads", lambda pair, unit, key: heads)
    monkeypatch.setattr(B.schemas, "read", lambda kind, path: heads)                        # the family's own stored transfer heads
    monkeypatch.setattr(B, "checked", lambda path, stage: native if path.name.startswith("native_head") else path)
    monkeypatch.setattr(B.manifest, "read", lambda path: {"settings": {"trial": path.stem.rsplit("_s", 1)[0], "seed": int(path.stem.rsplit("_s", 1)[1])}})
    monkeypatch.setattr(B, "age_loss_cytoset", lambda pair, unit: False)
    monkeypatch.setattr(B, "has_per_sample_route", lambda pair, unit, encoder: True)
    monkeypatch.setattr(B, "candidate_names", lambda pair, unit, encoder: ["base"])
    monkeypatch.setattr(B, "pseudobulk_n", lambda pair, unit, c_: [(n, "all", "native", X, X_bulk) for n in (100, 1000)])
    monkeypatch.setattr(B, "cells_of", lambda cells_file, ids: (X, np.array(ids), np.array(ids)))
    monkeypatch.setattr(B, "pooled_bags", lambda X_, ids, sample, sample_ids, n, pair, unit, pool="mean": rows.loc[list(sample_ids)].to_numpy())
    monkeypatch.setattr(B.anndata, "read_h5ad", lambda path: anndata.AnnData(X[:2], obs=pd.DataFrame({"sample_id": c["train_ids"][:2]}, index=["x0", "x1"])))
    monkeypatch.setattr(B, "counts_per_side", lambda pair, unit, c_: (single, bulk, genes))
    monkeypatch.setattr(B, "fm_full_file", lambda pair, model, what: fm(what))
    monkeypatch.setattr(B.pd, "read_csv", lambda path, **k: panel if path.name == "panel.csv" else read_csv(path, **k))
    monkeypatch.setattr(B.manifest, "temporary", lambda path: path)
    monkeypatch.setattr(B.manifest, "finish", lambda *a, **k: None)


@pytest.mark.parametrize("unit", [("crossfit", 0), ("all", 0)])
def test_every_family_returns_its_kinds(tmp_path, monkeypatch, unit):
    """Every job jobs() lists (all ten configurations; the reference rows are fold_zero_tree's) through B.run: the tables a
    compute function returns are exactly kinds() of its family and name, none of them empty, and each passes its schema.
    The alternative read-outs: heads for the elastic net only (the smoke run's pseudobulk/pls failure)."""
    B = stage08b()
    every_family_fixture(B, monkeypatch, tmp_path)
    jobs = [(family, name, compute) for family, name, compute, _ in B.jobs("soundlife", unit) if family != "reference_rows"]
    assert {family for family, _, _ in jobs} == {"permuted", "learning_curve", "bag_size", "own_recipe", "one_type", "per_sample_route", "mse",
                                                 "search_candidates", "thinned", "observed_library", "outside_moments", "single_cell_moments",
                                                 "alternative_readouts", "numeric_age", "native_head", "feature_variants"}
    assert {name for family, name, _ in jobs if family == "alternative_readouts"} == {f"{k}/{w}" for k in B.KEYS for w in C.ALTERNATIVE_READOUTS}
    for family, name, compute in jobs:
        B.run("soundlife", unit, family, name, compute, {}, {}, [])
        written = {p.name for p in B.outputs("soundlife", unit, family, name)}
        assert all((B.paths.output(B.STAGE, "soundlife", family, unit) / n).exists() for n in written), f"{family} {name}"
    for which in C.ALTERNATIVE_READOUTS:
        assert set(B.alternative_readouts("soundlife", unit, f"pseudobulk/{which}")) == ({"predictions", "heads"} if which == "elasticnet" else {"predictions"})
    within = ("predictions",) if unit[0] == "crossfit" else ("predictions", "heads")
    assert B.kinds(unit, "one_type", "pseudobulk_within") == B.kinds(unit, "per_sample_route", "scrat_within") == within


# ---- run records under the smoke run's layout (the smoke run's tms known_signature failure) ----

def test_job_records_under_the_smoke_layout(tmp_path, monkeypatch):
    """Every reference-row job's record (the only jobs listing raw downloads: tms's ortholog tables) is built with
    OUTPUTS = root/smoke and the raw downloads under root/01_download, as the smoke run lays them out, and comes out the
    same as under the full run's layout (both under root): upstream outputs by their manifests' content hashes, raw
    downloads by their pinned SHA-256 (they have no manifests)."""
    from sc2bulk import training_inputs
    from sc2bulk.config import sources
    B = stage08b()
    monkeypatch.setattr(training_inputs, "encoder_seeds", lambda pair, row: list(range(10)))
    monkeypatch.setattr(B, "untrained_selection", lambda pair, unit: B.paths.output("05_search", pair, "cytoset", unit) / "selection.json")
    monkeypatch.setattr(B.manifest, "read", lambda path: (path.is_relative_to(settings.OUTPUTS) or pytest.fail(f"{path}: no manifest"))
                        and {"content_hash": path.name})
    c = {"train_ids": ["s0"], "bulk_ids": ["b0"]}
    records = {}
    for layout, outputs in (("smoke", tmp_path / "smoke"), ("full", tmp_path)):
        monkeypatch.setattr(settings, "OUTPUTS", outputs)
        monkeypatch.setattr(settings, "RAW", tmp_path / "01_download")
        monkeypatch.setattr(B.folds, "SAMPLES", outputs / "02_preprocess")
        for pair in ("hammond", "tms", "lupus"):
            for name, _, extra in B.reference_jobs(pair, ("all", 0)):
                records[layout, pair, name] = B.record_for(pair, ("all", 0), "reference_rows", name, extra, "code", c)[0]
    assert {k[1:] for k in records if k[0] == "smoke"} == {k[1:] for k in records if k[0] == "full"}
    assert all(records["smoke", pair, name] == records["full", pair, name] for _, pair, name in records)
    ensembl = {f"ensembl/{n}": sources.DOWNLOADS[("ensembl", n)][1] for n in (*sources.ORTHOLOG_FILES.values(), sources.HUMAN_GTF)}
    for (layout, pair, name), record in records.items():
        signature = pair == "tms" and name.startswith("known_signature")
        assert record["raw_inputs"] == (ensembl if signature else {}), f"{pair} {name}"
        assert not any(k.startswith("01_download") or k.startswith("..") for k in record["inputs"]), f"{pair} {name}"
