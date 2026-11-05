"""The read-out."""
import numpy as np
from sklearn.linear_model import Ridge

from sc2bulk import readout


def data(seed=0, n=40, p=6, shift=1.0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, p)) + shift * y[:, None] * (np.arange(p) < 2)
    folds = np.arange(n) % 5
    bulk = rng.normal(3.0, 2.0, size=(15, p))
    return X, y, folds, bulk


def test_per_side_scaling_and_strata():
    X, y, folds, bulk = data()
    out = readout.run(X, y, folds, bulk)
    mean, sd = out["scalers"]["bulk"]["all"]
    assert np.allclose(mean, bulk.mean(axis=0)) and np.allclose(sd, bulk.std(axis=0))   # bulk's own moments
    mean, sd = out["scalers"]["single_cell"]["all"]
    assert np.allclose(mean, X.mean(axis=0))
    strata = np.array(["a", "b"] * 20)
    moments = readout.scaler(X, strata)
    assert np.allclose(moments["a"][0], X[strata == "a"].mean(axis=0))
    Z = readout.scale(X, moments, strata)
    assert np.allclose(Z[strata == "b"].mean(axis=0), 0) and np.allclose(Z[strata == "b"].std(axis=0), 1)
    assert np.abs(readout.scale(X * 100, readout.scaler(X))).max() > 10                    # uncapped


def test_zero_and_near_zero_variance():
    X = np.column_stack([np.ones(10), np.full(10, 5.0) + 1e-15 * np.arange(10), np.arange(10.0)])
    (mean, sd), = readout.scaler(X).values()
    assert sd[0] == 1.0 and sd[1] == 1.0 and sd[2] == np.arange(10.0).std()


def test_alpha_is_grouped_cv_mse_with_ties_to_smaller():
    X, y, folds, bulk = data(1)
    alphas = (0.1, 1.0, 10.0, 100.0)
    out = readout.run(X, y, folds, bulk, alphas=alphas)
    Z = readout.scale(X, readout.scaler(X))
    mse = [np.mean([np.mean((Ridge(alpha=a).fit(Z[folds != f], y[folds != f]).predict(Z[folds == f]) - y[folds == f]) ** 2)
                    for f in range(5)]) for a in alphas]
    assert out["alpha"] == alphas[int(np.argmin(mse))]
    assert out["alpha_at_boundary"] == (out["alpha"] in (alphas[0], alphas[-1]))
    assert np.allclose(out["probability"], np.clip(out["score"], 0, 1))
    tie = readout.run(np.column_stack([np.zeros(40), np.zeros(40)]), y, folds, bulk[:, :2], alphas=alphas)
    assert tie["alpha"] == 0.1                                   # all alphas tie: the smallest


def test_reproducible_and_bulk_label_free():
    X, y, folds, bulk = data(2)
    a, b = readout.run(X, y, folds, bulk), readout.run(X, y, folds, bulk)
    assert np.array_equal(a["score"], b["score"]) and a["alpha"] == b["alpha"]
    # bulk rows are scored independently of each other except through the bulk scaler
    shuffled = readout.run(X, y, folds, bulk[::-1])
    assert np.allclose(shuffled["score"], a["score"][::-1])


def test_lasso_records_convergence():
    X, y, folds, bulk = data(3)
    out = readout.run(X, y, folds, bulk, readout="lasso", alphas=(1e-3, 1e-2, 1e-1, 1.0))
    assert out["converged"] and out["n_nonzero"] == (out["coef"] != 0).sum()


def test_uncapped_through_run_and_scored_rows():
    X, y, folds, _ = data(4)
    bulk = np.random.default_rng(4).normal(size=(200, X.shape[1]))
    bulk[0, 0] = 1e4                                              # an outlier: |z| ≈ sqrt(199) > 10 on the bulk side
    out = readout.run(X, y, folds, bulk, X_scored=X[:5])
    Z = readout.scale(bulk, out["scalers"]["bulk"])
    assert np.abs(Z).max() > 10
    assert np.allclose(out["score"], Z @ out["coef"] + out["intercept"])                       # the uncapped rows scored
    single = readout.scale(X[:5], out["scalers"]["single_cell"])
    assert np.allclose(out["scored_score"], single @ out["coef"] + out["intercept"])          # single-cell scaler


def test_float32_noise_counts_as_zero_variance():
    X = np.column_stack([np.float32(3.0) + np.float32(1e-7) * np.arange(10, dtype=np.float32), np.arange(10.0)])
    (mean, sd), = readout.scaler(X.astype(np.float64)).values()
    assert sd[0] == 1.0


def test_alpha_grid_must_ascend():
    import pytest
    X, y, folds, bulk = data(5)
    with pytest.raises(AssertionError):
        readout.run(X, y, folds, bulk, alphas=(10.0, 1.0))


def test_lasso_alpha_is_the_grouped_cv_argmin():
    from sklearn.linear_model import Lasso
    X, y, folds, bulk = data(7, n=60, p=8)
    alphas = (1e-3, 1e-2, 3e-2, 1e-1)
    out = readout.run(X, y, folds, bulk, readout="lasso", alphas=alphas)
    Z = readout.scale(X, readout.scaler(X))
    mse = [np.mean([np.mean((Lasso(alpha=a, max_iter=200_000, tol=1e-4).fit(Z[folds != f], y[folds != f])
                             .predict(Z[folds == f]) - y[folds == f]) ** 2) for f in range(5)]) for a in alphas]
    assert out["alpha"] == alphas[int(np.argmin(mse))]


def driver():
    """Stage 08's script as a module (its name starts with a digit), so tests can patch its globals."""
    import importlib.util
    from sc2bulk.config import settings
    spec = importlib.util.spec_from_file_location("stage08", settings.REPO / "pipeline" / "08_readout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scored_bulk_rows_and_scoring_units(tmp_path, monkeypatch):
    import pandas as pd
    from sc2bulk import folds
    from sc2bulk.config import settings
    G = driver()
    (tmp_path / "02_preprocess" / "soundlife").mkdir(parents=True)
    (tmp_path / "03_splits" / "soundlife").mkdir(parents=True)
    pd.DataFrame({"sample_id": ["b1", "b2", "b3", "b4"], "donor_id": ["d1", "d1", "d2", "d3"],
                  "drop_reason": [np.nan, np.nan, np.nan, "dropped"]}).to_csv(tmp_path / "02_preprocess" / "soundlife" / "bulk_samples.csv", index=False)
    pd.DataFrame({"fold_set": "crossfit", "fold": [1, 1, 1], "donor_id": ["d1", "d2", "d3"],
                  "side": ["scored", "train", "scored"]}).to_csv(tmp_path / "03_splits" / "soundlife" / "donor_folds.csv", index=False)
    monkeypatch.setattr(settings, "OUTPUTS", tmp_path)
    monkeypatch.setattr(folds, "FOLDS", tmp_path / "03_splits")
    table = G.scored("soundlife", ("crossfit", 1))
    assert list(table["sample_id"]) == ["b1", "b2"]                      # fold 1's scored, kept donors only
    assert list(table["scoring_unit_id"]) == ["d1", "d1"]                # the pair's scoring unit: donor


def fake_unit(G, monkeypatch, tmp_path, n=40, p=6, splits=None):
    """Stage 08's driver on a synthetic unit: training samples (two organs), bulk, features (fixed: one
    seed -1 set), stage 03's folds and splits as simple functions, outputs under tmp_path."""
    import types
    import pandas as pd
    rng = np.random.default_rng(0)
    organ = np.array(["kidney", "lung"] * (n // 2))
    train = pd.DataFrame({"sample_id": [f"s{i}" for i in range(n)], "donor_id": [f"d{i}" for i in range(n)],
                          "holdout_group": [f"d{i}" for i in range(n)], "group": [0, 1] * (n // 2), "organ": organ})
    bulk = pd.DataFrame({"sample_id": [f"b{i}" for i in range(12)], "organ": ["kidney", "lung"] * 6})
    bulk = bulk.assign(scoring_unit_id=bulk["sample_id"])
    X_train, X_bulk = rng.normal(size=(n, p)) + 0.8 * train["group"].to_numpy()[:, None], rng.normal(size=(12, p))
    splits = splits or {e: {s: "validation" if (i + e) % 3 == 0 else "train" for i, s in enumerate(train["sample_id"])} for e in range(2)}
    monkeypatch.setattr(G, "folds", types.SimpleNamespace(training_samples=lambda pair, *unit: train))
    monkeypatch.setattr(G, "scored", lambda pair, unit: bulk)
    monkeypatch.setattr(G, "features", lambda pair, unit, key, t, b: [(-1, X_train[[int(s[1:]) for s in t]], X_bulk)])
    monkeypatch.setattr(G, "readout_folds", lambda pair, unit, ids, r, context="training": np.arange(len(ids)) % 5)
    monkeypatch.setattr(G, "training_inputs", types.SimpleNamespace(encoder_sides=lambda pair, unit, e: splits[e],
                                                                    panel=lambda pair, unit: [f"g{i}" for i in range(p)]))
    monkeypatch.setattr(G, "N_READOUT", 2)
    monkeypatch.setattr(G, "N_ENCODER", len(splits))
    monkeypatch.setattr(G, "manifest", types.SimpleNamespace(temporary=lambda path: path, finish=lambda *a, **k: None))
    monkeypatch.setattr(G, "paths", types.SimpleNamespace(output=lambda stage, pair, name, unit: tmp_path / name))
    monkeypatch.setattr(G, "inputs", lambda *a: [])
    return train, bulk, X_train, X_bulk


def test_run_unit_keys_and_heads(tmp_path, monkeypatch):
    from sc2bulk import schemas
    G = driver()
    train, bulk, _, _ = fake_unit(G, monkeypatch, tmp_path)
    G.run("tms", ("all", 0), "geneformer_cells", "primary", {}, {})
    pred = schemas.read("predictions", tmp_path / "geneformer_cells_predictions.parquet")
    heads = schemas.read("heads", tmp_path / "geneformer_cells_heads.parquet")
    assert set(pred["sample_id"]) == set(bulk["sample_id"]) and len(pred) == 12 * 2 * 2   # rows x read-outs x seeds
    assert (pred["encoder_seed"] == -1).all() and (pred["n_cells"] == 500).all() and (pred["pooling"] == "mean").all()
    assert set(zip(heads["side"], heads["stratum"])) == {(s, o) for s in ("single_cell", "bulk") for o in ("kidney", "lung")}


def test_within_single_cell_rows(tmp_path, monkeypatch):
    from sc2bulk import readout, schemas
    G = driver()
    train, _, X_train, _ = fake_unit(G, monkeypatch, tmp_path)
    def refuse(*args):
        raise AssertionError("Fig 2B's validation rows read held-out cells")
    monkeypatch.setattr(G, "heldout_features", refuse)
    G.run("tms", ("all", 0), "geneformer_cells", "within", {}, {})
    pred = schemas.read("predictions", tmp_path / "geneformer_cells_within_predictions.parquet")
    heads = schemas.read("heads", tmp_path / "geneformer_cells_within_heads.parquet")
    assert set(pred["encoder_seed"]) == {0, 1} and (pred["side"] == "single_cell_validation").all()   # the split index, for a fixed representation
    validation = [s for i, s in enumerate(train["sample_id"]) if i % 3 == 0]                              # split 0's validation side
    first = pred[(pred["encoder_seed"] == 0) & (pred["readout"] == "ridge") & (pred["readout_seed"] == 0)]
    assert list(first["sample_id"]) == validation and list(first["scoring_unit_id"]) == validation      # tms scores samples
    fit = np.array([i % 3 != 0 for i in range(len(train))])
    out = readout.run(X_train[fit], train["group"].to_numpy()[fit], np.arange(fit.sum()) % 5, None,
                      strata_train=train["organ"].to_numpy()[fit], X_scored=X_train[~fit], strata_scored=train["organ"].to_numpy()[~fit])
    assert np.allclose(first["score"], out["scored_score"]) and "score" not in out and set(out["scalers"]) == {"single_cell"}
    assert set(heads["side"]) == {"single_cell"}


def test_out_of_fold_rows_and_guard(tmp_path, monkeypatch):
    from sc2bulk import readout, schemas, seeds
    G = driver()
    n = 40
    ids = [f"s{i}" for i in range(n)]
    splits = {0: {s: "validation" if i < 20 else "train" for i, s in enumerate(ids)},     # 10 groups of each label: passes
              1: {s: "validation" if i < 4 else "train" for i, s in enumerate(ids)},      # 2 + 2 groups: fewer than READOUT_FOLDS
              2: {s: "validation" if i < 24 else "train" for i, s in enumerate(ids)}}
    train, bulk, X_train, X_bulk = fake_unit(G, monkeypatch, tmp_path, splits=splits)
    table = G.guard("tms", ("all", 0))
    assert list(table["passes"]) == [True, False, True] and list(table["groups_label_0"]) == [10, 2, 12]
    G.run("tms", ("all", 0), "pseudobulk", "out_of_fold", {}, {})
    pred = schemas.read("predictions", tmp_path / "pseudobulk_out_of_fold_predictions.parquet")
    heads = schemas.read("heads", tmp_path / "pseudobulk_out_of_fold_heads.parquet")
    assert set(pred["encoder_seed"]) == {0, 2} and (pred["variant"] == "out_of_fold_readout").all() and (pred["side"] == "bulk").all()
    assert set(heads["variant"]) == {"out_of_fold_readout"} and set(heads["side"]) == {"single_cell", "bulk"}
    samples = train.iloc[:20]                                                            # split 0's validation side only
    folds_0 = readout.readout_folds(samples, seeds.derived("readout_folds", "tms", 0, "all", 0)).to_numpy()
    out = readout.run(X_train[:20], samples["group"].to_numpy(), folds_0, X_bulk, strata_train=samples["organ"].to_numpy(),
                      strata_bulk=bulk["organ"].to_numpy())
    first = pred[(pred["encoder_seed"] == 0) & (pred["readout"] == "ridge") & (pred["readout_seed"] == 0)]
    assert np.allclose(first["score"], out["score"])


def test_families_and_layout():
    G = driver()
    assert G.families("soundlife", ("split67", 0)) == ["primary"]                       # split67: no Fig 2B, no out-of-fold row
    names = [p.name for p in G.outputs("soundlife", ("crossfit", 1), "pseudobulk", "within")]
    assert names == ["pseudobulk_within_predictions.parquet"]                           # held-out rows apply the primary heads
    names = [p.name for p in G.outputs("tms", ("all", 0), "scvi_cells", "out_of_fold")]
    assert names == ["scvi_cells_out_of_fold_predictions.parquet", "scvi_cells_out_of_fold_heads.parquet"]
    assert [p.name for p in G.outputs("tms", ("all", 0), "scvi_cells", "primary")] == ["scvi_cells_predictions.parquet", "scvi_cells_heads.parquet"]


def test_heldout_rows_apply_the_primary_heads(tmp_path, monkeypatch):
    import pandas as pd
    from sc2bulk import readout, schemas
    G = driver()
    train, bulk, X_train, X_bulk = fake_unit(G, monkeypatch, tmp_path)
    train = train.drop(columns="organ")
    monkeypatch.setattr(G.folds, "training_samples", lambda pair, *unit: train)
    X_held = np.random.default_rng(1).normal(size=(5, X_train.shape[1]))
    held = pd.DataFrame({"sample_id": [f"h{i}" for i in range(5)], "donor_id": [f"hd{i}" for i in range(5)]})
    monkeypatch.setattr(G, "heldout_samples", lambda pair, unit: held.assign(scoring_unit_id=held["donor_id"]))
    monkeypatch.setattr(G, "heldout_features", lambda pair, unit, key, e, ids: X_held)
    calls = []
    monkeypatch.setattr(G, "features", lambda pair, unit, key, t, b: calls.append(key) or [(-1, X_train, X_bulk)])
    G.run("soundlife", ("crossfit", 1), "geneformer_pseudobulk", "primary", {}, {})
    G.run("soundlife", ("crossfit", 1), "geneformer_pseudobulk", "within", {}, {})
    assert calls == ["geneformer_pseudobulk"]                                           # Fig 2B adds no fit
    pred = schemas.read("predictions", tmp_path / "geneformer_pseudobulk_within_predictions.parquet")
    assert not (tmp_path / "geneformer_pseudobulk_within_heads.parquet").exists()
    assert (pred["side"] == "single_cell_heldout").all() and (pred["encoder_seed"] == -1).all() and len(pred) == 5 * 2 * 2
    for name in ("ridge", "lasso"):
        out = readout.run(X_train, train["group"].to_numpy(), np.arange(len(train)) % 5, X_bulk, readout=name, X_scored=X_held)
        rows = pred[(pred["readout"] == name) & (pred["readout_seed"] == 0)]
        assert np.allclose(rows["score"], out["scored_score"]) and list(rows["scoring_unit_id"]) == list(held["donor_id"])


def test_readout_folds_copy_matches_stage03():
    import importlib.util
    import pandas as pd
    from sc2bulk.config import settings
    spec = importlib.util.spec_from_file_location("stage03", settings.REPO / "pipeline" / "03_splits.py")
    stage03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage03)
    rng = np.random.default_rng(0)
    groups = [f"g{i}" for i in range(23)]
    label = dict(zip(groups, rng.integers(0, 2, len(groups))))
    samples = pd.DataFrame({"sample_id": [f"s{i}" for i in range(60)], "holdout_group": rng.choice(groups, 60)})
    samples["group"] = samples["holdout_group"].map(label)
    for seed in (0, 7, 123456):
        assert readout.readout_folds(samples, seed).equals(stage03.readout_folds(samples, seed))


def test_alpha_grid_extension():
    from sc2bulk.config import readout as R
    base = R.ridge_alphas(())
    assert base == R.RIDGE_ALPHAS and len(base) == 60 and base[0] == 1e-3 and np.isclose(base[-1], 1e4)
    both = R.ridge_alphas(("lower", "upper"))
    assert both[26:86] == base and len(both) == 112                                     # the 60 kept, unchanged
    assert np.allclose(np.diff(np.log10(both)), 7 / 59)                                 # the same log spacing
    assert np.log10(base[0]) - np.log10(both[0]) >= 3 and np.isclose(both[0], 8.2e-7, rtol=0.01)
    assert R.ridge_alphas(("upper",))[:60] == base and np.isclose(R.ridge_alphas(("upper",))[-1], 1.2e7, rtol=0.02)


def test_extended_lowest_alpha_is_the_minimum_norm_fit():
    rng = np.random.default_rng(0)
    n, p = 41, 2000
    X = rng.normal(size=(n, p))
    y = rng.integers(0, 2, n).astype(float)
    from sc2bulk.config import readout as R
    alpha = R.ridge_alphas(("lower",))[0]
    X_new = rng.normal(size=(10, p))
    score = Ridge(alpha=alpha).fit(X, y).predict(X_new)
    mean = X.mean(axis=0)
    beta = np.linalg.pinv(X - mean) @ (y - y.mean())
    assert np.isfinite(score).all() and np.allclose(score, (X_new - mean) @ beta + y.mean(), rtol=1e-6)


def stage08a():
    import importlib.util
    from sc2bulk.config import settings
    spec = importlib.util.spec_from_file_location("stage08a", settings.REPO / "pipeline" / "08a_alpha_edges.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alpha_edge_rule(tmp_path, monkeypatch):
    import json
    import sys
    import pytest
    from sc2bulk.config import configurations
    from sc2bulk.config import readout as R
    A = stage08a()
    low, high, mid = R.RIDGE_ALPHAS[0], R.RIDGE_ALPHAS[-1], R.RIDGE_ALPHAS[30]
    fits = {key: [mid] * 10 for key in configurations.CONFIGURATION_ORDER}
    fits["pseudobulk"] = [low] * 2 + [mid] * 8                                          # 20% at the lower edge, pooled over pairs
    fits["scvi_cells"] = [high] + [mid] * 9                                              # exactly 10% at the upper edge: not more
    monkeypatch.setattr(A, "counted_alphas", lambda pair, key: fits[key] if pair == "hammond" else [])
    monkeypatch.setattr(A, "DECISION", tmp_path / "decision.json")
    monkeypatch.setattr(A, "AFTER", tmp_path / "after.json")
    monkeypatch.setattr(A.manifest, "git_state", lambda: {})
    monkeypatch.setattr(sys, "argv", ["08a"])
    counts = A.edges()
    assert counts["pseudobulk"] == {"fits": 10, "lower": 2, "upper": 0, "fraction_lower": 0.2, "fraction_upper": 0.0}
    with pytest.raises(SystemExit, match="lower"):
        A.main()
    assert json.loads((tmp_path / "decision.json").read_text())["extension"] == ["lower"]
    fits["pseudobulk"] = [mid] * 10                                                      # a rerun can't change the recorded decision
    with pytest.raises(AssertionError):
        A.main()
    assert json.loads((tmp_path / "decision.json").read_text())["extension"] == ["lower"]
    fits["pseudobulk"] = [low] * 3 + [mid] * 7                                          # counts drift, same decision: recorded, not fatal
    with pytest.raises(SystemExit, match="lower"):
        A.main()
    assert json.loads((tmp_path / "after.json").read_text())["configurations"]["pseudobulk"]["lower"] == 3
    assert json.loads((tmp_path / "decision.json").read_text())["configurations"]["pseudobulk"]["lower"] == 2
    monkeypatch.setattr(R, "RIDGE_ALPHA_EXTENSION", ("upper",))                          # report mode checks the committed value
    with pytest.raises(AssertionError):
        A.main()
    monkeypatch.setattr(R, "RIDGE_ALPHA_EXTENSION", ("lower",))
    fits["pseudobulk"] = [low] * 10                                                      # report mode never decides again
    A.main()
    assert json.loads((tmp_path / "after.json").read_text())["configurations"]["pseudobulk"]["fraction_lower"] == 1.0


def test_reading_rule():
    import pandas as pd
    from sc2bulk import reading_rule
    columns = ["key", "delta", "u", "p_better_holm", "p_worse_holm"]
    primary = pd.DataFrame([["a", -0.02, 0.01, 0.9, 0.02], ["b", 0.0, 0.03, 0.5, 0.5], ["c", -0.01, 0.04, 0.8, 0.2],
                            ["d", -0.01, 0.04, 0.8, 0.2]], columns=columns)
    out_of_fold = pd.DataFrame([["a", 0.01, 0.06, 0.3, 0.6], ["b", 0.02, 0.04, 0.5, 0.5], ["c", -0.03, 0.02, 0.9, 0.04]],
                               columns=columns)
    table = reading_rule.changes(primary, out_of_fold).set_index("key")
    assert table.loc["a", ["sign", "better", "worse", "margin", "flagged"]].tolist() == [True, False, True, True, True]
    assert not table.loc["b", "flagged"]                                                 # 0 to positive is not a sign change
    assert table.loc["c", ["sign", "worse", "margin", "flagged"]].tolist() == [False, True, False, True]
    assert not table.loc["d", "estimable"] and not table.loc["d", "flagged"]             # not estimable: never a changed call


def test_trained_features_and_seeds(tmp_path, monkeypatch):
    import anndata
    import pandas as pd
    import pytest
    from sc2bulk import training_inputs
    from sc2bulk.config import settings
    G = driver()
    obs = pd.DataFrame({"side": ["bulk", "single_cell_validation", "single_cell_heldout", "single_cell_train", "bulk", "single_cell_train"],
                        "sample_id": ["b2", "s2", "h1", "s1", "b1", "s3"], "variant": ["primary", "primary", "primary", "primary",
                                                                                         "observed_library", "primary"],
                        "scoring_unit_id": ["b2", "s2", "h1", "s1", "b1", "s3"]})
    obs = pd.concat([obs, pd.DataFrame({"side": ["bulk"], "sample_id": ["b1"], "variant": ["primary"], "scoring_unit_id": ["b1"]})],
                    ignore_index=True)
    X = np.arange(len(obs) * 2, dtype=np.float32).reshape(-1, 2)
    path = tmp_path / "s0.h5ad"
    anndata.AnnData(X, obs=obs.set_index(obs.index.astype(str))).write_h5ad(path)
    monkeypatch.setattr(G, "trained_file", lambda pair, unit, key, e: path)
    import types
    monkeypatch.setattr(G, "manifest", types.SimpleNamespace(read=lambda p: {"settings": {"code": "c"}},
                                                             stage_code_hash=lambda *a, **k: "c"))
    X_train, X_bulk = G.trained_features("p", ("all", 0), "cytoset_cells", 0, ["s1", "s2", "s3"], ["b1", "b2"])
    assert np.array_equal(X_train, X[[3, 1, 5]]) and np.array_equal(X_bulk, X[[6, 0]])   # primary bulk only, in the given order
    with pytest.raises(AssertionError):
        G.trained_features("p", ("all", 0), "cytoset_cells", 0, ["s1", "s2"], ["b1", "b2"])
    monkeypatch.setattr(G, "manifest", types.SimpleNamespace(read=lambda p: {"settings": {"code": "old"}},
                                                             stage_code_hash=lambda *a, **k: "c"))
    with pytest.raises(AssertionError):                                     # a stale stage 06 embedding
        G.trained_features("p", ("all", 0), "cytoset_cells", 0, ["s1", "s2", "s3"], ["b1", "b2"])
    monkeypatch.setattr(settings, "SMOKE", False)
    assert training_inputs.encoder_seeds("soundlife", "A") == list(range(10))
    assert training_inputs.encoder_seeds("lupus", "A") == list(range(20))
    assert training_inputs.encoder_seeds("lupus", "J") == list(range(20))   # scVI and CytoSet on pseudobulk too
    assert training_inputs.encoder_seeds("lupus", "B") == list(range(20))
    assert training_inputs.encoder_seeds("soundlife", "J") == list(range(10))


def test_alpha_edge_rule_smoke_writes_nothing(tmp_path, monkeypatch):
    import sys
    from sc2bulk.config import configurations
    from sc2bulk.config import readout as R
    A = stage08a()
    fits = {key: [R.RIDGE_ALPHAS[0]] * 10 for key in configurations.CONFIGURATION_ORDER}   # every fit at the lower edge
    monkeypatch.setattr(A, "counted_alphas", lambda pair, key: fits[key] if pair == "hammond" else [])
    monkeypatch.setattr(A, "DECISION", tmp_path / "decision.json")
    monkeypatch.setattr(A, "AFTER", tmp_path / "after.json")
    monkeypatch.setattr(A.settings, "SMOKE", True)
    monkeypatch.setattr(sys, "argv", ["08a"])
    A.main()                                                                             # no exit, no record: smoke reruns stay free
    assert not list(tmp_path.iterdir())
