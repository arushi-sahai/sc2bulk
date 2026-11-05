"""Fig 6A's functions (sc2bulk.properties): the property definitions, the ridge copy against readout.run, R², PCA-64, the scoring-unit AUROC, the
null draws, the split mean and the Fig 6 title licence."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from sc2bulk import properties, readout
from sc2bulk.config import configurations
from sc2bulk.config import properties as P
from sc2bulk.config import readout as R
from sc2bulk.config.pairs import PAIR_ORDER

LEARNED = [k for k in configurations.CONFIGURATION_ORDER if k != "pseudobulk"]


def test_depth_and_mito_are_cell_means():
    sample = np.array(["s1", "s1", "s2"])
    assert properties.depth([100, 1000, 10], sample).to_dict() == {"s1": 2.5, "s2": 1.0}
    assert properties.mito_fraction([10.0, 30.0, 0.0], sample).to_dict() == pytest.approx({"s1": 0.2, "s2": 0.0})


def test_clr_fractions_pseudocount():
    clr = properties.clr_fractions(["a", "a", "b", "x"], ["s1", "s1", "s1", "s1"], ["s1"], ["a", "b"])   # "x" isn't a kind
    expected = np.log([2.5 / 4, 1.5 / 4])
    assert np.allclose(clr.loc["s1"], expected - expected.mean())


def composition(n_a, n_b, n_c, sample):
    return ["a"] * n_a + ["b"] * n_b + ["c"] * n_c, [sample] * (n_a + n_b + n_c)


def toy_types(held_out_b=5):
    types, samples = [], []
    for i, (a, b, c) in enumerate([(50, 10, 5), (40, 20, 5), (30, 30, 8), (20, 40, 6), (10, 50, 9), (30, held_out_b, 30)]):
        t, s = composition(a, b, c, f"s{i}")
        types += t
        samples += s
    return np.array(types), np.array(samples)


def test_pc1_fit_on_training_samples_only_and_signed():
    train = [f"s{i}" for i in range(5)]
    stratum = pd.Series("all", index=[f"s{i}" for i in range(6)])
    types, samples = toy_types()
    score = properties.composition_pc1(types, samples, train, stratum)
    moved = properties.composition_pc1(*toy_types(held_out_b=90), train, stratum)
    assert np.allclose(score.loc[train], moved.loc[train])          # the held-out sample doesn't move the basis
    assert not np.isclose(score["s5"], moved["s5"])                  # it is projected
    # the sign: the loading on the most abundant training type ("a": 150 cells) is positive, so s0 (most "a") scores high
    assert score["s0"] > score["s4"]


def test_pc1_per_stratum():
    types, samples = toy_types()
    stratum = pd.Series(["x", "x", "x", "y", "y", "y"], index=[f"s{i}" for i in range(6)])
    score = properties.composition_pc1(types, samples, [f"s{i}" for i in range(6)], stratum)
    alone = properties.composition_pc1(types[np.isin(samples, ["s0", "s1", "s2"])], samples[np.isin(samples, ["s0", "s1", "s2"])],
                                       ["s0", "s1", "s2"], stratum.loc[["s0", "s1", "s2"]])
    assert np.allclose(score.loc[["s0", "s1", "s2"]], alone)        # one stratum's PC1 ignores the other's samples


def test_centering_uses_fit_side_organ_means_only():
    y_fit, y_scored = properties.centered([1.0, 3.0, 10.0, 20.0], [5.0, 100.0], ["k", "k", "l", "l"], ["k", "l"])
    assert np.allclose(y_fit, [-1, 1, -5, 5]) and np.allclose(y_scored, [3.0, 85.0])
    assert np.allclose(properties.centered([1.0, 2.0], [7.0])[1], [7.0])   # no strata: unchanged


def binary_problem(seed=0, n=40, p=30):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    folds = np.arange(n) % 5
    return X, y, folds, rng.normal(size=(12, p))


def test_ridge_regression_equals_readout_run_on_a_binary_target():
    X, y, folds, X_scored = binary_problem()
    ours = properties.ridge_regression(X, y, folds, X_scored)
    theirs = readout.run(X, y, folds, None, X_scored=X_scored)
    assert ours["alpha"] == theirs["alpha"] and ours["alpha_at_boundary"] == theirs["alpha_at_boundary"]
    assert np.array_equal(ours["coef"], theirs["coef"]) and np.array_equal(ours["prediction"], theirs["scored_score"])
    strata = np.array(["k", "l"] * 20)
    ours = properties.ridge_regression(X, y, folds, X_scored, strata, np.array(["k", "l"] * 6))
    theirs = readout.run(X, y, folds, None, strata_train=strata, X_scored=X_scored, strata_scored=np.array(["k", "l"] * 6))
    assert np.array_equal(ours["prediction"], theirs["scored_score"])


def test_r2_matches_sklearn_and_broadcasts():
    rng = np.random.default_rng(1)
    y, prediction = rng.normal(size=20), rng.normal(size=(3, 20))
    assert np.allclose(properties.r2(y, prediction), [r2_score(y, p) for p in prediction])
    assert properties.r2(y, prediction[0]) < 0                    # held-out R² can be negative


def test_pca64_components_and_fit_rows_only():
    rng = np.random.default_rng(2)
    for n_fit, k in ((30, 29), (100, 64)):
        X_fit, X_scored = rng.normal(size=(n_fit, 80)), rng.normal(size=(10, 80))
        Z_fit, Z_scored, components = properties.pca64(X_fit, X_scored)
        assert components == k and Z_fit.shape == (n_fit, k) and Z_scored.shape == (10, k)
        order = rng.permutation(10)
        Z_fit2, Z_scored2, _ = properties.pca64(X_fit, X_scored[order])   # held-out rows change no component
        assert np.array_equal(Z_fit, Z_fit2) and np.allclose(Z_scored[order], Z_scored2)


def test_mean_log_cells_per_sample():
    X = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 5.0]])
    assert np.array_equal(properties.mean_log_cells(X, ["a", "a", "b"], ["b", "a"]), [[5.0, 5.0], [2.0, 1.0]])


def test_unit_auroc_averages_within_scoring_units():
    # donor d1 (label 1) has two samples whose mean (0.5) is below d2's (label 0) single score 0.6
    assert properties.unit_auroc([0.9, 0.1, 0.6, 0.0], [1, 1, 0, 0], ["d1", "d1", "d2", "d3"]) == 0.5
    with pytest.raises(AssertionError):
        properties.unit_auroc([0.1, 0.2], [0, 1], ["d1", "d1"])


def test_null_draws():
    strata = np.array(["k", "k", "k", "l", "l"])
    a = properties.within_strata_permutations(strata, 50, np.random.default_rng(3))
    assert np.array_equal(a, properties.within_strata_permutations(strata, 50, np.random.default_rng(3)))   # seeded
    assert all(set(row[:3]) == {0, 1, 2} and set(row[3:]) == {3, 4} for row in a)                           # within strata
    group, label = np.array(["g1", "g1", "g2", "g3", "g3", "g4"]), np.array([1, 1, 0, 1, 1, 0])
    labels = properties.group_label_permutations(group, label, 50, np.random.default_rng(4))
    assert all(row[0] == row[1] and row[3] == row[4] for row in labels)                  # a group's samples move together
    assert all(sorted(row[[0, 2, 3, 5]]) == [0, 0, 1, 1] for row in labels)              # class counts (in groups) kept
    assert len({tuple(row) for row in labels}) > 1
    # aggregated exactly as the observed statistic: identity permutations give the observed mean over read-out seeds
    rng = np.random.default_rng(5)
    y, prediction = rng.normal(size=5), rng.normal(size=(4, 5))
    identity = np.tile(np.arange(5), (3, 1))
    assert np.allclose(properties.null_r2(y, prediction, identity), properties.r2(y, prediction).mean())
    scores = rng.normal(size=(2, 6))
    assert np.allclose(properties.null_auroc(scores, np.tile(label, (2, 1)), group),
                       np.mean([properties.unit_auroc(s, label, group) for s in scores]))


def test_split_mean_over_readout_seeds_then_fits():
    rows = [{"pair": "soundlife", "key": "pseudobulk", "property": "depth", "fold": f, "encoder_seed": -1, "readout_seed": r,
             "cv_r2": 0.1 * f + 0.01 * r} for f in range(4) for r in range(10)]
    value = properties.split_mean(pd.DataFrame(rows))
    assert value["r2"].iloc[0] == pytest.approx(np.mean([0.1 * f + 0.045 for f in range(4)]))


def grid(exceed: dict, depth_only=("tms",), base=(0.2, 0.1), win=0.5, lose=0.05, missing=()):
    """Split-mean values for the six pairs: pseudobulk / PCA-64 at `base`; in each pair, the configurations exceed[pair][t]
    (or exceed[pair] for every property) at `win`, the others at `lose`."""
    rows = []
    for pair in PAIR_ORDER:
        for t in (("depth",) if pair in depth_only else P.TECHNICAL_PROPERTIES):
            winners = exceed.get(pair, ())
            winners = winners.get(t, ()) if isinstance(winners, dict) else winners
            rows += [{"pair": pair, "key": b, "property": t, "r2": v} for b, v in zip(P.BASELINES, base)]
            rows += [{"pair": pair, "key": c, "property": t, "r2": win if c in winners else lose}
                     for c in LEARNED if (pair, c) not in missing]
    return pd.DataFrame(rows)


def test_licence_majority_of_present_configurations():
    five, four = LEARNED[:5], LEARNED[:4]
    result = properties.licence(grid({p: five for p in PAIR_ORDER[:4]}), LEARNED)
    assert result["licensed"] and result["qualifying_pairs"] == list(PAIR_ORDER[:4]) and result["title"] == P.LICENSED_TITLE
    assert result["per_pair"]["tms"] == {"depth": 5} and result["per_pair"]["pineda"] == {"depth": 0, "mito_fraction": 0}
    assert not properties.licence(grid({p: four for p in PAIR_ORDER}), LEARNED)["licensed"]      # 4 of 9 is no majority
    cut = LEARNED[:8]                                                                              # one configuration cut: 5 of 8
    assert properties.licence(grid({p: five for p in PAIR_ORDER[:4]}), cut)["licensed"]
    assert not properties.licence(grid({p: four for p in PAIR_ORDER[:4]}), cut)["licensed"]       # 4 of 8 isn't > 4


def test_licence_three_versus_four_pairs_and_missing_pairs():
    five = LEARNED[:5]
    result = properties.licence(grid({p: five for p in PAIR_ORDER[:3]}), LEARNED)
    assert not result["licensed"] and len(result["qualifying_pairs"]) == 3 and result["title"] == P.DESCRIPTIVE_TITLE
    values = grid({p: five for p in PAIR_ORDER})
    values = values[~values["pair"].isin(PAIR_ORDER[:3])]                                           # three pairs without values
    result = properties.licence(values, LEARNED)
    assert result["qualifying_pairs"] == list(PAIR_ORDER[3:]) and not result["licensed"]


def test_licence_every_technical_property_but_not_the_same_configurations():
    five, other_five = LEARNED[:5], LEARNED[4:]
    # every pair: depth won by one five, mito by another; the pair qualifies (each property counted on its own)
    split = {p: {"depth": five, "mito_fraction": other_five} for p in PAIR_ORDER}
    result = properties.licence(grid(split), LEARNED)
    assert result["licensed"] and len(result["qualifying_pairs"]) == 6
    assert result["per_configuration"][LEARNED[0]] == ["tms"]           # exceeds on every property only where depth is all
    # mito lost in every pair with mito: only mouse organs (depth only) qualifies
    lost = {p: {"depth": five} for p in PAIR_ORDER}
    assert properties.licence(grid(lost), LEARNED)["qualifying_pairs"] == ["tms"]


def test_licence_different_configurations_qualify_in_different_pairs():
    # a different majority wins in each of four pairs: the class claim is licensed, though no single configuration
    # exceeds in >= 4 pairs (the rejected same-configuration rule would not license it)
    rotated = {pair: [LEARNED[(5 * i + k) % 9] for k in range(5)] for i, pair in enumerate(PAIR_ORDER[:4])}
    result = properties.licence(grid(rotated), LEARNED)
    assert result["licensed"] and result["qualifying_pairs"] == list(PAIR_ORDER[:4])
    assert result["named"] == [] and max(len(p) for p in result["per_configuration"].values()) == 3
    assert result["per_configuration"][LEARNED[0]] != result["per_configuration"][LEARNED[2]]   # they qualify in different pairs
    winners = {pair: [LEARNED[(i + k) % 9] for k in range(5)] for i, pair in enumerate(PAIR_ORDER)}
    # the overlapping rotation: some configurations reach >= 4 pairs and every such one is named
    result = properties.licence(grid(winners), LEARNED)
    named = [c for c in LEARNED if len(result["per_configuration"][c]) >= P.LICENCE_MIN_PAIRS]
    assert result["licensed"] and result["named"] == named and len(named) >= 2


def test_licence_ties_floor_and_missing_configurations():
    bar = 0.2
    at_tie = grid({p: LEARNED[:5] for p in PAIR_ORDER}, win=bar + 1e-13)        # within the tolerance: no exceed
    assert properties.licence(at_tie, LEARNED)["qualifying_pairs"] == []
    above = grid({p: LEARNED[:5] for p in PAIR_ORDER}, win=bar + 1e-11)
    assert len(properties.licence(above, LEARNED)["qualifying_pairs"]) == 6
    # all rows negative, the learned ones above both baselines: R² > 0 is required, so nothing exceeds
    negative = grid({p: LEARNED[:5] for p in PAIR_ORDER}, base=(-0.3, -0.4), win=-0.05, lose=-0.5)
    result = properties.licence(negative, LEARNED)
    assert not result["licensed"] and all(v == 0 for counts in result["per_pair"].values() for v in counts.values())
    # a configuration with no value doesn't exceed: 5 winners with one missing leave 4
    missing = {(p, LEARNED[0]) for p in PAIR_ORDER}
    assert properties.licence(grid({p: LEARNED[:5] for p in PAIR_ORDER}, missing=missing), LEARNED)["qualifying_pairs"] == []


def test_licence_applicability_is_pre_declared_not_read_from_the_rows():
    # a pair outside MITO_NA_PAIRS whose mito rows are missing fails loudly rather than being judged on depth alone
    values = grid({p: LEARNED[:5] for p in PAIR_ORDER})
    values = values[~((values["pair"] == "pineda") & (values["property"] == "mito_fraction"))]
    with pytest.raises(AssertionError, match="baseline"):
        properties.licence(values, LEARNED)


# ---- stage 12a on toy units (pipeline/12a_properties.py) ----

import importlib.util
import re
import types

import anndata

from sc2bulk import folds, training_inputs
from sc2bulk.config import settings

N_E, N_R = 2, 2


def driver():
    spec = importlib.util.spec_from_file_location("stage12a", settings.REPO / "pipeline" / "12a_properties.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def toy_obs(samples, organ_types, annotation, mito):
    rng = np.random.default_rng(len(samples))
    rows = []
    for _, s in samples.iterrows():
        kinds = organ_types[s.get("organ", "all")]
        for c in range(30):
            rows.append({"sample_id": s["sample_id"], "total_counts": float(rng.integers(500, 5000)),
                         "percent_mito": float(rng.uniform(0, 10)) if mito else 0.0, annotation: kinds[rng.integers(len(kinds))]})
    obs = pd.DataFrame(rows)
    return anndata.AnnData(obs=obs.set_axis([f"c{i}" for i in range(len(obs))]))


class Rows:
    """A stand-in for a backed stage 07 file: file[ids].to_memory().X."""
    def __init__(self, X): self.X = X
    def __getitem__(self, ids): return types.SimpleNamespace(to_memory=lambda: Rows(toy_features("fm_profile", ids)))


def toy_features(name, ids, width=6):
    return np.vstack([np.random.default_rng(abs(hash((name, s))) % 2**32).normal(size=width) for s in ids])


def patched_12a(monkeypatch, train, held, sides, annotation, organ_types, mito):
    M = driver()
    monkeypatch.setattr(M, "N_ENCODER", N_E)
    monkeypatch.setattr(M, "N_READOUT", N_R)
    monkeypatch.setattr(P, "N_NULL", 20)
    monkeypatch.setattr(R, "RIDGE_ALPHAS", (0.1, 1.0, 10.0, 100.0))   # a short grid keeps the toy quick
    monkeypatch.setattr(folds, "training_samples", lambda pair, fold_set, fold: train.copy())
    monkeypatch.setattr(folds, "training_cells", lambda pair, fold_set, fold: toy_obs(train, organ_types, annotation, mito))
    monkeypatch.setattr(folds, "scored_cells", lambda pair, fold_set, fold: toy_obs(held, organ_types, annotation, mito))
    monkeypatch.setattr(training_inputs, "encoder_sides", lambda pair, unit, e: sides[e])
    monkeypatch.setattr(training_inputs, "pseudobulk_log", lambda pair, unit: dict(zip(train["sample_id"], toy_features("pb", train["sample_id"], 20))))
    M.S["heldout_samples"] = lambda pair, unit: held.assign(scoring_unit_id=held["donor_id"])
    M.S["readout_folds"] = lambda pair, unit, ids, r, context: (np.arange(len(ids)) + r) % 5
    M.S["fm_cells"] = lambda pair, unit, model, ids: model
    M.S["bag_means"] = lambda model, ids, pair, unit: toy_features(model, ids)
    M.S["fm_file"] = lambda pair, unit, model, what: Rows(None)
    M.S["heldout_features"] = lambda pair, unit, key, e, ids: toy_features("pb" if key == "pseudobulk" else key, ids, 20 if key == "pseudobulk" else 6)

    def trained_embedding(pair, unit, key, e):
        ids = list(train["sample_id"]) + list(held["sample_id"])
        side = [("single_cell_train" if sides[e].get(s, "train") == "train" else "single_cell_validation") if s in set(train["sample_id"])
                else "single_cell_heldout" for s in ids] + ["bulk", "bulk"]
        obs = pd.DataFrame({"side": side, "sample_id": ids + ["b0", "b1"]})
        return obs.assign(row=np.arange(len(obs))), toy_features(f"{key}{e}", ids + ["b0", "b1"])
    M.S["trained_embedding"] = trained_embedding
    monkeypatch.setattr(M, "did_train", lambda pair, unit, key, e: True if key in M.TRAINED and key != "scvi_cells" else None)
    scores = lambda key, e, ids: np.vstack([toy_features(f"score{key}{e}{r}", ids, 1).ravel() for r in range(N_R)])
    monkeypatch.setattr(M, "within_scores", lambda pair, unit, key, e, ids: scores(key, e, ids))
    return M, scores


def tms_toy():
    mice = [f"m{i:02d}" for i in range(12)]
    train = pd.DataFrame([{"sample_id": f"{m}_{o}", "donor_id": m, "holdout_group": m, "group": i % 2, "organ": o}
                          for i, m in enumerate(mice) for o in ("Lung", "Kidney")])
    sides = {e: {s: "validation" if s.split("_")[0] in mice[4 * e:4 * e + 4] else "train" for s in train["sample_id"]} for e in range(N_E)}
    return train, train.iloc[:0], sides


def test_stage12a_encoder_split_unit(monkeypatch):
    train, held, sides = tms_toy()
    M, scores = patched_12a(monkeypatch, train, held, sides, "cell_type", {"Lung": ["a", "b", "c"], "Kidney": ["d", "e", "f"]}, mito=False)
    out = M.unit_tables("tms", ("all", 0))
    table = out["samples"]
    assert table["mito_fraction"].isna().all() and table["composition_pc1"].notna().all()   # mouse organs: mito n/a
    for key in M.ROWS:
        rows = pd.DataFrame(out["rows"][key])
        assert set(rows["property"]) == {"depth", "composition_pc1", "phenotype"}
        assert len(rows) == N_E * N_R * 3 and set(rows["encoder_seed"]) == set(range(N_E))   # Fig 2B's encoder_seed e
        assert set(rows["n_scored"]) == {8} and set(rows["n_fit"]) == {16}
    pca = pd.DataFrame(out["rows"]["pseudobulk_pca64"])
    assert set(pca["n_components"]) == {15} and set(pca["row_kind"]) == {"reference"}
    # the phenotype column is Fig 2B's AUROC of stage 08's within predictions (no refit)
    first = pd.DataFrame(out["rows"]["cytoset_cells"]).query("property == 'phenotype' and encoder_seed == 0").sort_values("readout_seed")
    ids = [s for s in train["sample_id"] if sides[0][s] == "validation"]
    expected = [properties.unit_auroc(scores("cytoset_cells", 0, ids)[r], table.loc[ids, "group"], ids) for r in range(N_R)]
    assert np.allclose(first["cv_r2"], expected)
    assert all(len(v) == P.N_NULL for v in out["null"].values()) and ("scvi_cells", "depth") in out["null"]
    again = M.unit_tables("tms", ("all", 0))                                                  # deterministic
    assert all(pd.DataFrame(out["rows"][k]).equals(pd.DataFrame(again["rows"][k])) for k in M.ROWS)
    assert all(np.array_equal(out["null"][k], again["null"][k]) for k in out["null"])


def test_stage12a_crossfit_unit_and_outputs(tmp_path, monkeypatch):
    donors = [f"d{i:02d}" for i in range(16)]
    samples = pd.DataFrame([{"sample_id": f"S{d}", "donor_id": d, "holdout_group": d, "group": i % 2} for i, d in enumerate(donors)])
    train, held = samples.iloc[:12].reset_index(drop=True), samples.iloc[12:].reset_index(drop=True)
    sides = {e: {s: "validation" if i < 4 else "train" for i, s in enumerate(train["sample_id"])} for e in range(N_E)}
    M, _ = patched_12a(monkeypatch, train, held, sides, "AIFI_L2", {"all": ["T", "B", "NK", "Mono"]}, mito=True)
    out = M.unit_tables("soundlife", ("crossfit", 0))
    assert set(out["samples"]["side"]) == {"train", "heldout"} and out["samples"]["mito_fraction"].notna().all()
    for key in M.ROWS:
        rows = pd.DataFrame(out["rows"][key])
        seeds_used = set(range(N_E)) if key in M.TRAINED else {-1}                           # fixed rows: one fit per fold
        assert set(rows["encoder_seed"]) == seeds_used and set(rows["n_scored"]) == {4} and set(rows["n_fit"]) == {12}
        assert set(rows["property"]) == {"depth", "mito_fraction", "composition_pc1", "phenotype"}
    monkeypatch.setattr(M, "primary_units", lambda pair: [("crossfit", 0)])
    from sc2bulk import manifest, paths, schemas
    monkeypatch.setattr(paths, "output", lambda stage, pair, name, unit=None: tmp_path / stage / name)
    monkeypatch.setattr(manifest, "finish", lambda p, r, c, seed=None, inputs=(): manifest.temporary(p).replace(p))
    monkeypatch.setattr(M, "inputs", lambda pair, units: [])
    M.run("soundlife", [("crossfit", 0)], {}, {})
    written = pd.read_csv(tmp_path / M.STAGE / "sample_properties.csv")
    assert "group" not in written and len(written) == 16                                       # label-free
    r2 = schemas.read("property_r2", tmp_path / M.STAGE / "cytoset_cells_property_r2.parquet")
    assert r2.groupby("property")["null_q95"].nunique().eq(1).all()                           # one pair-level null per property
    draws = pd.read_csv(tmp_path / M.STAGE / "null_draws.csv")
    assert len(draws) == P.N_NULL * len(M.ROWS) * 4


def test_stages_12_read_no_bulk_data():
    """Neither stage names a bulk file or stage 08's bulk-side loaders; neither imports a label module (firewall test)."""
    bulk = re.compile(r"""bulk\.h5ad|bulk_samples|bulk_profiles|scored_bulk|S\[["'](scored|features|trained_features|pseudobulk_features|fm_features|split_features|inputs)["']\]""")
    for name in ("pipeline/12_simulation.py", "pipeline/12a_properties.py", "src/sc2bulk/simulation.py", "src/sc2bulk/properties.py"):
        assert not bulk.search((settings.REPO / name).read_text()), name
