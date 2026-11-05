"""Stage 09: the alignment methods, the stacked scaler's matrix, the silhouettes and coordinates, and the
driver on a toy unit. The read-out's scaling="stacked" is tested in test_alignment.py; the fold leak and the reproduction
of stage 08's predictions are run-level checks (tests/run_level/test_fold_leak09.py, check09.py)."""
import inspect
import runpy

import anndata
import harmonypy
import numpy as np
import pandas as pd
import pytest
import scanpy
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from sc2bulk import alignment, folds, manifest, readout, schemas
from sc2bulk.config import alignment as A
from sc2bulk.config import settings
from sc2bulk.config.pairs import PAIR_ORDER

D = runpy.run_path(str(settings.REPO / "pipeline" / "09_alignment.py"), run_name="stage09")
METHODS = ("combat", "coral", "harmony")


def two_sides(seed=0, n_sc=40, n_bulk=30, p=8, shift=2.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_sc, p)), rng.normal(shift, 1.5, size=(n_bulk, p))


# 1. ComBat

def test_combat_is_scanpys_pooled_combat_by_side():
    X_sc, X_bulk = two_sides()
    Z_sc, Z_bulk = alignment.align("combat", X_sc, X_bulk)
    stacked = np.vstack([X_sc, X_bulk])
    data = anndata.AnnData(stacked.copy(), obs=pd.DataFrame({"batch": ["x"] * 40 + ["y"] * 30}, index=[str(i) for i in range(70)]))
    scanpy.pp.combat(data, key="batch")   # wiring only: the side key, the row order, no covariate
    assert np.allclose(np.vstack([Z_sc, Z_bulk]), data.X)


def test_combat_closes_a_per_feature_shift():
    rng = np.random.default_rng(1)
    X_sc = rng.normal(size=(40, 200))
    X_bulk = rng.normal(size=(35, 200)) + rng.normal(0, 3, size=200)
    Z_sc, Z_bulk = alignment.align("combat", X_sc, X_bulk)
    before = np.abs(X_sc.mean(axis=0) - X_bulk.mean(axis=0))
    after = np.abs(Z_sc.mean(axis=0) - Z_bulk.mean(axis=0))
    assert after.mean() < 0.05 * before.mean()   # over features: EB shrinkage leaves small residuals where the shift is near 0


def test_combat_keeps_constant_features_finite():
    X_sc, X_bulk = two_sides(2)
    X_sc[:, 0], X_bulk[:, 0] = 3.0, 3.0      # constant over both sides
    X_sc[:, 1] = 1.0                          # constant within one side (collapsed embedding dimensions)
    Z_sc, Z_bulk = alignment.align("combat", X_sc, X_bulk)
    assert np.isfinite(Z_sc).all() and np.isfinite(Z_bulk).all()


# 2. CORAL

def test_coral_recolors_the_single_cell_side_exactly():
    rng = np.random.default_rng(3)
    for n_sc, d in ((60, 5), (12, 30)):        # full rank, and rank-deficient (n < d, as pseudobulk)
        X_sc = rng.normal(size=(n_sc, d)) @ rng.normal(size=(d, d))
        X_bulk = rng.normal(1.0, 2.0, size=(50, d)) @ rng.normal(size=(d, d))
        Z_sc, Z_bulk = alignment.align("coral", X_sc, X_bulk)
        assert np.array_equal(Z_bulk, X_bulk)                                        # bulk unchanged
        assert np.allclose(Z_sc.mean(axis=0), X_bulk.mean(axis=0))                   # the bulk mean exactly
        lam = A.CORAL_SHRINKAGE
        C_s = np.cov(X_sc, rowvar=False) + lam * np.eye(d)
        root_t = alignment.matrix_power(np.cov(X_bulk, rowvar=False) + lam * np.eye(d), 0.5)
        expected = root_t @ (np.eye(d) - lam * np.linalg.inv(C_s)) @ root_t        # holds at any rank
        assert np.allclose(np.cov(Z_sc, rowvar=False), expected, atol=1e-6 * np.abs(expected).max())


def test_coral_shrinkage_comes_from_config(monkeypatch):
    X_sc, X_bulk = two_sides(4, n_sc=10, p=20)
    default = alignment.align("coral", X_sc, X_bulk)[0]
    monkeypatch.setattr(A, "CORAL_SHRINKAGE", 1.0)
    assert not np.allclose(alignment.align("coral", X_sc, X_bulk)[0], default)


# 3. Harmony

def test_harmony_cluster_rule():
    assert [alignment.n_clusters(n) for n in (28, 75, 79, 3100)] == [2, 2, 3, 100]   # floor 2; Python round (75 -> 2)


def test_harmony_seed_is_a_signed_32_bit_int():
    for pair in PAIR_ORDER:
        for unit in folds.units(pair):
            assert 0 <= D["harmony_seed"](pair, unit) < 2 ** 31


def test_harmony_needs_a_seed():
    with pytest.raises(AssertionError):
        alignment.align("harmony", *two_sides(5, n_sc=20, n_bulk=10))


def test_harmony_is_seeded_shaped_and_runs_below_45_rows():
    X_sc, X_bulk = two_sides(5, n_sc=20, n_bulk=10)       # 30 rows: harmonypy's own rule would crash
    a, b = alignment.align("harmony", X_sc, X_bulk, seed=7), alignment.align("harmony", X_sc, X_bulk, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[0].shape == X_sc.shape and a[1].shape == X_bulk.shape


def test_harmony_orientation_on_a_square_matrix():
    X_sc, X_bulk = two_sides(6, n_sc=20, n_bulk=20, p=40)  # n == d
    Z_sc, Z_bulk = alignment.align("harmony", X_sc, X_bulk, seed=3)
    stacked = np.vstack([X_sc, X_bulk])
    side = pd.DataFrame({"side": ["single_cell"] * 20 + ["bulk"] * 20})
    direct = harmonypy.run_harmony(stacked.T, side, ["side"], theta=2.0, sigma=0.1, nclust=2, random_state=3, verbose=False)
    assert np.array_equal(np.vstack([Z_sc, Z_bulk]), np.asarray(direct.Z_corr))


# 4. The stacked scaler's matrix

def test_stacked_matrix_is_the_readouts():
    X_sc, X_bulk = two_sides(7)
    strata_sc, strata_bulk = np.array(["a", "b"] * 20), np.array(["a", "b"] * 15)
    y = np.array([0, 1, 1, 0] * 10)
    out = readout.run(X_sc, y, np.arange(40) % 5, X_bulk, strata_train=strata_sc, strata_bulk=strata_bulk, alphas=(1.0,),
                      scaling="stacked")
    moments = out["scalers"]["stacked"]
    expected = np.vstack([readout.scale(X_sc, moments, strata_sc), readout.scale(X_bulk, moments, strata_bulk)])
    assert np.array_equal(alignment.stacked_scaled(X_sc, X_bulk, strata_sc, strata_bulk), expected)


def test_unaligned_stacked_returns_its_inputs():
    X_sc, X_bulk = two_sides(8)
    Z_sc, Z_bulk = alignment.align("unaligned_stacked", X_sc, X_bulk)
    assert np.array_equal(Z_sc, X_sc) and np.array_equal(Z_bulk, X_bulk)


def test_per_side_matrix_is_the_readouts():
    X_sc, X_bulk = two_sides(9)
    out = readout.run(X_sc, np.array([0, 1] * 20), np.arange(40) % 5, X_bulk, alphas=(1.0,))
    expected = np.vstack([readout.scale(X_sc, out["scalers"]["single_cell"]), readout.scale(X_bulk, out["scalers"]["bulk"])])
    assert np.array_equal(alignment.per_side_scaled(X_sc, X_bulk), expected)


# 5. Strata (mouse organs): align, the stacked scaler and the silhouette

def organs(seed=10):
    X_sc, X_bulk = two_sides(seed, n_sc=40, n_bulk=24)
    return X_sc, X_bulk, np.repeat(["a", "b"], 20), np.repeat(["a", "b"], 12)


def test_another_organ_leaves_an_organs_rows_unchanged():
    X_sc, X_bulk, strata_sc, strata_bulk = organs()
    Y_sc, Y_bulk = X_sc.copy(), X_bulk.copy()
    rng = np.random.default_rng(0)
    Y_sc[strata_sc == "b"] = rng.normal(5, 3, size=(20, X_sc.shape[1]))
    Y_bulk[strata_bulk == "b"] = rng.normal(-5, 3, size=(12, X_sc.shape[1]))
    a_rows = np.concatenate([strata_sc, strata_bulk]) == "a"
    for method in ("unaligned_stacked", *METHODS):
        x = alignment.align(method, X_sc, X_bulk, strata_sc, strata_bulk, seed=1)
        y = alignment.align(method, Y_sc, Y_bulk, strata_sc, strata_bulk, seed=1)
        assert np.array_equal(x[0][strata_sc == "a"], y[0][strata_sc == "a"]) and np.array_equal(x[1][strata_bulk == "a"], y[1][strata_bulk == "a"])
        x, y = alignment.stacked_scaled(*x, strata_sc, strata_bulk), alignment.stacked_scaled(*y, strata_sc, strata_bulk)
        assert np.array_equal(x[a_rows], y[a_rows]), method


def test_within_organ_silhouette_is_sample_weighted():
    X_sc, X_bulk, strata_sc, strata_bulk = organs(11)
    keep = (strata_sc == "a") | (np.arange(40) % 2 == 0)        # organ a: 20 + 12 rows; organ b: 10 + 12
    Z, strata = np.vstack([X_sc[keep], X_bulk]), np.concatenate([strata_sc[keep], strata_bulk])
    side = alignment.sides(keep.sum(), len(X_bulk))
    expected = 0.0
    for s in ("a", "b"):
        rows = strata == s
        k = min(A.SILHOUETTE_PCS, rows.sum() - 1, Z.shape[1])
        expected += silhouette_score(PCA(n_components=k, svd_solver="full").fit_transform(Z[rows]), side[rows]) * rows.sum() / len(Z)
    assert np.isclose(alignment.silhouette(Z, side, strata), expected)


def test_a_stratum_with_too_few_rows_on_a_side_fails():
    X_sc, X_bulk, strata_sc, strata_bulk = organs(12)
    strata_bulk = strata_bulk.copy()
    strata_bulk[:] = "a"
    strata_bulk[0] = "b"                                          # organ b: one bulk row
    for method in ("unaligned_stacked", *METHODS):
        with pytest.raises(AssertionError):
            alignment.align(method, X_sc, X_bulk, strata_sc, strata_bulk, seed=1)
    with pytest.raises(AssertionError):
        alignment.align("combat", X_sc, X_bulk, strata_sc, None)  # strata on one side only


# 6. Label-freeness: nothing in the alignment takes a label

def test_alignment_takes_no_label():
    for function in (alignment.align, alignment.combat, alignment.coral, alignment.harmony, alignment.silhouette,
                     alignment.coordinates, alignment.stacked_scaled, alignment.per_side_scaled):
        assert not {"y", "group", "groups", "label", "labels"} & set(inspect.signature(function).parameters), function


# 9. Silhouette and coordinates

def test_silhouette_known_values_and_components():
    rng = np.random.default_rng(13)
    side = alignment.sides(30, 30)
    apart = np.vstack([rng.normal(0, 0.01, size=(30, 20)), rng.normal(100, 0.01, size=(30, 20))])
    assert alignment.silhouette(apart, side) > 0.99
    mixed = rng.normal(size=(60, 20))
    assert np.isclose(alignment.silhouette(mixed, side),
                      silhouette_score(PCA(n_components=10, svd_solver="full").fit_transform(mixed), side))
    assert [alignment.components(rng.normal(size=(n, d)), 10).shape[1] for n, d in ((5, 20), (50, 3), (50, 40))] == [4, 3, 10]
    X = rng.normal(size=(600, 1000))                              # sklearn's "auto" would pick the randomized solver here
    assert np.array_equal(alignment.components(X, 10), PCA(n_components=10, svd_solver="full").fit_transform(X))
    assert alignment.coordinates(mixed).shape == (60, 2)


# 10, 11 and B1: the driver on a toy unit

def toy_unit(monkeypatch, pair):
    """Stage 08's readers replaced by a toy unit of 40 training and 24 bulk samples (two organs for mouse
    organs); encoder seeds 0, 1, 10 and 11 offered (10 and 11 must be dropped)."""
    train = pd.DataFrame({"sample_id": [f"sc{i}" for i in range(40)], "group": [0, 1] * 20, "organ": np.repeat(["a", "b"], 20)})
    bulk = pd.DataFrame({"sample_id": [f"b{i}" for i in range(24)], "organ": np.repeat(["a", "b"], 12)})
    bulk["scoring_unit_id"] = bulk["sample_id"]

    def matrices(e):
        X_sc, X_bulk = two_sides(100 + e, n_sc=40, n_bulk=24, p=6)
        return X_sc + np.array([0, 1] * 20)[:, None] * (np.arange(6) < 2), X_bulk

    S08 = D["S08"]
    monkeypatch.setitem(S08, "N_READOUT", 1)
    monkeypatch.setitem(S08, "scored", lambda pair, unit: bulk)
    monkeypatch.setitem(S08, "readout_folds", lambda pair, unit, ids, r: np.arange(len(ids)) % 5)
    monkeypatch.setitem(S08, "inputs", lambda pair, unit, key: [])
    monkeypatch.setitem(S08, "encoder_seeds", lambda pair, key: [0, 1, 10, 11])
    monkeypatch.setitem(S08, "trained_features", lambda pair, unit, key, e, t, b: matrices(e))
    monkeypatch.setitem(S08, "features", lambda pair, unit, key, t, b: [(-1, *matrices(0))])
    monkeypatch.setitem(S08, "feature_names", lambda pair, unit, key, X: [f"f{i}" for i in range(X.shape[1])])
    monkeypatch.setattr(folds, "training_samples", lambda pair, fold_set, fold: train)
    rows = manifest.settings_hash({"training": list(train["sample_id"]), "bulk": list(bulk["sample_id"])})
    monkeypatch.setattr(manifest, "read", lambda path: {"settings": {"code": "c", "inputs": {}, "rows": rows}})
    monkeypatch.setattr(manifest, "stage_code_hash", lambda *args: "c")
    monkeypatch.setattr(manifest, "input_hashes", lambda paths: {})
    return train, bulk


def frames(pair, key, unit=("all", 0)):
    made = D["tables"](pair, unit, key)
    return {kind: schemas.check(kind, schemas.with_neutral_keys(pd.concat(made[kind], ignore_index=True))) for kind in made}


@pytest.mark.parametrize("pair", ["hammond", "tms"])
def test_driver_outputs_keys_and_seeds(monkeypatch, pair):
    train, bulk = toy_unit(monkeypatch, pair)
    out = frames(pair, "cytoset_cells")
    predictions, heads, geometry = out["predictions"], out["heads"], out["alignment_geometry"]
    fit = ["alignment", "encoder_seed", "readout", "readout_seed"]
    assert len(heads.drop_duplicates(fit)) == len(predictions.drop_duplicates(fit))   # every fit has heads
    assert set(heads["side"]) == {"single_cell", "bulk"} and set(heads["gene_or_dim"]) == {f"f{i}" for i in range(6)}
    for (_, stratum), group in heads.groupby(["readout_seed", "stratum"]):   # one scaler: the same moments on both sides
        a, b = (group[group["side"] == s].sort_values(fit + ["gene_or_dim"]) for s in ("single_cell", "bulk"))
        assert np.array_equal(a["scaler_mean"], b["scaler_mean"]) and np.array_equal(a["scaler_scale"], b["scaler_scale"])
    assert heads.loc[heads["readout"] == "lasso", "converged"].notna().all()
    assert set(predictions["alignment"]) == set(A.ALIGNED) and set(predictions["side"]) == {"bulk"}
    assert set(predictions["encoder_seed"]) == {0, 1} and set(geometry["encoder_seed"]) == {0, 1}   # < N_ENCODER_SEEDS
    assert set(predictions["readout"]) == {"ridge", "lasso"}
    assert (predictions["n_cells"] > 0).all() and set(predictions["variant"]) == {"primary"}
    assert set(geometry["alignment"]) == {"none", *A.ALIGNED} and set(geometry["readout"]) == {"none"}
    assert set(geometry.loc[geometry["measure"] == "silhouette_raw", "alignment"]) == {"none"}
    assert set(geometry.loc[geometry["measure"] == "harmony_relative_change", "alignment"]) == {"harmony"}
    pcs = geometry[geometry["measure"].isin(["pc1", "pc2"])]
    assert set(pcs["encoder_seed"]) == {A.GEOMETRY_ENCODER_SEED}
    ids = set(train["sample_id"]) | set(bulk["sample_id"])
    assert not set(train["sample_id"]) & set(bulk["sample_id"])
    for _, group in pcs.groupby(["alignment", "measure"]):
        assert set(group["sample_id"]) == ids and len(group) == len(ids)


def test_driver_coordinates_only_for_the_geometry_set_at_fold_0(monkeypatch):
    toy_unit(monkeypatch, "soundlife")
    pseudobulk = frames("soundlife", "pseudobulk", ("crossfit", 0))["alignment_geometry"]
    assert set(pseudobulk.loc[pseudobulk["measure"] == "pc1", "encoder_seed"]) == {-1}
    assert not frames("soundlife", "pseudobulk", ("crossfit", 1))["alignment_geometry"]["measure"].isin(["pc1"]).any()
    assert not frames("soundlife", "geneformer_cells", ("crossfit", 0))["alignment_geometry"]["measure"].isin(["pc1"]).any()


def test_driver_is_deterministic(monkeypatch):
    toy_unit(monkeypatch, "tms")
    a, b = frames("tms", "pseudobulk"), frames("tms", "pseudobulk")
    for kind in a:
        pd.testing.assert_frame_equal(a[kind], b[kind])


def test_driver_checks_stage_08s_record(monkeypatch):
    toy_unit(monkeypatch, "hammond")
    monkeypatch.setattr(manifest, "read", lambda path: {"settings": {"code": "old", "inputs": {}, "rows": "r"}})
    with pytest.raises(AssertionError):
        D["tables"]("hammond", ("all", 0), "pseudobulk")


def test_driver_primary_fold_set_only():
    assert D["units"]("soundlife") == [("crossfit", f) for f in range(4)]
    assert D["units"]("hammond") == [("all", 0)]
    assert {str(p.name) for p in D["outputs"]("hammond", ("all", 0), "pseudobulk")} == {"pseudobulk_predictions.parquet",
                                                                                         "pseudobulk_heads.parquet",
                                                                                         "pseudobulk_geometry.parquet"}
