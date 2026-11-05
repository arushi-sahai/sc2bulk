"""Stage 06 on a toy cross-fitted unit : every row's outputs
and sides, the pseudobulk step budgets, each row's configuration difference from row A, row C's untrained
network, row D's permuted labels, row E's nested subsets, row G's mixup, row H's one-type store, row I's route,
H3's thinned bulk, the MSE check, the streams each job draws (and their disjointness across stages 05, 05s
and 06), the job list."""
import contextlib
import importlib.util
import io
import json
import runpy
import sys
import types
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from sc2bulk import harness, mixup, protocol, seeds, training_inputs
from sc2bulk.config import search_space, settings
from sc2bulk.config import training as T
from sc2bulk.encoders import cytoset
from test_harness import CONFIGS, GENES, toy

UNIT = ("crossfit", 0)
TRAIN_S, HELD_S = [f"s{i}" for i in range(20)], [f"s{i}" for i in range(20, 28)]
SIDES = {s: "validation" if int(s[1:]) < 10 else "train" for s in TRAIN_S}   # 5 + 5 validation groups


def driver():
    spec = importlib.util.spec_from_file_location("stage06", settings.REPO / "pipeline" / "06_train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def toy_samples():
    """The toy unit's training samples, as folds.training_samples gives them (group 1 older)."""
    groups = np.array([i % 2 for i in range(20)])
    return pd.DataFrame({"sample_id": TRAIN_S, "holdout_group": TRAIN_S, "group": groups,
                         "label_value": 20.0 + 50 * groups + np.arange(20), "organ": ["liver", "lung"] * 10})


def patched(tmp_path, monkeypatch):
    """The driver on the toy unit: training inputs, stage 05's selection, outputs and seeds stubbed (the row
    functions of training_inputs run for real on toy samples); every seed drawn is recorded."""
    G = driver()
    cells = toy(n_samples=28, cells=60)
    unit_cells, held = training_inputs.subset(cells, TRAIN_S), training_inputs.subset(cells, HELD_S)
    one_type = training_inputs.subset(cells, [s for s in TRAIN_S if s not in ("s18", "s19")])   # s18, s19: not in the store
    pseudobulk = {s: np.log1p(np.asarray(cells.X[cells.sample_id == s].mean(axis=0)).ravel()) for s in cells.samples.index}
    config = {**CONFIGS["cytoset"], "ncell": 500, "n_epochs": 15, "patience": 3, "objective": "bce"}
    route = {**config, "cell_normalization": "population"}
    monkeypatch.setattr(T, "FROZEN_NCELL", 500)
    monkeypatch.setattr(T, "SCRAT_PCA_COMPONENTS", 5)
    monkeypatch.setattr(T, "SCRAT_MIXUP", {"augment_num": 10, "alpha": 0.5, "min_size": 200, "noise_var": 1e-5, "bags_per_sample": 10})
    monkeypatch.setattr(T, "OWN_RECIPES", {**T.OWN_RECIPES, "scrat": {"lr": 1e-2, "pca": True, "mixup": True, "n_epochs": 14, "patience": None}})
    monkeypatch.setattr(G, "frozen", lambda pair, unit, encoder, key="frozen": dict(route if key == "per_sample_route" else config))
    monkeypatch.setattr(G, "search_space", types.SimpleNamespace(base=lambda encoder: dict(config), OUTSIDE_SEARCH=search_space.OUTSIDE_SEARCH,
                                                                 FIG4A_BAG_SIZES=search_space.FIG4A_BAG_SIZES))
    # every encoder name builds a small CytoSet (ScRAT needs its clone); a PCA input has the components' width
    monkeypatch.setattr(G, "network", lambda encoder, c, in_dim: cytoset.network(c, T.SCRAT_PCA_COMPONENTS if c.get("pca") else in_dim))
    monkeypatch.setattr(G.harness, "micro_batch_for", lambda encoder, c: c["batch_size"])
    monkeypatch.setattr(training_inputs, "folds", types.SimpleNamespace(training_samples=lambda pair, fold_set, fold: toy_samples()))
    monkeypatch.setattr(training_inputs, "encoder_sides", lambda pair, unit, e: dict(SIDES))
    drawn = []

    def derived(purpose, pair, index, fold_set="all", fold=0):
        drawn.append((purpose, index))
        return seeds.derived(purpose, pair, index, fold_set, fold)
    recorder = types.SimpleNamespace(derived=derived)
    monkeypatch.setattr(G, "seeds", recorder)
    monkeypatch.setattr(training_inputs, "seeds", recorder)

    def thinned_bulk(pair, unit, bulk_table):
        for s in bulk_table["sample_id"]:
            counts = np.random.default_rng(int(s[1:])).poisson(3.0, (2, 5, GENES))
            yield s, counts, counts.sum(axis=2) + 10
    store = {"all": unit_cells, "one_type": one_type}
    monkeypatch.setattr(G, "training_inputs", types.SimpleNamespace(
        unit_cells=lambda pair, unit, counts=False, cell_subset="all", deconvolution_types=False: store[cell_subset],
        unit_cells_all_available=lambda pair, unit, counts=False: unit_cells,
        pseudobulk_log_all_available=lambda pair, unit: {s: pseudobulk[s] + 1.0 for s in TRAIN_S},   # told apart from the stored cells'
        encoder_sides=training_inputs.encoder_sides, split_sides=lambda pair, unit, purpose, i: dict(SIDES),
        split_cells=training_inputs.split_cells,
        pseudobulk_log=lambda pair, unit, cell_subset="all": {s: pseudobulk[s] for s in store[cell_subset].samples.index
                                                              if s in set(store[cell_subset].sample_id)},
        scored_bulk=lambda pair, unit: pd.DataFrame({"sample_id": ["b0", "b1"], "scoring_unit_id": ["d0", "d1"]}),
        bulk_profiles=lambda pair, unit, ids, counts=False: protocol.Profiles(
            X=sp.csr_matrix(np.vstack([pseudobulk["s0"], pseudobulk["s1"]])), genes=unit_cells.genes,
            sample_id=np.array(list(ids)), library_size=np.ones(2)),
        training_ages=training_inputs.training_ages, encoder_seeds=training_inputs.encoder_seeds,
        permuted_groups=training_inputs.permuted_groups, learning_curve_subset=training_inputs.learning_curve_subset,
        sample_strata=training_inputs.sample_strata, thinned_bulk=thinned_bulk))
    monkeypatch.setattr(G, "scoring_units", lambda pair: {s: "D" + s for s in cells.samples.index})
    held_one_type = training_inputs.subset(cells, HELD_S[:6])
    monkeypatch.setattr(G, "heldout_cells", lambda pair, unit, counts=False, cell_subset="all": (
        held if cell_subset == "all" else held_one_type, {s: pseudobulk[s] for s in HELD_S}))
    monkeypatch.setattr(G, "paths", types.SimpleNamespace(output=lambda stage, pair, name, unit=None: tmp_path / stage / name))
    monkeypatch.setattr(G, "manifest", types.SimpleNamespace(temporary=lambda p: p.with_name(".tmp." + p.name),
                                                             finish=lambda p, r, c: p.with_name(".tmp." + p.name).replace(p)))
    return G, config, unit_cells, drawn


def run_row(G, drawn, row, model, e=0, point=""):
    """One job; asserts it drew only its declared streams (the splits are read from stage 03, not drawn)."""
    out = G.paths.output(G.STAGE, "hammond", row, UNIT) / model / point
    out.mkdir(parents=True, exist_ok=True)
    drawn.clear()
    G.set_encoder_row(row, "hammond", UNIT, model, e, point, {}, {}, out)
    declared = set(G.draws(row, model, e).values())
    assert set(drawn) <= declared and declared - set(drawn) <= {("encoder_split", e), ("search_split", e), ("spare_3", 0)}
    return out


def sides(path) -> dict:
    return anndata.read_h5ad(path).obs["side"].value_counts().to_dict()


def test_rows_outputs_budget_and_untrained_init(tmp_path, monkeypatch):
    G, config, unit_cells, drawn = patched(tmp_path, monkeypatch)
    outs = {row: run_row(G, drawn, row, "cytoset") for row in ("A", "B", "C")}
    for row in ("A", "B", "C"):
        data = anndata.read_h5ad(outs[row] / "embeddings" / "s0.h5ad")
        thinned = 2 if row == "A" else 0   # H3: rows A and J only
        assert data.obs["side"].value_counts().to_dict() == {"single_cell_train": 10, "single_cell_validation": 10,
                                                             "single_cell_heldout": 8, "bulk": 2 + thinned}
        assert (data.obs["variant"] == "thinned_pseudocells").sum() == thinned
        assert np.isfinite(data.X).all()
        assert set(data.obs.loc[data.obs["side"] != "bulk", "scoring_unit_id"]) == {"D" + s for s in TRAIN_S + HELD_S}
    a, b = (json.loads((outs[r] / "record_s0.json").read_text()) for r in ("A", "B"))
    assert b["steps"] == a["steps"]                                        # row B's budget
    a_config, b_config = (torch.load(outs[r] / "checkpoints" / "s0.pt", weights_only=False)["config"] for r in ("A", "B"))
    assert {k for k in a_config if a_config[k] != b_config.get(k)} | (set(b_config) - set(a_config)) <= \
        {"ncell", "abundance_weighting", "pseudobulk_gene_scaler", "n_epochs", "patience", "max_steps", "micro_batch"}
    assert not (outs["C"] / "record_s0.json").exists()                  # row C: embeddings only
    heads = pd.read_csv(outs["A"] / "native_head_s0.csv")
    assert heads["side"].value_counts().to_dict() == {"single_cell_train": 10, "single_cell_validation": 10,
                                                      "single_cell_heldout": 8, "bulk": 2}
    # row C is the network as built under encoder_init 0, embedded through the same Inputs
    train, _ = training_inputs.split_cells(unit_cells, SIDES)
    inputs = harness.Inputs({**config, "encoder": "cytoset", "micro_batch": 16}, train, G.training_inputs.pseudobulk_log("hammond", UNIT))
    torch.manual_seed(G.stream("hammond", UNIT, "C", "cytoset", 0)["init_seed"])
    net = G.network("cytoset", {**config, "encoder": "cytoset", "micro_batch": 16}, harness.input_width(config, len(unit_cells.genes)))
    expected = harness.embed_samples(net.eval(), unit_cells, inputs, sorted(TRAIN_S), 500, "hammond", UNIT)
    row_c = anndata.read_h5ad(outs["C"] / "embeddings" / "s0.h5ad")
    single = row_c.obs["side"].isin(["single_cell_train", "single_cell_validation"]).to_numpy()
    assert np.allclose(row_c.X[single], expected, atol=1e-6)


def test_bulk_logs(tmp_path, monkeypatch):
    """The record's capped bulk fraction and embedding norms."""
    G, config, unit_cells, drawn = patched(tmp_path, monkeypatch)
    out = run_row(G, drawn, "A", "cytoset")
    record = json.loads((out / "record_s0.json").read_text())
    data = anndata.read_h5ad(out / "embeddings" / "s0.h5ad")
    norm = lambda mask: np.linalg.norm(data.X[mask.to_numpy()], axis=1).mean()
    bulk = norm((data.obs["side"] == "bulk") & (data.obs["variant"] == "primary"))
    cell = norm(data.obs["side"].isin(["single_cell_train", "single_cell_validation"]))
    assert np.isclose(record["bulk_embedding_norm"], bulk, rtol=1e-5) and np.isclose(record["single_cell_embedding_norm"], cell, rtol=1e-5)
    assert np.isclose(record["embedding_norm_ratio"], bulk / cell, rtol=1e-5)
    assert record["capped_bulk_fraction"] == 0.0                        # "none" standardizes nothing
    # a standardizing level: 3 of each profile's genes at 20 SD, one at exactly 9 SD
    inputs = harness.Inputs({"cell_normalization": "population"}, unit_cells, {})
    x = np.tile(inputs.mean, (2, 1))
    x[:, :3] += 20 * inputs.sd[:3]
    x[:, 3] -= 9 * inputs.sd[3]
    profiles = protocol.Profiles(X=sp.csr_matrix(x), genes=unit_cells.genes, sample_id=np.array(["b0", "b1"]), library_size=np.ones(2))
    logged = G.bulk_logs(inputs, profiles, np.ones((2, 4)), np.ones((5, 4)))
    assert np.isclose(logged["capped_bulk_fraction"], 3 / x.shape[1]) and logged["embedding_norm_ratio"] == 1.0


def test_configurations_differ_from_row_a_only_in_their_change(monkeypatch):
    G = driver()
    frozen = {**search_space.base("cytoset"), "cell_normalization": "per_sample", "h_dim": 128}
    route = {**frozen, "cell_normalization": "population"}
    monkeypatch.setattr(G, "frozen", lambda pair, unit, encoder, key="frozen": dict(route if key == "per_sample_route" else frozen))
    a = G.configuration("A", "hammond", UNIT, "cytoset", "")
    diff = lambda c, ref: {k for k in set(c) | set(ref) if c.get(k) != ref.get(k)}
    for row, model in (("C", "cytoset"), ("D", "cytoset"), ("D", G.PSEUDOBULK), ("E", "cytoset"), ("H", "cytoset")):
        assert G.configuration(row, "hammond", UNIT, model, "f25") == a
    assert diff(G.configuration("F", "hammond", UNIT, "cytoset", "n25"), a) == {"ncell"}
    assert G.configuration("F", "hammond", UNIT, "cytoset", "n_all")["ncell"] == 3000
    assert diff(G.configuration("I", "hammond", UNIT, "cytoset", ""), a) == {"cell_normalization"}
    base = {**search_space.base("cytoset"), "encoder": "cytoset"}
    assert diff(G.configuration("MSE", "hammond", UNIT, "cytoset", ""), base) == {"objective"}
    for encoder in ("cytoset", "set_transformer", "scrat"):   # own recipes: the base plus the recipe (D5)
        assert G.configuration("G", "hammond", UNIT, encoder, "") == {**search_space.base(encoder), **T.OWN_RECIPES[encoder], "encoder": encoder}
    assert T.OWN_RECIPES["set_transformer"]["nsubset"] == 1024


def test_permuted_row(tmp_path, monkeypatch):
    G, _, _, drawn = patched(tmp_path, monkeypatch)
    seen, train = [], G.harness.train   # the (training, validation) labels each training is given
    monkeypatch.setattr(G.harness, "train", lambda *a, **k: seen.append((a[2].samples["group"], a[3].samples["group"])) or train(*a, **k))
    cells_out = run_row(G, drawn, "D", "cytoset", e=3)
    pseudo_out = run_row(G, drawn, "D", G.PSEUDOBULK, e=3)
    permuted = training_inputs.permuted_groups("hammond", UNIT, 3)["group"]
    assert (permuted != toy_samples().set_index("sample_id")["group"]).any()
    for side_labels in seen:                                             # both sides, cells and pseudobulk
        for groups in side_labels:
            assert groups.equals(permuted.loc[groups.index])
    for out in (cells_out, pseudo_out):
        assert sides(out / "embeddings" / "s3.h5ad") == {"single_cell_train": 10, "single_cell_validation": 10, "bulk": 2}
        assert not (out / "checkpoints").exists()                       # embeddings and records only
    assert json.loads((pseudo_out / "record_s3.json").read_text())["steps"] == json.loads((cells_out / "record_s3.json").read_text())["steps"]


def test_permuted_groups_within_side_and_by_group(monkeypatch):
    holdout = [f"g{i // 2}" for i in range(24)]   # two samples per holdout group
    samples = pd.DataFrame({"sample_id": [f"s{i}" for i in range(24)], "holdout_group": holdout,
                            "group": [int(h[1:]) % 2 for h in holdout], "label_value": [float(int(h[1:])) for h in holdout]})
    side = {s: "validation" if int(h[1:]) < 4 else "train" for s, h in zip(samples["sample_id"], holdout)}
    monkeypatch.setattr(training_inputs, "folds", types.SimpleNamespace(training_samples=lambda pair, fold_set, fold: samples))
    monkeypatch.setattr(training_inputs, "encoder_sides", lambda pair, unit, j: side)
    changed = False
    for j in range(5):
        permuted = training_inputs.permuted_groups("hammond", UNIT, j)
        assert list(permuted.index) == list(samples["sample_id"])     # single-cell training samples only: bulk untouched
        table = samples.set_index("sample_id").assign(new=permuted["group"], source=permuted["source"], side=pd.Series(side))
        assert (table.groupby("holdout_group")["new"].nunique() == 1).all()                   # a group's samples move together
        for s in ("train", "validation"):                                                      # class counts per side kept
            part = table[table["side"] == s]
            assert sorted(part.groupby("holdout_group")["new"].first()) == sorted(part.groupby("holdout_group")["group"].first())
        source_side = table.groupby("holdout_group")["side"].first()
        assert (source_side.loc[table["source"]].to_numpy() == table["side"].to_numpy()).all()
        assert (table["new"] == table.groupby("holdout_group")["group"].first().loc[table["source"]].to_numpy()).all()
        changed |= (table["new"] != table["group"]).any()
        assert permuted.equals(training_inputs.permuted_groups("hammond", UNIT, j))          # seeded
        ages = training_inputs.training_ages("hammond", UNIT, table.index[table["side"] == "train"], source=permuted["source"])
        mean_age = table.groupby("holdout_group")["label_value"].mean()
        raw = mean_age.loc[table.loc[list(ages), "source"]].to_numpy()                         # the source group's mean age, z-scored
        assert np.allclose(list(ages.values()), (raw - raw.mean()) / raw.std())
    assert changed


def test_learning_curve_subsets_nested_and_counted(monkeypatch):
    monkeypatch.setattr(training_inputs, "folds", types.SimpleNamespace(training_samples=lambda pair, fold_set, fold: toy_samples()))
    monkeypatch.setattr(training_inputs, "encoder_sides", lambda pair, unit, e: dict(SIDES))
    for e in range(3):
        quarter, half = (training_inputs.learning_curve_subset("lupus", UNIT, e, f) for f in (0.25, 0.5))
        assert set(quarter) <= set(half) <= set(SIDES)                  # nested
        for subset, f in ((quarter, 0.25), (half, 0.5)):
            assert all(subset[s] == SIDES[s] for s in subset)
            for side in ("train", "validation"):
                for label in (0, 1):
                    n = sum(1 for s in subset if subset[s] == side and int(s[1:]) % 2 == label)
                    assert n == int(np.ceil(f * 5))
    with pytest.raises(AssertionError):                                  # fewer than 2 validation groups per label
        training_inputs.learning_curve_subset("lupus", UNIT, 0, 0.1)


def test_curve_rows(tmp_path, monkeypatch):
    G, _, _, drawn = patched(tmp_path, monkeypatch)
    given, inputs = [], G.harness.Inputs   # the gene scaler's p each training is given
    monkeypatch.setattr(G.harness, "Inputs", lambda config, train, pseudobulk: given.append(pseudobulk) or inputs(config, train, pseudobulk))
    for model in ("cytoset", G.PSEUDOBULK):
        out = run_row(G, drawn, "E", model, point="f25")
        assert sides(out / "embeddings" / "s0.h5ad") == {"single_cell_train": 4, "single_cell_validation": 4, "bulk": 2}
    assert json.loads((out / "record_s0.json").read_text())["steps"] == \
        json.loads((out.parents[1] / "cytoset" / "f25" / "record_s0.json").read_text())["steps"]
    for point in ("n25", "n_all"):
        out = run_row(G, drawn, "F", "cytoset", point=point)
        assert sides(out / "embeddings" / "s0.h5ad") == {"single_cell_train": 10, "single_cell_validation": 10, "bulk": 2}
        assert (out / "record_s0.json").exists() and not (out / "checkpoints").exists()
        expected = (G.training_inputs.pseudobulk_log_all_available if point == "n_all" else G.training_inputs.pseudobulk_log)("hammond", UNIT)
        assert given[-1].keys() == expected.keys() and all(np.array_equal(given[-1][s], expected[s]) for s in expected)   # "all": its own pseudobulk (F2)


def test_own_recipe_mixup_row(tmp_path, monkeypatch):
    G, _, _, drawn = patched(tmp_path, monkeypatch)
    out = run_row(G, drawn, "G", "scrat")
    assert ("train_bags", T.MIXUP_OFFSET) in drawn
    checkpoint = torch.load(out / "checkpoints" / "s0.pt", weights_only=False)
    assert checkpoint["recipe"] == "scrat" and checkpoint["components"].shape == (GENES, 5) and checkpoint["config"]["mixup"]
    assert json.loads((out / "record_s0.json").read_text())["recipe"] == "scrat"
    assert sides(out / "embeddings" / "s0.h5ad") == {"single_cell_train": 10, "single_cell_validation": 10, "bulk": 2}


def test_mixup_follows_upstream_rule(monkeypatch):
    """Two types: a control of type a only, a case half a and half b. Each synthetic sample's type counts are
    max(int(min_size x (lam f1 + (1 - lam) f2)), 1), its label lam y1 + (1 - lam) y2, and the absent type's
    cells are (1 - lam) x the present partner's (zero partner) plus small noise; only synthetic samples train."""
    monkeypatch.setattr(T, "SCRAT_MIXUP", {"augment_num": 20, "alpha": 0.5, "min_size": 100, "noise_var": 1e-5, "bags_per_sample": 3})
    X = sp.csr_matrix(np.vstack([np.full((40, 3), 1.0), np.full((20, 3), 2.0), np.full((20, 3), 3.0)]))
    train = protocol.Cells(X=X, genes=["g0", "g1", "g2"], cell_id=np.array([f"c{i}" for i in range(80)]),
                           sample_id=np.array(["ctrl"] * 40 + ["case"] * 40), cell_type=np.array(["a"] * 60 + ["b"] * 20),
                           library_size=np.ones(80), samples=pd.DataFrame({"group": [0, 1], "holdout_group": ["ctrl", "case"]},
                                                                           index=["ctrl", "case"]))
    inputs = types.SimpleNamespace(components=np.eye(3), cells=lambda x, s: x)   # identity transform
    mixed = mixup.synthetic(train, inputs, pd.Series("all", index=["ctrl", "case"]), np.random.default_rng(0))
    assert set(mixed.samples.index) == {f"mix{k}" for k in range(20)} and all(s.startswith("mix") for s in mixed.sample_id)
    for s, label in mixed.samples["group"].items():
        rows = mixed.sample_id == s
        lam_case = label   # lam weights y1; the soft label is the case's weight
        n_a, n_b = int(100 * (lam_case * 0.5 + (1 - lam_case))), max(int(100 * lam_case * 0.5), 1)
        x = mixed.X[rows]
        a, b = x[:n_a], x[n_a:]
        assert len(x) == n_a + n_b
        assert np.allclose(a, (1 - lam_case) * 1.0 + lam_case * 2.0)                         # type a: control 1s with case 2s
        assert np.allclose(b, lam_case * 3.0, atol=0.02) and not np.allclose(b, lam_case * 3.0, atol=1e-12)   # b: zero partner + noise
    bag_list = mixup.fixed_bags(mixed, 30, np.random.default_rng(1))
    assert len(bag_list) == 20 * 3 and {len(ids) for _, ids in bag_list} == {30}


def test_one_type_row(tmp_path, monkeypatch):
    G, _, _, drawn = patched(tmp_path, monkeypatch)
    for model in ("cytoset", G.PSEUDOBULK):
        out = run_row(G, drawn, "H", model)
        assert sides(out / "embeddings" / "s0.h5ad") == {"single_cell_train": 8, "single_cell_validation": 10,
                                                         "single_cell_heldout": 6, "bulk": 2}
    sizes = pd.read_csv(out.parent / "cytoset" / "bag_sizes.csv")
    assert len(sizes) == 18 and (sizes["cells"] == 60).all()


def test_one_type_split_floor_and_smoke_skip(tmp_path, monkeypatch):
    """Row H needs >= 2 holdout groups per label on each side; under smoke the training side needs 1, and a unit
    with any seed still short has its row H jobs (every seed's and model's) left out of the job list, with a marker."""
    G, _, _, _ = patched(tmp_path, monkeypatch)
    cells = G.training_inputs.unit_cells("hammond", UNIT, cell_subset="one_type")
    one_train = {**SIDES, "s10": "validation", "s12": "validation", "s14": "validation"}   # training label 0: s16 (s18 isn't stored)
    one_validation = {**SIDES, **{s: "train" for s in ("s3", "s5", "s7", "s9")}}            # validation label 1: s1
    split = {0: dict(SIDES), 1: one_train, 2: one_validation}
    monkeypatch.setattr(G.training_inputs, "encoder_sides", lambda pair, unit, e: split[e])
    monkeypatch.setattr(G.training_inputs, "encoder_seeds", lambda pair, row: [0, 1, 2])
    monkeypatch.setattr(settings, "SMOKE", False)
    G.row_sides("H", "hammond", UNIT, 0, "", cells)
    for e in (1, 2):
        with pytest.raises(AssertionError, match="one-type"):
            G.row_sides("H", "hammond", UNIT, e, "", cells)
    monkeypatch.setattr(settings, "SMOKE", True)
    assert G.row_sides("H", "hammond", UNIT, 1, "", cells) == G.one_type_sides(one_train, cells)
    with pytest.raises(AssertionError, match="1 one-type validation groups of label 1"):
        G.row_sides("H", "hammond", UNIT, 2, "", cells)
    marker = G.smoke_marker("hammond", UNIT, "H")
    assert G.smoke_skips_one_type("hammond", UNIT) and json.loads(marker.read_text()) == {"2": ["1 one-type validation groups of label 1"]}
    monkeypatch.setattr(G.training_inputs, "encoder_seeds", lambda pair, row: [0, 1])
    assert not G.smoke_skips_one_type("hammond", UNIT) and not marker.exists()   # a rerun clears the marker
    monkeypatch.setattr(G.folds, "units", lambda pair: [("crossfit", 0)])
    monkeypatch.setattr(G, "has_per_sample_route", lambda pair, unit, encoder: False)
    monkeypatch.setattr(G, "frozen", lambda pair, unit, encoder, key="frozen": {})
    monkeypatch.setattr(G, "smoke_skips_learning_curve", lambda pair, unit: False)
    for skips in (True, False):
        monkeypatch.setattr(G, "smoke_skips_one_type", lambda pair, unit: skips)
        for environment in ("main", "scvi"):
            rows = {row for _, row, *_ in G.jobs("soundlife", environment)}
            assert ("H" in rows) != skips and rows & {"A", "J"}


def test_learning_curve_smoke_skip(tmp_path, monkeypatch):
    """Row E's validation groups per label (learning_curve_counts) are those training_inputs.learning_curve_subset keeps,
    which asserts >= 2; under smoke a unit with any seed or point short has its row E jobs (every seed's, point's and
    model's) left out of the job list, with a marker. The real run lists them all."""
    G, _, _, _ = patched(tmp_path, monkeypatch)
    short = {**SIDES, "s1": "train", "s3": "train"}   # validation label 1: s5, s7, s9 -> 1 group at 25%, 2 at 50%
    split = {0: dict(SIDES), 1: short}
    monkeypatch.setattr(training_inputs, "encoder_sides", lambda pair, unit, e: split[e])
    monkeypatch.setattr(G.training_inputs, "encoder_sides", lambda pair, unit, e: split[e])
    monkeypatch.setattr(G.training_inputs, "encoder_seeds", lambda pair, row: [0, 1])
    monkeypatch.setattr(G.folds, "training_samples", lambda pair, fold_set, fold: toy_samples())
    for e in (0, 1):
        for f in (0.25, 0.5):
            counts = G.learning_curve_counts("soundlife", UNIT, e, f)
            if (e, f) == (1, 0.25):
                assert counts == {0: 2, 1: 1}
                with pytest.raises(AssertionError, match="1 validation groups of label 1 at 0.25"):
                    training_inputs.learning_curve_subset("soundlife", UNIT, e, f)
                continue
            subset = training_inputs.learning_curve_subset("soundlife", UNIT, e, f)
            assert counts == {label: sum(subset[s] == "validation" and int(s[1:]) % 2 == label for s in subset) for label in (0, 1)}
    marker = G.smoke_marker("soundlife", UNIT, "E")
    assert G.smoke_skips_learning_curve("soundlife", UNIT) and json.loads(marker.read_text()) == {"s1 f25": ["1 validation groups of label 1"]}
    monkeypatch.setattr(G.training_inputs, "encoder_seeds", lambda pair, row: [0])
    assert not G.smoke_skips_learning_curve("soundlife", UNIT) and not marker.exists()   # a rerun clears the marker
    monkeypatch.setattr(G.folds, "units", lambda pair: [("crossfit", 0)])
    monkeypatch.setattr(G, "has_per_sample_route", lambda pair, unit, encoder: False)
    monkeypatch.setattr(G, "frozen", lambda pair, unit, encoder, key="frozen": {})
    monkeypatch.setattr(G, "smoke_skips_one_type", lambda pair, unit: False)
    for smoke, skips in ((True, True), (True, False), (False, True)):
        monkeypatch.setattr(settings, "SMOKE", smoke)
        monkeypatch.setattr(G, "smoke_skips_learning_curve", lambda pair, unit: skips)
        for environment in ("main", "scvi"):
            jobs = G.jobs("soundlife", environment)
            rows = {row for _, row, *_ in jobs}
            assert ("E" in rows) != (smoke and skips) and rows & {"A", "J"}
            assert smoke and skips or {point for _, row, _, _, point in jobs if row == "E"} == {"f25", "f50"}


def test_per_sample_route_row(tmp_path, monkeypatch):
    G, _, _, drawn = patched(tmp_path, monkeypatch)
    out = run_row(G, drawn, "I", "cytoset")
    assert torch.load(out / "checkpoints" / "s0.pt", weights_only=False)["config"]["cell_normalization"] == "population"
    assert sides(out / "embeddings" / "s0.h5ad")["single_cell_heldout"] == 8
    assert not (out / "native_head_s0.csv").exists()


def test_mse_row(tmp_path, monkeypatch):
    G, _, _, drawn = patched(tmp_path, monkeypatch)
    out = run_row(G, drawn, "MSE", "cytoset", e=1)
    assert set(drawn) == {("search_init", 1), ("search_sampling", 1), (T.VALIDATION_BAGS_PURPOSE, T.SEARCH_INDEX_OFFSET + 1)}
    record = json.loads((out / "record_search_s1.json").read_text())
    assert record["objective"] == "mse" and record["margin_d_star"] is None
    assert sides(out / "embeddings" / "search_s1.h5ad") == {"single_cell_train": 10, "single_cell_validation": 10, "bulk": 2}
    assert not (out / "checkpoints").exists()


def test_did_it_train_mse():
    names = [f"s{i}" for i in range(12)]
    samples = pd.DataFrame({"group": [i % 2 for i in range(12)], "holdout_group": names}, index=names)
    targets = {s: (1.0 if g else -1.0) + 0.1 * i for i, (s, g) in enumerate(samples["group"].items())}
    val_bags = [(s, None) for s in names for _ in range(3)]
    constant = harness.did_it_train_mse(np.zeros(len(val_bags)), val_bags, samples, targets, seed=0)
    assert abs(constant[1]) < 1e-12 and not constant[0]                 # the training-mean predictor: d = 0
    perfect = harness.did_it_train_mse(np.array([targets[s] for s, _ in val_bags]), val_bags, samples, targets, seed=0)
    assert perfect[1] == pytest.approx(1.0) and perfect[0] and perfect[4] < 0.05 and perfect[2] is None
    few = [(s, None) for s in names[:4]]                                # C(4, 2) labellings: the floor alone, p NaN
    small = harness.did_it_train_mse(np.array([targets[s] for s, _ in few]), few, samples.iloc[:4], targets, seed=0)
    assert small[0] and np.isnan(small[4]) and not small[5]


def test_thinned_bulk_same_draws_within_organ(tmp_path, monkeypatch):
    genes = [f"g{i}" for i in range(6)]
    obs = pd.DataFrame({"sample_id": ["s0"] * 50 + ["s1"] * 50}, index=[f"c{i}" for i in range(100)])
    X = np.vstack([np.full((50, 6), 20.0), np.full((50, 6), 2000.0)])   # liver cells total 120, lung cells 12,000
    samples = pd.DataFrame({"sample_id": ["s0", "s1"], "organ": ["liver", "lung"]})
    monkeypatch.setattr(training_inputs, "folds", types.SimpleNamespace(training_cells=lambda pair, fs, f: anndata.AnnData(X, obs=obs, var=pd.DataFrame(index=genes)),
                                                                         training_samples=lambda pair, fs, f: samples))
    monkeypatch.setattr(training_inputs, "shared", lambda pair: genes)
    monkeypatch.setattr(training_inputs, "panel", lambda pair, unit: genes[:4])
    monkeypatch.setattr(settings, "OUTPUTS", tmp_path)
    monkeypatch.setattr(T, "FROZEN_NCELL", 30)
    (tmp_path / "02_preprocess" / "tms").mkdir(parents=True)
    bulk = anndata.AnnData(np.full((2, 6), 1e5), obs=pd.DataFrame(index=["b0", "b1"]), var=pd.DataFrame(index=genes))
    bulk.write_h5ad(tmp_path / "02_preprocess" / "tms" / "bulk.h5ad")
    table = pd.DataFrame({"sample_id": ["b0", "b1"], "organ": ["liver", "lung"]})
    first, second = (list(training_inputs.thinned_bulk("tms", ("all", 0), table)) for _ in range(2))
    for (s, counts, totals), (_, counts2, totals2) in zip(first, second):   # the same draws for every model
        assert np.array_equal(counts, counts2) and np.array_equal(totals, totals2)
        assert counts.shape == (10, 30, 4) and (totals >= counts.sum(axis=2)).all()
    assert first[0][2].mean() < 200 and first[1][2].mean() > 5000         # library sizes from the bulk sample's organ


def test_scvi_rows(tmp_path, monkeypatch):
    """Rows J and E fit (stubbed scvi-tools); F and H embed with row J's saved model and copy its bulk rows."""
    from sc2bulk.encoders import scvi as adapter
    G, _, _, drawn = patched(tmp_path, monkeypatch)

    class Model:
        def save(self, path, overwrite):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.pt").write_bytes(b"model")
    first3 = lambda counts: np.asarray(sp.csr_matrix(counts)[:, :3].todense(), dtype=float)
    monkeypatch.setattr(adapter, "fit", lambda train, val, config, seed: (Model(), {"median_library_size": 100.0, "epochs": 1}))
    monkeypatch.setattr(adapter, "load", lambda directory, counts: Model())
    monkeypatch.setattr(adapter, "latent", lambda model, counts: first3(counts))
    monkeypatch.setattr(adapter, "embed_profiles", lambda model, counts, median, seed, thinned=True: first3(counts) + (0 if thinned else 1))
    outs = {}
    for row, point in (("J", ""), ("E", "f50"), ("F", "n25"), ("H", "")):
        outs[row] = G.paths.output(G.STAGE, "hammond", row, UNIT) / "scvi" / point
        outs[row].mkdir(parents=True, exist_ok=True)
        G.scvi_row(row, "hammond", UNIT, 0, point, {}, {}, outs[row])
    j = anndata.read_h5ad(outs["J"] / "embeddings" / "s0.h5ad")
    assert (outs["J"] / "checkpoints" / "s0" / "model.pt").exists()
    assert j.obs.loc[j.obs["side"] == "bulk", "variant"].value_counts().to_dict() == {"primary": 2, "observed_library": 2, "thinned_pseudocells": 2}
    assert sides(outs["E"] / "embeddings" / "s0.h5ad") == {"single_cell_train": 6, "single_cell_validation": 6, "bulk": 4}
    for row, held in (("F", 0), ("H", 6)):
        data = anndata.read_h5ad(outs[row] / "embeddings" / "s0.h5ad")
        assert np.array_equal(data[data.obs["side"] == "bulk"].X, j[j.obs["side"] == "bulk"].X)   # row J's bulk rows
        assert (data.obs["side"] == "single_cell_heldout").sum() == held
        assert not (outs[row] / "record_s0.json").exists()


def test_streams_disjoint_across_stages():
    """Every (purpose, index) that stages 03b, 05, 05v, 05s, 06 and the blinded run draw serves one role; a
    shared stream is only a declared pairing (rows reusing row A's, the MSE variant with stage 05's base,
    row E's scVI with row J's), which carries the same role name."""
    G = driver()
    spec = importlib.util.spec_from_file_location("stage05s", settings.REPO / "pipeline" / "05s_eligibility_simulation.py")
    s05s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s05s)
    roles = {}
    use = lambda key, role: roles.setdefault(key, set()).add(role)
    for i in range(settings.N_SEARCH_SEEDS):   # stage 05 (and 05t): search seed i; 05v: scVI's search init
        for key, role in ((("search_split", i), "search_split"), (("search_init", i), "search_init"),
                          (("search_sampling", i), "search_bags"), ((T.VALIDATION_BAGS_PURPOSE, T.SEARCH_INDEX_OFFSET + i), "validation_bags"),
                          (("scvi_init", T.SEARCH_INDEX_OFFSET + i), "scvi_search_init")):
            use(key, role)
    for r in range(1000):
        use((s05s.PURPOSE, r), "eligibility_simulation")
    for b in range(T.PERMUTATION_OFFSET):   # the blinded run's bulk-label permutation (sc2bulk.labels, index BLINDED)
        use(("permutation", b), "blinded_permutation")
    for e in range(20):                     # stages 03 and 03b
        use(("encoder_split", e), "split")
    for row, models in (("A", ("cytoset", "set_transformer", "scrat")), ("B", ("cytoset",)), ("C", ("cytoset",)),
                        ("D", ("cytoset", "set_transformer", "scrat", G.PSEUDOBULK)), ("E", ("cytoset", "scrat", G.PSEUDOBULK, "scvi")),
                        ("F", ("cytoset", "scrat", "scvi")), ("G", ("cytoset", "set_transformer", "scrat")),
                        ("H", ("cytoset", G.PSEUDOBULK, "scvi")), ("I", ("cytoset",)), ("J", ("scvi",)), ("MSE", ("cytoset",))):
        for model in models:
            for e in range(20 if row in ("A", "D") else 10 if row != "MSE" else settings.N_SEARCH_SEEDS):
                for role, key in G.draws(row, model, e).items():
                    use(key, role)
    shared = {key: r for key, r in roles.items() if len(r) > 1}
    assert not shared, f"streams serving two roles: {shared}"


def test_blinded_index_below_the_permutation_offset(monkeypatch):
    path = settings.REPO / "src" / "sc2bulk" / "config" / "training.py"
    monkeypatch.setattr(settings, "BLINDED", T.PERMUTATION_OFFSET - 1)
    runpy.run_path(str(path))
    monkeypatch.setattr(settings, "BLINDED", T.PERMUTATION_OFFSET)
    with pytest.raises(AssertionError):
        runpy.run_path(str(path))


def test_jobs(monkeypatch):
    G = driver()
    monkeypatch.setattr(settings, "SMOKE", False)
    monkeypatch.setattr(G.folds, "units", lambda pair: [("crossfit", 0), ("split67", 0)])
    monkeypatch.setattr(G, "has_per_sample_route", lambda pair, unit, encoder: encoder == "cytoset")
    monkeypatch.setattr(G, "frozen", lambda pair, unit, encoder, key="frozen": {})   # no age loss
    main = G.jobs("soundlife", "main")
    assert {row for u, row, *_ in main if u[0] == "split67"} == {"A", "B"}       # split67: rows A, B, J
    assert {e for _, row, _, e, _ in main if row == "A"} == set(range(10))       # Sound Life: no attribution seeds
    assert {e for _, row, _, e, _ in main if row == "D"} == set(range(20))       # permutation j at encoder seed j
    assert {e for _, row, _, e, _ in main if row in ("E", "F", "H")} == set(range(5))
    assert {(m, e) for _, row, m, e, _ in main if row == "MSE"} == {("cytoset", i) for i in range(settings.N_SEARCH_SEEDS)}
    assert {m for _, row, m, _, _ in main if row == "I"} == {"cytoset"}
    assert {p for _, row, _, _, p in main if row == "F"} == {"n1", "n25", "n50", "n100", "n200", "n300", "n600", "n1000", "n_all"}
    for unit in (("crossfit", 0), ("split67", 0)):                               # B after A within each unit
        rows = [row for u, row, *_ in main if u == unit]
        assert rows.index("B") > max(i for i, r in enumerate(rows) if r == "A")
    for row in ("D", "E", "H"):                                                  # each pseudobulk retraining after its cells job
        order = [(m, e, p) for _, r, m, e, p in main if r == row]
        for m, e, p in order:
            if m == G.PSEUDOBULK:
                assert order.index(("cytoset", e, p)) < order.index((m, e, p))
    scvi = [row for u, row, *_ in G.jobs("soundlife", "scvi") if u == ("crossfit", 0)]   # F and H embed with row J's models
    assert min(scvi.index("F"), scvi.index("H")) > max(i for i, r in enumerate(scvi) if r == "J")
    monkeypatch.setattr(G.folds, "units", lambda pair: [("all", 0)])
    assert {e for _, row, _, e, _ in G.jobs("lupus", "scrat") if row == "A"} == set(range(20))
    assert {(row, m) for _, row, m, _, _ in G.jobs("lupus", "scvi")} == {("J", "scvi"), ("E", "scvi"), ("F", "scvi"), ("H", "scvi")}
    assert not {row for _, row, *_ in G.jobs("hammond", "main")} & {"E", "H"}   # no learning curve; no one-type store
    assert not {row for _, row, *_ in G.jobs("lupus", "main")} & {"MSE"}        # MSE: the aging pairs


def test_age_loss_host_gather(tmp_path, monkeypatch):
    """With the age loss above 600 cells both residents take the host gather, whatever
    fits_on_device says; at 600 and below, or without the age loss, fits_on_device decides. The record says which."""
    G, config, _, drawn = patched(tmp_path, monkeypatch)
    reason = "age-similarity loss: whole batch in one pass, above 600 cells"
    config["age_similarity_loss"] = True
    placed, resident = [], G.harness.Resident
    monkeypatch.setattr(G.harness, "Resident", lambda cells, inputs, device, on_device=True: placed.append(on_device) or
                        resident(cells, inputs, device, on_device))
    for point, on_device in (("n1000", False), ("n600", True)):   # the toy's samples cap the bags; the rule reads the configured size
        placed.clear()
        record = json.loads((run_row(G, drawn, "F", "cytoset", point=point) / "record_s0.json").read_text())
        assert placed == [on_device, on_device] and record["on_device"] is on_device
        assert record.get("host_reason") == (None if on_device else reason)
    for fits in (True, False):
        monkeypatch.setattr(G.harness, "fits_on_device", lambda *cells: fits)
        assert G.on_device({"age_similarity_loss": True, "ncell": 1000}, None, None) == (False, reason)
        for c in ({"age_similarity_loss": True, "ncell": 600}, {"age_similarity_loss": True, "ncell": 500},
                  {"age_similarity_loss": False, "ncell": 1000}, {"ncell": 1000}):
            assert G.on_device(c, None, None) == (fits, None)


def test_jobs_drop_age_loss_cytoset_all(monkeypatch):
    """No row F "all" for CytoSet in a unit whose frozen CytoSet configuration has the age loss;
    every other job unchanged."""
    G = driver()
    monkeypatch.setattr(settings, "SMOKE", False)
    monkeypatch.setattr(G.folds, "units", lambda pair: [("crossfit", 0), ("crossfit", 1), ("split67", 0)])
    monkeypatch.setattr(G, "has_per_sample_route", lambda pair, unit, encoder: encoder == "cytoset")
    asked = set()

    def frozen(age_loss_units):
        return lambda pair, unit, encoder, key="frozen": asked.add(encoder) or {"age_similarity_loss": unit in age_loss_units}
    environments = ("main", "scrat", "scvi")
    monkeypatch.setattr(G, "frozen", frozen(set()))
    before = {env: G.jobs("soundlife", env) for env in environments}
    monkeypatch.setattr(G, "frozen", frozen({("crossfit", 1)}))
    after = {env: G.jobs("soundlife", env) for env in environments}
    dropped = [j for j in before["main"] if j not in after["main"]]
    assert dropped == [(("crossfit", 1), "F", "cytoset", e, "n_all") for e in range(5)]
    assert [j for j in before["main"] if j not in dropped] == after["main"]
    assert after["scrat"] == before["scrat"] and after["scvi"] == before["scvi"]
    kept = {(u, m, p) for u, row, m, _, p in after["main"] + after["scrat"] + after["scvi"] if row == "F"}
    assert {(("crossfit", 0), "cytoset", "n_all"), (("crossfit", 1), "cytoset", "n1000"), (("crossfit", 1), "set_transformer", "n_all"),
            (("crossfit", 1), "scrat", "n_all"), (("crossfit", 1), "scvi", "n_all")} <= kept
    assert asked == {"cytoset"}


def listed(tmp_path, monkeypatch, files):
    """06_train --list on Sound Life's crossfit_0 with outputs under tmp_path: a job is done when its embeddings file
    exists; its record reads its upstream's record (row B, cytoset_pseudobulk: cells_steps) or row J's model (scVI
    F, H), raising when that file is missing, as run_record does."""
    G = driver()
    monkeypatch.setattr(settings, "SMOKE", False)
    monkeypatch.setattr(G.folds, "units", lambda pair: [UNIT])
    monkeypatch.setattr(G, "has_per_sample_route", lambda pair, unit, encoder: encoder == "cytoset")
    monkeypatch.setattr(G, "frozen", lambda pair, unit, encoder, key="frozen": {})
    monkeypatch.setattr(G, "paths", types.SimpleNamespace(output=lambda stage, pair, name, unit=None: tmp_path / stage / name))
    out = lambda row, model, e, point="", name=None: tmp_path / G.STAGE / row / model / point / (name or f"embeddings/s{e}.h5ad")

    def run_record(pair, unit, row, model, e, point, code):
        assert code == "code"
        for u, r, m, s, p in G.upstream(unit, row, model, e, point):
            (out(r, m, s, p, "checkpoints/s0/model.pt") if r == "J" else out(r, m, s, p, f"record_s{s}.json")).read_text()
        return {"job": (row, model, e, point)}
    monkeypatch.setattr(G, "run_record", run_record)
    monkeypatch.setattr(G, "manifest", types.SimpleNamespace(stage_code_hash=lambda script, pair: "code",
                                                             is_done=lambda outputs, record: all(o.exists() for o in outputs)))
    for path in files(out):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        G.listing("soundlife")
    return G, [json.loads(line) for line in printed.getvalue().splitlines()]


def test_list_dependency_order(tmp_path, monkeypatch):
    """A job is listed done only when its upstream jobs are, and --list never builds a record
    that reads a missing upstream output."""
    def files(out):
        return [out("B", "cytoset", 0), out("A", "cytoset", 1, name="record_s1.json"), out("B", "cytoset", 1),   # leftover A record
                out("F", "scvi", 0, "n100"), out("H", "scvi", 0),                                                # row J missing
                out("A", "cytoset", 2, name="record_s2.json"), out("A", "cytoset", 2), out("B", "cytoset", 2)]
    G, lines = listed(tmp_path, monkeypatch, files)
    done = {(x["row"], x["model"], x["seed"], x["point"]): x["done"] for x in lines}
    assert not done[("B", "cytoset", 0, "")] and not done[("B", "cytoset", 1, "")]
    assert not done[("F", "scvi", 0, "n100")] and not done[("H", "scvi", 0, "")]
    assert done[("A", "cytoset", 2, "")] and done[("B", "cytoset", 2, "")]
    assert sum(done.values()) == 2


def test_list_jobs_parse_and_done(tmp_path, monkeypatch):
    """--list lists every environment's jobs in jobs()' order; each after names a listed
    job; each job's command, as the scheduler builds it, is one 06_train.main() accepts; done is run()'s is_done."""
    G, lines = listed(tmp_path, monkeypatch, lambda out: [out("A", "cytoset", 3, name="record_s3.json"), out("A", "cytoset", 3),
                                                          out("J", "scvi", 4, name="checkpoints/s0/model.pt"), out("J", "scvi", 4)])
    key = lambda x: (x["unit"], x["row"], x["model"], x["seed"], x["point"])
    jobs = [job for env in ("main", "scrat", "scvi") for job in G.jobs("soundlife", env)]
    assert [key(x) for x in lines] == [("crossfit_0", *job[1:]) for job in jobs]
    done = {key(x): x["done"] for x in lines}
    assert all(tuple(a) in done for x in lines for a in x["after"])
    assert {k for k, d in done.items() if d} == {("crossfit_0", "A", "cytoset", 3, ""), ("crossfit_0", "J", "scvi", 4, "")}
    monkeypatch.setenv("SC2BULK_SERVER_ROOT", "/server")   # read when schedule.py loads
    spec = importlib.util.spec_from_file_location("schedule", settings.REPO / "scripts" / "schedule.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    monkeypatch.setattr(S, "age_loss", lambda pair, unit: False)
    monkeypatch.setattr(G.determinism, "configure", lambda: None)
    monkeypatch.setattr(G.manifest, "begin", lambda device: {}, raising=False)
    trained = []
    monkeypatch.setattr(G, "set_encoder_row", lambda row, pair, unit, model, e, point, *rest: trained.append((row, model, e, point)))
    monkeypatch.setattr(G, "scvi_row", lambda row, pair, unit, e, point, *rest: trained.append((row, "scvi", e, point)))
    for x in lines:
        if not all(done[tuple(a)] for a in x["after"]):   # its record would read a missing upstream output
            continue
        monkeypatch.setattr(sys, "argv", ["06_train.py", *S.stage06_job("soundlife", x).args])
        trained.clear()
        G.main()
        assert trained == ([] if x["done"] else [(x["row"], x["model"], x["seed"], x["point"])])
