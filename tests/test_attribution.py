"""Stage 10's computations (sc2bulk.attribution):
the averaged head, pseudobulk's exact completeness, TorchInputs reproducing stage 06's transform and embeddings,
bag attribution against brute force and finite differences, Geneformer deletion, scGPT's value gradient, the
sample and cell draws, rankings, the checks, and the seed purposes."""
import functools
import gzip
import re
import runpy

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from sc2bulk import attribution, bags, folds, harness, manifest, paths, protocol, schemas
from sc2bulk.config import genes as G
from sc2bulk.config import settings
from sc2bulk.config import training as T
from sc2bulk.encoders import cytoset


def head_table(n_seeds=3, dims=4, strata=("all",), seed=0) -> pd.DataFrame:
    """A stage 08 heads table: per read-out and read-out seed, coefficients, intercept and per-stratum scalers."""
    rng = np.random.default_rng(seed)
    rows = []
    for readout in ("ridge", "lasso"):
        for r in range(n_seeds):
            coef, intercept = rng.normal(size=dims), rng.normal()
            for st in strata:
                mean, scale = rng.normal(size=dims), rng.uniform(0.5, 2.0, dims)
                rows.append(pd.DataFrame({"encoder_seed": -1, "readout": readout, "readout_seed": r, "side": "single_cell", "stratum": st,
                                          "gene_or_dim": [f"g{i}" for i in range(dims)], "coef": coef, "intercept": intercept,
                                          "scaler_mean": mean, "scaler_scale": scale}))
    return pd.concat(rows, ignore_index=True)


def test_averaged_head_is_the_mean_prediction_and_linear():
    table = head_table(strata=("kidney", "lung"))
    heads, dims = attribution.heads(table)
    per_seed, _ = attribution.heads(table, per_seed=True)
    X = np.random.default_rng(1).normal(size=(6, 4))
    strata = np.array(["kidney", "lung"] * 3)
    for readout in ("ridge", "lasso"):
        expected = np.zeros(len(X))
        for r in range(3):
            rows = table[(table["readout"] == readout) & (table["readout_seed"] == r)]
            for i, (x, st) in enumerate(zip(X, strata)):
                one = rows[rows["stratum"] == st]
                expected[i] += ((x - one["scaler_mean"]) / one["scaler_scale"] * one["coef"]).sum() + one["intercept"].iloc[0]
        assert np.allclose(attribution.predict(heads[(-1, readout)], X, strata), expected / 3)
        # linearity: the averaged head's gradient x input = the mean of the per-seed heads' (test 4)
        A = attribution.direction(heads[(-1, readout)], strata) * X
        A_seeds = np.mean([attribution.direction(per_seed[(-1, readout, r)], strata) * X for r in range(3)], axis=0)
        assert np.allclose(A, A_seeds)
    assert dims == [f"g{i}" for i in range(4)]


def test_pseudobulk_completeness_is_exact():
    heads, _ = attribution.heads(head_table())
    head = heads[(-1, "ridge")]
    x = np.abs(np.random.default_rng(2).normal(size=4))
    A = head["all"][0] * x
    yhat = attribution.predict(head, x[None], ["all"])[0]
    zero = attribution.predict(head, np.zeros((1, 4)), ["all"])[0]
    assert np.isclose(A.sum(), yhat - zero)


GENES = 20


def toy_cells(n_samples=8, cells=40, genes=GENES, seed=0) -> protocol.Cells:
    rng = np.random.default_rng(seed)
    X, sample, ids = [], [], []
    for i in range(n_samples):
        x = np.abs(rng.normal(1.0, 0.7, (cells, genes))) * (rng.random((cells, genes)) > 0.3)
        x[:, 0] += i % 2
        X.append(x)
        sample += [f"s{i}"] * cells
        ids += [f"s{i}_c{j}" for j in range(cells)]
    samples = pd.DataFrame({"group": [i % 2 for i in range(n_samples)], "holdout_group": [f"s{i}" for i in range(n_samples)]},
                           index=[f"s{i}" for i in range(n_samples)])
    return protocol.Cells(X=sp.csr_matrix(np.vstack(X)), genes=[f"g{k}" for k in range(genes)], cell_id=np.array(ids),
                          sample_id=np.array(sample), cell_type=np.array(["t"] * len(ids)),
                          library_size=rng.uniform(500, 3000, len(ids)), samples=samples)


def pseudobulk_of(cells):
    return {s: np.log1p(np.asarray(cells.X[cells.sample_id == s].mean(axis=0)).ravel()) for s in np.unique(cells.sample_id)}


INPUT_CONFIGS = [
    {"cell_normalization": "none"},
    {"cell_normalization": "population"},
    {"cell_normalization": "per_sample"},
    {"cell_normalization": "population", "pseudobulk_gene_scaler": True, "population_distance_feature": True},
    {"cell_normalization": "none", "pseudobulk_gene_scaler": True},
    {"cell_normalization": "per_sample", "population_distance_feature": True},
    {"cell_normalization": "none", "pca": True},
    {"cell_normalization": "population", "pca": True},
]


@pytest.mark.parametrize("config", INPUT_CONFIGS)
def test_torch_inputs_equal_harness_inputs(config):
    genes = 60 if config.get("pca") else GENES   # ScRAT's PCA keeps 50 components
    cells = toy_cells(genes=genes)
    inputs = harness.Inputs(config, cells, pseudobulk_of(cells))
    inputs.prepare(cells)
    torch_inputs = attribution.TorchInputs(inputs)
    for n in (1, 7):   # a 1-cell bag takes the population route
        x = cells.X[cells.sample_id == "s3"].toarray()[:n]
        expected = inputs.cells(x, "s3")
        got = torch_inputs.cells(torch.tensor(x[None], dtype=torch.float64), "s3")[0].numpy()
        assert np.allclose(got, expected, atol=1e-5)
    profile = pseudobulk_of(cells)["s1"]
    assert np.allclose(torch_inputs.profile(torch.tensor(profile[None]))[0].numpy(), inputs.profile(profile), atol=1e-5)


CYTOSET = {"h_dim": 8, "nblock": 1, "pool": "max", "out_pool": "mean", "decoder": "mlp", "ncell": 10}


@pytest.mark.parametrize("config", [c for c in INPUT_CONFIGS if not c.get("pca")])
@pytest.mark.parametrize("weighted", [False, True])
def test_torch_inputs_reproduce_stage06_embeddings(config, weighted):
    """TorchInputs + net.embed on the shared test bags = harness.embed_samples (stage 06's embedding path), and
    the one-profile route = harness.embed_profiles (row B)."""
    cells = toy_cells()
    inputs = harness.Inputs(config, cells, pseudobulk_of(cells))
    torch.manual_seed(0)
    net = cytoset.network({**CYTOSET, "abundance_weighting": weighted}, harness.input_width(config, GENES)).eval()
    ids = ["s0", "s5"]
    expected = harness.embed_samples(net, cells, inputs, ids, CYTOSET["ncell"], "hammond", ("all", 0))
    torch_inputs = attribution.TorchInputs(inputs)
    test = bags.test_bags(cells.cell_id, cells.sample_id, ids, CYTOSET["ncell"], 10, "hammond", ("all", 0))
    for i, s in enumerate(ids):
        rows = np.stack([cells.row_of(b) for b in test[s]])
        x = cells.X[rows.ravel()].toarray().reshape(*rows.shape, -1)
        weights = attribution.depth_weights(cells.library_size[rows])
        _, _, mean_embedding = attribution.gradient_x_input(lambda xt: net.embed(torch_inputs.cells(xt, s), weights), x, {"v": np.ones(8)})
        assert np.abs(mean_embedding - expected[i]).max() <= G.REPRODUCTION_TOLERANCE * max(1.0, np.abs(expected[i]).max())
    profile = pseudobulk_of(cells)["s2"]
    expected = harness.embed_profiles(net, protocol.Profiles(X=sp.csr_matrix(profile[None]), genes=cells.genes, sample_id=np.array(["s2"]),
                                                              library_size=np.ones(1)), inputs)
    got = net.embed(torch_inputs.profile(torch.tensor(profile[None])), torch.ones((1, 1))).detach().numpy()
    assert np.abs(got - expected).max() < 1e-5


def test_bag_attribution_matches_a_linear_network_exactly():
    """embed = mean over cells of x W: A_g = sum_b sum_c x_bcg (W v)_g / (bags x cells), a cell repeated across bags
    counted once per occurrence (test 3)."""
    rng = np.random.default_rng(3)
    W = torch.tensor(rng.normal(size=(5, 3)), dtype=torch.float32)
    x = np.abs(rng.normal(size=(4, 6, 5)))
    x[1, 2] = x[0, 0]   # the same cell in two bags
    v = rng.normal(size=3)
    A, delta, _ = attribution.gradient_x_input(lambda xt: (xt.float() @ W).mean(dim=1), x, {"v": v})
    expected = (x.sum(axis=(0, 1)) * (W.numpy() @ v)) / (4 * 6)
    assert np.allclose(A["v"], expected, atol=1e-5)
    assert np.isclose(A["v"].sum(), delta["v"], atol=1e-5)   # complete for a linear, bias-free network


def test_bag_attribution_equals_per_bag_brute_force_and_finite_differences():
    cells = toy_cells()
    config = {"cell_normalization": "per_sample", "population_distance_feature": True}
    inputs = harness.Inputs(config, cells, pseudobulk_of(cells))
    torch_inputs = attribution.TorchInputs(inputs)
    torch.manual_seed(1)
    net = cytoset.network({**CYTOSET, "out_pool": "mean", "pool": "mean"}, harness.input_width(config, GENES)).eval()
    x = cells.X[cells.sample_id == "s4"].toarray()[:12].reshape(3, 4, GENES).astype(np.float64)
    w = torch.full((3, 4), 0.25)
    v = np.random.default_rng(4).normal(size=8)
    embed = lambda xt: net.embed(torch_inputs.cells(xt, "s4"), w)
    A, _, _ = attribution.gradient_x_input(embed, x, {"v": v})
    brute = np.zeros(GENES)
    for b in range(3):   # each bag on its own, 1/3 of the target each
        xb = torch.tensor(x[b:b + 1], requires_grad=True)
        (net.embed(torch_inputs.cells(xb, "s4"), w[:1]) @ torch.tensor(v, dtype=torch.float32)).sum().backward()
        brute += (xb.grad * xb).detach().sum(dim=(0, 1)).numpy() / 3
    assert np.allclose(A["v"], brute, atol=1e-5)
    target = lambda arr: float((embed(torch.tensor(arr)).mean(dim=0) @ torch.tensor(v, dtype=torch.float32)).detach())
    grad = np.zeros_like(x)
    eps = 1e-3
    for index in [(0, 0, 0), (1, 2, 3), (2, 3, 7)]:
        up, down = x.copy(), x.copy()
        up[index] += eps
        down[index] -= eps
        grad[index] = (target(up) - target(down)) / (2 * eps)
    xt = torch.tensor(x, requires_grad=True)
    (embed(xt).mean(dim=0) @ torch.tensor(v, dtype=torch.float32)).backward()
    for index in [(0, 0, 0), (1, 2, 3), (2, 3, 7)]:
        assert np.isclose(xt.grad[index].item(), grad[index], rtol=2e-2, atol=1e-4)


def test_capped_entries_get_no_gradient():
    cells = toy_cells()
    inputs = harness.Inputs({"cell_normalization": "population"}, cells, {})
    torch_inputs = attribution.TorchInputs(inputs)
    x = cells.X[:3].toarray()[None].astype(np.float64)
    x[0, 0, 1] = 1e6   # far above the +10 cap
    xt = torch.tensor(x, requires_grad=True)
    torch_inputs.cells(xt, "s0").sum().backward()
    assert xt.grad[0, 0, 1] == 0 and xt.grad[0, 0, 2] != 0


def test_scvi_weights_match_the_bag_mean_of_means():
    rng = np.random.default_rng(5)
    ids = np.array([f"c{i}" for i in range(9)])
    bag_list = [ids[[0, 1, 2]], ids[[2, 3]], ids[[2, 4, 5, 6]]]   # unequal sizes (a sample below the bag size)
    z = rng.normal(size=(9, 2))
    expected = np.mean([z[[int(c[1:]) for c in b]].mean(axis=0) for b in bag_list], axis=0)
    w = attribution.scvi_weights(bag_list, ids)
    assert np.allclose(w @ z, expected) and np.isclose(w.sum(), 1.0)


class FakeGeneformer:
    """fm.Geneformer's _encode only: rank by count / median, truncate to input_size - 2, wrap in <cls> <eos>."""

    def __init__(self, input_size):
        from sc2bulk import fm
        self.input_size, self.token = input_size, {"<cls>": 0, "<eos>": 1, "<pad>": 2}
        self._encode = fm.Geneformer._encode.__get__(self)


def test_geneformer_deletion_inputs_and_backfill():
    model = FakeGeneformer(input_size=5)   # room for 3 genes
    tokens, medians = np.array([10, 11, 12, 13, 14]), np.ones(5)
    row = np.array([5.0, 0.0, 3.0, 4.0, 1.0])   # gene 1 absent; ranked 0, 3, 2 kept, 4 truncated
    ids, variants, rank, truncated = attribution.geneformer_inputs(model._encode, row, tokens, medians)
    assert list(ids) == [0, 10, 13, 12, 1] and truncated == 1
    assert [j for j, _ in variants] == [0, 3, 2]                     # only genes in the input; the absent gene 1 never
    assert rank == {0: 1, 3: 2, 2: 3}
    assert dict(variants)[3].tolist() == [0, 10, 12, 14, 1]          # deleting a kept gene backfills the truncated one
    assert {len(v) for _, v in variants} == {len(ids)}               # one length per input: no padding


def test_geneformer_deltas_count_absent_cells_as_zero_and_are_linear_in_the_head():
    model = FakeGeneformer(input_size=10)
    tokens, medians = np.array([10, 11, 12]), np.ones(3)
    table = np.random.default_rng(6).normal(size=(20, 4))
    embed_ids = lambda lists: np.vstack([table[np.asarray(i)].mean(axis=0) for i in lists])
    rows = [np.array([1.0, 2.0, 0.0]), np.array([0.0, 1.0, 3.0])]
    per_input = []
    for row in rows:
        ids, variants, _, _ = attribution.geneformer_inputs(model._encode, row, tokens, medians)
        e0, deltas = attribution.deletion_deltas(embed_ids, ids, variants, batch=2)
        assert np.allclose(e0, embed_ids([ids])[0])
        per_input.append(deltas)
    mean = attribution.mean_deltas(per_input, 3, 4)
    assert np.allclose(mean[0], per_input[0][0] / 2)                 # gene 0 only in the first cell: halved, not averaged over 1
    w1, w2 = np.random.default_rng(7).normal(size=(2, 4))
    assert np.allclose(mean @ (w1 + w2), mean @ w1 + mean @ w2)
    sets = {"both": [0, 1]}
    e0 = [embed_ids([model._encode(r, tokens, medians)[0]])[0] for r in rows]
    joint = attribution.joint_deltas(embed_ids, model._encode, rows, e0, tokens, medians, sets, batch=4)
    by_hand = np.mean([e - embed_ids([model._encode(np.where(np.isin(np.arange(3), [0, 1]), 0, r), tokens, medians)[0]])[0]
                       for e, r in zip(e0, rows)], axis=0)
    assert np.allclose(joint["both"], by_hand)


def test_scgpt_gradient_reaches_values_only_through_the_normalization():
    torch.manual_seed(2)
    vocab, d = 12, 5
    token_embedding, value = torch.nn.Embedding(vocab, d), torch.nn.Linear(1, d)

    def encode(g, v, mask):   # <cls>'s output pools the other unmasked positions, as attention would
        keep = (~mask).float()[..., None]
        keep[:, 0] = 0
        h = torch.tanh(token_embedding(g) + value(v[..., None])) * keep
        return torch.cat([(h.sum(dim=1) / keep.sum(dim=1))[:, None, :], h[:, 1:]], dim=1)
    genes = torch.tensor([[0, 3, 4, 5], [0, 6, 7, 2]])   # 2 = <pad>
    expr = torch.tensor([[-2.0, 3.0, 1.0, 7.0], [-2.0, 2.0, 5.0, -2.0]])
    v = np.random.default_rng(8).normal(size=d)
    weight = np.array([0.5, 0.5])
    A, delta, embedding = attribution.scgpt_gradient_x_input(encode, genes, expr, 2, {"v": v}, weight, autocast=False)
    a = A["v"]
    assert a[0, 0] == 0 and a[1, 0] == 0 and a[1, 3] == 0            # <cls> and padding
    assert np.allclose(np.linalg.norm(embedding, axis=1), 1.0)
    def target(values):
        out = encode(genes, values, genes.eq(2))[:, 0, :]
        out = out / out.norm(dim=1, keepdim=True)
        return float(((out @ torch.tensor(v, dtype=torch.float32)) * torch.tensor(weight, dtype=torch.float32)).sum().detach())
    eps = 1e-3
    up, down = expr.clone(), expr.clone()
    up[0, 2] += eps
    down[0, 2] -= eps
    assert np.isclose(a[0, 2], expr[0, 2].item() * (target(up) - target(down)) / (2 * eps), rtol=2e-2, atol=1e-5)


def samples_table():
    rows = []
    for d in range(12):
        for k in range(1 + d % 3):
            rows.append({"sample_id": f"d{d}_v{k}", "donor_id": f"d{d}", "holdout_group": f"d{d}", "group": d % 2})
    return pd.DataFrame(rows)


def test_deletion_samples_per_donor_nested_and_fold_independent():
    samples = samples_table()
    one, two = (attribution.deletion_samples("soundlife", samples, k, True) for k in (1, 2))
    assert set(one) <= set(two)
    assert len(one) == 12 and all(len([s for s in two if s.startswith(f"d{d}_")]) == min(2, 1 + d % 3) for d in range(12))
    fold = samples[samples["donor_id"] != "d0"]   # another fold's training donors: the same draws for the shared donors
    assert set(attribution.deletion_samples("soundlife", fold, 2, True)) == set(two) - {s for s in two if s.startswith("d0_")}


def test_deletion_cap_round_robin(monkeypatch):
    monkeypatch.setattr(G, "DELETION_CAP", 12)
    samples = samples_table()
    chosen = attribution.deletion_samples("crohns", samples, 2, False)
    assert len(chosen) == 12 and len(set(chosen)) == 12
    donors = {s.split("_")[0] for s in chosen}
    assert donors == set(samples["donor_id"])                           # every donor before any donor's second sample
    assert chosen == attribution.deletion_samples("crohns", samples, 2, False)


def test_fm_cells_pair_level_nested_and_halves():
    ids = [f"c{i}" for i in range(200)]
    first = attribution.fm_cells("soundlife", "s1", ids)
    assert len(first) == G.FM_CELLS and len(set(first)) == G.FM_CELLS
    assert list(attribution.fm_cells("soundlife", "s1", ids[::-1])) == list(first)   # order-free; no fold or model argument
    assert list(attribution.fm_cells("soundlife", "s1", ids, 25)) == list(first[:25])
    screen = attribution.screen_cells("soundlife", "s1", ids, first, 10)
    assert not set(screen) & set(first)


def test_donor_halves_stratified_and_seeded():
    samples = samples_table()
    halves = attribution.donor_halves(samples, "soundlife", ("crossfit", 0))
    by_donor = samples.assign(half=halves.loc[samples["sample_id"]].to_numpy()).groupby("donor_id")["half"].nunique()
    assert (by_donor == 1).all()                                        # a donor's samples stay together
    group = samples.groupby("donor_id")["group"].first()
    half = samples.assign(half=halves.loc[samples["sample_id"]].to_numpy()).groupby("donor_id")["half"].first()
    for g in (0, 1):
        counts = half[group == g].value_counts()
        assert counts.max() - counts.min() <= 1
    assert halves.equals(attribution.donor_halves(samples, "soundlife", ("crossfit", 0)))


def test_ranking_union_of_panels_ties_and_nested_top_k():
    fold0 = [pd.Series({"A": 2.0, "B": 1.0, "C": 1.0}), pd.Series({"A": 4.0, "B": 1.0, "C": 3.0})]
    fold1 = [pd.Series({"A": 1.0, "D": 6.0})]
    score = attribution.across({0: fold0, 1: fold1})
    assert np.isclose(score["A"], (3.0 + 1.0) / 2) and np.isclose(score["D"], 3.0) and np.isclose(score["C"], 1.0)
    ranks = attribution.ranks(pd.Series({"x": 1.0, "b": 2.0, "a": 2.0}))
    assert ranks.to_dict() == {"x": 3, "b": 2, "a": 1}                  # ties by symbol
    s = pd.Series(np.random.default_rng(9).random(200), index=[f"g{i}" for i in range(200)])
    assert attribution.top(s, 25) == attribution.top(s, 50)[:25] == attribution.top(s, 100)[:25]


def test_modality_direction_label_free_per_organ():
    rng = np.random.default_rng(10)
    X_train, X_bulk = rng.normal(size=(6, 3)), rng.normal(2.0, 1.0, size=(4, 3))
    out = attribution.modality_direction(X_train, X_bulk, ["k", "k", "k", "l", "l", "l"], ["k", "l", "k", "l"])
    expected = X_bulk[[0, 2]].mean(axis=0) - X_train[:3].mean(axis=0)
    assert np.allclose(out["k"][0], expected / np.linalg.norm(expected)) and np.isclose(np.linalg.norm(out["l"][0]), 1.0)


def test_level_difference_within_organ():
    X_bulk = np.array([[3.0, 1.0], [5.0, 1.0], [0.0, 2.0], [2.0, 2.0]])
    X_sc = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 2.0], [1.0, 2.0]])
    table = attribution.level_difference(X_bulk, X_sc, ["k", "k", "l", "l"], ["k", "k", "l", "l"], ["A", "B"], {"B"})
    pooled_k = np.sqrt((1 * 2.0 + 1 * 0.0) / 2)
    assert np.isclose(table.loc[0, "delta"], (3.0 + 0.0) / 2) and np.isclose(table.loc[0, "smd"], (3.0 / pooled_k + 0.0) / 2)
    assert table.loc[1, "delta"] == 0 and table.loc[1, "smd"] == 0 and table["marker"].tolist() == [False, True]


def test_matched_random_sets_follow_the_top_bins():
    genes = [f"g{i}" for i in range(500)]
    rng = np.random.default_rng(11)
    frac, rank = pd.Series(rng.random(500), index=genes), pd.Series(rng.random(500), index=genes)
    top_genes = genes[:10]
    grid, bins = attribution.matching_grid(frac, rank, top_genes)
    assert grid == 5 and bins.equals(attribution.matching_bins(frac, rank, 5))   # independent axes: the full grid
    sets = attribution.matched_random_sets(bins, top_genes, 5, np.random.default_rng(0))
    for chosen in sets:
        assert not set(chosen) & set(top_genes) and len(set(chosen)) == 10
        assert sorted(bins.loc[chosen]) == sorted(bins.loc[top_genes])
    assert np.isclose(attribution.additivity(np.array([2.0, 4.0]), np.array([1.0, 2.0])), 2.0)


def test_housekeeping_dominance_statistics():
    attributable = [f"g{i}" for i in range(100)] + ["Rpl3", "mt-Co1"]   # mouse panel genes
    symbols = {**{g: g.upper() for g in attributable[:100]}, "Rpl3": "RPL3", "mt-Co1": "MT-CO1"}
    hk = set(attributable[:20])
    out = attribution.housekeeping(attributable[:10], attributable, hk, symbols, pseudobulk_top=attributable[50:60])
    assert out["hk_fraction"] == 1.0 and np.isclose(out["hk_base_rate"], 20 / 102) and out["hk_p"] < 1e-5
    assert out["hk_excess_vs_pseudobulk"] == 1.0
    assert attribution.housekeeping(["Rpl3", "mt-Co1"], attributable, hk, symbols)["rp_mt_fraction"] == 1.0


def hrt_fixture(tmp_path) -> tuple:
    """A tiny HRT CSV (CRLF, semicolons) and GTF: T1 and T2 are two transcripts of one gene, T3 another gene's
    (versioned gene id), T4 is retired (absent from the GTF)."""
    csv = tmp_path / "hrt.csv"
    csv.write_bytes(b"Ensembl;Gene.name;Refseq;CCDS.ID\r\nENST1;ACTB;NM_1;CCDS1\r\nENST2;ACTB;NM_2;CCDS2\r\n"
                    b"ENST3;C19orf53;NM_3;CCDS3\r\nENST4;OLD1;NM_4;CCDS4\r\n")
    attrs = lambda g, t: f'gene_id "{g}"; transcript_id "{t}"; gene_name "x";'
    lines = ["#!genome-build GRCh38", "\t".join(["7", "ensembl", "gene", "1", "9", ".", "+", ".", 'gene_id "ENSG2";']),
             *["\t".join(["7", "ensembl", feature, "1", "9", ".", "+", ".", attrs(g, t)])
               for feature in ("transcript", "exon") for g, t in (("ENSG2", "ENST1"), ("ENSG2", "ENST2"), ("ENSG1.5", "ENST3"))]]
    gtf = tmp_path / "Homo_sapiens.GRCh38.111.gtf.gz"
    gtf.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode()))
    return csv, gtf


def test_housekeeping_list_conversion_to_gene_ids(tmp_path):
    """scripts/hrt_atlas_ensembl.py: transcripts -> gene ids (version stripped), retired ones counted and dropped,
    one row per gene id sorted, with the header naming both inputs' SHA-256 and the counts."""
    script = runpy.run_path(str(settings.REPO / "scripts" / "hrt_atlas_ensembl.py"), run_name="hrt")
    csv, gtf = hrt_fixture(tmp_path)
    out = tmp_path / "hrt_atlas_v1.tsv"
    assert script["write"](csv, gtf, out) == {"transcripts": 4, "retired": 1, "genes": 2}
    table = pd.read_csv(out, sep="\t", comment="#", keep_default_na=False)
    assert list(table.columns) == ["gene", "ensembl"]
    assert table.values.tolist() == [["C19orf53", "ENSG1"], ["ACTB", "ENSG2"]]
    header = out.read_text()
    assert script["sha256"](csv) in header and script["sha256"](gtf) in header and "1 transcripts retired" in header


def test_housekeeping_matched_on_gene_ids(monkeypatch, tmp_path):
    """Stage 10 reads the list's gene ids, and a panel gene is housekeeping when any of its ids is on the list
    (human: the ';'-joined single-cell ids; mouse: the ortholog), whatever its symbol."""
    S = stage10()
    script = runpy.run_path(str(settings.REPO / "scripts" / "hrt_atlas_ensembl.py"), run_name="hrt")
    out = tmp_path / "hrt_atlas_v1.tsv"
    script["write"](*hrt_fixture(tmp_path), out)
    monkeypatch.setattr(G, "HOUSEKEEPING_FILE", out)
    monkeypatch.setattr(G, "HOUSEKEEPING_SHA256", script["sha256"](out))
    hk = S["housekeeping_set"]()
    assert hk == {"ENSG1", "ENSG2"}
    var = pd.DataFrame({"ensembl": ["ENSG9;ENSG2", "ENSG1", "", "ENSG3"]}, index=["ACTB", "C19ORF53", "NOID", "OTHER"])
    monkeypatch.setattr(folds, "single_cell_genes", lambda pair: var)
    ids = S["human_ensembl"]("soundlife", ("crossfit", 0), list(var.index))
    assert {k for k, i in ids.items() if hk & set(i.split(";"))} == {"ACTB", "C19ORF53"}
    orthologs = tmp_path / "orthologs.csv"
    pd.DataFrame({"gene": ["Actb", "Gm1"], "human_ensembl": ["ENSG2", ""]}).to_csv(orthologs, index=False)
    monkeypatch.setattr(paths, "output", lambda *a: orthologs)
    assert S["human_ensembl"]("hammond", ("all", 0), ["Actb", "Gm1"]) == {"Actb": "ENSG2", "Gm1": ""}


def test_fallback_order():
    days = {"per_donor_2": 4.3, "per_donor_1": 3.1, "cells_25": 2.0, "two_stage_screen": 1.0}
    assert attribution.fallback(days, "fp16")["step"] == "per_donor_1"
    assert attribution.fallback({**days, "per_donor_1": 10.0}, "fp16")["step"] == "two_stage_screen"   # cells_25 only under fp32
    assert attribution.fallback({**days, "per_donor_1": 10.0}, "fp32")["step"] == "cells_25"
    assert attribution.fallback({k: 9.0 for k in days}, "fp32")["step"] == "triage"
    assert [s["step"] for s in G.GF_FALLBACK_ORDER] == ["per_donor_2", "per_donor_1", "cells_25", "two_stage_screen", "triage"]


def test_attribution_and_deletion_purposes_are_stage_10s_only():
    """The `attribution` and `deletion` seed purposes are drawn only by stage 10's files,
    with distinct (purpose, index) pairs: attribution 0 (FM cells), 1 (deletion samples), 2 (donor halves);
    deletion 0/1 (the matched random sets per half) and 10 (the screen's stage-1 cells)."""
    assert {"attribution", "deletion"} <= set(settings.SEED_PURPOSES)
    pattern = re.compile(r"""derived\(\s*["'](attribution|deletion)["']""")
    users = {str(p.relative_to(settings.REPO)) for d in ("src", "pipeline", "figures", "scripts")
             for p in (settings.REPO / d).rglob("*.py") if pattern.search(p.read_text())}
    assert users == {"src/sc2bulk/attribution.py", "pipeline/10_attribution.py"}, users
    calls = set(re.findall(r"""derived\(\s*["'](attribution|deletion)["'],\s*\w+,\s*(\d+|index)""",
                           "".join((settings.REPO / f).read_text() for f in users)))
    assert calls == {("attribution", "0"), ("attribution", "1"), ("attribution", "2"), ("deletion", "10"), ("deletion", "index")}
    assert {T.SEARCH_INDEX_OFFSET, T.SCVI_THIN_OFFSET, T.PERMUTATION_OFFSET, T.MIXUP_OFFSET} >= {1000, 2000}   # other purposes' offsets


def test_stage10_reads_heads_never_predictions_or_labels():
    for name in ("pipeline/10_attribution.py", "src/sc2bulk/attribution.py"):
        text = (settings.REPO / name).read_text()
        assert "_predictions" not in text and "labels" not in re.findall(r"from sc2bulk import ([\w, ]+)", text)[0].split(", ")


@functools.cache
def stage10() -> dict:
    return runpy.run_path(str(settings.REPO / "pipeline" / "10_attribution.py"), run_name="stage10")


def test_top_k_lists_keep_positive_attribution_only(monkeypatch, tmp_path):
    """A sparse averaged LASSO head's top-k lists hold only its nonzero genes (no zero-score padding in symbol
    order), and each list's length is reported."""
    S = stage10()
    g = S["write_ranking"].__globals__
    monkeypatch.setitem(g, "units", lambda pair: [("all", 0)])
    monkeypatch.setitem(g, "out_dir", lambda pair, unit=None: tmp_path)
    written = {}
    monkeypatch.setattr(schemas, "write", lambda kind, frame, path: written.update(frame=frame))
    monkeypatch.setattr(manifest, "finish", lambda *a, **k: None)
    genes = [f"g{i:02d}" for i in range(63)]
    lasso = pd.Series([0.0] * 60 + [3.0, 2.0, 1.0], index=genes)   # 3 genes selected
    ridge = pd.Series(np.arange(63, 0, -1.0), index=genes)
    tops, lengths = S["write_ranking"]("hammond", "pseudobulk", {("lasso", "phenotype"): {("all", 0): [lasso]},
                                                                 ("ridge", "phenotype"): {("all", 0): [ridge]}}, {}, {})
    frame = written["frame"]
    chosen = frame[frame["readout"] == "lasso"]
    assert chosen["top25"].sum() == chosen["top100"].sum() == 3 and set(chosen.loc[chosen["top50"], "gene"]) == {"g60", "g61", "g62"}
    assert tops[("lasso", "phenotype")] == ["g60", "g61", "g62"] and tops[("ridge", "phenotype")] == genes[:50]
    by = {(r["readout"], r["check"]): r["value"] for r in lengths}
    assert by[("lasso", "top50_length")] == 3 and by[("ridge", "top50_length")] == 50 and by[("ridge", "top100_length")] == 63


def test_matching_grid_coarsens_a_thin_corner(monkeypatch):
    """Correlated axes with the top genes in the (most cells, best rank) corner: the finest grid whose
    every top cell holds >= GF_MATCH_POOL x other genes, and random sets matched on it; a pool of exactly 2 x the
    need keeps a grid, one gene fewer fails it."""
    genes = [f"g{i}" for i in range(500)]
    rng = np.random.default_rng(11)
    frac = pd.Series(rng.random(500), index=genes)
    rank = frac + rng.normal(0, 0.15, 500)
    top = genes[:10]
    frac[top], rank[top] = 1 + rng.random(10) * 0.01, -1 + rng.random(10) * 0.01
    def feasible(n):
        bins = attribution.matching_bins(frac, rank, n)
        need = bins.loc[top].value_counts()
        return (bins.drop(top).value_counts().reindex(need.index, fill_value=0) >= G.GF_MATCH_POOL * need).all()
    grid, bins = attribution.matching_grid(frac, rank, top)
    assert grid < 5 and feasible(grid) and not any(feasible(n) for n in G.GF_MATCH_BINS if n > grid)
    assert bins.equals(attribution.matching_bins(frac, rank, grid))
    for chosen in attribution.matched_random_sets(bins, top, 5, np.random.default_rng(0)):
        assert not set(chosen) & set(top) and len(set(chosen)) == 10 and sorted(bins.loc[chosen]) == sorted(bins.loc[top])
    # boundary: 20 genes on a 2 x 2 grid, 2 top genes in one cell holding exactly 4 others; one fewer stops the stage
    small = [f"h{i}" for i in range(20)]
    f2 = pd.Series(np.r_[np.full(6, 0.9) + np.arange(6) * 0.01, np.linspace(0, 0.5, 14)], index=small)
    r2 = pd.Series(np.r_[np.arange(6) * 0.01, np.linspace(0.6, 1, 14)], index=small)
    monkeypatch.setattr(G, "GF_MATCH_BINS", (2,))
    assert attribution.matching_grid(f2, r2, small[:2])[0] == 2   # h2-h5 share the top cell: 4 = 2 x 2
    with pytest.raises(AssertionError, match="no matching grid"):
        attribution.matching_grid(f2.drop(small[5]), r2.drop(small[5]), small[:2])


def gf_part(top_50: float, random: float, n: int, seed: int, cells=True, grid=5) -> dict:
    """One unit's Geneformer check ingredients: n evaluated samples per half with these joint effects."""
    rng = np.random.default_rng(seed)
    genes = [f"g{i:03d}" for i in range(120)]
    score = lambda: pd.Series(rng.random(120), index=genes)
    J = [{"top_50": top_50, **{f"random_{i}": random for i in range(G.GF_RANDOM_SETS)}} for _ in range(n)]
    summed = list(rng.normal(size=n))
    return {"cells": (score(), score()) if cells else None, "seeds": [score() for _ in range(3)], "J": {"a": J, "b": J},
            "joint": {10: summed, 50: [2 * v for v in summed]}, "summed": {10: summed, 50: summed},
            "matching": {"a": (grid, 50), "b": (grid, 49)}}


def test_geneformer_gate_pools_the_folds():
    """Human blood, aging: the gate pools the folds' ingredients (one decision per (configuration, pair) on the
    across-fold footing), so one weak fold doesn't exclude Geneformer by itself; one unit's gate is unchanged."""
    gate = stage10()["gate_rows"]
    weak, strong = gf_part(1.0, 2.0, 3, 0), gf_part(10.0, 1.0, 9, 1, grid=3)
    passed = lambda rows, check: [r["passed"] for r in rows if r["check"] == check]
    assert passed(gate("soundlife", ("crossfit", 0), "geneformer_cells", [weak]), "random_T_top") == [False, False]
    pooled = gate("soundlife", ("crossfit", -1), "geneformer_cells", [weak, strong])
    assert passed(pooled, "random_T_top") == [True, True]
    grid = {r["half"]: r for r in pooled if r["check"] == "random_match_grid"}   # reported: the coarsest over the folds
    assert grid["a"]["value"] == grid["b"]["value"] == 3 and grid["a"]["threshold"] is None and grid["a"]["passed"] is None
    assert [r["value"] for r in pooled if r["check"] == "random_set_size"] == [50, 49]
    assert [r["value"] for r in gate("soundlife", ("crossfit", 0), "geneformer_cells", [weak]) if r["check"] == "random_match_grid"] == [5, 5]
    beta = next(r for r in pooled if r["check"] == "additivity_beta_k50")
    assert np.isclose(beta["value"], 2.0) and beta["passed"]                     # pooled over every fold's evaluated samples
    a = attribution.across({0: [weak["cells"][0]], 1: [strong["cells"][0]]})
    b = attribution.across({0: [weak["cells"][1]], 1: [strong["cells"][1]]})
    expected = attribution.overlap(attribution.top(a, G.TOP_GENES), attribution.top(b, G.TOP_GENES))
    assert next(r for r in pooled if r["check"] == "stability_cells")["value"] == expected
    one = gate("hammond", ("all", 0), "geneformer_pseudobulk", [gf_part(1.0, 1.0, 4, 2, cells=False)])
    seeds_ = [r for r in one if r["check"] == "stability_readout_seeds"][0]
    part = gf_part(1.0, 1.0, 4, 2, cells=False)
    assert seeds_["value"] == attribution.mean_pairwise_overlap(part["seeds"], G.TOP_GENES) and seeds_["threshold"] == G.ATTRIBUTION_CHECKS["stability"]


def test_precision_gate_ignores_genes_absent_at_both_precisions():
    gate = stage10()["precision_gate"]
    rng = np.random.default_rng(0)
    fp32 = np.zeros((4, 200))
    fp32[:, :20] = rng.normal(size=(4, 20))
    fp16 = fp32.copy()
    fp16[:, :20] = rng.normal(size=(4, 20))                          # the expressed genes disagree completely
    assert abs(gate(fp16, fp32)["spearman"]) < 0.5                   # tied zeros (180 absent genes) don't lift rho


def test_genes_sharing_a_token_are_credited_to_none(monkeypatch, tmp_path):
    """Genes stage 07 summed into one token keep their token in the input but get no credit, like an
    unreadable gene; the configuration's creditable genes are the union over the units."""
    S = stage10()
    g = S["readable"].__globals__
    maps = {("crossfit", 0): pd.DataFrame({"gene": ["A", "B", "C", "D", "E"], "target": [7, 7, 3, -1, 5],
                                           "readable": [True, True, True, False, True]}),
            ("crossfit", 1): pd.DataFrame({"gene": ["A", "B", "C", "D", "E"], "target": [7, 9, 3, -1, 3],
                                           "readable": [True, True, True, False, True]})}
    for unit, frame in maps.items():
        frame.to_csv(tmp_path / f"{unit[1]}.csv", index=False)
    monkeypatch.setitem(g, "paths", type("P", (), {"output": staticmethod(lambda stage, pair, name, unit: tmp_path / f"{unit[1]}.csv")}))
    monkeypatch.setitem(g, "training_inputs", type("T", (), {"panel": staticmethod(lambda pair, unit: ["A", "B", "C", "D", "E"])}))
    monkeypatch.setitem(g, "units", lambda pair: list(maps))
    table = S["readable"]("soundlife", ("crossfit", 0), "geneformer")
    assert list(table["credited"]) == [False, False, True, False, True]
    assert S["token_genes"](table) == {3: 2, 5: 4}          # the shared token 7 and the unreadable gene map to no gene
    assert list(table["target"]) == [7, 7, 3, -1, 5]       # the input map (collapsed()) is unchanged
    assert S["attributable_genes"]("soundlife", "geneformer_cells") == ["A", "B", "C", "E"]   # A, B: credited in fold 1; C, E: in fold 0


def test_scgpt_chunks_are_exact(monkeypatch):
    """Scgpt_sample in chunks of SCGPT_CHUNK inputs gives one pass's attribution, deltas and embeddings
    (a partial last chunk, unequal lengths so chunks pad differently, unequal weights, two targets, a token with no
    credited gene), through real attention with a key-padding mask."""
    torch.manual_seed(3)
    vocab, d, pad = 20, 8, 1
    layer = torch.nn.TransformerEncoderLayer(d, 2, 16, dropout=0.0, batch_first=True)
    S = stage10()
    device = S["DEVICE"]
    encoder, token_embedding, value = (m.to(device) for m in (torch.nn.TransformerEncoder(layer, 2).eval(), torch.nn.Embedding(vocab, d), torch.nn.Linear(1, d)))
    encode = lambda g, v, mask: encoder(token_embedding(g) + value(v[..., None]), src_key_padding_mask=mask)
    model = type("M", (), {"vocab": {"<pad>": pad}, "configs": {"pad_token": "<pad>", "pad_value": -2.0},
                           "model": type("I", (), {"_encode": staticmethod(encode)})})
    rng = np.random.default_rng(4)
    collated = [(torch.tensor([0, *rng.choice(np.arange(2, vocab), n, replace=False)]), torch.tensor([0.0, *rng.integers(1, 6, n)], dtype=torch.float32))
                for n in (3, 7, 4, 6, 5)]
    token_gene = {t: t - 2 for t in range(2, vocab - 1)}   # token vocab - 1: no credited gene
    targets = {"a": rng.normal(size=d), "b": rng.normal(size=d)}
    weight = rng.dirichlet(np.ones(5))
    run = lambda: S["scgpt_sample"](model, collated, targets, weight, False, token_gene, vocab)
    monkeypatch.setattr(G, "SCGPT_CHUNK", 5)
    whole = run()
    monkeypatch.setattr(G, "SCGPT_CHUNK", 2)
    chunked = run()
    for name in targets:
        assert np.allclose(whole[0][name], chunked[0][name], atol=1e-6) and np.abs(whole[0][name]).sum() > 0
        assert np.isclose(whole[1][name], chunked[1][name], atol=1e-6)
    assert np.allclose(whole[2], chunked[2], atol=1e-6)    # embeddings in input order


def test_geneformer_files_url_encode_sample_ids(monkeypatch, tmp_path):
    """A "/" in a sample id (mouse organs, e.g. Lung_3-M-5/6) is URL-encoded in its file's name, and
    every reader finds the files by the deletion plan's ids (a stale file is never read; a missing one is named)."""
    S = stage10()
    g = S["gf_file"].__globals__
    ids = ["Liver_18-F-51", "Lung_3-M-5/6"]
    monkeypatch.setitem(g, "out_dir", lambda pair, unit=None: tmp_path)
    monkeypatch.setitem(g, "deletion_plan", lambda pair, unit: ({}, "fp16", ids))
    monkeypatch.setitem(g, "folds", type("F", (), {"training_samples": staticmethod(lambda pair, *unit: pd.DataFrame({"sample_id": ids}))}))
    monkeypatch.setitem(g, "training_inputs", type("T", (), {"panel": staticmethod(lambda pair, unit: ["a", "b", "c"])}))
    monkeypatch.setitem(g, "strata", lambda pair, table: np.full(len(table), "all"))
    unit, key = ("all", 0), "geneformer_cells"
    folder = tmp_path / "geneformer" / key
    assert S["gf_file"]("tms", unit, key, "delta", "Lung_3-M-5/6") == folder / "delta" / "Lung_3-M-5%2F6.npz"
    assert S["gf_file"]("tms", unit, key, "joint", "Lung_3-M-5/6", "a") == folder / "joint" / "Lung_3-M-5%2F6_a.npz"
    assert S["gf_file"]("tms", unit, key, "delta", "Liver_18-F-51").name == "Liver_18-F-51.npz"
    for sub in ("delta", "screen1"):
        (folder / sub).mkdir(parents=True)
        for i, s in enumerate(ids):
            np.savez(S["gf_file"]("tms", unit, key, sub, s), gene_idx=np.array([i, 2]), delta_A=np.full((2, 1), i + 1.0),
                     delta_B=np.zeros((2, 0)), n_present=np.array([1.0, 2.0]), rank_sum=np.array([3.0, 4.0]), n_inputs=2, n_half_a=2)
    np.savez(folder / "delta" / ".tmp.stale.npz", gene_idx=np.array([0]))   # never read
    head = {"all": [np.array([2.0])]}
    got, D, _, present, _ = S["gf_attributions"]("tms", unit, key, {"ridge": head}, "delta")
    assert got == ids and np.allclose(D["ridge"], [[2.0, 0, 2.0], [0, 4.0, 4.0]]) and present.sum() == 4
    fraction, rank = S["joint_stats"]("tms", unit, key, "delta", ids, ["a", "b", "c"])(ids)
    assert np.allclose(fraction, [0.25, 0.25, 1.0]) and np.allclose(rank, [3.0, 3.0, 2.0])
    assert np.allclose(S["screen_scores"]("tms", unit, key, {(-1, "ridge"): head}), [1.0, 2.0, 3.0])
    S["gf_file"]("tms", unit, key, "delta", "Lung_3-M-5/6").unlink()
    with pytest.raises(AssertionError, match="Lung_3-M-5%2F6"):
        S["gf_attributions"]("tms", unit, key, {"ridge": head}, "delta")
    text = (settings.REPO / "pipeline" / "10_attribution.py").read_text()
    assert 'glob("*.npz")' not in text and 'f"{s}.npz"' not in text and 'f"{s}_{h}.npz"' not in text


def test_untrainable_seed_is_skipped_by_stages_08_and_10(monkeypatch):
    """Mouse organs' scVI seed 14 is left out by stage 08 (scoring) and stage 10 (expected seeds);
    every other pair and configuration keeps stage 06's seeds, and stage 06's rule itself is unchanged."""
    from sc2bulk import training_inputs
    monkeypatch.setattr(settings, "SMOKE", False)
    S = stage10()
    S08 = S["S08"]
    assert S["TRAINED"] == S08["TRAINED"]
    assert S["expected_seeds"]("tms", "scvi_cells") == [e for e in range(20) if e != 14]
    assert S["expected_seeds"]("tms", "cytoset_cells") == list(range(20))
    assert S["expected_seeds"]("lupus", "scvi_cells") == list(range(20))
    assert S["expected_seeds"]("tms", "pseudobulk") == [-1]
    assert training_inputs.encoder_seeds("tms", "J") == list(range(20))
    names = [p.name for p in S08["inputs"]("tms", ("all", 0), "scvi_cells")]
    assert "s13.h5ad" in names and "s14.h5ad" not in names
