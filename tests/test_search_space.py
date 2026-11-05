"""The search grids and ScRAT's PCA input."""
import os

import numpy as np
import pytest
import torch

from sc2bulk import harness
from sc2bulk.config import search_space as S
from sc2bulk.config import training as T
from sc2bulk.encoders import set_transformer
from test_harness import GENES, toy


@pytest.mark.parametrize("encoder", ["cytoset", "set_transformer", "scrat"])
def test_counts_and_one_change_each(encoder):
    for pair, expected in (("hammond", 26), ("lupus", 25 if encoder == "cytoset" else 26)):
        groups = S.candidates(encoder, pair)
        names = [name for group in groups.values() for name in group]
        assert len(names) == expected and len(set(names)) == expected
        base = S.base(encoder)
        for group in groups.values():
            for name, change in group.items():
                assert set(change) <= set(base), name                        # a setting the base has
                assert any(base[k] != v for k, v in change.items()), name     # and a real change


def test_scvi_candidates_are_single_real_changes():
    names = [n for group in S.SCVI_GROUPS.values() for n in group]
    assert len(names) == len(set(names)) == 3
    for group in S.SCVI_GROUPS.values():
        for name, change in group.items():
            assert set(change) <= set(S.SCVI_BASE) and any(S.SCVI_BASE[k] != v for k, v in change.items()), name


def test_cytoset_bases_agree():
    shared = set(S.BASE) & set(T.BASES["cytoset"])
    assert shared and all(S.BASE[k] == T.BASES["cytoset"][k] for k in shared)


def test_batch_sizes_are_multiples_of_eight():
    for encoder in ("cytoset", "set_transformer", "scrat"):
        sizes = [c["batch_size"] for c in S.GROUPS_BY_ENCODER[encoder]["batch_size"].values()] + [S.base(encoder)["batch_size"]]
        assert all(b % 8 == 0 for b in sizes), encoder


def test_fig4a_bag_sizes_match_the_fm_sweep():
    assert S.FIG4A_BAG_SIZES[:-1] == (1, 25, 50, 100, 200, 300, 500, 600, 1000) and S.FIG4A_BAG_SIZES[-1] == "all"


def test_every_set_transformer_candidate_changes_the_network():
    base = {**S.base("set_transformer"), "dim_hidden": 16, "num_inds": 4}   # base heads (4) kept
    x = torch.rand(3, 7, GENES)

    def output(config):
        torch.manual_seed(0)
        net = set_transformer.network(config, GENES).train()   # train mode: dropout acts
        torch.manual_seed(1)
        with torch.no_grad():
            return net.head(net.embed(x, None))
    reference = output(base)
    network_keys = {"dim_hidden", "num_heads", "num_inds", "encoder_layers", "ln", "encoder_block", "decoder_sab", "dropout"}
    for group in S.SET_TRANSFORMER_GROUPS.values():
        for name, change in group.items():
            if set(change) & network_keys:
                assert not torch.allclose(output({**base, **change}), reference), name


def test_scrat_base_and_changes():
    # the base reads the PCA input; norm_per_sample is the one two-setting candidate
    base = S.base("scrat")
    assert base["pca"] is True
    for group in S.SCRAT_GROUPS.values():
        for name, change in group.items():
            changed = {k for k, v in change.items() if base[k] != v}
            assert changed == {"cell_normalization", "pca"} if name == "norm_per_sample" else len(changed) == len(change) == 1, name


def test_pca_inputs(monkeypatch):
    monkeypatch.setattr(T, "SCRAT_PCA_COMPONENTS", 5)   # the toy has 20 genes
    cells = toy()
    for level in ("none", "population"):
        inputs = harness.Inputs({"cell_normalization": level, "pca": True}, cells, {})
        c = inputs.components
        assert c.shape == (GENES, 5)
        assert np.allclose(c.T @ c, np.eye(c.shape[1]), atol=1e-8)                       # orthonormal
        assert (c[np.abs(c).argmax(axis=0), np.arange(c.shape[1])] > 0).all()             # sign rule
        z = np.vstack([inputs._standardize(cells.X[cells.sample_id == s].toarray(), own=False)
                       for s in np.unique(cells.sample_id)])
        projected = (z - z.mean(axis=0)) @ c
        variances = projected.var(axis=0)
        assert np.all(np.diff(variances) <= 1e-9)                                        # descending
        assert np.allclose(variances, np.sort(np.linalg.eigvalsh(np.cov(z.T, bias=True)))[::-1][:c.shape[1]])
        x = cells.X[:5].toarray()
        assert np.allclose(inputs.profile(x[0])[0], inputs.cells(x, "s0")[0])             # same projection
        resident = harness.Resident(cells, inputs, "cpu")
        ids = np.sort(cells.cell_id[cells.sample_id == "s3"][:7])
        groups = resident.indices([("s3", ids)])
        expected = inputs.cells(cells.X[cells.row_of(ids)].toarray(), "s3")
        assert np.allclose(resident.gather(groups[7][0])[0][0].numpy(), expected, atol=1e-5)
    with pytest.raises(AssertionError):
        harness.Inputs({"cell_normalization": "per_sample", "pca": True}, cells, {})


@pytest.mark.skipif(not os.environ.get("SC2BULK_SCRAT"), reason="needs a ScRAT clone (SC2BULK_SCRAT)")
def test_every_scrat_candidate_changes_the_network():
    from sc2bulk.encoders import scrat
    base = {**S.base("scrat"), "model_dim": 32}
    x = torch.rand(3, 7, GENES)

    def output(config):
        torch.manual_seed(0)
        in_dim = T.SCRAT_PCA_COMPONENTS if config["pca"] else GENES
        net = scrat.network(config, GENES, os.environ["SC2BULK_SCRAT"]).train()   # train mode: dropout acts
        torch.manual_seed(1)
        with torch.no_grad():
            return net.head(net.embed(torch.rand(3, 7, in_dim) if config["pca"] else x, None))
    reference = output(base)
    network_keys = {"model_dim", "num_layers", "num_heads", "dropout", "norm_first", "pca", "input_dropout"}
    for group in S.SCRAT_GROUPS.values():
        for name, change in group.items():
            if set(change) & network_keys:
                assert not torch.allclose(output({**base, **change}), reference), name
