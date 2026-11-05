"""The training harness (sc2bulk.harness) on a tiny synthetic pair, on CPU: encoders learn a
real signal and pass "did it train?", fail it on permuted labels, train deterministically, and
mixed-size batches equal bag-by-bag computation; normalization and the bag drawer behave as specified."""
import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from sc2bulk import bags, harness, protocol
from sc2bulk.encoders import cytoset, set_transformer

GENES = 20


def toy(n_samples=20, cells=60, shift=1.5, seed=0, permute=False):
    """Samples s0.. with alternating groups; group 1's cells have gene 0 raised by `shift`."""
    rng = np.random.default_rng(seed)
    groups = np.array([i % 2 for i in range(n_samples)])
    if permute:
        groups = rng.permutation(groups)
    true = np.array([i % 2 for i in range(n_samples)])
    X, sample, ids = [], [], []
    for i in range(n_samples):
        x = np.abs(rng.normal(1.0, 0.5, (cells, GENES)))
        x[:, 0] += shift * true[i]
        X.append(x)
        sample += [f"s{i}"] * cells
        ids += [f"s{i}_c{j}" for j in range(cells)]
    samples = pd.DataFrame({"group": groups, "holdout_group": [f"s{i}" for i in range(n_samples)]},
                           index=[f"s{i}" for i in range(n_samples)])
    return protocol.Cells(X=sp.csr_matrix(np.vstack(X)), genes=[f"g{k}" for k in range(GENES)], cell_id=np.array(ids),
                          sample_id=np.array(sample), cell_type=np.array(["t"] * len(ids)),
                          library_size=np.full(len(ids), 1000.0), samples=samples)


def split(cells, validation):
    keep = ~np.isin(cells.sample_id, validation)
    part = lambda m, rows: protocol.Cells(X=cells.X[m], genes=cells.genes, cell_id=cells.cell_id[m], sample_id=cells.sample_id[m],
                                          cell_type=cells.cell_type[m], library_size=cells.library_size[m], samples=cells.samples.loc[rows])
    train_rows = [s for s in cells.samples.index if s not in validation]
    return part(keep, train_rows), part(~keep, list(validation))


CONFIGS = {
    "cytoset": {"h_dim": 16, "nblock": 1, "pool": "max", "out_pool": "mean", "decoder": "mlp", "lr": 1e-2, "beta1": 0.9, "beta2": 0.999,
                "wts_decay": 0.0, "batch_size": 16, "ncell": 20, "nsubset": 64, "patience": 3, "n_epochs": 20,
                "cell_normalization": "none"},
    "set_transformer": {"dim_hidden": 16, "num_heads": 2, "num_inds": 4, "ln": False, "dropout": 0.1, "lr": 1e-2,
                        "encoder_block": "isab", "encoder_layers": 2, "decoder_sab": False,
                        "beta1": 0.9, "beta2": 0.999, "wts_decay": 0.0, "batch_size": 16, "ncell": 20, "nsubset": 64,
                        "patience": 3, "n_epochs": 20, "cell_normalization": "none"},
}
MODULES = {"cytoset": cytoset, "set_transformer": set_transformer}
VALIDATION = [f"s{i}" for i in range(10)]   # 5 + 5 holdout groups: 252 labellings, testable


def fit(name, permute=False):
    cells = toy(permute=permute)
    train, val = split(cells, VALIDATION)
    config = CONFIGS[name]
    inputs = harness.Inputs(config, train, {})
    return harness.train(lambda: MODULES[name].network(config, GENES), config, train, val, inputs,
                         init_seed=1, bag_seed=2, val_seed=3)[1:]


@pytest.mark.parametrize("name", MODULES)
def test_learns_and_passes(name):
    _, record, _, _ = fit(name)
    assert record.did_train, record.did_train_reason
    assert record.stop_epoch >= 10                      # the burn-in


@pytest.mark.parametrize("name", MODULES)
def test_permuted_labels_fail(name):
    _, record, _, _ = fit(name, permute=True)
    assert not record.did_train, record.did_train_reason


def test_deterministic():
    a, record_a, _, _ = fit("cytoset")
    b, record_b, _, _ = fit("cytoset")
    assert all(torch.equal(a[k], b[k]) for k in a) and record_a.validation_curve == record_b.validation_curve


def test_init_seed_sets_initial_weights():
    cells = toy()
    train, val = split(cells, VALIDATION)
    config = {**CONFIGS["cytoset"], "n_epochs": 10}
    first = lambda seed: harness.train(lambda: cytoset.network(config, GENES), config, train, val, harness.Inputs(config, train, {}),
                                       init_seed=seed, bag_seed=2, val_seed=3)[1]
    a, b, c = first(1), first(1), first(5)
    assert all(torch.equal(a[k], b[k]) for k in a) and not all(torch.equal(a[k], c[k]) for k in a)


@pytest.mark.parametrize("name", MODULES)
def test_mixed_sizes_equal_bag_by_bag(name):
    cells = toy()
    inputs = harness.Inputs(CONFIGS[name], cells, {})
    torch.manual_seed(0)
    net = MODULES[name].network(CONFIGS[name], GENES).eval()
    rng = np.random.default_rng(0)
    mixed = [(s, np.sort(rng.choice(cells.cell_id[cells.sample_id == s], n, replace=False)))
             for s, n in (("s0", 5), ("s1", 9), ("s2", 5), ("s3", 1))]
    resident = harness.Resident(cells, inputs, "cpu")
    with torch.no_grad():
        together, _ = harness.forward_bags(net, mixed, resident)
        alone = torch.cat([harness.forward_bags(net, [bag], resident)[0] for bag in mixed])
    assert torch.allclose(together, alone, atol=1e-6)


def test_host_gather_equals_device_gather(monkeypatch):
    cells = toy()
    inputs = harness.Inputs({"cell_normalization": "per_sample", "population_distance_feature": True}, cells, {})
    device = harness.Resident(cells, inputs, "cpu", harness.fits_on_device(cells))
    monkeypatch.setattr(harness.T, "RESIDENT_MAX_BYTES", 0)
    host = harness.Resident(cells, inputs, "cpu", harness.fits_on_device(cells))
    assert device.on_device and not host.on_device
    ids = np.sort(cells.cell_id[cells.sample_id == "s2"][:9])
    groups = device.indices([("s2", ids)])
    assert torch.allclose(device.gather(groups[9][0])[0], host.gather(groups[9][0])[0], atol=1e-5)


@pytest.mark.parametrize("level", ["none", "population", "per_sample"])
@pytest.mark.parametrize("extras", [{}, {"population_distance_feature": True}])
def test_resident_equals_per_bag_transform(level, extras):
    # the device-resident gather equals transforming each bag on the host (Inputs.cells)
    cells = toy()
    config = {"cell_normalization": level, **extras}
    inputs = harness.Inputs(config, cells, {})
    resident = harness.Resident(cells, inputs, "cpu")
    rng = np.random.default_rng(1)
    for n in (1, 7):
        ids = np.sort(rng.choice(cells.cell_id[cells.sample_id == "s3"], n, replace=False))
        groups = resident.indices([("s3", ids)])
        x, w = resident.gather(groups[n][0])
        expected = inputs.cells(cells.X[cells.row_of(ids)].toarray(), "s3")
        assert np.allclose(x[0].numpy(), expected, atol=1e-5)
        assert np.isclose(float(w.sum()), 1.0)


def test_normalization_levels():
    cells = toy()
    x = cells.X[:30].toarray()
    per = harness.Inputs({"cell_normalization": "per_sample"}, cells, {})
    assert np.allclose(per.cells(x, "s0").mean(axis=0), 0, atol=1e-6)
    pop = harness.Inputs({"cell_normalization": "population"}, cells, {})
    assert np.allclose(pop.profile(x[0]), np.clip((x[0] - pop.mean) / pop.sd, -10, 10))
    assert np.abs(pop.cells(x * 100, "s0")).max() <= 10.0      # the cap
    assert np.array_equal(per.profile(x[0]), pop.profile(x[0]))  # one-profile inputs: population moments


def test_train_bags_even_and_balanced():
    cells = toy()
    groups = cells.samples["group"].to_dict()
    epoch = bags.train_bags(cells.cell_id, cells.sample_id, groups, 10, 64, np.random.default_rng(0))
    counts = pd.Series([s for s, _ in epoch]).value_counts()
    assert set(counts) == {6} and len(epoch) == 120              # 64 per class over 10 samples: 6 each
    fixed = bags.fixed_bags(cells.cell_id, cells.sample_id, ["s1", "s0"], 10, 3, np.random.default_rng(0))
    assert [s for s, _ in fixed] == ["s0"] * 3 + ["s1"] * 3


@pytest.mark.skipif(not os.environ.get("SC2BULK_SCRAT"), reason="needs a ScRAT clone (SC2BULK_SCRAT)")
def test_scrat_learns():
    from sc2bulk.encoders import scrat
    cells = toy()
    train, val = split(cells, VALIDATION)
    config = {"model_dim": 16, "num_heads": 2, "num_layers": 1, "dropout": 0.1, "input_dropout": 0.0, "pca": False, "norm_first": False,
              **{k: CONFIGS["cytoset"][k] for k in ("lr", "beta1", "beta2", "wts_decay", "batch_size", "ncell", "nsubset",
                                                    "patience", "n_epochs", "cell_normalization")}}
    _, _, record, _, _ = harness.train(lambda: scrat.network(config, GENES, os.environ["SC2BULK_SCRAT"]), config, train, val,
                                       harness.Inputs(config, train, {}), 1, 2, 3)
    assert record.did_train, record.did_train_reason


def test_every_base_is_complete():
    from sc2bulk.config import training
    shared = {"lr", "beta1", "beta2", "wts_decay", "batch_size", "ncell", "nsubset", "patience", "n_epochs", "cell_normalization"}
    for name, base in training.BASES.items():
        assert shared <= set(base), (name, shared - set(base))



def test_did_it_train_tolerates_overconfidence():
    # 6 groups per class, 5 bags each: five groups per class confidently right, one per class wrong
    # (mean d about 0.26, but a large spread: the old 2-SE rule's threshold was about 0.6); the
    # permutation rule passes it, and logits unrelated to the labels fail
    import pandas as pd
    groups = [f"g{i}" for i in range(12)]
    samples = pd.DataFrame({"group": [i % 2 for i in range(12)], "holdout_group": groups}, index=groups)
    val_bags = [(g, None) for g in groups for _ in range(5)]
    y = np.array([samples.loc[g, "group"] for g, _ in val_bags])
    logits = np.where(y == 1, 6.0, -6.0)
    logits[:10] = np.where(y[:10] == 1, -2.5, 2.5)            # groups g0 and g1: wrong
    passed, d, d_star, t_star, p, testable, _ = harness.did_it_train(logits, val_bags, samples, seed=0)
    assert testable and passed and 0.2 < d < 0.3 and p <= 0.05, (d, p)
    noise = np.random.default_rng(0).normal(0, 3, len(y))
    assert not harness.did_it_train(noise, val_bags, samples, seed=0)[0]



def test_collapse_screen_is_scale_invariant():
    # d* is d's maximum over the temperature grid, so sharper or flatter logits of the same
    # ranking give the same screen (within the grid's bounds); a constant output never clears it
    groups = [f"g{i}" for i in range(12)]
    samples = pd.DataFrame({"group": [i % 2 for i in range(12)], "holdout_group": groups}, index=groups)
    val_bags = [(g, None) for g in groups for _ in range(5)]
    y = np.array([samples.loc[g, "group"] for g, _ in val_bags])
    rng = np.random.default_rng(1)
    logits = np.where(y == 1, 0.8, -0.8) + rng.normal(0, 1.0, len(y))
    out = [harness.did_it_train(logits * scale, val_bags, samples, seed=0) for scale in (0.5, 1.0, 3.0)]
    d_stars = [o[2] for o in out]
    assert max(d_stars) - min(d_stars) < 0.01 * max(d_stars)        # the grid's resolution
    assert all(o[2] >= o[1] for o in out)                           # d* >= d (T = 1 is on the grid)
    assert abs(out[2][1] - out[0][1]) > 0.05 * max(d_stars)          # while raw d moves with the scale
    assert 1.0 in harness.T.TEMPERATURES and len(harness.T.TEMPERATURES) == 61
    for constant in (0.0, 0.7, -2.0):
        assert harness.did_it_train(np.full(len(y), constant), val_bags, samples, seed=0)[2] <= 1e-12


def test_too_few_labellings_is_not_testable():
    groups = ["a", "b", "c", "d"]   # 2 + 2 groups: 6 labellings, below MIN_TESTABLE
    samples = pd.DataFrame({"group": [0, 1, 0, 1], "holdout_group": groups}, index=groups)
    val_bags = [(g, None) for g in groups for _ in range(3)]
    y = np.array([samples.loc[g, "group"] for g, _ in val_bags])
    passed, _, _, _, p, testable, _ = harness.did_it_train(np.where(y == 1, 5.0, -5.0), val_bags, samples, seed=0)
    assert passed and not testable and np.isnan(p)                       # the collapse screen alone
    assert not harness.did_it_train(np.zeros(len(y)), val_bags, samples, seed=0)[0]   # stuck at the constant



def test_every_cytoset_architecture_candidate_changes_the_network():
    # each search candidate that changes the network gives a different output from the base on the
    # same input and init seed (catches candidates the adapter ignores)
    from sc2bulk.config import search_space
    network_keys = {"h_dim", "nblock", "pool", "out_pool", "decoder", "abundance_weighting"}
    base = {**search_space.BASE}
    x = torch.rand(3, 7, GENES)
    w = torch.rand(3, 7)
    w = w / w.sum(dim=1, keepdim=True)   # unequal depth weights (equal ones make abundance weighting a plain mean)

    def output(config):
        torch.manual_seed(0)
        net = cytoset.network(config, GENES).eval()
        with torch.no_grad():
            return net.head(net.embed(x, w))
    reference = output(base)
    for group, candidates in search_space.GROUPS.items():
        for name, change in candidates.items():
            if set(change) & network_keys:
                assert not torch.allclose(output({**base, **change}), reference), name


class _Identity(torch.nn.Module):
    """A network whose embedding is the bag's first cell's input (to check positions end to end)."""

    def embed(self, x, w):
        return x[:, 0, :]

    def head(self, e):
        return e[:, 0]


@pytest.mark.parametrize("config", [{"cell_normalization": "none", "pseudobulk_gene_scaler": True},
                                    {"cell_normalization": "population", "pseudobulk_gene_scaler": True,
                                     "population_distance_feature": True},
                                    {"cell_normalization": "per_sample", "population_distance_feature": True}])
def test_resident_mixed_sizes_order_scaler_and_weights(config):
    # one shuffled list of bags from several samples, sizes 1, 7 and "all cells" (60); each gathered
    # bag equals the host transform, its weights equal depth_weights, and forward keeps the order
    cells = toy()
    cells.library_size = np.random.default_rng(3).uniform(500, 5000, len(cells.cell_id))
    pseudobulk = {s: np.log1p(cells.X[cells.sample_id == s].toarray().sum(axis=0)) for s in cells.samples.index}
    inputs = harness.Inputs(config, cells, pseudobulk)
    resident = harness.Resident(cells, inputs, "cpu")
    rng = np.random.default_rng(4)
    bag_list = [(s, np.sort(rng.choice(cells.cell_id[cells.sample_id == s], n, replace=False)))
                for s, n in (("s1", 7), ("s4", 1), ("s2", 60), ("s1", 1), ("s7", 7), ("s3", 60))]
    bag_list = [bag_list[i] for i in rng.permutation(len(bag_list))]
    groups = resident.indices(bag_list)
    for n, (idx, positions) in groups.items():
        x, w = resident.gather(idx)
        for k, p in enumerate(positions):
            s, ids = bag_list[p]
            rows = cells.row_of(ids)
            assert np.allclose(x[k].numpy(), inputs.cells(cells.X[rows].toarray(), s), atol=1e-5)
            depth = np.log1p(cells.library_size[rows])
            assert np.allclose(w[k].numpy(), depth / depth.sum(), atol=1e-6)   # depth weights, value by value
    e, _ = harness.forward(_Identity(), groups, len(bag_list), resident, embed_only=True)
    for p, (s, ids) in enumerate(bag_list):
        first = inputs.cells(cells.X[cells.row_of(ids)].toarray(), s)[0]
        assert np.allclose(e[p].numpy(), first, atol=1e-5), p



def test_every_search_setting_is_read():
    # each setting a search candidate changes is read by the CytoSet network, the harness or its inputs
    # (catches candidates that would train the base: decoder_linear, the age loss)
    from pathlib import Path as P
    from sc2bulk.config import search_space
    source = "".join((P(harness.__file__).parent / f).read_text() for f in ("harness.py", "encoders/cytoset.py"))
    for candidates in search_space.GROUPS.values():
        for name, change in candidates.items():
            for key in change:
                assert f'"{key}"' in source, (name, key)



def test_untestable_units_match_stage_03():
    # training.UNTESTABLE_UNITS is recomputed from stage 03's search splits when present
    from math import comb
    from sc2bulk.config import settings, training
    from sc2bulk.config.pairs import PAIR_ORDER
    found = set()
    for pair in PAIR_ORDER:
        path = settings.OUTPUTS / "03_splits" / pair / "splits.csv"
        if not path.exists():
            pytest.skip("stage 03 outputs not present")
        t = pd.read_csv(path)
        v = t[(t["side"] == "validation") & (t["purpose"] == "search_split")]
        counts = v.groupby(["fold_set", "fold", "index", "group"])["holdout_group"].nunique().unstack("group")
        for (fold_set, fold), part in counts.groupby(level=["fold_set", "fold"]):
            if min(comb(int(a + b), int(b)) for a, b in zip(part[0], part[1])) < training.MIN_TESTABLE:
                found.add((pair, fold_set, int(fold)))
    assert found == training.UNTESTABLE_UNITS


def test_micro_batch_is_the_largest_divisor_under_the_cap():
    assert harness.micro_batch(200, 64) == 50 and harness.micro_batch(256, 64) == 64 and harness.micro_batch(32, 64) == 32
    assert harness.micro_batch(200, 7) == 5 and all(200 % harness.micro_batch(200, c) == 0 for c in range(1, 300))


def test_permutation_p_uses_raw_logits():
    # 5 v 4 groups (126 labellings, enumerated): p is the share of labellings whose raw-logit d is >= the
    # observed one; confidently wrong groups make the rescaled-logit p differ, so this pins raw logits
    from itertools import combinations
    groups = [f"g{i}" for i in range(9)]
    label = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    samples = pd.DataFrame({"group": label, "holdout_group": groups}, index=groups)
    val_bags = [(g, None) for g in groups for _ in range(4)]
    rng = np.random.default_rng(4)
    member = np.repeat(np.arange(9), 4)
    logits = np.where(label[member] == 1, 4.0, -4.0) + rng.normal(0, 2.0, len(member))
    logits[member == 0] = 6.0                                                   # g0 confidently wrong
    logits[member == 5] = -5.0                                                  # g5 confidently wrong

    def d(lab):
        per = np.array([np.mean(harness.LN2 - harness.bce(logits[member == g], np.full(4, float(lab[g])))) for g in range(9)])
        return np.mean([per[lab == c].mean() for c in (0, 1)])
    observed = d(label)
    null = [d(np.isin(np.arange(9), ones).astype(int)) for ones in combinations(range(9), 4)]
    expected = np.mean(np.array(null) >= observed - 1e-12)
    _, d_raw, _, _, p, testable, _ = harness.did_it_train(logits, val_bags, samples, seed=0)
    assert testable and np.isclose(d_raw, observed) and np.isclose(p, expected)


def test_micro_batch_for_uses_the_bag_size_cap_and_the_whole_batch_with_the_age_loss():
    config = {"batch_size": 200, "ncell": 500}
    assert harness.micro_batch_for("cytoset", config) == harness.micro_batch(200, harness.T.MICRO_BATCH["cytoset"][500])
    assert harness.micro_batch_for("cytoset", {**config, "age_similarity_loss": True}) == 200
    assert harness.micro_batch_for("cytoset", {**config, "ncell": 1}) == harness.micro_batch_for("cytoset", config)  # under 500's cap
    with pytest.raises(AssertionError):                               # nothing pinned at or above: fails loudly
        harness.micro_batch_for("cytoset", {**config, "ncell": max(harness.T.MICRO_BATCH["cytoset"]) + 1})


def test_distance_feature_trains_at_the_right_width_and_shared_inputs_match_fresh():
    # the population-distance feature adds 2 input columns (harness.input_width); and training with
    # Residents shared across candidates (stage 05) equals building them fresh
    cells = toy()
    train, val = split(cells, VALIDATION)
    base = {**CONFIGS["cytoset"], "n_epochs": 11, "population_distance_feature": True}
    other = {**base, "h_dim": 8}

    def fit(config, residents=None, inputs=None):
        inputs = inputs or harness.Inputs(config, train, {})
        build = lambda: cytoset.network(config, harness.input_width(config, GENES))
        return harness.train(build, config, train, val, inputs, 1, 2, 3, residents=residents)
    shared_inputs = harness.Inputs(base, train, {})
    residents = harness.Resident(train, shared_inputs, "cpu"), harness.Resident(val, shared_inputs, "cpu")
    shared = [fit(c, residents, shared_inputs) for c in (base, other)]
    fresh = [fit(c) for c in (other, base)][::-1]
    for (_, s1, r1, _, l1), (_, s2, r2, _, l2) in zip(shared, fresh):
        assert r1.validation_curve == r2.validation_curve and np.array_equal(l1, l2)
        assert all(torch.equal(s1[k], s2[k]) for k in s1)


def test_step_budget_and_no_patience():
    cells = toy()
    train, val = split(cells, VALIDATION)
    config = {**CONFIGS["cytoset"], "n_epochs": 50, "patience": None, "max_steps": 97}   # 7 steps per epoch here
    inputs = harness.Inputs(config, train, {})
    _, _, record, _, _ = harness.train(lambda: cytoset.network(config, GENES), config, train, val, inputs, 1, 2, 3)
    assert record.steps == 97 and record.epochs == 14 == len(record.validation_curve)   # exact budget; that epoch scored
