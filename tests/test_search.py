"""Stage 05's selection rules on hand-built results (every branch)."""
import itertools
import math

from sc2bulk import harness, search
from sc2bulk.config import search_space as S
from sc2bulk.config import training as T
from test_harness import toy

FLOOR = T.MARGIN_FLOOR * math.log(2)


def test_eligible():
    good_d, good_p = [0.1] * 5, [0.01, 0.01, 0.01, 0.2, 0.2]
    assert search.eligible(good_d, good_p, testable=True)
    assert search.eligible([0.1] * 3 + [FLOOR] * 2, good_p, testable=True)          # a majority clears the floor
    assert not search.eligible([0.1] * 2 + [FLOOR] * 3, good_p, testable=True)      # a minority does
    assert not search.eligible(good_d, [0.01, 0.01, 0.2, 0.2, 0.2], testable=True)  # 2 of 5
    assert search.eligible(good_d, [0.5] * 5, testable=False)                      # untestable: floor only
    assert search.eligible([0.1] * 3, [0.01, 0.05, 0.9], testable=True)             # 2 of 3, p = alpha counts


def test_beats_uses_paired_sd():
    base = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert search.beats(base, [0.9, 0.9, 0.9, 0.9, 0.85])          # consistent gain
    assert not search.beats(base, [0.5, 1.2, 1.1, 0.6, 1.3])       # mean gain 0.06, SD large
    assert not search.beats(base, base)                            # mean 0 = SD 0: not strictly greater
    mean, sd = search.improvement([1, 2, 3], [0, 2, 3])
    assert abs(sd - (1 / 3) ** 0.5) < 1e-12                        # ddof 1


def test_group_winner():
    base = [1.0] * 5
    cands = {"a": [0.8] * 4 + [0.79], "b": [0.7] * 4 + [0.69], "c": [0.5] * 4 + [0.49]}
    assert search.group_winner(base, cands, {"a", "b"}) == "b"      # c is the best but ineligible
    assert search.group_winner(base, cands, set()) is None
    assert search.group_winner(base, {"a": [1.2, 0.4, 1.3, 0.5, 1.1]}, {"a"}) is None


CHANGES = {"abund": {"abundance_weighting": True}, "out_sum": {"out_pool": "sum"},
           "scaler": {"pseudobulk_gene_scaler": True}, "per_sample": {"cell_normalization": "per_sample", "pca": False},
           "population": {"cell_normalization": "population"}, "h128": {"h_dim": 128}, "panel_input": {"pca": False}}


def test_resolve_conflicts():
    winners = {"abundance_weighting": "abund", "out_pool": "out_sum", "h_dim": "h128"}
    assert search.resolve(winners, CHANGES, {"abund": 0.1, "out_sum": 0.2, "h128": 0.3}) == {"out_pool": "out_sum", "h_dim": "h128"}
    assert search.resolve(winners, CHANGES, {"abund": 0.3, "out_sum": 0.2, "h128": 0.3}) == {"abundance_weighting": "abund", "h_dim": "h128"}
    scaler = {"pseudobulk_gene_scaler": "scaler", "cell_normalization": "per_sample"}
    assert search.resolve(scaler, CHANGES, {"scaler": 0.1, "per_sample": 0.2}) == {"cell_normalization": "per_sample"}
    assert search.resolve(scaler, CHANGES, {"scaler": 0.3, "per_sample": 0.2}) == {"cell_normalization": "per_sample"}   # always dropped
    population = {"pseudobulk_gene_scaler": "scaler", "cell_normalization": "population"}
    assert search.resolve(population, CHANGES, {"scaler": 0.1, "population": 0.2}) == population   # no conflict
    panel = {"pca": "panel_input", "cell_normalization": "per_sample"}
    assert search.resolve(panel, CHANGES, {"panel_input": 0.3, "per_sample": 0.2}) == panel   # both kept


def test_scrat_pairwise_merges_build_inputs(monkeypatch):
    # every pair of ScRAT winners from two groups merges into inputs the harness accepts (never per_sample + PCA)
    monkeypatch.setattr(T, "SCRAT_PCA_COMPONENTS", 5)   # the toy has 20 genes
    cells, base = toy(), S.base("scrat")
    changes = {name: change for group in S.SCRAT_GROUPS.values() for name, change in group.items()}
    gains = dict.fromkeys(changes, 0.1)
    for (ga, a), (gb, b) in itertools.combinations(S.SCRAT_GROUPS.items(), 2):
        for na, nb in itertools.product(a, b):
            config = search.merged_config(base, search.resolve({ga: na, gb: nb}, changes, gains), changes)
            harness.Inputs(config, cells, {})


def test_frozen_branches():
    base = {"h_dim": 64, "out_pool": "mean"}
    gains = {"h128": 0.3, "out_sum": 0.2}
    assert search.frozen(base, {}, CHANGES, gains, None) == (base, "base")
    assert search.frozen(base, {"h_dim": "h128"}, CHANGES, gains, None) == ({"h_dim": 128, "out_pool": "mean"}, "single_winner")
    two = {"h_dim": "h128", "out_pool": "out_sum"}
    assert search.frozen(base, two, CHANGES, gains, (True, True)) == ({"h_dim": 128, "out_pool": "sum"}, "merged")
    assert search.frozen(base, two, CHANGES, gains, (True, False)) == ({"h_dim": 128, "out_pool": "mean"}, "best_single")
    assert search.frozen(base, two, CHANGES, gains, (False, True)) == ({"h_dim": 128, "out_pool": "mean"}, "best_single")


def test_modal():
    base, a, b = {"x": 0}, {"x": 1}, {"x": 2}
    assert search.modal([a, a, b, base], base) == a
    assert search.modal([a, b, base, base], base) == base       # the base in 2 folds
    assert search.modal([a, a, base, base], base) == base       # tie with the base
    assert search.modal([b, a, a, b], base) == b                # tie without the base: fold 0's
    assert search.modal([a, b, base, {"x": 3}], base) == base   # all different, base among them
    tupled = {"step_lr": (50, 0.5)}                             # ScRAT's base: a tuple; stored selections: lists
    stored = {"step_lr": [50, 0.5]}
    assert search.modal([stored, stored, a, a], tupled) == tupled   # tie with the base, across the round trip


def test_per_sample_sensitivity_rerun():
    # the selection rule rerun without norm_per_sample
    base = [1.0] * 5
    groups = {"cell_normalization": {"norm_per_sample": [0.6] * 4 + [0.59], "norm_population": [0.8] * 4 + [0.79]},
              "h_dim": {"h128": [0.9] * 4 + [0.89]}}
    names = {"norm_per_sample", "norm_population", "h128"}
    assert search.winners(base, groups, names) == {"cell_normalization": "norm_per_sample", "h_dim": "h128"}
    assert search.winners(base, groups, names, exclude={"norm_per_sample"}) == {"cell_normalization": "norm_population", "h_dim": "h128"}
    weak = {**groups, "cell_normalization": {"norm_per_sample": groups["cell_normalization"]["norm_per_sample"],
                                             "norm_population": [1.2, 0.8, 1.1, 0.9, 1.0]}}
    assert search.winners(base, weak, names, exclude={"norm_per_sample"}) == {"h_dim": "h128"}   # remaining winners only


def test_margin_needs_two_seeds():
    import pytest
    with pytest.raises(AssertionError):
        search.beats([1.0], [0.5])
