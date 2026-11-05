"""Output schemas (sc2bulk.schemas) and the bag drawer (sc2bulk.bags)."""
import numpy as np
import pandas as pd
import pytest

from sc2bulk import bags, schemas


def table(**extra):
    base = {"pair": ["hammond"], "fold_set": ["all"], "fold": [0], "row_kind": ["configuration"],
            "key": ["pseudobulk"], "model": ["pseudobulk"], "input": ["pseudobulk"], "n_cells": [0], "pooling": ["native"]}
    return schemas.with_neutral_keys(pd.DataFrame({**base, **extra}))


def test_round_trip_and_checks(tmp_path):
    frame = table(side=["bulk"], sample_id=["s1"], scoring_unit_id=["s1"], score=[0.3], probability=[0.6])
    schemas.write("predictions", frame, tmp_path / "p.parquet")
    back = schemas.read("predictions", tmp_path / "p.parquet")
    assert back["readout"].iloc[0] == "ridge" and back["permutation"].iloc[0] == -1
    for bad in (frame.assign(alignment="nonsense"), frame.drop(columns="score"), pd.concat([frame, frame]),
                frame.assign(key="cytoset_cells"),                      # model/input disagree with the key
                frame.assign(row_kind="reference", key="not_a_reference")):
        with pytest.raises(AssertionError):
            schemas.check("predictions", bad)
    with pytest.raises(AssertionError):   # a label-bearing kind must use the marked name
        schemas.write("scores", frame, tmp_path / "s.parquet")


def test_secondary_kinds_and_values(tmp_path):
    """Stage 11b's additions (spec stage08b_11b P2-P5): several secondary rows per configuration, the scores
    kind's signed -log10 p, the bulk-side variants and the own-recipe candidate ids."""
    stats = {"delta": [0.1, np.nan], "ci_low": [0.0, np.nan], "ci_high": [0.2, np.nan], "u": [0.5, np.nan],
             "p_better": [0.04, np.nan], "p_worse": [0.96, np.nan], "p_better_holm": [np.nan, np.nan],
             "p_worse_holm": [np.nan, np.nan], "mdd_obs": [0.1, np.nan], "resamples": [2000, 2000], "n0": [9, 4],
             "n1": [10, 5], "excluded_pairs": ["", ""], "estimable": [True, False]}
    one = table(row=["subgroups"], pool_set=["per_pair"], statistic=["auroc_difference"])
    two = pd.concat([one, one], ignore_index=True).assign(bulk_subset=["condition=a", "condition=b"], **stats)
    path = tmp_path / f"{schemas.LABELLED}_secondary_statistics.parquet"
    schemas.write("secondary_statistics", two, path)
    assert schemas.read("secondary_statistics", path)["estimable"].tolist() == [True, False]
    for bad in (two.assign(bulk_subset="all"), two.drop(columns="estimable")):   # one row per bulk subset; estimable required
        with pytest.raises(AssertionError):
            schemas.check("secondary_statistics", bad)
    scores = table(bulk_subset=["all"], side=["bulk"], n0=[9], n1=[10], auroc=[0.7], auprc=[0.6], brier=[0.2], signed_log10_p=[1.3])
    schemas.check("scores", scores)
    with pytest.raises(AssertionError):
        schemas.check("scores", scores.drop(columns="signed_log10_p"))
    for variant in ("thinned_pseudocells", "observed_library", "seed_ensemble"):
        schemas.check("scores", scores.assign(variant=variant))
    assert {"cytoset_own_recipe", "set_transformer_own_recipe", "scrat_own_recipe"} <= schemas.candidate_ids()


def test_test_bags_are_shared_and_order_free():
    ids = np.array([f"c{i}" for i in range(85)])
    sample = np.repeat(np.array(["a", "b", "c"]), [50, 5, 30])
    one = bags.test_bags(ids, sample, ["a", "b", "c"], 10, 3, "hammond", ("all", 0))
    shuffled = np.random.default_rng(0).permutation(85)
    two = bags.test_bags(ids[shuffled], sample[shuffled], ["c", "b", "a"], 10, 3, "hammond", ("all", 0))
    for s in "abc":
        assert all(np.array_equal(x, y) for x, y in zip(one[s], two[s]))
    assert len(one["b"][0]) == 5
    assert all(set(b) <= set(ids[sample == "a"]) for b in one["a"])
    assert len(bags.test_bags(ids, sample, ["a"], 0, 10, "hammond", ("all", 0))["a"]) == 1
