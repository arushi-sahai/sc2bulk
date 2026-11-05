"""The blinded-run permutation (sc2bulk.labels.permute): whole label rows move
together among scoring units, within strata, reproducibly by seed; a donor pair's strata are the units its
donors are scored in."""
import json
import subprocess

import pandas as pd
import pytest

from sc2bulk import folds, labels, manifest, seeds
from sc2bulk.config import settings


def example():
    samples = [f"s{i}" for i in range(12)]
    table = pd.DataFrame({"sample_id": samples,
                          "label_value": [20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70],
                          "group": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                          "litter": list("aabbccddeeff")})
    unit = pd.Series([f"d{i // 2}" for i in range(12)], index=samples)      # two samples per donor
    stratum = pd.Series(["x"] * 6 + ["y"] * 6, index=samples)
    stratum.iloc[[0, 1, 6, 7]] = ["y", "y", "x", "x"]                         # strata mix the groups
    return table, unit, stratum


def test_rows_move_whole_by_unit_within_strata():
    table, unit, stratum = example()
    out = labels.permute(table, unit, stratum, seed=3)
    assert list(out["sample_id"]) == list(table["sample_id"])
    rows = lambda t: sorted(map(tuple, t[["label_value", "group", "litter"]].to_numpy()))
    assert rows(out) == rows(table)                                           # whole rows, counts kept
    assert (out.groupby(unit.to_numpy())[["label_value", "group", "litter"]].nunique() == 1).all().all()
    for s in ("x", "y"):                                                      # within each stratum
        mask = (stratum == s).to_numpy()
        assert rows(out[mask]) == rows(table[mask])


def test_seeded():
    table, unit, stratum = example()
    a = labels.permute(table, unit, stratum, seed=3)
    assert a.equals(labels.permute(table, unit, stratum, seed=3))
    assert any(not a.equals(labels.permute(table, unit, stratum, seed=s)) for s in range(4, 10))


def test_labels_varying_within_a_unit_fail():
    table, unit, stratum = example()
    table.loc[1, "litter"] = "z"
    with pytest.raises(AssertionError):
        labels.permute(table, unit, stratum, seed=3)


def test_label_value_varying_within_a_unit():
    # Sound Life: a donor's age differs between visits; a permuted donor takes its source donor's
    # earliest age for every sample
    table, unit, stratum = example()
    table.loc[1, "label_value"] = 25
    out = labels.permute(table, unit, stratum, seed=3)
    assert (out.groupby(unit.to_numpy())["label_value"].nunique() == 1).all()
    assert 25 not in set(out["label_value"]) and out.dtypes.equals(table.dtypes)


def blinded_tree(tmp_path, monkeypatch, pair, fold_rows):
    """A synthetic stage 02/03 tree for a donor pair: 16 donors (8 per group), two samples each, the second
    one older; donor_folds.csv from fold_rows (donor index -> [(fold_set, fold, side)])."""
    monkeypatch.setattr(settings, "OUTPUTS", tmp_path)
    monkeypatch.setattr(folds, "FOLDS", tmp_path / "03_splits")
    monkeypatch.setattr(settings, "BLINDED", 1)
    monkeypatch.setattr(manifest, "LABELS_READ", False)
    samples = [f"s{i}" for i in range(32)]
    donors = [f"d{i // 2}" for i in range(32)]
    table = pd.DataFrame({"sample_id": samples, "label_value": [40 + i // 2 + i % 2 for i in range(32)],
                          "group": [int(i >= 16) for i in range(32)]})
    (tmp_path / "02_preprocess" / pair).mkdir(parents=True)
    stage02, stage03 = tmp_path / "02_preprocess" / pair, tmp_path / "03_splits" / pair
    table.to_csv(stage02 / "bulk_labels.csv", index=False)
    pd.DataFrame({"sample_id": samples + ["dropped"], "donor_id": donors + ["d99"],
                  "drop_reason": [None] * 32 + ["smoke run"]}).to_csv(stage02 / "bulk_samples.csv", index=False)
    stage03.mkdir(parents=True)
    pd.DataFrame([{"pair": pair, "fold_set": fs, "fold": f, "donor_id": f"d{d}", "side": side}
                  for d in range(16) for fs, f, side in fold_rows(d)]).to_csv(stage03 / "donor_folds.csv", index=False)
    return table, pd.Series(donors, index=samples)


def test_bulk_labels_keep_each_scored_units_class_counts(tmp_path, monkeypatch):
    # Sound Life's donors are permuted within the units they are scored in (four crossfit folds of
    # 2 + 2 donors; split67's scored side, five donors), so every scored unit keeps its real class counts
    split67 = {0, 3, 9, 12, 14}
    def fold_rows(d):
        crossfit = [("crossfit", f, "scored" if f == d % 4 else "train") for f in range(4)]
        return crossfit + [("split67", 0, "scored" if d in split67 else "train")]
    table, donor = blinded_tree(tmp_path, monkeypatch, "soundlife", fold_rows)
    moved = False
    for blinded in range(1, 6):
        monkeypatch.setattr(settings, "BLINDED", blinded)
        out = labels.bulk_labels("soundlife")
        assert list(out["sample_id"]) == list(table["sample_id"])
        index = donor.str[1:].astype(int).to_numpy()
        for scored in [index % 4 == f for f in range(4)] + [pd.Series(index).isin(split67).to_numpy()]:
            assert out.loc[scored, "group"].sum() == table.loc[scored, "group"].sum()
        moved |= not out["group"].equals(table["group"])
    assert moved


def test_bulk_labels_unchanged_without_scored_donors(tmp_path, monkeypatch):
    # an "all"-only donor pair (lupus) scores no donor in a unit: one stratum
    table, donor = blinded_tree(tmp_path, monkeypatch, "lupus", lambda d: [("all", 0, "train")])
    expected = labels.permute(table, donor, pd.Series("all", index=donor.index), seeds.derived("permutation", "lupus", 1))
    assert labels.bulk_labels("lupus").equals(expected)


def test_analysis_fixed_marker_is_checked(tmp_path, monkeypatch):
    """Real labels unlock only when the marker names an annotated tag at its recorded commit, and that
    commit is an ancestor of the checkout. A throwaway repository; never the real one or the real marker."""
    config = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false"]
    run = lambda *a: subprocess.run(["git", *config, *a], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
    run("init", "-q")
    (tmp_path / "f").write_text("a")
    run("add", "f")
    run("commit", "-q", "-m", "fixed")
    run("tag", "-a", "analysis-fixed", "-m", "fixed")
    run("tag", "light")
    commit = run("rev-parse", "HEAD")
    marker = tmp_path / "analysis_fixed.json"
    monkeypatch.setattr(labels.settings, "BLINDED", None)
    monkeypatch.setattr(labels.settings, "REPO", tmp_path)
    monkeypatch.setattr(labels, "ANALYSIS_FIXED", marker)
    with pytest.raises(AssertionError, match="locked"):
        labels.require_access()                                   # no marker
    marker.write_text(json.dumps({"tag": "light", "commit": commit}))
    with pytest.raises(AssertionError, match="annotated"):
        labels.require_access()                                   # a lightweight tag has no tagger date
    marker.write_text(json.dumps({"tag": "analysis-fixed", "commit": "0" * 40}))
    with pytest.raises(AssertionError, match="not the recorded"):
        labels.require_access()                                   # the tag moved
    marker.write_text(json.dumps({"tag": "analysis-fixed", "commit": commit}))
    labels.require_access()                                       # the matching annotated tag
    run("checkout", "-q", "--orphan", "other")
    run("commit", "-q", "-m", "unrelated")
    with pytest.raises(AssertionError, match="predates"):
        labels.require_access()                                   # a checkout without the fixed commit
    monkeypatch.setattr(labels.settings, "BLINDED", 7)
    labels.require_access()                                       # blinded runs need no marker
