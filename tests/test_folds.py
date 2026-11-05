"""Folds (sc2bulk.folds): the cross-fitting deal is a stratified, seeded partition of donors,
and per-fold stages read single-cell data only through sc2bulk.folds."""
import re

import pandas as pd
import pytest

from sc2bulk.config import settings

from sc2bulk.folds import deal


def test_deal_partitions_stratified_and_seeded():
    group = pd.Series([0] * 49 + [1] * 47, index=[f"d{i:02d}" for i in range(96)])
    fold_of = deal(group, 4, 7)
    assert set(fold_of) == set(group.index) and set(fold_of.values()) == {0, 1, 2, 3}
    sizes = pd.Series(fold_of).value_counts()
    assert sizes.max() - sizes.min() <= 1
    for g in (0, 1):
        per_group = pd.Series({d: f for d, f in fold_of.items() if group[d] == g}).value_counts()
        assert per_group.max() - per_group.min() <= 1
    assert deal(group, 4, 7) == fold_of                                  # seeded
    assert deal(group.sample(frac=1, random_state=0), 4, 7) == fold_of   # row order doesn't matter


SINGLE_CELL_FILES = re.compile(r"single_cell_samples\.csv|cells\.h5ad|pseudobulk\.h5ad|cells_one_type\.h5ad|pseudobulk_one_type\.h5ad|cell_qc\.csv|shards/")
ALLOWED = {"pipeline/02_preprocess.py", "src/sc2bulk/folds.py",
           "pipeline/04d_characteristics.py",   # descriptive: per-cell annotations of every donor (cell_qc), never model features
           "pipeline/02b_all_available.py", "src/sc2bulk/all_available.py",   # the all-available store's writer and reader
           "figures/common.py"}   # S2-S7 Fig panel A: per-cell QC distributions (cell_qc), descriptive, never model features


def test_only_folds_reads_single_cell_data():
    offenders = []
    for directory in ("src", "pipeline", "figures"):
        for path in (settings.REPO / directory).rglob("*.py"):
            name = str(path.relative_to(settings.REPO))
            if name not in ALLOWED and SINGLE_CELL_FILES.search(path.read_text()):
                offenders.append(name)
    assert not offenders, f"read single-cell data outside sc2bulk.folds: {offenders}"


# 06_train and 08_readout read scored_cells for Fig 2B's held-out donors (human blood, aging)
RESTRICTED_READERS = {"scored_cells": {"pipeline/04b_deconvolve.py", "pipeline/06_train.py", "pipeline/08_readout.py",
                                       "pipeline/12a_properties.py", "pipeline/08b_secondary.py"}, "all_cells": {"pipeline/07_embed.py", "pipeline/04c_qc_embed.py"},
                      "all_pseudobulk": {"pipeline/07_embed.py", "pipeline/04d_characteristics.py", "pipeline/02b_all_available.py"},
                      "all_shards": {"pipeline/02b_all_available.py"}}   # allowed caller files


def test_restricted_readers_have_listed_callers_only():
    for reader, allowed in RESTRICTED_READERS.items():
        callers = [str(p.relative_to(settings.REPO)) for d in ("src", "pipeline", "figures")
                   for p in (settings.REPO / d).rglob("*.py")
                   if re.search(rf"\b{reader}\b", p.read_text())
                   and str(p.relative_to(settings.REPO)) not in allowed | {"src/sc2bulk/folds.py", "src/sc2bulk/all_available.py"}]
        assert not callers, f"folds.{reader} called outside its allowed callers: {callers}"


def test_one_type_readers_keep_to_the_unit(tmp_path, monkeypatch):
    # a crossfit unit with two training and one scored donor; the one-type store lacks one
    # training sample (too few cells of the type)
    import anndata
    import numpy as np
    from sc2bulk import folds
    monkeypatch.setattr(folds, "SAMPLES", tmp_path / "02")
    monkeypatch.setattr(folds, "FOLDS", tmp_path / "03")
    (tmp_path / "02" / "p").mkdir(parents=True)
    (tmp_path / "03" / "p").mkdir(parents=True)
    pd.DataFrame({"sample_id": ["s1", "s2", "s3"], "donor_id": ["d1", "d2", "d3"], "holdout_group": ["d1", "d2", "d3"],
                  "group": [0, 1, 0], "drop_reason": [None, None, None],
                  "one_type_stored": [True, False, True]}).to_csv(tmp_path / "02/p/single_cell_samples.csv", index=False)
    pd.DataFrame({"fold_set": "crossfit", "fold": 0, "donor_id": ["d1", "d2", "d3"],
                  "side": ["train", "train", "scored"]}).to_csv(tmp_path / "03/p/donor_folds.csv", index=False)
    cells = ["s1", "s1", "s3", "s3"]                     # s2 has no one-type cells
    anndata.AnnData(np.ones((4, 2), dtype=np.float32),
                    obs=pd.DataFrame({"sample_id": cells}, index=[f"c{i}" for i in range(4)])).write_h5ad(tmp_path / "02/p/cells_one_type.h5ad")
    anndata.AnnData(np.ones((2, 2), dtype=np.float32),
                    obs=pd.DataFrame(index=["s3", "s1"])).write_h5ad(tmp_path / "02/p/pseudobulk_one_type.h5ad")
    assert set(folds.training_cells("p", "crossfit", 0, "one_type").obs["sample_id"]) == {"s1"}
    assert list(folds.training_pseudobulk("p", "crossfit", 0, "one_type").obs_names) == ["s1"]
    assert set(folds.scored_cells("p", "crossfit", 0, "one_type").obs["sample_id"]) == {"s3"}
    # a store that disagrees with stage 02's record fails loudly
    anndata.AnnData(np.ones((1, 2), dtype=np.float32),
                    obs=pd.DataFrame({"sample_id": ["s3"]}, index=["c0"])).write_h5ad(tmp_path / "02/p/cells_one_type.h5ad")
    with pytest.raises(AssertionError):
        folds.training_cells("p", "crossfit", 0, "one_type")
    anndata.AnnData(np.ones((3, 2), dtype=np.float32),                      # an extra sample also fails
                    obs=pd.DataFrame(index=["s3", "s1", "s2"])).write_h5ad(tmp_path / "02/p/pseudobulk_one_type.h5ad")
    with pytest.raises(AssertionError):
        folds.training_pseudobulk("p", "crossfit", 0, "one_type")


def test_all_available_training_shards_keep_to_the_unit(tmp_path, monkeypatch):
    from sc2bulk import all_available, folds
    monkeypatch.setattr(folds, "SAMPLES", tmp_path / "02")
    monkeypatch.setattr(folds, "FOLDS", tmp_path / "03")
    (tmp_path / "02" / "p").mkdir(parents=True)
    (tmp_path / "03" / "p").mkdir(parents=True)
    pd.DataFrame({"sample_id": ["s1", "s2", "s3"], "donor_id": ["d1", "d2", "d3"], "holdout_group": ["d1", "d2", "d3"],
                  "group": [0, 1, 0], "drop_reason": [None, None, None]}).to_csv(tmp_path / "02/p/single_cell_samples.csv", index=False)
    pd.DataFrame({"fold_set": "crossfit", "fold": 0, "donor_id": ["d1", "d2", "d3"],
                  "side": ["train", "scored", "train"]}).to_csv(tmp_path / "03/p/donor_folds.csv", index=False)
    assert set(all_available.training_shards("p", "crossfit", 0)) == {"s1", "s3"}
    assert set(all_available.all_shards("p")) == {"s1", "s2", "s3"}
    assert all_available.shard_path("p", "Lung_3-M-5/6").name == "Lung_3-M-5%2F6.h5ad"
