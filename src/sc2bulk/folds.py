"""Which single-cell samples train, and which bulk samples are scored, for each fold.

Every pair has one or more fold sets (pairs.py `fold_sets`; the first is primary), each with
one or more folds. A fold is a unit of everything downstream: its panel, splits, models,
read-outs and scores are made from its training samples only.
- "all" (one fold): every single-cell sample trains and every bulk sample is scored. Used
  where no donor is on both sides.
- "crossfit" (k folds): the donors (on both sides) are dealt into k folds, stratified by the
  single-cell 0/1 group; fold f's donors are scored, all others train. Every donor's bulk is
  scored exactly once, by models that never saw that donor's single-cell data.
- "split67": the 2026-09-30 donor split (30% of donors per age group to scoring), kept as a
  sensitivity fold set; sc2bulk.splits.donor_split reproduces the pinned list from the
  single-cell age groups.

training_samples(), training_cells() and training_pseudobulk() are the only readers of
single-cell data for every per-fold stage (each reads the full store or, for Fig 6C, the
one-cell-type store: cell_subset "all" or "one_type"), so a fold's held-out donors can't reach
its models (tests/test_folds.py checks no other module names those files; the run-level
fold-leak check verifies it end to end).
"""
import anndata
import numpy as np
import pandas as pd

from sc2bulk import seeds
from sc2bulk.config import settings
from sc2bulk.config.pairs import PAIRS
from sc2bulk.splits import donor_split

SAMPLES = settings.OUTPUTS / "02_preprocess"
FOLDS = settings.OUTPUTS / "03_splits"


def units(pair: str) -> list:
    """Every (fold_set, fold) of a pair, the primary fold set first."""
    return [(name, fold) for name, n in PAIRS[pair]["fold_sets"].items() for fold in range(n)]


def unit_name(fold_set: str, fold: int) -> str:
    return f"{fold_set}_{fold}"


def inputs(pair: str, one_type: bool = False) -> list:
    """Stage 02's single-cell outputs a per-fold stage depends on (for its run record)."""
    names = ["single_cell_samples.csv", "cells.h5ad", "pseudobulk.h5ad"] + (["cells_one_type.h5ad", "pseudobulk_one_type.h5ad"] if one_type else [])
    return [SAMPLES / pair / name for name in names]


def single_cell_genes(pair: str) -> pd.DataFrame:
    """The single-cell gene axis (stage 02's var: symbol index, Ensembl ids), no counts."""
    return anndata.read_h5ad(SAMPLES / pair / "cells.h5ad", backed="r").var


def single_cell_samples(pair: str) -> pd.DataFrame:
    """Stage 02's kept single-cell samples (sample_id, donor_id, holdout_group, group, ...)."""
    table = pd.read_csv(SAMPLES / pair / "single_cell_samples.csv", dtype={"holdout_group": str, "donor_id": str})
    kept = table[table["drop_reason"].isna()].reset_index(drop=True)
    kept["group"] = kept["group"].astype(int)
    return kept


def deal(group: pd.Series, n: int, seed: int) -> dict:
    """Donor -> fold (0..n-1): each 0/1 group's donors (sorted, then shuffled with the seed)
    dealt to the folds in turn from a seeded start, both groups continuing the same deal, so
    folds differ in size by at most one donor overall and per group."""
    rng = np.random.default_rng(seed)
    order = [d for g in (0, 1) for d in rng.permutation(sorted(group.index[group == g]))]
    start = rng.integers(n)
    return {d: int((start + i) % n) for i, d in enumerate(order)}


def donor_folds(pair: str, samples: pd.DataFrame) -> pd.DataFrame:
    """fold_set, fold, donor_id, side ("train" / "scored") for every unit and single-cell
    donor. `side` refers to the single-cell donors; under "all" every donor trains, and
    scored_bulk() returns every bulk sample."""
    donors = samples.groupby("donor_id")["group"].agg(["nunique", "first"])
    assert (donors["nunique"] == 1).all(), "a donor has single-cell samples of both groups"
    group = donors["first"]
    rows = []
    for fold_set, n in PAIRS[pair]["fold_sets"].items():
        if fold_set == "all":
            scored = {0: set()}
        elif fold_set == "crossfit":
            fold_of = deal(group, n, seeds.derived("donor_fold", pair, 0, fold_set))
            scored = {f: {d for d, k in fold_of.items() if k == f} for f in range(n)}
        else:
            split = PAIRS[pair]["donor_split"]
            age_groups = samples[["donor_id", "age_group"]]
            test = donor_split(age_groups, split["test_fraction"], split["seed"])
            if not settings.SMOKE:   # the smoke run keeps a subset of donors
                assert test == set(split["test_donors"]), f"{pair}: the {fold_set} donors differ from the pinned list"
            scored = {0: test}
        for fold, held in scored.items():
            for donor in group.index:
                rows.append({"pair": pair, "fold_set": fold_set, "fold": fold, "donor_id": donor,
                             "side": "scored" if donor in held else "train"})
    return pd.DataFrame(rows)


def training_samples(pair: str, fold_set: str, fold: int) -> pd.DataFrame:
    """The kept single-cell samples of a unit's training donors: the only way a per-fold
    stage gets single-cell samples."""
    samples = single_cell_samples(pair)
    folds = pd.read_csv(FOLDS / pair / "donor_folds.csv", dtype={"donor_id": str})
    unit = folds[(folds["fold_set"] == fold_set) & (folds["fold"] == fold)]
    train = set(unit.loc[unit["side"] == "train", "donor_id"])
    assert set(unit["donor_id"]) == set(samples["donor_id"]), f"{pair} {fold_set} {fold}: donor list differs from stage 02"
    return samples[samples["donor_id"].isin(train)].reset_index(drop=True)


CELL_FILES = {"all": ("cells.h5ad", "pseudobulk.h5ad"), "one_type": ("cells_one_type.h5ad", "pseudobulk_one_type.h5ad")}


def in_store(samples: pd.DataFrame, cell_subset: str) -> pd.DataFrame:
    """The samples a store holds: all of them, or (one-type store) those stage 02 recorded as
    stored, i.e. with enough cells of the type (settings.ONE_TYPE_MIN_CELLS)."""
    return samples if cell_subset == "all" else samples[samples["one_type_stored"]]


def training_cells(pair: str, fold_set: str, fold: int, cell_subset: str = "all") -> anndata.AnnData:
    """Stage 02's stored cells (or one-type cells) of a unit's training samples, read from disk
    row by row (only the training cells are loaded)."""
    train = training_samples(pair, fold_set, fold)
    cells = anndata.read_h5ad(SAMPLES / pair / CELL_FILES[cell_subset][0], backed="r")
    keep = cells.obs["sample_id"].astype(str).isin(train["sample_id"]).to_numpy()
    assert set(cells.obs.loc[keep, "sample_id"].astype(str)) == set(in_store(train, cell_subset)["sample_id"]), \
        f"{pair} {fold_set} {fold}: the store's training samples differ from stage 02's record"
    return cells[keep].to_memory()


def training_pseudobulk(pair: str, fold_set: str, fold: int, cell_subset: str = "all") -> anndata.AnnData:
    """Stage 02's pseudobulk profiles (or one-type profiles) of a unit's training samples."""
    train = training_samples(pair, fold_set, fold)
    pseudobulk = anndata.read_h5ad(SAMPLES / pair / CELL_FILES[cell_subset][1])
    expected = in_store(train, cell_subset)["sample_id"]
    assert set(pseudobulk.obs_names) & set(train["sample_id"]) == set(expected), \
        f"{pair} {fold_set} {fold}: the store's training samples differ from stage 02's record"
    return pseudobulk[expected.to_numpy()].copy()


def scored_cells(pair: str, fold_set: str, fold: int, cell_subset: str = "all") -> anndata.AnnData:
    """The stored cells (or one-type cells) of a cross-fitting fold's scored donors, for the
    analyses that score held-out single-cell samples with that fold's models (Fig 2B for the
    cross-fitted pair, Fig 6A, Fig 6C). Only "crossfit" units have scored single-cell donors;
    split67's held-out donors are never read here (their single-cell data stay unused, as in the
    67/29 design). Callers are restricted by tests/test_folds.py."""
    assert fold_set == "crossfit", f"{pair} {fold_set}: only cross-fitting folds have scored single-cell donors"
    folds = pd.read_csv(FOLDS / pair / "donor_folds.csv", dtype={"donor_id": str})
    unit = folds[(folds["fold_set"] == fold_set) & (folds["fold"] == fold)]
    scored = set(unit.loc[unit["side"] == "scored", "donor_id"])
    samples = single_cell_samples(pair)
    samples = samples[samples["donor_id"].isin(scored)]
    cells = anndata.read_h5ad(SAMPLES / pair / CELL_FILES[cell_subset][0], backed="r")
    keep = cells.obs["sample_id"].astype(str).isin(samples["sample_id"]).to_numpy()
    assert set(cells.obs.loc[keep, "sample_id"].astype(str)) == set(in_store(samples, cell_subset)["sample_id"]), \
        f"{pair} {fold_set} {fold}: the store's scored samples differ from stage 02's record"
    return cells[keep].to_memory()


def all_cells(pair: str, cell_subset: str = "all") -> anndata.AnnData:
    """Every stored cell of the pair (or its one-type cells), for label-free uses that cover
    every sample: the FM embeddings (stage 07; which samples a unit then uses is decided where
    embeddings are read, through this module) and descriptive outputs (QC embeddings, dataset
    characteristics, annotated deconvolution fractions). Callers are restricted by
    tests/test_folds.py."""
    return anndata.read_h5ad(SAMPLES / pair / CELL_FILES[cell_subset][0])


def all_pseudobulk(pair: str, cell_subset: str = "all") -> anndata.AnnData:
    """Every pseudobulk profile of the pair (or its one-type profiles); the same restrictions as all_cells."""
    return anndata.read_h5ad(SAMPLES / pair / CELL_FILES[cell_subset][1])


def scored_bulk(pair: str, fold_set: str, fold: int, bulk_samples: pd.DataFrame) -> pd.DataFrame:
    """The bulk samples a unit scores: all of them under "all", else the scored donors'."""
    if fold_set == "all":
        return bulk_samples
    folds = pd.read_csv(FOLDS / pair / "donor_folds.csv", dtype={"donor_id": str})
    unit = folds[(folds["fold_set"] == fold_set) & (folds["fold"] == fold)]
    return bulk_samples[bulk_samples["donor_id"].isin(set(unit.loc[unit["side"] == "scored", "donor_id"]))]
