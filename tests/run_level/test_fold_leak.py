"""Run-level fold-leak check: a cross-fitting fold's outputs must
not depend on its scored donors' single-cell data or raw labels.

On a copy of the smoke run's stage-02 outputs for the cross-fitted pair, fold 0's scored donors
get their cells' genes permuted (cells.h5ad and pseudobulk.h5ad) and their label_value changed;
their 0/1 group is kept, since it legitimately drives the donor deal. Stages 03, 04 and 04b are
then rerun on the copy, and stage 08 for pseudobulk (end to end for pseudobulk only;
the trained and FM configurations need stage 06's GPU runs and stage 07's pinned store, so they rest
on unit tests of what their readers keep). Fold 0's splits, read-out folds, panel and per-unit
deconvolution files must be identical to the original smoke run's (04b's unit-free diagnostic/
files, which describe the held-out donors, are exempt), and so must its stage 08 pseudobulk files
but Fig 2B's (the primary and out-of-fold read-outs and the guard table); the other folds' panels
(which train on those donors) and fold 0's Fig 2B held-out predictions (which score them) must
change, which shows the perturbation reaches the outputs.

Needs the smoke outputs (pipeline/run_all.py --smoke through 08_readout). Run with
  SC2BULK_RUN_LEVEL=1 SC2BULK_OUTPUTS=<outputs> python -m pytest tests/run_level
"""
import os
import shutil
import subprocess
import sys

import anndata
import numpy as np
import pandas as pd
import pytest

from sc2bulk.config import settings
from sc2bulk.config.pairs import PAIRS

PAIR = next(p for p, spec in PAIRS.items() if "crossfit" in spec["fold_sets"])
pytestmark = pytest.mark.skipif(os.environ.get("SC2BULK_RUN_LEVEL") != "1", reason="run-level check")


def permute_genes(adata, rows, rng):
    """Permute gene columns within the given rows (one permutation, applied to each row)."""
    X = adata.X.tocsr() if hasattr(adata.X, "tocsr") else np.asarray(adata.X)
    perm = rng.permutation(X.shape[1])
    if isinstance(X, np.ndarray):
        X[np.ix_(rows, np.arange(X.shape[1]))] = X[np.ix_(rows, perm)]
    else:
        X = X.tolil()
        X[rows] = X[rows][:, perm]
        X = X.tocsr()
    adata.X = X


def test_fold_zero_ignores_its_scored_donors(tmp_path):
    smoke = settings.OUTPUTS if settings.SMOKE else settings.OUTPUTS / "smoke"
    copy = tmp_path / "smoke"
    shutil.copytree(smoke / "02_preprocess" / PAIR, copy / "02_preprocess" / PAIR)
    folds = pd.read_csv(smoke / "03_splits" / PAIR / "donor_folds.csv", dtype={"donor_id": str})
    scored = set(folds.loc[(folds["fold_set"] == "crossfit") & (folds["fold"] == 0) & (folds["side"] == "scored"), "donor_id"])
    assert scored
    rng = np.random.default_rng(0)
    for name in ("cells.h5ad", "pseudobulk.h5ad"):
        path = copy / "02_preprocess" / PAIR / name
        adata = anndata.read_h5ad(path)
        rows = np.flatnonzero(adata.obs["donor_id"].astype(str).isin(scored).to_numpy())
        permute_genes(adata, rows, rng)
        adata.write_h5ad(path)
    table_path = copy / "02_preprocess" / PAIR / "single_cell_samples.csv"
    table = pd.read_csv(table_path, dtype=str)
    table.loc[table["donor_id"].isin(scored), "label_value"] = "999"
    table.to_csv(table_path, index=False)

    env = dict(os.environ, SC2BULK_OUTPUTS=str(tmp_path), SC2BULK_SMOKE="1")
    for stage in ("03_splits", "04_panel", "04b_deconvolve"):
        subprocess.run([sys.executable, str(settings.REPO / "pipeline" / f"{stage}.py"), "--pair", PAIR], env=env, check=True)
    subprocess.run([sys.executable, str(settings.REPO / "pipeline" / "08_readout.py"), "--pair", PAIR, "--configuration", "pseudobulk"],
                   env=env, check=True)

    def fold_rows(root, name, fold):
        t = pd.read_csv(root / "03_splits" / PAIR / name, dtype=str)
        return t[(t["fold_set"] == "crossfit") & (t["fold"] == str(fold))].reset_index(drop=True)

    for name in ("splits.csv", "readout_folds.csv"):
        pd.testing.assert_frame_equal(fold_rows(smoke, name, 0), fold_rows(copy, name, 0))
    panel = lambda root, f: pd.read_csv(root / "04_panel" / PAIR / f"crossfit_{f}" / "panel.csv")
    pd.testing.assert_frame_equal(panel(smoke, 0), panel(copy, 0))
    assert any(not panel(smoke, f).equals(panel(copy, f)) for f in (1, 2, 3)), "the perturbation reached no output"
    for name in ("types.csv", "markers.csv", "fractions.parquet", "fractions_splits.parquet", "check_v1.csv",
                 "check_v2.csv", "check_v4.csv"):
        a, b = (root / "04b_deconvolve" / PAIR / "crossfit_0" / name for root in (smoke, copy))
        assert a.read_bytes() == b.read_bytes(), name
    readouts = lambda root: root / "08_readout" / PAIR / "crossfit_0"
    names = sorted(p.name for p in readouts(smoke).glob("pseudobulk_*.parquet")) + ["out_of_fold_guard.csv"]
    assert names == sorted(p.name for p in readouts(copy).glob("pseudobulk_*.parquet")) + ["out_of_fold_guard.csv"]
    for name in names:
        a, b = (readouts(root) / name for root in (smoke, copy))
        assert ("_within_" in name) == (a.read_bytes() != b.read_bytes()), name   # only Fig 2B's held-out rows see those donors
