"""Run-level fold-leak check of stages 09 and 09c: a cross-fitting fold's alignment outputs and coordinates must not depend on its scored
donors' single-cell data or raw labels.

As tests/run_level/test_fold_leak.py (whose perturbation it reuses): on a copy of the smoke run's stage-02
outputs for the cross-fitted pair, fold 0's scored donors get their cells' genes permuted and their
label_value changed; stages 03, 04, 08, 09 and 09c are rerun on the copy for pseudobulk (end to end for
pseudobulk only, as stage 08's check). Fold 0's stage 09 and 09c files must be byte-identical to the original
smoke run's; the positive control: folds 1-3, which train on those donors, must change in stage 09 (09c writes fold 0
only, so it has none of its own).

Needs the smoke outputs (pipeline/run_all.py --smoke through 09c_geometry). Run with
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
from test_fold_leak import PAIR, permute_genes

pytestmark = pytest.mark.skipif(os.environ.get("SC2BULK_RUN_LEVEL") != "1", reason="run-level check")


def test_stages_09_and_09c_fold_zero_ignore_its_scored_donors(tmp_path):
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
    for stage in ("03_splits", "04_panel"):
        subprocess.run([sys.executable, str(settings.REPO / "pipeline" / f"{stage}.py"), "--pair", PAIR], env=env, check=True)
    for stage in ("08_readout", "09_alignment", "09c_geometry"):
        subprocess.run([sys.executable, str(settings.REPO / "pipeline" / f"{stage}.py"), "--pair", PAIR, "--configuration", "pseudobulk"],
                       env=env, check=True)

    def files(root, fold):
        return {name: (root / "09_alignment" / PAIR / f"crossfit_{fold}" / name).read_bytes()
                for name in ("pseudobulk_predictions.parquet", "pseudobulk_heads.parquet", "pseudobulk_geometry.parquet")}

    assert files(smoke, 0) == files(copy, 0), "fold 0's stage 09 outputs depend on its scored donors"
    coordinates = [root / "09c_geometry" / PAIR / "crossfit_0" / "pseudobulk_geometry.parquet" for root in (smoke, copy)]
    assert coordinates[0].read_bytes() == coordinates[1].read_bytes(), "fold 0's stage 09c coordinates depend on its scored donors"
    assert any(files(smoke, f)["pseudobulk_predictions.parquet"] != files(copy, f)["pseudobulk_predictions.parquet"]
               for f in (1, 2, 3)), "the perturbation reached no stage 09 output"
