"""Stage 02b: the all-available-cells store (Fig 4A's "up to 3,000 cells" point).

For every sample stage 02 kept, the raw data are read again through the pair's reader, put on
stage 02's gene axis, QC'd with stage 02's rule, and the first ALL_AVAILABLE_CAP post-QC cells in
the sample's seeded cell order (preprocess.cell_order) are written as one shard per sample. The
store nests with stage 02's: a sample's first MAX_STORED_CELLS shard rows are its stored cells
(same rule, same seeded order; asserted: their number equals stage 02's cells_stored and their
summed counts equal its pseudobulk profile exactly).
Streaming one sample at a time bounds memory.

Read only through sc2bulk.all_available, which restricts samples to a unit's training samples.

Outputs, in <outputs>/02b_all_available/<pair>/:
  shards/<sample_id>.h5ad   (id URL-encoded) int32 counts (cells x stage 02's genes; gzip); obs: cell_id
  index.csv                 sample_id, cells, shard content hash

  python pipeline/02b_all_available.py [--pair PAIR]
"""
import argparse
import importlib

import anndata
import numpy as np
import pandas as pd

from sc2bulk import all_available, codehash, determinism, folds, manifest, paths, preprocess
from sc2bulk.config import representations, settings
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

STAGE = "02b_all_available"


def record_for(pair) -> dict:
    return {"stage": STAGE, "pair": pair, "smoke": settings.SMOKE, "cap": representations.ALL_AVAILABLE_CAP,
            "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair, codehash.extra(STAGE, pair)),
            "inputs": manifest.input_hashes(folds.inputs(pair))}


def shard(pair, sample, X, symbols, cells, genes, n_stored, pseudobulk) -> anndata.AnnData:
    """The sample's first ALL_AVAILABLE_CAP post-QC cells in cell order, checked against stage 02."""
    X = preprocess.to_gene_axis(X, symbols, genes)
    qc = preprocess.cell_qc(X, genes, PAIRS[pair]["species"],
                            PAIRS[pair].get("qc_max_percent_mito", settings.QC_MAX_PERCENT_MITO))
    passed = np.flatnonzero(qc["passed_qc"].to_numpy())
    keep = passed[preprocess.cell_order(pair, sample, len(passed))[:representations.ALL_AVAILABLE_CAP]]
    ids = cells["cell_id"].astype(str).to_numpy()[keep]
    first = keep[:settings.MAX_STORED_CELLS]
    assert len(first) == n_stored, f"{pair} {sample}: {len(first)} stored cells, stage 02 stored {n_stored}"
    assert np.array_equal(preprocess.summed(X[first]), pseudobulk), f"{pair} {sample}: counts differ from stage 02's pseudobulk"
    return anndata.AnnData(X[keep].astype(np.int32), obs=pd.DataFrame(index=pd.Index(ids, name="cell_id")),
                           var=pd.DataFrame(index=genes))


def main():
    determinism.configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER)
    args = parser.parse_args()
    for pair in [args.pair] if args.pair else PAIR_ORDER:
        record = record_for(pair)
        samples = folds.single_cell_samples(pair)
        index_path = paths.output(STAGE, pair, "index.csv")
        shard_paths = all_available.all_shards(pair)
        if manifest.is_done([index_path, *shard_paths.values()], record):
            print(f"{pair}: done", flush=True)
            continue
        shard_paths[samples["sample_id"].iloc[0]].parent.mkdir(parents=True, exist_ok=True)
        unit = manifest.begin()
        genes = folds.single_cell_genes(pair).index
        pseudobulk = folds.all_pseudobulk(pair)
        n_stored = samples.set_index("sample_id")["cells_stored"]
        reader = importlib.import_module(f"sc2bulk.data.{pair}")
        rows = []
        for sample, X, symbols, cells in reader.single_cell_samples(settings.RAW):
            sample_id = sample["sample_id"]
            if sample_id not in shard_paths:
                continue
            data = shard(pair, sample_id, X, symbols, cells, genes, int(n_stored[sample_id]),
                         np.asarray(pseudobulk[sample_id].X).ravel())
            path = shard_paths[sample_id]
            data.write_h5ad(manifest.temporary(path), compression="gzip")   # gzip: the shared disk is near full
            manifest.finish(path, record, unit, inputs=folds.inputs(pair))
            rows.append({"sample_id": sample_id, "cells": data.n_obs, "content_hash": manifest.read(path)["content_hash"]})
        index = pd.DataFrame(rows).sort_values("sample_id")
        assert set(index["sample_id"]) == set(shard_paths), f"{pair}: the reader didn't yield every kept sample"
        index.to_csv(manifest.temporary(index_path), index=False)
        manifest.finish(index_path, record, unit, inputs=folds.inputs(pair))
        print(f"{pair}: {len(index)} samples, {index['cells'].sum()} cells (median {index['cells'].median():.0f})", flush=True)


if __name__ == "__main__":
    main()
