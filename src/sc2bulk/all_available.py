"""Reader of the all-available-cells store (stage 02b; Fig 4A's "up to 3,000 cells" point).

The store is one shard per sample (up to ALL_AVAILABLE_CAP post-QC cells, nested with stage 02's
stored cells). Shards are loaded one at a time, so memory stays bounded. Like sc2bulk.folds, a
per-fold stage gets only its unit's training samples (training_shards). all_shards covers every
sample, for label-free uses that embed every sample (the FM stage 07d); its callers are restricted
by tests/test_folds.py.
"""
from pathlib import Path
from urllib.parse import quote

import anndata

from sc2bulk import folds, paths

STAGE = "02b_all_available"


def shard_path(pair: str, sample_id: str) -> Path:
    """A sample's shard; the id is URL-encoded in the file name (mouse-organ ids contain "/")."""
    return paths.output(STAGE, pair, f"shards/{quote(sample_id, safe='')}.h5ad")


def training_shards(pair: str, fold_set: str, fold: int) -> dict:
    """{sample_id: shard path} for a unit's training samples (folds.training_samples)."""
    return {s: shard_path(pair, s) for s in folds.training_samples(pair, fold_set, fold)["sample_id"]}


def all_shards(pair: str) -> dict:
    """{sample_id: shard path} for every kept sample."""
    return {s: shard_path(pair, s) for s in folds.single_cell_samples(pair)["sample_id"]}


def load(path: Path) -> anndata.AnnData:
    """One shard: int32 counts (cells x stage 02's genes), obs index cell_id."""
    return anndata.read_h5ad(path)
