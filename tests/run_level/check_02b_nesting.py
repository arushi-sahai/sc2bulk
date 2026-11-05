"""Run-level check of the all-available store's nesting: for every pair
and every sample, the shard holds min(ALL_AVAILABLE_CAP, post-QC cells) cells, and its first rows are
exactly stage 02's stored cells, with the same cell ids, in the same order, with identical counts.
Stage 02b's own assert compares only the summed pseudobulk, which can't see ids or order. Read-only,
CPU, one sample at a time. Run before stage 07d's output is used (Fig 4A's "up to 3,000 cells"
point). Writes <outputs>/checks/check_02b_nesting.json.

  python tests/run_level/check_02b_nesting.py [PAIR ...]
"""
import json
import sys

import anndata
import numpy as np
import scipy.sparse as sp

from sc2bulk import all_available, folds, manifest, paths
from sc2bulk.config import representations, settings
from sc2bulk.config.pairs import PAIR_ORDER

result = {"git": manifest.git_state(), "pairs": {}}
for pair in sys.argv[1:] or PAIR_ORDER:
    cells = anndata.read_h5ad(folds.SAMPLES / pair / "cells.h5ad", backed="r")
    sample_of = cells.obs["sample_id"].astype(str).to_numpy()
    samples = folds.single_cell_samples(pair).set_index("sample_id")
    shards = all_available.all_shards(pair)
    assert set(sample_of) == set(shards), f"{pair}: stored samples and shards differ"
    for sample_id, path in sorted(shards.items()):
        shard = all_available.load(path)
        assert shard.obs_names.is_unique
        assert shard.n_obs == min(representations.ALL_AVAILABLE_CAP, samples.loc[sample_id, "cells_after_qc"]), \
            f"{pair} {sample_id}: {shard.n_obs} cells in the shard"
        rows = np.flatnonzero(sample_of == sample_id)
        assert (np.diff(rows) == 1).all()
        block = cells[rows[0]:rows[-1] + 1]
        first = shard[:len(rows)]
        assert list(first.obs_names) == list(block.obs_names), f"{pair} {sample_id}: ids or order differ"
        assert list(shard.var_names) == list(cells.var_names), f"{pair} {sample_id}: genes differ"
        assert (sp.csr_matrix(first.X) != sp.csr_matrix(block.X)).nnz == 0, f"{pair} {sample_id}: counts differ"
    result["pairs"][pair] = {"samples": len(shards), "cells": int(cells.n_obs),
                             "store": manifest.read(paths.output("02b_all_available", pair, "index.csv"))["content_hash"]}
    print(pair, result["pairs"][pair], flush=True)
out = settings.OUTPUTS / "checks" / "check_02b_nesting.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=1))
print("PASS", flush=True)
