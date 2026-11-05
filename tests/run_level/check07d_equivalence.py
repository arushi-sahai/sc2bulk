"""Pre-declared 07d equivalence check: stage 07d's own functions
(shard_chunks, embed_chunk) applied to Hammond's first 5 training samples' stored rows 0-999 vs stage
07's cells embeddings of the same cells (per-cell cosine >= 0.9999, max |diff| <= 1e-3). Also checks
that those shard rows' counts equal stage 02's stored cells exactly (the store's nesting). Run on its
own before 07d resumes beyond its launch test; if it fails, stop and report (never loosen the
tolerance). Writes its record to <outputs>/checks/check07d_equivalence_<model>.json.

  python tests/run_level/check07d_equivalence.py {geneformer,scgpt}     (in the model's FM env)
"""
import json
import runpy
import sys

import anndata
import numpy as np
import scipy.sparse as sp

from sc2bulk import determinism, folds, manifest, paths
from sc2bulk.config import settings

model_name = sys.argv[1]
determinism.configure()
D = runpy.run_path(str(settings.REPO / "pipeline" / "07d_embed_all_available.py"), run_name="check07d")
E = D["EMBED"]
unit = ("all", 0)
model, _ = E["load_model"](model_name)
genes = E["gene_table"]("hammond", "panel", unit)
target = E["targets"](model, model_name, genes)
chunks = list(D["shard_chunks"]("hammond", unit, genes["gene"].tolist(),
                                rows=slice(0, settings.MAX_STORED_CELLS), limit=5))
X, ids = sp.vstack([c[0] for c in chunks]).tocsr(), np.concatenate([c[1] for c in chunks])
embedded = [D["embed_chunk"](model, model_name, "hammond", target, c[0], c[1]) for c in chunks]
e, dropped = np.vstack([x[0] for x in embedded]), np.concatenate([x[1] for x in embedded])

stored = folds.training_cells("hammond", *unit)[ids, genes["gene"].tolist()].X
assert (sp.csr_matrix(stored) != X).nnz == 0, "shard rows 0-999 differ from stage 02's stored cells"

reference = paths.output("07_embed", "hammond", f"{model_name}_panel_cells_all.h5ad", unit)
reference_cells = anndata.read_h5ad(reference)[ids]
ref = reference_cells.X
assert (dropped == reference_cells.obs["genes_dropped"].to_numpy()).all(), "dropped-gene counts differ"
assert isinstance(ref, np.ndarray) and e.shape == ref.shape
e64, ref64 = e.astype(np.float64), ref.astype(np.float64)
cos = (e64 * ref64).sum(1) / np.linalg.norm(e64, axis=1) / np.linalg.norm(ref64, axis=1)
result = {"model": model_name, "git": manifest.git_state(), "reference": str(reference),
          "reference_hash": manifest.read(reference)["content_hash"], "cells": len(ids),
          "min_cosine": float(cos.min()), "max_abs_diff": float(np.abs(e64 - ref64).max())}
result["passed"] = bool(result["min_cosine"] >= 0.9999 and result["max_abs_diff"] <= 1e-3)
out = settings.OUTPUTS / "checks" / f"check07d_equivalence_{model_name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=1))
print(json.dumps(result), flush=True)
assert result["passed"], f"07d equivalence failed: {result}"
