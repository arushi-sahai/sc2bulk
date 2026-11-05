"""Stage 07d: foundation-model embeddings of the all-available store's extra cells (Fig 4A's
"up to 3,000 cells" point).

The store is nested (stage 02b): each sample's first MAX_STORED_CELLS cells are stage 02's stored
cells, already embedded by stage 07. This stage embeds only the rest (shard rows after the first
MAX_STORED_CELLS) of each unit's training samples (Fig 4A's read-out trains on them; held-out
donors' extra cells are never needed), on each unit's panel for the primary fold set (the headline features), with
stage 07's own gene mapping and embedding loop (its functions, loaded from pipeline/07_embed.py), so a
sample's up-to-3,000-cell embeddings are stage 07's first 1,000 plus these. Label-free.

Outputs, in <outputs>/07d_embed_all_available/<pair>/<unit>/:
  <model>_panel_cells_extra.h5ad   obs: id (cell_id), sample_id, side, genes_dropped; X float32

  python pipeline/07d_embed_all_available.py --model {geneformer,scgpt} --pair PAIR [--set UNIT]
"""
import argparse
import runpy

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk import all_available, codehash, determinism, fm, folds, manifest, paths
from sc2bulk.config import settings
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

STAGE = "07d_embed_all_available"
EMBED = runpy.run_path(str(settings.REPO / "pipeline" / "07_embed.py"), run_name="stage07")   # its functions, not its main


def shard_chunks(pair, unit, genes, rows=slice(settings.MAX_STORED_CELLS, None), limit=None):
    """(counts on `genes`, cell ids, sample ids), about CHUNK cells at a time: `rows` of the unit's
    training samples' shards (default: the rows after the stored ones), streamed one shard at a time
    so memory stays bounded; `limit` takes the first shards only (the equivalence check)."""
    X, ids, samples = [], [], []
    for sample_id, path in sorted(all_available.training_shards(pair, *unit).items())[:limit]:
        part = all_available.load(path)[rows]
        if part.n_obs:
            X.append(sp.csr_matrix(part[:, genes].X))
            ids.extend(part.obs_names.astype(str))
            samples.extend([sample_id] * part.n_obs)
        if sum(x.shape[0] for x in X) >= EMBED["CHUNK"]:
            yield sp.vstack(X).tocsr(), np.array(ids), np.array(samples)
            X, ids, samples = [], [], []
    if X:
        yield sp.vstack(X).tocsr(), np.array(ids), np.array(samples)


def embed_chunk(model, model_name, pair, target, X, ids):
    """Stage 07's embedding of one chunk of cells: (embeddings, genes dropped per cell)."""
    Xc, tokens = fm.collapse(X, target)
    if model_name == "geneformer":
        return model.embed(Xc, tokens)
    return model.embed(Xc, tokens, fm.profile_seeds(pair, ids, EMBED["FEATURE_SET_IDS"]["panel"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("geneformer", "scgpt"), required=True)
    parser.add_argument("--pair", choices=PAIR_ORDER, required=True)
    parser.add_argument("--set", help="one unit (e.g. crossfit_0); default: every unit of the primary fold set")
    args = parser.parse_args()
    determinism.configure()
    pair = args.pair
    primary = next(iter(PAIRS[pair]["fold_sets"]))
    units = [u for u in folds.units(pair) if u[0] == primary and (not args.set or folds.unit_name(*u) == args.set)]
    assert units, f"{pair}: no primary unit named {args.set}"
    model, model_record = EMBED["load_model"](args.model)
    index = paths.output("02b_all_available", pair, "index.csv")
    base = {"stage": STAGE, "model": args.model, "pair": pair, "smoke": settings.SMOKE, **model_record,
            "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair, codehash.extra(STAGE, pair)),
            "inputs": manifest.input_hashes([index, paths.output("04_panel", pair, "shared_genes.csv")]),
            "shards": manifest.read(index)["content_hash"]}
    for unit in units:
        record = {**base, "unit": unit, "panel": manifest.read(paths.output("04_panel", pair, "panel.csv", unit))["content_hash"],
                  "orthologs": manifest.read(paths.output("04_panel", pair, "orthologs.csv", unit))["content_hash"]
                  if PAIRS[pair]["species"] == "mouse" else None}
        path = paths.output(STAGE, pair, f"{args.model}_panel_cells_extra.h5ad", unit)
        if manifest.is_done([path], record):
            print(f"{pair} {args.model} {unit}: done", flush=True)
            continue
        cost = manifest.begin(device="cuda")
        genes = EMBED["gene_table"](pair, "panel", unit)
        target = EMBED["targets"](model, args.model, genes)
        emb, dropped, ids, samples = [], [], [], []
        for X, chunk_ids, chunk_samples in shard_chunks(pair, unit, genes["gene"].tolist()):
            e, d = embed_chunk(model, args.model, pair, target, X, chunk_ids)
            emb.append(e)
            dropped.append(d)
            ids.append(chunk_ids)
            samples.append(chunk_samples)
        emb, dropped, ids, samples = np.vstack(emb), np.concatenate(dropped), np.concatenate(ids), np.concatenate(samples)
        assert pd.Index(ids).is_unique
        assert np.isfinite(emb).all(), f"{pair} {args.model} {unit}: non-finite embeddings"
        obs = pd.DataFrame({"id": ids, "side": "single_cell", "genes_dropped": dropped, "sample_id": samples}).set_index("id")
        path.parent.mkdir(parents=True, exist_ok=True)
        anndata.AnnData(emb, obs=obs).write_h5ad(manifest.temporary(path))
        manifest.finish(path, record, cost)
        print(f"{pair} {args.model} {folds.unit_name(*unit)}: {emb.shape[0]} extra cells embedded", flush=True)


if __name__ == "__main__":
    main()
