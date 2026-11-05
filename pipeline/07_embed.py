"""Stage 07a: frozen foundation-model embeddings (Geneformer V2-104M, scGPT whole-human), in the
model's own environment.

Per pair, every stored cell, one-type cell, pseudobulk profile and bulk sample is embedded,
label-free, on two gene sets:
  panel       each unit's 5,000-gene panel (so per unit; the cross-fitted pair has 5 panels)
  full_genes  every shared gene (fold-independent; pair level)
Which samples a unit then uses (its training donors, and for cross-fitting its scored donors)
is decided where embeddings are read, through sc2bulk.folds; split67's held-out donors'
single-cell embeddings are never read by a split67 unit.

Genes: human pairs by the single-cell annotation (Geneformer: Ensembl ids through its
collapsing dictionary, symbols for genes with no id; scGPT: symbols, with genes named only by an
Ensembl id given their Ensembl 111 symbol); mouse pairs by one-to-one orthologs (panel:
<unit>/orthologs.csv; full genes: orthologs_shared.csv). Genes reaching the same token are
summed (Geneformer's collapse_gene_ids); unreadable genes are dropped and counted. Cell-level
embeddings are stored; pooling to samples (mean/max) and the Fig 4A n-cell bags are done
downstream from the shared bag drawer. Outputs, in <outputs>/07_embed/<pair>/[<unit>/]:
  <model>_<feature_set>_<input>_<cell_subset>.h5ad   obs: cell_id or sample_id, side (single_cell or
                                                     bulk: these files are fold-independent in membership;
                                                     a unit's sides are assigned where they're read); X float32
  <model>_<feature_set>_genes.csv                    readable genes, collapsed tokens, counts

  python pipeline/07_embed.py --model {geneformer,scgpt} [--pair PAIR] [--set SET]

--set runs one gene set only ("full_genes", or a unit's panel by its unit name, e.g. crossfit_0),
so one pair's sets can run in parallel on several GPUs. It selects work only: each set's outputs
and records are the same as when all sets run in one process.
"""
import argparse
import hashlib
import subprocess
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from sc2bulk import determinism, fm, folds, manifest, orthologs, paths
from sc2bulk.config import settings
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

FEATURE_SET_IDS = {"panel": 0, "full_genes": 1}
CHUNK = 20_000   # profiles embedded at a time (bounds memory: tokens, collated tensors)
INPUTS = (("cells", "all"), ("cells", "one_type"), ("pseudobulk", "all"), ("pseudobulk", "one_type"), ("bulk", "all"))


def file_hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(name):
    assert settings.GENEFORMER_REPO and settings.SCGPT_DIR, "set SC2BULK_GENEFORMER and SC2BULK_SCGPT (the frozen models' locations)"
    if name == "geneformer":
        repo = Path(settings.GENEFORMER_REPO)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        assert head == settings.GENEFORMER_COMMIT, f"Geneformer clone at {head}, not {settings.GENEFORMER_COMMIT}"
        for name_, expected in settings.GENEFORMER_FILES.items():
            assert file_hash(repo / name_) == expected, f"Geneformer {name_} differs from the pinned file (a Git LFS pointer?)"
        return fm.Geneformer(repo, repo / settings.GENEFORMER_CHECKPOINT), {
            "commit": settings.GENEFORMER_COMMIT, "files": settings.GENEFORMER_FILES}
    directory = Path(settings.SCGPT_DIR)
    assert hashlib.md5((directory / "best_model.pt").read_bytes()).hexdigest() == settings.SCGPT_CHECKPOINT_MD5, "scGPT checkpoint differs"
    return fm.ScGPT(directory), {"weights_md5": settings.SCGPT_CHECKPOINT_MD5,
                                 "vocab": file_hash(directory / "vocab.json"), "args": file_hash(directory / "args.json")}


def gene_table(pair, feature_set, unit) -> pd.DataFrame:
    """The gene set's genes with what each model reads: gene, ensembl, human_ensembl, human_symbol."""
    shared = pd.read_csv(paths.output("04_panel", pair, "shared_genes.csv"), keep_default_na=False)
    if feature_set == "panel":
        panel = pd.read_csv(paths.output("04_panel", pair, "panel.csv", unit), keep_default_na=False)
        shared = shared.set_index("gene").loc[panel["gene"]].reset_index()
    if PAIRS[pair]["species"] == "mouse":
        name = "orthologs.csv" if feature_set == "panel" else "orthologs_shared.csv"
        mapped = pd.read_csv(paths.output("04_panel", pair, name, unit if feature_set == "panel" else None),
                             keep_default_na=False).set_index("gene")
        shared["human_ensembl"] = mapped.loc[shared["gene"], "human_ensembl"].to_numpy()
        shared["human_symbol"] = mapped.loc[shared["gene"], "human_symbol"].to_numpy()
    else:
        names = orthologs.human_names(settings.RAW)   # Ensembl 111 symbols, for genes named only by an id
        shared["human_ensembl"] = shared["ensembl"]
        shared["human_symbol"] = [names.get(g, g) if g.startswith("ENSG") else g for g in shared["gene"]]
    return shared


def targets(model, name, genes: pd.DataFrame) -> np.ndarray:
    if name == "geneformer":
        return model.targets(genes["human_ensembl"].tolist(), genes["human_symbol"].tolist())
    return model.targets(genes["human_symbol"].tolist())


def profiles(pair, input_, cell_subset):
    """(counts AnnData on the single-cell or bulk gene axis, id column, side) for one input."""
    if input_ == "cells":
        data = folds.all_cells(pair, cell_subset)
        return data, data.obs_names.astype(str), "single_cell"
    if input_ == "pseudobulk":
        data = folds.all_pseudobulk(pair, cell_subset)
        return data, data.obs_names.astype(str), "single_cell"
    data = anndata.read_h5ad(settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad")
    return data, data.obs_names.astype(str), "bulk"


def embed_set(model, name, pair, feature_set, unit, record, cost):
    genes = gene_table(pair, feature_set, unit)
    target = targets(model, name, genes)
    readable = genes.assign(target=target, readable=target >= 0)
    write_csv(readable, paths.output("07_embed", pair, f"{name}_{feature_set}_genes.csv", unit), record, cost)
    for input_, cell_subset in INPUTS:
        if cell_subset == "one_type" and "one_type" not in PAIRS[pair]:
            continue
        data, ids, side = profiles(pair, input_, cell_subset)
        assert ids.is_unique, f"{pair} {input_}/{cell_subset}: ids aren't unique (seeds are keyed by id)"
        columns = data.var_names.get_indexer(genes["gene"])
        assert (columns >= 0).all()
        emb, dropped = [], []
        for start in range(0, data.n_obs, CHUNK):   # chunked: tokens and collated tensors stay bounded
            X, tokens = fm.collapse(data.X[start:start + CHUNK][:, columns], target)
            if name == "geneformer":
                e, d = model.embed(X, tokens)
            else:
                e, d = model.embed(X, tokens, fm.profile_seeds(pair, ids[start:start + CHUNK], FEATURE_SET_IDS[feature_set]))
            emb.append(e)
            dropped.append(d)
        emb, dropped = np.vstack(emb), np.concatenate(dropped)
        assert np.isfinite(emb).all(), f"{pair} {name} {feature_set} {input_}: non-finite embeddings"
        obs = pd.DataFrame({"id": ids, "side": side, "genes_dropped": dropped,
                            "sample_id": data.obs["sample_id"].astype(str).to_numpy() if input_ == "cells" else ids})
        out = anndata.AnnData(emb, obs=obs.set_index("id"))
        path = paths.output("07_embed", pair, f"{name}_{feature_set}_{input_}_{cell_subset}.h5ad", unit)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.write_h5ad(manifest.temporary(path))
        manifest.finish(path, record, cost)
        print(f"{pair} {name} {feature_set} {unit} {input_}/{cell_subset}: {emb.shape[0]} embedded, "
              f"{int(readable['readable'].sum())} of {len(genes)} genes readable, "
              f"{int((dropped > 0).sum())} profiles truncated", flush=True)


def write_csv(table, path, record, cost):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(manifest.temporary(path), index=False)
    manifest.finish(path, record, cost)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("geneformer", "scgpt"), required=True)
    parser.add_argument("--pair", choices=PAIR_ORDER)
    parser.add_argument("--set", help='one gene set: "full_genes" or a unit name (e.g. crossfit_0)')
    args = parser.parse_args()
    assert args.pair or not args.set, "--set needs --pair"
    determinism.configure()
    model, model_record = load_model(args.model)
    for pair in [args.pair] if args.pair else PAIR_ORDER:
        base = {"stage": "07_embed", "model": args.model, "pair": pair, "smoke": settings.SMOKE, **model_record,
                "code": manifest.stage_code_hash(settings.REPO / "pipeline" / "07_embed.py", pair),
                "inputs": manifest.input_hashes([*folds.inputs(pair, one_type="one_type" in PAIRS[pair]),
                                                 settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad",
                                                 paths.output("04_panel", pair, "shared_genes.csv")])}
        sets = [("full_genes", None)] + [("panel", u) for u in folds.units(pair)]
        if args.set:
            sets = [(f, u) for f, u in sets if (f if u is None else folds.unit_name(*u)) == args.set]
            assert len(sets) == 1, f"{pair}: no gene set named {args.set}"
        for feature_set, unit in sets:
            record = {**base, "feature_set": feature_set, "unit": unit,
                      "panel": manifest.read(paths.output("04_panel", pair, "panel.csv", unit))["content_hash"]
                      if unit else None,
                      "orthologs": manifest.read(paths.output("04_panel", pair, "orthologs.csv" if unit else "orthologs_shared.csv", unit))["content_hash"]
                      if PAIRS[pair]["species"] == "mouse" else None}
            done = [paths.output("07_embed", pair, f"{args.model}_{feature_set}_genes.csv", unit)] + \
                   [paths.output("07_embed", pair, f"{args.model}_{feature_set}_{i}_{c}.h5ad", unit)
                    for i, c in INPUTS if c == "all" or "one_type" in PAIRS[pair]]
            if manifest.is_done(done, record):
                print(f"{pair} {args.model} {feature_set} {unit}: done", flush=True)
                continue
            cost = manifest.begin(device="cuda")
            embed_set(model, args.model, pair, feature_set, unit, record, cost)


if __name__ == "__main__":
    main()
