"""Stage 04c: QC embeddings (S2-S7 Fig panels C, D, G), label-free.

Single-cell: a UMAP of each pair's stored cells (each sample's first n stored cells, which are
the first n of its seeded cell order, so no new seed; n = QC_EMBED_CELLS // samples, a cap),
log-normalized over the shared genes and restricted to the primary unit's panel, then PCA,
a neighbor graph and UMAP with fixed random states and pinned numba threads. Both PCAs are
centered, not scaled. Bulk: PCA of the log-normalized bulk on the same panel. Labels are joined in the figure scripts.

Outputs, per pair, in <outputs>/04c_qc_embed/<pair>/:
  umap.parquet           cell_id, sample_id, cell_type (and organ, mouse organs), umap1, umap2
  bulk_pca.parquet       sample_id, pc1..pcN
  bulk_pca_variance.csv  component, explained_variance_ratio

  python pipeline/04c_qc_embed.py [--pair PAIR]
"""
import argparse

import anndata
import numba
import numpy as np
import pandas as pd
import pynndescent
import scanpy as sc
import sklearn
import umap
from sklearn.decomposition import PCA

from sc2bulk import determinism, folds, manifest, paths, preprocess
from sc2bulk.config import descriptive, settings
from sc2bulk.config.pairs import PAIR_ORDER

STAGE = "04c_qc_embed"
OUTPUTS = ("umap.parquet", "bulk_pca.parquet", "bulk_pca_variance.csv")


def panel_path(pair):
    return paths.output("04_panel", pair, "panel.csv", folds.units(pair)[0])


def input_paths(pair) -> list:
    return folds.inputs(pair) + [settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad",
                                 paths.output("04_panel", pair, "shared_genes.csv"), panel_path(pair)]


def record_for(pair) -> dict:
    return {"stage": STAGE, "pair": pair, "smoke": settings.SMOKE,
            "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair),
            "inputs": manifest.input_hashes(input_paths(pair)),
            "versions": {"scanpy": sc.__version__, "umap": umap.__version__, "pynndescent": pynndescent.__version__,
                         "numba": numba.__version__, "sklearn": sklearn.__version__}}


def first_cells(obs: pd.DataFrame, total: int) -> np.ndarray:
    """Each sample's first n rows (stage 02 stores a sample's cells in its seeded cell order), with
    n = total // samples: a cap, so a smaller sample gives all its cells."""
    n = total // obs["sample_id"].nunique()
    return (obs.groupby("sample_id", observed=True).cumcount() < n).to_numpy()


def embedded_cells(pair, shared, panel) -> pd.DataFrame:
    cells = folds.all_cells(pair)
    cells = cells[first_cells(cells.obs, descriptive.QC_EMBED_CELLS)].copy()   # a copy frees the full store
    X = preprocess.log_normalize(cells[:, shared].X)[:, pd.Index(shared).get_indexer(panel)]
    adata = anndata.AnnData(X)
    numba.set_num_threads(settings.N_THREADS)   # pynndescent and UMAP run on numba threads
    sc.pp.pca(adata, n_comps=descriptive.N_PCS, random_state=descriptive.RANDOM_STATE)
    sc.pp.neighbors(adata, n_neighbors=descriptive.N_NEIGHBORS, random_state=descriptive.RANDOM_STATE)
    sc.tl.umap(adata, random_state=descriptive.RANDOM_STATE)
    table = pd.DataFrame({"cell_id": cells.obs_names.astype(str), "sample_id": cells.obs["sample_id"].astype(str).to_numpy(),
                          "cell_type": cells.obs["cell_type"].astype(str).to_numpy()})
    if "organ" in cells.obs:
        table["organ"] = cells.obs["organ"].astype(str).to_numpy()
    return table.assign(umap1=adata.obsm["X_umap"][:, 0], umap2=adata.obsm["X_umap"][:, 1])


def bulk_pca(pair, shared, panel) -> tuple:
    bulk = anndata.read_h5ad(settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad")   # label-free bulk expression
    X = preprocess.log_normalize(bulk[:, shared].X)[:, pd.Index(shared).get_indexer(panel)].toarray()
    pca = PCA(n_components=descriptive.BULK_PCS, svd_solver="full")
    coords = pca.fit_transform(X)
    table = pd.DataFrame(coords, columns=[f"pc{i + 1}" for i in range(coords.shape[1])])
    table.insert(0, "sample_id", bulk.obs_names.astype(str))
    variance = pd.DataFrame({"component": table.columns[1:], "explained_variance_ratio": pca.explained_variance_ratio_})
    return table, variance


def main():
    determinism.configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER)
    args = parser.parse_args()
    for pair in [args.pair] if args.pair else PAIR_ORDER:
        record = record_for(pair)
        outputs = [paths.output(STAGE, pair, name) for name in OUTPUTS]
        if manifest.is_done(outputs, record):
            print(f"{pair}: done", flush=True)
            continue
        outputs[0].parent.mkdir(parents=True, exist_ok=True)
        unit = manifest.begin()
        shared = list(pd.read_csv(paths.output("04_panel", pair, "shared_genes.csv"), keep_default_na=False)["gene"])
        panel = list(pd.read_csv(panel_path(pair), keep_default_na=False)["gene"])
        coords = embedded_cells(pair, shared, panel)
        pcs, variance = bulk_pca(pair, shared, panel)
        inputs = input_paths(pair)
        for frame, path in zip((coords, pcs), outputs):
            frame.to_parquet(manifest.temporary(path), index=False)
            manifest.finish(path, record, unit, inputs=inputs)
        variance.to_csv(manifest.temporary(outputs[2]), index=False)
        manifest.finish(outputs[2], record, unit, inputs=inputs)
        print(f"{pair}: {len(coords)} cells embedded; bulk PC1-2 explain "
              f"{variance['explained_variance_ratio'].iloc[:2].sum():.0%}", flush=True)


if __name__ == "__main__":
    main()
