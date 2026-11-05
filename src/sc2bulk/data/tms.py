"""Tabula Muris Senis: 10x droplet single-cell data of four organs (CELLxGENE) -> whole-organ
bulk RNA-seq of the same organs (Schaum et al. 2020, GEO GSE132040).

Single-cell: one CELLxGENE file per organ; raw counts in `.raw`, Ensembl ids as feature ids
and symbols in `feature_name`. The release is already QC'd by its providers (no
mitochondrial genes; at least 229 genes per cell). A sample is one organ of one mouse.
Ages come from the original `age` column ("18m"), because CELLxGENE's development_stage
merges every age above 20 months.

Each cell's release cohort is read from its index suffix, which the CELLxGENE release
carries over from the source objects: a bare "-1" for the 3-month cells (the 2018 Tabula
Muris; Tabula Muris Consortium 2018, Nature 562:367), "-<n>-0-0" for the 18/21/24-month
cells and "-<n>-1-0" for the 1/30-month cells (the later Tabula Muris Senis cohorts; the
split between those two is inferred from the suffix). Ages and cohorts are therefore
confounded (disclosed; Fig 1E).

Bulk: htseq counts (gene symbols as row names) and sample metadata. The bulk mice can't be
linked across organs (the sample name "<organ>_<n>" numbers samples within an organ), so
a bulk sample is its own donor. Samples between 6 and 18 months are outside both groups
and dropped by the 0/1 grouping.
"""
import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk.config import sources

ORGANS = tuple(sources.TMS_SINGLE_CELL)
COHORT = {None: "tabula_muris_2018", "0": "tms_cohort_0", "1": "tms_cohort_1"}


def release_cohort(cell_index: str) -> str:
    parts = cell_index.split("-")
    return COHORT[None if len(parts) == 2 else parts[3]]


def _single_cell_file(raw, organ):
    return raw / "tms" / f"tms_droplet_{organ.lower()}.h5ad"


def ensembl(raw) -> pd.DataFrame:
    var = anndata.read_h5ad(_single_cell_file(raw, ORGANS[0]), backed="r").raw.var
    return pd.DataFrame({"symbol": var["feature_name"].astype(str).to_numpy(), "ensembl": var.index.astype(str)})


def single_cell_genes(raw):
    genes = set()
    for organ in ORGANS:
        genes |= set(anndata.read_h5ad(_single_cell_file(raw, organ), backed="r").raw.var["feature_name"].astype(str))
    return sorted(genes)


def single_cell_samples(raw):
    for organ in ORGANS:
        adata = anndata.read_h5ad(_single_cell_file(raw, organ))
        X = sp.csr_matrix(adata.raw.X)
        symbols = list(adata.raw.var["feature_name"].astype(str))
        age = adata.obs["age"].astype(str).str.removesuffix("m").astype(float).to_numpy()
        mouse = adata.obs["donor_id"].astype(str).to_numpy()
        cohort = np.array([release_cohort(c) for c in adata.obs_names.astype(str)])
        for m in sorted(set(mouse)):
            rows = np.flatnonzero(mouse == m)
            assert len(set(age[rows])) == 1, f"mouse {m} has two ages"
            assert len(set(cohort[rows])) == 1, f"mouse {m} has cells of two release cohorts in {organ}"
            sample = {"sample_id": f"{organ}_{m}", "donor_id": m, "holdout_group": m, "label_value": age[rows[0]], "organ": organ,
                      "sex": adata.obs["sex"].astype(str).to_numpy()[rows[0]], "release_cohort": cohort[rows[0]],
                      "drop_reason": None}
            cells = pd.DataFrame({"cell_id": [f"{organ}_{c}" for c in adata.obs_names[rows]],
                                  "cell_type": adata.obs["cell_type"].astype(str).to_numpy()[rows]})
            yield sample, X[rows], symbols, cells


def bulk(raw):
    meta = pd.read_csv(raw / "tms" / sources.TMS_BULK_METADATA)
    meta = meta[~meta["source name"].astype(str).str.startswith("NA")]
    counts = pd.read_csv(raw / "tms" / sources.TMS_BULK_COUNTS, index_col=0, keep_default_na=False)
    counts.columns = counts.columns.str.removesuffix(".gencode.vM19")
    meta = meta[meta["Sample name"].isin(counts.columns)]
    organ = meta["source name"].str.rsplit("_", n=1).str[0]
    samples = pd.DataFrame({
        "sample_id": meta["source name"].to_numpy(),
        "donor_id": meta["source name"].to_numpy(),
        "label_value": pd.to_numeric(meta["characteristics: age"], errors="raise").astype(float).to_numpy(),
        "organ": organ.to_numpy(),
        "sex": meta["characteristics: sex"].map({"f": "female", "m": "male"}).to_numpy(),
        "drop_reason": np.where(organ.isin(ORGANS), None, "organ not in the four studied"),
    })
    X = counts[meta["Sample name"]].to_numpy().T
    return X, list(counts.index.astype(str)), samples
