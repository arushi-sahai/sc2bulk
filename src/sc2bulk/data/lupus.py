"""Human blood, lupus: Perez et al. 2022 PBMC single-cell (CELLxGENE) -> Hung et al. 2015
whole-blood bulk RNA-seq (SRP062966, via recount3). Selection rules:
pre-declared, from metadata only.

Single-cell: raw counts in `.raw`, Ensembl ids as feature ids, symbols in `feature_name`;
read from the backed file one sample at a time. The ImmVar controls (donor ids "IGTB…") are
excluded, as by the authors. Processing cohort is confounded with the label (cohort 2 holds
SLE donors only), so only donors whose majority processing cohort is 4 are kept, and only
their processing-cohort-4 cells (17 such donors also have cells processed in another
cohort). Each donor keeps the sample with the most cohort-4 cells (ties by sample id).

Bulk: recount3 stores per-gene base-coverage sums. Read-equivalent counts are coverage
divided by the read length (single-end 50 bp for every run, asserted), rounded to the
nearest integer. Each sample's transformed total must lie within 0.8-1.0 of its sequenced
spots (asserted). Genes are identified by version-stripped Ensembl ids (recount3's G026 annotation).
Bulk genes take the single-cell symbol for every Ensembl id the single-cell data has (else their own
Ensembl id), so both sides name a gene the same way and a bulk-only gene can't take a shared name. One run per donor. Only the disease status is read from the metadata: the other attributes
(titles "SLE#"/"control#", anti-Ro, and an interferon score computed from the expression
itself) encode the label and are never stored.
"""
import gzip
import re

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk.config import sources

COHORT = "4.0"
READ_LENGTH = 50
DISEASE = {"single_cell": {"systemic lupus erythematosus": 1.0, "normal": 0.0},
           "bulk": {"systemic lupus erythematosus (SLE)": 1.0, "healthy": 0.0}}


def _single_cell(raw):
    return anndata.read_h5ad(raw / "lupus" / sources.LUPUS_SINGLE_CELL, backed="r")


def ensembl(raw) -> pd.DataFrame:
    var = _single_cell(raw).raw.var
    return pd.DataFrame({"symbol": var["feature_name"].astype(str).to_numpy(), "ensembl": var.index.astype(str)})


def single_cell_genes(raw):
    return sorted(set(_single_cell(raw).raw.var["feature_name"].astype(str)))


def selection(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per (donor, sample): its cohort-4 cell count and the drop reason (None = kept)."""
    obs = obs[["donor_id", "sample_uuid", "Processing_Cohort"]].astype(str)
    majority = obs.groupby("donor_id")["Processing_Cohort"].agg(lambda c: c.value_counts().idxmax())
    rows = obs.groupby(["donor_id", "sample_uuid"]).agg(
        cells=("Processing_Cohort", "size"), cohort4_cells=("Processing_Cohort", lambda c: int((c == COHORT).sum())))
    rows = rows.reset_index().sort_values(["donor_id", "cohort4_cells", "sample_uuid"], ascending=[True, False, True])
    best = ~rows.duplicated("donor_id")
    reason = np.select(
        [rows["donor_id"].str.startswith("IGTB"),
         rows["donor_id"].map(majority).ne(COHORT),
         ~best],
        ["ImmVar control (excluded, as by the authors)",
         "donor's majority processing cohort is not 4",
         "the donor's other sample has more processing-cohort-4 cells"], default="")
    rows["drop_reason"] = pd.Series(reason, index=rows.index).replace("", None)
    return rows.reset_index(drop=True)


def single_cell_samples(raw):
    adata = _single_cell(raw)
    obs = adata.obs
    symbols = list(adata.raw.var["feature_name"].astype(str))
    table = selection(obs)
    sample_of = obs["sample_uuid"].astype(str).to_numpy()
    cohort = obs["Processing_Cohort"].astype(str).to_numpy()
    first = obs.drop_duplicates("sample_uuid").set_index("sample_uuid")   # attributes of each sample, not each donor
    for row in table.itertuples():
        d = first.loc[row.sample_uuid]
        sample = {"sample_id": row.sample_uuid, "donor_id": row.donor_id, "holdout_group": row.donor_id,
                  "label_value": DISEASE["single_cell"][str(d["disease"])], "sex": str(d["sex"]),
                  "ethnicity": str(d["self_reported_ethnicity"]), "development_stage": str(d["development_stage"]),
                  "disease_state": str(d["disease_state"]), "cells_in_sample": int(row.cells),
                  "cohort4_cells": int(row.cohort4_cells), "drop_reason": row.drop_reason}
        if row.drop_reason is not None:
            yield sample, None, None, None
            continue
        rows = np.flatnonzero((sample_of == row.sample_uuid) & (cohort == COHORT))
        X = sp.csr_matrix(adata.raw.X[rows])
        cells = pd.DataFrame({"cell_id": adata.obs_names[rows].astype(str),
                              "cell_type": obs["cell_type"].astype(str).to_numpy()[rows],
                              "author_cell_type": obs["author_cell_type"].astype(str).to_numpy()[rows]})
        yield sample, X, symbols, cells


def gene_names(path) -> dict:
    """Version-stripped Ensembl id -> gene name, from a GTF's gene lines."""
    names = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if "\tgene\t" in line:
                gene = re.search(r'gene_id "([^".]+)', line).group(1)
                names[gene] = re.search(r'gene_name "([^"]+)"', line).group(1)
    return names


def bulk(raw):
    meta = pd.read_csv(raw / "lupus" / sources.LUPUS_BULK_METADATA, sep="\t", keep_default_na=False)
    lengths = meta["read_info"].str.extract(r"average:(\d+),stdev:(\d+)").astype(int)
    assert (meta["library_layout"] == "single").all() and (lengths[0] == READ_LENGTH).all() and (lengths[1] == 0).all()
    assert meta["sample_name"].is_unique, "a donor has two runs"
    disease = meta["sample_attributes"].str.extract(r"disease status;;([^|]*)")[0]
    sums = pd.read_csv(raw / "lupus" / sources.LUPUS_BULK_COUNTS, sep="\t", comment="#", index_col=0)
    sums = sums[meta["external_id"]]
    counts = np.rint(sums.to_numpy(dtype=np.float64) / READ_LENGTH).T
    ratio = counts.sum(axis=1) / meta["run_total_spots"].to_numpy()
    assert ((ratio >= 0.8) & (ratio <= 1.0)).all(), f"recount3 transform: totals/spots outside 0.8-1.0 ({ratio.min():.2f}-{ratio.max():.2f})"
    # Genes by Ensembl id (version stripped; symbol matching across annotation versions
    # lost ~9,600 shared genes): the single-cell annotation's symbol where the id is in the single-cell
    # data, else the Ensembl id itself (such genes can't be shared, and a G026 symbol could collide
    # with another id's single-cell symbol). Ids sharing a single-cell symbol are summed, as stage 02
    # sums them on the single-cell side; a bulk-only id never shares a name.
    single_cell = ensembl(raw).drop_duplicates("ensembl").set_index("ensembl")["symbol"].to_dict()
    ids = sums.index.str.split(".").str[0]
    assert ids.isin(list(gene_names(raw / "lupus" / sources.LUPUS_BULK_GENES))).all(), "a recount3 gene is not in the G026 annotation"
    names = {i: single_cell.get(i, i) for i in ids}
    assert not set(single_cell.values()) & {i for i in ids if i not in single_cell}, "a bulk-only id is named like a single-cell gene"
    samples = pd.DataFrame({"sample_id": meta["sample_name"].to_numpy(), "donor_id": meta["sample_name"].to_numpy(),
                            "label_value": disease.map(DISEASE["bulk"]).to_numpy(), "drop_reason": None})
    assert samples["label_value"].notna().all(), f"unexpected disease status: {set(disease)}"
    return counts.astype(np.int64), [names[i] for i in ids], samples
