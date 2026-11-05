"""Human ileum, Crohn's: IBDverse terminal-ileum single-cell (CELLxGENE) -> MSCCR ileal-biopsy
bulk RNA-seq (GEO GSE193677). Selection rules: pre-declared, from
metadata only.

Single-cell: raw counts in `.raw`, Ensembl ids as feature ids, symbols in `feature_name`; the
file (32 GB) is read backed, one sample at a time (each sample is one contiguous block). Terminal ileum only (the rectum samples are
healthy only and the blood samples Crohn's only); Crohn's disease vs. normal; one sample per
donor (a Crohn's donor with an inflamed and an uninflamed sample keeps the inflamed one;
otherwise the sample with the most cells). Two samples hold fewer than 100 cells in the
provider's data (PREDECLARED_SMALL); they are dropped by id before any counts are read.

Bulk: counts per biopsy with GRCh37 Ensembl 75 ids. Each id takes the single-cell side's name
for that id where the single-cell annotation has it (symbols changed since Ensembl 75, so
matching by the old symbol would lose ~10,000 shared genes), and its own Ensembl id otherwise
(its GTF name could collide with another id's single-cell symbol and be summed into it: 46 cases).
Ileal biopsies of Crohn's disease and non-IBD controls (ulcerative colitis excluded); samples
with any non-integer count (exact half-integers, so not raw counts) are dropped, never
rounded; one biopsy per donor (a Crohn's donor's inflamed biopsy first, then the lowest
biopsy number). Only region, diagnosis, inflammation, age and sex are read from the
metadata: the clinical scores (SES-CD, HBI, endoscopic severity and remission, clinician
activity), CRP and calprotectin encode or track the label and are never stored. Inflammation
is defined for Crohn's biopsies only, so it encodes the label (a bulk label column). Age is
kept in the label-free metadata for the age-adjusted rows; it differs by group (controls are
older) without encoding the label, and no model reads it.
"""
import gzip
import re

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk.config import sources

PREDECLARED_SMALL = {"OTARscRNA13430447": 14, "5892STDY14478085": 17}   # sample -> cells (provider data)
DISEASE = {"single_cell": {"Crohn disease": 1.0, "normal": 0.0}, "bulk": {"CD": 1.0, "Control": 0.0}}
BATCH_PROXIES = ("bead_lot", "chip_lot", "gem_lot", "enzyme_lot_BLP", "chip_version", "bead_version")


def _single_cell(raw):
    return anndata.read_h5ad(raw / "crohns" / sources.CROHNS_SINGLE_CELL, backed="r")


def ensembl(raw) -> pd.DataFrame:
    var = _single_cell(raw).raw.var
    return pd.DataFrame({"symbol": var["feature_name"].astype(str).to_numpy(), "ensembl": var.index.astype(str)})


def single_cell_genes(raw):
    return sorted(set(_single_cell(raw).raw.var["feature_name"].astype(str)))


def selection(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per ileal sample of a Crohn's or normal donor, with its drop reason (None = kept)."""
    obs = obs[["donor_id", "sample_id", "tissue", "disease", "author_inflammation_status"]].astype(str)
    obs = obs[(obs["tissue"] == "ileum") & obs["disease"].isin(list(DISEASE["single_cell"]))]
    rows = obs.groupby(["donor_id", "sample_id", "disease", "author_inflammation_status"]).size().rename("cells").reset_index()
    rows["inflamed"] = (rows["author_inflammation_status"] == "inflamed").astype(int)
    rows = rows.sort_values(["donor_id", "inflamed", "cells", "sample_id"], ascending=[True, False, False, True])
    small = rows["sample_id"].isin(list(PREDECLARED_SMALL))
    assert set(rows.loc[rows["cells"] < 100, "sample_id"]) == set(PREDECLARED_SMALL), "the samples under 100 cells differ"
    reason = np.select([small, rows.duplicated("donor_id")],
                       ["fewer than 100 cells in the provider's data (pre-declared)",
                        "the donor's other sample is kept (inflamed first, then most cells)"], default="")
    rows["drop_reason"] = pd.Series(reason, index=rows.index).replace("", None)
    return rows.reset_index(drop=True)


def single_cell_samples(raw):
    adata = _single_cell(raw)
    obs = adata.obs
    symbols = list(adata.raw.var["feature_name"].astype(str))
    table = selection(obs)
    sample_of = obs["sample_id"].astype(str).to_numpy()
    first = obs.drop_duplicates("sample_id").set_index("sample_id")
    for row in table.itertuples():
        s = first.loc[row.sample_id]
        sample = {"sample_id": row.sample_id, "donor_id": row.donor_id, "holdout_group": row.donor_id,
                  "label_value": DISEASE["single_cell"][row.disease], "inflammation": row.author_inflammation_status,
                  "age_bin": str(s["author_age_binned_10"]), "sex": str(s["sex"]), "cells_in_sample": int(row.cells),
                  **{p: str(s[p]) for p in BATCH_PROXIES}, "drop_reason": row.drop_reason}
        if row.drop_reason is not None:
            yield sample, None, None, None
            continue
        rows = np.flatnonzero(sample_of == row.sample_id)
        assert rows[-1] - rows[0] + 1 == len(rows), f"{row.sample_id}: cells not contiguous in the file"
        X = sp.csr_matrix(adata.raw.X[rows[0]:rows[-1] + 1])   # one slice read (each sample is one block)
        cells = pd.DataFrame({"cell_id": adata.obs_names[rows].astype(str),
                              "cell_type": obs["cell_type"].astype(str).to_numpy()[rows],
                              "cell_category": obs["Clustering_category"].astype(str).to_numpy()[rows]})
        yield sample, X, symbols, cells


def series_metadata(path) -> pd.DataFrame:
    """The kept metadata fields of every biopsy, from the GEO series matrix."""
    fields, titles = {}, None
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = [p.strip('"') for p in line.rstrip("\n").split("\t")]
            if parts[0] == "!Sample_title":
                titles = [p.split(",")[0] for p in parts[1:]]
            elif parts[0] == "!Sample_characteristics_ch1":
                key = parts[1].split(": ")[0]
                if key in ("regionre", "ibd_disease", "typere", "study_eligibility_age_at_endo", "demographics_gender"):
                    fields[key] = [p.split(": ", 1)[1] for p in parts[1:]]
    meta = pd.DataFrame({"biopsy": titles, **fields})
    meta["donor_id"] = meta["biopsy"].str.split("_Biopsy").str[0]
    meta["biopsy_number"] = meta["biopsy"].str.split("_Biopsy_").str[1].astype(int)
    return meta


def gene_names(path) -> dict:
    names = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if "\tgene\t" in line:
                names[re.search(r'gene_id "([^"]+)"', line).group(1)] = re.search(r'gene_name "([^"]+)"', line).group(1)
    return names


def bulk(raw):
    meta = series_metadata(raw / "crohns" / sources.CROHNS_BULK_METADATA)
    counts = pd.read_csv(raw / "crohns" / sources.CROHNS_BULK_COUNTS, sep=" ", index_col=0)
    meta = meta[meta["biopsy"].isin(counts.columns)]
    ileum = meta[meta["regionre"] == "Ileum"].copy()
    values = counts[ileum["biopsy"]].to_numpy()
    integer = (values == np.round(values)).all(axis=0)
    ileum["inflamed"] = (ileum["typere"] == "I").astype(int)
    reason = np.select([~ileum["ibd_disease"].isin(list(DISEASE["bulk"])), ~integer],
                       ["ulcerative colitis (excluded)", "non-integer counts (not raw counts)"], default="")
    ileum["drop_reason"] = pd.Series(reason, index=ileum.index).replace("", None)
    eligible = ileum["drop_reason"].isna()
    order = ileum[eligible].sort_values(["donor_id", "inflamed", "biopsy_number"], ascending=[True, False, True])
    second = order.index[order.duplicated("donor_id")]
    ileum.loc[second, "drop_reason"] = "the donor's other ileal biopsy is kept (inflamed first, then lowest number)"
    # A bulk id takes the single-cell symbol if the id is in the single-cell data, else its own Ensembl id
    # (second freeze exception: the GTF75 name of a bulk-only id collided with another id's
    # single-cell symbol in 46 cases, summing unrelated genes). Ids sharing a single-cell symbol are summed,
    # as on the single-cell side.
    single_cell = ensembl(raw).drop_duplicates("ensembl").set_index("ensembl")["symbol"].to_dict()
    names = {i: single_cell.get(i, i) for i in gene_names(raw / "crohns" / sources.CROHNS_BULK_GENES)}
    assert not set(single_cell.values()) & {i for i in names if i not in single_cell}, "a bulk-only id is named like a single-cell gene"
    kept_genes = counts.index.isin(list(names))
    samples = pd.DataFrame({
        "sample_id": ileum["biopsy"].to_numpy(), "donor_id": ileum["donor_id"].to_numpy(),
        "label_value": ileum["ibd_disease"].map(DISEASE["bulk"]).to_numpy(),
        "inflammation": np.where(ileum["typere"] == "I", "inflamed", "uninflamed"),
        "age": pd.to_numeric(ileum["study_eligibility_age_at_endo"], errors="coerce").to_numpy(),
        "sex": ileum["demographics_gender"].str.lower().to_numpy(),
        "drop_reason": ileum["drop_reason"].to_numpy()})
    X = np.round(values[kept_genes]).T   # integer-valued for every kept sample (non-integer samples are dropped)
    return X, [names[g] for g in counts.index[kept_genes]], samples
