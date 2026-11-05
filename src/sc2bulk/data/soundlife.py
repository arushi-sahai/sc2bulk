"""AIFI Sound Life: PBMC scRNA-seq -> whole-blood bulk RNA-seq from the same donors and the
same blood draws (Gong et al. 2025; the bulk also Smithmyer et al. 2025; data doi:10.57785/bn17-3h16;
no license is stated).

Single-cell input: one "slim" file per portal file. The eight portal files (age group x
sex x CMV; 10x 3' v3.1, GRCh38-2020-A, raw counts already QC'd by the provider; ~248 GB)
are reduced to a uniform random 3,000 cells per sample (all cells of a sample with fewer),
drawn before our QC. The kept cell barcodes of each slim are shipped with this repository
(slim_barcodes/; derived from the data above), and pipeline/00_soundlife_slims.py
rebuilds the slims from the portal files by selecting exactly those barcodes;
content_checksum() verifies the result. A uniform random subset of a uniform random subset
is a uniform random subset, so taking <= 1,000 post-QC cells from the slim is distributed
like taking them from all cells (stage 02 asserts that at least 1,000 of every training
sample's slim cells pass QC; test samples aren't used).

A sample is one blood draw (sampleKitGuid); a donor gives several. The provider made
repeated gene symbols unique by appending "-1", "-2" ("SOD2-1"); those suffixes are
removed, so the copies are summed with their gene like every other repeated symbol,
wherever the base symbol is present and the suffixed name is not itself a gene symbol of
the bulk annotation.

Every donor's single-cell samples are read and stored: the same donors appear on both
sides, so the pair is cross-fitted by donor (sc2bulk.folds), and which donors train a given
model is decided there, never here.

Bulk: one file of unstimulated whole-blood samples, raw counts, gene symbols with Ensembl
ids. Two kits were sequenced twice; the two libraries of a kit are summed (technical
replicates of one blood draw).
"""
import gzip
import hashlib
import re
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk.config import sources
from sc2bulk.config.pairs import PAIRS

BARCODES = Path(__file__).resolve().parent / "slim_barcodes"
SAMPLE_COLUMNS = ("sample_id", "donor_id", "numeric", "age_group", "sex", "cmv", "timepoint", "batch_id")
CELL_TYPES = ("AIFI_L1", "AIFI_L2", "AIFI_L3")
SLIM_OBS = SAMPLE_COLUMNS + CELL_TYPES + ("n_cells_full",)
# portal obs column -> slim obs column
PORTAL_COLUMNS = {"sample.sampleKitGuid": "sample_id", "subject.subjectGuid": "donor_id",
                  "sample.subjectAgeAtDraw": "numeric", "subject.ageGroup": "age_group",
                  "subject.biologicalSex": "sex", "subject.cmv": "cmv", "sample.visitName": "timepoint",
                  "batch_id": "batch_id", "AIFI_L1": "AIFI_L1", "AIFI_L2": "AIFI_L2", "AIFI_L3": "AIFI_L3"}


def slim_name(portal_name):
    return portal_name.removesuffix(".h5ad") + "__slim.h5ad"


def _slim_files(raw):
    return [raw / "soundlife" / slim_name(name) for name in sources.SOUNDLIFE_SINGLE_CELL]


def content_checksum(adata) -> str:
    """SHA-256 of a slim's content, independent of the h5ad library version: the shape, the
    canonical CSR counts (indices sorted, duplicates summed, zeros removed; data uint16,
    indices int32, indptr int64, little-endian), the cell barcodes and gene names, and the
    obs columns in a fixed order, as strings."""
    X = sp.csr_matrix(adata.X)
    assert np.array_equal(X.data, np.round(X.data)) and X.data.min() >= 0 and X.data.max() <= np.iinfo(np.uint16).max, \
        "counts must be integers in the uint16 range"
    X = X.astype(np.uint16)
    X.sum_duplicates()
    X.eliminate_zeros()
    X.sort_indices()
    sections = [np.asarray(X.shape, dtype="<i8").tobytes(), X.data.astype("<u2").tobytes(),
                X.indices.astype("<i4").tobytes(), X.indptr.astype("<i8").tobytes(),
                "\n".join(adata.obs_names.astype(str)).encode(), "\n".join(adata.var_names.astype(str)).encode(),
                *["\n".join(adata.obs[column].astype(str)).encode() for column in SLIM_OBS]]
    digest = hashlib.sha256()
    for section in sections:   # each section prefixed by its length, so sections can't run into each other
        digest.update(len(section).to_bytes(8, "little"))
        digest.update(section)
    return digest.hexdigest()


def build_slim(portal_path, barcodes):
    """The slim of one portal file: the listed cells (in the listed order) with their raw
    counts as uint16, the slim obs columns, and the gene symbols as given."""
    portal = anndata.read_h5ad(portal_path, backed="r")
    assert portal.var_names.is_unique, f"{portal_path}: gene names are not unique"
    rows = portal.obs_names.get_indexer(barcodes)
    assert (rows >= 0).all(), f"{portal_path}: some listed barcodes are not in the file"
    sub = portal[np.sort(rows)].to_memory()
    sub = sub[pd.Index(sub.obs_names).get_indexer(barcodes)]
    n_cells_full = pd.Series(portal.obs["sample.sampleKitGuid"].astype(str).to_numpy()).value_counts()
    portal.file.close()
    obs = pd.DataFrame({slim: sub.obs[col].to_numpy() for col, slim in PORTAL_COLUMNS.items()}, index=sub.obs_names.astype(str))
    obs["numeric"] = pd.to_numeric(obs["numeric"], errors="raise").astype(float)
    obs["n_cells_full"] = obs["sample_id"].astype(str).map(n_cells_full).astype(int).to_numpy()
    for column in SAMPLE_COLUMNS + CELL_TYPES:
        if column != "numeric":
            obs[column] = obs[column].astype(str)
    X = sp.csr_matrix(sub.X)
    assert X.max() <= np.iinfo(np.uint16).max, f"{portal_path}: a count exceeds uint16"
    return anndata.AnnData(X=X.astype(np.uint16), obs=obs, var=pd.DataFrame(index=sub.var_names.astype(str)))


def slim_barcodes(portal_name):
    with gzip.open(BARCODES / (portal_name.removesuffix(".h5ad") + ".txt.gz"), "rt") as f:
        return [line.rstrip("\n") for line in f]


def ensembl(raw) -> pd.DataFrame:
    var = anndata.read_h5ad(raw / "soundlife" / sources.SOUNDLIFE_BULK, backed="r").var
    return pd.DataFrame({"symbol": var.index.astype(str), "ensembl": var["id"].astype(str).to_numpy()})


def restore_names(names, known) -> list:
    """Remove the "-N" suffix the provider appended to repeated symbols, where the base
    symbol is also among the names and the suffixed name is not a known symbol."""
    names = [str(n) for n in names]
    present, known = set(names), set(known)
    restored = []
    for name in names:
        base = re.sub(r"-\d+$", "", name)
        restored.append(base if base != name and base in present and name not in known else name)
    return restored


def single_cell_genes(raw):
    known = ensembl(raw)["symbol"]
    genes = set()
    for path in _slim_files(raw):
        genes |= set(restore_names(anndata.read_h5ad(path, backed="r").var_names, known))
    return sorted(genes)


def single_cell_samples(raw):
    known = ensembl(raw)["symbol"]
    for path in _slim_files(raw):
        adata = anndata.read_h5ad(path, backed="r")
        obs = adata.obs
        symbols = restore_names(adata.var_names, known)
        for sample_id, rows in obs.groupby("sample_id", observed=True).indices.items():
            rows = np.sort(rows)
            first = obs.iloc[rows[0]]
            for column in SAMPLE_COLUMNS + ("n_cells_full",):
                assert obs[column].iloc[rows].nunique() == 1, f"{sample_id}: {column} varies within the sample"
            donor = str(first["donor_id"])
            sample = {"sample_id": str(sample_id), "donor_id": donor, "holdout_group": donor,
                      "label_value": float(first["numeric"]), "age_group": str(first["age_group"]),
                      "sex": str(first["sex"]), "cmv": str(first["cmv"]), "visit": str(first["timepoint"]),
                      "batch_id": str(first["batch_id"]), "cells_in_portal_file": int(first["n_cells_full"]),
                      "file": path.name, "drop_reason": None}
            X = sp.csr_matrix(adata.X[rows])
            cells = pd.DataFrame({"cell_id": adata.obs_names[rows].astype(str),
                                  **{c: obs[c].iloc[rows].astype(str).to_numpy() for c in CELL_TYPES}})
            cells["cell_type"] = cells["AIFI_L1"]
            yield sample, X, symbols, cells


def bulk(raw):
    adata = anndata.read_h5ad(raw / "soundlife" / sources.SOUNDLIFE_BULK)
    columns = {"sample.sampleKitGuid": "sample_id", "subject.subjectGuid": "donor_id",
               "sample.subjectAgeAtDraw": "label_value", "subject.ageGroup": "age_group",
               "subject.biologicalSex": "sex", "subject.cmv": "cmv", "sample.visitName": "visit"}
    obs = adata.obs[list(columns)].rename(columns=columns)
    X = np.asarray(adata.X)
    kits = obs["sample_id"].astype(str).to_numpy()
    unique_kits = list(dict.fromkeys(kits))
    X = np.vstack([X[kits == k].sum(axis=0) for k in unique_kits])
    for column in ("donor_id", "label_value", "age_group"):
        assert (obs.groupby("sample_id", observed=True)[column].nunique() == 1).all(), f"a kit has two values of {column}"
    first = obs.drop_duplicates("sample_id")
    samples = pd.DataFrame({
        "sample_id": first["sample_id"].astype(str).to_numpy(), "donor_id": first["donor_id"].astype(str).to_numpy(),
        "label_value": first["label_value"].astype(float).to_numpy(), "age_group": first["age_group"].astype(str).to_numpy(),
        "sex": first["sex"].astype(str).to_numpy(), "cmv": first["cmv"].astype(str).to_numpy(),
        "visit": first["visit"].astype(str).to_numpy(),
        "libraries": [int((kits == k).sum()) for k in unique_kits],
        "drop_reason": None,
    })
    assert list(samples["sample_id"]) == unique_kits
    return X, list(adata.var_names.astype(str)), samples
