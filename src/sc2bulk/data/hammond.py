"""Hammond et al. 2019 mouse microglia scRNA-seq (GEO GSE121654) -> Reemst et al. 2022
hippocampal microglia bulk RNA-seq (GEO GSE207067).

Single-cell: one genes x cells raw-count file per sample (10x 3' digital expression;
Cell Ranger 1.2, mm10). Each file lists the genes detected in that sample, so a gene
missing from a file has zero counts there. The files' gene names went through R's make.names: a dash became a dot
("mt-Nd1" -> "mt.Nd1", "H2-K1" -> "H2.K1") and a leading digit gained an "X"
("0610007P14Rik" -> "X0610007P14Rik"). The "X" before a leading digit is always removed
(no mouse symbol starts with X and a digit); dots are turned back into dashes only where
the dashed name is a mouse gene symbol of Reemst's Ensembl annotation and the dotted one
is not. The injury experiment (P100 LPC demyelination and its SALINE controls) is not
part of the ageing series and is dropped. Recorded per sample from the GEO records:
  chemistry   10x v1 for GSM3442006-21 (GEO read layout 98/14/10 bp) and v2 for the rest
              (26/98 bp); GEO doesn't name the chemistry, it is inferred from the layouts.
              The young group is mostly v1 and the old group entirely v2 (disclosed).
  batch       "ageing series"; "injury experiment" for the injury experiment's samples:
              the dropped LPC/SALINE samples and GSM3442038-40, its untreated whole-brain
              P100 controls (kept: same tissue and condition as the ageing series);
              "Percoll comparison" for GSM3442047-52.
  holdout_group  the sample itself, except the P5 Percoll comparison: "P5 female no k
              percoll" and "no percoll" (GSM3442047-52) share one holdout group per k, so
              a split never separates them. GEO doesn't say whether they are the same
              animal; their possible double weight is disclosed.

Bulk: one count file per sample (Ensembl id, symbol, raw count) and the series matrix
(age, stress condition, treatment). LPS-treated animals are dropped; the P200 animals kept
are the PBS-injected controls of the LPS experiment. A sample title "P9_9.1_ELS" is read as
age P9, litter 9, animal 1, condition ELS.
"""
import gzip
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk.config import sources

AGE_DAYS = {"E14": -4.5, "P4": 4.0, "P5": 5.0, "P30": 30.0, "P100": 100.0, "OLD": 540.0}
INJURY = ("LPC", "SALINE")
V1_CHEMISTRY = {f"GSM{n}" for n in range(3442006, 3442022)}
INJURY_BATCH_CONTROLS = {"GSM3442038", "GSM3442039", "GSM3442040"}


def _reemst_count_files(raw):
    return sorted((raw / "reemst").glob("GSM*.count.txt.gz"))


def ensembl(raw) -> pd.DataFrame:
    """Mouse gene symbol -> Ensembl id, from Reemst's count files (one row per Ensembl id)."""
    table = pd.read_csv(_reemst_count_files(raw)[0], sep="\t", header=None, names=["ensembl", "symbol", "count"],
                        keep_default_na=False)
    return table[["symbol", "ensembl"]]


def restore_names(names, known) -> list:
    """Undo R's make.names on gene names: the "X" before a leading digit removed, and dots
    turned back into dashes where the dashed name is a known symbol and the dotted one is
    not."""
    known = set(known)
    restored = []
    for name in names:
        name = re.sub(r"^X(?=\d)", "", str(name))   # no mouse symbol starts with X and a digit
        if name not in known and name.replace(".", "-") in known:
            name = name.replace(".", "-")
        restored.append(name)
    return restored


def _single_cell_files(raw):
    return sorted((raw / "hammond").glob("GSM*.dge.txt.gz"))


def _parse_file_name(path):
    """GSM3442008_E14_F_B12.dge.txt.gz -> (gsm, age in days, sex, drop reason)."""
    stem = path.name.removesuffix(".dge.txt.gz")
    parts = stem.split("_")
    if any(token in stem.upper() for token in INJURY):
        return parts[0], None, None, "injury experiment (LPC demyelination or its SALINE control)"
    sex = {"f": "female", "m": "male"}[parts[2][0].lower()] if parts[2][0].lower() in "fm" else None
    return parts[0], AGE_DAYS[parts[1].upper()], sex, None


def _read_dge(path, known):
    with gzip.open(path, "rt") as f:
        table = pd.read_csv(f, sep="\t", index_col=0, keep_default_na=False)
    X = sp.csr_matrix(table.to_numpy().T)
    return X, restore_names(table.index, known), list(table.columns)


def single_cell_genes(raw):
    known = ensembl(raw)["symbol"]
    genes = set()
    for path in _single_cell_files(raw):
        with gzip.open(path, "rt") as f:
            genes |= set(restore_names(pd.read_csv(f, sep="\t", usecols=[0], keep_default_na=False).iloc[:, 0], known))
    return sorted(genes)


def single_cell_samples(raw):
    known = ensembl(raw)["symbol"]
    for path in _single_cell_files(raw):
        gsm, age, sex, drop = _parse_file_name(path)
        percoll = re.search(r"_(no)?percoll_(\d)$", path.name.removesuffix(".dge.txt.gz"))
        batch = ("injury experiment" if gsm in INJURY_BATCH_CONTROLS or drop
                 else "Percoll comparison" if percoll else "ageing series")
        sample = {"sample_id": gsm, "donor_id": gsm,
                  "holdout_group": f"P5_female_{percoll.group(2)}" if percoll else gsm,
                  "label_value": age, "sex": sex, "chemistry": "v1" if gsm in V1_CHEMISTRY else "v2",
                  "batch": batch, "file": path.name, "drop_reason": drop}
        if drop:
            yield sample, None, None, None
            continue
        X, symbols, barcodes = _read_dge(path, known)
        cells = pd.DataFrame({"cell_id": [f"{gsm}_{b}" for b in barcodes], "cell_type": "microglia"})
        yield sample, X, symbols, cells


def _series_matrix(raw) -> pd.DataFrame:
    """One row per GEO sample: gsm, title, the characteristics, and the count file name."""
    with gzip.open(raw / "reemst" / sources.REEMST_SERIES_MATRIX, "rt") as f:
        lines = [line.rstrip("\n").split("\t") for line in f if line.startswith("!Sample_")]
    fields = {}
    for key, *values in lines:
        values = [v.strip('"') for v in values]
        if key == "!Sample_characteristics_ch1":
            name = values[0].split(":")[0].strip()
            fields[name] = [v.split(":", 1)[1].strip() for v in values]
        elif key in ("!Sample_geo_accession", "!Sample_title", "!Sample_supplementary_file_1"):
            fields[key] = values
    table = pd.DataFrame(fields).rename(columns={"!Sample_geo_accession": "gsm", "!Sample_title": "title"})
    table["file"] = table.pop("!Sample_supplementary_file_1").str.rsplit("/", n=1).str[1]
    return table


def bulk(raw):
    meta = _series_matrix(raw)
    title = meta["title"].str.extract(r"^P(?P<age>\d+)_(?P<litter>\d+)\.(?P<animal>\d+)_(?P<rest>.+)$")
    assert title.notna().all().all(), "unexpected Reemst sample title"
    samples = pd.DataFrame({
        "sample_id": meta["gsm"], "title": meta["title"],
        "label_value": title["age"].astype(float),
        "litter": "P" + title["age"] + "_" + title["litter"],
        "condition": meta["condition"], "treatment": meta["treatment"],
    })
    samples["donor_id"] = samples["sample_id"]
    samples["drop_reason"] = np.where(samples["treatment"].str.upper() == "LPS", "LPS-treated", None)
    columns, symbols = [], None
    for file in meta["file"]:
        table = pd.read_csv(raw / "reemst" / file, sep="\t", header=None, names=["ensembl", "symbol", "count"],
                            keep_default_na=False)
        if symbols is None:
            ids, symbols = table["ensembl"].to_numpy(), table["symbol"].astype(str).to_numpy()
        assert (table["ensembl"].to_numpy() == ids).all(), f"{file}: genes differ between count files"
        columns.append(table["count"].to_numpy())
    X = np.column_stack(columns).T
    return X, list(symbols), samples
