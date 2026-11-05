"""Pineda et al. 2024 human motor cortex single-nucleus RNA-seq (Synapse syn51105515) -> NYGC
ALS Consortium / Target ALS motor cortex bulk RNA-seq (Prudencio et al. 2020, GEO GSE153960).
The label is ALS (1) vs. control (0).

Single-nucleus: one library per donor, from the authors' processed data (counts_fil.mtx:
integer counts, genes x nuclei, already QC'd by the authors; row_metadata: Ensembl id and
symbol; col_metadata: per-nucleus donor, condition, group and cell annotation). The sample
table S01 lists every library's donor, region and species, and every donor's group, sex,
age at death and postmortem interval. Selection: human motor cortex (BA4) libraries of
sporadic ALS ("ALS" in S01, "SALS" in col_metadata), C9ALS and pathologically normal (PN)
donors: 49 libraries, one per donor. FTLD, BA9 and mouse libraries are not used. Library
201019_ALS_102_snRNA-B2 is excluded by name (config: pairs.py), leaving 48 donors.

Genes without a symbol in row_metadata are named by their Ensembl id, on both sides of the
pair, so they stay separate genes.

Bulk: GEO GSE153960 RSEM gene counts (Ensembl ids with versions; RSEM expected counts,
rounded to integers) and the two series matrices (sample metadata). Motor cortex samples
(lateral, medial, unspecified) of "ALS Spectrum MND" (1) and "Non-Neurological Control" (0)
donors; other groups are not used. A subject (donor) can give a lateral and a medial
sample. Genes are restricted to Pineda's annotation (row_metadata ids, versions stripped,
summed per id) and named as on the single-nucleus side; the share of each sample's counts
outside that annotation is recorded.
"""
import gzip

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from sc2bulk.config import sources
from sc2bulk.config.pairs import PAIRS

GROUP_LABEL = {"ALS": 1, "C9ALS": 1, "PN": 0}          # S01 groups used
COL_GROUP = {"SALS": "ALS", "C9ALS": "C9ALS", "PN": "PN"}   # col_metadata group -> S01 group
EXCLUDED = PAIRS["pineda"]["excluded_samples"]
NYGC_ALS, NYGC_CONTROL = "ALS Spectrum MND", "Non-Neurological Control"


def _libraries():
    return sorted(set(sources.pineda_files()["library"]))


def sample_table(raw):
    """S01's donor table (real donors only; legend rows have no Donor) and library table."""
    sheets = pd.read_excel(raw / "pineda" / sources.PINEDA_SAMPLE_TABLE[0], sheet_name=None)
    donors = sheets["donor_information"].dropna(subset=["Donor"])
    donors = donors.assign(Donor=donors["Donor"].map(_donor_id))
    libraries = sheets["sample_mapping"].assign(Donor=lambda t: t["Donor"].map(_donor_id))
    return donors, libraries


def _donor_id(value) -> str:
    """Donor numbers as text ("101"); mouse donors are already text ("MS1")."""
    return str(int(value)) if isinstance(value, (int, float)) else str(value)


def _selected(raw):
    """The selected libraries with their donor rows; asserted to equal the pinned library list."""
    donors, libraries = sample_table(raw)
    chosen = libraries[(libraries["Species"] == "Human") & (libraries["Region"] == "BA4")].merge(donors, on="Donor")
    chosen = chosen[chosen["Group"].isin(list(GROUP_LABEL))]
    assert sorted(chosen["Sample_ID"]) == _libraries(), "the selection rule doesn't give the pinned library list"
    assert chosen["Donor"].is_unique, "a selected donor has two libraries"
    return chosen.set_index("Sample_ID")


def _row_metadata(raw):
    """Pineda's gene table, with `symbol` = the gene symbol, or the Ensembl id where the table
    has none (written as "N/A" or left empty)."""
    table = pd.read_csv(raw / "pineda" / _libraries()[0] / "row_metadata.tsv", sep="\t", keep_default_na=False)
    missing = table["Gene"].str.strip().isin(["", "N/A"])
    assert missing.sum() == PAIRS["pineda"]["expected"]["genes_without_symbol"], f"{missing.sum()} genes without a symbol"
    table["symbol"] = table["Gene"].where(~missing, table["ENSEMBL"])
    return table


def ensembl(raw) -> pd.DataFrame:
    table = _row_metadata(raw)
    return pd.DataFrame({"symbol": table["symbol"], "ensembl": table["ENSEMBL"]})


def single_cell_genes(raw):
    return sorted(set(_row_metadata(raw)["symbol"]))


def single_cell_samples(raw):
    chosen = _selected(raw)
    genes = _row_metadata(raw)
    mito = genes["Gene"].str.startswith("MT-").to_numpy()
    for library in _libraries():
        folder = raw / "pineda" / library
        row_metadata = pd.read_csv(folder / "row_metadata.tsv", sep="\t", keep_default_na=False)
        assert row_metadata["ENSEMBL"].equals(genes["ENSEMBL"]), f"{library}: genes differ between libraries"
        donor = chosen.loc[library]
        sample = {"sample_id": library, "donor_id": donor["Donor"], "holdout_group": donor["Donor"],
                  "label_value": GROUP_LABEL[donor["Group"]], "pineda_group": donor["Group"],
                  "sex": donor["Sex"], "age_at_death": float(donor["Age of death (Y)"]),
                  "pmi_hours": pd.to_numeric(donor["PMI_h"], errors="coerce"), "pmi_raw": str(donor["PMI_h"]),
                  "run": library.split("_")[0], "drop_reason": EXCLUDED.get(library)}
        if sample["drop_reason"]:
            yield sample, None, None, None
            continue
        cells = pd.read_csv(folder / "col_metadata.tsv", sep="\t", keep_default_na=False)
        assert (cells["Sample_ID"] == library).all() and (cells["Donor"].astype(str) == donor["Donor"]).all()
        assert (cells["Group"].map(COL_GROUP) == donor["Group"]).all(), f"{library}: group differs from S01"
        assert (cells["Sex"] == donor["Sex"]).all(), f"{library}: sex differs from S01"
        X = sp.csr_matrix(scipy.io.mmread(folder / "counts_fil.mtx").T)
        assert X.shape == (len(cells), len(genes)), f"{library}: matrix shape doesn't match its metadata"
        # the authors' QC, asserted so a changed provider file fails loudly
        total = np.asarray(X.sum(axis=1)).ravel()
        assert (np.diff(X.indptr) >= 200).all() and (np.asarray(X[:, mito].sum(axis=1)).ravel() < 0.2 * total).all(), \
            f"{library}: nuclei outside the authors' QC"
        cells = pd.DataFrame({"cell_id": cells["Barcode"].astype(str), "cell_type": cells["CellType"].astype(str),
                              "cell_class": cells["CellClass"].astype(str), "subtype": cells["SubType"].astype(str)})
        yield sample, X, list(genes["symbol"]), cells


def _series_matrices(raw) -> pd.DataFrame:
    """One row per GEO sample of both series matrices: gsm, title, source and the
    characteristics (tissue, subject id, group, sample id alt, project, ...)."""
    rows = []
    for name in sources.NYGC_SERIES_MATRICES:
        with gzip.open(raw / "nygc" / name, "rt") as f:
            lines = [line.rstrip("\n").split("\t") for line in f if line.startswith("!Sample_")]
        fields = {}
        for key, *values in lines:
            values = [v.strip('"') for v in values]
            if key == "!Sample_characteristics_ch1":
                fields[values[0].split(":")[0].strip()] = [v.split(":", 1)[1].strip() if ":" in v else "" for v in values]
            elif key in ("!Sample_geo_accession", "!Sample_title", "!Sample_source_name_ch1"):
                fields[key] = values
        rows.append(pd.DataFrame(fields).assign(platform=name.split("-")[1].split("_")[0]))
    return pd.concat(rows, ignore_index=True)


def bulk(raw):
    meta = _series_matrices(raw)
    meta = meta[meta["tissue"].str.contains("Cortex Motor") & meta["group"].isin([NYGC_ALS, NYGC_CONTROL])]
    counts = pd.read_csv(raw / "nygc" / sources.NYGC_COUNTS, sep="\t", index_col=0, keep_default_na=False)
    counts = counts.set_index("EnsemblID")
    assert meta["sample id alt"].is_unique
    has_counts = meta["sample id alt"].isin(counts.columns).to_numpy()
    assert (~has_counts).sum() == PAIRS["pineda"]["expected"]["bulk_without_counts"], f"{(~has_counts).sum()} samples without counts"
    # samples without counts stay in the sample table, with zero counts and a drop reason
    values = np.zeros((len(counts), len(meta)))
    values[:, has_counts] = counts[meta["sample id alt"][has_counts]].to_numpy(dtype=float)
    values = np.rint(values)   # RSEM expected counts; rounding changes totals by ~4e-5
    ids = counts.index.str.split(".").str[0]
    annotation = _row_metadata(raw).set_index("ENSEMBL")["symbol"]
    inside = ids.isin(annotation.index)
    total = values.sum(axis=0)
    X = values[inside].T
    samples = pd.DataFrame({
        "sample_id": meta["sample id alt"].to_numpy(), "donor_id": meta["subject id"].to_numpy(),
        "label_value": (meta["group"] == NYGC_ALS).astype(int).to_numpy(), "nygc_group": meta["group"].to_numpy(),
        "tissue": meta["tissue"].to_numpy(), "project": meta["project"].to_numpy(),
        "library_prep": meta["library preparation method"].to_numpy(), "platform": meta["platform"].to_numpy(),
        "fraction_counts_outside_annotation": np.divide(total - X.sum(axis=1), total, out=np.full(len(total), np.nan),
                                                        where=total > 0),
        "drop_reason": np.where(has_counts, None, "no counts in the GEO matrix"),
    })
    assert (samples.groupby("donor_id")["label_value"].nunique() == 1).all(), "a subject has two labels"
    return X, list(annotation.loc[ids[inside]]), samples
