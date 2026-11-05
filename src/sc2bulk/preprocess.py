"""Per-cell and per-sample preprocessing rules shared by every pair (stage 02).

Nothing here is fit across samples, and nothing reads a label: every rule is fixed in
advance (config/settings.py) and applied to one cell or one sample at a time. Only raw
counts are produced; normalization happens where the data are used.
"""
import hashlib

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk.config import settings

MITO_PREFIX = {"mouse": "mt-", "human": "MT-"}


def assert_symbols(symbols):
    """Every feature must have a real name: no missing, empty or "nan" symbols (a reader
    gives genes without a symbol their Ensembl id instead). Otherwise all of them would be
    summed into one fake gene."""
    s = pd.Series(list(symbols), dtype=object)
    assert s.notna().all() and not s.astype(str).str.strip().isin(["", "nan", "NaN", "None", "N/A", "NA"]).any(), \
        "a feature has no gene symbol"


def is_gene(symbols) -> np.ndarray:
    """False for features that are not genes: htseq summary counters (__no_feature, ...) and
    ERCC spike-ins."""
    assert_symbols(symbols)
    s = pd.Index(symbols).astype(str)
    return ~(s.str.startswith("__") | s.str.startswith("ERCC-"))


def gene_axis(symbols):
    """The unique gene symbols of a feature list, with non-genes removed, in sorted order."""
    s = pd.Index(symbols).astype(str)
    return pd.Index(sorted(set(s[is_gene(s)])))


def to_gene_axis(X, symbols, genes):
    """Columns of X (samples or cells x features named by symbols) summed onto the gene axis
    `genes`: features sharing a symbol are summed, non-gene features are dropped, and genes
    absent from this feature list get zero counts. Returns a CSR matrix."""
    symbols = pd.Index(symbols).astype(str)
    keep = np.flatnonzero(is_gene(symbols))
    target = genes.get_indexer(symbols[keep])
    assert (target >= 0).all(), "a gene of this feature list is missing from the gene axis"
    collapse = sp.csr_matrix((np.ones(len(keep)), (keep, target)), shape=(len(symbols), len(genes)))
    return (sp.csr_matrix(X) @ collapse).tocsr()


def merged_symbols(symbols) -> pd.Series:
    """Symbols carried by more than one feature (their counts are summed), with the number
    of features each one merges."""
    s = pd.Series(pd.Index(symbols).astype(str))
    s = s[is_gene(s)]
    counts = s.value_counts()
    return counts[counts > 1]


def assert_counts(X, what):
    data = X.data if sp.issparse(X) else np.asarray(X)
    assert (data >= 0).all() and np.array_equal(data, np.round(data)), f"{what}: not non-negative integer counts"


def cell_qc(X, genes, species, max_percent_mito=settings.QC_MAX_PERCENT_MITO) -> pd.DataFrame:
    """Per-cell QC on the gene axis: genes detected, total counts, % mitochondrial counts,
    and whether the cell passes the QC thresholds (the mito ceiling can be set per pair:
    pairs.py `qc_max_percent_mito`)."""
    X = sp.csr_matrix(X)
    mito = np.asarray(genes.str.startswith(MITO_PREFIX[species]))
    total = np.asarray(X.sum(axis=1)).ravel()
    mito_counts = np.asarray(X[:, mito].sum(axis=1)).ravel()
    qc = pd.DataFrame({
        "genes_detected": np.diff(X.indptr),
        "total_counts": total,
        "percent_mito": np.divide(100 * mito_counts, total, out=np.zeros_like(total, dtype=float), where=total > 0),
    })
    qc["passed_qc"] = ((qc.genes_detected >= settings.QC_MIN_GENES_PER_CELL)
                       & (qc.percent_mito < max_percent_mito))
    return qc


def sample_seed(pair: str, sample_id: str) -> int:
    """A seed fixed by the pair and sample id (stable across runs and machines)."""
    return int.from_bytes(hashlib.sha256(f"{pair}:{sample_id}".encode()).digest()[:8], "little")


def cell_order(pair: str, sample_id: str, n_cells: int) -> np.ndarray:
    """A seeded random permutation of a sample's post-QC cells. Every selection of a
    sample's cells takes them in this order, so "the first n" is a random n and the
    selections nest."""
    return np.random.default_rng(sample_seed(pair, sample_id)).permutation(n_cells)


def stored_cells(pair: str, sample_id: str, n_cells: int) -> np.ndarray:
    """Indices of the cells kept for a sample: the first MAX_STORED_CELLS in cell_order."""
    return cell_order(pair, sample_id, n_cells)[:settings.MAX_STORED_CELLS]


def stored_type_cells(pair: str, sample_id: str, is_type) -> np.ndarray:
    """Indices of the cells of one type kept for a sample (Fig 6C): cell_order filtered to
    the type, the first MAX_STORED_CELLS. The type's cells among the main stored cells come
    first. `is_type` is a boolean array over the sample's post-QC cells."""
    order = cell_order(pair, sample_id, len(is_type))
    return order[np.asarray(is_type)[order]][:settings.MAX_STORED_CELLS]


def summed(X) -> np.ndarray:
    """Counts summed over the rows (cells) of X: a pseudobulk profile of raw counts."""
    return np.asarray(sp.csr_matrix(X).sum(axis=0)).ravel()


def group_of(value, groups: dict):
    """The 0/1 group a label value falls in (inclusive ranges from config), or None."""
    for group, (low, high) in groups.items():
        if low <= value <= high:
            return group
    return None


def log_normalize(X):
    """The one normalization of the pipeline: counts on the shared
    genes, scaled to TARGET_SUM per row over those genes, then log1p. Returns float32 CSR."""
    X = sp.csr_matrix(X, dtype=np.float32, copy=True)
    totals = np.asarray(X.sum(axis=1, dtype=np.float64)).ravel()
    assert (totals > 0).all(), "a profile has no counts on the shared genes"
    X.data *= np.repeat(settings.TARGET_SUM / totals, np.diff(X.indptr)).astype(np.float32)   # in place: one copy only
    np.log1p(X.data, out=X.data)
    return X


def seurat_dispersion(X, chunk=20_000) -> pd.DataFrame:
    """scanpy's highly_variable_genes(flavor="seurat") statistics, computed in row chunks:
    per gene, the mean and variance (ddof 1) of expm1 of the log-normalized values, the log
    dispersion, the log1p mean, and the dispersion normalized within N_DISPERSION_BINS
    equal-width mean bins (a bin with one gene gets mean 0 and that gene's dispersion as SD,
    as scanpy does). X: log-normalized cells x genes (CSR)."""
    n, total, total_sq = X.shape[0], np.zeros(X.shape[1]), np.zeros(X.shape[1])
    for start in range(0, n, chunk):
        block = sp.csr_matrix(X[start:start + chunk], dtype=np.float64)
        block.data = np.expm1(block.data)
        total += np.asarray(block.sum(axis=0)).ravel()
        total_sq += np.asarray(block.multiply(block).sum(axis=0)).ravel()
    mean = total / n
    var = (total_sq / n - mean ** 2) * n / (n - 1)
    mean[mean == 0] = 1e-12
    dispersion = var / mean
    dispersion[dispersion == 0] = np.nan
    table = pd.DataFrame({"means": np.log1p(mean), "dispersions": np.log(dispersion)})
    table["mean_bin"] = pd.cut(table["means"], bins=settings.N_DISPERSION_BINS)
    grouped = table.groupby("mean_bin", observed=False)["dispersions"]
    bin_mean, bin_sd = grouped.mean(), grouped.std(ddof=1)
    one_gene = bin_sd.isnull()
    bin_sd[one_gene.values] = bin_mean[one_gene.values].values
    bin_mean[one_gene.values] = 0
    table["dispersions_norm"] = ((table["dispersions"].values - bin_mean[table["mean_bin"].values].values)
                                 / bin_sd[table["mean_bin"].values].values)
    return table.drop(columns="mean_bin")
