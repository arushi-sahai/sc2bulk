"""The deconvolution baseline's computations (stage 04b_deconvolve; numbers in config/deconvolution.py).

Weighted NNLS on a training-only marker signature, in count space. Every input here is training
single-cell data or label-free bulk; nothing reads a label.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import nnls
from scipy.stats import spearmanr

from sc2bulk.config import deconvolution as config


def profiles(X, sample_ids: np.ndarray, types: np.ndarray, totals: np.ndarray) -> dict:
    """Per (sample, type): summed counts over X's genes and the number of cells; per sample: mean
    total counts per cell. X: cells x genes (counts); totals: each cell's total counts."""
    samples, s_index = np.unique(sample_ids, return_inverse=True)
    names, k_index = np.unique(types, return_inverse=True)
    group = s_index * len(names) + k_index
    indicator = sp.csr_matrix((np.ones(len(group), dtype=np.float32), (group, np.arange(len(group)))),
                              shape=(len(samples) * len(names), len(group)))
    summed = np.asarray((indicator @ sp.csr_matrix(X, dtype=np.float32)).todense()).reshape(len(samples), len(names), -1)
    n = np.bincount(group, minlength=len(samples) * len(names)).reshape(len(samples), len(names))
    depth = pd.Series(totals).groupby(s_index).mean().to_numpy()
    return {"samples": samples, "types": names, "summed": summed, "n": n, "depth": depth}


def merge(prof: dict, source: str, target: str) -> dict:
    """Merge type `source` into `target` (counts and cells are additive)."""
    types = list(prof["types"])
    i, j = types.index(source), types.index(target)
    summed, n = prof["summed"].copy(), prof["n"].copy()
    summed[:, j] += summed[:, i]
    n[:, j] += n[:, i]
    keep = [t for t in range(len(types)) if t != i]
    return {**prof, "types": np.array(types)[keep], "summed": summed[:, keep], "n": n[:, keep]}


def signature(prof: dict, samples=None) -> np.ndarray:
    """(genes x types signature, global mean counts per cell): per-cell mean profiles, depth-normalized
    per sample (divided by the sample's mean counts per cell, times the global mean), averaged over
    samples with >= MIN_CELLS_PER_SAMPLE_TYPE cells of the type; a type no sample reaches
    that for (possible in a small V1 split) averages over the samples that have any of its cells.
    samples: a subset of prof["samples"] (default all)."""
    rows = np.arange(len(prof["samples"])) if samples is None else np.flatnonzero(np.isin(prof["samples"], samples))
    depth = prof["depth"][rows]
    scale = (depth.mean() / depth)[:, None, None]
    n = prof["n"][rows]
    per_cell = prof["summed"][rows] * scale / np.maximum(n, 1)[:, :, None]
    use = n >= config.MIN_CELLS_PER_SAMPLE_TYPE
    use[:, ~use.any(axis=0)] = n[:, ~use.any(axis=0)] > 0
    assert use.any(axis=0).all(), "a reference type has no cells in these samples"
    use = use[:, :, None]
    count = use.sum(axis=0)
    return ((per_cell * use).sum(axis=0) / count).T, depth.mean()


def type_lists(prof: dict) -> tuple:
    """(reference types, feature types) by their share of the training cells (inclusive thresholds)."""
    cells = prof["n"].sum(axis=0)
    share = cells / cells.sum()
    reference = (share >= config.REFERENCE_FRACTION) & (cells >= config.REFERENCE_MIN_CELLS)
    feature = reference & (share >= config.FEATURE_FRACTION)
    return prof["types"][reference], prof["types"][feature]


def cpm(X) -> np.ndarray:
    X = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float64)
    return X / X.sum(axis=1, keepdims=True) * 1e6


def candidates(genes: pd.Index, bulk_cpm: np.ndarray, pseudobulk_cpm: np.ndarray, excluded=()) -> np.ndarray:
    """Marker candidates among `genes`: technical genes and `excluded` removed, then the label-free
    mismatch filter: |log2(mean bulk CPM / mean training pseudobulk CPM)| <= MISMATCH_LOG2."""
    technical = genes.str.startswith(config.EXCLUDED_PREFIXES) | genes.isin(excluded)
    b, p = bulk_cpm.mean(axis=0), pseudobulk_cpm.mean(axis=0)
    ratio = np.log2((b + 1e-9) / (p + 1e-9))
    return ~technical & (np.abs(ratio) <= config.MISMATCH_LOG2) & (b > 0) & (p > 0)


def markers(S: np.ndarray, global_depth: float, allowed: np.ndarray) -> tuple:
    """(marker gene indices, per-type count of genes reaching MARKER_MIN_LOG2_RATIO): per type, the
    top MARKERS_PER_TYPE allowed genes by log2(type mean / max other type mean), among genes whose
    type mean is >= MARKER_MIN_CP10K counts per 10k."""
    cp10k = S / global_depth * 1e4
    chosen, strong = set(), []
    for k in range(S.shape[1]):
        other = np.delete(S, k, axis=1).max(axis=1) if S.shape[1] > 1 else np.zeros(len(S))
        ratio = np.log2((S[:, k] + 1e-9) / (other + 1e-9))
        ok = np.flatnonzero(allowed & (cp10k[:, k] >= config.MARKER_MIN_CP10K))
        best = ok[np.argsort(-ratio[ok], kind="stable")][:config.MARKERS_PER_TYPE]
        chosen |= set(best.tolist())
        strong.append(int((ratio[ok] >= config.MARKER_MIN_LOG2_RATIO).sum()))
    return np.array(sorted(chosen)), strong


def fit(S: np.ndarray, mixtures: np.ndarray) -> tuple:
    """Weighted NNLS of each mixture (rows; CPM on the marker genes) on the signature S (markers x
    types): min ||D(b - S w)||, D = 1 / each gene's mean over types. Returns (fractions w / sum w,
    residual 1 - R^2 on the weighted genes)."""
    d = 1.0 / S.mean(axis=1)
    A = S * d[:, None]
    fractions, residuals = [], []
    for b in mixtures:
        y = b * d
        w, _ = nnls(A, y)
        assert w.sum() > 0, "a mixture got all-zero weights"
        fractions.append(w / w.sum())
        residuals.append(((y - A @ w) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return np.array(fractions), np.array(residuals)


def close(fractions: np.ndarray, columns, keep) -> pd.DataFrame:
    """Fractions over `columns`, restricted to `keep` and re-closed to sum to 1."""
    table = pd.DataFrame(fractions, columns=columns)[list(keep)]
    return table.div(table.sum(axis=1), axis=0)


def clr(fractions: pd.DataFrame) -> pd.DataFrame:
    """Centered log-ratio after a pseudocount and re-closure."""
    f = fractions + config.CLR_PSEUDOCOUNT
    f = f.div(f.sum(axis=1), axis=0)
    logs = np.log(f)
    return logs.sub(logs.mean(axis=1), axis=0)


def most_similar(S: np.ndarray, types, k: str, feature) -> str:
    """The feature type whose log signature correlates most (Spearman, over the given genes) with k's."""
    i = list(types).index(k)
    scores = {t: spearmanr(np.log1p(S[:, i]), np.log1p(S[:, list(types).index(t)]))[0] for t in feature if t != k}
    return max(sorted(scores), key=lambda t: -np.inf if np.isnan(scores[t]) else scores[t])
