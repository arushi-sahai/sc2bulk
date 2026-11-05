"""Stage 10c's computations: gene shifts on both sides and the shared-gene shares (Fig 5A-B, S14B-C, S9 Table). Label-bearing: the bulk shift
uses bulk labels, which the caller reads through sc2bulk.labels.bulk_labels (whitelisted in
tests/test_firewall.py, as this module is).

A shift is (mean of group 1 - mean of group 0) / pooled SD (ddof 1), per gene on log-CP10k, computed per scoring
unit (a donor's samples averaged first where the pair scores donors) and within each stratum (mouse organs:
within organ, then the mean over organs with equal weight). A gene is shared when its two shifts have the same,
nonzero sign. No bootstrap, no test (descriptive).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sc2bulk.config import genes as G


def shift(X: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Per column, (mean of group 1 - mean of group 0) / pooled SD (ddof 1); 0 where the pooled SD is 0."""
    group = np.asarray(group)
    a, b = X[group == 1], X[group == 0]
    assert len(a) >= 2 and len(b) >= 2, "a shift needs >= 2 scoring units per group"
    pooled = np.sqrt(((len(a) - 1) * a.var(axis=0, ddof=1) + (len(b) - 1) * b.var(axis=0, ddof=1)) / (len(a) + len(b) - 2))
    delta = a.mean(axis=0) - b.mean(axis=0)
    return np.divide(delta, pooled, out=np.zeros_like(delta), where=pooled > 0)


def side_shift(X: np.ndarray, table: pd.DataFrame, skip_short: bool = False) -> np.ndarray:
    """One side's shift: table (rows of X) holds group, scoring_unit_id and stratum. Rows are averaged per scoring
    unit (its group asserted constant), the shift is taken within each stratum, then averaged over strata.
    skip_short (the smoke run only): a stratum with fewer than 2 scoring units in a group is skipped."""
    out = []
    for _, part in table.reset_index(drop=True).groupby("stratum", sort=True):
        assert (part.groupby("scoring_unit_id")["group"].nunique() == 1).all(), "a scoring unit has two groups"
        units = part.groupby("scoring_unit_id", sort=True)
        group = units["group"].first().to_numpy()
        if skip_short and min((group == 1).sum(), (group == 0).sum()) < 2:
            continue
        means = np.vstack([X[rows.index.to_numpy()].mean(axis=0) for _, rows in units])
        out.append(shift(means, group))
    assert out, "no stratum has 2 scoring units per group"
    return np.mean(out, axis=0)


def is_shared(single_cell: pd.Series, bulk: pd.Series) -> pd.Series:
    """Same nonzero sign on both sides (a zero shift on either side is not shared)."""
    return (np.sign(single_cell) == np.sign(bulk)) & (single_cell != 0) & (bulk != 0)


def matched_base(top_genes, single_cell: pd.Series, shared: pd.Series, n: int = G.MATCHED_NEIGHBOURS) -> float:
    """The expected shared share among genes matched on single-cell shift size: for each top gene,
    its n non-top genes nearest in |single-cell shift| (ties by symbol), averaged with multiplicity."""
    rest = single_cell.drop(list(top_genes)).abs()
    names = rest.index.to_numpy()
    values = []
    for g in top_genes:
        distance = np.abs(rest.to_numpy() - abs(single_cell[g]))
        order = np.lexsort((names, distance))[:n]
        values += list(shared.loc[names[order]])
    return float(np.mean(values))


def shares(top_genes, single_cell: pd.Series, bulk: pd.Series) -> dict:
    """Fig 5B's statistic and its two reference marks for one top-k list."""
    shared = is_shared(single_cell, bulk)
    return {"shared_share": float(shared.loc[list(top_genes)].mean()), "base_all": float(shared.mean()),
            "base_matched": matched_base(top_genes, single_cell, shared)}


def rho(single_cell: pd.Series, bulk: pd.Series) -> float:
    """The Spearman correlation of the two shifts over the panel genes (S9 Table; Fig 5A)."""
    return float(spearmanr(single_cell, bulk.loc[single_cell.index])[0])
