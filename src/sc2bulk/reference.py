"""The within-bulk reference: ridge cross-fitted
within the scored bulk itself, for stage 11b. It is not an upper bound (stated), and human blood, aging's is per
fold on that fold's scored bulk only.

Features: the unit's scored bulk as log-CP10k over the shared genes, on the unit's panel (pseudobulk's bulk
features). Outer CV: READOUT_FOLDS folds dealt per label over grouping units (mouse microglia: litter; human pairs:
donor; mouse organs: sample, dealt within organ), seed seeds.derived("within_bulk", pair, r, *unit), r = 0-9; inner
folds dealt the same way on the outer-training rows with the same seed. Each outer fold's rows are scored by a ridge
fit on the others (readout.run, alpha by inner grouped CV), standardized with the training rows' moments (within
organ on mouse organs). Labels are arguments; the caller reads them.
"""
import numpy as np
import pandas as pd

from sc2bulk import readout
from sc2bulk.folds import deal
from sc2bulk.config import settings


def dealt(groups: pd.DataFrame, seed: int) -> pd.Series:
    """Fold per grouping unit (groups: index = grouping unit, columns label and stratum): within each stratum
    (sorted), its units dealt per label to READOUT_FOLDS folds (folds.deal), with a stratum-specific seed."""
    out = {}
    for k, stratum in enumerate(sorted(set(groups["stratum"]))):
        part = groups[groups["stratum"] == stratum]["label"]
        out.update(deal(part, settings.READOUT_FOLDS, int(np.random.SeedSequence([seed, k]).generate_state(1)[0])))
    return pd.Series(out)


def folds_of(group, label, stratum, seed) -> np.ndarray:
    """Each row's fold: its grouping unit's (labels and strata constant within a grouping unit, asserted)."""
    table = pd.DataFrame({"group": np.asarray(group), "label": np.asarray(label), "stratum": np.asarray(stratum)})
    units = table.groupby("group")[["label", "stratum"]].agg(["nunique", "first"])
    assert (units[("label", "nunique")] == 1).all() and (units[("stratum", "nunique")] == 1).all(), \
        "a grouping unit spans labels or strata"
    of = dealt(pd.DataFrame({"label": units[("label", "first")], "stratum": units[("stratum", "first")]}), seed)
    return table["group"].map(of).to_numpy()


def cross_fit(X, label, group, stratum, outer, seed) -> np.ndarray:
    """Out-of-fold ridge scores of every row: each outer fold scored by a fit on the other folds' rows (inner
    folds dealt on those rows with the same seed), scaled with the training rows' moments per stratum."""
    X, label, group, stratum, outer = (np.asarray(a) for a in (X, label, group, stratum, outer))
    score = np.empty(len(X))
    for f in range(settings.READOUT_FOLDS):
        test = outer == f
        if not test.any():   # fewer grouping units than outer folds: the smoke run only
            assert settings.SMOKE, "an empty outer fold"
            continue
        train = ~test
        inner = folds_of(group[train], label[train], stratum[train], seed)
        out = readout.run(X[train], label[train], inner, None, strata_train=stratum[train], X_scored=X[test], strata_scored=stratum[test])
        score[test] = out["scored_score"]
    return score
