"""Absolute scores of a set of predictions: the scores kind's
columns, from scoring-unit means. Called by stage 11b (and by the released evaluate()); it takes labels as
arguments and reads no file.

- AUROC: statistics.auroc (ties 1/2).
- AUPRC: average precision with label 1 positive (secondary; tables only).
- Brier: the mean over scoring units of (unit-mean probability - label)^2. Per-side centering puts bulk
  probabilities near the single-cell prevalence, so Brier partly measures the prevalence shift (disclosed).
- signed -log10 p: sign(AUROC - 1/2) x (-log10 p), p the two-sided Mann-Whitney p of the unit means (asymptotic,
  tie-corrected; tables only).
"""
import numpy as np
import pandas as pd
from scipy import stats

from sc2bulk import statistics


def unit_means(table: pd.DataFrame) -> pd.DataFrame:
    """Per scoring unit, the mean score and probability over its samples (sorted by unit)."""
    return table.groupby("scoring_unit_id")[["score", "probability"]].mean().sort_index()


def average_precision(score, label) -> float:
    """sklearn's average_precision_score (label 1 positive): sum over distinct thresholds, descending, of the recall
    step times the precision there."""
    order = np.argsort(-score, kind="stable")
    s, y = score[order], label[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]   # each distinct threshold's last row
    tps = np.cumsum(y)[last]
    precision, recall = tps / (last + 1), tps / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def mann_whitney_p(score, label, auroc) -> float:
    """scipy's two-sided asymptotic Mann-Whitney p (tie-corrected, with the continuity correction), from the AUROC."""
    n1, n0 = int((label == 1).sum()), int((label == 0).sum())
    n = n0 + n1
    u = max(auroc, 1 - auroc) * n0 * n1
    ties = np.unique(score, return_counts=True)[1]
    sd = np.sqrt(n0 * n1 / 12 * ((n + 1) - (ties ** 3 - ties).sum() / (n * (n - 1))))
    return float(min(1.0, 2 * stats.norm.sf((u - n0 * n1 / 2 - 0.5) / sd))) if sd > 0 else 1.0


def score(unit_score, unit_probability, label) -> dict:
    """n0, n1, auroc, auprc, brier and signed_log10_p of scoring-unit means against their 0/1 labels."""
    unit_score, unit_probability, label = (np.asarray(x, dtype=float) for x in (unit_score, unit_probability, label))
    auroc = statistics.auroc(unit_score, label)
    p = mann_whitney_p(unit_score, label, auroc)
    return {"n0": int((label == 0).sum()), "n1": int((label == 1).sum()), "auroc": auroc,
            "auprc": average_precision(unit_score, label),
            "brier": float(np.mean((unit_probability - label) ** 2)),
            "signed_log10_p": float(np.sign(auroc - 0.5) * -np.log10(p))}
