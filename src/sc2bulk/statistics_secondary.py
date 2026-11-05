"""Stage 11b's numerical primitives: the pieces stage 11's sc2bulk.statistics lacks, as pure functions on arrays. sc2bulk.statistics itself is
used unedited. Labels are arguments; this module reads no file (tests/test_statistics_secondary.py).

- point_delta: a configuration's point difference without resampling (the oracle's selection).
- max_statistic: Fig 4's best-candidate test: T = max_v delta_v / se_v over the candidate
  set, the null max_v (draw_v - delta_v) / se_v, se_v = SD of the draws; one-sided p = (1 + #{null >= T}) / (B + 1);
  candidates with se_v = 0 (both separating perfectly) are left out and listed.
- age_residuals / residual_auroc: the covariate-adjusted AUROC (Janes & Pepe 2008): scores
  residualized on age by a least-squares line among the controls.
- permuted_summary, summary_rank, decomposition: the rows without a stream.
- aggregate_scores: the scores table's averaging (stage 11's rule) and the S11B seed variability.
"""
import numpy as np
import pandas as pd
from scipy import stats

from sc2bulk import statistics


def point_delta(scores: dict, label, reference: str) -> dict:
    """configuration -> the mean over encoder seeds of the per-seed AUROC (mean over read-out seeds) minus the
    reference's, as statistics.fold_bootstrap's point estimate. scores: arrays (encoder seeds, read-out seeds, units)."""
    mean = {k: float(np.mean([np.mean([statistics.auroc(s[e, r], label) for r in range(s.shape[1])]) for e in range(s.shape[0])]))
            for k, s in scores.items()}
    return {k: mean[k] - mean[reference] for k in scores if k != reference}


def max_statistic(results: dict, candidates) -> dict:
    """The max-statistic test over the candidate set (results: configuration -> (point delta, draws), from one
    fold_bootstrap call or its folds mean). Returns T, the p-value, each candidate's se, and the candidates left
    out for se = 0; T and p are NaN if none remains."""
    se = {v: float(np.std(results[v][1], ddof=1)) for v in candidates}
    kept = [v for v in candidates if se[v] > 0]
    excluded = sorted(v for v in candidates if se[v] == 0)
    if not kept:
        return {"t": np.nan, "p": np.nan, "se": se, "excluded": excluded, "kept": []}
    assert all(se[v] > 0 for v in kept)
    t = max(results[v][0] / se[v] for v in kept)
    null = np.max(np.stack([(results[v][1] - results[v][0]) / se[v] for v in kept]), axis=0)
    return {"t": float(t), "p": float((1 + (null >= t).sum()) / (len(null) + 1)), "se": se, "excluded": excluded, "kept": kept}


def age_residuals(scores, age, label) -> np.ndarray:
    """scores (..., units) residualized on age, per leading index, by the least-squares line fit among the
    controls (label 0): the covariate-adjusted AUROC's input (Janes & Pepe 2008)."""
    scores, age, label = np.asarray(scores, dtype=float), np.asarray(age, dtype=float), np.asarray(label)
    controls = label == 0
    design = np.column_stack([np.ones(controls.sum()), age[controls]])
    flat = scores.reshape(-1, scores.shape[-1])
    beta = np.linalg.lstsq(design, flat[:, controls].T, rcond=None)[0]   # (2, rows)
    fitted = (np.column_stack([np.ones(len(age)), age]) @ beta).T
    return (flat - fitted).reshape(scores.shape)


def residual_auroc(score, age, label, min_controls: int):
    """The covariate-adjusted AUROC of one set of scores, or None if fewer than min_controls controls with
    distinct ages are available to fit the line."""
    age, label = np.asarray(age, dtype=float), np.asarray(label)
    if len(np.unique(age[label == 0])) < min_controls:
        return None
    return statistics.auroc(age_residuals(score, age, label), label)


def permuted_summary(deltas) -> dict:
    """The label-permuted control's summary over permutations j: the mean and the 95th percentile of delta_j."""
    deltas = np.asarray(deltas, dtype=float)
    return {"permuted_mean": float(deltas.mean()), "permuted_q95": float(np.quantile(deltas, 0.95))}


def summary_rank(per_pair: dict) -> pd.DataFrame:
    """Per pair, each configuration's rank of its point difference (pseudobulk = 0 included; rank 1 = the largest;
    ties share the mean rank), and the mean rank over pairs. per_pair: pair -> {configuration: delta}."""
    ranks = {pair: pd.Series(stats.rankdata([-d for d in deltas.values()]), index=list(deltas)) for pair, deltas in per_pair.items()}
    table = pd.DataFrame(ranks)
    return table.assign(mean_rank=table.mean(axis=1))


def decomposition(results: dict, models) -> dict:
    """Fig 2C's step from a model's pseudobulk input to its cells input: model -> (delta(model, cells) -
    delta(model, pseudobulk), the paired draws' difference), from one set of stage 11 results."""
    return {f"{m}_cells": (results[f"{m}_cells"][0] - results[f"{m}_pseudobulk"][0], results[f"{m}_cells"][1] - results[f"{m}_pseudobulk"][1])
            for m in models}


def aggregate_scores(table: pd.DataFrame, by) -> pd.DataFrame:
    """The scores table's per-configuration AUROC by stage 11's averaging: per unit (fold), the mean over read-out
    seeds then over encoder seeds; then the mean over the folds. With it, the SD over encoder seeds of the per-seed
    AUROC and the SD over read-out seeds of the per-read-out-seed AUROC (S11B), each averaged over folds. by: the
    key columns that name a configuration (besides fold and the seeds)."""
    by = list(by)
    per_e = table.groupby([*by, "fold", "encoder_seed"])["auroc"].mean().reset_index()
    per_r = table.groupby([*by, "fold", "readout_seed"])["auroc"].mean().reset_index()
    fold = per_e.groupby([*by, "fold"])["auroc"].agg(["mean", lambda a: a.std(ddof=1) if len(a) > 1 else np.nan]).reset_index()
    fold.columns = [*by, "fold", "auroc", "sd_encoder_seeds"]
    fold_r = per_r.groupby([*by, "fold"])["auroc"].agg(lambda a: a.std(ddof=1) if len(a) > 1 else np.nan).rename("sd_readout_seeds").reset_index()
    return fold.merge(fold_r, on=[*by, "fold"]).groupby(by)[["auroc", "sd_encoder_seeds", "sd_readout_seeds"]].mean().reset_index()
