"""Stage 11's numerical primitives: AUROC with resampling multiplicities,
Holm's step-down adjustment, bootstrap p-values, and the pooled-margin estimators. Pure functions,
tested against closed forms (tests/test_statistics.py)."""
import numpy as np
import pandas as pd
from scipy import stats


def auroc(score, label, weight=None) -> float:
    """Mann-Whitney AUROC of scores for 0/1 labels, ties counted 1/2; weight = each unit's
    multiplicity in a bootstrap resample (default 1)."""
    score, label = np.asarray(score, dtype=float), np.asarray(label)
    weight = np.ones(len(score)) if weight is None else np.asarray(weight, dtype=float)
    pos, neg = label == 1, label == 0
    assert pos.any() and neg.any() and (pos | neg).all()
    order = np.argsort(score, kind="stable")
    s, w, p = score[order], weight[order], pos[order]
    # for each distinct score: weighted negatives below it, plus half the negatives tied with it
    uniq, start = np.unique(s, return_index=True)
    neg_w = np.where(p, 0.0, w)
    neg_at = np.add.reduceat(neg_w, start)
    below = np.concatenate([[0.0], np.cumsum(neg_at)[:-1]])
    credit = below + neg_at / 2
    group = np.searchsorted(uniq, s)
    return float((w[p] * credit[group[p]]).sum() / (w[p].sum() * w[~p].sum()))


def holm(p) -> np.ndarray:
    """Holm's step-down adjusted p-values (monotone, capped at 1), in the input order."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def bootstrap_p(draws) -> tuple:
    """One-sided p-values from bootstrap draws of a difference: (better, worse) =
    ((1 + #{draw <= 0}) / (B + 1), (1 + #{draw >= 0}) / (B + 1))."""
    draws = np.asarray(draws, dtype=float)
    b = len(draws)
    return float((1 + (draws <= 0).sum()) / (b + 1)), float((1 + (draws >= 0).sum()) / (b + 1))


def inverse_variance(estimates, variances) -> tuple:
    """Fixed-effect pooled estimate and its variance."""
    w = 1 / np.asarray(variances, dtype=float)
    return float((w * estimates).sum() / w.sum()), float(1 / w.sum())


def heterogeneity(estimates, variances) -> dict:
    """Cochran's Q, I^2 and the DerSimonian-Laird tau^2."""
    y, v = np.asarray(estimates, dtype=float), np.asarray(variances, dtype=float)
    w = 1 / v
    mu, _ = inverse_variance(y, v)
    q = float((w * (y - mu) ** 2).sum())
    k = len(y)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return {"q": q, "i2": i2, "tau2": tau2}


def hartung_knapp(estimates, variances, level=0.95) -> dict:
    """Random-effects pooled estimate (DerSimonian-Laird tau^2) with the modified Hartung-Knapp
    variance, max(1, q) / sum(w) (never narrower than the random-effects normal variance), and its one-sided upper bound at `level` (t, k - 1 df)."""
    y, v = np.asarray(estimates, dtype=float), np.asarray(variances, dtype=float)
    k = len(y)
    tau2 = heterogeneity(y, v)["tau2"]
    w = 1 / (v + tau2)
    mu = float((w * y).sum() / w.sum())
    q = float((w * (y - mu) ** 2).sum() / (k - 1))
    var = max(1.0, q) / float(w.sum())
    return {"estimate": mu, "se": var ** 0.5, "tau2": tau2, "q": q,
            "upper": mu + stats.t.ppf(level, k - 1) * var ** 0.5}


# ---- The two-level paired bootstrap ----
#
# AUROC with unit multiplicities w is bilinear: AUROC = wp' M wn / (sum wp * sum wn), where M[i, j] =
# 1, 1/2 or 0 as case i's score is above, tied with or below control j's. A configuration's per-seed
# AUROC averaged over read-out seeds is therefore one product with M averaged over read-out seeds,
# so every resample is a small matrix product.

def comparison_matrix(score, label) -> np.ndarray:
    """M (cases x controls): 1 if the case scores higher, 1/2 if tied, 0 if lower."""
    score, label = np.asarray(score, dtype=float), np.asarray(label)
    case, control = score[label == 1][:, None], score[label == 0][None, :]
    return (case > control) + 0.5 * (case == control)


def unit_weights(label, strata, clusters, n: int, rng) -> tuple:
    """n resamples of scoring-unit multiplicities, as (cases (n x n1), controls (n x n0)) in the unit
    order of comparison_matrix. Within each stratum (always nested in label), either units are drawn
    with replacement (clusters None) or the stratum's clusters are, each bringing all its units."""
    label, strata = np.asarray(label), np.asarray(strata)
    if clusters is not None:
        frame = pd.DataFrame({"cluster": np.asarray(clusters), "stratum": strata})
        assert (frame.groupby("cluster")["stratum"].nunique() == 1).all(), "a cluster spans resampling strata"
    w = np.zeros((n, len(label)))
    for s in sorted(set(strata)):   # sorted: the draws don't depend on row order
        members = np.flatnonzero(strata == s)
        assert len(set(label[members])) == 1, "a resampling stratum mixes labels"
        if clusters is None:
            picks = rng.integers(0, len(members), size=(n, len(members)))
            np.add.at(w, (np.arange(n)[:, None], members[picks]), 1.0)
        else:
            names = sorted(set(np.asarray(clusters)[members]))
            of = {c: members[np.asarray(clusters)[members] == c] for c in names}
            picks = rng.integers(0, len(names), size=(n, len(names)))
            for k, c in enumerate(names):
                counts = (picks == k).sum(axis=1)
                w[:, of[c]] += counts[:, None]
    return w[:, np.asarray(label) == 1], w[:, np.asarray(label) == 0]


def auroc_resamples(M: np.ndarray, cases: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """AUROC under each resample's multiplicities: (n,)."""
    return np.einsum("bi,ij,bj->b", cases, M, controls, optimize=True) / (cases.sum(axis=1) * controls.sum(axis=1))


def logit_auroc(U, pairs):
    """logit of the shrunk AUROC (U + 1/2) / (pairs + 1): finite at AUROC 0 and 1."""
    a = (U + 0.5) / (pairs + 1)
    return np.log(a / (1 - a))


def fold_bootstrap(scores: dict, label, strata, clusters, n: int, rng, fixed: set, reference: str = "pseudobulk",
                   logit: bool = False) -> dict:
    """One unit's (fold's) paired bootstrap. scores: configuration -> array (encoder seeds in
    encoder_seed order, read-out seeds, scoring units); `fixed` names the configurations without an
    encoder level (one row, encoder_seed -1); every other configuration needs >= 2 seeds (with one it
    is "not estimable"). Returns configuration ->
    (point difference, n resampled differences) vs the reference: the mean over encoder seeds of the
    per-seed AUROC (mean over read-out seeds) minus the reference's. One set of uniforms per resample is
    shared by every configuration with an encoder level, each resampling its k seeds as floor(u k), so
    the seed draws are coupled across configurations whatever their seed counts;
    the unit multiplicities are shared by every configuration. logit: the logit-scale sensitivity row
    (per encoder and read-out seed, logit of the shrunk AUROC, then averaged over read-out seeds)."""
    label = np.asarray(label)
    for key, s in scores.items():
        assert (s.shape[0] == 1) if key in fixed else (s.shape[0] >= 2), f"{key}: {s.shape[0]} encoder seeds"
    cases, controls = unit_weights(label, strata, clusters, n, rng)
    widest = max([s.shape[0] for k, s in scores.items() if k not in fixed], default=1)
    uniform = rng.random(size=(n, widest))   # shared: configuration c resamples its k_c seeds as floor(u k_c)
    point, draws = {}, {}
    ones_c, ones_n = np.ones((1, int((label == 1).sum()))), np.ones((1, int((label == 0).sum())))
    for key, s in scores.items():
        if logit:
            M = [[comparison_matrix(s[e, r], label) for r in range(s.shape[1])] for e in range(s.shape[0])]   # 0, 1/2, 1: exact
            pairs, pairs1 = cases.sum(axis=1) * controls.sum(axis=1), ones_c.sum() * ones_n.sum()
            per_seed = np.stack([np.mean([logit_auroc(np.einsum("bi,ij,bj->b", cases, m, controls), pairs) for m in row], axis=0)
                                 for row in M])
            point[key] = float(np.mean([np.mean([logit_auroc((m.sum()), pairs1) for m in row]) for row in M]))
        else:
            # integer sums of 2M over read-out seeds, divided at the end: every product is exact in float64, so
            # the draws (and exact-zero ties) don't depend on the BLAS build
            R = s.shape[1]
            M = [np.sum([2 * comparison_matrix(s[e, r], label) for r in range(R)], axis=0) for e in range(s.shape[0])]
            per_seed = np.stack([auroc_resamples(m, cases, controls) for m in M]) / (2 * R)   # (seeds, n)
            point[key] = float(np.mean([auroc_resamples(m, ones_c, ones_n)[0] for m in M]) / (2 * R))
        if key in fixed:
            draws[key] = per_seed[0]
        else:
            k = s.shape[0]
            picks = np.floor(uniform[:, :k] * k).astype(int)                    # (resamples, k)
            draws[key] = np.take_along_axis(per_seed, picks.T, axis=0).mean(axis=0)
    return {k: (point[k] - point[reference], draws[k] - draws[reference]) for k in scores if k != reference}


def lower(draws, tail: float) -> float:
    """The lower bound with `tail` below it: the k-th smallest draw, k = tail x B (an integer, asserted)."""
    draws = np.sort(np.asarray(draws, dtype=float))
    k = tail * len(draws)
    assert np.isclose(k, round(k)), "tail x resamples must be an integer"
    return float(draws[round(k) - 1])


def upper(draws, tail: float) -> float:
    """The upper bound with `tail` above it: the k-th largest draw. With bootstrap_p's (1 + count) /
    (B + 1), p_worse <= tail iff upper < 0, and p_better <= tail iff lower > 0 (tested)."""
    return -lower(-np.asarray(draws, dtype=float), tail)


def summarize(results: dict, family: bool, level: float = 0.95) -> pd.DataFrame:
    """Per configuration: delta, the two-sided percentile CI, the one-sided upper bound u, the one-sided
    p-values and, when the rows are a family (a pair's primary rows, the pooled primary rows, the
    lupus + Crohn's row), their Holm adjustments within the "better" and "worse" families; other rows
    (LASSO, variants, logit) are unadjusted (Holm columns NaN)."""
    rows = []
    for key, (delta, draws) in results.items():
        better, worse = bootstrap_p(draws)
        rows.append({"key": key, "delta": delta, "ci_low": lower(draws, (1 - level) / 2),
                     "ci_high": upper(draws, (1 - level) / 2), "u": upper(draws, 1 - level),
                     "p_better": better, "p_worse": worse, "resamples": len(draws),
                     "mdd_obs": mdd(np.var(draws, ddof=1))})   # post hoc: from the bootstrap SD
    table = pd.DataFrame(rows)
    table["p_better_holm"] = holm(table["p_better"]) if family else np.nan
    table["p_worse_holm"] = holm(table["p_worse"]) if family else np.nan
    return table


def folds_mean(per_fold: list) -> dict:
    """The cross-fitted pair: per configuration, the mean over folds of the fold differences (each fold
    resampled independently: its donors within label, its encoder seeds)."""
    assert all(set(f) == set(per_fold[0]) for f in per_fold), "folds with different configurations"
    return {k: (float(np.mean([f[k][0] for f in per_fold])), np.mean([f[k][1] for f in per_fold], axis=0))
            for k in per_fold[0]}


def at_ceiling(pseudobulk_auroc: float, margin: float) -> bool:
    """A pair is excluded from pooling if pseudobulk's AUROC >= 1 - X (with a tolerance:
    AUROCs are rationals that can hit 1 - X exactly)."""
    return pseudobulk_auroc >= 1 - margin - 1e-12


def pool_equal(per_pair: dict) -> dict:
    """The equal-weight pooled difference over pairs ({pair: {configuration: (point, draws)}}); pairs
    are independent datasets, so their resamples are combined index by index."""
    pairs = list(per_pair)
    keys = per_pair[pairs[0]]
    assert all(set(per_pair[p]) == set(keys) for p in pairs), "pairs with different configurations"
    return {k: (float(np.mean([per_pair[p][k][0] for p in pairs])), np.mean([per_pair[p][k][1] for p in pairs], axis=0))
            for k in keys}


def pool_rows(per_pair: dict, level: float = 0.95) -> pd.DataFrame:
    """Inverse-variance (fixed-effect) and modified Hartung-Knapp rows from each pair's point difference
    and bootstrap variance, with heterogeneity."""
    assert len(per_pair) >= 2, "pooling needs at least 2 pairs (t has K - 1 df)"
    rows = []
    for k in per_pair[next(iter(per_pair))]:
        y = np.array([per_pair[p][k][0] for p in per_pair])
        v = np.array([np.var(per_pair[p][k][1], ddof=1) for p in per_pair])
        est, var = inverse_variance(y, v)
        hk, het = hartung_knapp(y, v, level), heterogeneity(y, v)
        rows.append({"key": k, "pool_set": "nonceiling_inverse_variance", "delta": est,
                     "u": est + stats.norm.ppf(level) * var ** 0.5})
        rows.append({"key": k, "pool_set": "nonceiling_hartung_knapp", "delta": hk["estimate"], "u": hk["upper"],
                     "hk_q": hk["q"], "q": het["q"], "i2": het["i2"], "tau2": het["tau2"]})
    return pd.DataFrame(rows)


# ---- Minimum detectable differences ----

def hanley_mcneil_variance(auroc: float, n0: int, n1: int) -> float:
    q1, q2 = auroc / (2 - auroc), 2 * auroc ** 2 / (1 + auroc)
    return (auroc * (1 - auroc) + (n1 - 1) * (q1 - auroc ** 2) + (n0 - 1) * (q2 - auroc ** 2)) / (n0 * n1)


def binormal_variance(auroc: float, n0: int, n1: int) -> float:
    """The variance of an AUROC estimate (Hanley-McNeil's form) with Q1 and Q2 exact for the binormal
    equal-variance model at that AUROC, not the exponential approximation: at unbalanced sizes the latter
    underestimates (lupus 18 v 99, A = 0.8: 0.0022 vs simulated 0.0033; binormal 0.0031; stage 11's
    simulation test checks this within 10%)."""
    from scipy import integrate
    a = stats.norm.ppf(auroc) * np.sqrt(2)                      # cases N(a, 1), controls N(0, 1)
    q1 = integrate.quad(lambda y: stats.norm.pdf(y) * stats.norm.sf(y - a) ** 2, -12, 12 + a)[0]   # two cases above a control
    q2 = integrate.quad(lambda x: stats.norm.pdf(x - a) * stats.norm.cdf(x) ** 2, -12, 12 + a)[0]  # a case above two controls
    return (auroc * (1 - auroc) + (n1 - 1) * (q1 - auroc ** 2) + (n0 - 1) * (q2 - auroc ** 2)) / (n0 * n1)


def difference_variance(n0: int, n1: int, auroc: float, rho: float, seed_sd: float = 0.0, seeds: int = 1) -> float:
    """Variance of a paired AUROC difference: 2 V (1 - rho) for the bulk units (V binormal), plus seed variance."""
    return 2 * binormal_variance(auroc, n0, n1) * (1 - rho) + seed_sd ** 2 / seeds


def mdd(variance: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """The difference a one-sided level-alpha test detects with the given power."""
    return float((stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) * variance ** 0.5)


def mdd_table(counts: dict, seed_sd: float, seeds: int, margin: float = 0.05, alpha: float = 0.05,
              holm_family: int = 9) -> pd.DataFrame:
    """The a priori MDD table: per pair and pooled (K = the pairs given), over
    rho x A, with the Holm first-step column and P(pooled u < X | delta = 0). counts: pair -> list of
    (n0, n1) per fold (the cross-fitted pair's folds: its per-fold-mean variance is the mean of the fold
    variances / folds)."""
    rows = []
    for rho in (0.3, 0.5, 0.7):
        for auroc in (0.7, 0.8, 0.9):
            variance = {p: np.mean([difference_variance(n0, n1, auroc, rho, seed_sd, seeds) for n0, n1 in f]) / len(f)
                        for p, f in counts.items()}
            pooled = sum(variance.values()) / len(variance) ** 2
            for name, v in [*variance.items(), (f"pooled K={len(variance)}", pooled)]:
                rows.append({"pair": name, "rho": rho, "auroc": auroc, "mdd": mdd(v, alpha),
                             "mdd_holm_first_step": mdd(v, alpha / holm_family),
                             "p_within_margin_at_zero": float(stats.norm.cdf(margin / v ** 0.5 - stats.norm.ppf(1 - alpha)))})
    return pd.DataFrame(rows)
