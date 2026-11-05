"""Fig 6A's pure functions (stage 12a_properties), and the two reference-row features other stages share:
PCA-64 (Fig 6A, Fig 2A's 08b row) and the mean of log cells (stage 12's simulation, Fig 2A's 08b row).

Properties per single-cell sample, from its stored cells: depth (mean log10 total counts), mitochondrial fraction (mean
percent_mito / 100) and composition PC1 (first principal component of CLR cell-type fractions at the deconvolution
level, fit on the unit's training samples within the level's stratum). Each is predicted from a representation by ridge
with the read-out's scaling and folds; held-out R² (AUROC for the phenotype). The licence is a pure function of the
split-mean R² table. Nothing here reads a label beyond the single-cell 0/1 group.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from sc2bulk import readout, statistics
from sc2bulk.config import properties as P
from sc2bulk.config import readout as R
from sc2bulk.config.pairs import PAIR_ORDER


def depth(total_counts, sample_id) -> pd.Series:
    """sample -> the mean over its stored cells of log10(total counts)."""
    return pd.Series(np.log10(np.asarray(total_counts, dtype=float))).groupby(np.asarray(sample_id)).mean()


def mito_fraction(percent_mito, sample_id) -> pd.Series:
    """sample -> the mean over its stored cells of percent_mito / 100."""
    return (pd.Series(np.asarray(percent_mito, dtype=float)) / 100).groupby(np.asarray(sample_id)).mean()


def clr_fractions(cell_type, sample_id, sample_ids, kinds) -> pd.DataFrame:
    """samples x kinds: CLR of (count of type k + CLR_PSEUDOCOUNT) / (cells + CLR_PSEUDOCOUNT K), counting only the
    sample's cells of the given kinds."""
    cells = pd.DataFrame({"sample_id": np.asarray(sample_id), "type": np.asarray(cell_type)})
    cells = cells[cells["type"].isin(kinds)]
    counts = pd.crosstab(cells["sample_id"], cells["type"]).reindex(index=list(sample_ids), columns=list(kinds), fill_value=0)
    fraction = (counts + P.CLR_PSEUDOCOUNT).div(counts.sum(axis=1) + P.CLR_PSEUDOCOUNT * len(kinds), axis=0)
    logged = np.log(fraction)
    return logged.sub(logged.mean(axis=1), axis=0)


def composition_pc1(cell_type, sample_id, train_ids, stratum: pd.Series) -> pd.Series:
    """sample -> its composition PC1 score. Within each stratum (stratum: sample -> key; mouse organs: the organ), the
    kinds are the types of the stratum's training samples' cells, PC1 is fit on the training samples' CLR fractions
    (centered on their mean) and every sample of the stratum is projected; PC1's sign makes its loading on the stratum's
    most abundant training type positive (label-free; R² doesn't depend on it)."""
    cell_type, sample_id = np.asarray(cell_type), np.asarray(sample_id)
    out = {}
    for key in sorted(set(stratum)):
        members = sorted(stratum.index[stratum == key])
        train = [s for s in members if s in set(train_ids)]
        in_train = np.isin(sample_id, train)
        abundance = pd.Series(cell_type[in_train]).value_counts()
        kinds = sorted(abundance.index)
        assert len(kinds) >= 2 and len(train) >= 2, f"stratum {key}: composition needs >= 2 types and >= 2 training samples"
        clr = clr_fractions(cell_type, sample_id, members, kinds)
        mean = clr.loc[train].mean(axis=0)
        loading = np.linalg.svd((clr.loc[train] - mean).to_numpy(), full_matrices=False)[2][0]
        top = kinds.index(abundance.sort_index().idxmax())   # ties: the first type in sorted order
        loading = loading if loading[top] > 0 else -loading
        out.update(((clr - mean) @ loading).to_dict())
    return pd.Series(out).sort_index()


def centered(y_fit, y_scored, strata_fit=None, strata_scored=None) -> tuple:
    """Mouse organs: both sides' targets centered within organ with the fit rows' organ means (unchanged without strata)."""
    y_fit, y_scored = np.asarray(y_fit, dtype=float), np.asarray(y_scored, dtype=float)
    if strata_fit is None:
        return y_fit, y_scored
    means = pd.Series(y_fit).groupby(np.asarray(strata_fit)).mean()
    return y_fit - means.loc[np.asarray(strata_fit)].to_numpy(), y_scored - means.loc[np.asarray(strata_scored)].to_numpy()


def ridge_regression(X_fit, y, folds, X_scored, strata_fit=None, strata_scored=None, alphas=None) -> dict:
    """readout.run's ridge branch for a continuous target (readout.run asserts a 0/1 one): per-side scaling on the fit
    rows (scored rows with the fit side's scaler), alpha by grouped CV on the given folds (MSE; ties to the smaller
    alpha), refit on every fit row. Returns alpha, alpha_at_boundary, coef, intercept and the scored rows' predictions."""
    X_fit, y = np.asarray(X_fit, dtype=np.float64), np.asarray(y, dtype=np.float64)
    alphas = tuple(R.RIDGE_ALPHAS) if alphas is None else tuple(alphas)
    assert list(alphas) == sorted(alphas), "the alpha grid must be ascending"
    moments = readout.scaler(X_fit, strata_fit)
    search = GridSearchCV(Ridge(fit_intercept=R.READOUT_FIT_INTERCEPT), {"alpha": list(alphas)}, cv=PredefinedSplit(np.asarray(folds)),
                          scoring="neg_mean_squared_error", error_score="raise").fit(readout.scale(X_fit, moments, strata_fit), y)
    model, alpha = search.best_estimator_, float(search.best_params_["alpha"])
    return {"alpha": alpha, "alpha_at_boundary": bool(alpha in (min(alphas), max(alphas))), "coef": model.coef_,
            "intercept": float(model.intercept_),
            "prediction": model.predict(readout.scale(np.asarray(X_scored, dtype=np.float64), moments, strata_scored))}


def r2(y, prediction) -> np.ndarray:
    """Held-out R² over the last axis (sklearn's r2_score, broadcast): 1 - SS_res / SS_tot about y's own mean."""
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    total = ((y - y.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)
    assert (total > 0).all(), "a held-out target without variance"
    return 1.0 - ((y - prediction) ** 2).sum(axis=-1) / total


def pca64(X_fit, X_scored, strata_fit=None, strata_scored=None) -> tuple:
    """The PCA-64 reference row: both sides standardized with the fit rows' scaler (mouse organs within
    organ), PCA with min(PCA_COMPONENTS, fit rows - 1) components fit on the fit rows only, both sides projected.
    Returns (fit scores, scored scores, components). Where the fit rows are fewer than 65 it is a full-rank rotation
    (whitened again by the read-out's own scaling)."""
    moments = readout.scaler(np.asarray(X_fit, dtype=np.float64), strata_fit)
    Z_fit = readout.scale(np.asarray(X_fit, dtype=np.float64), moments, strata_fit)
    k = min(P.PCA_COMPONENTS, len(Z_fit) - 1)
    pca = PCA(n_components=k, svd_solver="full").fit(Z_fit)
    return pca.transform(Z_fit), pca.transform(readout.scale(np.asarray(X_scored, dtype=np.float64), moments, strata_scored)), k


def mean_log_cells(X, sample_id, sample_ids) -> np.ndarray:
    """The mean-of-log-cells reference row: each sample's mean over its cells of their log-CP10k panel rows (X: cells x
    genes, sparse or dense), in the given sample order."""
    sample_id = np.asarray(sample_id)
    return np.vstack([np.asarray(X[sample_id == s].mean(axis=0), dtype=np.float64).ravel() for s in sample_ids])


def unit_auroc(score, label, scoring_unit) -> float:
    """Fig 2B's single-cell AUROC (stage 11's rule): scores averaged within each scoring unit, then the AUROC over units."""
    table = pd.DataFrame({"score": np.asarray(score, dtype=float), "label": np.asarray(label), "unit": np.asarray(scoring_unit)})
    by_unit = table.groupby("unit").agg(score=("score", "mean"), low=("label", "min"), high=("label", "max"))
    assert (by_unit["low"] == by_unit["high"]).all(), "a scoring unit with mixed labels"
    return statistics.auroc(by_unit["score"].to_numpy(), by_unit["low"].to_numpy())


def within_strata_permutations(strata, n_draws: int, rng) -> np.ndarray:
    """n_draws x n index permutations of n held-out samples, each permuting only within strata (sorted)."""
    strata = np.asarray(strata)
    out = np.tile(np.arange(len(strata)), (n_draws, 1))
    for b in range(n_draws):
        for key in sorted(set(strata)):
            rows = np.flatnonzero(strata == key)
            out[b, rows] = rows[rng.permutation(len(rows))]
    return out


def group_label_permutations(holdout_group, label, n_draws: int, rng) -> np.ndarray:
    """n_draws x n permuted 0/1 labels of n held-out samples: the holdout groups' labels (sorted groups) permuted across
    groups, so a group's samples move together and class counts (in groups) are kept."""
    holdout_group, label = np.asarray(holdout_group), np.asarray(label)
    per_group = pd.Series(label).groupby(holdout_group).agg(["min", "max"])
    assert (per_group["min"] == per_group["max"]).all(), "a holdout group with mixed labels"
    groups, values = per_group.index.to_numpy(), per_group["min"].to_numpy()
    member = np.searchsorted(groups, holdout_group)
    return np.vstack([values[rng.permutation(len(groups))][member] for _ in range(n_draws)])


def null_r2(y, predictions, permutations) -> np.ndarray:
    """(draws,) the mean over read-out seeds of R² between fixed predictions (seeds x n) and the permuted held-out
    target (permutations: draws x n indices)."""
    permuted = np.asarray(y, dtype=float)[permutations]                  # draws x n
    return r2(permuted[:, None, :], np.asarray(predictions)[None, :, :]).mean(axis=1)


def null_auroc(scores, permuted_labels, scoring_unit) -> np.ndarray:
    """(draws,) the mean over read-out seeds of unit_auroc between fixed scores (seeds x n) and permuted labels."""
    return np.array([np.mean([unit_auroc(s, labels, scoring_unit) for s in scores]) for labels in permuted_labels])


def split_mean(table: pd.DataFrame) -> pd.DataFrame:
    """pair, key, property, r2: each row's split-mean value: cv_r2 averaged over read-out seeds, then over
    (fold, encoder seed): encoder splits 0-9, or human blood, aging's 4 folds x 10 seeds (fixed rows: over folds)."""
    per_fit = table.groupby(["pair", "key", "property", "fold", "encoder_seed"])["cv_r2"].mean()
    return per_fit.groupby(["pair", "key", "property"]).mean().rename("r2").reset_index()


def licence(values: pd.DataFrame, present) -> dict:
    """The Fig 6 title licence, from values (pair, key, property, r2: split-mean
    R²) and the present learned configurations (the nine minus CUT_CONFIGURATIONS). A configuration exceeds on a
    technical property in a pair if R²(c) - max(R²(pseudobulk), R²(PCA-64), 0) > EXCEED_TOLERANCE (missing: doesn't);
    the property is won if more than n/2 of the n present exceed; the pair qualifies if every applicable technical
    property (all of them, but mito_fraction in MITO_NA_PAIRS; a missing baseline row fails loudly) is won; the title is licensed if >= LICENCE_MIN_PAIRS of the six pairs
    qualify (a pair without values doesn't). Also reported: per pair and property the count exceeding, and per
    configuration the pairs where it exceeds on every applicable technical property (named if >= LICENCE_MIN_PAIRS)."""
    present, n = list(present), len(present)
    value = values.set_index(["pair", "key", "property"])["r2"]
    assert not value.index.duplicated().any(), "one value per pair, key and property"
    per_pair, qualifying, per_configuration = {}, [], {c: [] for c in present}
    with_values = set(values["pair"])
    for pair in PAIR_ORDER:
        if pair not in with_values:   # a pair without 12a outputs doesn't qualify
            per_pair[pair] = {}
            continue
        # applicable unless n/a by the pre-declared rule (mouse organs: no mitochondrial genes), not by which rows exist
        applicable = [t for t in P.TECHNICAL_PROPERTIES if not (t == "mito_fraction" and pair in P.MITO_NA_PAIRS)]
        exceeding = {}
        for t in applicable:
            assert all((pair, b, t) in value.index for b in P.BASELINES), f"{pair} {t}: a baseline row without a value"
            baselines = [value[(pair, b, t)] for b in P.BASELINES]
            assert np.isfinite(baselines).all(), f"{pair} {t}: a baseline row without a value"
            bar = max(*baselines, 0.0)
            exceeding[t] = {c for c in present if (pair, c, t) in value.index and value[(pair, c, t)] - bar > P.EXCEED_TOLERANCE}
        per_pair[pair] = {t: len(exceeding[t]) for t in applicable}
        if applicable and all(len(exceeding[t]) > n / 2 for t in applicable):
            qualifying.append(pair)
        for c in present:
            if applicable and all(c in exceeding[t] for t in applicable):
                per_configuration[c].append(pair)
    licensed = len(qualifying) >= P.LICENCE_MIN_PAIRS
    return {"n_present": n, "present": present, "per_pair": per_pair, "qualifying_pairs": qualifying, "licensed": licensed,
            "title": P.LICENSED_TITLE if licensed else P.DESCRIPTIVE_TITLE, "per_configuration": per_configuration,
            "named": [c for c in present if len(per_configuration[c]) >= P.LICENCE_MIN_PAIRS]}
