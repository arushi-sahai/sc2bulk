"""Stage 08b's numerics: label-free,
pure functions, tested in tests/test_readout_secondary.py. sc2bulk.readout itself is used unchanged.

- apply_head: a stored primary head applied to other bulk rows with their own bulk scaler (H3's thinned bulk,
  scVI's observed library size, the outside-moments row). The ridge or LASSO fit sees only single-cell rows, so
  this equals a refit exactly.
- halves / outside_moments: two label-free halves of the scored bulk's scoring units (mouse organs within organ),
  each half scaled with the other half's moments.
- alternative: the S9 Fig read-outs (elastic net, PLS, k-NN, MLP), with readout.run's per-side scaling and
  grouped folds; ridge_continuous: ridge on a continuous target (the numeric-age read-out).
- tmm_log_cpm, vst: TMM log-CPM (edgeR's defaults) and DESeq2's parametric blind VST, per side.
- bag_pseudobulk: Fig 4A's pseudobulk-n line (log-CP10k per bag, then the mean over bags).

Fig 2A's reference rows:
- cell_ridge: the per-cell classifier's ridge, readout.run's model and selection rule solved in closed form (Gram
  matrices, one eigh per CV training set), on already-scaled cells.
- fraction_features, type_list_sha256: deconvolution fractions as features ("<stratum>:<type>" blocks, zero outside
  the sample's stratum; CLR or the closed fractions) and the review gate's hash of a type list.
- match_signature, signature_detected, signature_used, signature_score, median_centered: the known-signature score. Scoring stays out of sc2bulk.signatures, which is hashed into stage 04b.
"""
import hashlib

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sc2bulk import deconvolution, preprocess
from sc2bulk.readout import scale, scaler
from sc2bulk.config import readout as R
from sc2bulk.config import secondary as C


def apply_head(X, coef, intercept, strata=None, moments=None) -> np.ndarray:
    """Scores of bulk rows X under a head (coef, intercept), standardized with the given moments (default:
    X's own, per stratum, as readout.run scales its bulk side)."""
    X = np.asarray(X, dtype=np.float64)
    moments = scaler(X, strata) if moments is None else moments
    return scale(X, moments, strata) @ np.asarray(coef, dtype=np.float64) + float(intercept)


def halves(units, strata, seed) -> dict:
    """scoring unit -> half (0 or 1): within each stratum, the sorted units in a seeded random order, the first
    half (rounded down) to half 0. Label-free; each half is asserted non-empty in every stratum."""
    units, strata = np.asarray(units), np.asarray(strata)
    rng = np.random.default_rng(seed)
    out = {}
    for s in sorted(set(strata)):
        members = np.array(sorted(set(units[strata == s])))
        assert len(members) >= 2, f"stratum {s}: fewer than 2 scoring units to halve"
        order = members[rng.permutation(len(members))]
        out.update({u: int(i >= len(members) // 2) for i, u in enumerate(order)})
    return out


def outside_moments(X, half, strata=None) -> np.ndarray:
    """X standardized row-wise with the moments of the other half's rows (per stratum)."""
    X, half = np.asarray(X, dtype=np.float64), np.asarray(half)
    strata = np.full(len(X), "all") if strata is None else np.asarray(strata)
    out = np.empty_like(X)
    for h in (0, 1):
        rows, other = half == h, half != h
        assert set(strata[rows]) <= set(strata[other]), "a stratum without rows in the other half"
        out[rows] = scale(X[rows], scaler(X[other], strata[other]), strata[rows])
    return out


def smallest_training_fold(folds) -> int:
    folds = np.asarray(folds)
    return int(min((folds != f).sum() for f in np.unique(folds)))


def alternative(X_train, y, folds, X_bulk, name, strata_train=None, strata_bulk=None, random_state=None) -> dict:
    """One S9 Fig read-out of the 0/1 group: per-side scaling as readout.run, the hyperparameter by grouped CV
    MSE on the given folds (ties to the first value listed in config.secondary). Returns the bulk scores, the
    chosen hyperparameters, and for the elastic net the head (coef, intercept, alpha, convergence) and scalers."""
    X_train, y = np.asarray(X_train, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert set(np.unique(y)) == {0.0, 1.0}
    single_cell = scaler(X_train, strata_train)
    Z = scale(X_train, single_cell, strata_train)
    split = PredefinedSplit(np.asarray(folds))
    largest = smallest_training_fold(folds)
    out = {"scalers": {"single_cell": single_cell}}
    if name == "elasticnet":
        model = ElasticNetCV(l1_ratio=list(C.ELASTICNET_L1_RATIOS), alphas=list(R.LASSO_ALPHAS), cv=split, max_iter=R.LASSO_MAX_ITER,
                             tol=R.LASSO_TOL, fit_intercept=R.READOUT_FIT_INTERCEPT).fit(Z, y)
        out.update(alpha=float(model.alpha_), alpha_at_boundary=bool(model.alpha_ in (min(R.LASSO_ALPHAS), max(R.LASSO_ALPHAS))),
                   coef=model.coef_, intercept=float(model.intercept_), converged=bool(model.n_iter_ < R.LASSO_MAX_ITER),
                   n_iter=int(model.n_iter_), n_nonzero=int((model.coef_ != 0).sum()), hyper={"l1_ratio": float(model.l1_ratio_)})
    else:
        if name == "pls":
            estimator, grid = PLSRegression(scale=False), {"n_components": [c for c in C.PLS_COMPONENTS if c < largest and c <= Z.shape[1]]}
        elif name == "knn":
            estimator, grid = KNeighborsRegressor(weights="uniform"), {"n_neighbors": [k for k in C.KNN_NEIGHBOURS if k <= largest]}
        else:
            estimator = MLPRegressor(hidden_layer_sizes=C.MLP_HIDDEN, max_iter=C.MLP_MAX_ITER, early_stopping=False, random_state=random_state)
            grid = {"alpha": list(C.MLP_ALPHAS)}
        assert all(grid.values()), f"{name}: no hyperparameter value fits the folds"
        model = GridSearchCV(estimator, grid, cv=split, scoring="neg_mean_squared_error", error_score="raise").fit(Z, y).best_estimator_
        out["hyper"] = {k: getattr(model, k) for k in grid}
    bulk = np.asarray(X_bulk, dtype=np.float64)
    out["scalers"]["bulk"] = scaler(bulk, strata_bulk)
    out["score"] = np.asarray(model.predict(scale(bulk, out["scalers"]["bulk"], strata_bulk)), dtype=np.float64).ravel()
    return out


def ridge_continuous(X_train, y, folds, X_bulk, strata_train=None, strata_bulk=None) -> dict:
    """readout.run's ridge for a continuous target (the numeric-age read-out): the same per-side
    scaling, grid and grouped CV MSE (ties to the smaller alpha)."""
    X_train, y = np.asarray(X_train, dtype=np.float64), np.asarray(y, dtype=np.float64)
    single_cell = scaler(X_train, strata_train)
    alphas = list(R.RIDGE_ALPHAS)
    search = GridSearchCV(Ridge(fit_intercept=R.READOUT_FIT_INTERCEPT), {"alpha": alphas}, cv=PredefinedSplit(np.asarray(folds)),
                          scoring="neg_mean_squared_error", error_score="raise").fit(scale(X_train, single_cell, strata_train), y)
    model, alpha = search.best_estimator_, search.best_params_["alpha"]
    bulk = np.asarray(X_bulk, dtype=np.float64)
    moments = scaler(bulk, strata_bulk)
    return {"alpha": float(alpha), "alpha_at_boundary": bool(alpha in (alphas[0], alphas[-1])), "coef": model.coef_,
            "intercept": float(model.intercept_), "scalers": {"single_cell": single_cell, "bulk": moments},
            "score": model.predict(scale(bulk, moments, strata_bulk))}


# ---- TMM (edgeR calcNormFactors, method "TMM", at its defaults) ----

def tmm_pair(obs, ref, n_obs, n_ref) -> float:
    """edgeR's .calcFactorTMM: the weighted trimmed mean of the log ratios of obs against ref (genes with a
    zero in either are excluded), with asymptotic precision weights."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_r = np.log2((obs / n_obs) / (ref / n_ref))
        abs_e = (np.log2(obs / n_obs) + np.log2(ref / n_ref)) / 2
        v = (n_obs - obs) / n_obs / obs + (n_ref - ref) / n_ref / ref
    keep = np.isfinite(log_r) & np.isfinite(abs_e) & (abs_e > C.TMM_A_CUTOFF)
    log_r, abs_e, v = log_r[keep], abs_e[keep], v[keep]
    if np.max(np.abs(log_r)) < 1e-6:
        return 1.0
    n = len(log_r)
    lo_l, lo_s = np.floor(n * C.TMM_LOGRATIO_TRIM) + 1, np.floor(n * C.TMM_SUM_TRIM) + 1
    rank_r, rank_e = stats.rankdata(log_r), stats.rankdata(abs_e)
    kept = (rank_r >= lo_l) & (rank_r <= n + 1 - lo_l) & (rank_e >= lo_s) & (rank_e <= n + 1 - lo_s)
    return float(2 ** (np.sum(log_r[kept] / v[kept]) / np.sum(1 / v[kept])))


def tmm_factors(counts) -> np.ndarray:
    """Per sample (rows of counts, samples x genes) TMM normalization factors, rescaled to geometric mean 1.
    The reference is the sample whose upper-quartile factor is closest to their mean (edgeR: the largest sum of
    square-root counts if the median upper-quartile factor is below 1e-20)."""
    x = np.asarray(counts, dtype=np.float64)
    x = x[:, x.sum(axis=0) > 0]
    library = x.sum(axis=1)
    f75 = np.quantile(x / library[:, None], 0.75, axis=1)
    ref = int(np.argmax(np.sqrt(x).sum(axis=1))) if np.median(f75) < 1e-20 else int(np.argmin(np.abs(f75 - f75.mean())))
    f = np.array([tmm_pair(x[i], x[ref], library[i], library[ref]) for i in range(len(x))])
    return f / np.exp(np.mean(np.log(f)))


def tmm_log_cpm(counts) -> np.ndarray:
    """log2 CPM with TMM effective library sizes, as edgeR's cpm(log=TRUE, prior.count=2): the prior scaled by
    each library's size over the mean, and twice the scaled prior added to the library."""
    x = np.asarray(counts, dtype=np.float64)
    library = x.sum(axis=1) * tmm_factors(x)
    prior = library / library.mean() * C.TMM_PRIOR_COUNT
    return np.log2((x + prior[:, None]) / (library + 2 * prior)[:, None] * 1e6)


# ---- DESeq2's parametric blind VST ----

def size_factors(counts) -> np.ndarray:
    """DESeq2's median-of-ratios size factors (genes with a zero in any sample left out)."""
    x = np.asarray(counts, dtype=np.float64)
    positive = (x > 0).all(axis=0)
    assert positive.any(), "no gene is counted in every sample"
    log_geo = np.log(x[:, positive]).mean(axis=0)
    return np.exp(np.median(np.log(x[:, positive]) - log_geo, axis=1))


def gamma_identity_fit(y, x, start) -> np.ndarray:
    """Coefficients of y ~ b0 + b1 x by a Gamma GLM with the identity link (IRLS: weights 1 / mu^2)."""
    design = np.column_stack([np.ones_like(x), x])
    beta = np.asarray(start, dtype=np.float64)
    for _ in range(100):
        mu = design @ beta
        w = 1 / mu ** 2
        new = np.linalg.solve(design.T @ (design * w[:, None]), design.T @ (w * y))
        if np.max(np.abs(new - beta) / np.abs(beta)) < 1e-10:
            return new
        beta = new
    raise AssertionError("the Gamma dispersion fit did not converge")


def vst(counts) -> np.ndarray:
    """DESeq2's blind parametric VST of counts (samples x genes): size factors, moment estimates of the gene-wise
    dispersions (DESeq2's momentsDispEstimate; not its Cox-Reid MLE), the parametric trend asymptDisp +
    extraPois / mean (its parametricDispersionFit loop, outlier genes dropped, on genes with dispersion above
    100 x the minimum), then the closed-form transform of the normalized counts, in log2 units."""
    x = np.asarray(counts, dtype=np.float64)
    sf = size_factors(x)
    q = x / sf[:, None]
    mean, var = q.mean(axis=0), q.var(axis=0, ddof=1)
    used = mean > 0
    disp = np.maximum((var[used] - np.mean(1 / sf) * mean[used]) / mean[used] ** 2, C.VST_MIN_DISPERSION)
    fit_on = disp > 100 * C.VST_MIN_DISPERSION
    means, disps = mean[used][fit_on], disp[fit_on]
    coefs = np.array([0.1, 1.0])
    for iteration in range(C.VST_FIT_ITERATIONS + 1):
        residual = disps / (coefs[0] + coefs[1] / means)
        good = (residual > 1e-4) & (residual < 15)
        old, coefs = coefs, gamma_identity_fit(disps[good], 1 / means[good], coefs)
        assert (coefs > 0).all(), "the parametric dispersion trend has a non-positive coefficient"
        if np.sum(np.log(coefs / old) ** 2) < 1e-6:
            break
        assert iteration < C.VST_FIT_ITERATIONS, "the dispersion trend fit did not converge"
    a, extra = coefs
    return np.log((1 + extra + 2 * a * q + 2 * np.sqrt(a * q * (1 + extra + a * q))) / (4 * a)) / np.log(2)


# ---- Fig 4A's pseudobulk-n ----

def bag_pseudobulk(counts, bags, columns) -> np.ndarray:
    """One sample's pseudobulk-n feature: per bag (row indices into counts, cells x shared genes), its summed
    counts' log-CP10k over the shared genes on the panel columns; then the mean over bags."""
    profiles = [preprocess.log_normalize(preprocess.summed(counts[b])[None, :])[:, columns].toarray().ravel() for b in bags]
    return np.mean(profiles, axis=0)


# ---- Fig 2A's reference rows ----

def ridge_path(C, c, alphas) -> np.ndarray:
    """features x alphas: the ridge coefficients (C + alpha I)^-1 c at every alpha, from one eigendecomposition of the
    centered Gram C."""
    values, vectors = np.linalg.eigh(C)
    return vectors @ ((vectors.T @ c)[:, None] / (values[:, None] + np.asarray(alphas)[None, :]))


def cell_ridge(Z_cells, y_cells, folds_cells, Z_bulk, alphas) -> dict:
    """The per-cell classifier's ridge: readout.run's ridge on already-scaled cells, solved in closed form.
    Per CV fold, the training cells' Gram is the total minus the fold's, centered analytically (the unpenalized
    intercept); the fold's held-out MSE at every alpha comes from one eigh. alpha = the smallest minimum of the
    equal-weight mean over folds (GridSearchCV's rule on an ascending grid); then the refit on every cell. Returns
    readout.run's keys but scalers (the caller's), and cv_mse (the fold-mean curve)."""
    X, y, folds = np.asarray(Z_cells, dtype=np.float64), np.asarray(y_cells, dtype=np.float64), np.asarray(folds_cells)
    alphas = np.asarray(alphas, dtype=np.float64)
    assert set(np.unique(y)) == {0.0, 1.0}
    assert list(alphas) == sorted(alphas), "the alpha grid must be ascending"
    gram, cross, total, y_total = X.T @ X, X.T @ y, X.sum(axis=0), y.sum()
    mse = []
    for f in np.unique(folds):
        held = folds == f
        X_held, y_held = X[held], y[held]
        n = len(y) - held.sum()
        mean, y_mean = (total - X_held.sum(axis=0)) / n, (y_total - y_held.sum()) / n
        C = gram - X_held.T @ X_held - n * np.outer(mean, mean)
        c = cross - X_held.T @ y_held - n * mean * y_mean
        prediction = (X_held - mean) @ ridge_path(C, c, alphas) + y_mean
        mse.append(((prediction - y_held[:, None]) ** 2).mean(axis=0))
    cv_mse = np.mean(mse, axis=0)
    best = int(np.argmin(cv_mse))
    alpha = float(alphas[best])
    mean, y_mean = total / len(y), y_total / len(y)
    coef = np.linalg.solve(gram - len(y) * np.outer(mean, mean) + alpha * np.eye(len(mean)), cross - len(y) * mean * y_mean)
    intercept = float(y_mean - mean @ coef)
    score = np.asarray(Z_bulk, dtype=np.float64) @ coef + intercept
    return {"alpha": alpha, "alpha_at_boundary": best in (0, len(alphas) - 1), "coef": coef, "intercept": intercept,
            "score": score, "probability": np.clip(score, 0.0, 1.0), "cv_mse": cv_mse}


def fraction_features(table: pd.DataFrame, sample_ids, transform: str) -> tuple:
    """(samples x features, feature names) from long fractions (sample_id, stratum, type, fraction) closed over each
    stratum's feature types: per stratum, the stratum's samples' fractions under deconvolution.clr ("clr") or as they
    are ("raw"), in columns "<stratum>:<type>" (sorted), zero outside the sample's stratum (mouse organs' blocks)."""
    wide = table.pivot_table(index="sample_id", columns=["stratum", "type"], values="fraction", aggfunc="first")
    stratum = table.groupby("sample_id")["stratum"].agg(["nunique", "first"])
    assert (stratum["nunique"] == 1).all(), "a sample in two strata"
    assert set(sample_ids) <= set(wide.index), f"samples without fractions: {sorted(set(sample_ids) - set(wide.index))[:3]}"
    blocks = []
    for s in sorted(table["stratum"].unique()):
        block = wide[s].loc[stratum.index[stratum["first"] == s]].sort_index(axis=1)
        assert block.notna().all().all(), f"stratum {s}: a sample without a fraction of every type"
        block = deconvolution.clr(block) if transform == "clr" else block
        blocks.append(block.set_axis([f"{s}:{t}" for t in block.columns], axis=1))
    out = pd.concat(blocks, axis=1).fillna(0.0)
    return out.loc[list(sample_ids)].to_numpy(dtype=np.float64), list(out.columns)


def type_list_sha256(table: pd.DataFrame) -> str:
    """SHA-256 of a fractions table's sorted (stratum, type) list, one tab-separated pair per line."""
    pairs = sorted(set(zip(table["stratum"].astype(str), table["type"].astype(str))))
    return hashlib.sha256("".join(f"{s}\t{t}\n" for s, t in pairs).encode()).hexdigest()


def match_signature(listed: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    """The bulk genes a signature matches: keys maps each bulk gene (index) to the symbol it is matched on (its own,
    uppercased, or its one-to-one human ortholog's); listed has gene and direction. Returns gene (bulk), symbol,
    direction, in bulk order."""
    direction = listed.set_index("gene")["direction"]
    matched = keys[keys.isin(direction.index)]
    return pd.DataFrame({"gene": matched.index, "symbol": matched.to_numpy(), "direction": direction.loc[matched.to_numpy()].to_numpy()})


def signature_detected(cpm) -> np.ndarray:
    """Per matched gene, the detection filter: CPM >= SIGNATURE_MIN_CPM in >= SIGNATURE_MIN_FRACTION of the scored
    bulk (all of it, not within organ)."""
    return (np.asarray(cpm, dtype=np.float64) >= C.SIGNATURE_MIN_CPM).mean(axis=0) >= C.SIGNATURE_MIN_FRACTION


def signature_used(cpm) -> np.ndarray:
    """Per matched gene, whether the score uses it: detected and not constant over the scored bulk."""
    return signature_detected(cpm) & (np.ptp(np.asarray(cpm, dtype=np.float64), axis=0) > 0)


def signature_score(cpm, up, strata=None) -> np.ndarray:
    """Each bulk sample's mean z of the up-genes minus that of the down-genes (an up-only list: the up term), z of
    log1p CPM with readout's scaler over the scored bulk (per stratum when given; a gene constant within a stratum
    gets z = 0 there)."""
    L = np.log1p(np.asarray(cpm, dtype=np.float64))
    up = np.asarray(up, dtype=bool)
    assert up.any(), "no up-regulated signature gene is used (the score would be NaN)"
    z = scale(L, scaler(L, strata), strata)
    return z[:, up].mean(axis=1) - (z[:, ~up].mean(axis=1) if (~up).any() else 0.0)


def median_centered(cpm) -> np.ndarray:
    """Yao's median form (lupus sensitivity): per bulk sample, the median over genes of log1p CPM
    minus the gene's mean over the scored bulk."""
    L = np.log1p(np.asarray(cpm, dtype=np.float64))
    assert L.shape[1] > 0, "no signature gene is used (the score would be NaN)"
    return np.median(L - L.mean(axis=0), axis=1)
