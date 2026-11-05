"""Stage 09's alignment: one code path for every configuration.

ComBat, CORAL and Harmony are each fit on the stacked training single-cell and scored bulk rows of a
stratum (mouse organs: within organ), on the representation's own features (no PCA, no prior scaling),
with side as the only batch variable and no covariate: no label on either side. "unaligned_stacked"
returns its inputs, so the read-out's stacked scaler is the only change from "none".
- ComBat: scanpy.pp.combat with batch = side, the pooled form (both sides move to a common mean and
  scale).
- CORAL: the single-cell rows are recolored to the bulk side's covariance and mean, the bulk rows left
  unchanged: X_sc' = (X_sc - mean_sc) (C_sc + lambda I)^(-1/2) (C_bulk + lambda I)^(1/2) + mean_bulk.
- Harmony: harmonypy.run_harmony, clusters floored at 2, its dynamic lambda, seeded per unit.

The modality silhouette and the Fig 3A coordinates are computed on the matrix the read-out sees, and
see no label either.
"""
import anndata
import harmonypy
import numpy as np
import pandas as pd
import scanpy
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

from sc2bulk import readout
from sc2bulk.config import alignment as A


def strata_or_all(strata, n) -> np.ndarray:
    return np.full(n, "all") if strata is None else np.asarray(strata)


def sides(n_sc, n_bulk) -> np.ndarray:
    return np.array(["single_cell"] * n_sc + ["bulk"] * n_bulk)


def combat(X_sc, X_bulk) -> np.ndarray:
    stacked = np.vstack([X_sc, X_bulk])
    data = anndata.AnnData(X=stacked.copy(), obs=pd.DataFrame({"side": sides(len(X_sc), len(X_bulk))},
                                                              index=[str(i) for i in range(len(stacked))]))
    scanpy.pp.combat(data, key="side")
    return np.asarray(data.X, dtype=np.float64)


def matrix_power(C, power) -> np.ndarray:
    """C ** power for a symmetric positive semi-definite C (eigenvalues clipped at the floor)."""
    w, v = np.linalg.eigh(C)
    return (v * np.clip(w, A.CORAL_EIGENVALUE_FLOOR, None) ** power) @ v.T


def coral(X_sc, X_bulk) -> np.ndarray:
    shrink = A.CORAL_SHRINKAGE * np.eye(X_sc.shape[1])
    transform = matrix_power(np.cov(X_sc, rowvar=False) + shrink, -0.5) @ matrix_power(np.cov(X_bulk, rowvar=False) + shrink, 0.5)
    return np.vstack([(X_sc - X_sc.mean(axis=0)) @ transform + X_bulk.mean(axis=0), X_bulk])


def n_clusters(n) -> int:
    return min(max(round(n / A.HARMONY["cells_per_cluster"]), A.HARMONY["min_clusters"]), A.HARMONY["max_clusters"])


def harmony(X_sc, X_bulk, seed) -> np.ndarray:
    assert seed is not None, "Harmony needs the unit's seed (harmonypy would silently use 0)"
    stacked = np.vstack([X_sc, X_bulk])
    side = pd.DataFrame({"side": sides(len(X_sc), len(X_bulk))})
    # passed d x n: harmonypy guesses the orientation and would misread a square n x d matrix
    out = harmonypy.run_harmony(stacked.T, side, ["side"], theta=float(A.HARMONY["theta"]), sigma=float(A.HARMONY["sigma"]),
                                nclust=n_clusters(len(stacked)), random_state=seed, verbose=False)
    Z = np.asarray(out.Z_corr, dtype=np.float64)
    assert Z.shape == stacked.shape, Z.shape
    return Z


def align(method, X_sc, X_bulk, strata_sc=None, strata_bulk=None, seed=None) -> tuple:
    """(X_sc', X_bulk'): the method fit within each stratum on the stacked rows. Every stratum has at least
    MIN_ROWS_PER_SIDE rows on each side (asserted, for every method)."""
    X_sc, X_bulk = np.asarray(X_sc, dtype=np.float64), np.asarray(X_bulk, dtype=np.float64)
    assert (strata_sc is None) == (strata_bulk is None), "strata on both sides or on neither"
    strata_sc, strata_bulk = strata_or_all(strata_sc, len(X_sc)), strata_or_all(strata_bulk, len(X_bulk))
    for s in dict.fromkeys([*strata_sc, *strata_bulk]):
        assert min((strata_sc == s).sum(), (strata_bulk == s).sum()) >= A.MIN_ROWS_PER_SIDE, f"stratum {s}: too few rows on a side"
    if method == "unaligned_stacked":
        return X_sc, X_bulk
    out_sc, out_bulk = np.empty_like(X_sc), np.empty_like(X_bulk)
    for s in dict.fromkeys(strata_sc):
        sc, bulk = strata_sc == s, strata_bulk == s
        if method == "combat":
            Z = combat(X_sc[sc], X_bulk[bulk])
        elif method == "coral":
            Z = coral(X_sc[sc], X_bulk[bulk])
        else:
            assert method == "harmony", method
            Z = harmony(X_sc[sc], X_bulk[bulk], seed)
        out_sc[sc], out_bulk[bulk] = Z[:sc.sum()], Z[sc.sum():]
    assert np.isfinite(out_sc).all() and np.isfinite(out_bulk).all(), f"{method}: non-finite values"
    return out_sc, out_bulk


def strata_stacked(strata_sc, strata_bulk):
    return None if strata_sc is None else np.concatenate([strata_sc, strata_bulk])


def stacked_scaled(X_sc, X_bulk, strata_sc=None, strata_bulk=None) -> np.ndarray:
    """The stacked rows as readout.run(..., scaling="stacked") scales them: one scaler per stratum over both sides."""
    stacked, strata = np.vstack([X_sc, X_bulk]).astype(np.float64), strata_stacked(strata_sc, strata_bulk)
    return readout.scale(stacked, readout.scaler(stacked, strata), strata)


def per_side_scaled(X_sc, X_bulk, strata_sc=None, strata_bulk=None) -> np.ndarray:
    """The stacked rows as readout.run's per-side scaling scales them ("none")."""
    return np.vstack([readout.scale(X, readout.scaler(X, s), s)
                      for X, s in ((np.asarray(X_sc, dtype=np.float64), strata_sc), (np.asarray(X_bulk, dtype=np.float64), strata_bulk))])


def components(Z, k) -> np.ndarray:
    """The first min(k, n - 1, d) principal-component scores of Z's rows (centered; the full, deterministic SVD)."""
    return PCA(n_components=min(k, len(Z) - 1, Z.shape[1]), svd_solver="full").fit_transform(Z)


def silhouette(Z, side, strata=None) -> float:
    """The modality silhouette: within each stratum, the side label's silhouette (Euclidean) on the top
    SILHOUETTE_PCS components; the mean over all rows of their within-stratum values (0 mixed, 1 separate)."""
    strata = strata_or_all(strata, len(Z))
    values = np.empty(len(Z))
    for s in dict.fromkeys(strata):
        rows = strata == s
        values[rows] = silhouette_samples(components(Z[rows], A.SILHOUETTE_PCS), side[rows])
    return float(values.mean())


def coordinates(Z) -> np.ndarray:
    """PC1 and PC2 of all of Z's rows (one PCA, mouse organs included): Fig 3A / S13A-E."""
    return components(Z, 2)
