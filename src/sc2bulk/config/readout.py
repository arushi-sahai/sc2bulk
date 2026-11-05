"""Read-outs (ridge primary, LASSO secondary), on the 0/1 group (stage 08 and later).

Hashed only into the stages that import it (sc2bulk.codehash).
"""

READOUT_FIT_INTERCEPT = True
SCALER_DDOF = 0                        # per-side StandardScaler; read-out features are not capped
# The ridge grid: 60 points 10^(-3 + 7i/59), 1e-3 .. 1e4. An edge is extended
# once, by RIDGE_EXTENSION_STEPS more points at the same log spacing (26 x 7/59 = 3.08 decades, the first whole
# number of steps covering 3), keeping the 60. RIDGE_ALPHA_EXTENSION, a subset of ("lower", "upper"), is set
# only as pipeline/08a_alpha_edges.py's decision requires: the one pre-declared post-deposit edit.
RIDGE_ALPHA_EXTENSION = ("lower",)   # 08a, 2026-10-22: pseudobulk at the lower edge in 53 of 90 fits (0.59 > 0.10)
RIDGE_EXTENSION_STEPS = 26
ALPHA_EDGE_FRACTION = 0.10             # an edge is extended if more than this fraction of a configuration's fits sit on it


def ridge_alphas(extension: tuple) -> tuple:
    """The ridge grid with the given edges extended."""
    low = -RIDGE_EXTENSION_STEPS if "lower" in extension else 0
    high = 60 + (RIDGE_EXTENSION_STEPS if "upper" in extension else 0)
    return tuple(float(10.0 ** (-3 + 7 * i / 59)) for i in range(low, high))


RIDGE_ALPHAS = ridge_alphas(RIDGE_ALPHA_EXTENSION)
LASSO_ALPHAS = tuple(float(a) for a in (10.0 ** (-4 + 6 * i / 99) for i in range(100)))    # 1e-4 .. 1e2
LASSO_MAX_ITER = 200_000
LASSO_TOL = 1e-4
OUT_OF_FOLD_MIN_GROUPS = 2             # the guard: holdout groups per label on split e's validation side (and READOUT_FOLDS in total)
