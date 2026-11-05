"""Stages 08b and 11b: the secondary read-outs and statistics. Imported only by 08b, 11b and their modules, so nothing here enters stage 05's,
08's or 11's code hash.

Hashed only into the stages that import it (sc2bulk.codehash).
"""

# ---- 08b: alternative read-outs (S9 Fig, S3 Table) ----
# The hyperparameter is chosen by grouped CV MSE on the primary read-out folds; ties go to the first value in
# the listed order (GridSearchCV's first best), so each grid lists its simplest model first.
ELASTICNET_L1_RATIOS = (0.1, 0.5, 0.9)          # ElasticNetCV with readout.LASSO_ALPHAS, LASSO_MAX_ITER, LASSO_TOL
PLS_COMPONENTS = (1, 2, 3, 5, 10)               # capped below the smallest CV training fold (and the feature count)
KNN_NEIGHBOURS = (15, 11, 7, 5, 3)              # descending: ties go to the smoother model; capped at the smallest CV training fold
MLP_HIDDEN = (64,)
MLP_ALPHAS = (1.0, 1e-2, 1e-4)                  # descending: ties go to the stronger penalty
MLP_MAX_ITER = 2000
ALTERNATIVE_READOUTS = ("elasticnet", "pls", "knn", "mlp")

# ---- 08b: TMM and VST (S11C) ----
# TMM as edgeR 4.0's calcNormFactors(method="TMM") defaults; log-CPM as edgeR's cpm(log=TRUE, prior.count=2)
TMM_EDGER_VERSION = "4.0"
TMM_LOGRATIO_TRIM = 0.3
TMM_SUM_TRIM = 0.05
TMM_A_CUTOFF = -1e10
TMM_PRIOR_COUNT = 2.0
# VST: DESeq2's parametric blind VST, in-house (pydeseq2 is not pinned in requirements.txt)
VST_MIN_DISPERSION = 1e-8
VST_FIT_ITERATIONS = 10

# ---- 08b and 11b: Fig 2A's reference rows ----
# The per-cell classifier: each training sample's cells are bag 0 of the shared n = PER_CELL_CAP test bags (= the
# store's MIN_CELLS_PER_SAMPLE, so every sample gives exactly this many outside the smoke run); its ridge grid is
# PER_CELL_CAP x readout.RIDGE_ALPHAS (each cell weighted 1/PER_CELL_CAP); ridge only
PER_CELL_CAP = 100
# The known-signature score's detection filter: a matched gene is used if its CPM is
# >= SIGNATURE_MIN_CPM in >= SIGNATURE_MIN_FRACTION of the unit's scored bulk (over all of it, not within organ)
SIGNATURE_MIN_CPM = 1.0
SIGNATURE_MIN_FRACTION = 0.10
# The sensitivity variants: (reference key, variant) -> the pairs it is formed for (None: every
# pair the key is formed for). A pair-specific variant is never pooled
REFERENCE_VARIANTS = {("deconvolution_clr", "raw_fractions"): None, ("deconvolution_true", "raw_fractions"): None,
                      ("deconvolution_clr", "no_ifn_markers"): ("lupus",), ("deconvolution_true", "no_ifn_markers"): ("lupus",),
                      ("known_signature", "pooled_z"): ("tms",), ("known_signature", "median_centered"): ("lupus",)}
# The deconvolution review gate: (pair, unit, run) -> SHA-256, filled after the V1-V4 review
# and before the analysis-fixed tag. Run "main": stage 04b's types.csv; "no_ifn" (lupus): the sorted (stratum, type)
# list of fractions_no_ifn.parquet (04b writes no types.csv for that run). 08b asserts them outside the smoke run.
# Pinned by the V1-V4 review, human blood,
# aging's four folds included after its 04b run (2026-10-08); its split67 unit takes no 08b row, so it has no pin.
DECONVOLUTION_TYPES_SHA256 = {
    ("tms", "all_0", "main"): "79e6076262cb23129a3e1da7af8db6b2929dc000b861688708d55f1df9bf9459",
    ("pineda", "all_0", "main"): "d25de59a88f6750a6b5ad8433a74990c8191d6ad4dd19812e71c219ccfdad391",
    ("lupus", "all_0", "main"): "ccc44e2eda4bb971332a70346bdd95ccbfd8b772b26552315db21f8bc16e3be1",
    ("lupus", "all_0", "no_ifn"): "7d6dc1bca27f2a64f7c176f2b0e2548637ae64bb8431c620178534a839aaaa57",
    ("crohns", "all_0", "main"): "c4960f2804cb1610136184a25bfa38d3f290f1c9b0df30a176656b584d700a2b",
    ("soundlife", "crossfit_0", "main"): "69f7cec49671ef2474e9706294e22e26b73c7cb744cfdbf7adc99ccf77b95038",
    ("soundlife", "crossfit_1", "main"): "79bcfcce3d715385c711fec2abb938660ccbd4e433df225b3b102e9d400967e5",
    ("soundlife", "crossfit_2", "main"): "6ae9f8a02225e690f95358b8feb251d61930a92b38d2866c3f440eb6a2a44897",
    ("soundlife", "crossfit_3", "main"): "fd31c0bb4fe41acb56e61597b8928c13053594a7778f61a8562593f3ff2de5b7",
}

# ---- 11b ----
# Stage 11's import-closure files that may differ from the analysis-fixed tag without stopping 11b: the
# triage cuts only. Every difference is written into 11b's run record (the deviation report).
CLOSURE_ALLOWLIST = ("src/sc2bulk/config/triage.py",)
# 11b's own files, which a real run asserts equal the analysis-fixed tag's
SECONDARY_FILES = ("pipeline/11b_secondary.py", "pipeline/08b_secondary.py", "src/sc2bulk/statistics_secondary.py",
                   "src/sc2bulk/readout_secondary.py", "src/sc2bulk/scoring.py", "src/sc2bulk/reference.py",
                   "src/sc2bulk/reading_rule.py", "src/sc2bulk/config/secondary.py")

# Subgroup rows (stream 14): per pair, the column (bulk_samples.csv, or Crohn's bulk
# label column) whose subsets are rows. Sub-stream k is the subset's position: the levels of the column sorted,
# or Crohn's two fixed subsets (inflamed or uninflamed Crohn's, each with every control)
SUBGROUP_COLUMNS = {"hammond": "condition", "tms": "organ", "soundlife": "cmv", "pineda": "project", "crohns": "inflammation"}
CROHNS_SUBSETS = ("inflamed_vs_controls", "uninflamed_vs_controls")
MIN_SUBGROUP_LITTERS = 2                        # Hammond, litter clusters: a subset with fewer per label group is listed and skipped

# The age-adjusted rows (the covariate-adjusted AUROC, Janes & Pepe 2008, for Crohn's and ALS).
# ALS's within-single-cell score: a split needs this many validation controls with distinct ages, else it is not
# estimable
MIN_AGE_CONTROLS = 3

# The rows 11b writes besides ROW_STREAMS' keys
UNSTREAMED_ROWS = ("label_permuted", "summary_rank", "decomposition", "leave_one_pair_out")   # leave_one_pair_out
