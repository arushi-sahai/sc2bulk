"""Fig 6A's sample properties and the Fig 6 title licence (stage 12a_properties).

Hashed only into the stages that import it (sc2bulk.codehash), so its own file: config/training.py and
config/statistics.py feed other stages' hashes.
"""
from sc2bulk.config import training as T

PROPERTIES = ("depth", "mito_fraction", "composition_pc1", "phenotype")   # Fig 6A's columns, in order
CONTINUOUS = ("depth", "mito_fraction", "composition_pc1")                # cross-validated R²; phenotype: AUROC
MITO_NA_PAIRS = ("tms",)               # no mitochondrial genes in the store: every percent_mito is 0 (asserted), so n/a
CLR_PSEUDOCOUNT = 0.5                  # per sample: (count of type k + 0.5) / (cells + 0.5 K), then CLR
PCA_COMPONENTS = 64                    # the PCA-64 reference row: min(64, fit rows - 1) components

# The chance level: predictions fixed, held-out targets permuted, N_NULL draws per unit. Continuous properties
# from `permutation` at PROPERTY_NULL_OFFSET, the phenotype at PROPERTY_NULL_OFFSET + 1 (unit-keyed): clear of the
# blinded run's index (< PERMUTATION_OFFSET, asserted in config/training.py) and stage 06 row D's
# PERMUTATION_OFFSET + j, j < N_PERMUTED
N_NULL = 200
NULL_QUANTILE = 0.95
PROPERTY_NULL_OFFSET = 2000
assert PROPERTY_NULL_OFFSET >= T.PERMUTATION_OFFSET + T.N_PERMUTED

# The Fig 6 title licence, mechanical. A learned configuration exceeds on a
# technical property in a pair if R²(c) - max(R²(pseudobulk), R²(PCA-64), 0) > EXCEED_TOLERANCE (split-mean R², raw;
# missing = not exceeding); the property is won if more than half of the present learned configurations exceed; a
# pair qualifies if every applicable technical property is won; the title is licensed if >= LICENCE_MIN_PAIRS of the
# six pairs qualify.
TECHNICAL_PROPERTIES = ("depth", "mito_fraction")
EXCEED_TOLERANCE = 1e-12
LICENCE_MIN_PAIRS = 4
BASELINES = ("pseudobulk", "pseudobulk_pca64")
LICENSED_TITLE = "Technical sample properties are more predictable from learned representations than from pseudobulk"
DESCRIPTIVE_TITLE = "Sample properties predictable from each representation"
