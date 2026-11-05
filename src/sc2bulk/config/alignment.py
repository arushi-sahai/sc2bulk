"""Stage 09's alignment settings.

Hashed only into stage 09 (sc2bulk.codehash).
"""

ALIGNED = ("unaligned_stacked", "combat", "coral", "harmony")   # read out by stage 09; "none" is stage 08's primary
MIN_ROWS_PER_SIDE = 2                  # per stratum and side: ComBat needs a within-side variance
CORAL_SHRINKAGE = 1e-3                 # absolute, added to both covariances' diagonals
CORAL_EIGENVALUE_FLOOR = 1e-12         # the matrix powers' eigenvalues are clipped here (the old coral_correct_domains)
# Harmony (harmonypy 2.0.2, "as now"): nclust = round(N / cells_per_cluster) with Python's round, clamped to
# [min_clusters, max_clusters], N the stacked rows of the stratum. The floor is required: harmonypy's own rule
# gives 1 cluster below 45 rows, where it crashes. lambda is left at harmonypy's dynamic estimate (its default).
HARMONY = {"theta": 2.0, "sigma": 0.1, "cells_per_cluster": 30, "min_clusters": 2, "max_clusters": 100}
HARMONY_SEED_MODULUS = 2 ** 31         # harmonypy's C++ binding takes a signed 32-bit seed; seeds.derived gives uint32
SEED_PURPOSE = "alignment"             # Harmony's seed: seeds.derived(SEED_PURPOSE, pair, 0, *unit), one per unit
SILHOUETTE_PCS = 10                    # the modality silhouette on the top principal components (captions Fig 3B)
# Fig 3A / S13A-E coordinates: these configurations at each primary unit of fold 0; trained ones at this
# encoder seed (the first, chosen on no outcome), fixed representations at -1
GEOMETRY_CONFIGURATIONS = ("pseudobulk", "cytoset_cells")
GEOMETRY_ENCODER_SEED = 0
