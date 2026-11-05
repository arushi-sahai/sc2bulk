"""Encoder inputs and how sample embeddings are drawn (stages 05-07b and later).

Hashed only into the stages that import it (sc2bulk.codehash).
"""

INPUT_CAP = 10.0                       # standardized inputs capped at +-10 (population, per-sample)
TEST_TIME_BAGS = 10                    # a sample's embedding = mean over this many random bags
BAG_SIZES = (1, 25, 50, 100, 200, 300, 500, 600, 1000)   # Fig 4A: CytoSet's ncell candidates, and the FMs' mean of n cells
FM_CELLS_PER_SAMPLE = 1000             # covers the largest Fig 4A bag size
ALL_AVAILABLE_CAP = 3000               # Fig 4A "up to 3,000 cells" store (stage 02b): post-QC cells per sample

# the smoke run's values (settings.SMOKE)
SMOKE = {"TEST_TIME_BAGS": 2, "FM_CELLS_PER_SAMPLE": 20, "N_EPOCHS": 2, "NSUBSET": 16,
         "SEARCH_GROUPS": ("out_pool",),          # one group, so two candidates (base and out_max) ...
         "SEARCH_CANDIDATES": ("out_max",)}       # ... are trained
