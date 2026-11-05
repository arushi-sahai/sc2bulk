"""The smoke run: every stage on a tiny subset, to catch wiring bugs in minutes. It is the
dry run before the full run, and the quick check that an installation works.

Selected with `python pipeline/run_all.py --smoke` (sets SC2BULK_SMOKE=1). Its outputs go
to <outputs>/smoke/, never mixed with the full run's.

Only stages 02-03's smoke values are here (this file is hashed into them and is frozen with
them); later stages keep their smoke values beside their settings (e.g. representations.SMOKE).
"""

HOLDOUT_GROUPS_PER_LABEL = 8            # holdout groups (sample, mouse or donor) per 0/1 group, with all their samples:
                                       # enough for 2 validation groups per label and 4 cross-fitting folds
BULK_SAMPLES_PER_GROUP = 6
N_ENCODER_SEEDS = 1
N_READOUT_SEEDS = 1
N_SEARCH_SEEDS = 1
