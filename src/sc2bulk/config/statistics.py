"""Statistics (stage 11).

Hashed only into the stages that import it (sc2bulk.codehash).
"""

BOOTSTRAP_RESAMPLES = 2000            # 0.025 x and 0.05 x resamples are integers (statistics.lower/upper)
CI_LEVEL = 0.95
SMOKE_BOOTSTRAP_RESAMPLES = 40         # the smoke run's (settings.SMOKE); 0.025 x 40 = 1
MARGIN = 0.05                          # X, pre-declared
ALPHA = 0.05                           # one-sided tests; Holm within each family
# Mouse microglia's bootstrap scheme: "litters" if each label group has >= 5 litters,
# else "samples"; pinned before the analysis was fixed, from the blinded run's litter counts
HAMMOND_SCHEME = "litters"              # blinded run 2026-10-06: 9 and 10 litters per label group
# Each row kind's bootstrap stream: seeds.derived("bootstrap", pair, ROW_STREAMS[kind], fold_set, fold),
# so adding a row never shifts another's draws
# The a priori MDD table's inputs: each pair's scoring units per label (n0, n1) per fold,
# from the blinded run (permutation keeps the counts), pinned before the analysis was fixed; the encoder-seed SD
# planning value (the AUROC SD across encoder seeds; reviewer 1's planning value, stated) and the seeds
# blinded run 2026-10-06 (scripts/blinded_counts.py): totals exact; human blood, aging's split across its four
# folds is the permuted one, not the real near-balanced split (slightly conservative MDD; disclosed)
CLASS_COUNTS = {"hammond": [(25, 13)], "tms": [(69, 76)], "soundlife": [(15, 9), (14, 10), (12, 12), (8, 16)],
                "lupus": [(18, 99)], "crohns": [(113, 314)], "pineda": [(30, 140)]}
SEED_SD = 0.03
ROW_STREAMS = {"primary": 0, "logit": 1, "passing_seeds": 2, "lasso": 3,
               # stage 11b's rows, reserved before the analysis was fixed
               "out_of_fold": 4, "thinned_bulk": 5, "scvi_observed_library": 6, "numeric_age": 7, "alignment_combat": 8,
               "alignment_coral": 9, "alignment_harmony": 10, "alignment_unaligned_stacked": 11, "own_recipe": 12,
               "oracle": 13, "subgroups": 14, "age_adjusted": 15, "split67": 16, "pooled_predictions": 17,
               "best_candidate": 18, "outside_moments": 19,
               "female_only": 20,   # unused: Hung has no sex; its stream stays reserved, never reused
               "readout_elasticnet": 21, "readout_pls": 22, "readout_knn": 23, "readout_mlp": 24, "native_head": 25,
               "per_sample_route": 26, "learning_curve_25": 27, "learning_curve_50": 28, "one_type": 29,
               "feature_variants": 30, "within_bulk_reference": 31, "reference_rows": 32, "learning_curve_100": 33,
               "single_cell_moments": 34}
