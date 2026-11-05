"""The training harness's settings.

Hashed only into the stages that import it (sc2bulk.codehash).
"""
from sc2bulk.config import settings

# Early stopping and checkpoints
BURN_IN_EPOCHS = 10          # no checkpoint kept and no patience counted before this epoch, every encoder
MIN_STEPS = 50               # and no checkpoint kept before this many optimizer steps
VALIDATION_BAGS_PER_SAMPLE = 20   # a fixed set, drawn once per training, equal per validation sample

# "Did it train?": d = class-balanced mean over validation holdout groups of ln 2 - the
# group's mean BCE; its permutation p-value (labels permuted across holdout groups, logits fixed; all
# labellings up to NULL_PERMUTATIONS, else that many seeded draws). A seed passes if d* (below) >
# MARGIN_FLOOR x ln 2 and, on a split with at least MIN_TESTABLE labellings, p <= NULL_ALPHA (below it: "not testable", the collapse screen alone). Seeds are combined for eligibility in stage 05.
NULL_PERMUTATIONS = 1000
# The collapse screen's statistic d*: d's maximum over temperatures T on this fixed log grid
# in [1/20, 20] (61 points, T = 1 included), so the screen doesn't depend on the scale of the logits
TEMPERATURES = (*(20.0 ** (k / 30) for k in range(-30, 0)), 1.0, *(20.0 ** (k / 30) for k in range(1, 31)))
NULL_ALPHA = 0.05
MIN_TESTABLE = 100
MARGIN_FLOOR = 0.02

# Seed purposes (settings.SEED_PURPOSES): spare ids reserved for the harness
VALIDATION_BAGS_PURPOSE = "spare_1"

# Normalization
INPUT_CAP = 10.0
GENE_SCALER_MIN_Q = 1e-3     # pseudobulk gene scaler: s_g = 1 where the cells' mean log-CP10k < this
GENE_SCALER_MAX = 20.0       # ... and clamped to [0, 20]

# Training additions (CytoSet only)
AGE_LOSS_WEIGHT = 0.1
AGE_LOSS_SIGMA = 0.5         # in SD of the training samples' ages
AGE_LOSS_PAIRS = ("hammond", "tms", "soundlife")   # aging pairs only (n/a for Pineda, lupus, Crohn's)

# The frozen bag size, shared by every set encoder; also the main-row FM and
# scVI single-cell test bags. Each base's published size below is kept for Fig 4A's sweep.
FROZEN_NCELL = 500

# Each encoder's base: its authors' published settings
BASES = {
    "cytoset": {   # the original train.py defaults (config/search_space.BASE adds the model settings)
        "lr": 1e-4, "beta1": 0.9, "beta2": 0.999, "wts_decay": 1e-3, "batch_size": 200,
        "ncell": FROZEN_NCELL, "nsubset": 1024, "patience": 5, "n_epochs": 200, "cell_normalization": "none"},
    "set_transformer": {   # juho-lee/set_transformer @73432c6, main_pointcloud.py (classification)
        "lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "wts_decay": 0.0, "batch_size": 64,
        "ncell": FROZEN_NCELL, "nsubset": 1024, "patience": 5, "n_epochs": 200, "cell_normalization": "none",
        "dim_hidden": 256, "num_heads": 4, "num_inds": 16, "ln": False, "dropout": 0.5,
        "encoder_block": "isab", "encoder_layers": 2, "decoder_sab": False},
    "scrat": {   # yuzhenmao/ScRAT @cbf0772, main.py defaults (pca=True, as main.py and run.sh; the panel path is a candidate)
        "lr": 3e-3, "beta1": 0.9, "beta2": 0.999, "wts_decay": 1e-4, "batch_size": 16,
        "ncell": FROZEN_NCELL, "nsubset": 1024, "patience": 5, "n_epochs": 200, "cell_normalization": "none",
        "model_dim": 128, "num_heads": 8, "num_layers": 1, "dropout": 0.3, "input_dropout": 0.0, "pca": True,
        "norm_first": False,
        "step_lr": (50, 0.5)},   # its main.py's StepLR
}

# ScRAT's PCA input (pca=True, the base's input; the panel path is a candidate): components of the exact covariance
# of the split's training-side cells after the none/population transform (scanpy's default number)
SCRAT_PCA_COMPONENTS = 50

# Search streams that share a purpose with stage 06 (the validation bags, scVI's init) use index
# SEARCH_INDEX_OFFSET + i, so no stream serves two purposes
SEARCH_INDEX_OFFSET = 1000
# scVI's D3 bulk thinning draws (stage 06): scvi_init at SCVI_THIN_OFFSET + e, disjoint from the search's
SCVI_THIN_OFFSET = 2000
assert SCVI_THIN_OFFSET >= SEARCH_INDEX_OFFSET + 1000
# Stage 06 row D's label permutations: permutation at PERMUTATION_OFFSET + j (unit-keyed), clear of the
# blinded run's bulk-label permutation at index BLINDED (sc2bulk.labels); asserted here, at import, so it
# holds in every blinded process
PERMUTATION_OFFSET = 1000
assert settings.BLINDED is None or settings.BLINDED < PERMUTATION_OFFSET, "the blinded index must stay below PERMUTATION_OFFSET"
# Stage 06 row G's ScRAT mixup and its fixed bags: train_bags at MIXUP_OFFSET + e, clear of row A's 0-19 (attribution seeds included)
MIXUP_OFFSET = 1000
assert MIXUP_OFFSET >= 20

# ScRAT: the authors' repository at a pinned commit, fetched by the user (never redistributed)
SCRAT_REPO = "https://github.com/yuzhenmao/ScRAT"
SCRAT_COMMIT = "cbf07723233e2bac0a62be667dbda29a7afe06cd"

# Each encoder's published bag size (Fig 4A shows them; the frozen size is shared)
PUBLISHED_NCELL = {"cytoset": 200, "set_transformer": 1000, "scrat": 500}

# a unit is testable if every one of its search splits has >= MIN_TESTABLE labellings, fixed
# before the run from stage 03's splits (single-cell labels only; computed 2026-10-06: mouse organs'
# search splits have 35). Untestable units rely on the collapse screen and the adoption margin.
UNTESTABLE_UNITS = {("tms", "all", 0)}

# Dense device-resident inputs above this size use the host gather instead (the up-to-3,000-cell
# point on the large pairs; harness.Resident). Pre-declared; both paths are equal by test.
RESIDENT_MAX_BYTES = 14 * 2**30

# Micro-batch memory caps (bags per forward pass), per encoder and bag size: the micro-batch is the largest
# divisor of the configured batch at or below the cap (harness.micro_batch), so gradients accumulate to
# exactly the batch. Pinned from the timing test's memory block (block F, waves 1 and 1b): per size, the smallest over the memory-worst variants of the largest
# measured micro-batch with peak reserved GPU memory <= 22 GiB at one job per GPU on the largest resident unit
# (Sound Life crossfit_0, 12.35 GiB of cells; at 3,000 also Crohn's, resident at 13.7 GiB) and, on Sound Life's
# 3,000-cell host gather, peak host RSS <= 36 GiB. 1,024 (CytoSet's ICS recipe) bags whole stored samples
# (<= 1,000 cells), so it takes the 1,000 cap. With the age-similarity loss the micro-batch is the whole batch
# (its loss needs every bag of the batch).
MICRO_BATCH = {"cytoset": {500: 200, 600: 128, 1000: 64, 1024: 64, 3000: 16},
               "set_transformer": {500: 64, 600: 64, 1000: 32, 3000: 4},
               "scrat": {500: 32, 600: 16, 1000: 8, 3000: 1}}

# Stage 06's control, curve and recipe rows
N_PERMUTED = 20              # row D: label permutations j = 0-19, each at encoder seed j (10 under scale-down step 1)
CURVE_SEEDS = 5              # rows E, F, H: encoder seeds 0-4 (F 0-2 under scale-down step 3)
# Row E: the learning curve on "human blood, aging + the large disease pairs" (figure plan S12C):
# pinned from that wording (lupus 84 and Crohn's 359 holdout groups; ALS's 48 left out on "large"), not from a
# computed criterion. 100% is row A, reused.
LEARNING_CURVE_PAIRS = ("soundlife", "lupus", "crohns")
LEARNING_CURVE_FRACTIONS = (0.25, 0.5)
# Row F's "all" point (the 02b store, up to 3,000 cells): the pairs whose units have training samples with more
# than MAX_STORED_CELLS post-QC cells (stage 02's sample tables, 2026-10-05: 85-100% of every pair's samples;
# asserted per unit from 02b's index.csv at run time). One list for every configuration's "all" point.
FIG4A_ALL_PAIRS = ("hammond", "tms", "soundlife", "lupus", "crohns", "pineda")
# Pairs whose ScRAT mixup and H3 library sizes stay within a stratum: the samples' column
ORGAN_STRATUM = {"tms": "organ"}

# Row G: each encoder's own recipe on its base. Set Transformer:
# main_pointcloud.py's 1,000 points per set, 2,000 harness epochs, or the pre-declared fallback of 200 if their projected
# GPU-hours exceed 25% of stage 06's (set in stage 05's launch commit). Timing wave 2 projected ~280 of
# ~770 GPU-h at 2,000 epochs (36%), so 200 (ST_RECIPE_EPOCHS); nsubset pinned
ST_RECIPE_EPOCHS = 200
ST_RECIPE_NSUBSET = 1024
OWN_RECIPES = {
    "cytoset": {"h_dim": 256, "nblock": 3, "pool": "max", "out_pool": "max", "ncell": 1024, "nsubset": 20000,   # ICS
                "n_epochs": 100, "patience": 100},
    "set_transformer": {"ncell": 1000, "nsubset": ST_RECIPE_NSUBSET, "n_epochs": ST_RECIPE_EPOCHS, "patience": None},
    "scrat": {"lr": 1e-2, "batch_size": 256, "pca": True, "mixup": True, "n_epochs": 100, "patience": None},   # run.sh
}
# ScRAT's run.sh sample mixup (utils.mixups at SCRAT_COMMIT): augment_num, alpha, min_size, add_noise's variance,
# and train_num_sample (bags per synthetic sample, drawn once)
SCRAT_MIXUP = {"augment_num": 300, "alpha": 0.5, "min_size": 10000, "noise_var": 1e-5, "bags_per_sample": 20}
