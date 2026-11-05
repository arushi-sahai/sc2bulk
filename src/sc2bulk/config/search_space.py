"""The configuration searches (stage 05): CytoSet's grid below, then Set Transformer's,
ScRAT's and scVI's (one change from each base; ScRAT's norm_per_sample: two); the selection rules are in sc2bulk.search.

CytoSet's base is the original CytoSet's own defaults (train.py at CompCy-lab/cytoset @830f5a2),
plus the changes logged in third_party/cytoset/CHANGES.txt. The search changes one setting
at a time. In each group the candidate with the lowest mean validation BCE over the search
seeds is adopted if it beats the base by more than the SD of the seed-paired differences, and
only if it is eligible (d* above the collapse floor on a majority of seeds; on
testable units, a majority of seeds with permutation p <= 0.05). The groups' winners are then merged, and the merged configuration
must beat the base by the same rule; otherwise the single best change is kept, or the base
if no change cleared.

Every candidate keeps its checkpoint and gets the full transfer read-out: Fig 4's CytoSet
points are the candidates' bulk transfer (change from the base), with the frozen
configuration marked. Selection never reads the bulk side. Options marked "added in this study"
are not in the original CytoSet.
"""
from sc2bulk.config import representations
from sc2bulk.config import training as T

BASE = {
    # model (upstream defaults)
    "h_dim": 64,
    "nblock": 1,
    "pool": "max",           # block-level pooling
    "out_pool": "mean",      # output pooling
    "decoder": "mlp",        # upstream: Dropout(0.5) -> Linear -> ELU -> Dropout(0.5) -> Linear
    # optimizer (upstream defaults; constant learning rate, no gradient clipping)
    "lr": 1e-4,
    "beta1": 0.9,
    "beta2": 0.999,
    "wts_decay": 1e-3,
    "batch_size": 200,
    # bags: ncell random cells per bag (the shared frozen size; upstream's default 200 is a
    # Fig 4A sweep point); nsubset bags per class per
    # epoch, spread evenly over that class's training samples
    "ncell": 500,
    "nsubset": 1024,
    # early stopping on validation BCE (upstream defaults)
    "patience": 5,
    "n_epochs": 200,
    # input (changed from upstream, which applied arcsinh to cytometry): log-CP10k, no
    # per-gene standardization
    "cell_normalization": "none",
    # objective (upstream): BCE on the 0/1 group
    "objective": "bce",
    # training additions (added in this study), all off in the base
    "abundance_weighting": False,
    "age_similarity_loss": False,
    "population_distance_feature": False,
    "pseudobulk_gene_scaler": False,   # applied before any standardization
}

# group -> {candidate name: the setting it changes}. The base is every group's reference.
GROUPS = {
    "nblock": {"nblock2": {"nblock": 2}, "nblock3": {"nblock": 3}},
    "h_dim": {"h128": {"h_dim": 128}, "h256": {"h_dim": 256}, "h512": {"h_dim": 512}},
    "pool": {"pool_mean": {"pool": "mean"}, "pool_sum": {"pool": "sum"},
             "pool_attention": {"pool": "attention"}},            # attention: added in this study
    "out_pool": {"out_max": {"out_pool": "max"}, "out_sum": {"out_pool": "sum"},
                 "out_attention": {"out_pool": "attention"}},     # attention: added in this study
    "decoder": {"decoder_linear": {"decoder": "linear"}},
    "cell_normalization": {"norm_population": {"cell_normalization": "population"},
                           "norm_per_sample": {"cell_normalization": "per_sample"}},
    "wts_decay": {"wd0": {"wts_decay": 0.0}, "wd1e-4": {"wts_decay": 1e-4}, "wd1e-2": {"wts_decay": 1e-2}},
    "lr": {"lr3e-5": {"lr": 3e-5}, "lr3e-4": {"lr": 3e-4}},
    "batch_size": {"bs32": {"batch_size": 32}, "bs64": {"batch_size": 64}, "bs128": {"batch_size": 128}},
    # training additions (added in this study), each switched on alone
    "abundance_weighting": {"abundance_weighting": {"abundance_weighting": True}},
    "age_similarity_loss": {"age_similarity_loss": {"age_similarity_loss": True}},
    "population_distance_feature": {"population_distance_feature": {"population_distance_feature": True}},
    "pseudobulk_gene_scaler": {"pseudobulk_gene_scaler": {"pseudobulk_gene_scaler": True}},
}

# Bag size is not searched (validation BCE at different bag sizes measures different
# tasks); it is Fig 4A's sweep, one encoder per size (train = test = n), at each encoder's frozen
# configuration with only the bag size varied, the same seeds for every encoder. "all" = the
# up-to-3,000-cell store (stage 02b).
FIG4A_BAG_SIZES = representations.BAG_SIZES + ("all",)   # figure plan v3: incl. 50 and 500

# One change from the base, trained at the search seeds, but outside the search: its
# validation loss (MSE) can't be compared with BCE. A Fig 4C variant.
OUTSIDE_SEARCH = {"objective_mse": {"objective": "mse"}}
# Search seeds: settings.N_SEARCH_SEEDS (5), drawn by seeds.derived("search_init"/"search_split", ...)


def _shared(lr, wd, batch):
    """The knobs every set encoder searches: normalization, learning rate (about
    ±½ decade around the base), weight decay, batch size."""
    return {"cell_normalization": {"norm_population": {"cell_normalization": "population"},
                                   "norm_per_sample": {"cell_normalization": "per_sample"}},
            "lr": {f"lr{_name(v)}": {"lr": v} for v in lr},
            "wts_decay": {f"wd{_name(v)}": {"wts_decay": v} for v in wd},
            "batch_size": {f"bs{v}": {"batch_size": v} for v in batch}}


def _name(v) -> str:
    """A level in candidate ids, as CytoSet's are written: 0, 1e-4, 3e-3, 1e-2, 0.1, 0.25, 64."""
    if isinstance(v, float) and v and v < 0.05:
        mantissa, exponent = f"{v:.0e}".split("e")
        return f"{mantissa}e{int(exponent)}"
    return f"{v:g}" if isinstance(v, float) else f"{v}"


def _levels(key, values, prefix):
    return {key: {f"{prefix}{_name(v)}": {key: v} for v in values}}


# Set Transformer (base: training.BASES; the repository's classification configuration). Pooling stays
# PMA; every candidate is a repository or paper option.
SET_TRANSFORMER_GROUPS = {
    **_shared((3e-4, 3e-3), (1e-4, 1e-3, 1e-2), (32, 128, 256)),
    **_levels("dim_hidden", (64, 128, 512), "d"),
    **_levels("num_heads", (2, 8), "heads"),
    **_levels("num_inds", (8, 32, 64), "inds"),
    **_levels("encoder_layers", (1, 3), "layers"),
    "ln": {"ln": {"ln": True}},
    **_levels("dropout", (0.0, 0.1, 0.25), "dropout"),
    "encoder_block": {"sab": {"encoder_block": "sab"}},          # the paper's SAB encoder
    "decoder_sab": {"decoder_sab": {"decoder_sab": True}},       # models.SetTransformer's SAB x 2 after PMA
}

# ScRAT (base: training.BASES; its main.py defaults, PCA input included). Dropped:
# positional encoding, a 4x feed-forward width, no StepLR, cosine warm-up.
SCRAT_GROUPS = {
    **_shared((1e-3, 1e-2), (0.0, 1e-3, 1e-2), (8, 32, 64)),
    # the one two-setting candidate: PCA is defined on the none/population routes only
    "cell_normalization": {"norm_population": {"cell_normalization": "population"},
                           "norm_per_sample": {"cell_normalization": "per_sample", "pca": False}},
    **_levels("model_dim", (64, 256, 512), "dim"),
    **_levels("num_layers", (2, 3, 4), "layers"),
    **_levels("num_heads", (2, 4, 16), "heads"),
    **_levels("dropout", (0.0, 0.1, 0.5), "dropout"),
    "norm_first": {"norm_first": {"norm_first": True}},
    "pca": {"panel_input": {"pca": False}},                       # the panel instead of the PCA input
    **_levels("input_dropout", (0.1, 0.3), "input_dropout"),     # a model option; main.py leaves it at 0
}

# scVI (outside the set-encoder budget): one change from the scvi-tools
# defaults, selected on validation ELBO with the same margin, merge and fallback rules.
SCVI_BASE = {"n_latent": 10, "n_layers": 1}
SCVI_GROUPS = {**_levels("n_latent", (20, 30), "latent"), **_levels("n_layers", (2,), "layers")}

GROUPS_BY_ENCODER = {"cytoset": GROUPS, "set_transformer": SET_TRANSFORMER_GROUPS, "scrat": SCRAT_GROUPS}


def base(encoder: str) -> dict:
    """An encoder's full base configuration."""
    return {**(BASE if encoder == "cytoset" else {}), **T.BASES[encoder]}


def candidates(encoder: str, pair: str) -> dict:
    """{group: {candidate: change}} for one encoder on one pair: the age-similarity loss only on the
    aging pairs (25 CytoSet candidates elsewhere, reported)."""
    groups = dict(GROUPS_BY_ENCODER[encoder])
    if encoder == "cytoset" and pair not in T.AGE_LOSS_PAIRS:
        del groups["age_similarity_loss"]
    return groups
