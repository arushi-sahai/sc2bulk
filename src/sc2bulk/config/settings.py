"""The pinned numbers of stages 01-04 and 07, the seeds, and where outputs go.

This file is hashed into every stage's record (sc2bulk.codehash), so it is frozen with Phase 0
and Phase 1: an edit here reruns everything. Numbers for later stages live in their own config
modules (representations.py, readout.py, genes.py, statistics.py, search_space.py), hashed only
into the stages that import them.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# The smoke run (python pipeline/run_all.py --smoke) sets this; config/smoke.py holds its
# overrides.
SMOKE = os.environ.get("SC2BULK_SMOKE") == "1"

# Where every stage writes (SC2BULK_OUTPUTS, else outputs/ in the repository). The smoke run
# writes to its own subdirectory.
OUTPUTS = Path(os.environ.get("SC2BULK_OUTPUTS", REPO / "outputs"))
# The raw downloads, shared by the full and smoke runs (the smoke run never downloads twice)
RAW = OUTPUTS / "01_download"
if SMOKE:
    OUTPUTS = OUTPUTS / "smoke"
# Full-run mode: a clean tree at a tag, inputs made in this environment (sc2bulk.manifest)
FULL_RUN = os.environ.get("SC2BULK_FULL_RUN") == "1"
# The blinded-run switch: with a seed, the bulk
# labels are permuted at read time (sc2bulk.labels) and label-bearing outputs go to their own
# root, so no real result exists before the analysis choices are fixed.
BLINDED = int(os.environ["SC2BULK_PERMUTE_BULK_LABELS"]) if os.environ.get("SC2BULK_PERMUTE_BULK_LABELS") else None
LABELLED_OUTPUTS = OUTPUTS / f"blinded_{BLINDED}" if BLINDED is not None else OUTPUTS
LABELLED = "labelled"   # file-name prefix of label-bearing outputs (written with an underscore; sc2bulk.schemas)
# Written at the tagged freeze (the handoff checksum and
# the tag); label-bearing stages refuse to run without it unless the labels are blinded.
DEPOSIT_MARKER = RAW.parent / "deposit.json"
N_THREADS = int(os.environ.get("SC2BULK_THREADS", 16))   # BLAS/OpenMP/torch threads (sc2bulk.determinism)

# Preprocessing (every pair). No gene filtering here: genes are selected in stage 04, on
# each unit's training cells only.
QC_MIN_GENES_PER_CELL = 200
QC_MAX_PERCENT_MITO = 20.0
MAX_STORED_CELLS = 1000                # post-QC cells kept per sample (seeded random); pseudobulk sums these
MIN_CELLS_PER_SAMPLE = 100             # a floor that must drop nothing (asserted in stage 02)
ONE_TYPE_MIN_CELLS = 50                # Fig 6C: samples with fewer cells of the type are not stored
BULK_MIN_LIBRARY = 1_000_000           # bulk libraries with fewer counts are dropped
TARGET_SUM = 1e4                       # normalize_total over the shared genes, then log1p

# Gene panel: built from training single-cell data only
PANEL_SIZE = 5000
PANEL_MIN_CELL_FRACTION = 0.01         # a gene must be detected in >= 1% of training cells
PANEL_SIZE_SENSITIVITY = (1000, 2000)  # S11D: pseudobulk on smaller panels (5,000 = the panel; "all_shared" = every shared gene)
N_DISPERSION_BINS = 20                 # scanpy seurat-flavor normalized dispersion

# Seeds. Every random choice draws its seed from seeds.derived(purpose, pair, index): one
# SeedSequence per (base, purpose id, pair id, index), so no seed value serves two purposes
# and adding a pair or purpose changes no existing seed. Indices are 0, 1, 2, ...
SEED_BASE = 20261004
# ids 3 and 7 (attribution split/init) are retired: attribution uses encoder seeds 0-19
SEED_PURPOSES = {"encoder_split": 1, "search_split": 2, "readout_folds": 4,
                 "encoder_init": 5, "search_init": 6, "readout": 8, "bootstrap": 9,
                 "donor_fold": 10, "train_bags": 11, "test_bags": 12, "permutation": 13,
                 "learning_curve": 14, "fm_sampling": 15,
                 # declared ahead for stages not yet written, so this file never changes for them
                 "scvi_init": 16, "uce_sampling": 17, "simulation": 18, "deletion": 19,
                 "search_sampling": 20, "deconvolution": 21, "alignment": 22, "attribution": 23,
                 "within_bulk": 24, "qc_embed": 25, "readout_init": 26, "subsample": 27,
                 "spare_1": 28, "spare_2": 29, "spare_3": 30}
FOLD_SET_IDS = {"all": 0, "crossfit": 1, "split67": 2}   # seed entropy (sc2bulk.seeds)
SEED_PAIRS = {"hammond": 1, "soundlife": 2, "tms": 3, "pineda": 4, "lupus": 5, "crohns": 6}
N_ENCODER_SEEDS = 10
N_READOUT_SEEDS = 10
N_SEARCH_SEEDS = 5
READOUT_FOLDS = 5                      # read-out CV folds, grouped by holdout group, stratified
VALIDATION_FRACTION = 0.3              # of holdout groups within each 0/1 group

# Foundation models
GENEFORMER_REVISION = "1f7fbae"        # huggingface.co/ctheodoris/Geneformer, V2-104M, gc104M dictionaries
# Where the frozen models live (a Geneformer clone at GENEFORMER_REVISION; the scGPT whole-human
# checkpoint directory: best_model.pt, vocab.json, args.json, from the official scGPT release)
GENEFORMER_REPO = os.environ.get("SC2BULK_GENEFORMER")   # required by the FM stages (asserted there)
GENEFORMER_CHECKPOINT = "Geneformer-V2-104M"
GENEFORMER_COMMIT = "1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5"
# SHA-256 of the pinned files (the Git LFS pointers' oids at GENEFORMER_COMMIT; asserted at load)
GENEFORMER_FILES = {
    "Geneformer-V2-104M/model.safetensors": "fff5cba29ddd8792991fa77b4872246fbe548a178cebda3775cdc72b67780e7f",
    "geneformer/ensembl_mapping_dict_gc104M.pkl": "0819bcbd869cfa14279449b037eb9ed1d09a91310e77bd1a19d927465030e95c",
    "geneformer/gene_median_dictionary_gc104M.pkl": "a51c53f6a771d64508dfaf61529df70e394c53bd20856926117ae5d641a24bf5",
    "geneformer/gene_name_id_dict_gc104M.pkl": "fabfa0c2f49c598c59ae432a32c3499a5908c033756c663b5e0cddf58deea8e1",
    "geneformer/token_dictionary_gc104M.pkl": "67c445f4385127adfc48dcc072320cd65d6822829bf27dd38070e6e787bc597f",
}
SCGPT_DIR = os.environ.get("SC2BULK_SCGPT")
SCGPT_CHECKPOINT_MD5 = "9922ec94305126e6e4f9c1575cf493ae"   # whole-human best_model.pt
