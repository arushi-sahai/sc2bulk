"""Gene lists and attribution (stages 10 and 10c).

Hashed only into the stages that import it (sc2bulk.codehash).
"""
from pathlib import Path

TOP_GENES = 50
TOP_K = (25, 50, 100)                  # every ranking's top-k flags (read from one score, no refit)

# The foundation models' cell subsample (shared by scGPT and Geneformer): m cells per sample,
# uniform without replacement from its stored cells (seeds: `attribution` index 0 and the sample), halves A and
# B = the first and second m/2 of the draw. Geneformer under fallback step "cells_25" takes the first 25 (nested).
FM_CELLS = 50
SCGPT_CHUNK = 10                       # scGPT gradient x input: cells per forward pass (exact; fp32 fits a 24 GB card)

# Geneformer deletion samples: pairs without cross-fitting explain every training
# sample, capped at DELETION_CAP (round robin over holdout groups within each 0/1 group, `attribution` index 1);
# human blood, aging explains per_donor of each training donor's samples (the first of the donor's draw from
# `attribution` index 1, so the lever's 1 is nested in the 2), per fold.
DELETION_CAP = 150

# The pilot: mouse microglia, PILOT_SAMPLES samples x
# PILOT_CELLS cells (the first of their deletion order and cell draw) plus their profiles. Gates:
# (a) reproduction: the unperturbed fp32 embedding vs stage 07's, cosine >= reproduction per sample (assert);
# (b) precision: fp16 autocast vs fp32 D (Geneformer) or gradient x input (scGPT), Spearman over the pooled
#     (sample, gene) values >= spearman and top-50 overlap of the 3-sample mean_abs >= overlap, on each
#     configuration separately; else fp32;
# (c) measured tokens/s, and L per pair for all six pairs (no forward passes); the budget is recomputed at the
#     precision (b) selects and compared with budget_days on `gpus` GPUs.
PILOT = {"pair": "hammond", "samples": 3, "cells": 10, "reproduction": 0.9999, "spearman": 0.98, "overlap": 0.9,
         "budget_days": 4.0, "gpus": 4}
# Attention overhead of a length-L pass relative to L ~ 1,000, interpolated linearly
ATTENTION_OVERHEAD = ((1000, 1.0), (2000, 1.2), (4094, 1.5))
CHECK_OVERHEAD = {"cells": 0.10, "pseudobulk": 0.03}   # the checks' joint deletions

# The Geneformer fallback order, fixed before the pilot: the first step whose recomputed
# budget is within PILOT["budget_days"] is run; a step marked fp32_only is skipped unless gate (b) forced fp32;
# "triage" drops the deletion.
GF_FALLBACK_ORDER = (
    {"step": "per_donor_2", "per_donor": 2, "cells": 50, "screen": False, "fp32_only": False},
    {"step": "per_donor_1", "per_donor": 1, "cells": 50, "screen": False, "fp32_only": False},
    {"step": "cells_25", "per_donor": 1, "cells": 25, "screen": False, "fp32_only": True},
    {"step": "two_stage_screen", "per_donor": 1, "cells": None, "screen": True, "fp32_only": False},   # cells: the last step's run
    {"step": "triage"},
)
# The two-stage screen: stage 1 deletes every expressed gene in a fresh 10-cell draw
# (`deletion` index 10), stage 2 the top `candidates` in the main cells; accepted if every final top-`top` gene
# was within stage 1's top `margin`, else the candidates expand to `expanded`.
GF_SCREEN = {"cells": 10, "candidates": 500, "margin": 250, "expanded": 1000, "top": 100}
GF_BATCH = 32                          # deletion variants per forward pass (one input's variants share a length)

# The Geneformer checks and inclusion rule: random sets matched on the finest n x n grid
# of GF_MATCH_BINS in which every cell holding top genes holds >= GF_MATCH_POOL x as many other genes,
# the cumulative deletion curve's k, and the thresholds
GF_RANDOM_SETS = 100
GF_MATCH_BINS = (5, 4, 3, 2)               # finest first
GF_MATCH_POOL = 2
GF_CURVE_K = (1, 5, 10, 25, 50, 100)
ATTRIBUTION_CHECKS = {"stability": 0.5, "hk_p": 0.01, "hk_excess": 0.20, "random_pct": 95, "additivity": (0.5, 2.0)}

# Housekeeping genes: HRT Atlas v1.0 human housekeeping genes (Hounkpe et al. 2021, Nucleic Acids
# Res 49:D947), pinned as a file (sources.py is frozen: spec Q8): columns gene (the HRT symbol, display only) and
# ensembl (the human Ensembl gene id it is matched on) under a header naming the source, written by
# scripts/hrt_atlas_ensembl.py from the HRT transcript list and the Ensembl 111 GTF.
HOUSEKEEPING_FILE = Path(__file__).resolve().parent / "attribution" / "hrt_atlas_v1.tsv"
HOUSEKEEPING_SHA256 = "f33c3b3ce3febdb1a594a095d3d51ed56b5b8546fdb0e7df2efc96f3c38aba35"   # 2156 genes, written 2026-10-10

# Fig 6B: "large-difference" genes = the unit's top LARGE_DIFFERENCE_FRACTION of panel genes by
# |SMD| between scored bulk and the training samples' pseudobulk
LARGE_DIFFERENCE_FRACTION = 0.05
# Fig 5B's matched reference: for each top-k gene, its MATCHED_NEIGHBOURS nearest non-top genes in
# |single-cell shift| (ties by symbol), averaged with multiplicity
MATCHED_NEIGHBOURS = 10
# The set-encoder reproduction assert: max |delta| <= REPRODUCTION_TOLERANCE x max(1, |e|)
REPRODUCTION_TOLERANCE = 1e-4
