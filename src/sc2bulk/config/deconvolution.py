"""The deconvolution baseline (stage 04b_deconvolve; reference rows deconvolution_clr /
deconvolution_true). Every number and map here is pre-declared, before any check runs.

Hashed only into the stages that import it (sc2bulk.codehash).
"""

REFERENCE_FRACTION = 0.002   # a reference type: >= 0.2% of the unit's training cells ...
REFERENCE_MIN_CELLS = 50     # ... and >= 50 cells
FEATURE_FRACTION = 0.01      # a feature type: a reference type with >= 1% (inclusive; Crohn's mesenchymal is exactly 1.00%)
MIN_CELLS_PER_SAMPLE_TYPE = 10   # a (sample, type) profile enters the signature only with >= 10 cells
MARKERS_PER_TYPE = 50
MARKER_MIN_CP10K = 1.0       # a marker's mean in its type, in counts per 10k
MARKER_MIN_LOG2_RATIO = 1.0  # types with fewer markers reaching this are logged
MISMATCH_LOG2 = 2.0          # genes with |log2(bulk CPM / training pseudobulk CPM)| > 2 are dropped
CLR_PSEUDOCOUNT = 0.005
V1_MIN_R = 0.5               # recovery gate: every feature type r >= 0.5 (else merged) ...
V1_MEDIAN_R = 0.8            # ... and the median >= 0.8
V3_FLAG_R = 0.5              # human blood, aging: mean of the folds' median r below this flags the row
V4_RESIDUAL_RATIO = 3.0      # bulk median residual > 3x the single-cell median flags "reference mismatch"
V1_SPLITS = "encoder_split"  # stage 03's shared validation splits

# Excluded from marker candidates: technical nuclear/cytoplasmic/bulk ratios
EXCLUDED_PREFIXES = ("MT-", "mt-", "RPS", "RPL", "Rps", "Rpl")
NEUTROPHIL_SCORE = ("CSF3R", "FCGR3B", "CXCR2")   # V4 leak indicator (blood pairs)

# Starting annotation per pair: the obs column, an optional map from its labels to types (labels
# not in a map are dropped), and the stratum a separate reference is built within.
LEVELS = {
    "hammond": None,            # one cell type: excluded
    "tms": {"column": "cell_type", "map": None, "within": "organ"},
    "soundlife": {"column": "AIFI_L2", "map": None, "within": None},
    "lupus": {"column": "cell_type", "map": None, "within": None},
    "crohns": {"column": "cell_type", "within": None, "map": {
        # nested in the provider's cell_category
        **dict.fromkeys(["enterocyte of epithelium proper of ileum", "BEST4+ enterocyte", "BEST4+ colonocyte",
                         "colonocyte", "early colonocyte"], "absorptive epithelium"),
        **dict.fromkeys(["ileal goblet cell", "anorectum goblet cell", "paneth cell of epithelium of small intestine",
                         "intestinal tuft cell", "intestinal enteroendocrine cell", "enterochromaffin-like cell"],
                        "secretory epithelium"),
        **dict.fromkeys(["intestinal crypt stem cell of small intestine", "intestinal crypt stem cell of colon"], "stem/TA"),
        **dict.fromkeys(["B cell", "memory B cell", "naive B cell"], "B"),
        **dict.fromkeys(["plasma cell", "plasmablast"], "plasma"),
        **dict.fromkeys(["CD4-positive, CD25-positive, alpha-beta regulatory T cell", "CD4-positive, alpha-beta T cell",
                         "CD4-positive, alpha-beta memory T cell", "CD4-positive, alpha-beta memory T cell, CD45RO-positive",
                         "central memory CD4-positive, alpha-beta T cell"], "CD4 T"),
        **dict.fromkeys(["CD8-positive, alpha-beta memory T cell, CD45RO-positive", "effector memory CD8-positive, alpha-beta T cell",
                         "effector memory CD8-positive, alpha-beta T cell, terminally differentiated",
                         "granzyme K-associated CD8 T cell", "naive thymus-derived CD8-positive, alpha-beta T cell"], "CD8 T"),
        **dict.fromkeys(["gamma-delta T cell", "innate lymphoid cell", "natural killer cell", "double negative thymocyte"],
                        "other lymphoid"),
        **dict.fromkeys(["classical monocyte", "conventional dendritic cell", "intermediate monocyte", "monocyte",
                         "non-classical monocyte", "plasmacytoid dendritic cell", "tissue-resident macrophage"], "myeloid"),
        **dict.fromkeys(["fibroblast", "glial cell", "gut endothelial cell", "myofibroblast cell", "pericyte",
                         "subepithelial intestinal fibroblast", "vascular associated smooth muscle cell"], "mesenchymal"),
        "mast cell": "mast", "platelet": "platelet"}},
    "pineda": {"column": "cell_type", "within": None, "map": {   # major classes
        **dict.fromkeys(["L2_L3", "L3_L5", "L4_L5", "L4_L6", "L5", "L5_L6", "L6"], "excitatory"),
        **dict.fromkeys(["5HT3aR", "PV", "Rosehip", "SOM"], "inhibitory"),
        "Astro": "astrocyte", "Oligo": "oligodendrocyte", "OPC": "OPC", "Micro": "microglia",
        **dict.fromkeys(["Endo", "Fibro", "Mural"], "vascular"), "T_Cell": "T cell"}},
}
# Monocyte types for the V4 leak indicator, matched by substring of the type name
MONOCYTE = "monocyte"
