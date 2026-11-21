"""The display items: numbering, citation plan, titles, figure text, and the figure scripts' per-item settings.

Imported by no pipeline stage, so editing it never invalidates a stage (tests/test_figures_config.py asserts it is
outside every stage's code hash). Every label string a figure draws and every number that carries meaning in a
figure comes from here, config/style.py, config/configurations.py or config/pairs.
"""
from sc2bulk.config import configurations, deconvolution, properties, statistics, training
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

# ---- the display order ----

# The display order of every key and row: pseudobulk, then the supervised set
# encoders, the unsupervised generative model and the frozen foundation models. Display only: the stages keep
# configurations.CONFIGURATION_ORDER, which is inside their code hashes (this file is not).
DISPLAY_ORDER = ("pseudobulk", "cytoset_cells", "cytoset_pseudobulk", "set_transformer_cells", "scrat_cells", "scvi_cells",
                 "geneformer_cells", "geneformer_pseudobulk", "scgpt_cells", "scgpt_pseudobulk")
assert sorted(DISPLAY_ORDER) == sorted(configurations.CONFIGURATION_ORDER)
MODEL_ORDER = tuple(dict.fromkeys(configurations.CONFIGURATIONS[k]["model"] for k in DISPLAY_ORDER))
# The family names, keyed by configurations.MODELS "family"; pseudobulk is the baseline, with no family label
FAMILY_LABELS = {"set_encoder": "Supervised set encoders", "generative": "Unsupervised generative model",
                 "foundation_model": "Frozen foundation models"}

# ---- the display items ----

# Items in their final numbering: type, the script that draws it ("writing" or None for tables, which are
# a separate item) and the width (style.WIDTH_MM key). Keys are the paper's labels (Genome Biology's "Fig. S1", "Table S1"; the data files are "Additional file 3"-"6"); internal file names keep
# the earlier stems (S8Fig.png, S9AFig.csv; figures/common.stem). Supplementary figures are numbered by first
# main-text citation (CITATIONS below).
QC_FIGURES = {pair: f"Fig. S{i + 2}" for i, pair in enumerate(PAIR_ORDER)}   # Figs. S2–S7, one per pair

ITEMS = {
    "Fig 1": {"type": "main", "script": "fig1", "width": "double"},      # A, B: figures/schematics.py
    "Fig 2": {"type": "main", "script": "fig2", "width": "double"},
    "Fig 3": {"type": "main", "script": "fig3", "width": "double"},
    "Fig 4": {"type": "main", "script": "fig4", "width": "double"},
    "Fig 5": {"type": "main", "script": "fig5", "width": "double"},
    "Fig 6": {"type": "main", "script": "fig6", "width": "double"},
    "Table 1": {"type": "table", "script": None},
    "Fig. S1": {"type": "s_fig", "script": "s01", "width": "double"},
    **{item: {"type": "s_fig", "script": "s02_s07_qc", "width": "double", "pair": pair} for pair, item in QC_FIGURES.items()},
    "Fig. S8": {"type": "s_fig", "script": "s08", "width": "double"},
    "Fig. S9": {"type": "s_fig", "script": "s09", "width": "double"},
    "Fig. S10": {"type": "s_fig", "script": "s10", "width": "double"},
    "Fig. S11": {"type": "s_fig", "script": "s11", "width": "double"},
    "Fig. S12": {"type": "s_fig", "script": "s12", "width": "double"},
    "Fig. S13": {"type": "s_fig", "script": "s13", "width": "double"},
    "Fig. S14": {"type": "s_fig", "script": "s14", "width": "double"},
    "Fig. S15": {"type": "s_fig", "script": "s16", "width": "double"},
    "Fig. S16": {"type": "s_fig", "script": "s17", "width": "double"},
    **{f"Table S{i}": {"type": "s_table", "script": None} for i in range(1, 13)},
    **{f"Additional file {i}": {"type": "s_data", "script": "data_files"} for i in range(3, 7)},
}

# Main figures whose figure + legend (style.MAX_HEIGHT_MM) is not yet within the limit; each leaves this set in its figure pass
LEGEND_FIT_PENDING = {"Fig 2"}
# Figures whose layout checks (figures/common.check_layout) are reported but not yet asserted; each leaves this set
# once its report is clean (empty since 2026-11-04: every figure is checked)
LAYOUT_PENDING = set()
# Each item's drawn height in mm (Genome Biology: <= 225 mm including the legend; a working cap below it, set from the
# rehearsal's PNGs; style.MAX_HEIGHT_MM is the hard limit)
HEIGHT_MM = {"Fig 1": 160, "Fig. S1": 114,
             "Fig 2": 159, "Fig 3": 170, "Fig 4": 163, "Fig 5": 160, "Fig 6": 157, "Fig. S8": 190, "Fig. S9": 215,
             "Fig. S10": 193, "Fig. S11": 220, "Fig. S12": 125, "Fig. S13": 225, "Fig. S14": 215, "Fig. S15": 225, "Fig. S16": 215,
             **{item: 150 if pair in ("hammond", "soundlife", "lupus") else 190 for pair, item in QC_FIGURES.items()}}   # S2-S7: three or four rows

# The S12B UCE panel is drawn iff UCE is run (not a configuration, so not in CUT_CONFIGURATIONS): asserted against the
# scores table both ways, as FIG4A_FM_ALL_AVAILABLE. No stage writes UCE predictions yet (07c_uce is planned).
S12B_UCE = False

# Per-pair QC panels after the template's A-G (captions Figs. S2–S7), from 02/03/04d outputs: panel -> (description,
# source, column). Sources: "single_cell" (stage 02's kept sample table), "bulk" (stage 02's bulk sample table), "labels"
# (the bulk label table's own columns), "folds" (stage 03's donor folds), "organs_per_donor" (each single-cell mouse's
# number of organs), "restriction" (stage 02's whole sample table: kept vs. dropped by the batch-4 restriction, by disease
# status). S5I (lupus bulk flowcell) is in no stage output and is dropped (Table S1 states it).
CROHNS_BATCH_PROXIES = ("bead_lot", "chip_lot", "gem_lot", "enzyme_lot_BLP", "chip_version", "bead_version")   # data/crohns.py
LUPUS_RESTRICTION = "donor's majority processing cohort is not 4"   # data/lupus.py's drop reason for the batch-4 restriction
QC_EXTRA = {
    "hammond": {"H": ("Bulk samples by early-life-stress group", "bulk", "condition")},
    "tms": {"H": ("Samples per organ on each side", "both", "organ"),
            "I": ("Mice contributing to more than one organ", "organs_per_donor", "organ")},
    "soundlife": {"H": ("Donors per cross-fitting fold by age group", "folds", "fold"),
                  "I": ("Cytomegalovirus serostatus by age group", "single_cell", "cmv")},
    "lupus": {"H": ("Single-cell samples kept and dropped by the batch-4 restriction, by disease status", "restriction", "restriction")},
    "crohns": {"H": ("Batch proxies by group", "single_cell", CROHNS_BATCH_PROXIES), "I": ("Donor age by group", "bulk", "age"), "J": ("Inflamed and non-inflamed bulk biopsies", "labels", "inflammation")},
    "pineda": {"H": ("Potential confounders by group", "single_cell", ("sex", "age_at_death", "pmi_hours", "run"))},
}
QC_TEMPLATE = {"A": "Per-cell QC with thresholds", "B": "Cells per sample by group", "C": "UMAP by annotated cell type",
               "D": "UMAP by group", "E": "Bulk samples per group", "F": "Bulk library size and genes detected",
               "G": "Bulk PCA by group"}
QC_LABELS = {"genes_detected": "Median genes\nper cell", "total_counts": "Median counts\nper cell",
             "percent_mito": "Median mito. %\nper cell", "cells_after_qc": "Cells per sample", "per_level": "Samples per level",
             "library_size": "Library size\n(millions)", "bulk_genes_detected": "Genes detected\n(thousands)"}
LEVELS_LABEL = "{label}\n{n} levels"  # a count panel whose level names don't fit flat drops them and states their number
# The 0/1 groups' display names per pair (QC figures), and across pairs (S8's key)
GROUP_NAMES = {"hammond": ("younger", "older"), "tms": ("younger", "older"), "soundlife": ("younger", "older"),
               "lupus": ("control", "lupus"), "crohns": ("control", "Crohn's"), "pineda": ("control", "ALS")}
GROUP_NAMES_ALL = ("control / younger", "case / older")
ANNOTATION_TEXT = {"SLE": "lupus"}   # stage 04d annotations as printed (config/fig1e, hashed into 04d, keeps its own)
# Display labels of sample-table columns in QC and S8 panels (others: underscores to spaces, sentence case)
COLUMN_LABELS = {"age": "Age (years)", "age_at_death": "Age at death (years)", "pmi_hours": "Postmortem interval (hours)",
                 "cmv": "CMV serostatus", "organs": "Organs per mouse", "enzyme_lot_BLP": "Enzyme lot (BLP)",
                 "batch_id": "Batch ID", "release_cohort": "Release cohort", "bead_version": "Bead version (v)"}
LEVEL_PREFIX = {"bead_version": "V"}  # dropped from the tick labels (the axis label carries it)
# Many-level columns sharing one axes (S6h's lots): short tick names, the shared x label
COLUMN_SHORT = {"bead_lot": "Bead", "chip_lot": "Chip", "gem_lot": "GEM", "enzyme_lot_BLP": "Enz.\n(BLP)"}
MANY_LEVELS_LABEL = "Lot (number of levels)"
QC_KEY_COLUMNS = 4                     # S2-S7's key (sides and groups) at the top right
MAX_ROTATED_LEVELS = 6                 # ... unless there are this few (then at 45 degrees)
LEVEL_WRAP = 10                        # characters per line of a level name or a count panel's x label
LEVEL_GAP_MM = 1.0                     # level labels lie flat when neighbours are at least this far apart (measured)
# Fig. S11E's subset titles where the level alone is long or unclear (others: the level, its first letter capitalized)
SUBGROUP_SHORT = {"Negative": "CMV−", "Positive": "CMV+", "NYGC ALS Consortium": "NYGC"}

# Short display names for long annotated cell types (UMAP labels, S8B's legend); others are shown as annotated,
# wrapped at TYPE_WRAP characters. The CSVs keep the annotated names.
TYPE_WRAP = 22
# single-pair main panels: the pair's name beside the letter, this many points above the axes (clears the plot titles)
PAIR_TAG_PT = {"Fig 3A": 28, "Fig 5D": 16}
# the figures' alignment names where they differ from config/configurations (that file is in frozen closures)
# ("none": stage 08's per-side standardization; "unaligned_stacked": one scaler fit over both sides)
ALIGNMENT_DISPLAY = {"none": "within-assay", "unaligned_stacked": "across-assay"}
ALIGNMENT_TICKS = {"unaligned_stacked": "across-assay", "none": "within-assay", "combat": "ComBat", "coral": "CORAL",
                   "harmony": "Harmony"}   # Fig 3b and Fig. S13f, h: named on the first pair's pseudobulk dots
# Fig 6C's cell type per pair, in its column title: each pair's one-type annotation (config/pairs one_type)
# as displayed; a pair with a type per stratum (mouse organs, per organ) shows ONE_TYPE_PER_STRATUM, its types in the caption
FIG6C_TYPE_NAMES = {"T cell": "T cells", "CD4-positive, alpha-beta T cell": "CD4+ αβ T cells",
                    "enterocyte of epithelium proper of ileum": "ileal enterocytes", "Oligo": "oligodendrocytes"}
ONE_TYPE_PER_STRATUM = "one type per organ"
CELL_TYPE_SHORT = {
    "mesenchymal stem cell": "MSC", "mesenchymal stem cell of adipose": "adipose MSC",
    "granulocytopoietic cell": "granulocyte progenitor",
    "kidney proximal convoluted tubule epithelial cell": "PCT epithelial cell",
    "epithelial cell of proximal tubule": "PT epithelial cell",
    "CD4-positive, alpha-beta T cell": "CD4 T cell", "CD8-positive, alpha-beta T cell": "CD8 T cell",
    "CD4-positive, alpha-beta memory T cell": "CD4 memory T cell",
    "CD4-positive, alpha-beta memory T cell, CD45RO-positive": "CD4 memory T cell",
    "CD8-positive, alpha-beta memory T cell, CD45RO-positive": "CD8 memory T cell",
    "CD4-positive, CD25-positive, alpha-beta regulatory T cell": "Treg",
    "central memory CD4-positive, alpha-beta T cell": "CD4 TCM",
    "effector memory CD8-positive, alpha-beta T cell": "CD8 TEM",
    "effector memory CD8-positive, alpha-beta T cell, terminally differentiated": "CD8 TEMRA",
    "naive thymus-derived CD8-positive, alpha-beta T cell": "naive CD8 T cell",
    "granzyme K-associated CD8 T cell": "GZMK+ CD8 T cell",
    "classical monocyte": "classical mono.", "non-classical monocyte": "non-classical mono.",
    "intermediate monocyte": "intermediate mono.",
    "conventional dendritic cell": "cDC", "plasmacytoid dendritic cell": "pDC", "natural killer cell": "NK cell",
    "tissue-resident macrophage": "resident macrophage",
    "enterocyte of epithelium proper of ileum": "ileal enterocyte",
    "intestinal crypt stem cell of small intestine": "SI crypt stem cell",
    "intestinal crypt stem cell of colon": "colon crypt stem cell",
    "paneth cell of epithelium of small intestine": "Paneth cell",
    "intestinal enteroendocrine cell": "enteroendocrine cell",
    "subepithelial intestinal fibroblast": "subepithelial fibroblast",
    "vascular associated smooth muscle cell": "vascular SMC",
    "absorptive epithelium": "absorptive epith.", "secretory epithelium": "secretory epith.",
}

FIG13_OTHER_PAIRS = tuple(p for p in PAIR_ORDER if p != "crohns")    # S13A-E: Fig 3A for the other pairs, in pair order
S15_KEYS = tuple(k for k in DISPLAY_ORDER if k.endswith("_cells") and k != "cytoset_cells")   # Fig 6B left's others
AGE_ADJUSTED = {"bulk": ("crohns",), "within": ("pineda",)}   # S11F: 11b's covariate-adjusted rows
S9_READOUTS = {"A": "lasso", "B": "elasticnet", "C": "pls", "D": "knn", "E": "mlp", "F": "native_head"}

# figure -> {panel: short description}: the registry every panel's CSV is checked against
PANELS = {
    "Fig 1": {"A": "The task", "B": "How a sample's cells become one vector", "C": "Pseudobulk and nine learned configurations",
              "D": "The dataset pairs", "E": "What each pair adds"},
    "Fig. S1": {"A": "CytoSet", "B": "Set Transformer", "C": "ScRAT", "D": "scVI, Geneformer and scGPT"},
    "Fig 2": {"A": "Difference in bulk AUROC from pseudobulk", "B": "Within-single-cell vs. bulk AUROC",
              "C": "Decomposition of the gap"},
    "Fig 3": {"A": "The gap illustrated (Crohn's)", "B": "Modality silhouette", "C": "Bulk AUROC under alignment"},
    "Fig 4": {"A": "Cells per bag", "B": "Output-level pooling", "C": "Input normalization"},
    "Fig 5": {"A": "Gene shifts", "B": "Same-sign share", "C": "Same-sign share vs. bulk AUROC", "D": "Simulation"},
    "Fig 6": {"A": "Sample properties", "B": "Modality-direction genes", "C": "One cell type only"},
    **{item: {**QC_TEMPLATE, **{k: v[0] for k, v in QC_EXTRA[pair].items()}} for pair, item in QC_FIGURES.items()},
    "Fig. S8": {"A": "Every characteristic", "B": "Cell-type composition: single-cell and deconvolved bulk",
               "C": "Tissue-difference markers", "D": "Batch x group", "E": "Samples per side by group", "F": "Mouse-organ counts"},
    "Fig. S9": {k: f"Read-out: {v}" for k, v in S9_READOUTS.items()},
    "Fig. S10": {"A": "Validation loss per candidate", "B": "Bulk AUROC vs. validation loss (best on bulk marked)",
                "C": "Outside the search", "D": "Training checks per encoder seed", "E": "scVI search"},
    "Fig. S11": {"A": "Bulk predictions by group", "B": "Seed variability and ensembling", "C": "Pseudobulk normalizations",
                "D": "Gene-panel size", "E": "Subgroups", "F": "Age adjustment"},
    "Fig. S12": {"A": "Full gene sets and orthologs", **({"B": "UCE"} if S12B_UCE else {}), "C": "Learning curves"},
    "Fig. S13": {**{chr(ord("A") + i): f"The gap illustrated ({p})" for i, p in enumerate(FIG13_OTHER_PAIRS)},
                "F": "Modality silhouette, configurations on cells", "G": "Bulk AUROC, configurations on cells",
                "H": "Modality silhouette, pseudobulk-input configurations", "I": "Bulk AUROC, pseudobulk-input configurations"},
    "Fig. S14": {"A": "Stability", "B": "Top 25 and top 100", "C": "LASSO genes", "D": "Attribution checks"},
    "Fig. S15": {chr(ord("A") + i): f"Modality-direction genes ({k})" for i, k in enumerate(S15_KEYS)},
    "Fig. S16": {"A": "Every comparison with pseudobulk"},
}

# ---- titles ----

TITLE_MAX_WORDS = 15
TITLES = {
    "Fig 1": "Six single-cell-to-bulk dataset pairs and nine learned configurations compared with pseudobulk for phenotype transfer.",
    "Fig 3": "Per-assay scaling removed most of the single-cell–bulk gap; label-free alignment did not reduce it further.",
    "Fig 5": "Single-cell–bulk gene-shift agreement varied by pair; supervised set encoders detected planted structure, not mean shifts.",
    "Table 1": "Dataset pairs.",
    "Fig. S1": "Supervised set-encoder architectures and the embedding paths of the unsupervised generative and frozen foundation models.",
    "Fig. S2": "Quality control: mouse microglia, aging.",
    "Fig. S3": "Quality control: mouse organs, aging.",
    "Fig. S4": "Quality control: human blood, aging.",
    "Fig. S5": "Quality control: human blood, lupus.",
    "Fig. S6": "Quality control: human ileum, Crohn's disease.",
    "Fig. S7": "Quality control: human motor cortex, ALS.",
    "Fig. S8": "Characteristics of the dataset pairs.",
    "Fig. S9": "Transfer under alternative readouts, as differences from pseudobulk.",
    "Fig. S10": "Configuration searches and training checks for the three supervised set encoders.",
    "Fig. S11": "Robustness of the comparison with pseudobulk across seeds, normalizations, panel sizes, and subgroups.",
    "Fig. S12": "Full gene sets, native mouse genes, and more training donors, compared with pseudobulk.",
    "Fig. S13": "The modality gap in every dataset pair.",
    "Fig. S14": "Robustness of the gene lists behind Fig 5.",
    "Fig. S15": "Modality-direction genes for each learned representation.",
    "Fig. S16": "Summary of all comparisons with pseudobulk.",
    "Table S1": "Selection criteria, characteristics, and quality control of the dataset pairs.",
    "Table S2": "Transfer performance of pseudobulk, the nine learned configurations, and reference analyses.",
    "Table S3": "Transfer performance under alternative readouts.",
    "Table S4": "Configuration searches and tuning budgets.",
    "Table S5": "Robustness of the comparison with pseudobulk.",
    "Table S6": "Full gene sets, native mouse genes, and learning curves.",
    "Table S7": "Modality gap and transfer before and after alignment.",
    "Table S8": "Transfer under shared supervised set-encoder design choices.",
    "Table S9": "Genes behind transfer and the simulation.",
    "Table S10": "Sample properties captured by each configuration.",
    "Table S11": "Computational cost.",
    "Table S12": "Sample-level methods considered and the inclusion criteria.",
    "Additional file 3": "Source data for every figure panel, including per-seed values.",
    "Additional file 4": "Gene rankings, top-50 lists and attributions.",
    "Additional file 5": "Every configuration-search candidate, per seed.",
    "Additional file 6": "Per-sample predictions.",
}
# Fig 2's title: the short form of the headline sentence the precedence rule selects: "pooled" forms for K >= 2, "single" for K = 1 (no "pooled"; the pair named in
# the legend); pooled "better": one or two configurations named, three or more counted. The legend's first sentence
# completes it (FIG2_SENTENCES); the abstract carries the full sentence. The strings live only here.
FIG2_TITLES = {
    "pooled": {
        "better_one": "{a} outperformed pseudobulk at bulk transfer, pooled over dataset pairs.",
        "better_two": "{a} and {b} outperformed pseudobulk at bulk transfer, pooled over dataset pairs.",
        "better_many": "{n} configurations outperformed pseudobulk at bulk transfer, pooled over dataset pairs.",
        "within_margin": "Under the main readout, no configuration outperformed pseudobulk, with every pooled upper bound below {margin}.",
        "inconclusive": "No configuration outperformed pseudobulk, and {k} of {n} configurations {verb} inconclusive at {margin}.",
    },
    "single": {
        "better_one": "{a} outperformed pseudobulk at bulk transfer.",
        "better_two": "{a} and {b} outperformed pseudobulk at bulk transfer.",
        "better_many": "{n} configurations outperformed pseudobulk at bulk transfer.",
        "within_margin": "Under the main readout, no configuration outperformed pseudobulk, with every upper bound below {margin}.",
        "inconclusive": "No configuration outperformed pseudobulk, and {k} of {n} configurations {verb} inconclusive at {margin}.",
    },
    "not_evaluable": "Pooled claims not evaluable: pseudobulk's bulk AUROC was at least 0.95 in every pair.",
}
FIG2_VERB = {1: "is"}                  # "1 of 9 configurations is inconclusive"; otherwise "are"
FIG2_SENTENCES = {
    "better": "{configuration} outperformed pseudobulk by {delta} (95% confidence interval {low}–{high})",
    "inconclusive": "Each configuration's pooled one-sided 95% upper bound: Additional file 2: Table S2; inconclusive at {margin}: {configurations}",
    "single_pair": "single pooled pair: {pair}",
    "per_pair_better": "{configuration} outperformed pseudobulk in {pair} (per pair; not adjusted across pairs)",
}
DECIMALS = 2                           # Fig 2's title numbers (AUROC differences)
# Fig 4's title follows the per-pair best-candidate test: pairs
# named by display name, joined with "and", or counted if the names break TITLE_MAX_WORDS
FIG4_TITLES = {
    "none": "No design choice of the supervised set encoders significantly outperformed pseudobulk in any pair.",
    "pairs": "A supervised set-encoder design choice significantly outperformed pseudobulk in {pairs}.",
    "count": "A supervised set-encoder design choice significantly outperformed pseudobulk in {n} of {total} dataset pairs.",
}
FIG4_NOT_FORMED = "The best-candidate test is not formed in {pairs}."
# Fig 4's test sentence: the no-rejection form, or per group of rejecting pairs whether the candidate
# carrying the max statistic (11b's notes, best_candidate carrier) is the unit's selected configuration
FIG4_TEST = {"none": "No candidate significantly exceeded pseudobulk in any pair (one-sided max-statistic bootstrap).",
             "pairs": "In {pairs}, {which} candidate significantly exceeded pseudobulk (one-sided max-statistic bootstrap).",
             "selected": "the selected", "unselected": "an unselected"}
# Fig 6's title: its Results heading, whichever of config/properties.LICENSED_TITLE / DESCRIPTIVE_TITLE
# stage 12a wrote into licence.json (figures/titles.fig6 still checks that file)
FIG6_TITLE = "Frozen foundation-model cell embeddings encoded depth and other technical properties beyond pseudobulk in every pair."
FIG6_NAMED = "Configurations above both reference rows on every technical property in at least {n} pairs: {names}."
FIG5_CRITERION = "Simulation criterion (structure only, margin {margin}): {result}; per supervised set encoder: {per_encoder}."

# Conditional sentences and drawn headers
FIGURE_TEXT = {
    # the pooled pairs' mark (a diamond: the figure fonts have no star, and "*" reads as significance)
    "pooled_mark": "♦",
    "pooled_header": "Average\n({mark} datasets)",
    "pooled_header_k1": "Single dataset\n({mark})",
    "pooled_k0": "No pair below {threshold}: no pooled average",
    "pooled_short": "Average\n({mark} datasets)",   # Figs. S9-S12's pooled column title
    "pooled_short_k1": "Single dataset\n({mark})",
    "pooled_strip": "Pseudobulk bulk AUROC",   # Fig 2A's strip above the columns
    "fig4a_fm_cut": "Frozen foundation models: up to 1,000 cells.",
    "fig4a_scgpt_only": "Geneformer: up to 1,000 cells.",     # the partial fallback (scGPT keeps its "all" point)
    "fig4a_not_estimable": "n.e.: not estimable (CytoSet, \"all\"; the age-similarity loss exceeds memory there, Methods).",
    "fig5d_per_sample": "Its mean-shift value is not read as mean-signal detection (Methods).",   # iff a set encoder uses per-sample normalization
    "not_estimable": "n.e.",
    "not_applicable": "n/a",
    "not_affordable": "not computed (computational cost)",
    "pending": "n/a (pending)",                                 # the Geneformer deletion not yet run; never final
    "coordinates_pending": "coordinates not yet saved",   # Fig 3a, S13a-e: a row outside config/geometry.CONFIGURATIONS
    "selected": "selected",
    "not_drawn": "Not drawn in a blinded run (needs stage 11/11b tables; frozen §29.2)",
    "oracle": "best on bulk (for reference only, never selected)",
}

# stage 10's inclusion code for a dropped Geneformer deletion -> its FIGURE_TEXT
DROPPED_TEXT = {"triaged": "not_affordable", "pending": "pending"}

# ---- what each figure draws ----

# Fig 1C: the ten configurations as a model x input grid (the design: drawn from config, no stage output)
FIG1C_INPUTS = ("cells", "pseudobulk")
# The dataset pairs (Table 1; Fig 1D draws the groups): per pair and side, (tissue or preparation, assay, groups compared,
# units spelled out). From config/pairs/<pair>.py
# ("single_cell" and "bulk" sources; "groups" and its unit comment) and config/fig1e.FIG1E's tissue annotations.
FIG1D = {
    "hammond": {"single_cell": ("whole-brain microglia", "scRNA-seq", "≤ 30 days vs. 100–540 days"),   # from E14.5 (-4.5 days)
                "bulk": ("hippocampal microglia", "bulk", "9 days vs. 200 days")},
    "tms": {"single_cell": ("4 organs, dissociated", "scRNA-seq", "1–3 vs. 18–30 months"),   # single-cell ages 1, 3, 18-30 months
            "bulk": ("4 organs, whole", "bulk", "≤ 6 vs. 18–30 months")},           # the groups' ranges (0-6, 18-30)
    "soundlife": {"single_cell": ("PBMC", "scRNA-seq", "25–37 vs. 55–67 years"),
                  "bulk": ("whole blood", "bulk", "25–37 vs. 55–67 years")},
    "lupus": {"single_cell": ("PBMC", "scRNA-seq", "control vs. lupus"),
              "bulk": ("whole blood", "bulk", "control vs. lupus")},
    "crohns": {"single_cell": ("ileal tissue", "scRNA-seq", "control vs. Crohn's"),   # non-IBD controls
               "bulk": ("ileal biopsy", "bulk", "control vs. Crohn's")},
    "pineda": {"single_cell": ("motor cortex", "snRNA-seq", "control vs. ALS"),
               "bulk": ("motor cortex", "bulk", "control vs. ALS")},
}
# Fig 1E's columns, in order -> header: 04d's characteristics (config/fig1e; S8 Fig shows all of them) and the typed-in
# FIG1E_TYPED
FIG1E_PANEL = {"Many cell types": "Many cell types", "Tissue or preparation differs between sides": "Tissue differs",
               "Phenotype defined differently": "Phenotype differs", "Independent cohorts": "Independent cohorts"}
FIG1E_TYPED = {"Phenotype defined differently": ("hammond", "tms", "lupus")}   # column -> the pairs marked yes
# Fig 1E's headers: the thresholds are in the legend, not the header
FIG1E_HEADERS = {"Few training donors (< 100)": "Few donors",   # two lines each; full wording in the legend
                 "Tissue or preparation differs between sides": "Tissue differs",
                 "Independent cohorts": "Indep. cohorts",
                 "Batch confounded with label": "Batch confounded"}
FIG1_FONT_SIZE = 7                     # Fig 1's text at print size: a target; style.MIN_FONT_SIZE is the floor
FIG2_FONT_SIZE = 7                     # Fig 2's tick labels, titles and key (the 7-8 pt target)
# Figs 3-6 and S9-S16 (layout pass, 2026-10-09): tick labels, titles, keys and
# annotations at 7 pt (axis labels stay at style.FONT_SIZE); dense labels (gene names, heatmap cells) at style.MIN_FONT_SIZE
TEXT_SIZE = 7
NARROW_TITLE = 15                      # characters per line of a pair title over a narrow column (plots.narrow)
TEXT_RC = {"xtick.labelsize": TEXT_SIZE, "ytick.labelsize": TEXT_SIZE, "legend.fontsize": TEXT_SIZE, "axes.titlesize": TEXT_SIZE}
# Short display names where a full configuration label doesn't fit a category axis (Fig 5D)
SHORT_LABELS = {"pseudobulk": "Pseudobulk", "mean_log_cells": "Mean of\nlog-\nnormalized\ncells", "cytoset_cells": "CytoSet",
                "set_transformer_cells": "Set\nTransformer", "scrat_cells": "ScRAT"}
FIG4_OPTION_LABELS = {"per_sample": "per-sample"}   # Fig 4b/c row labels; other options as stage 05 names them
FIG4_NATIVE_POOLING = {"set_transformer": "PMA", "scrat": "mean"}   # Fig 4b: option "native"
S10E_CANDIDATE_LABELS = {"base": "base", "latent20": "latent 20", "latent30": "latent 30",
                         "layers2": "2 layers"}   # Fig. S10e ticks: search_space.SCVI_GROUPS' candidates
PROPERTY_SHORT = {"depth": "Depth", "mito_fraction": "Mito.", "composition_pc1": "Comp.", "phenotype": "Pheno."}   # Fig 6A columns
FEATURE_LABELS = {"tmm": "TMM", "vst": "VST", "panel_1000": "1,000-gene panel", "panel_2000": "2,000-gene panel",
                  "all_shared": "All shared genes"}   # Fig. S11c/d rows
STABILITY_LABELS = {"stability_seeds": "across seeds", "stability_split_half": "seeds 0–9 vs. 10–19",
                    "stability_donors": "donor halves", "stability_folds": "across folds"}   # Fig. S14a, top to bottom in a row
# Fig. S16's columns (stage 11's and 11b's row names); a row not listed is shown by its name
S16_COLUMN_LABELS = {"primary (ridge)": "Primary (ridge)", "primary (lasso)": "Primary (LASSO)", "age_adjusted": "Age-adjusted",
                     "alignment_combat": "ComBat", "alignment_coral": "CORAL", "alignment_harmony": "Harmony",
                     "alignment_unaligned_stacked": "One scaler, both sides", "feature_variants (full_genes)": "Full gene set",
                     "learning_curve_25": "25% of donors", "learning_curve_50": "50% of donors", "learning_curve_100": "100% of donors",
                     "native_head": "CytoSet's own head", "numeric_age": "Numeric age", "one_type": "One cell type",
                     "out_of_fold": "Out-of-fold", "outside_moments": "Bulk scaled by other bulk", "passing_seeds": "Passing seeds only",
                     "per_sample_route": "Per-sample normalization", "pooled_predictions": "Folds pooled (blood, aging)",
                     "readout_elasticnet": "Elastic net", "readout_knn": "k-NN", "readout_mlp": "MLP", "readout_pls": "PLS",
                     "scvi_observed_library": "scVI, unthinned bulk", "single_cell_moments": "Bulk scaled by single cells", "split67": "67/29 donor split",
                     "thinned_bulk": "Thinned bulk"}   # leave_one_pair_out: pooled only, an unstreamed row (s17.LEFT_OUT)

# Fig 2A's ✕ rows in the captions' order; "label_permuted" is pseudobulk's label-permuted row (stage 11b)
LABEL_PERMUTED = "label_permuted"
FIG2A_REFERENCE_ROWS = ("pseudobulk_pca64", "cytoset_untrained", LABEL_PERMUTED)
assert set(FIG2A_REFERENCE_ROWS) <= set(configurations.REFERENCE_ROWS) | {LABEL_PERMUTED}
# Display labels that override configurations.py's (which is inside stage code hashes) in every figure, key and tick label:
# short forms so Fig 2a's row labels fit the label column; plain names for the configurations with one
# input, so only CytoSet, Geneformer and scGPT carry ", cells" / ", pseudobulk"; the reference row's full name (2026-11-04)
ROW_LABELS = {"deconvolution_clr": "Deconvolution (fractions)", LABEL_PERMUTED: "Label-permuted",
              "set_transformer_cells": "Set Transformer", "scrat_cells": "ScRAT", "scvi_cells": "scVI",
              "mean_log_cells": "Mean of log-normalized cells"}
FIG2C_MODELS = ("cytoset", "geneformer", "scgpt")
FIG3_KEYS = ("pseudobulk", "cytoset_cells", "scgpt_cells")   # Fig 3a's rows and b and c's configurations
FIG3A = {"pair": "crohns", "columns": ("unaligned_stacked", "none", "combat", "harmony")}
ALIGNMENT_COLUMN_LABELS = {"unaligned_stacked": "Across-assay standard scaling", "none": "Within-assay standard scaling",
                           "combat": "ComBat", "harmony": "Harmony"}
ALIGNMENT_COLUMN_SHORT = {"unaligned_stacked": "Across-\nassay", "none": "Within-\nassay", "combat": "ComBat",
                         "harmony": "Harmony"}   # Fig. S13a-e's narrower columns
FIG3_CONDITIONS = ("unaligned_stacked", "none", "combat", "harmony")   # Fig 3b and c, left to right in each configuration
S13_CONDITIONS = ("unaligned_stacked", "none", "combat", "coral", "harmony")   # Fig. S13f-i
# Fig. S13f-i's configurations by input, pseudobulk first in each as the reference: f and g on cells, h and i on pseudobulk
S13_KEYS = {"cells": tuple(k for k in DISPLAY_ORDER if k == "pseudobulk" or configurations.CONFIGURATIONS[k]["input"] == "cells"),
            "pseudobulk": tuple(k for k in DISPLAY_ORDER if configurations.CONFIGURATIONS[k]["input"] == "pseudobulk")}
# Fig 4A's foundation-model "all" points (stage 07d): ("scgpt",) under the partial fallback, () if cut.
# Fig 4 asserts the scores table agrees both ways.
FIG4A_FM_ALL_AVAILABLE = ("geneformer", "scgpt")
FIG4_SET_ENCODERS = ("cytoset", "set_transformer", "scrat")
FIG4_GROUPS = {"B": {"cytoset": "out_pool"}, "C": {e: "cell_normalization" for e in FIG4_SET_ENCODERS}}   # the one-change groups
FIG4_BASE_VALUE = {"B": {"cytoset": "mean", "set_transformer": "native", "scrat": "native"}, "C": {e: "none" for e in FIG4_SET_ENCODERS}}
FIG6B_PAIR = "crohns"                  # the left exemplar: Fig 3A's pair, for continuity
FIG6B_TOP = 10                         # Fig 6b left and Fig. S15 (15 overlapped at 7 pt; layout round 3, 2026-10-09)
FIG6A_ROWS = ("pseudobulk", "pseudobulk_pca64")   # above the separator; the nine others follow
PHENOTYPE_RANGE = (0.5, 1.0)           # Fig 6A's phenotype column scale

# The named cells drawn "n/a" instead of failing: item panel -> {(pair, column or key)}
NOT_APPLICABLE = {
    "Fig 6A": {(p, "mito_fraction") for p in properties.MITO_NA_PAIRS}
              | {(p, "composition_pc1") for p in PAIR_ORDER if deconvolution.LEVELS[p] is None},
    "Fig 6C": {(p, "all") for p in PAIR_ORDER if "one_type" not in PAIRS[p]},
    "Fig. S12C": {(p, "all") for p in PAIR_ORDER if p not in training.LEARNING_CURVE_PAIRS},
    "Fig. S12A": {(p, "pseudobulk") for p in PAIR_ORDER if PAIRS[p]["species"] == "human"},   # mapped orthologs: mouse pairs only
    "Fig. S10C": {(p, "objective_mse") for p in PAIR_ORDER if p not in training.AGE_LOSS_PAIRS},   # CytoSet's MSE objective: aging pairs only
}
# Fig 5D's values that hold by construction (simulation.BY_CONSTRUCTION): drawn as references, never results.
# (scenario, key or None = every row, column)
BY_CONSTRUCTION = {("structure_only", "pseudobulk", "within_sc_auroc"), ("structure_only", None, "simulated_bulk_auroc"),
                   ("mean_shift", "pseudobulk", "simulated_bulk_auroc"), ("mean_shift", "pseudobulk", "within_sc_auroc")}

# Difference-axis limits (cosmetic; set from the rehearsal's PNGs, then the real numbers). CI ends past a limit are
# clipped with an arrowhead; in every difference panel (Figs 2, 4, S9-S12) a point estimate outside is drawn as below. Fig 2's
# range is pre-declared: -0.3 to +0.15 shows the +0.05 margin's edge; there a point estimate
# outside is drawn as an arrowhead at the edge, a seed dot outside is left out and a bar is cut at the edge with an
# arrowhead (the panel CSVs keep every value; the legend states it).
# Figs 4 and S9-S12 use Fig 2's pre-declared range and clip the same way (layout pass round 2, 2026-10-09); Fig. S16's
# diverging color scale is symmetric about 0, with extend arrows past it.
DIFFERENCE_RANGE = (-0.3, 0.15)
XLIM_DIFFERENCE = {"Fig 2A": DIFFERENCE_RANGE, "Fig 4": DIFFERENCE_RANGE, "Fig. S9": DIFFERENCE_RANGE, "Fig. S10": DIFFERENCE_RANGE,
                   "Fig. S11": DIFFERENCE_RANGE, "Fig. S12": DIFFERENCE_RANGE, "Fig. S16": (-0.3, 0.3)}
SILHOUETTE_TOP = 1.0                   # Fig 3b, S13f, h: the modality silhouette's maximum (room for the condition names)
AUROC_RANGE = (0.5, 1.02)              # Fig 2b, 5c and 6c's shared AUROC axis; a value below is an arrowhead at the bottom edge
HOLM_GLYPH = {"better": "↑", "worse": "↓"}
HOLM_ALPHA = statistics.ALPHA

# Blinded runs: the panels whose inputs a blinded run legitimately writes (label-free stages, 10c's
# label-bearing outputs, and panels that read the bulk labels themselves). Every other panel needs stage 11/11b tables
# and is exercised in pytest only. A listed panel with a missing input fails.
BLINDED_PANELS = {
    "Fig 1": tuple(PANELS["Fig 1"]), "Fig. S1": tuple(PANELS["Fig. S1"]),
    "Fig 3": ("A", "B"), "Fig 5": ("A", "B", "D"), "Fig 6": ("A", "B"),
    **{item: tuple(PANELS[item]) for item in QC_FIGURES.values()},
    "Fig. S8": tuple(PANELS["Fig. S8"]), "Fig. S10": ("A", "D", "E"), "Fig. S11": ("A",),
    "Fig. S13": tuple(p for p in PANELS["Fig. S13"] if p not in ("G", "I")), "Fig. S14": tuple(PANELS["Fig. S14"]),
    "Fig. S15": tuple(PANELS["Fig. S15"]),
}

# Drawn text (axis labels, headers)
TEXT = {
    "difference": "Δ bulk AUROC vs. pseudobulk",
    "bulk_auroc": "Bulk AUROC",
    "auroc": "AUROC",
    "within": "single-cell",
    "bulk": "bulk",
    "silhouette": "Modality silhouette",
    "silhouette_short": "Silhouette",   # Fig 3a and Fig. S13a-e, under each plot
    "cells_per_bag": "Cells per bag (log scale; embedding averaged over 10 bags)",
    "stability_order": "Within each row, top to bottom: {checks}",   # Fig. S14a
    "chance_row": "Chance (95th pct.)",   # Fig 6a
    "groups_by_x": "Configuration (control / younger left, case / older right)",   # Fig. S11a
    "encoder_axis": "Supervised set encoder (candidates spread by value)",   # Fig. S10a
    "age_adjusted": "age-adjusted",   # Fig. S11f
    "within_auroc": "Within single-cell AUROC",
    "raw_to_adjusted": "Unadjusted → adjusted",
    "difference_short": "Δ bulk AUROC",
    "top_k": "Top {k}",   # Fig. S14b
    "lasso_top": "LASSO, top 50",   # Fig. S14c
    "shared_share_k": "Same-sign share",
    "all_point": "all",
    "single_cell_shift": "Single-cell shift",
    "bulk_shift": "Bulk shift",
    "shared_share": "Same-sign share (top 50)",
    "rho": "ρ = {rho:.2f}",
    "simulated_bulk": "simulated bulk",
    "level_difference": "bulk − single-cell",   # Fig 6b, S15: the number column's header
    "attribution": "Attribution",
    "modality_share": "Share of top genes: large difference or marker",
    "pc1": "PC1",
    "pc2": "PC2",
    "umap1": "UMAP1",
    "umap2": "UMAP2",
    "validation_bce": "Validation BCE (log scale)",   # Fig. S10a, b
    "validation_elbo": "Validation ELBO",
    "d_star": "d*",
    "encoder_seed": "Encoder seed",
    "fraction": "Training donors (%)",
    "count": "Count",
    "fraction_of_cells": "Fraction of cells",
    "score": "Bulk score",
    "seed": "seed",
    "ensemble": "seed ensemble",
    "by_construction": "holds by construction",   # Fig 5d's gray marks
    "base_all": "all panel genes",          # Fig 5b, S14b/c reference ticks
    "base_matched": "matched on single-cell shift",
    "pseudobulk_within_assay": "pseudobulk, within-assay",   # Fig 3c and Fig. S13g, i's dashed line (the key)
    "scale_r2": "R² (properties)",                    # Fig 6A's scale bars: the columns mix R² and AUROC
    "scale_phenotype": "AUROC (phenotype)",   # Fig 3c and S13f's dashed line: pseudobulk under the standard per-side standardization
    "pseudobulk_bulk": "pseudobulk's bulk AUROC",   # Fig 6c's dashed line (caption: "as in Fig 2b")
    "objective_mse": "CytoSet, MSE",   # Fig. S10c: CytoSet trained with the MSE objective (wider labels clip at left=0.12)
    "delta_elbo": "Δ validation ELBO\nvs. base",   # Fig. S10e
    "stability": "Top-50 overlap",
    "passed": "passed",
    "failed": "failed",
    "group": {0: "control / younger", 1: "case / older"},
    "side": {"single_cell": "single-cell", "bulk": "bulk"},
    "scenario": {"structure_only": "Signal in cell-level structure only", "mean_shift": "Signal in mean expression"},
    "simulation_sides": {"within_sc_auroc": "single-cell", "simulated_bulk_auroc": "simulated bulk"},   # Fig 5D: the first pair of points
    "readout": {"lasso": "LASSO", "elasticnet": "Elastic net", "pls": "Partial least squares", "knn": "k-nearest neighbors",
                "mlp": "Multilayer perceptron", "native_head": "CytoSet's own classification head"},
    "chance_line": "chance",
    "p_value": "p = {p:.2f}",
    "p_value_holm": "p = {p:.2f} (Holm {holm:.2f})",
    "reference_row": "reference row",
    "baseline_control": "baseline or control",   # Fig 2a's gray rows (key)
    "margin_band": f"> {statistics.MARGIN:g} above pseudobulk",   # Fig 2a's shading (key)
    "hollow": "hollow = pseudobulk input",
    "ring": "ring = selected",
    "within_bulk": "within-bulk reference",
    "within_bulk_fig2": "within-bulk reference\n(ridge trained on bulk)",   # Fig 2's key and panel b's line label
    "pseudobulk_bulk": "pseudobulk",                # Fig 2b: the dashed line's name at the right
    "pseudobulk_zero": "pseudobulk (Δ = 0)",   # the line is 0 on a difference axis
    "pseudobulk_fig2": "pseudobulk (Δ = 0 in a, c;\nits bulk AUROC in b)",   # Fig 2's one dashed line, both meanings
    "pseudobulk_n": "pseudobulk of the same cells",
    "log1p_cpm": "Mean log1p CPM",
    "fig1c_inputs": {"cells": "Cells", "pseudobulk": "Pseudobulk"},
    "fig1c_header": "Model input",
    "absent": "—",
    "fig1d_cohorts": {"single_cell": "Single-cell cohort", "bulk": "Bulk cohort"},
    "fig1d_counts": "n = {a} vs. {b} {unit}",
    "fig1d_units": {"donor": "donors", "sample": "samples"},
}

# ---- the data files' donor-level rule ----

# Donor-level columns that may appear in the data files (Additional files 3-6) only where a figure plots or a test uses them (and then only in that
# panel's CSV, Additional file 3)
SENSITIVE_COLUMNS = ("age", "age_at_death", "age_bin", "label_value", "numeric", "sex", "pmi_hours", "pmi_raw", "cmv", "litter",
                     "title", "treatment", "nygc_group", "age_group")
PANEL_SENSITIVE = {"Fig. S4I": ("cmv",), "Fig. S6I": ("age",), "Fig. S7H": ("sex", "age_at_death", "pmi_hours")}

# ---- the citation plan (manuscript reading order; Genome Biology: Background, Results, Discussion, Conclusions,
# Methods, then the legends) ----

# (section, citing context, item). Numbering (per type) and the Additional-file sequence follow first citation in
# the main body (not the legends). The data files (Additional files 3-6) are first cited in the Methods' data paragraph, in
# order, in the Methods' overview (their legends' citations don't count).
CITATIONS = (
    ("Results", "Fig 1A", "Fig 1"),
    ("Results", "Fig 1B", "Fig. S1"),
    ("Results", "Fig 1D", "Table 1"),
    *(("Results", "Fig 1D", item) for item in QC_FIGURES.values()),
    ("Results", "Fig 1E", "Fig. S8"),
    ("Results", "Fig 1E", "Table S1"),
    ("Results", "Fig 2A", "Fig 2"),
    ("Results", "Fig 2A", "Table S2"),
    ("Results", "Fig 2", "Fig. S9"),
    ("Results", "Fig 2", "Table S3"),
    ("Results", "Fig 2", "Fig. S10"),
    ("Results", "Fig 2", "Table S4"),
    ("Results", "Fig 2", "Fig. S11"),
    ("Results", "Fig 2", "Table S5"),
    ("Results", "Fig 2", "Fig. S12"),
    ("Results", "Fig 2", "Table S6"),
    ("Results", "Fig 3", "Fig 3"),
    ("Results", "Fig 3", "Fig. S13"),
    ("Results", "Fig 3", "Table S7"),
    ("Results", "Fig 4", "Fig 4"),
    ("Results", "Fig 4", "Table S8"),
    ("Results", "Fig 5", "Fig 5"),
    ("Results", "Fig 5", "Fig. S14"),
    ("Results", "Fig 5", "Table S9"),
    ("Results", "Fig 6", "Fig 6"),
    ("Results", "Fig 6B", "Fig. S15"),
    ("Results", "Fig 6", "Table S10"),
    ("Discussion", "", "Fig. S16"),
    ("Discussion", "", "Table S11"),
    *(("Methods", "Overview", f"Additional file {i}") for i in range(3, 7)),
    ("Methods", "", "Table S12"),
)
SECTIONS = ("Background", "Results", "Discussion", "Conclusions", "Methods")
FIRST_CITED_IN = {"Table S12": "Methods"}   # if the Background cites it, it becomes Table S1
# The legends' own order rules (Fig 2's legend cites Figs. S9 -> S10 -> S11 -> S12)
LEGEND_ORDER = {"Fig 2": ("Fig. S9", "Fig. S10", "Fig. S11", "Fig. S12")}
# The Results cite Tables S3–S6 in order
RESULTS_ORDER = ("Table S3", "Table S4", "Table S5", "Table S6")
# Additional files (Genome Biology: numbered by first main-text citation): every supplementary figure in
# one PDF and every supplementary table in one .xlsx workbook; each data file is its own Additional file (its key's
# number). citation_order.check fails if first citation would number them otherwise.
ADDITIONAL_FILE = {"s_fig": 1, "s_table": 2}
PANEL_LABEL_CASE = "lower"             # drawn letters and generated text: a, b, c; file names and
                                       # registry keys keep capitals (Fig1C.png, S8AFig.csv, PANELS)

# Table 1 (figures/table1.py): per pair and side, (short citation, short accession). The accession's
# id (before any comma) appears in config/pairs' accession; a version, in config/sources (tests/test_table1.py). Tissue, assay and groups come from FIG1D.
TABLE1 = {
    "studies": {
        "hammond": {"single_cell": ("Hammond et al. 2019", "GSE121654"), "bulk": ("Reemst et al. 2022", "GSE207067")},
        "tms": {"single_cell": ("Tabula Muris Consortium 2020", "CELLxGENE 0b9d8a04"), "bulk": ("Schaum et al. 2020", "GSE132040")},
        "soundlife": {"single_cell": ("Gong et al. 2025", "doi:10.57785/bn17-3h16"),
                      "bulk": ("Gong et al. 2025; Smithmyer et al. 2025", "doi:10.57785/bn17-3h16")},
        "lupus": {"single_cell": ("Perez et al. 2022", "CELLxGENE 218acb0f"), "bulk": ("Hung et al. 2015", "GSE72509")},
        "crohns": {"single_cell": ("Alegbe et al. 2026; Krzak et al. 2026", "CELLxGENE 0e5c75da, version e3460364"),
                   "bulk": ("Argmann et al. 2023", "GSE193677")},
        "pineda": {"single_cell": ("Pineda et al. 2024", "syn51105515"), "bulk": ("Prudencio et al. 2020", "GSE153960")},
    },   # every dataset is open; the NYGC ALS Consortium is credited in the Methods
    # Genome Biology: footnotes as superscript lower-case letters (Unicode superscripts), a, b, c, d in order of appearance
    "header_marks": {"Groups compared (n per group)": "ᵃ", "Samples (donors)": "ᵇ", "Cells per sample, median (range)": "ᶜ"},
    "group_marks": {"tms": "ᵈ"},   # on both sides' groups cells: the two sides' age groups differ
    "sides": {"single_cell": "Single-cell", "bulk": "Bulk"},
    "columns": ("Pair", "Side", "Study (accession)", "Tissue / preparation", "Assay", "Groups compared (n per group)",
                "Samples (donors)", "Cells per sample, median (range)"),
    "footnotes": ("ᵃn per group counts donors in the pairs scored by donor ({donor_pairs}) and samples in the others.",
                  "ᵇDonors or mice are given where a donor or mouse contributes more than one sample.",
                  "ᶜCells per sample after quality control; at most {max_cells} per sample were stored for modeling. In human blood, "
                  "aging, cells are counted in each sample's random 3000-cell subsample, drawn before quality control "
                  "(Additional file 1), hence the maximum of 3000.",
                  "ᵈIn mouse organs, aging, the age groups differ between sides: 1–3 vs. 18–30 months on the single-cell side "
                  "and ≤ 6 vs. 18–30 months on the bulk side."),
}
