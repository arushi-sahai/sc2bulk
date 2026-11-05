"""Fig 1E and the dataset characteristics (stage 04d_characteristics; S8 Fig, S1 Table).

Kept out of config/pairs, which is hashed into every stage of its pair (sc2bulk.codehash).
The computed columns' rules below are pre-declared and fixed before 04d runs.
"""

# The three columns marked "computed" are filled from the processed data by stage 04d, never
# typed in.
FIG1E_COLUMNS = (
    "Many cell types",               # computed: >= MANY_TYPES_MIN types, each >= MANY_TYPES_FRACTION of training cells
    "Tissue or preparation differs between sides",
    "Single-nucleus → bulk",
    "Independent cohorts",
    "Non-age phenotype",
    "Several tissues",
    "Few training donors (< 100)",   # computed: fewer than FEW_TRAINING_DONORS training donors (mice for mouse pairs)
    "No shared label values",
    "Batch confounded with label",   # computed from single-cell metadata
)
FEW_TRAINING_DONORS = 100
# Fig 1E's five main-figure columns (S8 Fig shows all of them)
FIG1E_MAIN = ("Many cell types", "Tissue or preparation differs between sides", "Independent cohorts",
              "Batch confounded with label", "Few training donors (< 100)")
# Typed-in cells: column -> (state, annotation), state "yes", "no" or "not determinable".
# Cells not listed are "no". Computed columns ("Many cell types", "Few training donors (< 100)",
# "Batch confounded with label") are filled from the data, never typed here.
# "Tissue or preparation differs between sides": yes when the sides come from
# different tissues (anatomical region included), or the bulk is whole tissue while the single-cell
# side is dissociated cells (cell types lost in dissociation are in bulk only). Isolation protocols
# are not part of the rule; a difference is mentioned in the annotation only.
FIG1E = {
    'hammond': {'Independent cohorts': ('yes', ''), 'No shared label values': ('yes', ''),
                'Tissue or preparation differs between sides': ('yes', 'brain region: whole brain vs. hippocampus (FACS-sorted microglia on both sides)')},
    'tms': {'Several tissues': ('yes', '4 organs'), 'Independent cohorts': ('no', '(same colony)'),
            'Tissue or preparation differs between sides': ('yes', 'whole organ vs. dissociated cells')},
    'soundlife': {'Tissue or preparation differs between sides': ('yes', 'PBMC vs. whole blood'), 'Independent cohorts': ('no', '(same donors, cross-fitted)')},
    'lupus': {'Tissue or preparation differs between sides': ('yes', 'PBMC vs. whole blood'), 'Independent cohorts': ('yes', ''), 'Non-age phenotype': ('yes', 'SLE')},
    'crohns': {'Independent cohorts': ('yes', ''), 'Non-age phenotype': ('yes', "Crohn's"),
               'Tissue or preparation differs between sides': ('yes', 'biopsy vs. dissociated cells')},
    'pineda': {'Single-nucleus → bulk': ('yes', ''), 'Independent cohorts': ('yes', ''), 'Non-age phenotype': ('yes', 'ALS'),
               'Tissue or preparation differs between sides': ('no', 'same tissue; nuclei isolated from tissue, not dissociated cells; nuclear vs. whole-cell RNA: see "Single-nucleus → bulk"')},
}

# "Many cell types": at least MANY_TYPES_MIN annotated types, each at least MANY_TYPES_FRACTION of
# the primary unit's post-QC training cells, at the pre-declared annotation level. Mouse organs
# count per organ; the pair is "yes" if most organs (more than half) qualify.
MANY_TYPES_MIN = 3
MANY_TYPES_FRACTION = 0.05
ANNOTATION = {"hammond": "cell_type",      # one annotated type (microglia), so "no"
              "tms": "cell_type", "soundlife": "AIFI_L1", "lupus": "cell_type", "crohns": "cell_type",
              "pineda": "cell_type"}       # the authors' 24 types

# "Batch confounded with label": bias-corrected Cramer's V (Bergsma 2013) between the batch and the
# 0/1 group over the primary unit's single-cell training samples, at least BATCH_V. None = no
# single-cell processing-batch variable recorded ("not determinable"). Lupus's pool is the
# connected component of the donor-library graph (each donor's cells span several libraries),
# read from the raw CELLxGENE obs; its processing cohort is constant (4) by the pair's restriction.
BATCH_V = 0.5
BATCH = {"hammond": "chemistry", "tms": "release_cohort", "soundlife": "batch_id", "lupus": "pool",
         "crohns": None, "pineda": "run"}
NOT_DETERMINABLE = {"crohns": "no single-cell processing-batch variable recorded; reagent lots are proxies (annotated)"}

# Tissue- and preparation-difference markers (S8 Fig; pre-declared 2026-10-06): mean
# log1p CPM on each side, per pair, for the cell types the "yes" rows name. Each entry: marker ->
# (genes, organ the samples are restricted to, or None).
BLOOD = {"neutrophil": (("CSF3R", "FCGR3B", "CXCR2"), None),
         "erythrocyte": (("HBB", "HBA1", "HBA2"), None),        # human blood, aging bulk is globin-depleted
         "platelet": (("PPBP", "PF4"), None)}
TISSUE_MARKERS = {
    "soundlife": BLOOD,
    "lupus": BLOOD,
    "crohns": {"stroma and muscularis": (("COL1A1", "DES", "ACTG2"), None)},
    "tms": {"myofiber": (("Acta1", "Ckm", "Myh4"), "Limb_Muscle"),
            "lung epithelium": (("Ager", "Sftpc"), "Lung")},
}
