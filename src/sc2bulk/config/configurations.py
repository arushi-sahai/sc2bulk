"""The configurations compared in every figure: pseudobulk; each set encoder and foundation
model applied to a sample's cells; CytoSet and the two foundation models also applied to its
pseudobulk profile. The same key, label, color, marker and order everywhere
(palette decided 2026-10-05).

Hue = model family: set encoders warm (CytoSet, Set Transformer, ScRAT), foundation models
blue (Geneformer, scGPT), the generative model purple (scVI), pseudobulk green. Within a
family the models differ in lightness, and every model has its own marker shape, so the
palette survives the three common color-vision deficiencies (minimum CIELAB distance after
Machado et al. 2009 simulation: 16) and grayscale (by marker). Pattern = input: "cells" is
solid; "pseudobulk" is hatched (bars) or hollow (markers) in the same hue. Markers were
chosen to stay distinct at 8 pt in grayscale: o, s, D, ^, v, p, *; reference
rows use a black X.

Reference rows are not configurations; they have their own keys, labels and a neutral
style (config/style.py)."""

MODELS = {
    "pseudobulk":      {"label": "Pseudobulk",      "family": "pseudobulk",        "color": "#009E73", "marker": "o"},
    "cytoset":         {"label": "CytoSet",         "family": "set_encoder",       "color": "#D55E00", "marker": "s"},
    "set_transformer": {"label": "Set Transformer", "family": "set_encoder",       "color": "#7A2E00", "marker": "D"},
    "scrat":           {"label": "ScRAT",           "family": "set_encoder",       "color": "#F5B041", "marker": "^"},
    "geneformer":      {"label": "Geneformer",      "family": "foundation_model",  "color": "#003F7F", "marker": "v"},
    "scgpt":           {"label": "scGPT",           "family": "foundation_model",  "color": "#56B4E9", "marker": "p"},
    "scvi":            {"label": "scVI",            "family": "generative",        "color": "#CC79A7", "marker": "*"},
}

CONFIGURATION_ORDER = (
    "pseudobulk",
    "cytoset_cells",
    "cytoset_pseudobulk",
    "set_transformer_cells",
    "scrat_cells",
    "geneformer_cells",
    "geneformer_pseudobulk",
    "scgpt_cells",
    "scgpt_pseudobulk",
    "scvi_cells",
)


def _configuration(model, input_):
    label = MODELS[model]["label"] if model == "pseudobulk" else f"{MODELS[model]['label']}, {input_}"
    return {"label": label, "model": model, "input": input_,
            "color": MODELS[model]["color"], "marker": MODELS[model]["marker"]}


CONFIGURATIONS = {key: _configuration(*(("pseudobulk", "pseudobulk") if key == "pseudobulk" else key.rsplit("_", 1)))
                  for key in CONFIGURATION_ORDER}

# Reference rows (Fig 2A/2C and S2 Table), drawn in the neutral reference style.
REFERENCE_ROWS = {   # each with the model behind it and its input (schemas check them)
    "pseudobulk_pca64":      {"label": "Pseudobulk, PCA-64",                  "model": "pseudobulk", "input": "pseudobulk"},
    "mean_log_cells":        {"label": "Mean of log cells",                   "model": "none",       "input": "cells"},
    "per_cell_classifier":   {"label": "Per-cell classifier",                 "model": "none",       "input": "cells"},
    "cytoset_untrained":     {"label": "CytoSet, untrained",                  "model": "cytoset",    "input": "cells"},
    "deconvolution_clr":     {"label": "Deconvolution (estimated fractions)", "model": "none",       "input": "none"},
    "deconvolution_true":    {"label": "Deconvolution (annotated fractions)", "model": "none",       "input": "none"},
    "known_signature":       {"label": "Known-signature score",               "model": "none",       "input": "none"},
    "within_bulk_reference": {"label": "Within-bulk reference",               "model": "none",       "input": "none"},
}

READOUTS = ("ridge", "lasso", "elasticnet", "pls", "knn", "mlp")   # ridge primary, LASSO secondary
ALIGNMENTS = ("none", "combat", "coral", "harmony")
ALIGNMENT_LABELS = {"none": "None", "combat": "ComBat", "coral": "CORAL", "harmony": "Harmony"}

# How the foundation models pool their cell embeddings into one per sample. Mean in every
# headline figure; max only as a Fig 4B variant.
FM_HEADLINE_POOLING = "mean"
