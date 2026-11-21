"""Additional file 2: Tables S1–S12 (figures/text/legends.md "## Additional file 2: supplementary tables"). Table 1 is
figures/table1.py's (label-free; run separately).

One function per table, each returning {sheet: (subtitle, frame, notes)}: a sheet per sub-table, its notes printed under
it. Sources, read only through figures/common.py (as the figure scripts read them; no bulk label is read):
  S1   figures/text/tables.md's hand-written S1A-S1C and S1E; S1D from stage 04d; S1F from stage 02 (sample tables, per-cell QC)
  S2   stage 11's rows (per pair, pooled); 11b's rows (reference, label-permuted, decomposition, rank, sensitivity),
       scores and notes (seeds passing the training check)
  S3   stage 11's LASSO rows; 11b's alternative-readout, native-head and numeric-age rows; the Brier score from 11b's scores
  S4   stage 05's candidates and selections, 11b's candidate scores, oracle and own-procedure rows, stage 06's row A
       records, stage 05v's scVI search
  S5   11b's subgroup, age-adjusted and feature-variant rows; seed variance and the seed ensemble from 11b's scores
  S6   11b's full-gene, ortholog and learning-curve rows (UCE iff config/figures.S12B_UCE)
  S7   stage 09's silhouettes and 11b's scores under each alignment (fig3's readers)
  S8   11b's best-candidate rows and notes; the encoder-seed SDs from 11b's candidate scores (fig4's differences)
  S9   stage 10c's concordance, stage 10's checks and inclusion, stage 12's simulation (Fig 5D's panel CSV) and criterion
  S10  stage 12a's property R² and licence, stage 10's modality rankings and summary, Fig 6C's panel CSV
  S11  every stage output's manifest cost (common.costs)
  S12  figures/text/tables.md's hand-written table
Legends and footnotes: figures/text/legends.md's block for each table. Text the pipeline has no source for is a constant here
(HAND, each with its source).

Outputs, in <labelled outputs>/tables/ (the outputs root in a real run): AdditionalFile2.xlsx (a sheet per sub-table:
title, subtitle, the table, its notes, the legend and footnotes), Table<S#>/Table<S#>[A].csv per sheet,
Table<S#>/Table<S#>.md (a review preview), sources.json. Numbers carry no thousands separators; AUROC-scale values 3
decimals, p values 3 significant digits.

  PYTHONPATH=src:third_party:figures python figures/tables.py   (after figures/make_all.py: it reads Figs 5D and 6C's CSVs)"""
import json
import re

import numpy as np
import pandas as pd

import common
import fig3
import fig4
import fig5
import s10
import s14
from sc2bulk import properties, simulation
from sc2bulk.config import configurations, fig1e, search_space, settings
from sc2bulk.config import figures as F
from sc2bulk.config import genes as G
from sc2bulk.config import properties as P
from sc2bulk.config import simulation as SIM
from sc2bulk.config import statistics as ST
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS
from sc2bulk.statistics import difference_variance, mdd

ITEMS = [f"Table S{i}" for i in range(1, 13)]
PANELS = (("Fig 5", "D"), ("Fig 6", "C"))   # the figure panel CSVs read (Tables S9E, S10E)
CAPTIONS = settings.REPO / "figures" / "text" / "legends.md"
DRAFTS = settings.REPO / "figures" / "text" / "tables.md"
NA = F.FIGURE_TEXT["not_applicable"]
GLYPH = {"yes": "●", "not determinable": "○", "no": ""}
SET_ENCODERS = ("cytoset", "set_transformer", "scrat")
SELECTED = tuple(f"{e}_cells" for e in SET_ENCODERS)   # 11b's best-candidate rows of a set encoder's selected configuration
MODELS_S8 = [*SET_ENCODERS, *(m for m in F.MODEL_ORDER if m not in SET_ENCODERS)]   # Table S8A: set encoders first
ALIGNMENTS = (*configurations.ALIGNMENTS, "unaligned_stacked")
ALIGNMENT_NAMES = {"none": "per-side (no alignment)", "combat": "ComBat", "coral": "CORAL", "harmony": "Harmony",
                   "unaligned_stacked": "one scaler for both sides (no alignment)"}
READOUT_NAMES = {"ridge": "ridge", "lasso": "LASSO", "elasticnet": "elastic net", "pls": "partial least squares",
                 "knn": "k-nearest neighbors", "mlp": "multilayer perceptron",
                 "native_head": "CytoSet's own classification head", "numeric_age": "ridge on numeric age"}
POOLS = {"nonceiling_equal": "Equal weights", "nonceiling_inverse_variance": "Inverse variance",
         "nonceiling_hartung_knapp": "Hartung–Knapp random effects", "nonceiling_logit": "Equal weights, logit scale",
         "lupus_crohns_equal": "Equal weights, criteria-accepted pairs (lupus and Crohn's)"}
# Table S2E's sensitivity rows (stage 11b's row names), in the legend's order, with their names
SENSITIVITY = {"pooled_predictions": "Human blood, aging: pooled-prediction AUROC",
               "split67": "Human blood, aging: 67/29 split", "passing_seeds": "Passing seeds only",
               "out_of_fold": "Readout fit on samples the encoder did not train on",
               "outside_moments": "Bulk scaled on moments from bulk outside the scored set",
               "single_cell_moments": "Bulk scaled with the single-cell training moments",
               "own_recipe": "Own published training procedure",
               "oracle": "Best candidate on bulk (selected with bulk labels; for reference only)ᵇ",
               "thinned_bulk": "Bulk as 10 bags of thinned pseudo-cellsᶜ",
               "scvi_observed_library": "scVI with the observed bulk library size",
               "per_sample_route": "Per-sample normalization route",
               "alignment_unaligned_stacked": "One scaler for both sides (no alignment)"}
FEATURES = {"tmm": "Pseudobulk, TMM", "vst": "Pseudobulk, VST", "panel_1000": "1000-gene panel", "panel_2000": "2000-gene panel",
            "all_shared": "Every shared gene", "full_genes": "Full gene set",
            "mapped_orthologs": "Panel genes with one-to-one human orthologs"}
PANELS_S5 = ("panel_1000", "panel_2000", "all_shared")   # Table S5E's gene panels (Fig. S11d)
# the key columns' values as printed (schemas.KEYS' variants and feature sets, the readouts, the alignments)
VALUE_NAMES = {
    "variant": {"primary": "primary", "tmm": "TMM", "vst": "VST", "training_fraction_0.25": "25% of training donors",
                "training_fraction_0.5": "50% of training donors", "out_of_fold_readout": "out-of-fold readout",
                "outside_moments_scaling": "outside-moments scaling", "per_sample_route_sensitivity": "per-sample route",
                "numeric_age_readout": "numeric-age readout", "pooled_predictions": "pooled predictions",
                "thinned_pseudocells": "thinned pseudo-cells", "observed_library": "observed library size",
                "seed_ensemble": "seed ensemble", "raw_fractions": "raw fractions (no log-ratio transform)",
                "no_ifn_markers": "without interferon markers", "pooled_z": "pooled z-scores",
                "median_centered": "median-centered"},
    "feature_set": {"panel": "shared panel", **FEATURES},
    "readout": {"none": "none (a score)", **READOUT_NAMES},
    "alignment": ALIGNMENT_NAMES,
    "cell_subset": {"all": "all cells", "one_type": "one cell type", "all_available": "all available cells"},
}
SUBGROUP_LEVELS = {"CTR": "control", "ELS": "early-life stress"}   # mouse microglia's bulk conditions
TARGETS = {"phenotype": "phenotype", "modality_direction": "single-cell-to-bulk direction"}   # stage 10's attribution targets
# the configuration-search settings as printed (config/search_space.py's parameters)
SETTINGS = {"h_dim": "hidden width", "nblock": "blocks", "pool": "block pooling", "out_pool": "output pooling",
            "decoder": "decoder",
            "cell_normalization": "cell normalization", "wts_decay": "weight decay", "lr": "learning rate",
            "batch_size": "batch size",
            "abundance_weighting": "abundance weighting", "age_similarity_loss": "age-similarity loss",
            "population_distance_feature": "population-distance feature", "pseudobulk_gene_scaler": "pseudobulk gene scaler",
            "dim_hidden": "hidden width", "num_heads": "attention heads", "num_inds": "inducing points",
            "encoder_layers": "encoder layers", "ln": "layer normalization", "dropout": "dropout",
            "encoder_block": "encoder block",
            "decoder_sab": "self-attention decoder", "model_dim": "model width", "num_layers": "layers",
            "norm_first": "normalization before attention", "pca": "PCA input", "input_dropout": "input dropout",
            "objective": "objective", "n_latent": "latent dimensions", "n_layers": "layers",
            "step_lr": "step decay (epochs, factor)",
            "ncell": "cells per bag"}
NAMED_SETTINGS = {"encoder_block": {"sab": "SAB encoder blocks"}, "objective": {"mse": "MSE objective (age)"}}   # printed whole
SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
# readable pipeline stage names (Table S11), one per pipeline/<stage>.py
STAGE_NAMES = {"00_soundlife_slims": "Sound Life subsampling", "01_download": "Download", "02_preprocess": "Preprocessing",
               "02b_all_available": "All-cell store", "03_splits": "Splits", "03b_attribution_splits": "Attribution splits",
               "04_panel": "Gene panel", "04b_deconvolve": "Deconvolution", "04c_qc_embed": "QC embedding",
               "04d_characteristics": "Pair characteristics", "05_search": "Configuration search",
               "05s_eligibility_simulation": "Eligibility simulation", "05t_timing": "Timing test",
               "05v_scvi_search": "scVI search",
               "06_train": "Training", "07_embed": "Foundation-model embedding",
               "07d_embed_all_available": "Embedding of all cells",
               "08_readout": "Readout", "08a_alpha_edges": "Readout penalty check", "08b_secondary": "Secondary readouts",
               "09_alignment": "Alignment", "09c_geometry": "Alignment coordinates",
               "10_attribution": "Gene attribution", "10c_concordance": "Gene concordance",
               "11_statistics": "Primary statistics", "11b_secondary": "Secondary statistics", "12_simulation": "Simulation",
               "12a_properties": "Sample properties"}
CHECK_NAMES = {"stability_cells": "stability across cell subsamples", "stability_readout_seeds": "stability across readout seeds",
               "random_T_top": "top-50 genes vs. matched random sets (median |T|)",
               "additivity_beta_k50": "additivity β (top 50)",
               "hk_dominated": "dominated by housekeeping genes"}   # Table S9D's gated checks (stage 10)
SEX = {"female": "female", "f": "female", "male": "male", "m": "male"}   # stage 02's sex labels, lowercased
BATCH_NAMES = {"chemistry": "chemistry", "release_cohort": "release cohort", "batch_id": "processing batch", "pool": "pool",
               "run": "sequencing run"}   # config/fig1e.BATCH's variables as printed
# house wording in every printed text (writing rules: American spelling, "readout"), and pipeline notes reworded whole
WORDING = {"read-out": "readout", "labellings": "labelings",
           "registered access": "controlled access", "SALINE": "saline", "1M counts": "1 million counts",
           "LPS-treated": "LPS-challenged",
           "fewer than 100 cells in the provider's data (pre-declared)":   # data/crohns.py's drop reason
               "fewer than 100 cells in the provider's data (excluded by identifier)",
           "n/a (unsupervised; ELBO finite)": "not applicable (unsupervised)",   # 11b's scVI training-check note
           simulation.READINGS[(True, True)]:   # stage 12's reading when both pass
               "Detects a signal that leaves every per-gene sum unchanged, as the mean of log-normalized cells also does, so it shows "
               "no capacity beyond a fixed summary."}
HOW = {"base": "base kept", "single_winner": "best single option", "merged": "merged group winners",
       "best_single": "best single option (the combined configuration failed)"}   # stage 05's decisions
GENERAL = ("Summary rank", "Mean summary rank", "Stop epoch, median")   # ranks and medians: whole where whole
INTEGERS = ("Mitochondrial ceiling (%)", "Rank", "Stop epoch, min", "Stop epoch, max")   # whole numbers stored as floats
# the a priori minimum detectable difference's planning values (correlation 0.5 at an AUROC of 0.8)
MDD_RHO, MDD_AUROC = 0.5, 0.8
PROPERTY_NAMES = {"depth": "Sequencing depth (R²)", "mito_fraction": "Mitochondrial fraction (R²)",
                  "composition_pc1": "Cell-type composition, PC1 (R²)", "phenotype": "Phenotype (AUROC)"}
MEASURES = {"within_sc_auroc": "Within-single-cell", "simulated_bulk_auroc": "Simulated-bulk"}   # Fig 5D's measures, named
# key column -> name: the descriptive columns a difference table shows where they vary within it
DESCRIPTORS = {"variant": "Variant", "feature_set": "Gene set", "readout": "Readout", "alignment": "Alignment",
               "bulk_subset": "Bulk subset", "cell_subset": "Cells"}
STATISTICS = {"n0": "n group 0", "n1": "n group 1", "delta": "Δ bulk AUROC", "ci_low": "95% CI low", "ci_high": "95% CI high",
              "u": "u (one-sided 95% upper bound)", "p_better": "p better", "p_worse": "p worse",
              "p_better_holm": "p better (Holm)", "p_worse_holm": "p worse (Holm)"}
DELTA_NOTE = "Δ, difference in bulk AUROC from pseudobulk."   # Table S2's sheets with a Δ bulk AUROC column
UNADJUSTED = {c: n for c, n in STATISTICS.items() if not c.endswith("holm")}
INTERVAL = {c: STATISTICS[c] for c in ("n0", "n1", "delta", "ci_low", "ci_high")}

# ---- hand-sourced text (no pipeline source) ----

HAND = {
    # S1D's definitions: figures/text/tables.md S1D, with config/fig1e.py's thresholds
    "definitions": {
        "Many cell types": f"≥ {fig1e.MANY_TYPES_MIN} annotated types, each ≥ {fig1e.MANY_TYPES_FRACTION:.0%} of the primary "
                           "unit's post-QC training cells, at one fixed annotation level per pair; mouse organs counted per organ, "
                           "\"has it\" if more than half the organs qualify",
        "Tissue or preparation differs between sides": "The sides come from different tissues (anatomical region included), "
                                                       "or the bulk is whole tissue while the single-cell side is dissociated "
                                                       "cells; isolation protocols are not part of the rule",
        "Single-nucleus → bulk": "Single-cell side is snRNA-seq (nuclear vs. whole-cell RNA)",
        "Independent cohorts": "No donor or animal on both sides",
        "Non-age phenotype": "The label is a disease status",
        "Several tissues": "The pair spans more than one tissue",
        "Few training donors (< 100)": f"< {fig1e.FEW_TRAINING_DONORS} single-cell training donors (mice for the mouse pairs)",
        "No shared label values": "No label value occurs on both sides",
        "Batch confounded with label": "Bias-corrected Cramér's V (Bergsma 2013) between the single-cell processing batch and "
                                       f"the 0/1 group over the primary unit's training samples ≥ {fig1e.BATCH_V}; ○ where no "
                                       "single-cell batch variable is recorded",
    },
    # S2A's input columns: figures/text/tables.md Table S12 ("Embeds a bulk profile?") and methods_supp_full_2026-10-19.md
    "training_input": {"cells": "bags of cells", "pseudobulk": "pseudobulk profiles"},
    "bulk_input": {"pseudobulk": "bulk profile", "cytoset": "bulk profile as a one-element set",
                   "set_transformer": "bulk profile as a one-element set", "scrat": "bulk profile as a one-element set",
                   "scvi": "bulk profile thinned to the median training-cell library size (10 draws, averaged)",
                   "geneformer": "bulk profile embedded directly", "scgpt": "bulk profile embedded directly"},
    "within": {"single_cell_heldout": "held-out donors' samples (cross-fitted folds)",
               "single_cell_validation": "the encoder split's validation samples"},
    # S1F's sex where stage 02 records none but the study reports it: the Discussion (Reemst et al. used male mice)
    "sex": {("hammond", "bulk"): "male (reported by the study)"},
    # S1F's footnotes: figures/text/tables.md S1F (ᵃ, ᶜ) and data/soundlife.py's docstring (ᵇ: a kit sequenced twice is summed)
    "s1f_marks": {"soundlife": "ᵃ ᵇ", "lupus": "ᶜ"},
    "s1f_notes": ["ᵃ Cells before QC: the provider's 3000-cell pre-QC subsamples (all cells where a sample had fewer).",
                  "ᵇ Bulk: the two libraries of each kit sequenced twice are summed.",
                  "ᶜ Single-cell cells: processing-cohort-4 cells only."],
    # S11's hardware: methods_supp_full_2026-10-19.md ("Trainings and embeddings ran on one server ...")
    "hardware": "One laboratory server with four NVIDIA RTX 3090 GPUs (24 GB each); at most two jobs per GPU.",
}


# ---- reading the notes ----

def clean(text) -> str:
    """Hand-written markdown as table text: no bold marks or comments, superscript footnote letters, no thousands
    separators."""
    text = re.sub(r"\^([abc])", lambda m: {"a": "ᵃ", "b": "ᵇ", "c": "ᶜ"}[m.group(1)], text.replace("**", ""))
    for old, new in WORDING.items():
        text = text.replace(old, new)
    return re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", re.sub(r"<!--.*?-->", "", text)).strip()


def pipe_table(lines) -> pd.DataFrame:
    rows = [[clean(c) for c in line.strip().strip("|").split("|")] for line in lines]
    return pd.DataFrame(rows[2:], columns=rows[0])


def block(lines) -> tuple:
    """(table, notes) of a hand-written block: its pipe table and the text around it."""
    table = [line for line in lines if line.startswith("|")]
    notes = [clean(line) for line in lines if line.strip() and not line.startswith("|")]
    return pipe_table(table), [n for n in notes if n]


def drafted(item) -> str:
    """figures/text/tables.md's '## <item>' section."""
    sections = re.split(r"^## ", common.read_note(DRAFTS), flags=re.M)[1:]
    return next(s for s in sections if s.partition("\n")[0].strip() == item).partition("\n")[2]


def drafted_parts(item) -> dict:
    """letter -> (subtitle, table, notes) of a section's '**S1A. Title.**' sub-tables."""
    text = drafted(item)
    heads = list(re.finditer(r"^\*\*S\d+([A-F])\. (.*?)\*\*(.*)$", text, flags=re.M))
    out = {}
    for i, m in enumerate(heads):
        table, notes = block(text[m.end():heads[i + 1].start() if i + 1 < len(heads) else len(text)].split("\n"))
        lead = clean(m.group(3))
        out[m.group(1)] = (clean(m.group(2)), table, [lead] * bool(lead) + notes)
    return out


def legend(item) -> list:
    """figures/text/legends.md's legend and footnotes for an item: the lines under its bold title, up to the first comment,
    quote or blank line."""
    lines = common.read_note(CAPTIONS).split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(f"**{item}."))
    out = []
    for line in lines[start + 1:]:
        if not line.strip() or line.startswith(("<!--", ">")):
            break
        out.append(clean(line.removeprefix("Legend: ")))
    return out


# ---- formatting ----

def pair_name(pair) -> str:
    return PAIRS[pair]["name"] if pair in PAIRS else "Pooled"


def not_pooled(r) -> str:
    """A pooled row's pairs left out: stage 11's excluded pairs, and for the criteria-accepted pool every other pair."""
    if r["pair"] != "pooled":
        return ""
    left_out = set(r["excluded_pairs"].split(","))
    if r["pool_set"] == "lupus_crohns_equal":
        left_out |= set(PAIR_ORDER) - {"lupus", "crohns"}
    return "; ".join(pair_name(p) for p in PAIR_ORDER if p in left_out)


def subset_name(subset) -> str:
    """A bulk subset as printed: "organ=Limb_Muscle" -> "organ: limb muscle", "cmv=Negative" -> "CMV-negative"; "all" and
    "age_adjusted" as words."""
    if "=" not in subset:
        return subset.replace("_", " ")
    variable, level = subset.split("=")
    if variable == "cmv":
        return f"CMV-{level.lower()}"
    if variable == "inflammation":
        return level.replace("_vs_", " vs. ").replace("_", " ")
    if variable == "condition":
        return f"condition: {SUBGROUP_LEVELS[level]}"
    return f"{variable}: {level.replace('_', ' ').lower() if variable == 'organ' else level}"


def value_text(value) -> str:
    """A setting's value as printed: 1 × 10⁻⁴ for small floats, per-sample for per_sample, (50, 0.5) for a pair."""
    if isinstance(value, float) and 0 < value < 0.05:
        mantissa, exponent = f"{value:.0e}".split("e")
        return f"{mantissa} × 10{str(int(exponent)).translate(SUPERSCRIPTS)}"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(map(value_text, value)) + ")"
    return f"{value:g}" if isinstance(value, float) else str(value).replace("_", "-")


def settings_text(change) -> str:
    """A configuration change as printed: "hidden width 128"; a switched-on option by its name; "PCA input off"."""
    words = [NAMED_SETTINGS[k][v] if k in NAMED_SETTINGS else SETTINGS[k] if v is True else f"{SETTINGS[k]} off" if v is False
             else f"{SETTINGS[k]} {value_text(v)}" for k, v in change.items()]
    return "; ".join(words) if words else "none"


def group_label(group) -> str:
    """A search group as printed: the setting it changes; the base and the merged winners as such."""
    return {"base": "base", "merged": "merged"}.get(group) or SETTINGS[group]


def change_text(change_value) -> str:
    """Stage 05's stored change from the base (JSON) as printed."""
    return settings_text(json.loads(change_value))


def candidate_label(encoder, key) -> str:
    """A search candidate as printed, from its change from the base (config/search_space.py); the base, the merged
    winners, and CytoSet's MSE objective by name."""
    named = {"base": "base", "merged": "merged group winners", "merged_no_per_sample": "merged group winners (no per-sample)"}
    if key in named:
        return named[key]
    grids = {**search_space.GROUPS_BY_ENCODER, "scvi": search_space.SCVI_GROUPS}[encoder]
    changes = {name: change for group in grids.values() for name, change in group.items()} | search_space.OUTSIDE_SEARCH
    return settings_text(changes[key])


def pool_name(pool_set) -> str:
    return POOLS[pool_set] if pool_set in POOLS else f"Equal weights without {pair_name(pool_set.split('_without_')[1])}"


def row_label(r) -> str:
    """A row's name: the configuration or reference label; a candidate as "Model: candidate"."""
    if "row" in r and r["row"] == "label_permuted":
        return "Label-permuted pseudobulk"
    if r["key"].endswith("_own_recipe"):
        return f"{model_label(r['model'])}, own published training procedure"
    if r["row_kind"] == "candidate" and r["model"] in configurations.MODELS:
        return f"{model_label(r['model'])}: {candidate_label(r['model'], r['key'])}"
    if r["key"] in configurations.REFERENCE_ROWS:   # the full labels (figures shorten some)
        return configurations.REFERENCE_ROWS[r["key"]]["label"]
    return common.label(r["key"])


def ordered(rows) -> pd.DataFrame:
    """Pairs in PAIR_ORDER then pooled, keys in the display order (then the reference rows); otherwise as read."""
    pairs = {p: i for i, p in enumerate([*PAIR_ORDER, "pooled"])}
    keys = {k: i for i, k in enumerate([*F.DISPLAY_ORDER, *configurations.REFERENCE_ROWS])}
    rank = rows.assign(_pair=rows["pair"].map(pairs), _key=rows["key"].map(keys).fillna(len(keys)))
    return rank.sort_values(["_pair", "_key"], kind="stable").drop(columns=["_pair", "_key"]).reset_index(drop=True)


def differences(rows, statistics=STATISTICS, row_name="Configuration", extra=None) -> pd.DataFrame:
    """Stage 11/11b difference rows as a table: the extra columns ({column: name}) first, the pair (and pool), the row's
    name, the key columns that vary within the rows, the statistics, and whether 11b could estimate it."""
    rows, extra = ordered(rows), extra or {}
    out = pd.DataFrame({name: rows[column].to_numpy() for column, name in extra.items()})
    out["Pair"] = rows["pair"].map(pair_name).to_numpy()
    out[row_name] = [row_label(r) for _, r in rows.iterrows()]
    if (rows["pair"] == "pooled").any():
        pools = [pool_name(s) if p == "pooled" else "" for p, s in zip(rows["pair"], rows["pool_set"])]
        out.insert(len(extra) + 1, "Pool", pools)
        out.insert(len(extra) + 2, "Pairs not pooled", [not_pooled(r) for _, r in rows.iterrows()])
    for column, name in DESCRIPTORS.items():
        if column in rows and rows[column].nunique() > 1:
            out[name] = rows[column].map(subset_name if column == "bulk_subset" else VALUE_NAMES[column]).to_numpy()
    for column, name in statistics.items():
        out[name] = rows[column].to_numpy()
    for column in ("n0", "n1"):
        if column in statistics:
            out[statistics[column]] = out[statistics[column]].astype("Int64")
    if "estimable" in rows:
        out["Estimable"] = rows["estimable"].map({True: "yes", False: "no"}).to_numpy()
    return out


def stacked(frames) -> pd.DataFrame:
    """Difference tables one under another, the pool columns (where any part has pooled rows) after the pair."""
    table = pd.concat(frames, ignore_index=True)
    pool = [c for c in ("Pool", "Pairs not pooled") if c in table]
    rest = [c for c in table.columns if c not in pool]
    at = rest.index("Pair") + 1
    return table[rest[:at] + pool + rest[at:]].fillna({c: "" for c in pool})


def secondary_rows(row, **conditions) -> pd.DataFrame:
    return common.pick(common.secondary(), neutral=False, row=row, statistic="auroc_difference", **conditions)


def notes_11b() -> dict:
    return common.read_json("11b_secondary", "pooled", f"{common.LABELLED}secondary_notes.json")


def sheet(frame, subtitle=None, notes=()) -> tuple:
    assert len(frame), f"an empty table: {subtitle}"
    return subtitle, frame.reset_index(drop=True), list(notes)


def is_p(column) -> bool:
    return column == "p" or column.startswith("p ")


def rounded(frame) -> pd.DataFrame:
    """p values (columns named "p" or "p ...") to 3 significant digits, other decimals to 3 places, no signed zeros;
    the whole-number columns stored as floats (INTEGERS) as integers."""
    out = frame.copy()
    for column in out.columns:
        if column in INTEGERS:
            out[column] = out[column].round().astype("Int64")
        elif pd.api.types.is_float_dtype(out[column]):
            values = [float(f"{v:.3g}") if is_p(column) else round(v, 3) for v in out[column]]
            out[column] = np.array(values) + 0.0   # -0.0 -> 0.0
    return out


def text(value, column) -> object:
    """A cell as printed in the CSV and preview: n/a for missing, yes/no for booleans, p values to 3 significant digits
    (1.00), ranks and medians as they are (2, 2.5), other decimals to 3 places."""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return NA
    if isinstance(value, float):
        return f"{value:#.3g}" if is_p(column) else f"{value:g}" if column in GENERAL else f"{value:.3f}"
    return value


def shown(frame) -> pd.DataFrame:
    """The rounded table as printed (text())."""
    table = rounded(frame).astype(object)
    return pd.DataFrame({c: [text(v, c) for v in table[c]] for c in table.columns}, columns=table.columns)


def unit_name(unit) -> str:
    """A unit as printed: a cross-fitted pair's "fold 0", the 67/29 split, or "all" for the other pairs' one split."""
    fold_set, fold = unit
    return f"fold {fold}" if fold_set == "crossfit" else "67/29 split" if fold_set == "split67" else "all"


def model_label(model) -> str:
    return configurations.MODELS[model]["label"]


# ---- shared measures ----

def mean_measure(pair, key, column, side="bulk", seeds=None, **conditions) -> float:
    """A configuration's mean of one scores column in stage 11's order (readout seeds, encoder seeds per fold, folds)."""
    seeds = common.seeds_of(key) if seeds is None else seeds
    part = common.pick(common.scores(), pair=pair, fold_set=common.primary_units(pair)[0][0], key=key, side=side,
                       row_kind="configuration", bulk_subset="all", encoder_seed=tuple(seeds),
                       **{**common.headline_conditions(key), **conditions})
    assert len(part), f"{pair} {key} {side} {conditions}: no scores"
    return float(part.groupby(["fold", "encoder_seed"])[column].mean().groupby("fold").mean().mean())


def seed_sd(raw) -> tuple:
    """(SD over encoder seeds of the per-seed AUROC, or over readout seeds for a fixed representation; which)."""
    if raw["encoder_seed"].eq(-1).all():
        per_readout = raw.groupby(["readout_seed", "fold"])["auroc"].mean().groupby("readout_seed").mean()
        return float(per_readout.std(ddof=1)), "readout seeds"
    return float(common.per_seed(raw).groupby("encoder_seed")["auroc"].mean().std(ddof=1)), "encoder seeds"


# ---- Table S1 ----

def characteristics_table() -> pd.DataFrame:
    """S1D: every characteristic, its definition, whether Fig 1e shows it, and each pair's state and annotation (stage
    04d)."""
    rows = []
    for column in fig1e.FIG1E_COLUMNS:
        row = {"Characteristic": column, "Definition and threshold": HAND["definitions"][column],
               "In Fig 1e": "yes" if column in F.FIG1E_PANEL else "no"}
        for pair in PAIR_ORDER:
            table = common.read_csv("04d_characteristics", pair, "characteristics.csv", keep_default_na=False)
            r = table.set_index("column").loc[column]
            row[pair_name(pair)] = f"{GLYPH[r['state']]} {annotation(r['annotation'])}".strip()
        rows.append(row)
    return pd.DataFrame(rows)


def annotation(note) -> str:
    """A stage 04d annotation as printed: no annotation-level prefix, readable batch variables and levels, Cramér's V, ≥."""
    note = note.replace(">=", "≥").replace("Cramer's", "Cramér's").replace(" 1 types", " 1 type")
    for level in set(fig1e.ANNOTATION.values()):
        note = note.replace(f"{level}: ", "")
    for variable, name in BATCH_NAMES.items():
        note = re.sub(rf"^{variable}: ", f"{name}, ", note)
    for old, new in F.ANNOTATION_TEXT.items():
        note = note.replace(old, new)
    return note.replace("_", " ")


def counted(reasons) -> str:
    return "; ".join(f"{n} {wording(reason)}" for reason, n in reasons.value_counts().items())


def wording(note) -> str:
    """A pipeline note in house wording (WORDING)."""
    for old, new in WORDING.items():
        note = note.replace(old, new)
    return note


def sex(table, pair, side) -> str:
    """Sex composition of kept samples; where stage 02's table has none, what the study reports (HAND) or "not recorded"."""
    if "sex" not in table:
        return HAND["sex"].get((pair, side), "not recorded")
    counts = table["sex"].str.lower().map(lambda s: SEX[s]).value_counts().sort_index()
    return "; ".join(f"{n} {s}" for s, n in counts.items())


def removals_table() -> pd.DataFrame:
    """S1F: samples and cells removed at each selection and QC step (stage 02's sample tables and per-cell QC)."""
    rows = []
    for pair in PAIR_ORDER:
        single = common.all_single_cell_samples(pair)
        kept = single[single["drop_reason"].isna()]
        ceiling = PAIRS[pair].get("qc_max_percent_mito", settings.QC_MAX_PERCENT_MITO)
        cells = common.cell_qc(pair, ("sample_id", "genes_detected", "percent_mito"))
        cells = cells[cells["sample_id"].isin(kept["sample_id"])]
        bulk = common.read_csv("02_preprocess", pair, "bulk_samples.csv", dtype={"sample_id": str, "donor_id": str})
        bulk_kept = bulk[bulk["drop_reason"].isna()]
        rows.append({"Pair": pair_name(pair), "Single-cell samples listed": len(single), "Single-cell samples kept": len(kept),
                     "Single-cell exclusions": counted(single["drop_reason"].dropna()),
                     "Cells before QC": int(kept["cells_before_qc"].sum()),
                     f"Removed: < {settings.QC_MIN_GENES_PER_CELL} genes":
                         int((cells["genes_detected"] < settings.QC_MIN_GENES_PER_CELL).sum()),
                     "Mitochondrial ceiling (%)": ceiling,
                     "Removed: mitochondrial ceiling": int((cells["percent_mito"] >= ceiling).sum()),
                     "Cells after QC": int(kept["cells_after_qc"].sum()),
                     "Smallest sample after QC": int(kept["cells_after_qc"].min()),
                     "Cells stored": int(kept["cells_stored"].sum()), "Bulk samples listed": len(bulk),
                     "Bulk samples kept": len(bulk_kept), "Bulk exclusions": counted(bulk["drop_reason"].dropna()),
                     "Sex, single-cell": sex(kept, pair, "single_cell"), "Sex, bulk": sex(bulk_kept, pair, "bulk"),
                     "Notes": HAND["s1f_marks"].get(pair, "")})
    return pd.DataFrame(rows)


def table_s1() -> dict:
    parts = drafted_parts("Table S1")
    out = {f"S1{letter}": sheet(parts[letter][1], parts[letter][0], parts[letter][2]) for letter in "ABC"}
    out["S1D"] = sheet(characteristics_table(), parts["D"][0], [
        "● has it; ○ not determinable; blank, does not have it. States and values computed by the rules in the Methods."])
    out["S1E"] = sheet(parts["E"][1], parts["E"][0], parts["E"][2])
    out["S1F"] = sheet(removals_table(), parts["F"][0], [
        "Samples listed and kept, with each exclusion's reason; cells summed over kept samples. A cell failing both QC "
        f"thresholds is counted in both removal columns. Cells stored: at most {settings.MAX_STORED_CELLS} per sample, drawn "
        "with a fixed seed.", *HAND["s1f_notes"]])
    return out


# ---- Table S2 ----

def configuration_rows(pair) -> list:
    """S2A's rows of one pair: pseudobulk, then every alternative in the display order."""
    stats = common.pick(common.statistics(), neutral=False, pair=pair, pool_set="per_pair", readout="ridge",
                        statistic="auroc_difference").set_index("key")
    assert sorted(stats.index) == sorted(common.alternatives()), f"S2A {pair}: statistics rows {sorted(stats.index)}"
    sec = common.secondary()
    permuted = common.pick(sec, neutral=False, pair=pair, row="label_permuted", permutation=-1)
    permuted = permuted.set_index(["key", "statistic"])["delta"]
    decomposed = common.pick(sec, neutral=False, pair=pair, row="decomposition").set_index("key")
    rank = common.pick(sec, neutral=False, pair=pair, row="summary_rank").set_index("key")["delta"]
    passing = notes_11b()["seeds_passed"][pair]
    side = common.within_side(pair)
    rows = []
    for key in common.kept():
        conditions = common.headline_conditions(key)
        raw = common.aurocs(pair, key, **conditions)
        sd, over = seed_sd(raw)
        within = common.aurocs(pair, key, common.seeds_of(key) if side == "single_cell_heldout" else common.encoder_seeds(),
                               side=side, **conditions)
        counts = stats.loc[key] if key in stats.index else stats.iloc[0]
        prefix = "permuted_minus_primary" if key == "pseudobulk" else "permuted"   # pseudobulk: Fig 2a's ✕ row
        seeds = passing.get(key, NA)
        rows.append({"Pair": pair_name(pair), "Configuration": common.label(key),
                     "Training input": HAND["training_input"][configurations.CONFIGURATIONS[key]["input"]],
                     "Bulk input": HAND["bulk_input"][configurations.CONFIGURATIONS[key]["model"]],
                     "n group 0": counts["n0"], "n group 1": counts["n1"],
                     "Bulk AUROC, mean": common.mean_over_seeds(raw), "Bulk AUROC, SD": sd, "SD over": over,
                     **{name: stats[c].get(key, np.nan) for c, name in STATISTICS.items() if c not in ("n0", "n1")},
                     "Brier score": mean_measure(pair, key, "brier"), "AUPRC": mean_measure(pair, key, "auprc"),
                     "Seeds passing the training check": passing_text(seeds),
                     "Within-single-cell AUROC": common.mean_over_seeds(within),
                     "Within-single-cell grouping": HAND["within"][side],
                     "Label-permuted Δ, mean": permuted.get((key, f"{prefix}_mean"), np.nan),
                     "Label-permuted Δ, 95th percentile": permuted.get((key, f"{prefix}_q95"), np.nan),
                     "Decomposition: cells − pseudobulk input": decomposed["delta"].get(key, np.nan),
                     "Decomposition 95% CI low": decomposed["ci_low"].get(key, np.nan),
                     "Decomposition 95% CI high": decomposed["ci_high"].get(key, np.nan), "Summary rank": rank.get(key, np.nan)})
    return rows


def passing_text(seeds) -> str:
    """11b's training-check label per unit ("trained (9/10 seeds)"): one unit's as is, a cross-fitted pair's per fold;
    n/a for a fixed representation, scVI's as 11b writes it."""
    if not isinstance(seeds, dict):
        return wording(seeds)
    if len(seeds) == 1:
        return next(iter(seeds.values()))
    return "; ".join(f"{unit_name(named_unit(u))}: {v}" for u, v in seeds.items())


def named_unit(name) -> tuple:
    """11b's unit name ("crossfit_0") as (fold set, fold)."""
    fold_set, fold = name.rsplit("_", 1)
    return fold_set, int(fold)


def a_priori_mdd(pair) -> float:
    """The a priori minimum detectable difference (scripts/mdd_table.py's formula): each pair's group sizes per fold
    (computed before bulk labels were used), the planning correlation and AUROC, the encoder-seed SD planning value and
    the encoder seeds; one-sided at alpha, 80% power."""
    folds_ = ST.CLASS_COUNTS[pair]
    variance = np.mean([difference_variance(n0, n1, MDD_AUROC, MDD_RHO, ST.SEED_SD, settings.N_ENCODER_SEEDS)
                        for n0, n1 in folds_])
    return mdd(variance / len(folds_), ST.ALPHA)


def pair_references() -> pd.DataFrame:
    """S2B: per pair, pseudobulk's bulk AUROC and the pooling rule, the within-bulk reference and the a priori minimum
    detectable difference."""
    within = secondary_rows("within_bulk_reference").set_index("pair")
    rows = []
    for pair in PAIR_ORDER:
        reference = common.aurocs(pair, "within_bulk_reference", [-1], row_kind="reference", n_cells=0, pooling="native")
        pooled = "yes" if pair in common.kept_pairs() else f"no (pseudobulk AUROC ≥ {common.threshold()})"
        rows.append({"Pair": pair_name(pair), "Pseudobulk bulk AUROC": common.pseudobulk_auroc(pair),
                     "In the pooled average": pooled,
                     "Within-bulk reference AUROC": common.mean_over_seeds(reference),
                     "Within-bulk reference Δ": within.loc[pair, "delta"], "95% CI low": within.loc[pair, "ci_low"],
                     "95% CI high": within.loc[pair, "ci_high"],
                     "Minimum detectable difference (a priori)ᵃ": a_priori_mdd(pair)})
    pooled = within.loc["pooled"]   # the equal-weight pool over the pooled pairs
    rows.append({"Pair": "Pooled", "Within-bulk reference Δ": pooled["delta"], "95% CI low": pooled["ci_low"],
                 "95% CI high": pooled["ci_high"]})
    return pd.DataFrame(rows)


def pooled_rows() -> pd.DataFrame:
    """S2C: stage 11's pooled rows (heterogeneity reported, not interpreted), 11b's leave-one-pair-out rows, the pooled
    decomposition, and the mean summary rank (on the equal-weight rows)."""
    primary = common.pick(common.statistics(), neutral=False, pair="pooled")
    decomposed = common.pick(common.secondary(), neutral=False, pair="pooled", row="decomposition")
    rows = ordered(pd.concat([primary.assign(estimable=np.isfinite(primary["delta"]), shown="difference from pseudobulk"),
                              secondary_rows("leave_one_pair_out").assign(shown="difference from pseudobulk"),
                              decomposed.assign(shown="decomposition: cells minus pseudobulk input")], ignore_index=True))
    table = differences(rows, {**STATISTICS, "q": "Cochran's Q", "i2": "I²", "tau2": "τ²"}, extra={"shown": "Statistic"})
    rank = common.pick(common.secondary(), neutral=False, pair="pooled", row="summary_rank").set_index("key")["delta"]
    equal = (rows["pool_set"] == "nonceiling_equal") & (rows["shown"] == "difference from pseudobulk")
    table["Mean summary rank"] = [rank.get(k, np.nan) if e else np.nan for k, e in zip(rows["key"], equal)]
    return table


def table_s2() -> dict:
    rows = pd.DataFrame([r for pair in PAIR_ORDER for r in configuration_rows(pair)])
    rows[["n group 0", "n group 1"]] = rows[["n group 0", "n group 1"]].astype("Int64")
    sensitivity = stacked([differences(secondary_rows(row).assign(shown=name), extra={"shown": "Sensitivity"})
                           for row, name in SENSITIVITY.items() if len(secondary_rows(row))])
    out_of_fold = pd.DataFrame([{"Pair": pair_name(p), "Configuration": common.label(k),
                                 "Bulk AUROC, out-of-fold readout": mean_measure(p, k, "auroc", seeds=common.encoder_seeds(),
                                                                                 variant="out_of_fold_readout"),
                                 "Bulk AUROC, primary readout": mean_measure(p, k, "auroc")}
                                for p in PAIR_ORDER for k in common.kept()])
    references = pd.concat([secondary_rows("reference_rows"),
                            common.pick(common.secondary(), neutral=False, row="label_permuted", key="pseudobulk",
                                        statistic="permuted_minus_primary_mean")], ignore_index=True)
    return {"S2A": sheet(rows, "Per pair and configuration", [
                "Bulk AUROC: the mean over readout seeds, then encoder seeds, then folds; SD over encoder seeds (fixed "
                "representations: readout seeds). Differences from pseudobulk with two-sided 95% bootstrap CIs; p values "
                "one-sided, Holm-adjusted within each family. Label-permuted Δ: the configuration under permuted labels "
                "(pseudobulk: its permuted minus primary AUROC). Summary rank: the configuration's rank by its difference "
                "from pseudobulk within the pair (pseudobulk counted at 0; 1 is the largest).", DELTA_NOTE]),
            "S2B": sheet(pair_references(), "Pseudobulk, the pooling rule, the within-bulk reference, and the minimum "
                         "detectable difference, per pair", [
                f"Minimum detectable difference: one-sided test at {ST.ALPHA} with 80% power, from each pair's group sizes, "
                f"computed before bulk labels were used, a between-configuration correlation of {MDD_RHO}, an AUROC of {MDD_AUROC}, "
                f"and an encoder-seed SD of {ST.SEED_SD}."]),
            "S2C": sheet(pooled_rows(), "Pooled estimates", [
                "Logit-scale row: Δ on the logit scale. Leave-one-pair-out rows: descriptive.", DELTA_NOTE]),
            "S2D": sheet(differences(references, UNADJUSTED, "Reference row"), "Reference analyses", [
                "Differences from primary pseudobulk, unadjusted; label-permuted row: pseudobulk's permuted minus primary "
                "AUROC (mean over permutations).", DELTA_NOTE]),
            "S2E": sheet(sensitivity, "Sensitivity analyses", [
                "Holm-adjusted only for the out-of-fold and thinned-bulk rows (each within its own families); the others "
                "are unadjusted.", DELTA_NOTE]),
            "S2F": sheet(out_of_fold, "Bulk AUROC with the readout fit on samples the encoder did not train on", [
                "Out-of-fold readout: every configuration, pseudobulk included, at each unit's passing splits (the mean over "
                "them); primary readout: as in S2A."])}


# ---- Table S3 ----

def table_s3() -> dict:
    lasso = common.pick(common.statistics(), neutral=False, pool_set="per_pair", readout="lasso", statistic="auroc_difference")
    parts = [lasso.assign(estimable=np.isfinite(lasso["delta"])),
             *(secondary_rows(f"readout_{r}") for r in ("elasticnet", "pls", "knn", "mlp")),
             secondary_rows("native_head"), secondary_rows("numeric_age")]
    frames = []
    for part in parts:
        part = ordered(part)
        numeric = (part["variant"] == "numeric_age_readout").all()
        table = differences(part.assign(shown=READOUT_NAMES["numeric_age" if numeric else part["readout"].iloc[0]]), UNADJUSTED,
                            extra={"shown": "Readout"})
        conditions = [{"variant": v} if numeric else {"readout": r} for r, v in zip(part["readout"], part["variant"])]
        for column, name in (("auroc", "Bulk AUROC, mean"), ("brier", "Brier score")):
            table[name] = [mean_measure(p, k, column, **c) if p != "pooled" else np.nan
                           for p, k, c in zip(part["pair"], part["key"], conditions)]
        if part["readout"].iloc[0] != "native_head":   # pseudobulk under the same readout (CytoSet's head has none)
            change = conditions[0]
            counts = part.groupby("pair")[["n0", "n1"]].first()   # the readout's scored groups in the pair
            pseudobulk = pd.DataFrame([{"Readout": table["Readout"].iloc[0], "Pair": pair_name(p),
                                        "Configuration": common.label("pseudobulk"), "n group 0": counts.loc[p, "n0"],
                                        "n group 1": counts.loc[p, "n1"],
                                        "Bulk AUROC, mean": mean_measure(p, "pseudobulk", "auroc", **change),
                                        "Brier score": mean_measure(p, "pseudobulk", "brier", **change)}
                                       for p in PAIR_ORDER if p in set(part["pair"])])   # numeric age: the aging pairs
            order = {pair_name(p): i for i, p in enumerate([*PAIR_ORDER, "pooled"])}
            table = pd.concat([pseudobulk.astype({"n group 0": "Int64", "n group 1": "Int64"}), table],
                              ignore_index=True)[table.columns]
            table = table.assign(_pair=table["Pair"].map(order)).sort_values("_pair", kind="stable").drop(columns="_pair")
        frames.append(table)
    return {"S3": sheet(stacked(frames), None, [
        "Pseudobulk rows: its bulk AUROC and Brier score under the same readout. Unadjusted: reported with intervals and "
        "never called significant. LASSO: per pair only (not pooled)."])}


# ---- Table S4 ----

def candidate_table(pair) -> pd.DataFrame:
    """Every search candidate per unit: mean validation BCE and bulk AUROC over the search seeds, seeds trained."""
    table = s10.candidates(pair)
    scores = common.pick(common.scores(), pair=pair, row_kind="candidate", side="bulk", bulk_subset="all", n_cells=T.FROZEN_NCELL,
                         fold_set=common.primary_units(pair)[0][0], model=tuple(s10.encoders()))
    auroc = scores.groupby(["model", "fold", "key", "encoder_seed"], as_index=False)["auroc"].mean()
    table = table.merge(auroc, on=["model", "fold", "key", "encoder_seed"], how="left", validate="one_to_one")
    assert table["auroc"].notna().all(), f"S4A {pair}: candidates without bulk scores"
    grouped = table.groupby(["fold_set", "fold", "model", "key", "change_group", "change_value", "merged"], sort=False)
    out = grouped.agg(bce=("validation_bce", "mean"), auroc=("auroc", "mean"), trained=("did_train", "sum"),
                      seeds=("did_train", "size"))
    out = out.reset_index().sort_values(["fold", "merged"], kind="stable")   # per unit, the merged configuration last
    return pd.DataFrame({"Pair": pair_name(pair), "Fold": [unit_name(u) for u in zip(out["fold_set"], out["fold"])],
                         "Encoder": out["model"].map(model_label),
                         "Candidate": [candidate_label(m, k) for m, k in zip(out["model"], out["key"])],
                         "Change group": out["change_group"].map(group_label), "Change": out["change_value"].map(change_text),
                         "Validation BCE, mean": out["bce"], "Bulk AUROC, mean": out["auroc"],
                         "Seeds trained": [f"{t}/{n}" for t, n in zip(out["trained"], out["seeds"])]})


def budget_table() -> pd.DataFrame:
    """Candidate trainings x search seeds per encoder and unit (the merged configuration counted apart); scVI's grid
    apart."""
    rows = []
    for pair in PAIR_ORDER:
        for (fold_set, fold, model), part in s10.candidates(pair).groupby(["fold_set", "fold", "model"], sort=False):
            searched = part[~part["merged"]]
            rows.append({"Pair": pair_name(pair), "Fold": unit_name((fold_set, fold)), "Encoder": model_label(model),
                         "Candidates (base included)": searched["key"].nunique(),
                         "Search seeds": searched["encoder_seed"].nunique(),
                         "Candidate trainings × seeds": len(searched), "Merged trainings": int(part["merged"].sum())})
        for unit in common.primary_units(pair):
            grid = common.read_csv("05v_scvi_search", pair, "elbo_s0.csv", unit)["candidate"].nunique()
            rows.append({"Pair": pair_name(pair), "Fold": unit_name(unit), "Encoder": "scVI (not in the budget)",
                         "Candidates (base included)": grid, "Search seeds": len(common.search_seeds()),
                         "Candidate trainings × seeds": grid * len(common.search_seeds()), "Merged trainings": 0})
    return pd.DataFrame(rows)


def selection_table() -> pd.DataFrame:
    """Per unit and encoder: how the configuration was chosen, the settings adopted, whether the base was kept, the
    merge check (stage 05's selection.json)."""
    rows = []
    for pair in PAIR_ORDER:
        for unit in common.primary_units(pair):
            for encoder in s10.encoders():
                selection = common.read_json("05_search", pair, f"{encoder}/selection.json", unit)
                base = json.loads(json.dumps(search_space.base(encoder)))   # as stored: tuples become lists
                adopted = {k: v for k, v in selection["frozen"].items()
                           if base.get(k) != v and k not in ("encoder", "micro_batch")}   # titles.selected_carrier's rule
                merged = selection["winners"]["merged_result"]
                rows.append({"Pair": pair_name(pair), "Fold": unit_name(unit), "Encoder": model_label(encoder),
                             "How chosen": HOW[selection["how"]],
                             "Settings adopted (changes from the base)": settings_text(adopted),
                             "Kept its base": "yes" if selection["how"] == "base" else "no",
                             "Group winners": "; ".join(candidate_label(encoder, n)
                                                        for n in selection["winners"]["kept"].values()) or "none",
                             "Combined configuration tested (trains; better than the base)": NA if merged is None else
                                 f"{'yes' if merged[0] else 'no'}, {'yes' if merged[1] else 'no'}"})
    return pd.DataFrame(rows)


def training_records() -> pd.DataFrame:
    """The frozen configuration's training records per encoder seed (stage 06 row A): the training check and the kept
    epoch."""
    rows = []
    for pair in PAIR_ORDER:
        for unit in common.primary_units(pair):
            for encoder in s10.encoders():
                for e in common.encoder_seeds():
                    record = common.read_json("06_train", pair, f"A/{encoder}/record_s{e}.json", unit)
                    rows.append({"Pair": pair_name(pair), "Fold": unit_name(unit), "Encoder": model_label(encoder),
                                 "Encoder seed": e, "Margin d*": record["margin_d_star"], "p": record["margin_p"],
                                 "Trained": "yes" if record["did_train"] else "no",
                                 "Reason": wording(record["did_train_reason"]), "Stop epoch": record["stop_epoch"]})
    return pd.DataFrame(rows)


def stop_epochs(records) -> pd.DataFrame:
    spread = records.groupby(["Pair", "Fold", "Encoder"], sort=False)["Stop epoch"].agg(["min", "median", "max"]).reset_index()
    return spread.rename(columns={"min": "Stop epoch, min", "median": "Stop epoch, median", "max": "Stop epoch, max"})


def scvi_search() -> pd.DataFrame:
    rows = []
    for pair in PAIR_ORDER:
        for unit in common.primary_units(pair):
            table = pd.concat([common.read_csv("05v_scvi_search", pair, f"elbo_s{i}.csv", unit) for i in common.search_seeds()])
            mean = table.groupby(["candidate", "group"], sort=False)["validation_elbo"].mean().reset_index()
            base = mean.loc[mean["candidate"] == "base", "validation_elbo"].iloc[0]
            rows += [{"Pair": pair_name(pair), "Fold": unit_name(unit), "Candidate": candidate_label("scvi", r.candidate),
                      "Group": group_label(r.group),
                      "Validation negative ELBO (lower is better), mean": r.validation_elbo,
                      "Negative ELBO minus the base's": r.validation_elbo - base}
                     for r in mean.itertuples()]
    return pd.DataFrame(rows)


def table_s4() -> dict:
    records = training_records()
    outside = stacked([differences(secondary_rows(row).assign(shown=name), UNADJUSTED, "Candidate", {"shown": "Row"})
                       for row, name in (("oracle", "Best candidate on bulk (selected with bulk labels; for reference only)"),
                                         ("own_recipe", "Own published training procedure"))])
    return {"S4A": sheet(pd.concat([candidate_table(p) for p in PAIR_ORDER], ignore_index=True), "Every search candidate", [
                "Means over the search seeds; candidates of every unit (human blood, aging: per fold). Per-seed values: "
                "Additional file 5."]),
            "S4B": sheet(budget_table(), "Tuning budgets"),
            "S4C": sheet(selection_table(), "Selections per fold"),
            "S4D": sheet(stop_epochs(records), "Stop-epoch distributions (trainings at the selected configuration)"),
            "S4E": sheet(records, "Training checks per encoder seed (trainings at the selected configuration)", [
                "d: ln 2 minus the class-balanced mean validation binary cross-entropy over held-out groups; margin d*: the "
                "largest d over logit temperatures T (scale-free); p: the permutation p value of d, labels permuted across "
                f"held-out groups. A seed passes if d* exceeds {T.MARGIN_FLOOR} × ln 2 and, where at least "
                f"{T.MIN_TESTABLE} labelings exist, p ≤ {T.NULL_ALPHA}."]),
            "S4F": sheet(outside, "Best candidate on bulk and each encoder's published training procedure",
                         ["Differences from primary pseudobulk, unadjusted."]),
            "S4G": sheet(scvi_search(), "scVI's search (not counted in the budget)")}


# ---- Table S5 ----

def seed_table() -> pd.DataFrame:
    """Per pair and configuration: the SDs over encoder and readout seeds, and the seed ensemble's AUROC."""
    rows = []
    for pair in PAIR_ORDER:
        for key in common.kept():
            conditions = common.headline_conditions(key)
            raw = common.aurocs(pair, key, **conditions)
            trained = key not in common.NO_ENCODER_LEVEL
            rows.append({"Pair": pair_name(pair), "Configuration": common.label(key),
                         "Bulk AUROC, mean": common.mean_over_seeds(raw),
                         "SD over encoder seeds": seed_sd(raw)[0] if trained else np.nan,
                         "SD over readout seeds": float(raw.groupby(["fold", "encoder_seed"])["auroc"].std(ddof=1).mean()),
                         "Seed ensemble AUROC": common.mean_over_seeds(common.aurocs(pair, key, [-1], variant="seed_ensemble",
                                                                                     **conditions)) if trained else np.nan})
    return pd.DataFrame(rows)


def within_adjusted() -> pd.DataFrame:
    """ALS's covariate-adjusted within-single-cell AUROC (11b's scores), beside the unadjusted value."""
    within = common.pick(common.scores(), bulk_subset="age_adjusted", side="single_cell_validation", row_kind="configuration")
    rows = []
    for pair in F.AGE_ADJUSTED["within"]:
        for key in common.kept():
            fits = within[(within["pair"] == pair) & (within["key"] == key)]
            raw = common.aurocs(pair, key, common.encoder_seeds(), side="single_cell_validation",
                                **common.headline_conditions(key))
            rows.append({"Pair": pair_name(pair), "Configuration": common.label(key),
                         "Within-single-cell AUROC": common.mean_over_seeds(raw),
                         "Within-single-cell AUROC, age-adjusted": fits["auroc"].mean(), "Estimable fits": len(fits)})
    return pd.DataFrame(rows)


def table_s5() -> dict:
    features = secondary_rows("feature_variants", key="pseudobulk")
    features = features[features["variant"].isin(("tmm", "vst")) | features["feature_set"].isin(PANELS_S5)]
    features = features.assign(change=[FEATURES[f if v == "primary" else v] for v, f in zip(features["variant"],
                                                                                            features["feature_set"])])
    changes = differences(features, UNADJUSTED, extra={"change": "Change"})
    return {"S5A": sheet(differences(secondary_rows("subgroups"), INTERVAL), "Within bulk subgroups", [
                "Group sizes per subgroup; not estimable where a label group is missing."]),
            "S5B": sheet(differences(secondary_rows("age_adjusted"), UNADJUSTED),
                         "Age-adjusted bulk scores (human ileum, Crohn's)"),
            "S5C": sheet(within_adjusted(), "Age-adjusted within-single-nucleus AUROC (human motor cortex, ALS)", [
                "Estimable fits: the encoder-seed × readout-seed fits for which an age-adjusted AUROC could be computed; the "
                "adjusted value is their mean."]),
            "S5D": sheet(seed_table(), "Encoder- and readout-seed variance and the seed ensemble"),
            "S5E": sheet(changes.drop(columns=[c for c in ("Variant", "Gene set") if c in changes]),
                         "Pseudobulk normalizations and gene panels", ["Pseudobulk under each change vs. primary pseudobulk."])}


# ---- Table S6 ----

def table_s6() -> dict:
    features = secondary_rows("feature_variants", feature_set=("full_genes", "mapped_orthologs"))
    genes = differences(features.assign(change=features["feature_set"].map(FEATURES)), UNADJUSTED, extra={"change": "Change"})
    curve = pd.concat([secondary_rows(f"learning_curve_{f}").assign(percent=f) for f in (25, 50, 100)], ignore_index=True)
    curve = differences(curve, UNADJUSTED, extra={"percent": "Training donors (%)"})   # stable: 25, 50, 100 per configuration
    uce = " UCE: Table S6C." if F.S12B_UCE else ""
    out = {"S6A": sheet(genes.drop(columns="Gene set"), "Full gene sets and native mouse genes",
                        [f"Differences from primary pseudobulk.{uce}"]),
           "S6B": sheet(curve.drop(columns="Variant"), "Learning curves (5 seeds)",
                        ["Differences from pseudobulk at the same fraction of training donors."])}
    if F.S12B_UCE:
        rows = [{"Pair": pair_name(p),
                 "UCE bulk AUROC": common.pick(common.scores(), pair=p, model="uce", side="bulk",
                                               bulk_subset="all")["auroc"].mean()}
                for p in PAIR_ORDER]
        out["S6C"] = sheet(pd.DataFrame(rows), "UCE")
    return out


# ---- Table S7 ----

def table_s7() -> dict:
    rows = []
    for pair in PAIR_ORDER:
        keys = common.kept()
        silhouettes = fig3.silhouettes(pair, keys, ALIGNMENTS).set_index(["key", "alignment"])["value"]
        aurocs = fig3.aurocs(pair, keys, ALIGNMENTS).set_index(["key", "alignment"])["value"]
        for key in keys:
            raw = pd.concat([fig3.geometry(pair, key, u) for u in common.primary_units(pair)])
            rows.append({"Pair": pair_name(pair), "Configuration": common.label(key),
                         "Silhouette, raw": float(raw.loc[raw["measure"] == "silhouette_raw", "value"].median()),
                         **{f"Silhouette, {ALIGNMENT_NAMES[a]}": silhouettes[(key, a)] for a in ALIGNMENTS},
                         **{f"Bulk AUROC, {ALIGNMENT_NAMES[a]}": aurocs[(key, a)] for a in ALIGNMENTS}})
    differences_ = stacked([differences(secondary_rows(f"alignment_{a}").assign(shown=ALIGNMENT_NAMES[a]),
                                        {**INTERVAL, "u": STATISTICS["u"]}, extra={"shown": "Alignment"})
                            for a in ALIGNMENTS if a != "none"])
    return {"S7A": sheet(pd.DataFrame(rows), "Modality silhouette and bulk AUROC under each alignment", [
                "Modality silhouette of single-cell vs. bulk samples: raw, and after per-side scaling under each alignment. "
                "Trained models: median silhouette and mean AUROC over seeds."]),
            "S7B": sheet(differences_, "Difference in bulk AUROC from pseudobulk under each alignment", [
                "Each configuration vs. pseudobulk aligned the same way, per pair and pooled with equal weights over the pairs "
                f"with pseudobulk bulk AUROC below {common.threshold()} ({F.FIGURE_TEXT['pooled_mark']}); two-sided 95% bootstrap "
                "CIs and the one-sided 95% upper bound u. Descriptive (no test)."])}


# ---- Table S8 ----

def candidate_name(r) -> str:
    """A best-candidate row's name: the search candidate, or the bag size (pooling) of a configuration's point."""
    if r["row_kind"] == "candidate":
        return candidate_label(r["model"], r["key"])
    if r["cell_subset"] == "all_available":
        return "all available cells"
    if r["pooling"] == "max":
        return "max pooling"
    return cells(r["n_cells"]) + (" (selected configuration)" if r["n_cells"] == T.FROZEN_NCELL and r["key"] in SELECTED else "")


def cells(n) -> str:
    return f"{n} cell" + ("s" if int(n) != 1 else "")


def member_id(r) -> str:
    """A best-candidate row's id as 11b's notes name it ("cytoset/out_max", "cytoset_cells@25all", "scgpt_cells@max")."""
    if r["row_kind"] == "candidate":
        return f"{r['model']}/{r['key']}"
    if r["pooling"] == "max":
        return f"{r['key']}@max"
    return f"{r['key']}@{r['n_cells']}" if r["n_cells"] == T.FROZEN_NCELL and r["key"] != "pseudobulk" else \
        f"{r['key']}@{r['n_cells']}{r['cell_subset']}"


def carrier_name(carrier) -> str:
    """11b's carrier id ("cytoset/out_max", "cytoset_cells@500", "cytoset_cells@0all_available") as table text."""
    if carrier is None:
        return NA
    if "/" in carrier:
        encoder, name = carrier.split("/")
        return f"{model_label(encoder)}: {candidate_label(encoder, name)}"
    key, n = carrier.split("@")
    bag = "max pooling" if n == "max" else "all available cells" if n.endswith("all_available") else cells(n.removesuffix("all"))
    return f"{common.label(key)} at {bag}" + (" (the selected configuration)" if n == str(T.FROZEN_NCELL) else "")


def in_test(r) -> bool:
    """The best-candidate test's candidate set: every set-encoder search candidate and the set encoders' bag-size points."""
    if r["row_kind"] == "candidate":
        return r["model"] in SET_ENCODERS and r["key"] != "objective_mse"
    return r["key"] in SELECTED


def best_candidate_members() -> pd.DataFrame:
    """S8A: every member of 11b's best-candidate call with its change from the base (stage 05) and its interval."""
    members = secondary_rows("best_candidate")
    members = members.assign(_pair=members["pair"].map(PAIR_ORDER.index), _model=members["model"].map(MODELS_S8.index),
                             _kind=members["row_kind"].ne("candidate"))
    members = members.sort_values(["_pair", "_model", "_kind", "n_cells"], kind="stable")
    changes = pd.concat([fig4.candidate_groups(p, e).assign(pair=p, model=e) for p in PAIR_ORDER for e in fig4.encoders()])
    members = members.merge(changes, on=["pair", "model", "key"], how="left")
    groups = [group_label(g) if isinstance(g, str) else "merged" if k.startswith("merged") else ""   # merged: in no group
              for g, k in zip(members["change_group"], members["key"])]
    excluded = {(p, c) for p, made in notes_11b()["best_candidate"].items() for c in made["excluded_zero_se"]}
    tested = ["no (interval only)" if not in_test(r) else "no (zero bootstrap SE)" if (r["pair"], member_id(r)) in excluded
              else "yes" for _, r in members.iterrows()]
    return pd.DataFrame({"Pair": members["pair"].map(pair_name), "Model": members["model"].map(model_label),
                         "Candidate": [candidate_name(r) for _, r in members.iterrows()],
                         "Change group": groups,
                         "Change": [change_text(v) if isinstance(v, str) else "" for v in members["change_value"]],
                         "Tested": tested,
                         **{name: members[c].to_numpy() for c, name in INTERVAL.items() if c not in ("n0", "n1")},
                         "Estimable": members["estimable"].map({True: "yes", False: "no"})})


def best_candidate_tests() -> pd.DataFrame:
    """S8B: each pair's max-statistic test, its Holm value across pairs, and the candidate carrying the maximum."""
    notes = notes_11b()["best_candidate"]
    tests = common.pick(common.secondary(), neutral=False, row="best_candidate", key="best_candidate").set_index("pair")
    selected = {f"{k}@{T.FROZEN_NCELL}" for k in SELECTED}
    return pd.DataFrame([{"Pair": pair_name(p), "Max statistic T": tests.loc[p, "delta"],
                          "p (max statistic)": tests.loc[p, "p_better"],
                          "p (Holm across pairs)": tests.loc[p, "p_better_holm"], "Candidates tested": notes[p]["candidates"],
                          "Candidate carrying the maximum": carrier_name(notes[p]["carrier"]),
                          "Maximum at the selected configuration": "yes" if notes[p]["carrier"] in selected else "no",
                          "Excluded (zero SE)": "; ".join(map(carrier_name, notes[p]["excluded_zero_se"]))} for p in PAIR_ORDER])


def untested() -> pd.DataFrame:
    """S8C: candidates listed but not tested: zero bootstrap SE, and 11b's not-estimable bag-size points."""
    notes = notes_11b()
    rows = [{"Pair": pair_name(p), "Candidate": carrier_name(c), "Reason": "zero bootstrap SE"}
            for p in PAIR_ORDER for c in notes["best_candidate"][p]["excluded_zero_se"]]
    rows += [{"Pair": pair_name(n.split()[1]), "Candidate": carrier_name(n.split(":")[0].split()[2]),
              "Reason": n.split(": ", 1)[1]}
             for n in notes["not_estimable"] if n.startswith("bag_size ")]
    return pd.DataFrame(rows or [{"Pair": "(none)", "Candidate": "", "Reason": "every listed candidate was tested"}])


def seed_spread() -> pd.DataFrame:
    """S8D: per pair and encoder, the SD over the search seeds of each candidate's per-seed bulk AUROC (each seed's mean
    over readout seeds, then folds: the encoder-seed SD the max-statistic simulation varied), its median and maximum
    over the candidates, and the selected configuration's SD over the encoder seeds (Table S2A)."""
    rows = []
    for pair in PAIR_ORDER:
        for encoder in fig4.encoders():
            sds = [seed_sd(common.aurocs(pair, name, common.search_seeds(), row_kind="candidate", model=encoder,
                                         n_cells=T.FROZEN_NCELL, pooling="native"))[0]
                   for name in fig4.candidate_groups(pair, encoder)["key"]]
            key = f"{encoder}_cells"
            selected = seed_sd(common.aurocs(pair, key, **common.headline_conditions(key)))[0]
            rows.append({"Pair": pair_name(pair), "Encoder": model_label(encoder), "Candidates": len(sds),
                         "SD of per-seed bulk AUROC over search seeds, median over candidates": float(np.median(sds)),
                         "SD of per-seed bulk AUROC over search seeds, max over candidates": float(np.max(sds)),
                         "Selected configuration: SD over encoder seeds": selected})
    return pd.DataFrame(rows)


def table_s8() -> dict:
    return {"S8A": sheet(best_candidate_members(), "Each candidate's difference from full pseudobulk (mean over 5 seeds)", [
                "Bag-size rows: the selected configuration at each bag size (Fig 4a). Interval-only analyses (the frozen foundation "
                "models, the unsupervised generative model (scVI), pseudobulk of n cells, and CytoSet's MSE objective) are not "
                "in the test."]),
            "S8B": sheet(best_candidate_tests(), "Best-candidate max-statistic test per pair", [
                "Per pair, unadjusted across pairs; the Holm value across the six pairs is given beside it."]),
            "S8C": sheet(untested(), "Candidates listed but not tested"),
            "S8D": sheet(seed_spread(), "Observed encoder-seed SD of bulk AUROC", [
                f"Search seeds: {len(common.search_seeds())}; encoder seeds of the selected configuration: "
                f"{len(common.encoder_seeds())}. In simulation, the test may reject a true null more often than 5% when this "
                "SD exceeds about 0.045."])}


# ---- Table S9 ----

def simulation_table(criterion) -> pd.DataFrame:
    """S9E: Fig 5D's means per scenario and model, the values that hold by construction named, and each supervised set
    encoder's seeds passing the training check (stage 12's criterion.json)."""
    table = common.read_written("Fig 5", "D")
    passed = criterion["did_train_passed"]
    rows = []
    for (scenario, key), part in table[table["seed"].isna()].groupby(["scenario", "key"], sort=False):
        values = part.set_index("measure")
        rows.append({"Scenario": F.TEXT["scenario"][scenario], "Model": common.label(key),
                     **{f"{name} AUROC": values.loc[measure, "auroc"] for measure, name in MEASURES.items()},
                     "Holds by constructionᵃ": ", ".join(name.lower() for measure, name in MEASURES.items()
                                                         if values.loc[measure, "by_construction"]),
                     "Seeds passing the training check": passed.get(f"{key} {scenario}", NA)})
    return pd.DataFrame(rows)


def passing_totals(criterion) -> str:
    """The supervised set encoders' seeds passing the training check, together, per scenario."""
    totals = []
    for scenario in SIM.SCENARIOS:
        counts = [criterion["did_train_passed"][f"{m}_cells {scenario}"].split("/") for m in criterion["set_encoders"]]
        passed, trained = sum(int(a) for a, _ in counts), sum(int(b) for _, b in counts)
        totals.append(f"{F.TEXT['scenario'][scenario].lower()}, {passed} of {trained}")
    return "Seeds passing the training check, the supervised set encoders together: " + "; ".join(totals) + "."


def pending_rows(pair, present, columns, among=None) -> pd.DataFrame:
    """Rows for the configurations stage 10 dropped in a pair (the Geneformer deletion pending or triaged) that the table
    lacks (among: the configurations the table covers; default all): the configuration named, the status in the first
    value column."""
    dropped = {k: v for k, v in common.dropped(pair).items() if k not in set(present) and (among is None or k in among)}
    return pd.DataFrame([{"Pair": pair_name(pair), "Configuration": common.label(k), columns[0]: F.FIGURE_TEXT[F.DROPPED_TEXT[v]]}
                         for k, v in dropped.items()], columns=["Pair", "Configuration", *columns])


def joined(table, extra) -> pd.DataFrame:
    """A table with the extra rows below it (none: the table as is)."""
    return pd.concat([table, extra], ignore_index=True) if len(extra) else table


def pending_note() -> list:
    """A sub-table note naming the pairs where the Geneformer deletion attribution was still running, if any."""
    pairs = [pair_name(p) for p in PAIR_ORDER if "pending" in common.dropped(p).values()]
    return [f"Geneformer: {F.FIGURE_TEXT['pending']} in {'; '.join(pairs)}; its gene-deletion attribution was still "
            "running."] if pairs else []


def threshold(value) -> object:
    """A check's threshold as printed: a range such as "[0.5, 2.0]" as written, a number to 3 decimals."""
    if isinstance(value, str) and value.startswith("["):
        return value
    return round(float(value), 3)


def included(inclusion, configuration) -> str:
    """Whether stage 10 included a configuration's genes in Fig 5, with the checks it failed (or why it was dropped)."""
    if configuration not in inclusion.index:
        return NA
    if inclusion.loc[configuration, "included"]:
        return "yes"
    failed = inclusion.loc[configuration, "failed_checks"]
    if failed in F.DROPPED_TEXT:
        return f"no ({F.FIGURE_TEXT[F.DROPPED_TEXT[failed]]})"
    return f"no ({'; '.join(CHECK_NAMES[c] for c in failed.split(';'))})"


def gene_tables(pair) -> tuple:
    """(concordance, shares, stability, checks) of one pair: stage 10c's concordance, stage 10's checks and inclusion."""
    concordance = fig5.concordance(pair)
    assert concordance["rho"].nunique() == 1, f"S9 {pair}: rho varies"
    rho = {"Pair": pair_name(pair), "Spearman ρ (single-cell vs. bulk phenotype shifts)": concordance["rho"].iloc[0]}
    part = ordered(common.pick(concordance, neutral=False, readout=("ridge", "lasso"), k=(25, 50, 100)))
    shares = pd.DataFrame({"Pair": pair_name(pair), "Configuration": part["key"].map(common.label),
                           "Readout": part["readout"].map(READOUT_NAMES), "Top k": part["k"],
                           "Same-sign share": part["shared_share"],
                           "Reference share, all genes": part["base_all"],
                           "Reference share, expression-matched": part["base_matched"],
                           "Included": part["included"].map({True: "yes", False: "no"})})
    shares = joined(shares, pending_rows(pair, part["key"], list(shares.columns[2:])))
    table = s14.checks(pair)
    stable = table[table["check"].isin(s14.STABILITY)].pivot_table(index=["configuration", "readout", "target"], columns="check",
                                                                    values="value")
    stable = ordered(stable.reset_index().rename(columns={"configuration": "key"}).assign(pair=pair))
    stability = pd.DataFrame({"Pair": pair_name(pair), "Configuration": stable["key"].map(common.label),
                              "Readout": stable["readout"].map(READOUT_NAMES), "Target": stable["target"].map(TARGETS),
                              **{f"Top-{G.TOP_GENES} overlap, {F.STABILITY_LABELS[c]}": stable[c] for c in s14.STABILITY
                                 if c in stable}})
    stability = joined(stability, pending_rows(pair, stable["key"], list(stability.columns[2:])))
    inclusion = common.read_csv("10_attribution", pair, "inclusion.csv", keep_default_na=False).set_index("configuration")
    gated = table[table["passed"].notna()]
    checks = pd.DataFrame({"Pair": pair_name(pair), "Configuration": gated["configuration"].map(common.label),
                           "Readout": gated["readout"].map(READOUT_NAMES), "Target": gated["target"].map(TARGETS),
                           "Fold": [unit_name(u) if u[1] != -1 else "across folds"
                                    for u in zip(gated["fold_set"], gated["fold"])],
                           "Half": gated["half"].map(lambda h: f"half {h.upper()}" if isinstance(h, str) and h else ""),
                           "Check": gated["check"].map(lambda c: CHECK_NAMES[c]),
                           "Value": [("yes" if float(v) else "no") if c == "hk_dominated" else float(v)
                                     for c, v in zip(gated["check"], gated["value"])],
                           "Threshold": gated["threshold"].map(threshold),
                           "Passed": gated["passed"].astype(str).map({"True": "yes", "False": "no"}),
                           "Included in Fig 5": gated["configuration"].map(lambda c: included(inclusion, c))})
    checks = joined(checks, pending_rows(pair, gated["configuration"], list(checks.columns[2:])))
    return rho, shares, stability, checks


def table_s9() -> dict:
    rho, shares, stability, checks = zip(*[gene_tables(p) for p in PAIR_ORDER])
    criterion = common.read_json("12_simulation", SIM.PAIR, "criterion.json", SIM.UNIT)
    met = [{"Model": common.label(f"{m}_cells" if m != "mean_log_cells" else m),
            "Within-single-cell AUROC minus pseudobulk's (structure only)": criterion["delta"][m],
            "Margin": criterion["margin"], "Criterion met": "yes" if criterion["passes"][m] else "no",
            "Interpretation": wording(criterion["structure_readings"][m]) if m != "mean_log_cells" else "reference (a fixed aggregator)"}
           for m in [*criterion["set_encoders"], "mean_log_cells"]]
    return {"S9A": sheet(pd.DataFrame(rho), "Concordance of single-cell and bulk phenotype shifts"),
            "S9B": sheet(pd.concat(shares, ignore_index=True), "Same-sign shares (top 25, 50, and 100 genes; ridge and LASSO)",
                         pending_note()),
            "S9C": sheet(pd.concat(stability, ignore_index=True), "Gene-list stability", [
                f"Each value: the share of the top {G.TOP_GENES} genes that two gene rankings have in common (across seeds "
                "and across folds: the mean over pairs of rankings)."] + pending_note()),
            "S9D": sheet(pd.concat(checks, ignore_index=True), "Attribution checks and Geneformer's inclusion", pending_note()),
            "S9E": sheet(simulation_table(criterion), f"Simulation ({pair_name(SIM.PAIR)}; mean over {len(SIM.SPLITS)} splits)",
                         ["ᵃ Values that hold by construction: references, not results.", passing_totals(criterion)]),
            "S9F": sheet(pd.DataFrame(met), "The simulation criterion", [
                "Criterion: under structure only, each supervised set encoder's within-single-cell AUROC exceeds "
                f"pseudobulk's by at least {criterion['margin']}; the mean of log-normalized cells, a fixed aggregator, is the "
                "reference. The positive control passes if every supervised set encoder meets the criterion: "
                f"{'yes' if criterion['control_passes'] else 'no'}."])}


# ---- Table S10 ----

def property_table() -> pd.DataFrame:
    """S10A: each representation's split-mean value per property (stage 12a) with its permutation null."""
    rows = []
    for pair in PAIR_ORDER:
        keys = [*F.FIG6A_ROWS, *[k for k in common.kept() if k not in F.FIG6A_ROWS]]
        table = pd.concat([common.read_kind("property_r2", "12a_properties", pair, f"{k}_property_r2.parquet", u)
                           for k in keys for u in common.primary_units(pair)], ignore_index=True)
        nulls = table.groupby(["key", "property"])[["null_mean", "null_q95"]].first().reset_index()
        values = properties.split_mean(table).merge(nulls, on=["key", "property"])
        values = ordered(values.assign(_p=values["property"].map(P.PROPERTIES.index)).sort_values("_p", kind="stable"))
        rows.append(pd.DataFrame({"Pair": pair_name(pair), "Representation": values["key"].map(common.label),
                                  "Property": values["property"].map(PROPERTY_NAMES), "Cross-validated value": values["r2"],
                                  "Permutation null, mean": values["null_mean"],
                                  "Permutation null, 95th percentile": values["null_q95"]}))
    return pd.concat(rows, ignore_index=True)


def modality_genes() -> pd.DataFrame:
    """S10C: each cells configuration's top modality-direction genes (stage 10), with the level difference (the primary
    units' mean); a dropped Geneformer deletion is named, and one whose checks failed is marked n/a."""
    rows = []
    for pair in PAIR_ORDER:
        units = common.primary_units(pair)
        level = pd.concat([common.read_csv("10_attribution", pair, "level_difference.csv", u) for u in units])
        delta = level.groupby("gene")["delta"].mean()
        dropped = common.dropped(pair)
        excluded = common.excluded(pair)
        for key in [k for k in common.kept() if k.endswith("_cells")]:
            if key in dropped:
                rows.append({"Pair": pair_name(pair), "Configuration": common.label(key), "Rank": np.nan,
                             "Gene": F.FIGURE_TEXT[F.DROPPED_TEXT[dropped[key]]]})
                continue
            if key in excluded:
                rows.append({"Pair": pair_name(pair), "Configuration": common.label(key), "Rank": np.nan,
                             "Gene": "n/a: Geneformer's checks failed (Additional file 1: Fig. S14d)"})
                continue
            ranking = common.read_kind("attributions", "10_attribution", pair, f"{key}_ranking.parquet")
            top = common.pick(ranking, neutral=False, target="modality_direction").sort_values("rank").head(F.FIG6B_TOP)
            rows += [{"Pair": pair_name(pair), "Configuration": common.label(key), "Rank": r.rank, "Gene": r.gene,
                      "Attribution": r.attribution, "Bulk minus single-cell pseudobulk expression": delta.loc[r.gene]}
                     for r in top.itertuples()]
    return pd.DataFrame(rows)


def modality_summary() -> pd.DataFrame:
    """S10D: stage 10's cross-model summary (the across-fold row for human blood, aging)."""
    rows = []
    for pair in PAIR_ORDER:
        units = common.primary_units(pair)
        summary = common.read_csv("10_attribution", pair, "modality_summary.csv")
        fold = -1 if len(units) > 1 else units[0][1]
        summary = ordered(summary[summary["fold"] == fold].rename(columns={"configuration": "key"}))
        rows.append(pd.DataFrame({"Pair": pair_name(pair), "Configuration": summary["key"].map(common.label),
                                  "Share, large level difference": summary["share_large"],
                                  "Share, tissue marker": summary["share_marker"],
                                  "Share, either": summary["share_either"], "Base rate, large": summary["base_large"],
                                  "Base rate, marker": summary["base_marker"], "Base rate, either": summary["base_either"]}))
        cells = [k for k in common.kept() if k.endswith("_cells")]
        rows[-1] = joined(rows[-1], pending_rows(pair, summary["key"], list(rows[-1].columns[2:]), cells))
    return pd.concat(rows, ignore_index=True)


def licence_count(licence, pair, prop) -> float:
    """licence.json's count of configurations exceeding both baselines (0 where it lists none; n/a for mouse organs'
    mitochondrial fraction, which has no mitochondrial genes)."""
    if pair in P.MITO_NA_PAIRS and prop == "mito_fraction":
        return np.nan
    return licence["per_pair"].get(pair, {}).get(prop, 0)


def table_s10() -> dict:
    licence = common.read_json("12a_properties", "", "licence.json")
    n = licence["n_present"]
    header = f"Learned configurations exceeding both pseudobulk baselines (of {n})"
    counts = pd.DataFrame([{"Pair": pair_name(p), "Property": PROPERTY_NAMES[prop], header: licence_count(licence, p, prop),
                            "Pair qualifies": "yes" if p in licence["qualifying_pairs"] else "no"}
                           for p in PAIR_ORDER for prop in P.TECHNICAL_PROPERTIES]).astype({header: "Int64"})
    qualifying = licence["qualifying_pairs"]
    one_type = common.read_written("Fig 6", "C")
    one_type = pd.DataFrame({"Pair": one_type["pair"].map(pair_name), "Cell type": one_type["cell_type"],
                             "Configuration": one_type["key"].map(common.label),
                             "Within-single-cell AUROC": one_type["within_auroc"],
                             "Bulk AUROC": one_type["bulk_auroc"]})
    return {"S10A": sheet(property_table(), "Cross-validated R² (continuous properties) and AUROC (phenotype)", [
                f"Pseudobulk on up to {P.PCA_COMPONENTS} principal components: the PCA-64 row. Permutation nulls over "
                f"{P.N_NULL} permutations."]),
            "S10B": sheet(counts, "Pairs in which the learned configurations exceeded both pseudobulk baselines", [
                f"The {n} learned configurations: every configuration but pseudobulk. Baselines: pseudobulk and its PCA-64 "
                f"row. A pair qualifies when more than half of the {n} exceed both baselines on every technical property that "
                f"applies to it: {len(qualifying)} of {len(PAIR_ORDER)} pairs"
                + (f" ({'; '.join(pair_name(p) for p in qualifying)})" if qualifying else "")
                + f"; at least {P.LICENCE_MIN_PAIRS} are needed for a general claim."]),
            "S10C": sheet(modality_genes(), f"Modality-direction genes (top {F.FIG6B_TOP} per cells configuration)",
                          pending_note()),
            "S10D": sheet(modality_summary(), "Cross-model summary of the modality-direction genes", pending_note()),
            "S10E": sheet(one_type, "One cell type only")}


# ---- Table S11 ----

def configuration_of(r) -> str:
    """A work unit's configuration, from its run record: the record's configuration; stage 06's row and model (row B is
    "CytoSet, pseudobulk"; rows C and MSE, the untrained and MSE controls, are shared; other rows train their model's
    cells configuration, or the configuration the model names); stage 08b's job name when it starts with a configuration;
    otherwise "shared"."""
    if pd.notna(r["configuration"]):
        return common.label(r["configuration"])
    if r["stage"] == "06_train":
        key = {"B": "cytoset_pseudobulk", "C": None, "MSE": None}.get(r["row"], r["model"] if r["model"] in F.DISPLAY_ORDER
                                                                       else f"{r['model']}_cells")
        return common.label(key) if key else "shared"
    if pd.notna(r["name"]) and r["name"].split("/")[0] in F.DISPLAY_ORDER:
        return common.label(r["name"].split("/")[0])
    return "shared"


def cost_units() -> pd.DataFrame:
    """One row per work unit, identified by its run record (stage, settings hash, seed): its outputs carry its cost so
    far, so the unit's cost is the largest, wherever its outputs are written; a GPU
    unit if any output ran on CUDA."""
    costs = common.costs()
    assert len(costs), "S11: no manifest costs under the outputs"
    costs = costs.assign(gpu_unit=costs["device"].str.startswith("cuda"), seed=costs["seed"].astype(str))
    units = costs.groupby(["stage", "settings_hash", "seed"], sort=False).agg(
        pair=("pair", "first"), seconds=("seconds", "max"), gpu_unit=("gpu_unit", "any"), outputs=("path", "size"),
        gpu=("gpu", "first"), configuration=("configuration", "first"), row=("row", "first"), model=("model", "first"),
        name=("name", "first")).reset_index()
    return units.assign(configuration=[configuration_of(r) for _, r in units.iterrows()],
                        gpu_seconds=units["seconds"].where(units["gpu_unit"], 0.0))


def table_s11() -> dict:
    units = cost_units()
    table = units.groupby(["stage", "pair", "configuration"]).agg(seconds=("seconds", "sum"), gpu_seconds=("gpu_seconds", "sum"),
                                                                  units=("seconds", "size"), outputs=("outputs", "sum"))
    table = table.reset_index().assign(_pair=lambda t: t["pair"].map({p: i for i, p in enumerate(PAIR_ORDER)}))
    table = table.sort_values(["stage", "_pair"], kind="stable")
    table = pd.DataFrame({"Analysis step": table["stage"].map(lambda s: STAGE_NAMES[s]),
                          "Pair": table["pair"].map(lambda p: pair_name(p) if p in PAIRS else "all pairs"),
                          "Configuration": table["configuration"], "Runs": table["units"], "Outputs": table["outputs"],
                          "Wall-clock hours": table["seconds"] / 3600, "GPU hours": table["gpu_seconds"] / 3600})
    totals = table.groupby("Analysis step", sort=False)[["Wall-clock hours", "GPU hours"]].sum().reset_index()
    gpus = sorted(units.loc[units["gpu_unit"], "gpu"].dropna().unique())
    hardware = pd.DataFrame([{"Machine": "Laboratory server", "Processor": f"GPU: {g}"} for g in gpus] +
                            [{"Machine": "Laboratory server", "Processor": "CPU"}])
    return {"S11A": sheet(table, "Per analysis step, pair, and configuration", [
                "A run is one execution of an analysis step (one configuration, seed, or fold, as the step is organized); "
                "outputs are the files it wrote. Each run's time is its recorded wall-clock time; a run that serves several "
                "configurations is listed as shared. GPU hours are the wall-clock hours of runs on a GPU, where up to "
                "two small jobs could share one GPU. Recorded time excludes starting Python and loading libraries (about "
                "0.6 s per run)."]),
            "S11B": sheet(totals, "Totals per analysis step"),
            "S11C": sheet(hardware, "Hardware", [HAND["hardware"]])}


# ---- Table S12 ----

def table_s12() -> dict:
    table, notes = block(drafted("Table S12").split("\n"))
    return {"S12": sheet(table, None, notes)}


TABLES = {"Table S1": table_s1, "Table S2": table_s2, "Table S3": table_s3, "Table S4": table_s4, "Table S5": table_s5,
          "Table S6": table_s6, "Table S7": table_s7, "Table S8": table_s8, "Table S9": table_s9, "Table S10": table_s10,
          "Table S11": table_s11, "Table S12": table_s12}


# ---- writing ----

def heading(item, name, subtitle) -> str:
    """A sub-table's heading ("Table S1A. ..."), or the table's title where the sheet has no subtitle."""
    return f"{item}{name.removeprefix(item.removeprefix('Table '))}. {subtitle}" if subtitle else f"{item}. {F.TITLES[item]}"


def markdown_table(frame) -> list:
    cells = shown(frame).astype(str).map(lambda v: v.replace("|", "\\|").replace("\n", " "))
    lines = ["| " + " | ".join(cells.columns) + " |", "|" + "---|" * len(cells.columns)]
    return lines + ["| " + " | ".join(r) + " |" for r in cells.itertuples(index=False)]


def markdown(item, sheets, lines) -> str:
    out = [f"**{item}. {F.TITLES[item]}**", ""]
    for name, (subtitle, frame, notes) in sheets.items():
        out += [f"**{heading(item, name, subtitle)}**", ""] if subtitle else []
        out += [*markdown_table(frame), "", *notes, ""]
    return "\n".join([*out, *lines, ""])


def workbook(path, built):
    """One sheet per sub-table: the title, the subtitle, the table, its notes, then the table's legend and footnotes."""
    def write(temporary):
        with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
            for item, (sheets, lines) in built.items():
                for name, (subtitle, frame, notes) in sheets.items():
                    numbers = rounded(frame)
                    cells_ = pd.DataFrame({c: [v if isinstance(v, float) and np.isfinite(v) else text(v, c) for v in numbers[c]]
                                           for c in numbers.columns}, columns=numbers.columns)   # numbers stay numbers
                    cells_.to_excel(writer, sheet_name=f"Table {name}", startrow=3, index=False)
                    page = writer.sheets[f"Table {name}"]
                    for j, column in enumerate(numbers.columns, start=1):   # decimals to 3 places, p values as they are
                        if pd.api.types.is_float_dtype(numbers[column]) and not is_p(column) and column not in GENERAL:
                            for i in range(len(numbers)):
                                page.cell(row=5 + i, column=j).number_format = "0.000"
                    page.cell(row=1, column=1, value=f"{item}. {F.TITLES[item]}")
                    page.cell(row=2, column=1, value=heading(item, name, subtitle) if subtitle else "")
                    for i, line in enumerate([*notes, "", *lines]):
                        page.cell(row=len(frame) + 6 + i, column=1, value=line)
    common.write_bytes(path, write)


def main():
    common.start()
    built = {}
    for item in ITEMS:
        built[item] = (TABLES[item](), legend(item))
        print(f"{item}: " + ", ".join(f"{name} {len(frame)} rows" for name, (_, frame, _) in built[item][0].items()), flush=True)
    out = settings.LABELLED_OUTPUTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for item, (sheets, lines) in built.items():
        directory = out / item.replace(" ", "")
        directory.mkdir(exist_ok=True)
        for name, (_, frame, _) in sheets.items():
            common.write_bytes(directory / f"Table{name}.csv", lambda p: shown(frame).to_csv(p, index=False))
        common.write_bytes(directory / f"{directory.name}.md", lambda p: p.write_text(markdown(item, sheets, lines)))
    workbook(out / "AdditionalFile2.xlsx", built)
    panels = [common.directory(item) / f"{common.panel_name(item, panel)}.csv" for item, panel in PANELS]
    record = {"item": "Additional file 2", "script": common.blob_hash(__file__),
              **common.sources(common.INPUTS, common.REPOSITORY),
              "figure_panels": [{"path": str(p.relative_to(settings.OUTPUTS)), "blob": common.blob_hash(p)} for p in panels]}
    common.write_bytes(out / "sources.json", lambda p: p.write_text(json.dumps(record, indent=1, sort_keys=True)))
    return built


if __name__ == "__main__":
    main()
