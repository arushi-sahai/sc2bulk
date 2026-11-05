"""One long-format schema per output kind.

Every table written through write() carries the key columns (KEYS) plus its kind's own
columns. write() and read() assert the columns, dtypes, allowed values, valid keys, uniqueness
of the key + identity columns, and SCHEMA_VERSION (stored in the Parquet metadata). Kinds
flagged label_bearing hold bulk-label-derived numbers: they're written under a marked file
name (LABELLED prefix + "_"), and the firewall test allows only its whitelist to name that
prefix, so no other file can reach them. Every difference is alternative - pseudobulk.
"""
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from sc2bulk.config import configurations, search_space, settings
from sc2bulk.config.pairs import PAIR_ORDER

SCHEMA_VERSION = 1
LABELLED = settings.LABELLED   # file-name prefix of label-bearing kinds (written with an underscore)

# key column -> (dtype, neutral value or None (= required), allowed values or None)
KEYS = {
    "pair": ("str", None, (*PAIR_ORDER, "pooled")),   # "pooled" only on pooled statistics rows
    "fold_set": ("str", None, tuple(settings.FOLD_SET_IDS)),
    "fold": ("int", None, None),            # -1 = across folds: a crossfit set's per-fold mean or pooled predictions; pooled rows (fold_set "all")
    "row_kind": ("str", None, ("configuration", "reference", "candidate")),
    "key": ("str", None, None),
    "model": ("str", None, (*configurations.MODELS, "uce", "none")),
    "input": ("str", None, ("cells", "pseudobulk", "none")),
    "encoder_seed": ("int", -1, None),
    "readout": ("str", "ridge", (*configurations.READOUTS, "native_head", "none")),
    "readout_seed": ("int", -1, None),
    "alignment": ("str", "none", (*configurations.ALIGNMENTS, "unaligned_stacked")),
    "n_cells": ("int", None, None),         # required: the actual test-time bag size (0 = all stored cells / a profile)
    "feature_set": ("str", "panel", ("panel", "full_genes", "mapped_orthologs", "all_shared",
                                     *(f"panel_{n}" for n in settings.PANEL_SIZE_SENSITIVITY))),
    "cell_subset": ("str", "all", ("all", "one_type", "all_available")),
    "permutation": ("int", -1, None),
    "pooling": ("str", None, ("native", "mean", "max")),   # required: native for set encoders, mean/max for embed-only
    "variant": ("str", "primary", ("primary", "tmm", "vst", "training_fraction_0.25", "training_fraction_0.5",
                                   "out_of_fold_readout", "outside_moments_scaling", "per_sample_route_sensitivity",
                                   "numeric_age_readout", "pooled_predictions", "thinned_pseudocells", "observed_library",
                                   "seed_ensemble", "raw_fractions", "no_ifn_markers", "pooled_z", "median_centered")),
}
RESERVED_CANDIDATES = ("base", "merged", "merged_no_per_sample", "selected", "objective_mse", "scrat_own_recipe", "best_candidate",
                       "cytoset_own_recipe", "set_transformer_own_recipe")
EMBED_ONLY = {"geneformer", "scgpt", "scvi", "uce"}

# kind -> (identity columns (with the keys, unique per row), other columns, label_bearing)
KINDS = {
    "candidates": ((), ("change_group", "change_value", "validation_bce", "did_train", "did_train_reason",
                        "selected", "merged"), False),
    "training_records": ((), ("epochs", "stop_epoch", "validation_curve", "did_train", "did_train_reason", "seconds"), False),
    "predictions": (("side", "sample_id"), ("scoring_unit_id", "score", "probability"), False),
    "heads": (("side", "stratum", "gene_or_dim"), ("coef", "intercept", "scaler_mean", "scaler_scale", "alpha",
                                                   "alpha_at_boundary", "converged", "n_iter", "n_nonzero"), False),
    "attributions": (("target", "method", "gene"), ("attribution", "rank"), False),
    "property_r2": (("property",), ("cv_r2", "null_mean", "null_q95"), False),
    "alignment_geometry": (("measure", "sample_id"), ("value",), False),
    "simulation": (("scenario",), ("within_sc_auroc", "simulated_bulk_auroc"), False),
    "reference_predictions": (("side", "sample_id"), ("scoring_unit_id", "score", "probability"), True),
    "scores": (("bulk_subset", "side"), ("n0", "n1", "auroc", "auprc", "brier", "signed_log10_p"), True),
    "statistics": (("pool_set", "statistic"), ("delta", "ci_low", "ci_high", "u", "p_better", "p_worse", "p_better_holm",
                                              "p_worse_holm", "mdd_obs", "resamples", "n0", "n1", "excluded_pairs"), True),
    # stage 11b's rows: several per configuration (one per row and bulk subset); estimable False -> NaN statistics
    "secondary_statistics": (("row", "pool_set", "statistic", "bulk_subset"),
                             ("delta", "ci_low", "ci_high", "u", "p_better", "p_worse", "p_better_holm", "p_worse_holm",
                              "mdd_obs", "resamples", "n0", "n1", "excluded_pairs", "estimable"), True),
    "gene_shifts": (("gene",), ("single_cell_shift", "bulk_shift"), True),
    "concordance": (("k",), ("shared_share", "base_all", "base_matched"), True),
}
SIDES = ("single_cell_train", "single_cell_validation", "single_cell_heldout", "bulk")
HEAD_SIDES = ("single_cell", "bulk")   # a head's scaler side (per-side scaling)


def candidate_ids() -> set:
    """Every valid candidate id: every encoder's search (CytoSet, Set Transformer, ScRAT, scVI), those
    outside it, and the reserved ids. Ids are unique within an encoder; the table's encoder column
    tells encoders apart."""
    grids = [*search_space.GROUPS_BY_ENCODER.values(), search_space.SCVI_GROUPS]
    ids = {c for grid in grids for group in grid.values() for c in group} | set(search_space.OUTSIDE_SEARCH)
    return ids | set(RESERVED_CANDIDATES)


def with_neutral_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill the optional key columns a table doesn't set with their neutral values."""
    frame = frame.copy()
    for column, (_, neutral, _) in KEYS.items():
        if column not in frame and neutral is not None:
            frame[column] = neutral
    return frame


def check(kind: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Assert that a table matches its kind; return it with the columns in schema order."""
    identity, other, _ = KINDS[kind]
    columns = [*KEYS, *identity, *other]
    missing = [c for c in columns if c not in frame]
    assert not missing, f"{kind}: missing columns {missing}"
    for column, (dtype, _, allowed) in KEYS.items():
        assert frame[column].notna().all(), f"{kind}: {column} has missing values"
        if dtype == "int":
            assert pd.api.types.is_integer_dtype(frame[column]), f"{kind}: {column} is not an integer column"
        else:
            assert frame[column].map(type).eq(str).all(), f"{kind}: {column} is not plain strings"
        if allowed is not None:
            bad = set(frame[column]) - set(allowed)
            assert not bad, f"{kind}: {column} has values outside the schema: {sorted(bad)[:5]}"
    across = frame["fold"] == -1   # across folds: a crossfit set's per-fold mean / pooled predictions, or a pooled row
    assert (frame.loc[across, "fold_set"].eq("crossfit") | frame.loc[across, "pair"].eq("pooled")).all() \
        and (frame["fold"] >= -1).all(), f"{kind}: fold -1 outside a crossfit fold set or a pooled row"
    pooled_rows = frame[frame["pair"] == "pooled"]
    assert (pooled_rows["fold_set"].eq("all") & pooled_rows["fold"].eq(-1)).all(), f"{kind}: pooled rows carry fold_set 'all', fold -1"
    configurations_ = frame[frame["row_kind"] == "configuration"]
    known = configurations_["key"].map(configurations.CONFIGURATIONS)
    assert known.notna().all(), f"{kind}: unknown configuration keys"
    assert (configurations_["model"] == known.map(lambda c: c["model"])).all() and \
        (configurations_["input"] == known.map(lambda c: c["input"])).all(), f"{kind}: model/input disagree with key"
    references = frame[frame["row_kind"] == "reference"]
    known = references["key"].map(configurations.REFERENCE_ROWS)
    assert known.notna().all(), f"{kind}: unknown reference keys"
    assert (references["model"] == known.map(lambda r: r["model"])).all() and \
        (references["input"] == known.map(lambda r: r["input"])).all(), f"{kind}: reference model/input disagree with key"
    candidates = frame[frame["row_kind"] == "candidate"]
    assert candidates["key"].isin(candidate_ids()).all(), f"{kind}: unknown candidate ids"
    embed_only = frame["model"].isin(EMBED_ONLY)
    assert frame.loc[embed_only, "pooling"].isin(["mean", "max"]).all(), f"{kind}: embed-only models need mean/max pooling"
    assert frame.loc[~embed_only, "pooling"].eq("native").all(), \
        f"{kind}: set encoders, pseudobulk and non-learned rows have native pooling"
    if "pool_set" in identity:
        pooled = frame["pair"] == "pooled"
        assert (pooled == (frame["pool_set"] != "per_pair")).all(), f"{kind}: pair 'pooled' goes with a pooled pool_set (and only then)"
    else:
        assert (frame["pair"] != "pooled").all(), f"{kind}: pair 'pooled' only in statistics"
    if "side" in identity:
        assert frame["side"].isin(HEAD_SIDES if kind == "heads" else SIDES).all(), f"{kind}: unknown side"
    assert not frame.duplicated([*KEYS, *identity]).any(), f"{kind}: two rows share their key + identity columns"
    return frame[columns + [c for c in frame.columns if c not in columns]]


def write(kind: str, frame: pd.DataFrame, path):
    """Check and write a table (Parquet, SCHEMA_VERSION in its metadata) to path (the caller passes
    manifest.temporary(...)). Label-bearing kinds must be written under the marked name."""
    marked = path.name.removeprefix(".tmp.").startswith(LABELLED + "_")
    assert marked == KINDS[kind][2], f"{kind}: label-bearing tables (only) are named {LABELLED}_*"
    table = pa.Table.from_pandas(check(kind, frame), preserve_index=False)
    meta = {**(table.schema.metadata or {}), b"sc2bulk_schema": f"{kind}:{SCHEMA_VERSION}".encode()}
    pq.write_table(table.replace_schema_metadata(meta), path)


def read(kind: str, path) -> pd.DataFrame:
    """Read and check a table, asserting its kind and schema version."""
    table = pq.read_table(path)
    assert table.schema.metadata[b"sc2bulk_schema"].decode() == f"{kind}:{SCHEMA_VERSION}", f"{path}: not a {kind} v{SCHEMA_VERSION} table"
    return check(kind, table.to_pandas())
