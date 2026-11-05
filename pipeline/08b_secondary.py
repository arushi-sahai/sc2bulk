"""Stage 08b: the secondary read-outs. Label-free: single-cell labels and bulk expression only; it
runs on real bulk only after the analysis-fixed tag, after stage 08, 08a and any stage 08 rerun.

Stage 08's functions are loaded from pipeline/08_readout.py (runpy, the 07d pattern; codehash.extra adds that
file to this stage's hash), and sc2bulk.readout is used unchanged, so stage 08's hash, its primary files and
08a's decision stay as they are. New numerics are in sc2bulk.readout_secondary; constants in config/secondary.py.
Stage 06's inputs are checked against stage 06's current code hash (as stage 08 does), stage 05's against stage
05's, stage 08's heads against stage 08's.

Common rules: the pair's primary fold set only (split67 takes no family); read-out seeds 0-9;
training rows = all of the unit's training samples or the family's subset; bulk rows = the
unit's scored bulk; CV folds = stage 03's read-out folds when the rows are the full training set (or an encoder
split's training side, context encoder_split_e), else dealt by readout.readout_folds with stage 03's seed.
Encoder seeds: families built on rows A, B and J read seeds 0-9 only (10-19 are attribution
only); D: j = 0..N_PERMUTED-1 (scVI: row J's seed j mod 10); E, F, H: 0-4; G, I: 0-9; MSE and the search: the
search seeds. Fixed representations carry encoder_seed -1, except where the family's subset or split is indexed
by e (rows E and H's within rows: encoder_seed e for every configuration).

Families, each file <outputs>/08b_secondary/<pair>/<unit>/<family>/<name>_{predictions,heads}.parquet
(heads only for ridge, LASSO and elastic net, and not for head application), one manifest record per file pair:
  permuted              row D: the permuted single-cell labels, folds dealt on them; ridge only
  learning_curve        row E (LEARNING_CURVE_PAIRS): the subset's samples; variant training_fraction_<f>
  bag_size              row F (Fig 4A): the set encoders and scVI at each n but 500 (the primary) and "all" (not
                        CytoSet's where its frozen configuration has the age-similarity loss);
                        the FMs' mean of n cells (shared bags, U7) and, at 500, their max pooling; pseudobulk-n
                        (log-CP10k per bag, then the mean over bags) at every n and "all"
  own_recipe            row G: candidate ids <encoder>_own_recipe
  one_type              row H (Fig 6C): transfer from the one-type store; <key>_within: Fig 2B's format (split e
                        within the store), or for a cross-fitting fold its held-out one-type samples scored by
                        this family's own heads; under smoke, a unit whose store falls short has no configuration's
                        one_type rows (one_type_absent)
  per_sample_route      row I, where the frozen configuration uses per_sample: transfer and <key>_within as above
  mse                   the MSE variant (aging pairs): candidate objective_mse, encoder_seed = search seed
  search_candidates     every stage 05 search candidate (and merged configuration) of the three set encoders
  thinned               H3: the primary heads applied to row A's and J's thinned-bulk rows (own bulk scaler)
  observed_library      scVI's observed-library-size bulk rows, by the primary heads
  outside_moments       the primary heads applied to bulk scaled with the other half's moments (halves per r)
  single_cell_moments   the primary heads applied to bulk scaled with stage 08's stored single-cell moments; variant "primary" (schemas.py, in stage 05's closure, takes no new label), told
                        apart by this directory
  alternative_readouts  elastic net, PLS, k-NN, MLP on the primary features (grids in config/secondary.py)
  numeric_age           aging pairs: ridge on the z-scored single-cell age (label_value)
  native_head           CytoSet's own head on bulk (row A's native_head_s<e>.csv); readout native_head
  feature_variants      pseudobulk TMM / VST (per side), panel_1000 / panel_2000 / all_shared, mapped
                        orthologs (mouse pairs), and the FMs on the full gene set
  reference_rows        Fig 2A's reference rows: pseudobulk_pca64, mean_log_cells, per_cell_classifier (ridge only), cytoset_untrained
                        (row C, seeds 0-9), deconvolution_clr / deconvolution_true (not mouse microglia) with their
                        <key>_within column (D6), known_signature (readout none, probability NaN), the sensitivity
                        variants "<key>/<variant>" (config/secondary.REFERENCE_VARIANTS); and two label-free tables,
                        deconvolution_flags.csv and known_signature_genes.csv

  python pipeline/08b_secondary.py --pair PAIR [--set UNIT] [--family FAMILY] [--name NAME]
"""
import argparse
import json
import runpy

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk import (bags, codehash, determinism, folds, manifest, orthologs, paths, preprocess, properties, readout, readout_secondary,
                     schemas, seeds, signatures, training_inputs)
from sc2bulk.config import configurations, representations, search_space, settings, sources
from sc2bulk.config import deconvolution as deconvolution_config
from sc2bulk.config import readout as R
from sc2bulk.config import secondary as C
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

STAGE = "08b_secondary"
R8 = runpy.run_path(str(settings.REPO / "pipeline" / "08_readout.py"), run_name="stage08")   # its functions, not its main
N_READOUT, N_ENCODER, TRAINED = R8["N_READOUT"], R8["N_ENCODER"], R8["TRAINED"]
KEYS = configurations.CONFIGURATION_ORDER
SET_ENCODERS = ("cytoset", "set_transformer", "scrat")
ROW_MODEL = {"cytoset_cells": "cytoset", "set_transformer_cells": "set_transformer", "scrat_cells": "scrat",
             "cytoset_pseudobulk": "cytoset_pseudobulk", "scvi_cells": "scvi"}   # stage 06's model directory per key (rows D-H)
FM_CELLS = ("geneformer_cells", "scgpt_cells")


# ---- inputs ----

def headline_seeds(pair, key) -> list:
    """Rows A, B and J's encoder seeds in 08b: 0-9 (the smoke count), never the attribution seeds."""
    return [e for e in R8["encoder_seeds"](pair, key) if e < N_ENCODER]


def row_file(pair, unit, row, model, point, e):
    return paths.output("06_train", pair, row, unit) / model / point / "embeddings" / f"{'search_s' if row == 'MSE' else 's'}{e}.h5ad"


def checked(path, stage):
    """Assert an upstream file was made by its stage's current code (a crashed or stale rerun can't mix in)."""
    pair = path.relative_to(settings.OUTPUTS).parts[1]
    made = manifest.read(path)["settings"]["code"]
    assert made == manifest.stage_code_hash(settings.REPO / "pipeline" / f"{stage}.py", pair, codehash.extra(stage, pair)), \
        f"{path}: stale {stage} output"
    return path


def embedding(path, stage="06_train"):
    data = anndata.read_h5ad(checked(path, stage))
    return data.obs.assign(row=np.arange(data.n_obs)), np.asarray(data.X, dtype=np.float64)


def rows_of(obs, X, sides, ids, variant="primary") -> np.ndarray:
    """X's rows of the given samples on the given sides (and bulk variant), in the given order."""
    part = obs[obs["side"].isin(sides)]
    if "variant" in part:
        part = part[part["variant"] == variant]
    part = part.set_index("sample_id")
    assert part.index.is_unique and set(ids) <= set(part.index), f"rows missing: {sorted(set(ids) - set(part.index))[:3]}"
    return X[part.loc[list(ids), "row"].to_numpy()]


TRAINING_SIDES = ("single_cell_train", "single_cell_validation")


def stage06_rows(pair, unit, row, model, point, e, train_ids, bulk_ids, variant="primary"):
    """A stage 06 row's embedding of seed e: the given training samples' rows and the bulk rows of a variant."""
    obs, X = embedding(row_file(pair, unit, row, model, point, e))
    return rows_of(obs, X, TRAINING_SIDES, train_ids), rows_of(obs, X, ("bulk",), bulk_ids, variant)


def primary_features(pair, unit, key, train_ids, bulk_ids) -> list:
    """[(e, training rows, bulk rows)] of a configuration's primary features, encoder seeds 0-9."""
    if key in TRAINED:
        return [(e, *R8["trained_features"](pair, unit, key, e, train_ids, bulk_ids)) for e in headline_seeds(pair, key)]
    return R8["features"](pair, unit, key, train_ids, bulk_ids)


def pseudobulk_bulk(pair, unit, bulk_ids) -> np.ndarray:
    """Pseudobulk's bulk rows: log-CP10k over the shared genes, on the unit's panel (stage 08's)."""
    return training_inputs.bulk_profiles(pair, unit, bulk_ids).X.toarray().astype(np.float64)


def fm_subset_file(pair, unit, model, what, cell_subset):
    """A stage 07 panel file of another cell subset (R8's fm_file reads "_all"), with stage 08's checks."""
    path = paths.output("07_embed", pair, f"{model}_panel_{what}_{cell_subset}.h5ad", unit)
    made = manifest.read(path)["settings"]
    assert all(made["inputs"][k] == v for k, v in manifest.input_hashes(folds.inputs(pair, one_type=True)).items()), \
        f"{path.name}: not made from the current single-cell store"
    assert tuple(made["unit"]) == tuple(unit) and made["panel"] == manifest.read(paths.output("04_panel", pair, "panel.csv", unit))["content_hash"], \
        f"{path.name}: not the unit's own panel"
    return anndata.read_h5ad(path, backed="r")


def fm_full_file(pair, model, what):
    """A stage 07 full-gene file (pair level), made from the current store."""
    path = paths.output("07_embed", pair, f"{model}_full_genes_{what}_all.h5ad")
    made = manifest.read(path)["settings"]
    assert all(made["inputs"][k] == v for k, v in manifest.input_hashes(folds.inputs(pair)).items()), \
        f"{path.name}: not made from the current single-cell store"
    return anndata.read_h5ad(path, backed="r")


def pooled_bags(X, ids, sample, sample_ids, n_cells, pair, unit, pool="mean") -> np.ndarray:
    """Each sample's mean over the shared test bags of n cells (bags.test_bags; 0: all its cells, once) of the
    bag's mean (or max) cell embedding."""
    row = pd.Series(np.arange(len(ids)), index=ids)
    test = bags.test_bags(ids, sample, list(sample_ids), n_cells, representations.TEST_TIME_BAGS, pair, unit)
    reduce = (lambda B: B.mean(axis=0, dtype=np.float64)) if pool == "mean" else (lambda B: B.max(axis=0).astype(np.float64))
    return np.vstack([np.mean([reduce(np.asarray(X[row[b].to_numpy()])) for b in test[s]], axis=0) for s in sample_ids])


def cells_of(cells_file, sample_ids):
    keep = cells_file.obs["sample_id"].astype(str).isin(set(sample_ids)).to_numpy()
    cells = cells_file[keep].to_memory()
    return np.asarray(cells.X), cells.obs_names.astype(str).to_numpy(), cells.obs["sample_id"].astype(str).to_numpy()


# ---- the unit's rows and folds ----

def context(pair, unit) -> dict:
    train = folds.training_samples(pair, *unit)
    bulk = R8["scored"](pair, unit)
    return {"train": train, "train_ids": list(train["sample_id"]), "y": train["group"].to_numpy(), "bulk": bulk,
            "bulk_ids": list(bulk["sample_id"]), "strata_train": R8["strata_of"](pair, train), "strata_bulk": R8["strata_of"](pair, bulk)}


def cv_folds(pair, unit, samples, full_ids, context_name="training") -> dict:
    """read-out seed r -> each sample's CV fold: stage 03's when the rows are its (the full training set, or
    split e's training side), else dealt as stage 03 deals them, with its seed."""
    ids = list(samples["sample_id"])
    if set(ids) == set(full_ids):
        return {r: R8["readout_folds"](pair, unit, ids, r, context_name) for r in range(N_READOUT)}
    return {r: readout.readout_folds(samples, seeds.derived("readout_folds", pair, r, *unit)).to_numpy() for r in range(N_READOUT)}


def key_columns(pair, unit, key, **change) -> dict:
    return {**R8["key_columns"](pair, unit, key), **change}


def candidate_columns(pair, unit, encoder, name, n_cells=T.FROZEN_NCELL) -> dict:
    return {"pair": pair, "fold_set": unit[0], "fold": unit[1], "row_kind": "candidate", "key": name, "model": encoder,
            "input": "cells", "n_cells": n_cells, "pooling": "native"}


def transfer(c, X_train, X_bulk, cv, row, dims, train=None, readouts=("ridge", "lasso"), X_scored=None, scored=None) -> tuple:
    """Ridge and LASSO at every read-out seed: bulk predictions (and, with X_scored, the scored single-cell rows'
    predictions, side single_cell_heldout), and the heads."""
    train = c["train"] if train is None else train
    strata_train = c["strata_train"] if train is c["train"] else R8["strata_of"](row["pair"], train)
    predictions, heads = [], []
    for name in readouts:
        for r in range(N_READOUT):
            out = readout.run(X_train, train["group"].to_numpy(), cv[r], X_bulk, readout=name, strata_train=strata_train,
                              strata_bulk=c["strata_bulk"], X_scored=X_scored,
                              strata_scored=None if scored is None else R8["strata_of"](row["pair"], scored))
            keys = {**row, "readout": name, "readout_seed": r}
            predictions.append(R8["prediction_rows"](keys, "bulk", c["bulk_ids"], c["bulk"]["scoring_unit_id"].to_numpy(), out["score"]))
            if X_scored is not None:
                predictions.append(R8["prediction_rows"](keys, "single_cell_heldout", list(scored["sample_id"]),
                                                         scored["scoring_unit_id"].to_numpy(), out["scored_score"]))
            heads += R8["head_rows"](keys, out, dims)
    return predictions, heads


def dims_of(pair, unit, key, X):
    return R8["feature_names"](pair, unit, key, X)


# ---- families ----

def permuted(pair, unit, key) -> dict:
    """Row D: the label-permuted control. y = training_inputs.permuted_groups(j); folds dealt on the permuted
    labels; trained configurations read row D's seed-j embedding (scVI: row J's seed j mod 10, not retrained);
    pseudobulk and the FMs their primary features. Ridge only."""
    c = context(pair, unit)
    fixed = None if key in TRAINED else primary_features(pair, unit, key, c["train_ids"], c["bulk_ids"])[0]
    predictions, heads = [], []
    for j in training_inputs.encoder_seeds(pair, "D"):
        samples = c["train"].assign(group=training_inputs.permuted_groups(pair, unit, j)["group"].loc[c["train_ids"]].to_numpy())
        if key == "scvi_cells":
            e = j % settings.N_ENCODER_SEEDS
            X_train, X_bulk = R8["trained_features"](pair, unit, key, e, c["train_ids"], c["bulk_ids"])
        elif key in TRAINED:
            e = j
            X_train, X_bulk = stage06_rows(pair, unit, "D", ROW_MODEL[key], "", j, c["train_ids"], c["bulk_ids"])
        else:
            e, X_train, X_bulk = fixed
        cv = {r: readout.readout_folds(samples, seeds.derived("readout_folds", pair, r, *unit)).to_numpy() for r in range(N_READOUT)}
        p, h = transfer(c, X_train, X_bulk, cv, key_columns(pair, unit, key, encoder_seed=e, permutation=j), dims_of(pair, unit, key, X_train),
                        train=samples, readouts=("ridge",))
        predictions += p
        heads += h
    return {"predictions": predictions, "heads": heads}


def learning_curve(pair, unit, key) -> dict:
    """Row E: split e's nested subset at each fraction (training_inputs.learning_curve_subset; both sides, R1);
    trained configurations read row E's embedding, the others their primary features on the subset's rows;
    encoder_seed e for every configuration (the subset is indexed by e)."""
    c = context(pair, unit)
    fixed = None if key in TRAINED else primary_features(pair, unit, key, c["train_ids"], c["bulk_ids"])[0]
    predictions, heads = [], []
    for f in T.LEARNING_CURVE_FRACTIONS:
        for e in training_inputs.encoder_seeds(pair, "E"):
            subset = training_inputs.learning_curve_subset(pair, unit, e, f)
            samples = c["train"][c["train"]["sample_id"].isin(set(subset))].reset_index(drop=True)
            ids = list(samples["sample_id"])
            if key in TRAINED:
                X_train, X_bulk = stage06_rows(pair, unit, "E", ROW_MODEL[key], f"f{round(100 * f)}", e, ids, c["bulk_ids"])
            else:
                X_train, X_bulk = fixed[1][c["train"]["sample_id"].isin(set(subset)).to_numpy()], fixed[2]
            p, h = transfer(c, X_train, X_bulk, cv_folds(pair, unit, samples, c["train_ids"]),
                            key_columns(pair, unit, key, encoder_seed=e, variant=f"training_fraction_{f}"), dims_of(pair, unit, key, X_train),
                            train=samples)
            predictions += p
            heads += h
    return {"predictions": predictions, "heads": heads}


def bag_points(pair) -> list:
    """Fig 4A's points besides the primary's 500: (stage 06's point name, n_cells key, cell_subset key)."""
    out = [(f"n{n}", n, "all") for n in representations.BAG_SIZES if n != T.FROZEN_NCELL]
    return out + ([("n_all", 0, "all_available")] if pair in T.FIG4A_ALL_PAIRS else [])


def age_loss_cytoset(pair, unit) -> bool:
    """Whether the unit's frozen CytoSet configuration has the age-similarity loss: stage 06 then trains no row F
    "all" for CytoSet (the whole batch would pass at once, above the 3,000-cell micro-batch cap)."""
    return bool(json.loads((paths.output("05_search", pair, "cytoset", unit) / "selection.json").read_text())["frozen"].get("age_similarity_loss"))


def trained_points(pair, unit, key) -> list:
    """A trained configuration's row F points: bag_points without "all" for an age-loss CytoSet."""
    skip = key == "cytoset_cells" and age_loss_cytoset(pair, unit)
    return [p for p in bag_points(pair) if not (skip and p[0] == "n_all")]


def bag_size(pair, unit, name) -> dict:
    """Row F, one line of Fig 4A: a trained model's row F embeddings (seeds 0-4), an FM's pooled cells (mean at
    every n; "<key>_max": max pooling at 500), or pseudobulk-n (every n incl. 500, and "all")."""
    c = context(pair, unit)
    predictions, heads = [], []
    key = name.removesuffix("_max")
    if key in TRAINED:
        for point, n, cell_subset in trained_points(pair, unit, key):
            for e in training_inputs.encoder_seeds(pair, "F"):
                X_train, X_bulk = stage06_rows(pair, unit, "F", ROW_MODEL[key], point, e, c["train_ids"], c["bulk_ids"])
                p, h = transfer(c, X_train, X_bulk, cv_folds(pair, unit, c["train"], c["train_ids"]),
                                key_columns(pair, unit, key, encoder_seed=e, n_cells=n, cell_subset=cell_subset), dims_of(pair, unit, key, X_train))
                predictions += p
                heads += h
        return {"predictions": predictions, "heads": heads}
    if key == "pseudobulk":
        lines = pseudobulk_n(pair, unit, c)
    else:
        model = configurations.CONFIGURATIONS[key]["model"]
        X, ids, sample = cells_of(R8["fm_file"](pair, unit, model, "cells"), c["train_ids"])
        if name.endswith("_max"):
            lines = [(T.FROZEN_NCELL, "all", "max", pooled_bags(X, ids, sample, c["train_ids"], T.FROZEN_NCELL, pair, unit, "max"))]
        else:
            lines = [(n, s, "mean", pooled_bags(X, ids, sample, c["train_ids"], n, pair, unit)) for _, n, s in bag_points(pair) if s == "all"]
            if pair in T.FIG4A_ALL_PAIRS:   # "all": stage 07's stored cells and 07d's extra cells, each sample as one bag
                extra = anndata.read_h5ad(checked(paths.output("07d_embed_all_available", pair, f"{model}_panel_cells_extra.h5ad", unit),
                                                  "07d_embed_all_available"))
                X_all = np.vstack([X, np.asarray(extra.X)])
                ids_all = np.concatenate([ids, extra.obs_names.astype(str).to_numpy()])
                sample_all = np.concatenate([sample, extra.obs["sample_id"].astype(str).to_numpy()])
                lines.append((0, "all_available", "mean", pooled_bags(X_all, ids_all, sample_all, c["train_ids"], 0, pair, unit)))
        X_bulk = np.asarray(R8["fm_file"](pair, unit, model, "bulk")[c["bulk_ids"]].to_memory().X, dtype=np.float64)
        lines = [(n, s, pool, X_train, X_bulk) for n, s, pool, X_train in lines]
    for n, cell_subset, pool, X_train, X_bulk in lines:
        p, h = transfer(c, X_train, X_bulk, cv_folds(pair, unit, c["train"], c["train_ids"]),
                        key_columns(pair, unit, key, n_cells=n, cell_subset=cell_subset, pooling=pool), dims_of(pair, unit, key, X_train))
        predictions += p
        heads += h
    return {"predictions": predictions, "heads": heads}


def pseudobulk_n(pair, unit, c) -> list:
    """[(n, cell_subset, pooling, training rows, bulk rows)] of the pseudobulk-n line: per sample, log-CP10k of
    each shared test bag's summed counts, then the mean over the bags; "all": the all-available
    store's pseudobulk. Bulk: the primary pseudobulk features."""
    genes, chosen = training_inputs.shared(pair), training_inputs.panel(pair, unit)
    columns = pd.Index(genes).get_indexer(chosen)
    cells = folds.training_cells(pair, *unit)
    counts = sp.csr_matrix(cells[:, genes].X)
    ids, sample = cells.obs_names.astype(str).to_numpy(), cells.obs["sample_id"].astype(str).to_numpy()
    row = pd.Series(np.arange(len(ids)), index=ids)
    X_bulk = pseudobulk_bulk(pair, unit, c["bulk_ids"])
    out = []
    for n in representations.BAG_SIZES:
        test = bags.test_bags(ids, sample, c["train_ids"], n, representations.TEST_TIME_BAGS, pair, unit)
        X = np.vstack([readout_secondary.bag_pseudobulk(counts, [row[b].to_numpy() for b in test[s]], columns) for s in c["train_ids"]])
        out.append((n, "all", "native", X, X_bulk))
    if pair in T.FIG4A_ALL_PAIRS:
        profiles = training_inputs.pseudobulk_log_all_available(pair, unit)
        out.append((0, "all_available", "native", np.vstack([profiles[s] for s in c["train_ids"]]), X_bulk))
    return out


def own_recipe(pair, unit, encoder) -> dict:
    """Row G: each set encoder's own recipe, seeds 0-9."""
    c = context(pair, unit)
    n_cells = T.OWN_RECIPES[encoder].get("ncell", search_space.base(encoder)["ncell"])
    predictions, heads = [], []
    for e in training_inputs.encoder_seeds(pair, "G"):
        X_train, X_bulk = stage06_rows(pair, unit, "G", encoder, "", e, c["train_ids"], c["bulk_ids"])
        p, h = transfer(c, X_train, X_bulk, cv_folds(pair, unit, c["train"], c["train_ids"]),
                        {**candidate_columns(pair, unit, encoder, f"{encoder}_own_recipe", n_cells), "encoder_seed": e},
                        list(np.arange(X_train.shape[1]).astype(str)))
        predictions += p
        heads += h
    return {"predictions": predictions, "heads": heads}


def one_type_features(pair, unit, key, ids, bulk_ids, e=None, heldout=False):
    """(single-cell rows, bulk rows) for row H: a trained model's row H embedding of seed e (its training or
    held-out one-type rows), else one-type features computed here: pseudobulk from the one-type profiles (the
    held-out donors' from their one-type cells), the FMs' one-type cells (bags of min(500, available)) or
    one-type profiles. Bulk: the full profile (row H's bulk rows; the primary bulk features for the others)."""
    if key in TRAINED:
        obs, X = embedding(row_file(pair, unit, "H", ROW_MODEL[key], "", e))
        return rows_of(obs, X, ("single_cell_heldout",) if heldout else TRAINING_SIDES, ids), rows_of(obs, X, ("bulk",), bulk_ids)
    if key == "pseudobulk":
        X_bulk = pseudobulk_bulk(pair, unit, bulk_ids)
        if not heldout:
            profiles = training_inputs.pseudobulk_log(pair, unit, cell_subset="one_type")
            return np.vstack([profiles[s] for s in ids]), X_bulk
        data = folds.scored_cells(pair, *unit, cell_subset="one_type")
        genes, chosen = training_inputs.shared(pair), training_inputs.panel(pair, unit)
        raw, sample = data[:, genes].X, data.obs["sample_id"].astype(str).to_numpy()
        columns = pd.Index(genes).get_indexer(chosen)
        return np.vstack([preprocess.log_normalize(preprocess.summed(raw[sample == s])[None, :])[:, columns].toarray().ravel()
                          for s in ids]), X_bulk
    model, input_ = configurations.CONFIGURATIONS[key]["model"], configurations.CONFIGURATIONS[key]["input"]
    X_bulk = np.asarray(R8["fm_file"](pair, unit, model, "bulk")[list(bulk_ids)].to_memory().X, dtype=np.float64)
    if input_ == "pseudobulk":
        return np.asarray(fm_subset_file(pair, unit, model, "pseudobulk", "one_type")[list(ids)].to_memory().X, dtype=np.float64), X_bulk
    X, cell_ids, sample = cells_of(fm_subset_file(pair, unit, model, "cells", "one_type"), ids)
    return pooled_bags(X, cell_ids, sample, ids, T.FROZEN_NCELL, pair, unit), X_bulk


def store_samples(train):
    return folds.in_store(train, "one_type").reset_index(drop=True)


def one_type(pair, unit, key) -> dict:
    """Row H transfer: the one-type store's training samples; trained e 0-4, fixed -1; cell_subset one_type."""
    c = context(pair, unit)
    samples = store_samples(c["train"])
    ids = list(samples["sample_id"])
    cv = cv_folds(pair, unit, samples, c["train_ids"])
    seeds_used = training_inputs.encoder_seeds(pair, "H") if key in TRAINED else [-1]
    predictions, heads = [], []
    for e in seeds_used:
        X_train, X_bulk = one_type_features(pair, unit, key, ids, c["bulk_ids"], e)
        p, h = transfer(c, X_train, X_bulk, cv, key_columns(pair, unit, key, encoder_seed=e, cell_subset="one_type"),
                        dims_of(pair, unit, key, X_train), train=samples)
        predictions += p
        heads += h
    return {"predictions": predictions, "heads": heads}


def within_split(pair, unit, key, samples, features_of, row_of, splits) -> dict:
    """Fig 2B's format: per split e, a fit on split e's training side (within the given samples), scoring its
    validation side; encoder_seed e for every configuration."""
    predictions, heads = [], []
    samples = samples.assign(scoring_unit_id=R8["scoring_units"](pair, samples))
    for e in splits:
        X = features_of(e)
        dims = dims_of(pair, unit, key, X)
        fit = (samples["sample_id"].map(training_inputs.encoder_sides(pair, unit, e)) == "train").to_numpy()
        fitted, validation = samples[fit], samples[~fit]
        full_side = [s for s, side in training_inputs.encoder_sides(pair, unit, e).items() if side == "train"]
        cv = cv_folds(pair, unit, fitted, full_side, f"encoder_split_{e}")
        for name in ("ridge", "lasso"):
            for r in range(N_READOUT):
                out = readout.run(X[fit], fitted["group"].to_numpy(), cv[r], None, readout=name, strata_train=R8["strata_of"](pair, fitted),
                                  X_scored=X[~fit], strata_scored=R8["strata_of"](pair, validation))
                keys = {**row_of(e), "readout": name, "readout_seed": r}
                predictions.append(R8["prediction_rows"](keys, "single_cell_validation", list(validation["sample_id"]),
                                                         validation["scoring_unit_id"].to_numpy(), out["scored_score"]))
                heads += R8["head_rows"](keys, out, dims)
    return {"predictions": predictions, "heads": heads}


def heldout_by_heads(pair, unit, heads_path, X_of, held, keys) -> dict:
    """A cross-fitting fold's held-out single-cell samples scored by stored heads (the family's own transfer
    heads, as stage 08 scores its held-out rows with the primary heads)."""
    stored = schemas.read("heads", checked(heads_path, STAGE))
    stored = stored[stored["side"] == "single_cell"]
    predictions = []
    for (e, name, r), head in stored.groupby(["encoder_seed", "readout", "readout_seed"], sort=False):
        first = head[head["stratum"] == head["stratum"].iloc[0]]
        score = readout_secondary.apply_head(X_of(int(e)), first["coef"].to_numpy(), first["intercept"].iloc[0],
                                             R8["strata_of"](pair, held), head_moments(head))
        row = {**keys, "encoder_seed": int(e), "readout": name, "readout_seed": int(r)}
        predictions.append(R8["prediction_rows"](row, "single_cell_heldout", list(held["sample_id"]), held["scoring_unit_id"].to_numpy(), score))
    return {"predictions": predictions}


def one_type_within(pair, unit, key) -> dict:
    """Row H's within rows: split e within the store (e 0-4), or a cross-fitting fold's held-out one-type samples
    scored by this family's transfer heads."""
    c = context(pair, unit)
    samples = store_samples(c["train"])
    keys = key_columns(pair, unit, key, cell_subset="one_type")
    if unit[0] == "crossfit":
        held = R8["heldout_samples"](pair, unit)
        held = held[held["sample_id"].isin(set(store_samples(folds.single_cell_samples(pair))["sample_id"]))].reset_index(drop=True)
        ids = list(held["sample_id"])
        return heldout_by_heads(pair, unit, paths.output(STAGE, pair, "one_type", unit) / f"{key}_heads.parquet",
                                lambda e: one_type_features(pair, unit, key, ids, [], e, heldout=True)[0], held, keys)
    ids = list(samples["sample_id"])
    fixed = None if key in TRAINED else one_type_features(pair, unit, key, ids, [])[0]
    return within_split(pair, unit, key, samples, lambda e: one_type_features(pair, unit, key, ids, [], e)[0] if key in TRAINED else fixed,
                        lambda e: {**keys, "encoder_seed": e}, training_inputs.encoder_seeds(pair, "H"))


def has_per_sample_route(pair, unit, encoder) -> bool:
    return "per_sample_route" in json.loads((paths.output("05_search", pair, encoder, unit) / "selection.json").read_text())


def per_sample_route(pair, unit, encoder) -> dict:
    """Row I transfer: stage 05's per_sample_route configuration, seeds 0-9."""
    c = context(pair, unit)
    key = f"{encoder}_cells"
    predictions, heads = [], []
    for e in training_inputs.encoder_seeds(pair, "I"):
        X_train, X_bulk = stage06_rows(pair, unit, "I", encoder, "", e, c["train_ids"], c["bulk_ids"])
        p, h = transfer(c, X_train, X_bulk, cv_folds(pair, unit, c["train"], c["train_ids"]),
                        key_columns(pair, unit, key, encoder_seed=e, variant="per_sample_route_sensitivity"), dims_of(pair, unit, key, X_train))
        predictions += p
        heads += h
    return {"predictions": predictions, "heads": heads}


def per_sample_route_within(pair, unit, encoder) -> dict:
    c = context(pair, unit)
    key = f"{encoder}_cells"
    keys = key_columns(pair, unit, key, variant="per_sample_route_sensitivity")
    if unit[0] == "crossfit":
        held = R8["heldout_samples"](pair, unit)
        ids = list(held["sample_id"])
        def X_of(e):
            obs, X = embedding(row_file(pair, unit, "I", encoder, "", e))
            return rows_of(obs, X, ("single_cell_heldout",), ids)
        return heldout_by_heads(pair, unit, paths.output(STAGE, pair, "per_sample_route", unit) / f"{encoder}_heads.parquet", X_of, held, keys)
    def X_of(e):
        obs, X = embedding(row_file(pair, unit, "I", encoder, "", e))
        return rows_of(obs, X, TRAINING_SIDES, c["train_ids"])
    return within_split(pair, unit, key, c["train"], X_of, lambda e: {**keys, "encoder_seed": e}, training_inputs.encoder_seeds(pair, "I"))


def mse(pair, unit, _name) -> dict:
    """CytoSet's MSE variant (aging pairs; Fig 4C): row MSE's embeddings at the search seeds."""
    c = context(pair, unit)
    predictions, heads = [], []
    for i in training_inputs.encoder_seeds(pair, "MSE"):
        X_train, X_bulk = stage06_rows(pair, unit, "MSE", "cytoset", "", i, c["train_ids"], c["bulk_ids"])
        p, h = transfer(c, X_train, X_bulk, cv_folds(pair, unit, c["train"], c["train_ids"]),
                        {**candidate_columns(pair, unit, "cytoset", "objective_mse"), "encoder_seed": i}, list(np.arange(X_train.shape[1]).astype(str)))
        predictions += p
        heads += h
    return {"predictions": predictions, "heads": heads}


def search_dir(pair, unit, encoder):
    return paths.output("05_search", pair, encoder, unit)


def candidate_names(pair, unit, encoder) -> list:
    """Every configuration stage 05 trained at every search seed (base, candidates, merged ones), from its
    candidates tables, each checked against stage 05's current code hash."""
    names = None
    for i in training_inputs.encoder_seeds(pair, "MSE"):
        tables = sorted(search_dir(pair, unit, encoder).glob(f"candidates*_s{i}.parquet"))
        seen = {k for path in tables for k in schemas.read("candidates", checked(path, "05_search"))["key"]}
        names = seen if names is None else names & seen
    return sorted(names)


def search_candidate(pair, unit, name) -> dict:
    """One stage 05 candidate (name "<encoder>/<candidate>") at every search seed i: its embedding, read only if its
    manifest carries stage 05's current code hash, the trial name and seed i (stage08 v2 note)."""
    encoder, candidate = name.split("/")
    c = context(pair, unit)
    predictions, heads = [], []
    for i in training_inputs.encoder_seeds(pair, "MSE"):
        path = search_dir(pair, unit, encoder) / "embeddings" / f"{candidate}_s{i}.h5ad"
        made = manifest.read(path)["settings"]
        assert made["trial"] == candidate and made["seed"] == i, f"{path}: not trial {candidate} at search seed {i}"
        obs, X = embedding(path, "05_search")
        X_train, X_bulk = rows_of(obs, X, TRAINING_SIDES, c["train_ids"]), rows_of(obs, X, ("bulk",), c["bulk_ids"])
        p, h = transfer(c, X_train, X_bulk, cv_folds(pair, unit, c["train"], c["train_ids"]),
                        {**candidate_columns(pair, unit, encoder, candidate), "encoder_seed": i}, list(np.arange(X_train.shape[1]).astype(str)))
        predictions += p
        heads += h
    return {"predictions": predictions, "heads": heads}


def primary_heads(pair, unit, key) -> pd.DataFrame:
    """Stage 08's stored primary heads (single-cell side), checked against stage 08's current code hash."""
    heads = schemas.read("heads", checked(paths.output("08_readout", pair, f"{key}_heads.parquet", unit), "08_readout"))
    return heads[(heads["side"] == "single_cell") & (heads["encoder_seed"] < N_ENCODER)]


def head_moments(head) -> dict:
    """{stratum: (mean, scale)} of one stored head's scaler side, as readout.scale takes them."""
    return {s: (h["scaler_mean"].to_numpy(), h["scaler_scale"].to_numpy()) for s, h in head.groupby("stratum", sort=False)}


def by_heads(pair, unit, key, X_bulk_of, variant, scale_of=None) -> dict:
    """Head application: per (encoder seed, read-out, read-out seed), the primary head's coefficients
    and intercept applied to bulk rows X_bulk_of(e) standardized with their own moments (per stratum), or with
    scale_of(X, r, head) (outside moments; the single-cell moments)."""
    c = context(pair, unit)
    predictions = []
    for (e, name, r), head in primary_heads(pair, unit, key).groupby(["encoder_seed", "readout", "readout_seed"], sort=True):
        first = head[head["stratum"] == head["stratum"].iloc[0]]
        X = X_bulk_of(int(e))
        assert len(first) == X.shape[1], f"{pair} {unit} {key} s{e}: the head's features differ from the bulk rows'"
        if scale_of is None:
            score = readout_secondary.apply_head(X, first["coef"].to_numpy(), first["intercept"].iloc[0], c["strata_bulk"])
        else:
            score = scale_of(X, int(r), head) @ first["coef"].to_numpy() + first["intercept"].iloc[0]
        row = {**key_columns(pair, unit, key, variant=variant), "encoder_seed": int(e), "readout": name, "readout_seed": int(r)}
        predictions.append(R8["prediction_rows"](row, "bulk", c["bulk_ids"], c["bulk"]["scoring_unit_id"].to_numpy(), score))
    return {"predictions": predictions}


def bulk_variant(pair, unit, key, variant):
    c = context(pair, unit)
    def X_of(e):
        obs, X = R8["trained_embedding"](pair, unit, key, e)   # checks stage 06's code hash
        return rows_of(obs, X, ("bulk",), c["bulk_ids"], variant)
    return by_heads(pair, unit, key, X_of, variant)


def thinned(pair, unit, key) -> dict:
    """H3: row A's and J's thinned-pseudocell bulk rows, by the primary heads."""
    return bulk_variant(pair, unit, key, "thinned_pseudocells")


def observed_library(pair, unit, key) -> dict:
    """scVI's observed-library-size bulk rows, by the primary heads."""
    return bulk_variant(pair, unit, key, "observed_library")


def bulk_halves(pair, unit, c, r) -> np.ndarray:
    """Each scored bulk sample's half for read-out seed r (scoring units halved within strata; label-free)."""
    strata = c["strata_bulk"] if c["strata_bulk"] is not None else np.full(len(c["bulk"]), "all")
    of = readout_secondary.halves(c["bulk"]["scoring_unit_id"].to_numpy(), strata, seeds.derived("subsample", pair, r, *unit))
    return c["bulk"]["scoring_unit_id"].map(of).to_numpy()


def outside(pair, unit, key) -> dict:
    """The outside-moments row: each bulk sample scaled with the other half's moments,
    halves drawn per read-out seed; the primary heads."""
    c = context(pair, unit)
    features = {e: X_bulk for e, _, X_bulk in primary_features(pair, unit, key, c["train_ids"], c["bulk_ids"])}
    half = {r: bulk_halves(pair, unit, c, r) for r in range(N_READOUT)}
    return by_heads(pair, unit, key, lambda e: features[e], "outside_moments_scaling",
                    lambda X, r, head: readout_secondary.outside_moments(X, half[r], c["strata_bulk"]))


def single_cell_moments(pair, unit, key) -> dict:
    """The single-cell-moments row: the primary heads applied to bulk standardized with the
    single-cell training rows' moments, stage 08's stored scaler (within organ for mouse organs, aging; per fold for
    human blood, aging). For AUROC this equals the single-cell-trained linear model on the unscaled bulk features."""
    c = context(pair, unit)
    features = {e: X_bulk for e, _, X_bulk in primary_features(pair, unit, key, c["train_ids"], c["bulk_ids"])}
    return by_heads(pair, unit, key, lambda e: features[e], "primary",
                    lambda X, r, head: readout.scale(X, head_moments(head), c["strata_bulk"]))


def alternative_readouts(pair, unit, name) -> dict:
    """One configuration under one S9 Fig read-out (name "<key>/<read-out>"), the primary features and folds."""
    key, which = name.split("/")
    c = context(pair, unit)
    cv = cv_folds(pair, unit, c["train"], c["train_ids"])
    predictions, heads = [], []
    for e, X_train, X_bulk in primary_features(pair, unit, key, c["train_ids"], c["bulk_ids"]):
        for r in range(N_READOUT):
            out = readout_secondary.alternative(X_train, c["y"], cv[r], X_bulk, which, c["strata_train"], c["strata_bulk"],
                                                random_state=seeds.derived("readout_init", pair, r, *unit))
            row = {**key_columns(pair, unit, key), "encoder_seed": e, "readout": which, "readout_seed": r}
            predictions.append(R8["prediction_rows"](row, "bulk", c["bulk_ids"], c["bulk"]["scoring_unit_id"].to_numpy(), out["score"]))
            if which == "elasticnet":
                heads += R8["head_rows"](row, out, dims_of(pair, unit, key, X_train))
    return {"predictions": predictions, "heads": heads} if which == "elasticnet" else {"predictions": predictions}


def numeric_age(pair, unit, key) -> dict:
    """The numeric-age read-out (aging pairs): ridge on the single-cell age (label_value) z-scored on
    the training rows, the primary features and folds."""
    c = context(pair, unit)
    z = training_inputs.training_ages(pair, unit, c["train_ids"])
    y = np.array([z[s] for s in c["train_ids"]])
    cv = cv_folds(pair, unit, c["train"], c["train_ids"])
    predictions, heads = [], []
    for e, X_train, X_bulk in primary_features(pair, unit, key, c["train_ids"], c["bulk_ids"]):
        for r in range(N_READOUT):
            out = readout_secondary.ridge_continuous(X_train, y, cv[r], X_bulk, c["strata_train"], c["strata_bulk"])
            row = {**key_columns(pair, unit, key, variant="numeric_age_readout"), "encoder_seed": e, "readout": "ridge", "readout_seed": r}
            predictions.append(R8["prediction_rows"](row, "bulk", c["bulk_ids"], c["bulk"]["scoring_unit_id"].to_numpy(), out["score"]))
            heads += R8["head_rows"](row, out, dims_of(pair, unit, key, X_train))
    return {"predictions": predictions, "heads": heads}


def native_head(pair, unit, _name) -> dict:
    """CytoSet's own head on bulk (S9 Fig): row A's native_head_s<e>.csv, seeds 0-9; score = its logit,
    probability = the logistic of the logit; readout native_head, one read-out seed (-1)."""
    c = context(pair, unit)
    predictions = []
    for e in headline_seeds(pair, "cytoset_cells"):
        path = checked(paths.output("06_train", pair, "A", unit) / "cytoset" / f"native_head_s{e}.csv", "06_train")
        table = pd.read_csv(path, dtype={"sample_id": str})
        logit = table[table["side"] == "bulk"].set_index("sample_id").loc[c["bulk_ids"], "logit"].to_numpy()
        row = {**key_columns(pair, unit, "cytoset_cells"), "encoder_seed": e, "readout": "native_head", "readout_seed": -1}
        predictions.append(pd.DataFrame({**row, "side": "bulk", "sample_id": c["bulk_ids"], "scoring_unit_id": c["bulk"]["scoring_unit_id"].to_numpy(),
                                         "score": logit, "probability": 1 / (1 + np.exp(-logit))}))
    return {"predictions": predictions}


def counts_per_side(pair, unit, c):
    """(training pseudobulk counts, scored bulk counts) on the shared genes, dense, and the gene names."""
    genes = training_inputs.shared(pair)
    single = folds.training_pseudobulk(pair, *unit)[c["train_ids"]][:, genes].X
    bulk = anndata.read_h5ad(settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad")[c["bulk_ids"]][:, genes].X
    dense = lambda X: X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return dense(single).astype(np.float64), dense(bulk).astype(np.float64), genes


def feature_variant(pair, unit, name) -> dict:
    """S11C/D, S12A: pseudobulk on other features, the same transform applied to each side on its own rows
    (label-free): tmm, vst (then the panel), panel_1000 / panel_2000 (the top k of stage 04's rank), all_shared
    (log-CP10k over every shared gene), mapped_orthologs (panel genes with a one-to-one ortholog, stage 04's
    orthologs.csv); or an FM on the full gene set (name "<key>/full_genes")."""
    c = context(pair, unit)
    cv = cv_folds(pair, unit, c["train"], c["train_ids"])
    if name.endswith("/full_genes"):
        key = name.split("/")[0]
        model, input_ = configurations.CONFIGURATIONS[key]["model"], configurations.CONFIGURATIONS[key]["input"]
        X_bulk = np.asarray(fm_full_file(pair, model, "bulk")[c["bulk_ids"]].to_memory().X, dtype=np.float64)
        if input_ == "pseudobulk":
            X_train = np.asarray(fm_full_file(pair, model, "pseudobulk")[c["train_ids"]].to_memory().X, dtype=np.float64)
        else:
            X, ids, sample = cells_of(fm_full_file(pair, model, "cells"), c["train_ids"])
            X_train = pooled_bags(X, ids, sample, c["train_ids"], T.FROZEN_NCELL, pair, unit)
        p, h = transfer(c, X_train, X_bulk, cv, key_columns(pair, unit, key, feature_set="full_genes"), list(np.arange(X_train.shape[1]).astype(str)))
        return {"predictions": p, "heads": h}
    single, bulk, genes = counts_per_side(pair, unit, c)
    panel = pd.read_csv(paths.output("04_panel", pair, "panel.csv", unit), keep_default_na=False)
    index = pd.Index(genes)
    log = lambda X: preprocess.log_normalize(X).toarray().astype(np.float64)
    if name in ("tmm", "vst"):
        transform = readout_secondary.tmm_log_cpm if name == "tmm" else readout_secondary.vst
        chosen, X_train, X_bulk, change = list(panel["gene"]), transform(single), transform(bulk), {"variant": name}
    else:
        if name == "all_shared":
            chosen = genes
        elif name == "mapped_orthologs":
            mapped = pd.read_csv(paths.output("04_panel", pair, "orthologs.csv", unit), keep_default_na=False)
            mapped = set(mapped.loc[mapped["status"] == "mapped", "gene"])
            chosen = [g for g in panel["gene"] if g in mapped]
        else:
            chosen = list(panel.sort_values("rank")["gene"].head(int(name.split("_")[1])))
        X_train, X_bulk, change = log(single), log(bulk), {"feature_set": name}
    columns = index.get_indexer(chosen)
    p, h = transfer(c, X_train[:, columns], X_bulk[:, columns], cv, key_columns(pair, unit, "pseudobulk", **change), chosen)
    return {"predictions": p, "heads": h}


# ---- Fig 2A's reference rows ----

def reference_columns(pair, unit, key, n_cells, **change) -> dict:
    """A reference row's key columns: model and input from configurations.REFERENCE_ROWS, pooling native."""
    row = configurations.REFERENCE_ROWS[key]
    return {"pair": pair, "fold_set": unit[0], "fold": unit[1], "row_kind": "reference", "key": key, "model": row["model"],
            "input": row["input"], "n_cells": n_cells, "pooling": "native", **change}


def pca64(pair, unit, _name) -> dict:
    """Pseudobulk, PCA-64: stage 08's pseudobulk features, each side standardized on its own rows
    (within organ), then properties.pca64 fit on the training rows; stage 08's ridge and LASSO on the scores."""
    c = context(pair, unit)
    (_, X_train, X_bulk), = R8["features"](pair, unit, "pseudobulk", c["train_ids"], c["bulk_ids"])
    Z_train = readout.scale(X_train, readout.scaler(X_train, c["strata_train"]), c["strata_train"])
    Z_bulk = readout.scale(X_bulk, readout.scaler(X_bulk, c["strata_bulk"]), c["strata_bulk"])
    P_train, P_bulk, k = properties.pca64(Z_train, Z_bulk, c["strata_train"], c["strata_bulk"])
    p, h = transfer(c, P_train, P_bulk, cv_folds(pair, unit, c["train"], c["train_ids"]),
                    reference_columns(pair, unit, "pseudobulk_pca64", 0, encoder_seed=-1), [f"pc{i}" for i in range(k)])
    return {"predictions": p, "heads": h}


def mean_log(pair, unit, _name) -> dict:
    """Mean of log cells: each training sample's mean over its stored cells' log-CP10k panel rows
    (properties.mean_log_cells); bulk: stage 08's pseudobulk bulk features; stage 08's ridge and LASSO."""
    c = context(pair, unit)
    cells = training_inputs.unit_cells(pair, unit)
    X_bulk = pseudobulk_bulk(pair, unit, c["bulk_ids"])
    X_train = properties.mean_log_cells(cells.X, cells.sample_id, c["train_ids"])
    p, h = transfer(c, X_train, X_bulk, cv_folds(pair, unit, c["train"], c["train_ids"]),
                    reference_columns(pair, unit, "mean_log_cells", 0, encoder_seed=-1), cells.genes)
    return {"predictions": p, "heads": h}


def per_cell(pair, unit, _name) -> dict:
    """The per-cell classifier: each training sample's cells are bag 0 of the shared
    n = PER_CELL_CAP test bags, each carrying its sample's group and read-out fold; cells standardized with the
    sample-level moments (the moments of the samples' mean capped cells, within organ), bulk with its own; ridge
    over PER_CELL_CAP x RIDGE_ALPHAS by readout_secondary.cell_ridge. Ridge only."""
    c = context(pair, unit)
    cells = training_inputs.unit_cells(pair, unit)
    test = bags.test_bags(cells.cell_id, cells.sample_id, c["train_ids"], C.PER_CELL_CAP, 1, pair, unit)
    if not settings.SMOKE:
        assert all(len(test[s][0]) == C.PER_CELL_CAP for s in c["train_ids"]), f"{pair} {unit}: a sample with fewer than {C.PER_CELL_CAP} cells"
    row = pd.Series(np.arange(len(cells.cell_id)), index=cells.cell_id)
    take = np.concatenate([row[test[s][0]].to_numpy() for s in c["train_ids"]])
    position = np.repeat(np.arange(len(c["train_ids"])), [len(test[s][0]) for s in c["train_ids"]])   # each cell's sample
    X_cells = cells.X[take].toarray().astype(np.float64)
    cell_strata = None if c["strata_train"] is None else c["strata_train"][position]
    moments = readout.scaler(properties.mean_log_cells(X_cells, position, range(len(c["train_ids"]))), c["strata_train"])
    Z_cells = readout.scale(X_cells, moments, cell_strata)
    X_bulk = pseudobulk_bulk(pair, unit, c["bulk_ids"])
    bulk_moments = readout.scaler(X_bulk, c["strata_bulk"])
    Z_bulk = readout.scale(X_bulk, bulk_moments, c["strata_bulk"])
    cv = cv_folds(pair, unit, c["train"], c["train_ids"])
    keys = reference_columns(pair, unit, "per_cell_classifier", C.PER_CELL_CAP, encoder_seed=-1, readout="ridge")
    predictions, heads = [], []
    for r in range(N_READOUT):
        out = readout_secondary.cell_ridge(Z_cells, c["y"][position], cv[r][position], Z_bulk, [C.PER_CELL_CAP * a for a in R.RIDGE_ALPHAS])
        out["scalers"] = {"single_cell": moments, "bulk": bulk_moments}
        row_keys = {**keys, "readout_seed": r}
        predictions.append(R8["prediction_rows"](row_keys, "bulk", c["bulk_ids"], c["bulk"]["scoring_unit_id"].to_numpy(), out["score"]))
        heads += R8["head_rows"](row_keys, out, cells.genes)
    return {"predictions": predictions, "heads": heads}


def untrained_selection(pair, unit):
    return paths.output("05_search", pair, "cytoset", unit) / "selection.json"


def cytoset_untrained(pair, unit, _name) -> dict:
    """CytoSet, untrained: stage 06 row C's embeddings at encoder seeds 0-9 (sides asserted to be
    split e), formed whatever the triage; stage 08's ridge and LASSO per seed."""
    c = context(pair, unit)
    assert json.loads(untrained_selection(pair, unit).read_text())["frozen"]["ncell"] == T.FROZEN_NCELL, "row C's bag size isn't FROZEN_NCELL"
    predictions, heads = [], []
    for e in training_inputs.encoder_seeds(pair, "C"):
        obs, X = embedding(row_file(pair, unit, "C", "cytoset", "", e))
        sides = training_inputs.encoder_sides(pair, unit, e)
        single = obs[obs["side"].isin(TRAINING_SIDES)].set_index("sample_id")
        assert all(single.loc[s, "side"] == f"single_cell_{'train' if sides[s] == 'train' else 'validation'}" for s in c["train_ids"]), \
            f"{pair} {unit} row C s{e}: sides differ from split {e}"
        X_train, X_bulk = rows_of(obs, X, TRAINING_SIDES, c["train_ids"]), rows_of(obs, X, ("bulk",), c["bulk_ids"])
        p, h = transfer(c, X_train, X_bulk, cv_folds(pair, unit, c["train"], c["train_ids"]),
                        reference_columns(pair, unit, "cytoset_untrained", T.FROZEN_NCELL, encoder_seed=e), list(np.arange(X_train.shape[1]).astype(str)))
        predictions += p
        heads += h
    return {"predictions": predictions, "heads": heads}


def deconvolution_file(pair, unit, name):
    return paths.output("04b_deconvolve", pair, name, unit)


def read_fractions(path) -> pd.DataFrame:
    """A stage 04b parquet table, checked against stage 04b's current code hash."""
    return pd.read_parquet(checked(path, "04b_deconvolve"))


def fractions(pair, unit, variant) -> pd.DataFrame:
    """Stage 04b's fractions of the unit (lupus no_ifn_markers: the run without Yao's genes among the markers), after
    the review gate: outside the smoke run, the type list's hash must equal the pin."""
    if variant == "no_ifn_markers":
        table = read_fractions(deconvolution_file(pair, unit, "fractions_no_ifn.parquet"))
        run, digest = "no_ifn", readout_secondary.type_list_sha256(table)
    else:
        table = read_fractions(deconvolution_file(pair, unit, "fractions.parquet"))
        run, digest = "main", manifest.content_hash(checked(deconvolution_file(pair, unit, "types.csv"), "04b_deconvolve"))
    if not settings.SMOKE:
        assert C.DECONVOLUTION_TYPES_SHA256.get((pair, folds.unit_name(*unit), run)) == digest, \
            f"{pair} {unit} {run}: the type list isn't the reviewed, pinned one (config/secondary.DECONVOLUTION_TYPES_SHA256)"
    return table


def deconvolution_row(pair, unit, name) -> dict:
    """Deconvolution: name "<key>[/<variant>]". Training rows: the training samples' estimated fractions
    (deconvolution_clr) or annotated fractions (deconvolution_true); bulk rows: the scored bulk's estimated fractions;
    CLR per stratum block (raw_fractions: the closed fractions); stage 08's ridge and LASSO."""
    key, _, variant = name.partition("/")
    c = context(pair, unit)
    table = fractions(pair, unit, variant)
    kind = "true" if key == "deconvolution_true" else "estimated"
    transform = "raw" if variant == "raw_fractions" else "clr"
    X_train, dims = readout_secondary.fraction_features(table[(table["side"] == "single_cell") & (table["kind"] == kind)], c["train_ids"], transform)
    X_bulk, bulk_dims = readout_secondary.fraction_features(table[(table["side"] == "bulk") & (table["kind"] == "estimated")], c["bulk_ids"], transform)
    assert dims == bulk_dims, f"{pair} {unit} {name}: the two sides' types differ"
    p, h = transfer(c, X_train, X_bulk, cv_folds(pair, unit, c["train"], c["train_ids"]),
                    reference_columns(pair, unit, key, 0, encoder_seed=-1, **({"variant": variant} if variant else {})), dims)
    return {"predictions": p, "heads": h}


def heldout_fractions(pair, unit, key) -> pd.DataFrame:
    """A cross-fitting fold's held-out single-cell samples' fractions, long (stage 04b's diagnostic files: estimated
    with the fold's reference, or annotated)."""
    which = "true" if key == "deconvolution_true" else "estimated"
    path = paths.output("04b_deconvolve", pair, f"diagnostic/{folds.unit_name(*unit)}_scored_{which}.parquet")
    wide = read_fractions(path).drop(columns=["residual"], errors="ignore")
    return wide.melt(id_vars="sample_id", var_name="type", value_name="fraction").assign(stratum="all")


def deconvolution_within(pair, unit, name) -> dict:
    """The within-single-cell column for the deconvolution rows. Per split e: a fit on its
    training side scoring its validation side, Fig 2B's format; estimated fractions from split e's own reference
    (fractions_splits.parquet: a training sample of a stratum without validation samples in split e has none, and
    the fit uses the training samples present), annotated fractions as in the transfer row. A cross-fitting fold: its
    held-out donors' samples scored by this family's transfer heads."""
    key = name.removesuffix("_within")
    c = context(pair, unit)
    keys = reference_columns(pair, unit, key, 0)
    if unit[0] == "crossfit":
        held = R8["heldout_samples"](pair, unit)
        ids = list(held["sample_id"])
        reference_types = fraction_types(fractions(pair, unit, ""), key)
        def X_of(_e):
            X, dims = readout_secondary.fraction_features(heldout_fractions(pair, unit, key), ids, "clr")
            assert dims == reference_types, f"{pair} {unit} {key}: the held-out fractions' types differ from the transfer row's"
            return X
        return heldout_by_heads(pair, unit, paths.output(STAGE, pair, "reference_rows", unit) / f"{key}_heads.parquet", X_of, held, keys)
    train = c["train"].assign(scoring_unit_id=R8["scoring_units"](pair, c["train"]))
    table = fractions(pair, unit, "")                    # the review gate, for both keys (deconvolution V1-V4 review)
    if key == "deconvolution_true":
        true = table[(table["side"] == "single_cell") & (table["kind"] == "true")]
    else:
        splits = read_fractions(deconvolution_file(pair, unit, "fractions_splits.parquet")).rename(columns={"estimated": "fraction"})
    predictions, heads = [], []
    for e in range(N_ENCODER):
        sides = training_inputs.encoder_sides(pair, unit, e)
        if key == "deconvolution_true":
            part = true
        else:
            part = splits[splits["split"] == e]
            assert all(sides[s] == side for s, side in zip(part["sample_id"], part["side"])), f"{pair} {unit}: split {e}'s sides differ from stage 03's"
        samples = train[train["sample_id"].isin(set(part["sample_id"]))]
        fit = (samples["sample_id"].map(sides) == "train").to_numpy()
        fitted, validation = samples[fit], samples[~fit]
        X, dims = readout_secondary.fraction_features(part, list(samples["sample_id"]), "clr")
        full_side = [s for s, side in sides.items() if side == "train"]
        cv = cv_folds(pair, unit, fitted, full_side, f"encoder_split_{e}")
        for readout_name in ("ridge", "lasso"):
            for r in range(N_READOUT):
                out = readout.run(X[fit], fitted["group"].to_numpy(), cv[r], None, readout=readout_name, strata_train=R8["strata_of"](pair, fitted),
                                  X_scored=X[~fit], strata_scored=R8["strata_of"](pair, validation))
                row = {**keys, "encoder_seed": e, "readout": readout_name, "readout_seed": r}
                predictions.append(R8["prediction_rows"](row, "single_cell_validation", list(validation["sample_id"]),
                                                         validation["scoring_unit_id"].to_numpy(), out["scored_score"]))
                heads += R8["head_rows"](row, out, dims)
    return {"predictions": predictions, "heads": heads}


def fraction_types(table, key) -> list:
    """The transfer row's feature names ("<stratum>:<type>") of a fractions table."""
    kind = "true" if key == "deconvolution_true" else "estimated"
    part = table[(table["side"] == "single_cell") & (table["kind"] == kind)]
    return readout_secondary.fraction_features(part, sorted(set(part["sample_id"])), "raw")[1]


def deconvolution_flags(pair, unit, _name) -> dict:
    """deconvolution_flags.csv, label-free, per stratum: V1's median and minimum r and gate,
    V4's reference_mismatch (against the out-of-sample single-cell baseline), its ratio and the in-sample ratio, and
    the neutrophil-leak Spearman; cross-fitting folds also V3's median r of this fold and the flag (the mean of the
    folds' medians below V3_FLAG_R). An undefined V3 r (a constant estimate) counts as
    r = 0 (deconvolution V1-V4 review)."""
    v1 = pd.read_csv(checked(deconvolution_file(pair, unit, "check_v1.csv"), "04b_deconvolve"), keep_default_na=False, na_values=[""])
    v4 = pd.read_csv(checked(deconvolution_file(pair, unit, "check_v4.csv"), "04b_deconvolve"), keep_default_na=False, na_values=[""])
    per_stratum = v1.groupby("stratum")["r"].agg(v1_median_r="median", v1_min_r="min").join(v1.groupby("stratum")["gate_passed"].first())
    table = per_stratum.reset_index().merge(v4[["stratum", "reference_mismatch"]].assign(
        v4_ratio=v4["bulk_median_residual"] / v4["single_cell_median_residual"],
        v4_in_sample_ratio=v4["bulk_median_residual"] / v4["single_cell_in_sample_median_residual"],
        neutrophil_leak_spearman=v4.get("neutrophil_leak_spearman", np.nan)), on="stratum", validate="one_to_one")
    table = table.assign(pair=pair, fold_set=unit[0], fold=unit[1])
    if unit[0] == "crossfit":
        median = {u: pd.read_csv(checked(v3_file(pair, u), "04b_deconvolve"))["r"].fillna(0).median() for u in folds.units(pair) if u[0] == "crossfit"}
        mean = float(np.mean(list(median.values())))
        table = table.assign(v3_median_r=median[unit], v3_mean_fold_median_r=mean, v3_flag=mean < deconvolution_config.V3_FLAG_R)
    return {"csv": [table]}


def v3_file(pair, unit):
    return paths.output("04b_deconvolve", pair, f"diagnostic/{folds.unit_name(*unit)}_v3.csv")


def signature_matches(pair, unit, var) -> pd.DataFrame:
    """The signature's bulk genes: human pairs by symbol (aliases applied by
    signatures.load); mouse microglia by uppercased symbol, a list gene matching two bulk genes failing; mouse organs
    through the one-to-one human ortholog of each bulk gene (orthologs.map_genes on the full bulk axis)."""
    listed = signatures.load(pair)
    if pair == "hammond":
        matched = readout_secondary.match_signature(listed, pd.Series(var.index.str.upper(), index=var.index))
        assert matched["symbol"].is_unique, f"{pair}: two bulk genes differ only in case and match one listed gene"
        return matched
    if pair == "tms":
        mapped = orthologs.map_genes(pd.DataFrame({"gene": var.index, "ensembl": var["ensembl"].to_numpy()}),
                                     orthologs.one2one(settings.RAW), orthologs.human_names(settings.RAW))
        mapped = mapped[mapped["status"] == "mapped"]
        return readout_secondary.match_signature(listed, pd.Series(mapped["human_symbol"].to_numpy(), index=mapped["gene"].to_numpy()))
    return readout_secondary.match_signature(listed, pd.Series(var.index, index=var.index))


def signature_inputs(pair, unit):
    """(the scored bulk's CPM of the matched genes over the full bulk axis, the matches, the used-gene mask)."""
    c = context(pair, unit)
    bulk = anndata.read_h5ad(settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad")[c["bulk_ids"]]
    matched = signature_matches(pair, unit, bulk.var)
    counts = np.asarray(bulk.X, dtype=np.float64)
    cpm = counts[:, bulk.var_names.get_indexer(matched["gene"])] / counts.sum(axis=1, keepdims=True) * 1e6
    return c, cpm, matched, readout_secondary.signature_used(cpm)


def known_signature(pair, unit, name) -> dict:
    """The known-signature score: no training; readout none, read-out seed -1, encoder
    seed -1, probability NaN (not a probability scale). Variants: pooled_z (z over all scored bulk, not within
    organ), median_centered (Yao's median form)."""
    _, _, variant = name.partition("/")
    c, cpm, matched, used = signature_inputs(pair, unit)
    up = (matched["direction"] == "up").to_numpy()[used]
    if variant == "median_centered":
        score = readout_secondary.median_centered(cpm[:, used])
    else:
        score = readout_secondary.signature_score(cpm[:, used], up, None if variant == "pooled_z" else c["strata_bulk"])
    keys = reference_columns(pair, unit, "known_signature", 0, encoder_seed=-1, readout="none", readout_seed=-1,
                             **({"variant": variant} if variant else {}))
    table = R8["prediction_rows"](keys, "bulk", c["bulk_ids"], c["bulk"]["scoring_unit_id"].to_numpy(), score)
    return {"predictions": [table.assign(probability=np.nan)]}


def known_signature_genes(pair, unit, _name) -> dict:
    """known_signature_genes.csv, label-free: per direction, the genes listed, matched to the
    bulk axis, passing the detection filter, and used (also not constant)."""
    _, cpm, matched, used = signature_inputs(pair, unit)
    listed = signatures.load(pair)
    detected = readout_secondary.signature_detected(cpm)
    rows = [{"pair": pair, "fold_set": unit[0], "fold": unit[1], "direction": d, "listed": int((listed["direction"] == d).sum()),
             "matched": int((matched["direction"] == d).sum()), "detected": int(detected[(matched["direction"] == d).to_numpy()].sum()),
             "used": int(used[(matched["direction"] == d).to_numpy()].sum())} for d in sorted(set(listed["direction"]))]
    return {"csv": [pd.DataFrame(rows)]}


REFERENCE_TABLES = ("deconvolution_flags", "known_signature_genes")   # label-free CSVs, not schema kinds


def reference_jobs(pair, unit) -> list:
    """[(name, compute, extra inputs)] of the reference_rows family: every reference row and its variants (config/secondary.REFERENCE_VARIANTS), the deconvolution rows' within column (D6), and the two label-free tables.
    Deconvolution only where stage 04b ran (not mouse microglia: one cell type)."""
    signature_extra = [settings.RAW / "ensembl" / n for n in (*sources.ORTHOLOG_FILES.values(), sources.HUMAN_GTF)] if pair == "tms" else []
    out = [("pseudobulk_pca64", pca64, []), ("mean_log_cells", mean_log, []), ("per_cell_classifier", per_cell, []),
           ("cytoset_untrained", cytoset_untrained, [row_file(pair, unit, "C", "cytoset", "", e) for e in training_inputs.encoder_seeds(pair, "C")]
            + [untrained_selection(pair, unit)]),
           ("known_signature", known_signature, signature_extra), ("known_signature_genes", known_signature_genes, signature_extra)]
    deconvolved = deconvolution_config.LEVELS[pair] is not None
    main = [deconvolution_file(pair, unit, n) for n in ("fractions.parquet", "types.csv")]
    for key in ("deconvolution_clr", "deconvolution_true") if deconvolved else ():
        out.append((key, deconvolution_row, main))
        heads = [paths.output(STAGE, pair, "reference_rows", unit) / f"{key}_heads.parquet"]
        held = [paths.output("04b_deconvolve", pair, f"diagnostic/{folds.unit_name(*unit)}_scored_{'true' if key == 'deconvolution_true' else 'estimated'}.parquet")]
        out.append((f"{key}_within", deconvolution_within, main + (heads + held if unit[0] == "crossfit" else [deconvolution_file(pair, unit, "fractions_splits.parquet")])))
    for (key, variant), pairs in C.REFERENCE_VARIANTS.items():
        if (pairs is None or pair in pairs) and (deconvolved or key == "known_signature"):
            extra = (signature_extra if key == "known_signature" else
                     [deconvolution_file(pair, unit, "fractions_no_ifn.parquet")] if variant == "no_ifn_markers" else main)
            out.append((f"{key}/{variant}", known_signature if key == "known_signature" else deconvolution_row, extra))
    if deconvolved:
        v3 = [v3_file(pair, u) for u in folds.units(pair) if u[0] == "crossfit"] if unit[0] == "crossfit" else []
        out.append(("deconvolution_flags", deconvolution_flags, [deconvolution_file(pair, unit, n) for n in ("check_v1.csv", "check_v4.csv")] + v3))
    return out


# ---- jobs ----

def base_inputs(pair, unit) -> list:
    return [*folds.inputs(pair, one_type="one_type" in PAIRS[pair]), settings.OUTPUTS / "02_preprocess" / pair / "bulk_samples.csv",
            settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad", settings.OUTPUTS / "03_splits" / pair / "donor_folds.csv",
            settings.OUTPUTS / "03_splits" / pair / "readout_folds.csv", settings.OUTPUTS / "03_splits" / pair / "splits.csv",
            paths.output("04_panel", pair, "panel.csv", unit), paths.output("04_panel", pair, "shared_genes.csv")]


def raw_inputs(files) -> dict:
    """Raw downloads among a job's inputs, by their pinned SHA-256 (as stages 04 and 04d record them): they have no
    manifests, and a smoke run keeps them outside settings.OUTPUTS."""
    return {f"{p.parent.name}/{p.name}": sources.DOWNLOADS[(p.parent.name, p.name)][1] for p in files}


def primary_inputs(pair, unit, key) -> list:
    """The upstream files of a configuration's primary features (stage 06 seeds 0-9, or stage 07's files)."""
    if key in TRAINED:
        return [R8["trained_file"](pair, unit, key, e) for e in headline_seeds(pair, key)]
    if key == "pseudobulk":
        return []
    model, input_ = configurations.CONFIGURATIONS[key]["model"], configurations.CONFIGURATIONS[key]["input"]
    return [paths.output("07_embed", pair, f"{model}_panel_{w}_all.h5ad", unit) for w in (input_, "bulk")]


def stage06_inputs(pair, unit, row, model, point, seeds_) -> list:
    return [row_file(pair, unit, row, model, point, e) for e in seeds_]


def row_absent(pair, unit, row) -> bool:
    """Smoke only: stage 06 skipped the unit's row (06_train.smoke_marker). Row H: no key's one_type rows are read
    out (one_type_absent); row E: no key's learning_curve rows are (the subset itself falls short). The real run has
    rows E and H at every seed."""
    absent = (paths.output("06_train", pair, row, unit) / "smoke_skipped.json").exists()
    assert settings.SMOKE or not absent, f"{pair} {unit}: stage 06's smoke row {row} marker in the real run"
    return absent


def fit_shortfall(pair, fitted, strata_scored, n_scored) -> list:
    """What readout.readout_folds and readout.run would assert on a fit, counted without calling them: fewer
    than READOUT_FOLDS holdout groups (folds.deal fills every fold from that many on), a label missing, a stratum of
    the scored single-cell rows (a within fit's validation side; readout.scale) without training rows, or no scored
    rows. Bulk is scaled on its own moments, so the transfer fit passes no strata."""
    out = [f"{n} holdout groups" for n in [fitted["holdout_group"].nunique()] if n < settings.READOUT_FOLDS]
    out += [f"no training samples of label {label}" for label in (0, 1) if not (fitted["group"] == label).any()]
    strata_fitted = R8["strata_of"](pair, fitted)
    if strata_fitted is not None:
        out += [f"no training samples of stratum {s}" for s in sorted(set(strata_scored) - set(strata_fitted))]
    return out + (["no scored samples"] if n_scored == 0 else [])


def one_type_shortfall(pair, unit) -> dict:
    """Smoke only: {fit: shortfalls} of the one-type store's fits in 08b: the transfer fit (the store's training
    samples, scoring the bulk) and, outside cross-fitting, each row H split e's within fit (its training side,
    scoring its validation side). A cross-fitting fold's held-out one-type samples aren't counted: with none,
    one_type_features would fail (np.vstack of nothing), which can't happen in the smoke run (human blood, aging:
    4 held-out donors per fold, unstratified)."""
    c = context(pair, unit)
    samples = store_samples(c["train"])
    out = {"transfer": fit_shortfall(pair, samples, [], len(c["bulk"]))}
    if unit[0] != "crossfit":
        for e in training_inputs.encoder_seeds(pair, "H"):
            fit = (samples["sample_id"].map(training_inputs.encoder_sides(pair, unit, e)) == "train").to_numpy()
            strata = R8["strata_of"](pair, samples[~fit])
            out[f"split {e}"] = fit_shortfall(pair, samples[fit], [] if strata is None else strata, int((~fit).sum()))
    return out


def one_type_marker(pair, unit):
    """Smoke only: written when the unit's one_type family is skipped (11b's one_type reads it)."""
    return paths.output(STAGE, pair, "one_type", unit) / "smoke_skipped.json"


def one_type_absent(pair, unit) -> bool:
    """Smoke only: whether every configuration's one_type rows (transfer and within) are skipped in the unit: stage
    06 skipped its row H, or a one-type fit falls short (one_type_shortfall). The reasons go to one_type_marker. The
    real run lists every one_type job and asserts there is no marker."""
    marker = one_type_marker(pair, unit)
    if not settings.SMOKE:
        assert not marker.exists() and not row_absent(pair, unit, "H"), f"{pair} {unit}: a smoke one_type marker in the real run"
        return False
    short = {"row H": ["stage 06 skipped it"]} if row_absent(pair, unit, "H") else {k: v for k, v in one_type_shortfall(pair, unit).items() if v}
    marker.unlink(missing_ok=True)
    if short:
        print(f"smoke: {pair} {folds.unit_name(*unit)} one_type skipped: {short}", flush=True)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(short))
    return bool(short)


def jobs(pair, unit) -> list:
    """Every (family, name, compute, extra inputs) of a primary unit (units)."""
    out = []
    def add(family, name, compute, extra):
        out.append((family, name, compute, extra))
    one_type_kept = "one_type" in PAIRS[pair] and not one_type_absent(pair, unit)
    for key in KEYS:
        D_seeds = training_inputs.encoder_seeds(pair, "D")
        extra = (stage06_inputs(pair, unit, "D", ROW_MODEL[key], "", D_seeds) if key in TRAINED and key != "scvi_cells"
                 else [R8["trained_file"](pair, unit, key, e) for e in sorted({j % settings.N_ENCODER_SEEDS for j in D_seeds})]
                 if key == "scvi_cells" else primary_inputs(pair, unit, key))
        add("permuted", key, lambda key=key: permuted(pair, unit, key), extra)
        if pair in T.LEARNING_CURVE_PAIRS and not row_absent(pair, unit, "E"):
            extra = ([row_file(pair, unit, "E", ROW_MODEL[key], f"f{round(100 * f)}", e) for f in T.LEARNING_CURVE_FRACTIONS
                      for e in training_inputs.encoder_seeds(pair, "E")] if key in TRAINED else primary_inputs(pair, unit, key))
            add("learning_curve", key, lambda key=key: learning_curve(pair, unit, key), extra)
        add("outside_moments", key, lambda key=key: outside(pair, unit, key),
            primary_inputs(pair, unit, key) + [paths.output("08_readout", pair, f"{key}_heads.parquet", unit)])
        add("single_cell_moments", key, lambda key=key: single_cell_moments(pair, unit, key),
            primary_inputs(pair, unit, key) + [paths.output("08_readout", pair, f"{key}_heads.parquet", unit)])
        for which in C.ALTERNATIVE_READOUTS:
            add("alternative_readouts", f"{key}/{which}", lambda name=f"{key}/{which}": alternative_readouts(pair, unit, name),
                primary_inputs(pair, unit, key))
        if pair in T.AGE_LOSS_PAIRS:
            add("numeric_age", key, lambda key=key: numeric_age(pair, unit, key), primary_inputs(pair, unit, key))
        if one_type_kept:
            extra = (stage06_inputs(pair, unit, "H", ROW_MODEL[key], "", training_inputs.encoder_seeds(pair, "H")) if key in TRAINED
                     else primary_inputs(pair, unit, key) + ([paths.output("07_embed", pair, f"{configurations.CONFIGURATIONS[key]['model']}_panel_"
                                                                          f"{configurations.CONFIGURATIONS[key]['input']}_one_type.h5ad", unit)]
                                                              if key != "pseudobulk" else []))
            add("one_type", key, lambda key=key: one_type(pair, unit, key), extra)
            add("one_type", f"{key}_within", lambda key=key: one_type_within(pair, unit, key),
                extra + ([paths.output(STAGE, pair, "one_type", unit) / f"{key}_heads.parquet"] if unit[0] == "crossfit" else []))
    for key in (*TRAINED, *FM_CELLS, "pseudobulk"):
        if key == "cytoset_pseudobulk":
            continue
        extra = ([row_file(pair, unit, "F", ROW_MODEL[key], point, e) for point, _, _ in trained_points(pair, unit, key)
                  for e in training_inputs.encoder_seeds(pair, "F")] if key in TRAINED else primary_inputs(pair, unit, key))
        if key in FM_CELLS and pair in T.FIG4A_ALL_PAIRS:
            extra.append(paths.output("07d_embed_all_available", pair, f"{configurations.CONFIGURATIONS[key]['model']}_panel_cells_extra.h5ad", unit))
        if key == "pseudobulk" and pair in T.FIG4A_ALL_PAIRS:
            extra.append(paths.output("02b_all_available", pair, "index.csv"))
        add("bag_size", key, lambda key=key: bag_size(pair, unit, key), extra)
        if key in FM_CELLS:
            add("bag_size", f"{key}_max", lambda key=key: bag_size(pair, unit, f"{key}_max"), primary_inputs(pair, unit, key))
    for encoder in SET_ENCODERS:
        add("own_recipe", encoder, lambda encoder=encoder: own_recipe(pair, unit, encoder),
            stage06_inputs(pair, unit, "G", encoder, "", training_inputs.encoder_seeds(pair, "G")))
        if has_per_sample_route(pair, unit, encoder):
            extra = stage06_inputs(pair, unit, "I", encoder, "", training_inputs.encoder_seeds(pair, "I"))
            add("per_sample_route", encoder, lambda encoder=encoder: per_sample_route(pair, unit, encoder), extra)
            add("per_sample_route", f"{encoder}_within", lambda encoder=encoder: per_sample_route_within(pair, unit, encoder),
                extra + ([paths.output(STAGE, pair, "per_sample_route", unit) / f"{encoder}_heads.parquet"] if unit[0] == "crossfit" else []))
        for candidate in candidate_names(pair, unit, encoder):
            K = training_inputs.encoder_seeds(pair, "MSE")
            add("search_candidates", f"{encoder}/{candidate}", lambda name=f"{encoder}/{candidate}": search_candidate(pair, unit, name),
                [search_dir(pair, unit, encoder) / "embeddings" / f"{candidate}_s{i}.h5ad" for i in K])
    if pair in T.AGE_LOSS_PAIRS:
        add("mse", "cytoset", lambda: mse(pair, unit, "cytoset"), stage06_inputs(pair, unit, "MSE", "cytoset", "", training_inputs.encoder_seeds(pair, "MSE")))
    for key in [k for k in TRAINED if k != "cytoset_pseudobulk"]:
        heads = [paths.output("08_readout", pair, f"{key}_heads.parquet", unit)]
        add("thinned", key, lambda key=key: thinned(pair, unit, key), primary_inputs(pair, unit, key) + heads)
    add("observed_library", "scvi_cells", lambda: observed_library(pair, unit, "scvi_cells"),
        primary_inputs(pair, unit, "scvi_cells") + [paths.output("08_readout", pair, "scvi_cells_heads.parquet", unit)])
    add("native_head", "cytoset_cells", lambda: native_head(pair, unit, "cytoset_cells"),
        [paths.output("06_train", pair, "A", unit) / "cytoset" / f"native_head_s{e}.csv" for e in headline_seeds(pair, "cytoset_cells")])
    variants = ["tmm", "vst", *(f"panel_{n}" for n in settings.PANEL_SIZE_SENSITIVITY), "all_shared"]
    variants += ["mapped_orthologs"] if PAIRS[pair]["species"] == "mouse" else []
    for name in variants:
        extra = [paths.output("04_panel", pair, "orthologs.csv", unit)] if name == "mapped_orthologs" else []
        add("feature_variants", name, lambda name=name: feature_variant(pair, unit, name), extra)
    for key in KEYS:
        if configurations.CONFIGURATIONS[key]["model"] in ("geneformer", "scgpt"):
            model, input_ = configurations.CONFIGURATIONS[key]["model"], configurations.CONFIGURATIONS[key]["input"]
            add("feature_variants", f"{key}/full_genes", lambda name=f"{key}/full_genes": feature_variant(pair, unit, name),
                [paths.output("07_embed", pair, f"{model}_full_genes_{w}_all.h5ad") for w in (input_, "bulk")])
    for name, compute, extra in reference_jobs(pair, unit):
        add("reference_rows", name, lambda name=name, compute=compute: compute(pair, unit, name), extra)
    return out


def kinds(unit, family, name) -> tuple:
    """A job's file kinds: no heads for head application, the native head, PLS / k-NN / MLP, held-out rows scored by
    stored heads, or the known-signature score; one label-free CSV ("csv") for the reference rows' two tables."""
    if family == "reference_rows" and name in REFERENCE_TABLES:
        return ("csv",)
    no_heads = family in ("thinned", "observed_library", "outside_moments", "single_cell_moments", "native_head") or \
        (name.endswith("_within") and unit[0] == "crossfit") or (family == "alternative_readouts" and not name.endswith("/elasticnet")) or \
        (family == "reference_rows" and name.startswith("known_signature"))
    return ("predictions",) if no_heads else ("predictions", "heads")


def outputs(pair, unit, family, name) -> list:
    directory = paths.output(STAGE, pair, family, unit)
    return [directory / (f"{name}.csv" if kind == "csv" else f"{name.replace('/', '__')}_{kind}.parquet") for kind in kinds(unit, family, name)]


def run(pair, unit, family, name, compute, record, cost, inputs):
    tables = compute()
    assert set(tables) == set(kinds(unit, family, name)), f"{family} {name}: tables {sorted(tables)}"
    for kind, path in zip(kinds(unit, family, name), outputs(pair, unit, family, name)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "csv":
            pd.concat(tables[kind], ignore_index=True).to_csv(manifest.temporary(path), index=False)
        else:
            schemas.write(kind, schemas.with_neutral_keys(pd.concat(tables[kind], ignore_index=True)), manifest.temporary(path))
        manifest.finish(path, record, cost, inputs=inputs)


def record_for(pair, unit, family, name, extra, code, c) -> tuple:
    """A job's run record and its input files: upstream outputs by their manifests' content hashes, raw downloads (the
    tms ortholog tables) by their pinned SHA-256."""
    inputs = base_inputs(pair, unit) + extra
    raw = [p for p in extra if p.is_relative_to(settings.RAW)]
    record = {"stage": STAGE, "pair": pair, "unit": unit, "family": family, "name": name, "smoke": settings.SMOKE, "code": code,
              "rows": manifest.settings_hash({"training": c["train_ids"], "bulk": c["bulk_ids"]}),
              "inputs": manifest.input_hashes([p for p in inputs if p not in raw]), "raw_inputs": raw_inputs(raw)}
    return record, inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER, required=True)
    parser.add_argument("--set", help="one unit (e.g. crossfit_0)")
    parser.add_argument("--family")
    parser.add_argument("--name")
    args = parser.parse_args()
    determinism.configure()
    pair = args.pair
    code = manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair, codehash.extra(STAGE, pair))
    primary = next(iter(PAIRS[pair]["fold_sets"]))
    for unit in [u for u in folds.units(pair) if u[0] == primary]:   # split67 takes no family
        if args.set and folds.unit_name(*unit) != args.set:
            continue
        c = context(pair, unit)
        for family, name, compute, extra in jobs(pair, unit):
            if (args.family and family != args.family) or (args.name and name != args.name):
                continue
            record, inputs = record_for(pair, unit, family, name, extra, code, c)
            if manifest.is_done(outputs(pair, unit, family, name), record):
                print(f"{pair} {folds.unit_name(*unit)} {family} {name}: done", flush=True)
                continue
            cost = manifest.begin()
            run(pair, unit, family, name, compute, record, cost, inputs)
            print(f"{pair} {folds.unit_name(*unit)} {family} {name}: read out", flush=True)


if __name__ == "__main__":
    main()
