"""Stage 08: the read-outs.

Per unit (pair, fold set, fold) and configuration: one row of features per training sample of the
unit (all of them) and per scored bulk sample; per-side scaling; ridge (primary) and
LASSO (secondary) of the 0/1 group with alpha chosen by grouped CV on stage 03's read-out folds, one
fit per read-out seed (sc2bulk.readout.run); the bulk scores written label-free. No bulk label is
read.

Configurations here: pseudobulk (log-CP10k over the shared genes, on the unit's panel) and the
foundation models on cells (each training sample = the mean over the shared 10 test bags of
FROZEN_NCELL cells of the mean cell embedding) and on pseudobulk (the profile's
embedding); every bulk sample is one profile. The trained configurations (CytoSet on cells and on
pseudobulk, Set Transformer, ScRAT, scVI) read stage 06's embeddings per encoder seed (0-9, plus the
attribution seeds 10-19 for rows A, B and J outside human blood, aging and the smoke run, less NOT_TRAINED;
stage 11 uses 0-9). Fixed representations carry encoder_seed -1 (no encoder level).

Three read-out families, each in its own files with its own run record (the primary files are the ones
stage 11 reads and hashes). On the pair's primary fold set only, besides the primary:
- Fig 2B, within single-cell: per encoder split e (0-9), the read-out fits on split e's
  training side (stage 03's encoder_split_e read-out folds) and scores its validation side, with the
  training side's scaler; encoder_seed e for every configuration. Human blood, aging (crossfit units):
  instead, the fold's held-out donors' single-cell samples are scored by the fold's primary heads, so the primary pass never reads them; no heads of its own, encoder_seed as the primary fit's.
- Out of fold (the key sensitivity): per encoder split e passing the guard, the read-out
  trains on split e's validation side only (CV folds dealt as stage 03's, readout.readout_folds) and
  scores the unit's bulk; encoder_seed e for every configuration, variant out_of_fold_readout. The
  guard: split e's validation side has >= OUT_OF_FOLD_MIN_GROUPS holdout groups of each label and >=
  READOUT_FOLDS in total; a failing split is dropped for every configuration of the unit (stage 11b:
  a pair with any primary unit of fewer than 2 passing splits is not estimable in this row).

Outputs, in <outputs>/08_readout/<pair>/<unit>/:
  <configuration>_predictions.parquet   predictions kind: bulk scores per encoder seed, read-out, read-out seed
  <configuration>_heads.parquet         heads kind: coef per feature, intercept, both sides' scalers, alpha
  <configuration>_within_predictions.parquet, _within_heads.parquet   Fig 2B (single-cell side only;
                                        human blood, aging: predictions only)
  <configuration>_out_of_fold_predictions.parquet, _out_of_fold_heads.parquet   the out-of-fold row
  out_of_fold_guard.csv                 per encoder split: validation holdout groups per label, passes

  python pipeline/08_readout.py --pair PAIR [--set UNIT] [--configuration KEY]
"""
import argparse

import anndata
import numpy as np
import pandas as pd

from sc2bulk import bags, determinism, folds, manifest, paths, preprocess, readout, schemas, seeds, training_inputs
from sc2bulk.config import configurations, representations, settings, smoke
from sc2bulk.config import readout as R
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

STAGE = "08_readout"
N_READOUT = smoke.N_READOUT_SEEDS if settings.SMOKE else settings.N_READOUT_SEEDS
N_ENCODER = smoke.N_ENCODER_SEEDS if settings.SMOKE else settings.N_ENCODER_SEEDS   # Fig 2B's and the out-of-fold row's splits
READY = configurations.CONFIGURATION_ORDER
# trained configurations: stage 06's row and encoder
TRAINED = {"cytoset_cells": ("A", "cytoset"), "cytoset_pseudobulk": ("B", "cytoset"), "set_transformer_cells": ("A", "set_transformer"),
           "scrat_cells": ("A", "scrat"), "scvi_cells": ("J", "scvi")}
# encoder seeds stage 06 lists but cannot train: mouse organs' scVI seed 14 (an attribution seed) fails
# in scvi-tools on a final training minibatch of one cell (BatchNorm); stage 06 still lists the job (FAILED)
NOT_TRAINED = {("tms", "scvi_cells", 14)}


def encoder_seeds(pair, key) -> list:
    """A trained configuration's encoder seeds: stage 06's rule for its row (training_inputs.encoder_seeds) less
    NOT_TRAINED; stage 11 uses 0-9."""
    return [e for e in training_inputs.encoder_seeds(pair, TRAINED[key][0]) if (pair, key, e) not in NOT_TRAINED]


def trained_file(pair, unit, key, e):
    row, encoder = TRAINED[key]
    return paths.output("06_train", pair, row, unit) / encoder / "embeddings" / f"s{e}.h5ad"


def trained_embedding(pair, unit, key, e):
    """Stage 06's embedding file of encoder seed e: its obs (with each row's position) and X."""
    path = trained_file(pair, unit, key, e)
    made = manifest.read(path)["settings"]["code"]   # stage 06's outputs must be current (a crashed rerun could mix)
    assert made == manifest.stage_code_hash(settings.REPO / "pipeline" / "06_train.py", pair), f"{path}: stale stage 06 embedding"
    data = anndata.read_h5ad(path)
    return data.obs.assign(row=np.arange(data.n_obs)), np.asarray(data.X, dtype=np.float64)


def trained_features(pair, unit, key, e, train_ids, bulk_ids, sides=None):
    """Stage 06's embeddings of encoder seed e: the training samples' rows (both sides of the encoder's
    split; with sides, stage 03's split e, asserted to be the split stage 06 trained on) and the scored
    bulk's primary rows."""
    obs, X = trained_embedding(pair, unit, key, e)
    single = obs[obs["side"].isin(["single_cell_train", "single_cell_validation"])].set_index("sample_id")
    bulk = obs[(obs["side"] == "bulk") & (obs["variant"] == "primary")].set_index("sample_id")
    assert set(single.index) == set(train_ids) and set(bulk.index) == set(bulk_ids), f"{pair} {unit} {key} s{e}: rows differ"
    if sides is not None:
        assert all(single.loc[s, "side"] == ("single_cell_train" if sides[s] == "train" else "single_cell_validation")
                   for s in train_ids), f"{pair} {unit} {key} s{e}: stage 06's sides differ from split {e}"
    return X[single.loc[list(train_ids), "row"].to_numpy()], X[bulk.loc[list(bulk_ids), "row"].to_numpy()]


def scoring_units(pair: str, table: pd.DataFrame) -> pd.Series:
    """Each row's scoring unit: the donor where the pair scores donors, else the sample."""
    return table["donor_id" if PAIRS[pair]["scoring_unit"] == "donor" else "sample_id"]


def scored(pair: str, unit: tuple) -> pd.DataFrame:
    """The unit's scored bulk samples (kept; label-free), in bulk.h5ad order, with their scoring unit."""
    bulk = pd.read_csv(settings.OUTPUTS / "02_preprocess" / pair / "bulk_samples.csv", dtype={"sample_id": str, "donor_id": str})
    kept = bulk[bulk["drop_reason"].isna()]
    table = folds.scored_bulk(pair, *unit, kept).reset_index(drop=True)
    return table.assign(scoring_unit_id=scoring_units(pair, table))


def heldout_samples(pair: str, unit: tuple) -> pd.DataFrame:
    """A cross-fitting fold's scored donors' single-cell samples (Fig 2B's held-out rows), in sample-id
    order, with their scoring unit."""
    donors = pd.read_csv(settings.OUTPUTS / "03_splits" / pair / "donor_folds.csv", dtype={"donor_id": str})
    held = donors[(donors["fold_set"] == unit[0]) & (donors["fold"] == unit[1]) & (donors["side"] == "scored")]
    samples = folds.single_cell_samples(pair)
    samples = samples[samples["donor_id"].isin(set(held["donor_id"]))].sort_values("sample_id").reset_index(drop=True)
    return samples.assign(scoring_unit_id=scoring_units(pair, samples))


def strata_of(pair, table):
    strata = PAIRS[pair].get("strata")
    return table[strata].to_numpy() if strata else None


def pseudobulk_features(pair, unit, train_ids, bulk_ids):
    profiles = training_inputs.pseudobulk_log(pair, unit)
    genes, chosen = training_inputs.shared(pair), training_inputs.panel(pair, unit)
    bulk = anndata.read_h5ad(settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad")[list(bulk_ids)]
    X_bulk = preprocess.log_normalize(bulk[:, genes].X)[:, pd.Index(genes).get_indexer(chosen)].toarray()
    return np.vstack([profiles[s] for s in train_ids]), X_bulk


def fm_file(pair, unit, model, what):
    """A stage 07 embedding file of the unit's panel, read backed (only the rows used are loaded), after
    asserting it was made from the current store and the unit's own panel. Stage 07 embeds every
    stored cell under its cell id, so its cells per sample are the store's and the shared drawer's bags
    are the set encoders' bags."""
    path = paths.output("07_embed", pair, f"{model}_panel_{what}_all.h5ad", unit)
    made = manifest.read(path)["settings"]
    assert all(made["inputs"][k] == v for k, v in manifest.input_hashes(folds.inputs(pair)).items()), \
        f"{path.name}: not made from the current single-cell store"
    assert tuple(made["unit"]) == tuple(unit) and made["panel"] == manifest.read(paths.output("04_panel", pair, "panel.csv", unit))["content_hash"], \
        f"{path.name}: not the unit's own panel"
    return anndata.read_h5ad(path, backed="r")


def fm_cells(pair, unit, model, sample_ids):
    """A stage 07 cells file's cells of the given samples only."""
    cells = fm_file(pair, unit, model, "cells")
    keep = cells.obs["sample_id"].astype(str).isin(set(sample_ids)).to_numpy()
    cells = cells[keep].to_memory()
    assert set(cells.obs["sample_id"].astype(str)) == set(sample_ids)
    return cells


def bag_means(cells, sample_ids, pair, unit) -> np.ndarray:
    """Each sample's mean over the shared test bags of FROZEN_NCELL cells of the mean cell embedding."""
    ids, sample = cells.obs_names.astype(str).to_numpy(), cells.obs["sample_id"].astype(str).to_numpy()
    row = pd.Series(np.arange(len(ids)), index=ids)
    test = bags.test_bags(ids, sample, list(sample_ids), T.FROZEN_NCELL, representations.TEST_TIME_BAGS, pair, unit)
    X = cells.X
    return np.vstack([np.mean([X[row[b].to_numpy()].mean(axis=0, dtype=np.float64) for b in test[s]], axis=0)
                      for s in sample_ids])


def fm_features(pair, unit, model, input_, train_ids, bulk_ids):
    bulk = np.asarray(fm_file(pair, unit, model, "bulk")[list(bulk_ids)].to_memory().X, dtype=np.float64)
    if input_ == "pseudobulk":
        return np.asarray(fm_file(pair, unit, model, "pseudobulk")[list(train_ids)].to_memory().X, dtype=np.float64), bulk
    return bag_means(fm_cells(pair, unit, model, train_ids), train_ids, pair, unit), bulk   # the unit's training samples' cells only


def features(pair, unit, key, train_ids, bulk_ids) -> list:
    """[(encoder seed, training rows, bulk rows)] for one configuration, rows in the given sample orders;
    fixed representations have one, encoder seed -1."""
    if key == "pseudobulk":
        return [(-1, *pseudobulk_features(pair, unit, train_ids, bulk_ids))]
    if key in TRAINED:
        return [(e, *trained_features(pair, unit, key, e, train_ids, bulk_ids)) for e in encoder_seeds(pair, key)]
    model, input_ = configurations.CONFIGURATIONS[key]["model"], configurations.CONFIGURATIONS[key]["input"]
    return [(-1, *fm_features(pair, unit, model, input_, train_ids, bulk_ids))]


def split_features(pair, unit, key, train_ids, bulk_ids, splits) -> list:
    """[(e, training rows, bulk rows)] per encoder split e (Fig 2B, the out-of-fold row): a trained
    configuration's seed-e embeddings (trained on split e, asserted), a fixed representation's primary
    rows for every e."""
    if key in TRAINED:
        return [(e, *trained_features(pair, unit, key, e, train_ids, bulk_ids, training_inputs.encoder_sides(pair, unit, e)))
                for e in splits]
    (_, X_train, X_bulk), = features(pair, unit, key, train_ids, bulk_ids)
    return [(e, X_train, X_bulk) for e in splits]


def heldout_features(pair, unit, key, e, ids) -> np.ndarray:
    """The held-out donors' single-cell samples' features: stage 06's single_cell_heldout rows of seed e;
    pseudobulk from their stored cells (summed over the shared genes, log-CP10k, then the panel: stage 02's
    pseudobulk, as stage 06 computes it); the FMs' shared test-bag means of their cells, or their profiles."""
    if key in TRAINED:
        obs, X = trained_embedding(pair, unit, key, e)
        held = obs[obs["side"] == "single_cell_heldout"].set_index("sample_id")
        assert set(held.index) == set(ids), f"{pair} {unit} {key} s{e}: held-out rows differ"
        return X[held.loc[ids, "row"].to_numpy()]
    if key == "pseudobulk":
        data = folds.scored_cells(pair, *unit)
        genes, chosen = training_inputs.shared(pair), training_inputs.panel(pair, unit)
        raw, sample = data[:, genes].X, data.obs["sample_id"].astype(str).to_numpy()
        columns = pd.Index(genes).get_indexer(chosen)
        return np.vstack([preprocess.log_normalize(preprocess.summed(raw[sample == s])[None, :])[:, columns].toarray().ravel()
                          for s in ids])
    model, input_ = configurations.CONFIGURATIONS[key]["model"], configurations.CONFIGURATIONS[key]["input"]
    if input_ == "pseudobulk":
        return np.asarray(fm_file(pair, unit, model, "pseudobulk")[ids].to_memory().X, dtype=np.float64)
    return bag_means(fm_cells(pair, unit, model, ids), ids, pair, unit)


def readout_folds(pair, unit, train_ids, r, context="training") -> np.ndarray:
    table = pd.read_csv(settings.OUTPUTS / "03_splits" / pair / "readout_folds.csv", dtype={"sample_id": str})
    rows = table[(table["fold_set"] == unit[0]) & (table["fold"] == unit[1]) & (table["context"] == context)
                 & (table["readout_index"] == r)].set_index("sample_id")
    assert set(rows.index) == set(train_ids), f"{pair} {unit}: read-out folds don't cover the training samples"
    return rows.loc[list(train_ids), "readout_fold"].to_numpy()


def guard(pair, unit) -> pd.DataFrame:
    """The minimum-n guard, per encoder split e: the validation side's holdout groups per label
    (stage 03 already asserts >= 2) and whether the split enters the out-of-fold row. Label-free: stage 03's
    splits and the single-cell labels only."""
    train = folds.training_samples(pair, *unit)
    rows = []
    for e in range(N_ENCODER):
        sides = training_inputs.encoder_sides(pair, unit, e)
        validation = train[train["sample_id"].map(sides) == "validation"]
        n0, n1 = (validation.loc[validation["group"] == g, "holdout_group"].nunique() for g in (0, 1))
        rows.append({"pair": pair, "fold_set": unit[0], "fold": unit[1], "encoder_seed": e, "groups_label_0": n0, "groups_label_1": n1,
                     "passes": min(n0, n1) >= R.OUT_OF_FOLD_MIN_GROUPS and n0 + n1 >= settings.READOUT_FOLDS})
    return pd.DataFrame(rows)


def key_columns(pair, unit, key) -> dict:
    config = configurations.CONFIGURATIONS[key]
    return {"pair": pair, "fold_set": unit[0], "fold": unit[1], "row_kind": "configuration", "key": key,
            "model": config["model"], "input": config["input"],
            "n_cells": T.FROZEN_NCELL if config["input"] == "cells" else 0,
            "pooling": "mean" if config["model"] in schemas.EMBED_ONLY else "native"}


def feature_names(pair, unit, key, X) -> list:
    return training_inputs.panel(pair, unit) if key == "pseudobulk" else list(np.arange(X.shape[1]).astype(str))


def fits(X_train, y, cv_folds, X_bulk, strata_train, strata_bulk, X_scored=None, strata_scored=None):
    """Ridge and LASSO at every read-out seed r: (read-out, r, sc2bulk.readout.run's output)."""
    for name in ("ridge", "lasso"):
        for r in range(N_READOUT):
            yield name, r, readout.run(X_train, y, cv_folds[r], X_bulk, readout=name, strata_train=strata_train,
                                       strata_bulk=strata_bulk, X_scored=X_scored, strata_scored=strata_scored)


def prediction_rows(row, side, ids, units, score) -> pd.DataFrame:
    return pd.DataFrame({**row, "side": side, "sample_id": ids, "scoring_unit_id": units, "score": score,
                         "probability": np.clip(score, 0.0, 1.0)})


def head_rows(row, out, dims) -> list:
    """One fit's heads: per scaler side and stratum, the coefficients, intercept, scaler moments and alpha."""
    return [pd.DataFrame({**row, "side": side, "stratum": str(stratum), "gene_or_dim": dims,
                          "coef": out["coef"], "intercept": out["intercept"], "scaler_mean": mean,
                          "scaler_scale": sd, "alpha": out["alpha"],
                          "alpha_at_boundary": out["alpha_at_boundary"],
                          "converged": out.get("converged", True), "n_iter": out.get("n_iter", -1),
                          "n_nonzero": out.get("n_nonzero", int((out["coef"] != 0).sum()))})
            for side, moments in out["scalers"].items() for stratum, (mean, sd) in moments.items()]


def primary(pair, unit, key) -> dict:
    """The transfer read-out: all of the unit's training samples, its scored bulk."""
    train = folds.training_samples(pair, *unit)
    train_ids = list(train["sample_id"])
    bulk = scored(pair, unit)
    bulk_ids = list(bulk["sample_id"])
    cv_folds = {r: readout_folds(pair, unit, train_ids, r) for r in range(N_READOUT)}
    keys = key_columns(pair, unit, key)
    predictions, heads = [], []
    for e, X_train, X_bulk in features(pair, unit, key, train_ids, bulk_ids):
        dims = feature_names(pair, unit, key, X_train)
        for name, r, out in fits(X_train, train["group"].to_numpy(), cv_folds, X_bulk, strata_of(pair, train), strata_of(pair, bulk)):
            row = {**keys, "encoder_seed": e, "readout": name, "readout_seed": r}
            predictions.append(prediction_rows(row, "bulk", bulk_ids, bulk["scoring_unit_id"].to_numpy(), out["score"]))
            heads += head_rows(row, out, dims)
    return {"predictions": predictions, "heads": heads}


def within(pair, unit, key) -> dict:
    """Fig 2B: split e's validation side scored by a fit on its training side; or, for a cross-fitting
    fold, its held-out donors' samples scored by the fold's stored primary heads."""
    train = folds.training_samples(pair, *unit)
    train = train.assign(scoring_unit_id=scoring_units(pair, train))
    train_ids = list(train["sample_id"])
    keys = key_columns(pair, unit, key)
    predictions, heads = [], []
    if unit[0] == "crossfit":
        held = heldout_samples(pair, unit)
        ids = list(held["sample_id"])
        stored = schemas.read("heads", paths.output(STAGE, pair, f"{key}_heads.parquet", unit))
        stored = stored[stored["side"] == "single_cell"]
        X = {e: heldout_features(pair, unit, key, e, ids) for e in stored["encoder_seed"].unique()}
        dims = feature_names(pair, unit, key, next(iter(X.values())))
        for (e, name, r), head in stored.groupby(["encoder_seed", "readout", "readout_seed"], sort=False):
            moments = {s: (h["scaler_mean"].to_numpy(), h["scaler_scale"].to_numpy()) for s, h in head.groupby("stratum", sort=False)}
            first = head[head["stratum"] == head["stratum"].iloc[0]]
            assert list(first["gene_or_dim"]) == list(dims), f"{pair} {unit} {key}: head features differ"
            score = readout.scale(X[e], moments, strata_of(pair, held)) @ first["coef"].to_numpy() + first["intercept"].iloc[0]
            row = {**keys, "encoder_seed": int(e), "readout": name, "readout_seed": int(r)}
            predictions.append(prediction_rows(row, "single_cell_heldout", ids, held["scoring_unit_id"].to_numpy(), score))
        return {"predictions": predictions}
    bulk_ids = list(scored(pair, unit)["sample_id"])
    for e, X_train, _ in split_features(pair, unit, key, train_ids, bulk_ids, range(N_ENCODER)):
        dims = feature_names(pair, unit, key, X_train)
        fit = (train["sample_id"].map(training_inputs.encoder_sides(pair, unit, e)) == "train").to_numpy()
        fitted, validation = train[fit], train[~fit]
        cv_folds = {r: readout_folds(pair, unit, list(fitted["sample_id"]), r, f"encoder_split_{e}") for r in range(N_READOUT)}
        for name, r, out in fits(X_train[fit], fitted["group"].to_numpy(), cv_folds, None, strata_of(pair, fitted), None,
                                 X_train[~fit], strata_of(pair, validation)):
            row = {**keys, "encoder_seed": e, "readout": name, "readout_seed": r}
            predictions.append(prediction_rows(row, "single_cell_validation", list(validation["sample_id"]),
                                               validation["scoring_unit_id"].to_numpy(), out["scored_score"]))
            heads += head_rows(row, out, dims)
    return {"predictions": predictions, "heads": heads}


def out_of_fold(pair, unit, key) -> dict:
    """The out-of-fold row: per passing encoder split e, a fit on split e's validation side only
    (CV folds dealt as stage 03 deals them, with its seed), scoring the unit's bulk."""
    train = folds.training_samples(pair, *unit)
    train_ids = list(train["sample_id"])
    bulk = scored(pair, unit)
    bulk_ids = list(bulk["sample_id"])
    passing = list(guard(pair, unit).query("passes")["encoder_seed"])
    keys = key_columns(pair, unit, key)
    predictions, heads = [], []
    for e, X_train, X_bulk in split_features(pair, unit, key, train_ids, bulk_ids, passing):
        dims = feature_names(pair, unit, key, X_train)
        validation = (train["sample_id"].map(training_inputs.encoder_sides(pair, unit, e)) == "validation").to_numpy()
        samples = train[validation]
        cv_folds = {r: readout.readout_folds(samples, seeds.derived("readout_folds", pair, r, *unit)).to_numpy() for r in range(N_READOUT)}
        for name, r, out in fits(X_train[validation], samples["group"].to_numpy(), cv_folds, X_bulk, strata_of(pair, samples),
                                 strata_of(pair, bulk)):
            row = {**keys, "encoder_seed": e, "readout": name, "readout_seed": r, "variant": "out_of_fold_readout"}
            predictions.append(prediction_rows(row, "bulk", bulk_ids, bulk["scoring_unit_id"].to_numpy(), out["score"]))
            heads += head_rows(row, out, dims)
    return {"predictions": predictions, "heads": heads}


FAMILIES = {"primary": primary, "within": within, "out_of_fold": out_of_fold}


def families(pair, unit) -> list:
    """The unit's read-out families: the primary; on the pair's primary fold set, Fig 2B and (if any split
    passes the guard) the out-of-fold row."""
    if unit[0] != next(iter(PAIRS[pair]["fold_sets"])):
        return ["primary"]
    return ["primary", "within"] + (["out_of_fold"] if guard(pair, unit)["passes"].any() else [])


def outputs(pair, unit, key, family) -> list:
    """A family's files, predictions then heads (human blood, aging's held-out rows have no heads)."""
    name = key if family == "primary" else f"{key}_{family}"
    kinds = ("predictions",) if family == "within" and unit[0] == "crossfit" else ("predictions", "heads")
    return [paths.output(STAGE, pair, f"{name}_{kind}.parquet", unit) for kind in kinds]


def run(pair, unit, key, family, record, cost):
    tables = FAMILIES[family](pair, unit, key)
    for kind, path in zip(("predictions", "heads"), outputs(pair, unit, key, family)):
        path.parent.mkdir(parents=True, exist_ok=True)
        schemas.write(kind, schemas.with_neutral_keys(pd.concat(tables[kind], ignore_index=True)), manifest.temporary(path))
        manifest.finish(path, record, cost, inputs=inputs(pair, unit, key, family))


def inputs(pair, unit, key, family="primary") -> list:
    out = [*folds.inputs(pair), settings.OUTPUTS / "02_preprocess" / pair / "bulk_samples.csv",
           settings.OUTPUTS / "03_splits" / pair / "donor_folds.csv",
           settings.OUTPUTS / "03_splits" / pair / "readout_folds.csv", paths.output("04_panel", pair, "panel.csv", unit)]
    if family != "primary":
        out.append(settings.OUTPUTS / "03_splits" / pair / "splits.csv")
    if family == "within" and unit[0] == "crossfit":
        out.append(paths.output(STAGE, pair, f"{key}_heads.parquet", unit))
    if key == "pseudobulk":
        return out + [settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad", paths.output("04_panel", pair, "shared_genes.csv")]
    if key in TRAINED:
        return out + [trained_file(pair, unit, key, e) for e in encoder_seeds(pair, key)]
    model, input_ = configurations.CONFIGURATIONS[key]["model"], configurations.CONFIGURATIONS[key]["input"]
    return out + [paths.output("07_embed", pair, f"{model}_panel_{w}_all.h5ad", unit) for w in (input_, "bulk")]


def write_guard(pair, unit):
    """The guard's table, for the deposit's pass counts per pair and fold."""
    path = paths.output(STAGE, pair, "out_of_fold_guard.csv", unit)
    needed = [*folds.inputs(pair), settings.OUTPUTS / "03_splits" / pair / "splits.csv"]
    record = {"stage": STAGE, "pair": pair, "unit": unit, "smoke": settings.SMOKE,
              "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair),
              "inputs": manifest.input_hashes(needed)}
    if manifest.is_done([path], record):
        return
    cost = manifest.begin()
    path.parent.mkdir(parents=True, exist_ok=True)
    guard(pair, unit).to_csv(manifest.temporary(path), index=False)
    manifest.finish(path, record, cost, inputs=needed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER, required=True)
    parser.add_argument("--set", help="one unit (e.g. crossfit_0)")
    parser.add_argument("--configuration", choices=READY)
    args = parser.parse_args()
    determinism.configure()
    pair = args.pair
    for unit in folds.units(pair):
        if args.set and folds.unit_name(*unit) != args.set:
            continue
        unit_families = families(pair, unit)
        if "within" in unit_families:
            write_guard(pair, unit)
        for key in [args.configuration] if args.configuration else READY:
            train_ids, bulk_ids = list(folds.training_samples(pair, *unit)["sample_id"]), list(scored(pair, unit)["sample_id"])
            for family in unit_families:
                record = {"stage": STAGE, "pair": pair, "unit": unit, "configuration": key, "family": family, "smoke": settings.SMOKE,
                          "rows": manifest.settings_hash({"training": train_ids, "bulk": bulk_ids}),   # same rows for every configuration
                          "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair),
                          "inputs": manifest.input_hashes(inputs(pair, unit, key, family))}
                if manifest.is_done(outputs(pair, unit, key, family), record):
                    print(f"{pair} {folds.unit_name(*unit)} {key} {family}: done", flush=True)
                    continue
                cost = manifest.begin()
                run(pair, unit, key, family, record, cost)
                print(f"{pair} {folds.unit_name(*unit)} {key} {family}: read out", flush=True)


if __name__ == "__main__":
    main()
