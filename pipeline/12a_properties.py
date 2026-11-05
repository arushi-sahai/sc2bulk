"""Stage 12a: Fig 6A, sample properties predictable from each representation, and the Fig 6 title licence.
No bulk data and no bulk label is read.

Per pair, its primary fold set's units, and each of 11 rows (the ten configurations and the PCA-64 reference row):
the cross-validated R² of predicting each sample property from the representation on held-out single-cell samples,
and the phenotype column's AUROC, on Fig 2B's rows unchanged:
- pairs other than human blood, aging: per encoder split e = 0-9, fit on split e's training side (read-out folds
  encoder_split_e), score its validation side; trained configurations use their seed-e embeddings (trained on split e;
  asserted); encoder_seed e on every row;
- human blood, aging (crossfit_0-3): fit on the fold's training samples (read-out folds "training"), score its held-out
  donors' single-cell samples; trained configurations at encoder seeds 0-9, fixed ones once (encoder_seed -1).
Properties (sc2bulk.properties): depth, mitochondrial fraction (n/a for mouse organs: all zero, asserted), composition
PC1 at the deconvolution level (n/a for mouse microglia: one type); mouse organs' targets centered within organ with the
fit rows' means. Ridge with the read-out's scaling, grid and folds, read-out seeds 0-9 (properties.ridge_regression).
The phenotype column (AUROC, scoring-unit means, stage 11's rule): the ten configurations' values are read from stage
08's within predictions (no refit, so equal to Fig 2B); PCA-64's is fit here with readout.run on the same rows.
Chance level: N_NULL draws with predictions fixed and held-out targets permuted (continuous: within strata; phenotype:
0/1 labels across holdout groups), one draw set per unit shared by every row and property, the statistic recomputed
and averaged exactly as the observed one; human blood, aging's pair-level null averages its folds' draws. The rows
reuse stage 08's feature loaders (runpy, as 03b and 07d do; codehash.extra), only their single-cell parts.

Outputs, in <outputs>/12a_properties/<pair>/<unit>/:
  sample_properties.csv         sample_id, side (train / heldout), depth, mito_fraction, composition_pc1[, organ]; label-free
  <key>_property_r2.parquet     property_r2 kind: one row per (encoder seed, read-out seed, property); cv_r2 (AUROC for
                                the phenotype), the pair-level null_mean and null_q95, n_fit, n_scored, alpha,
                                alpha_at_boundary (NaN for phenotype rows read from stage 08), n_components (PCA-64),
                                did_train (stage 06's record; set encoders and CytoSet on pseudobulk)
  null_draws.csv                draw, key, property, value: the unit's null of the displayed statistic
and, once every pair's outputs exist, <outputs>/12a_properties/licence.json (properties.licence on the split-mean R²).

  python pipeline/12a_properties.py --pair PAIR
"""
import argparse
import json
import runpy

import numpy as np
import pandas as pd

from sc2bulk import codehash, determinism, folds, manifest, paths, properties, readout, schemas, seeds, training_inputs
from sc2bulk.config import configurations, deconvolution, settings, triage
from sc2bulk.config import properties as P
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

STAGE = "12a_properties"
S = runpy.run_path(str(settings.REPO / "pipeline" / "08_readout.py"), run_name="stage08")   # its loaders, not its main
N_READOUT, N_ENCODER, TRAINED = S["N_READOUT"], S["N_ENCODER"], S["TRAINED"]
PCA = "pseudobulk_pca64"
# a configuration cut before unblinding (config/triage) has no outputs to read: it is left out here and in the licence
KEPT = tuple(k for k in configurations.CONFIGURATION_ORDER if k not in triage.CUT_CONFIGURATIONS)
ROWS = (*KEPT, PCA)


def primary_units(pair) -> list:
    return [u for u in folds.units(pair) if u[0] == next(iter(PAIRS[pair]["fold_sets"]))]


def sample_table(pair, unit) -> pd.DataFrame:
    """The unit's single-cell samples (training, plus a crossfit fold's held-out donors') with their 0/1 group, holdout
    group, scoring unit, side and properties, from their stored cells."""
    train = folds.training_samples(pair, *unit).assign(side="train")
    obs, parts = [folds.training_cells(pair, *unit).obs], [train]
    if unit[0] == "crossfit":
        obs.append(folds.scored_cells(pair, *unit).obs)
        parts.append(S["heldout_samples"](pair, unit).drop(columns="scoring_unit_id").assign(side="heldout"))
    obs = pd.concat(obs)
    table = pd.concat(parts, ignore_index=True)
    table = table.assign(scoring_unit_id=S["scoring_units"](pair, table).astype(str)).set_index("sample_id")
    sample_id = obs["sample_id"].astype(str).to_numpy()
    table["depth"] = properties.depth(obs["total_counts"], sample_id)
    mito = properties.mito_fraction(obs["percent_mito"], sample_id)
    assert bool((mito == 0).all()) == (pair in P.MITO_NA_PAIRS), f"{pair}: mitochondrial fractions all zero only for {P.MITO_NA_PAIRS}"
    table["mito_fraction"] = np.nan if pair in P.MITO_NA_PAIRS else mito
    level = deconvolution.LEVELS[pair]
    if level is None:   # one cell type (mouse microglia): n/a
        table["composition_pc1"] = np.nan
    else:
        stratum = table[level["within"]].astype(str) if level["within"] else pd.Series("all", index=table.index)
        table["composition_pc1"] = properties.composition_pc1(training_inputs.level_types(pair, obs), sample_id,
                                                              list(train["sample_id"]), stratum)
    assert table["depth"].notna().all()
    return table


def fit_sets(pair, unit, table) -> list:
    """[(fit sample ids, scored sample ids, read-out fold context)]: per encoder split e, or a crossfit fold's one."""
    train_ids = list(table.index[table["side"] == "train"])
    if unit[0] == "crossfit":
        return [(train_ids, list(table.index[table["side"] == "heldout"]), "training")]
    out = []
    for e in range(N_ENCODER):
        sides = training_inputs.encoder_sides(pair, unit, e)
        out.append(([s for s in train_ids if sides[s] == "train"], [s for s in train_ids if sides[s] == "validation"], f"encoder_split_{e}"))
    return out


def fits_of(unit, key) -> list:
    """[(encoder seed, index into fit_sets)] for one row (Fig 2B's encoder_seed convention)."""
    if unit[0] != "crossfit":
        return [(e, e) for e in range(N_ENCODER)]
    return [(e, 0) for e in range(N_ENCODER)] if key in TRAINED else [(-1, 0)]


def features(pair, unit, key, e, table) -> pd.DataFrame:
    """sample_id -> feature row, for every sample of the table: stage 06's seed-e embeddings (their single-cell rows; on
    an encoder-split unit the split must be split e), pseudobulk (training samples' stored profiles; held-out from their
    cells, as stage 08), or the FMs' shared test-bag means / profile embeddings."""
    ids = list(table.index)
    train_ids, held_ids = list(table.index[table["side"] == "train"]), list(table.index[table["side"] == "heldout"])
    if key in TRAINED:
        obs, X = S["trained_embedding"](pair, unit, key, e)
        single = obs[obs["side"] != "bulk"].set_index("sample_id")
        assert set(single.index) == set(ids), f"{pair} {unit} {key} s{e}: single-cell rows differ"
        if unit[0] != "crossfit":
            sides = training_inputs.encoder_sides(pair, unit, e)
            assert all(single.loc[s, "side"] == ("single_cell_train" if sides[s] == "train" else "single_cell_validation")
                       for s in ids), f"{pair} {unit} {key} s{e}: stage 06's sides differ from split {e}"
        return pd.DataFrame(X[single.loc[ids, "row"].to_numpy()], index=ids)
    if key in ("pseudobulk", PCA):
        profiles = training_inputs.pseudobulk_log(pair, unit)
        X_train = np.vstack([profiles[s] for s in train_ids])
        held_key = "pseudobulk"
    else:
        model, input_ = configurations.CONFIGURATIONS[key]["model"], configurations.CONFIGURATIONS[key]["input"]
        X_train = (np.asarray(S["fm_file"](pair, unit, model, "pseudobulk")[train_ids].to_memory().X, dtype=np.float64)
                   if input_ == "pseudobulk" else S["bag_means"](S["fm_cells"](pair, unit, model, train_ids), train_ids, pair, unit))
        held_key = key
    X = np.vstack([X_train, S["heldout_features"](pair, unit, held_key, -1, held_ids)]) if held_ids else X_train
    return pd.DataFrame(X, index=train_ids + held_ids)


def key_columns(pair, unit, key) -> dict:
    if key == PCA:
        reference = configurations.REFERENCE_ROWS[PCA]
        return {"pair": pair, "fold_set": unit[0], "fold": unit[1], "row_kind": "reference", "key": PCA,
                "model": reference["model"], "input": reference["input"], "n_cells": 0, "pooling": "native"}
    return S["key_columns"](pair, unit, key)


def did_train(pair, unit, key, e):
    """Stage 06's did-it-train flag for a set-encoder seed (None for scVI and the fixed representations)."""
    if key not in TRAINED or key == "scvi_cells":
        return None
    row, encoder = TRAINED[key]
    return bool(json.loads((paths.output("06_train", pair, row, unit) / encoder / f"record_s{e}.json").read_text())["did_train"])


def within_scores(pair, unit, key, e, scored_ids) -> np.ndarray:
    """Stage 08's Fig 2B ridge scores of encoder seed e: read-out seeds x scored samples."""
    table = schemas.read("predictions", paths.output("08_readout", pair, f"{key}_within_predictions.parquet", unit))
    table = table[(table["readout"] == "ridge") & (table["encoder_seed"] == e)]
    assert sorted(table["readout_seed"].unique()) == list(range(N_READOUT)), f"{pair} {unit} {key} s{e}: read-out seeds"
    scores = table.pivot(index="readout_seed", columns="sample_id", values="score")
    assert set(scores.columns) == set(scored_ids), f"{pair} {unit} {key} s{e}: Fig 2B's scored samples differ"
    return scores[scored_ids].to_numpy()


def unit_tables(pair, unit) -> dict:
    """The unit's sample properties, rows per key, and null draws: {"samples", "rows": {key: [dict]}, "null": {(key,
    property): draws}}."""
    table = sample_table(pair, unit)
    strata_column = PAIRS[pair].get("strata")
    strata = lambda ids: table.loc[ids, strata_column].astype(str).to_numpy() if strata_column else None
    continuous = [p for p in P.CONTINUOUS if table[p].notna().any()]
    assert all(table[p].notna().all() for p in continuous), f"{pair} {unit}: a property missing for some samples"
    sets = fit_sets(pair, unit, table)
    cv = {k: [S["readout_folds"](pair, unit, fit_ids, r, context) for r in range(N_READOUT)] for k, (fit_ids, _, context) in enumerate(sets)}
    rows, held = {}, {}   # held[(key, property)]: [(set index, held-out target or labels, read-out seeds x n predictions)]
    for key in ROWS:
        rows[key] = []
        fixed = None if key in TRAINED else features(pair, unit, key, -1, table)
        for e, k in fits_of(unit, key):
            fit_ids, scored_ids, _ = sets[k]
            X = features(pair, unit, key, e, table) if key in TRAINED else fixed
            X_fit, X_scored = X.loc[fit_ids].to_numpy(), X.loc[scored_ids].to_numpy()
            base = {**key_columns(pair, unit, key), "encoder_seed": e, "readout": "ridge", "n_fit": len(fit_ids),
                    "n_scored": len(scored_ids), "n_components": np.nan, "did_train": did_train(pair, unit, key, e)}
            if key == PCA:
                X_fit, X_scored, base["n_components"] = properties.pca64(X_fit, X_scored, strata(fit_ids), strata(scored_ids))
            for prop in continuous:
                y_fit, y_scored = properties.centered(table.loc[fit_ids, prop], table.loc[scored_ids, prop], strata(fit_ids), strata(scored_ids))
                out = [properties.ridge_regression(X_fit, y_fit, cv[k][r], X_scored, strata(fit_ids), strata(scored_ids)) for r in range(N_READOUT)]
                prediction = np.vstack([o["prediction"] for o in out])
                held.setdefault((key, prop), []).append((k, y_scored, prediction))
                rows[key] += [{**base, "readout_seed": r, "property": prop, "cv_r2": float(properties.r2(y_scored, prediction[r])),
                               "alpha": o["alpha"], "alpha_at_boundary": o["alpha_at_boundary"]} for r, o in enumerate(out)]
            if key == PCA:
                out = [readout.run(X_fit, table.loc[fit_ids, "group"].to_numpy(), cv[k][r], None, strata_train=strata(fit_ids),
                                   X_scored=X_scored, strata_scored=strata(scored_ids)) for r in range(N_READOUT)]
                scores, alphas = np.vstack([o["scored_score"] for o in out]), [(o["alpha"], o["alpha_at_boundary"]) for o in out]
            else:   # Fig 2B's own predictions: no refit
                scores, alphas = within_scores(pair, unit, key, e, scored_ids), [(np.nan, None)] * N_READOUT
            label, unit_ids = table.loc[scored_ids, "group"].to_numpy(), table.loc[scored_ids, "scoring_unit_id"].to_numpy()
            held.setdefault((key, "phenotype"), []).append((k, label, scores))
            rows[key] += [{**base, "readout_seed": r, "property": "phenotype", "cv_r2": properties.unit_auroc(scores[r], label, unit_ids),
                           "alpha": a, "alpha_at_boundary": b} for r, (a, b) in enumerate(alphas)]
    return {"samples": table, "rows": rows, "null": nulls(pair, unit, table, sets, held, strata)}


def nulls(pair, unit, table, sets, held, strata) -> dict:
    """(key, property) -> the unit's N_NULL null values of the displayed statistic: per draw, every fit set's held-out
    targets permuted (continuous: within strata, `permutation` PROPERTY_NULL_OFFSET; phenotype: labels across holdout
    groups, PROPERTY_NULL_OFFSET + 1; unit-keyed; the same permutations for every row and property), the statistic
    recomputed per read-out seed and averaged as the observed one (over read-out seeds, then fits)."""
    continuous = np.random.default_rng(seeds.derived("permutation", pair, P.PROPERTY_NULL_OFFSET, *unit))
    phenotype = np.random.default_rng(seeds.derived("permutation", pair, P.PROPERTY_NULL_OFFSET + 1, *unit))
    order = [properties.within_strata_permutations(np.full(len(ids), "all") if strata(ids) is None else strata(ids), P.N_NULL, continuous)
             for _, ids, _ in sets]
    labels = [properties.group_label_permutations(table.loc[ids, "holdout_group"], table.loc[ids, "group"], P.N_NULL, phenotype)
              for _, ids, _ in sets]
    out = {}
    for (key, prop), entries in held.items():
        if prop == "phenotype":
            draws = [properties.null_auroc(scores, labels[k], table.loc[sets[k][1], "scoring_unit_id"].to_numpy()) for k, _, scores in entries]
        else:
            draws = [properties.null_r2(y, prediction, order[k]) for k, y, prediction in entries]
        out[(key, prop)] = np.mean(draws, axis=0)
    return out


def outputs(pair, units) -> list:
    return [paths.output(STAGE, pair, name, unit) for unit in units
            for name in ["sample_properties.csv", "null_draws.csv", *(f"{key}_property_r2.parquet" for key in ROWS)]]


def inputs(pair, units) -> list:
    out = [*folds.inputs(pair), settings.OUTPUTS / "03_splits" / pair / "splits.csv", settings.OUTPUTS / "03_splits" / pair / "donor_folds.csv",
           settings.OUTPUTS / "03_splits" / pair / "readout_folds.csv", paths.output("04_panel", pair, "shared_genes.csv")]
    for unit in units:
        out.append(paths.output("04_panel", pair, "panel.csv", unit))
        out += [paths.output("08_readout", pair, f"{key}_within_predictions.parquet", unit) for key in KEPT]
        for key, (row, encoder) in [(k, v) for k, v in TRAINED.items() if k in KEPT]:
            out += [S["trained_file"](pair, unit, key, e) for e in range(N_ENCODER)]
            if key != "scvi_cells":
                out += [paths.output("06_train", pair, row, unit) / encoder / f"record_s{e}.json" for e in range(N_ENCODER)]
        for key in KEPT:
            config = configurations.CONFIGURATIONS[key]
            if config["model"] in ("geneformer", "scgpt"):
                out.append(paths.output("07_embed", pair, f"{config['model']}_panel_{config['input']}_all.h5ad", unit))
    return out


def run(pair, units, record, cost):
    tables = {unit: unit_tables(pair, unit) for unit in units}
    pair_null = {k: np.mean([t["null"][k] for t in tables.values()], axis=0) for k in next(iter(tables.values()))["null"]}   # over folds
    sources = inputs(pair, units)
    for unit, t in tables.items():
        columns = ["side", "depth", "mito_fraction", "composition_pc1"] + (["organ"] if "organ" in t["samples"] else [])
        write_csv(paths.output(STAGE, pair, "sample_properties.csv", unit), t["samples"][columns].rename_axis("sample_id").reset_index(),
                  record, cost, sources)
        draws = pd.DataFrame([{"draw": b, "key": key, "property": prop, "value": float(v)}
                              for (key, prop), values in t["null"].items() for b, v in enumerate(values)])
        write_csv(paths.output(STAGE, pair, "null_draws.csv", unit), draws, record, cost, sources)
        for key in ROWS:
            frame = pd.DataFrame(t["rows"][key])
            frame["null_mean"] = [float(pair_null[(key, p)].mean()) for p in frame["property"]]
            frame["null_q95"] = [float(np.quantile(pair_null[(key, p)], P.NULL_QUANTILE)) for p in frame["property"]]
            path = paths.output(STAGE, pair, f"{key}_property_r2.parquet", unit)
            schemas.write("property_r2", schemas.with_neutral_keys(frame), manifest.temporary(path))
            manifest.finish(path, record, cost, inputs=sources)


def write_csv(path, table, record, cost, sources):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(manifest.temporary(path), index=False)
    manifest.finish(path, record, cost, inputs=sources)


def write_licence():
    """licence.json, once every pair's 12a outputs exist: properties.licence on the split-mean R² of all six pairs, with
    the present learned configurations (the nine minus CUT_CONFIGURATIONS)."""
    files = [p for pair in PAIR_ORDER for p in outputs(pair, primary_units(pair)) if p.name.endswith("_property_r2.parquet")]
    if not all(p.exists() for p in files):
        return
    path = settings.OUTPUTS / STAGE / "licence.json"
    present = [k for k in KEPT if k != "pseudobulk"]
    record = {"stage": STAGE, "smoke": settings.SMOKE, "present": present,
              "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", None, codehash.extra(STAGE, None)),
              "inputs": manifest.input_hashes(files)}
    if manifest.is_done([path], record):
        return
    cost = manifest.begin()
    values = properties.split_mean(pd.concat([schemas.read("property_r2", p) for p in files], ignore_index=True))
    manifest.temporary(path).write_text(json.dumps(properties.licence(values, present), indent=1))
    manifest.finish(path, record, cost, inputs=files)
    print(f"Fig 6 title licence written: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER, required=True)
    args = parser.parse_args()
    determinism.configure()
    pair, units = args.pair, primary_units(args.pair)
    record = {"stage": STAGE, "pair": pair, "units": units, "smoke": settings.SMOKE, "n_null": P.N_NULL,
              "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair, codehash.extra(STAGE, pair)),
              "inputs": manifest.input_hashes(inputs(pair, units))}
    if manifest.is_done(outputs(pair, units), record):
        print(f"{pair}: done", flush=True)
    else:
        run(pair, units, record, manifest.begin())
        print(f"{pair}: sample properties predicted", flush=True)
    write_licence()


if __name__ == "__main__":
    main()
