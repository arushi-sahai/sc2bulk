"""Stage 09: alignment.

Per unit of the pair's primary fold set and configuration: stage 08's primary features (its own
functions, loaded from pipeline/08_readout.py as 03b and 07d load theirs), one row per training sample of
the unit and per scored bulk sample, at encoder seeds < N_ENCODER_SEEDS (seeds 10-19 are
attribution only) and -1 for fixed representations. For each alignment in ALIGNED (sc2bulk.alignment;
"unaligned_stacked" = none), the stacked scaler and ridge and LASSO of the 0/1 group at every read-out
seed (sc2bulk.readout.run(..., scaling="stacked"), stage 03's read-out folds), the bulk scores written
label-free. "none" is not refit: it is stage 08's primary predictions file, an input here, checked to be
made by the current stage 08 from the current inputs and the same rows, so Fig 3C's None column equals
Fig 2 by construction. The alignment and the scaler are fit once per encoder seed on all training rows,
before the read-out's CV (label-free, as stage 08's scaler). Harmony's seed: one per unit,
seeds.derived("alignment", pair, 0, *unit) mod 2^31. Heads as stage 08 writes them (every
read-out keeps its full ridge coefficients; LASSO's convergence recorded in every sweep), the stacked
moments under both sides' scaler rows (one scaler, applied to both). No bulk label is read.

Geometry (label-free), per encoder seed: the modality silhouette under every alignment ("none" after
per-side scaling) and silhouette_raw (alignment none, the unscaled rows); harmony_relative_change =
||Z_harmony - Z||_F / ||Z||_F over all stacked rows. For GEOMETRY_CONFIGURATIONS at fold 0 (Fig 3A,
S13A-E): pc1 and pc2 of every sample under all five alignments. Sides are told apart by sample id (a
unit's training and bulk ids are disjoint, asserted).

Outputs, in <outputs>/09_alignment/<pair>/<unit>/:
  <configuration>_predictions.parquet   predictions kind: bulk scores per alignment, encoder seed, read-out, read-out seed
  <configuration>_heads.parquet         heads kind: coef per feature, intercept, the stacked scaler, alpha, convergence
  <configuration>_geometry.parquet      alignment_geometry kind: silhouettes, Harmony's change, PC coordinates

  python pipeline/09_alignment.py --pair PAIR [--set UNIT] [--configuration KEY]
"""
import argparse
import runpy

import numpy as np
import pandas as pd

from sc2bulk import alignment, codehash, determinism, folds, manifest, paths, readout, schemas, seeds
from sc2bulk.config import alignment as A
from sc2bulk.config import configurations, settings
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

STAGE = "09_alignment"
STAGE08_SCRIPT = settings.REPO / "pipeline" / "08_readout.py"
S08 = runpy.run_path(str(STAGE08_SCRIPT), run_name="stage08")   # its functions, not its main


def units(pair) -> list:
    """The pair's primary fold set's units (S7 Table and Fig 3 use no other)."""
    return [unit for unit in folds.units(pair) if unit[0] == next(iter(PAIRS[pair]["fold_sets"]))]


def harmony_seed(pair, unit) -> int:
    return seeds.derived(A.SEED_PURPOSE, pair, 0, *unit) % A.HARMONY_SEED_MODULUS


def features(pair, unit, key, train_ids, bulk_ids) -> list:
    """Stage 08's [(encoder seed, training rows, bulk rows)], trained configurations at seeds < N_ENCODER_SEEDS."""
    if key in S08["TRAINED"]:
        return [(e, *S08["trained_features"](pair, unit, key, e, train_ids, bulk_ids))
                for e in S08["encoder_seeds"](pair, key) if e < settings.N_ENCODER_SEEDS]
    return S08["features"](pair, unit, key, train_ids, bulk_ids)


def stage08_predictions(pair, unit, key):
    return paths.output("08_readout", pair, f"{key}_predictions.parquet", unit)


def rows_hash(train_ids, bulk_ids) -> str:
    return manifest.settings_hash({"training": train_ids, "bulk": bulk_ids})   # stage 08's record field


def check_stage08(pair, unit, key, train_ids, bulk_ids):
    """Stage 08's primary file ("none") was made by the current stage 08, from the inputs stage 09 reads, on
    the same rows."""
    made = manifest.read(stage08_predictions(pair, unit, key))["settings"]
    assert made["code"] == manifest.stage_code_hash(STAGE08_SCRIPT, pair), f"{pair} {unit} {key}: stale stage 08 predictions"
    assert made["inputs"] == manifest.input_hashes(S08["inputs"](pair, unit, key)), f"{pair} {unit} {key}: stage 08 read other inputs"
    assert made["rows"] == rows_hash(train_ids, bulk_ids), f"{pair} {unit} {key}: stage 08 read other rows"


def inputs(pair, unit, key) -> list:
    """Stage 08's primary inputs for the configuration, and its primary predictions."""
    return S08["inputs"](pair, unit, key) + [stage08_predictions(pair, unit, key)]


KINDS = {"predictions": "predictions", "heads": "heads", "alignment_geometry": "geometry"}   # schema kind -> file suffix


def outputs(pair, unit, key) -> list:
    return [paths.output(STAGE, pair, f"{key}_{suffix}.parquet", unit) for suffix in KINDS.values()]


def geometry_rows(row, alignment_, measure, sample_ids, values) -> pd.DataFrame:
    return pd.DataFrame({**row, "alignment": alignment_, "readout": "none", "readout_seed": -1, "measure": measure,
                         "sample_id": sample_ids, "value": values})


def tables(pair, unit, key) -> dict:
    train = folds.training_samples(pair, *unit)
    train_ids = list(train["sample_id"])
    bulk = S08["scored"](pair, unit)
    bulk_ids = list(bulk["sample_id"])
    assert not set(train_ids) & set(bulk_ids), f"{pair} {unit}: a sample id on both sides"
    check_stage08(pair, unit, key, train_ids, bulk_ids)
    strata_sc, strata_bulk = S08["strata_of"](pair, train), S08["strata_of"](pair, bulk)
    strata = alignment.strata_stacked(strata_sc, strata_bulk)
    side = alignment.sides(len(train_ids), len(bulk_ids))
    y = train["group"].to_numpy()
    cv_folds = {r: S08["readout_folds"](pair, unit, train_ids, r) for r in range(S08["N_READOUT"])}
    keys = S08["key_columns"](pair, unit, key)
    seed = harmony_seed(pair, unit)
    coordinates_seed = A.GEOMETRY_ENCODER_SEED if key in S08["TRAINED"] else -1
    predictions, heads, geometry = [], [], []
    for e, X_train, X_bulk in features(pair, unit, key, train_ids, bulk_ids):
        row = {**keys, "encoder_seed": e}
        dims = S08["feature_names"](pair, unit, key, X_train)
        raw = np.vstack([X_train, X_bulk]).astype(np.float64)
        matrices = {"none": alignment.per_side_scaled(X_train, X_bulk, strata_sc, strata_bulk)}
        geometry.append(geometry_rows(row, "none", "silhouette_raw", ["all"], [alignment.silhouette(raw, side, strata)]))
        for a in A.ALIGNED:
            Z_train, Z_bulk = alignment.align(a, X_train, X_bulk, strata_sc, strata_bulk, seed)
            for name in ("ridge", "lasso"):
                for r in range(S08["N_READOUT"]):
                    out = readout.run(Z_train, y, cv_folds[r], Z_bulk, readout=name, strata_train=strata_sc, strata_bulk=strata_bulk,
                                      scaling="stacked")
                    fit = {**row, "alignment": a, "readout": name, "readout_seed": r}
                    predictions.append(S08["prediction_rows"](fit, "bulk", bulk_ids, bulk["scoring_unit_id"].to_numpy(), out["score"]))
                    stacked = out["scalers"]["stacked"]
                    heads += S08["head_rows"](fit, {**out, "scalers": {"single_cell": stacked, "bulk": stacked}}, dims)
            matrices[a] = alignment.stacked_scaled(Z_train, Z_bulk, strata_sc, strata_bulk)
            if a == "harmony":
                change = np.linalg.norm(np.vstack([Z_train, Z_bulk]) - raw) / np.linalg.norm(raw)
                geometry.append(geometry_rows(row, a, "harmony_relative_change", ["all"], [change]))
        for a, Z in matrices.items():
            geometry.append(geometry_rows(row, a, "silhouette", ["all"], [alignment.silhouette(Z, side, strata)]))
            if key in A.GEOMETRY_CONFIGURATIONS and e == coordinates_seed and unit[1] == 0:
                pcs = alignment.coordinates(Z)
                geometry += [geometry_rows(row, a, f"pc{i + 1}", train_ids + bulk_ids, pcs[:, i]) for i in range(2)]
    return {"predictions": predictions, "heads": heads, "alignment_geometry": geometry}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER, required=True)
    parser.add_argument("--set", help="one unit (e.g. crossfit_0)")
    parser.add_argument("--configuration", choices=configurations.CONFIGURATION_ORDER)
    args = parser.parse_args()
    determinism.configure()
    pair = args.pair
    for unit in units(pair):
        if args.set and folds.unit_name(*unit) != args.set:
            continue
        train_ids, bulk_ids = list(folds.training_samples(pair, *unit)["sample_id"]), list(S08["scored"](pair, unit)["sample_id"])
        for key in [args.configuration] if args.configuration else configurations.CONFIGURATION_ORDER:
            record = {"stage": STAGE, "pair": pair, "unit": unit, "configuration": key, "smoke": settings.SMOKE,
                      "rows": rows_hash(train_ids, bulk_ids),
                      "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair, codehash.extra(STAGE, pair)),
                      "inputs": manifest.input_hashes(inputs(pair, unit, key))}
            if manifest.is_done(outputs(pair, unit, key), record):
                print(f"{pair} {folds.unit_name(*unit)} {key}: done", flush=True)
                continue
            cost = manifest.begin()
            made = tables(pair, unit, key)
            for kind, path in zip(KINDS, outputs(pair, unit, key)):
                path.parent.mkdir(parents=True, exist_ok=True)
                schemas.write(kind, schemas.with_neutral_keys(pd.concat(made[kind], ignore_index=True)), manifest.temporary(path))
                manifest.finish(path, record, cost, seed=harmony_seed(pair, unit), inputs=inputs(pair, unit, key))
            print(f"{pair} {folds.unit_name(*unit)} {key}: aligned and read out", flush=True)


if __name__ == "__main__":
    main()
