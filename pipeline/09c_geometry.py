"""Stage 09c: the Fig 3A / S13A-E coordinates (stage 09's §5 coordinates).

Per pair, at fold 0 of its primary fold set, and per configuration in config/geometry.CONFIGURATIONS: stage 08's
features at encoder seed config/alignment.GEOMETRY_ENCODER_SEED (trained configurations) or -1 (fixed ones), one row per
training sample of the unit and per scored bulk sample; the matrix stage 09 reads out under each alignment ("none"
per-side scaled, every alignment in ALIGNED stacked-scaled after it, Harmony at stage 09's seed for the unit); pc1 and
pc2 of every sample under all five (one PCA over all rows). Stage 09's and stage 08's functions are loaded with runpy.
Silhouettes stay in stage 09's geometry files. Sides are told apart by sample id (disjoint, asserted). No label is read.

Outputs, in <outputs>/09c_geometry/<pair>/<unit>/:
  <configuration>_geometry.parquet      alignment_geometry kind: pc1 and pc2 per alignment and sample

  python pipeline/09c_geometry.py --pair PAIR [--configuration KEY]
"""
import argparse
import runpy

import pandas as pd

from sc2bulk import alignment, codehash, determinism, folds, manifest, paths, schemas
from sc2bulk.config import alignment as A
from sc2bulk.config import geometry, settings
from sc2bulk.config.pairs import PAIR_ORDER

STAGE = "09c_geometry"
S09 = runpy.run_path(str(settings.REPO / "pipeline" / "09_alignment.py"), run_name="stage09")   # its functions, not its main
S08 = S09["S08"]


def units(pair) -> list:
    return [unit for unit in S09["units"](pair) if unit[1] == 0]


def encoder_seed(key) -> int:
    return A.GEOMETRY_ENCODER_SEED if key in S08["TRAINED"] else -1


def features(pair, unit, key, train_ids, bulk_ids) -> tuple:
    """Stage 08's (training rows, bulk rows) at the configuration's encoder seed."""
    if key in S08["TRAINED"]:
        return S08["trained_features"](pair, unit, key, encoder_seed(key), train_ids, bulk_ids)
    (e, X_train, X_bulk), = S08["features"](pair, unit, key, train_ids, bulk_ids)
    assert e == -1, e
    return X_train, X_bulk


def inputs(pair, unit, key) -> list:
    """Stage 08's primary inputs for the configuration, less the embeddings of the encoder seeds not read."""
    unread = {S08["trained_file"](pair, unit, key, e) for e in S08["encoder_seeds"](pair, key) if e != encoder_seed(key)} \
        if key in S08["TRAINED"] else set()
    return [path for path in S08["inputs"](pair, unit, key) if path not in unread]


def output(pair, unit, key):
    return paths.output(STAGE, pair, f"{key}_geometry.parquet", unit)


def table(pair, unit, key) -> pd.DataFrame:
    train = folds.training_samples(pair, *unit)
    train_ids = list(train["sample_id"])
    bulk = S08["scored"](pair, unit)
    bulk_ids = list(bulk["sample_id"])
    assert not set(train_ids) & set(bulk_ids), f"{pair} {unit}: a sample id on both sides"
    strata_sc, strata_bulk = S08["strata_of"](pair, train), S08["strata_of"](pair, bulk)
    X_train, X_bulk = features(pair, unit, key, train_ids, bulk_ids)
    matrices = {"none": alignment.per_side_scaled(X_train, X_bulk, strata_sc, strata_bulk)}
    for a in A.ALIGNED:
        Z_train, Z_bulk = alignment.align(a, X_train, X_bulk, strata_sc, strata_bulk, S09["harmony_seed"](pair, unit))
        matrices[a] = alignment.stacked_scaled(Z_train, Z_bulk, strata_sc, strata_bulk)
    row = {**S08["key_columns"](pair, unit, key), "encoder_seed": encoder_seed(key)}
    frames = []
    for a, Z in matrices.items():
        pcs = alignment.coordinates(Z)
        frames += [S09["geometry_rows"](row, a, f"pc{i + 1}", train_ids + bulk_ids, pcs[:, i]) for i in range(2)]
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER, required=True)
    parser.add_argument("--configuration", choices=geometry.CONFIGURATIONS)
    args = parser.parse_args()
    determinism.configure()
    pair = args.pair
    for unit in units(pair):
        train_ids, bulk_ids = list(folds.training_samples(pair, *unit)["sample_id"]), list(S08["scored"](pair, unit)["sample_id"])
        for key in [args.configuration] if args.configuration else geometry.CONFIGURATIONS:
            record = {"stage": STAGE, "pair": pair, "unit": unit, "configuration": key, "smoke": settings.SMOKE,
                      "rows": S09["rows_hash"](train_ids, bulk_ids),
                      "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair, codehash.extra(STAGE, pair)),
                      "inputs": manifest.input_hashes(inputs(pair, unit, key))}
            path = output(pair, unit, key)
            if manifest.is_done([path], record):
                print(f"{pair} {folds.unit_name(*unit)} {key}: done", flush=True)
                continue
            cost = manifest.begin()
            path.parent.mkdir(parents=True, exist_ok=True)
            schemas.write("alignment_geometry", schemas.with_neutral_keys(table(pair, unit, key)), manifest.temporary(path))
            manifest.finish(path, record, cost, seed=S09["harmony_seed"](pair, unit), inputs=inputs(pair, unit, key))
            print(f"{pair} {folds.unit_name(*unit)} {key}: coordinates", flush=True)


if __name__ == "__main__":
    main()
