"""Stage 03b: encoder splits for the attribution seeds 10-19 (encoder seeds 0-19, each with
its own split). Stage 03 (frozen) wrote encoder splits 0-9; this
stage draws 10-19 with stage 03's own split function and seed purpose (encoder_split), loaded from
pipeline/03_splits.py, and asserts that it reproduces stage 03's rows for 0-9 exactly.

Outputs, in <outputs>/03b_attribution_splits/<pair>/:
  splits.csv   pair, fold_set, fold, purpose ("encoder_split"), index (10-19), split_seed, sample_id,
               holdout_group, group, side

  python pipeline/03b_attribution_splits.py [--pair PAIR]
"""
import argparse
import runpy

import pandas as pd

from sc2bulk import codehash, determinism, folds, manifest, paths, seeds
from sc2bulk.config import settings
from sc2bulk.config.pairs import PAIR_ORDER

STAGE = "03b_attribution_splits"
STAGE03 = runpy.run_path(str(settings.REPO / "pipeline" / "03_splits.py"), run_name="stage03")   # its split(), not its main
INDICES = range(10, 20)


def record_for(pair) -> dict:
    return {"stage": STAGE, "pair": pair, "smoke": settings.SMOKE, "indices": list(INDICES),
            "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair, codehash.extra(STAGE, pair)),
            "inputs": manifest.input_hashes(folds.inputs(pair) + [settings.OUTPUTS / "03_splits" / pair / "splits.csv"])}


def draw(pair, unit, samples, index):
    seed = seeds.derived("encoder_split", pair, index, *unit)
    return samples.assign(pair=pair, fold_set=unit[0], fold=unit[1], purpose="encoder_split", index=index,
                          split_seed=seed, side=STAGE03["split"](samples, seed))


def main():
    determinism.configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER)
    args = parser.parse_args()
    for pair in [args.pair] if args.pair else PAIR_ORDER:
        record = record_for(pair)
        path = paths.output(STAGE, pair, "splits.csv")
        if manifest.is_done([path], record):
            print(f"{pair}: done", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        unit_cost = manifest.begin()
        old = pd.read_csv(settings.OUTPUTS / "03_splits" / pair / "splits.csv", dtype={"sample_id": str, "holdout_group": str})
        rows = []
        for unit in folds.units(pair):
            samples = folds.training_samples(pair, *unit)[["sample_id", "holdout_group", "group"]]
            for index in range(STAGE03["N_ENCODER"]):   # stage 03's rows reproduced exactly (the smoke run: its one split)
                mine = draw(pair, unit, samples, index)
                theirs = old[(old["fold_set"] == unit[0]) & (old["fold"] == unit[1]) & (old["purpose"] == "encoder_split")
                             & (old["index"] == index)]
                assert list(mine["side"]) == list(theirs["side"]) and list(mine["sample_id"]) == list(theirs["sample_id"]), \
                    f"{pair} {unit} split {index} differs from stage 03"
            rows += [draw(pair, unit, samples, index) for index in INDICES]
        columns = ["pair", "fold_set", "fold", "purpose", "index", "split_seed", "sample_id", "holdout_group", "group", "side"]
        pd.concat(rows, ignore_index=True)[columns].to_csv(manifest.temporary(path), index=False)
        manifest.finish(path, record, unit_cost, inputs=folds.inputs(pair))
        print(f"{pair}: encoder splits 10-19 for {len(folds.units(pair))} units (0-9 reproduce stage 03)", flush=True)


if __name__ == "__main__":
    main()
