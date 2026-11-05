"""Stage 03: folds, train/validation splits and read-out CV folds, shared by every configuration.

Folds (sc2bulk.folds): which single-cell samples train and which bulk samples are scored, per
unit (fold set, fold). Pairs with no donor on both sides have one unit, "all"; the pair whose
donors appear on both sides is cross-fitted by donor (4 folds, stratified by the single-cell
0/1 group) plus the 67/29 sensitivity split. Everything below is made per unit, from the
unit's training samples only (folds.training_samples).

Splits (encoder training, early stopping and the configuration search; the held-out samples
of Fig 2B's within-single-cell scores): per split, 30% of the holdout groups within each
0/1 group go to validation, the rest to training. A holdout group (mouse microglia: the
sample, the P5 Percoll pairs together; human blood, aging: the donor; mouse organs: the
mouse; motor cortex: the donor) never has mixed labels, so the split is drawn at the group
level (StratifiedShuffleSplit on one row per group) and expanded to its samples.

Read-out CV folds (the read-out's hyperparameter search): READOUT_FOLDS folds per read-out
seed, grouped by holdout group and stratified by group: each label's holdout groups are
shuffled with the seed and dealt to the folds in turn. Made for two contexts: all of the
unit's training samples (the transfer read-out) and each encoder split's training side
(Fig 2B).

Every seed comes from sc2bulk.seeds.derived, keyed by the unit. Outputs, in
<outputs>/03_splits/<pair>/:
  donor_folds.csv    pair, fold_set, fold, donor_id, side ("train" / "scored")
  splits.csv         pair, fold_set, fold, purpose, index, split_seed, sample_id, holdout_group,
                     group, side
  split_summary.csv  per unit and split: validation samples, their fraction, holdout groups
                     per label
  readout_folds.csv  pair, fold_set, fold, context ("training" or "encoder_split_<i>"),
                     readout_index, fold_seed, sample_id, readout_fold

  python pipeline/03_splits.py [--pair PAIR]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from sc2bulk import determinism, folds, manifest, seeds
from sc2bulk.config import settings, smoke
from sc2bulk.config.pairs import PAIR_ORDER

OUT = settings.OUTPUTS / "03_splits"
OUTPUTS = ("donor_folds.csv", "splits.csv", "split_summary.csv", "readout_folds.csv")
N_ENCODER = smoke.N_ENCODER_SEEDS if settings.SMOKE else settings.N_ENCODER_SEEDS
N_SEARCH = smoke.N_SEARCH_SEEDS if settings.SMOKE else settings.N_SEARCH_SEEDS
N_READOUT = smoke.N_READOUT_SEEDS if settings.SMOKE else settings.N_READOUT_SEEDS


def record_for(pair) -> dict:
    return {"stage": "03_splits", "pair": pair, "smoke": settings.SMOKE,
            "code": manifest.stage_code_hash(settings.REPO / "pipeline" / "03_splits.py", pair),
            "inputs": manifest.input_hashes(folds.inputs(pair)),
            "validation_fraction": settings.VALIDATION_FRACTION, "readout_folds": settings.READOUT_FOLDS,
            "n": [N_ENCODER, N_SEARCH, N_READOUT]}


def holdout_groups(samples: pd.DataFrame) -> pd.DataFrame:
    """One row per holdout group with its 0/1 group (asserted unique within the group)."""
    labels = samples.groupby("holdout_group")["group"].agg(["nunique", "first"])
    assert (labels["nunique"] == 1).all(), "a holdout group has samples of both groups"
    return labels["first"].rename("group").reset_index()


def split(samples: pd.DataFrame, seed: int) -> pd.Series:
    """'train' / 'validation' per sample: VALIDATION_FRACTION of the holdout groups within each
    0/1 group to validation, drawn with the seed."""
    groups = holdout_groups(samples)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=settings.VALIDATION_FRACTION, random_state=seed)
    _, validation = next(splitter.split(groups, groups["group"]))
    side = samples["holdout_group"].isin(groups["holdout_group"].iloc[validation])
    held_out = samples[side]
    for g in (0, 1):
        assert held_out.loc[held_out["group"] == g, "holdout_group"].nunique() >= 2, f"validation has < 2 groups of label {g}"
        assert (samples.loc[~side, "group"] == g).any(), f"training side has no samples of label {g}"
    return pd.Series(np.where(side, "validation", "train"), index=samples.index)


def readout_folds(samples: pd.DataFrame, seed: int) -> pd.Series:
    """Read-out fold per sample: each label's holdout groups shuffled with the seed and dealt
    to the READOUT_FOLDS folds in turn, starting at a seeded fold (both labels continue the
    same deal, so each label's group count differs by at most one between folds)."""
    groups = holdout_groups(samples).set_index("holdout_group")["group"]
    fold = samples["holdout_group"].map(folds.deal(groups, settings.READOUT_FOLDS, seed))
    assert fold.nunique() == settings.READOUT_FOLDS, "a read-out fold is empty"
    return fold


def write_csv(frame, path, record, unit):
    frame.to_csv(manifest.temporary(path), index=False)
    manifest.finish(path, record, unit, inputs=folds.inputs(path.parent.name))


def main():
    determinism.configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER)
    args = parser.parse_args()
    for pair in [args.pair] if args.pair else PAIR_ORDER:
        out = OUT / pair
        record = record_for(pair)
        if manifest.is_done([out / name for name in OUTPUTS], record):
            print(f"{pair}: done", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        unit = manifest.begin()
        write_csv(folds.donor_folds(pair, folds.single_cell_samples(pair)), out / "donor_folds.csv", record, unit)

        rows, summaries, fold_rows = [], [], []
        for fold_set, fold in folds.units(pair):
            key = {"pair": pair, "fold_set": fold_set, "fold": fold}
            samples = folds.training_samples(pair, fold_set, fold)[["sample_id", "holdout_group", "group"]]
            unit_rows = []
            for purpose, n in (("encoder_split", N_ENCODER), ("search_split", N_SEARCH)):
                for index in range(n):
                    seed = seeds.derived(purpose, pair, index, fold_set, fold)
                    unit_rows.append(samples.assign(**key, purpose=purpose, index=index, split_seed=seed,
                                                    side=split(samples, seed)))
            unit_splits = pd.concat(unit_rows, ignore_index=True)
            held_out = unit_splits[unit_splits["side"] == "validation"]
            summary = held_out.groupby(["purpose", "index"]).agg(
                validation_samples=("sample_id", "size"),
                groups_label_0=("holdout_group", lambda g: g[held_out.loc[g.index, "group"] == 0].nunique()),
                groups_label_1=("holdout_group", lambda g: g[held_out.loc[g.index, "group"] == 1].nunique())).reset_index()
            summary.insert(0, "fold", fold)
            summary.insert(0, "fold_set", fold_set)
            summary.insert(0, "pair", pair)
            summary["validation_fraction"] = summary["validation_samples"] / len(samples)
            for r in range(N_READOUT):
                seed = seeds.derived("readout_folds", pair, r, fold_set, fold)
                fold_rows.append(samples[["sample_id"]].assign(**key, context="training", readout_index=r, fold_seed=seed,
                                                               readout_fold=readout_folds(samples, seed)))
                for index in range(N_ENCODER):
                    train = unit_splits[(unit_splits["purpose"] == "encoder_split") & (unit_splits["index"] == index)
                                        & (unit_splits["side"] == "train")]
                    train = train[["sample_id", "holdout_group", "group"]].reset_index(drop=True)
                    fold_rows.append(train[["sample_id"]].assign(**key, context=f"encoder_split_{index}", readout_index=r,
                                                                 fold_seed=seed, readout_fold=readout_folds(train, seed)))
            rows.append(unit_splits)
            summaries.append(summary)
            print(f"{pair} {fold_set} {fold}: {len(samples)} training samples, {samples['holdout_group'].nunique()} "
                  f"holdout groups; validation {summary['validation_samples'].min()}-{summary['validation_samples'].max()} "
                  f"samples per split", flush=True)

        write_csv(pd.concat(rows, ignore_index=True), out / "splits.csv", record, unit)
        write_csv(pd.concat(summaries, ignore_index=True), out / "split_summary.csv", record, unit)
        write_csv(pd.concat(fold_rows, ignore_index=True), out / "readout_folds.csv", record, unit)


if __name__ == "__main__":
    main()
