"""Run-level same-rows check of stage 08 (amendments draft F1), after its pass
over every pair. Per pair and unit, for every read-out family:
- primary, and human blood, aging's Fig 2B held-out rows: every fit of every configuration scores the same
  sample_id and scoring_unit_id sequence;
- Fig 2B validation sides and the out-of-fold row: per encoder split e, the same sequence for every
  configuration, and the same splits e (the guard drops a split for all of them).
Also checks that readout.readout_folds, stage 08's copy of stage 03's dealing, reproduces stage 03's
encoder_split_<e> read-out folds. Writes <outputs>/checks/check08_rows.json.

  python tests/run_level/check08_rows.py
"""
import json

import pandas as pd

from sc2bulk import folds, manifest, paths, readout, seeds, training_inputs
from sc2bulk.config import configurations, settings, smoke
from sc2bulk.config.pairs import PAIR_ORDER

FITS = ["encoder_seed", "readout", "readout_seed"]
N_ENCODER = smoke.N_ENCODER_SEEDS if settings.SMOKE else settings.N_ENCODER_SEEDS
N_READOUT = smoke.N_READOUT_SEEDS if settings.SMOKE else settings.N_READOUT_SEEDS


def sequences(path) -> dict:
    """encoder seed -> the (side, sample_id, scoring_unit_id) sequence its fits score (asserted equal within the seed)."""
    table = pd.read_parquet(path, columns=[*FITS, "side", "sample_id", "scoring_unit_id"])
    out = {}
    for (e, _, _), fit in table.groupby(FITS, sort=True):
        rows = tuple(zip(fit["side"], fit["sample_id"], fit["scoring_unit_id"]))
        assert out.setdefault(int(e), rows) == rows, f"{path}: fits of encoder seed {e} score different rows"
    return out


result = {"git": manifest.git_state(), "units": []}
for pair in PAIR_ORDER:
    for unit in folds.units(pair):
        for family in ("", "_within", "_out_of_fold"):
            files = [paths.output("08_readout", pair, f"{key}{family}_predictions.parquet", unit) for key in configurations.CONFIGURATION_ORDER]
            if not files[0].exists():
                assert not any(f.exists() for f in files), f"{pair} {unit} {family}: only some configurations were read out"
                continue
            per_key = [sequences(f) for f in files]
            if family == "_out_of_fold" or (family == "_within" and unit[0] != "crossfit"):   # rows depend on split e
                assert all(s == per_key[0] for s in per_key), f"{pair} {unit} {family}: configurations differ in splits or rows"
            else:
                assert len({rows for s in per_key for rows in s.values()}) == 1, f"{pair} {unit} {family}: configurations score different rows"
            result["units"].append({"pair": pair, "unit": folds.unit_name(*unit), "family": family or "_primary",
                                    "splits": sorted(per_key[0])})
        # the copied dealing reproduces stage 03's encoder-split read-out folds
        train = folds.training_samples(pair, *unit)
        table = pd.read_csv(settings.OUTPUTS / "03_splits" / pair / "readout_folds.csv", dtype={"sample_id": str})
        for e in range(N_ENCODER):
            side = train["sample_id"].map(training_inputs.encoder_sides(pair, unit, e))
            for r in range(N_READOUT):
                dealt = readout.readout_folds(train[side == "train"], seeds.derived("readout_folds", pair, r, *unit))
                rows = table[(table["fold_set"] == unit[0]) & (table["fold"] == unit[1]) & (table["context"] == f"encoder_split_{e}")
                             & (table["readout_index"] == r)]
                assert dict(zip(train.loc[side == "train", "sample_id"], dealt)) == dict(zip(rows["sample_id"], rows["readout_fold"])), \
                    f"{pair} {unit} e{e} r{r}: the copied dealing differs from stage 03's"
result["passed"] = True
out = settings.OUTPUTS / "checks" / "check08_rows.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=1))
print(json.dumps({"passed": True, "families": len(result["units"])}), flush=True)
