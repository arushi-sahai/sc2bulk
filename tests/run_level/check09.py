"""Run-level checks of stage 09, after its pass
over every pair. Per pair, primary unit and configuration:
- reproduction: for pseudobulk, readout.run with per-side scaling on stage 09's features (stage 08's functions
  as stage 09 loads them) reproduces stage 08's primary bulk scores exactly, every read-out and read-out seed:
  the strongest check that stage 09 reads stage 08's rows and features;
- stage 09's predictions score the same (sample_id, scoring_unit_id) sequence as stage 08's primary file,
  carry only the aligned values and encoder seeds < N_ENCODER_SEEDS (stage 08's seeds 0-9 exactly);
- every fit has heads, the stacked moments under both sides;
- its geometry carries all five alignment values; pc1/pc2 rows, where written, cover the unit's training and
  scored bulk ids exactly, which are disjoint.
Writes <outputs>/checks/check09.json.

  python tests/run_level/check09.py
"""
import json
import runpy

import numpy as np
import pandas as pd

from sc2bulk import determinism, folds, manifest
from sc2bulk.config import alignment as A
from sc2bulk.config import configurations, settings
from sc2bulk.config.pairs import PAIR_ORDER

D = runpy.run_path(str(settings.REPO / "pipeline" / "09_alignment.py"), run_name="check09")
S08 = D["S08"]
determinism.configure()   # stage 08's thread pools, so the reproduction can be exact


def sequence(table) -> list:
    return list(zip(table["sample_id"], table["scoring_unit_id"]))


result = {"git": manifest.git_state(), "units": []}
for pair in PAIR_ORDER:
    for unit in D["units"](pair):
        train = folds.training_samples(pair, *unit)
        train_ids, bulk_ids = list(train["sample_id"]), list(S08["scored"](pair, unit)["sample_id"])
        assert not set(train_ids) & set(bulk_ids), f"{pair} {unit}: a sample id on both sides"
        for key in configurations.CONFIGURATION_ORDER:
            mine, theirs = D["outputs"](pair, unit, key)[0], D["stage08_predictions"](pair, unit, key)
            predictions, primary = pd.read_parquet(mine), pd.read_parquet(theirs)
            geometry = pd.read_parquet(D["outputs"](pair, unit, key)[2])
            heads = pd.read_parquet(D["outputs"](pair, unit, key)[1])
            assert len(heads.drop_duplicates(["alignment", "encoder_seed", "readout", "readout_seed"])) == \
                len(predictions.drop_duplicates(["alignment", "encoder_seed", "readout", "readout_seed"])), f"{pair} {unit} {key}: a fit without heads"
            assert set(heads["side"]) == {"single_cell", "bulk"}, f"{pair} {unit} {key}: heads' scaler sides"
            single_cell, bulk = (heads[heads["side"] == s].drop(columns="side").reset_index(drop=True) for s in ("single_cell", "bulk"))
            assert single_cell.equals(bulk), f"{pair} {unit} {key}: the two sides' heads differ (one stacked scaler)"
            assert set(predictions["alignment"]) == set(A.ALIGNED), f"{mine}: alignment values"
            assert set(geometry["alignment"]) == {"none", *A.ALIGNED}, f"{pair} {unit} {key}: geometry alignment values"
            seeds_09 = set(predictions["encoder_seed"])
            assert seeds_09 == set(primary["encoder_seed"]) & set(range(-1, settings.N_ENCODER_SEEDS)), f"{mine}: encoder seeds"
            reference = sequence(primary[(primary["encoder_seed"] == min(seeds_09)) & (primary["readout"] == "ridge") & (primary["readout_seed"] == 0)])
            for _, fit in predictions.groupby(["alignment", "encoder_seed", "readout", "readout_seed"]):
                assert sequence(fit) == reference, f"{mine}: a fit scores other rows than stage 08"
            pcs = geometry[geometry["measure"].isin(["pc1", "pc2"])]
            for _, group in pcs.groupby(["alignment", "measure"]):
                assert sorted(group["sample_id"]) == sorted(train_ids + bulk_ids), f"{pair} {unit} {key}: coordinates' samples"
            if key == "pseudobulk":
                (_, X_train, X_bulk), = D["features"](pair, unit, key, train_ids, bulk_ids)
                cv_folds = {r: S08["readout_folds"](pair, unit, train_ids, r) for r in range(S08["N_READOUT"])}
                strata_train, strata_bulk = S08["strata_of"](pair, train), S08["strata_of"](pair, S08["scored"](pair, unit))
                for name, r, out in S08["fits"](X_train, train["group"].to_numpy(), cv_folds, X_bulk, strata_train, strata_bulk):
                    fit = primary[(primary["readout"] == name) & (primary["readout_seed"] == r)]
                    assert np.array_equal(fit["score"].to_numpy(), out["score"]), f"{pair} {unit} {name} r{r}: stage 08 not reproduced"
            result["units"].append({"pair": pair, "unit": folds.unit_name(*unit), "configuration": key,
                                    "encoder_seeds": sorted(int(e) for e in seeds_09), "coordinates": not pcs.empty})
result["passed"] = True
path = settings.OUTPUTS / "checks" / "check09.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, indent=1))
print(json.dumps({"passed": True, "units": len(result["units"])}), flush=True)
