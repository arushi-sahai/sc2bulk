"""Run-level checks of stage 09c, after its pass over every pair. Per pair, at fold 0 of
its primary fold set, and per configuration in config/geometry.CONFIGURATIONS:
- pc1/pc2 rows for all five alignment values, one per sample of the unit's training and scored bulk ids (disjoint);
- every input 09c records has the content hash stage 09's geometry file recorded for it (the figures print stage 09's
  silhouette under 09c's scatter);
- for pseudobulk and cytoset_cells (stage 09's config/alignment.GEOMETRY_CONFIGURATIONS): the same samples, alignments
  and encoder seed as stage 09's rows, each component with the same sign, and values within TOLERANCE when the thread
  count or environment differs from stage 09's record, exactly when both match (the largest difference reported).
Writes <outputs>/checks/check09c.json.

  python tests/run_level/check09c.py
"""
import json
import runpy

import pandas as pd

from sc2bulk import folds, manifest
from sc2bulk.config import alignment as A
from sc2bulk.config import geometry, settings
from sc2bulk.config.pairs import PAIR_ORDER

C = runpy.run_path(str(settings.REPO / "pipeline" / "09c_geometry.py"), run_name="check09c")
S08, S09 = C["S08"], C["S09"]
INDEX = ["alignment", "measure", "encoder_seed", "sample_id"]
TOLERANCE = 1e-6                       # absolute, on the PC scores of standardized rows (BLAS reductions over other thread counts)

result = {"git": manifest.git_state(), "units": []}
for pair in PAIR_ORDER:
    for unit in C["units"](pair):
        ids = sorted([*folds.training_samples(pair, *unit)["sample_id"], *S08["scored"](pair, unit)["sample_id"]])
        for key in geometry.CONFIGURATIONS:
            mine = pd.read_parquet(C["output"](pair, unit, key))
            assert set(mine["measure"]) == {"pc1", "pc2"}, f"{pair} {unit} {key}: measures"
            assert set(mine["alignment"]) == {"none", *A.ALIGNED}, f"{pair} {unit} {key}: alignment values"
            assert set(mine["encoder_seed"]) == {C["encoder_seed"](key)}, f"{pair} {unit} {key}: encoder seed"
            for _, group in mine.groupby(["alignment", "measure"]):
                assert sorted(group["sample_id"]) == ids, f"{pair} {unit} {key}: coordinates' samples"
            entry = {"pair": pair, "unit": folds.unit_name(*unit), "configuration": key, "rows": len(mine)}
            record, stage09 = manifest.read(C["output"](pair, unit, key)), manifest.read(S09["outputs"](pair, unit, key)[2])
            assert record["settings"]["rows"] == stage09["settings"]["rows"], f"{pair} {unit} {key}: other rows than stage 09's"
            assert record["seed"] == stage09["seed"], f"{pair} {unit} {key}: another Harmony seed than stage 09's"
            for name, content in record["settings"]["inputs"].items():
                assert stage09["settings"]["inputs"][name] == content, f"{pair} {unit} {key}: {name} differs from stage 09's"
            if key in A.GEOMETRY_CONFIGURATIONS:
                theirs = pd.read_parquet(S09["outputs"](pair, unit, key)[2])
                theirs = theirs[theirs["measure"].isin(["pc1", "pc2"])]
                a, b = mine.set_index(INDEX)["value"].sort_index(), theirs.set_index(INDEX)["value"].sort_index()
                assert a.index.equals(b.index), f"{pair} {unit} {key}: other rows than stage 09's"
                for component, values in a.groupby(level=["alignment", "measure"]):
                    assert (values.to_numpy() @ b.loc[component].to_numpy()) > 0, f"{pair} {unit} {key} {component}: sign flipped"
                same = record["n_threads"] == stage09["n_threads"] and record["environment"] == stage09["environment"]
                difference = float((a - b).abs().max())
                assert difference == 0 if same else difference <= TOLERANCE, f"{pair} {unit} {key}: {difference} from stage 09"
                entry.update(max_abs_diff_vs_stage09=difference, same_threads_and_environment=same)
            result["units"].append(entry)
result["max_abs_diff_vs_stage09"] = max(u["max_abs_diff_vs_stage09"] for u in result["units"] if "max_abs_diff_vs_stage09" in u)
result["passed"] = True
path = settings.OUTPUTS / "checks" / "check09c.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, indent=1))
print(json.dumps({"passed": True, "units": len(result["units"]), "max_abs_diff_vs_stage09": result["max_abs_diff_vs_stage09"]}), flush=True)
