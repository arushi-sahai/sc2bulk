"""Run-level check of stage 12a, after its pass over every pair: Fig 6A's
phenotype column equals the AUROC of stage 08's Fig 2B (within) ridge predictions, for every configuration, encoder seed
and read-out seed (scoring-unit means, stage 11's rule; single-cell labels only). Writes
<outputs>/checks/check12a_phenotype.json.

  python tests/run_level/check12a_phenotype.py
"""
import json

import numpy as np
import pandas as pd

from sc2bulk import folds, manifest, paths, properties, schemas
from sc2bulk.config import configurations, settings, triage
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

result = {"git": manifest.git_state(), "units": []}
for pair in PAIR_ORDER:
    labels = folds.single_cell_samples(pair).set_index("sample_id")["group"]
    for unit in [u for u in folds.units(pair) if u[0] == next(iter(PAIRS[pair]["fold_sets"]))]:
        worst = 0.0
        for key in [k for k in configurations.CONFIGURATION_ORDER if k not in triage.CUT_CONFIGURATIONS]:
            ours = schemas.read("property_r2", paths.output("12a_properties", pair, f"{key}_property_r2.parquet", unit))
            ours = ours[ours["property"] == "phenotype"].set_index(["encoder_seed", "readout_seed"])["cv_r2"]
            within = schemas.read("predictions", paths.output("08_readout", pair, f"{key}_within_predictions.parquet", unit))
            within = within[within["readout"] == "ridge"]
            fig2b = pd.Series({(int(e), int(r)): properties.unit_auroc(fit["score"], labels.loc[fit["sample_id"]], fit["scoring_unit_id"])
                               for (e, r), fit in within.groupby(["encoder_seed", "readout_seed"])})
            fig2b = fig2b.loc[ours.index]
            worst = max(worst, float(np.abs(ours.to_numpy() - fig2b.to_numpy()).max()))
        assert worst == 0.0, f"{pair} {unit}: Fig 6A's phenotype column differs from Fig 2B by up to {worst}"
        result["units"].append({"pair": pair, "unit": folds.unit_name(*unit), "max_difference": worst})
out = settings.OUTPUTS / "checks" / "check12a_phenotype.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=1))
print(f"Fig 6A's phenotype column equals Fig 2B's on {len(result['units'])} units: {out}")
