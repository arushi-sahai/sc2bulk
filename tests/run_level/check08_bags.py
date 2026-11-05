"""Run-level bag-equivalence check of stage 08 (amendments draft F2), after its
pass. For each pair's first unit (and human blood, aging's crossfit_0 held-out donors):
- each sample's cell ids in the FM cells files equal the single-cell store's (stage 07
  embeds every stored cell under its cell id);
- the bags stage 08 draws from them equal sc2bulk.bags.test_bags called as harness.embed_samples calls
  it in stage 06 (the store's cell ids, FROZEN_NCELL, TEST_TIME_BAGS, pair, unit);
- re-embedding 3 training samples with stage 06's CytoSet row A seed 0 checkpoint through
  harness.embed_samples reproduces stage 06's stored rows (allclose).
Writes <outputs>/checks/check08_bags.json.

  python tests/run_level/check08_bags.py
"""
import json
import runpy

import anndata
import numpy as np
import torch

from sc2bulk import bags, determinism, folds, harness, manifest, paths, training_inputs
from sc2bulk.config import configurations, representations, settings
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

determinism.configure()
S = runpy.run_path(str(settings.REPO / "pipeline" / "08_readout.py"), run_name="check08")
TRAIN = runpy.run_path(str(settings.REPO / "pipeline" / "06_train.py"), run_name="check08")
FM_CELLS = [k for k in configurations.CONFIGURATION_ORDER if k not in S["TRAINED"] and configurations.CONFIGURATIONS[k]["input"] == "cells"]


def store(pair):
    """The single-cell store's cell ids and samples (backed: no counts loaded)."""
    return anndata.read_h5ad(folds.SAMPLES / pair / folds.CELL_FILES["all"][0], backed="r").obs


def drawn(cell_ids, cell_sample, sample_ids, pair, unit) -> dict:
    return bags.test_bags(cell_ids, cell_sample, sample_ids, T.FROZEN_NCELL, representations.TEST_TIME_BAGS, pair, unit)


result = {"git": manifest.git_state(), "checks": []}
for pair in PAIR_ORDER:
    unit = folds.units(pair)[0]
    checks = [(unit, list(folds.training_samples(pair, *unit)["sample_id"]))]
    if "crossfit" in PAIRS[pair]["fold_sets"]:
        checks.append((("crossfit", 0), list(S["heldout_samples"](pair, ("crossfit", 0))["sample_id"])))
    obs = store(pair)
    for checked, ids in checks:
        stored = obs[obs["sample_id"].astype(str).isin(set(ids))]
        stored_ids, stored_sample = stored.index.astype(str).to_numpy(), stored["sample_id"].astype(str).to_numpy()
        reference = drawn(stored_ids, stored_sample, ids, pair, checked)   # as harness.embed_samples draws them
        for key in FM_CELLS:
            cells = S["fm_cells"](pair, checked, configurations.CONFIGURATIONS[key]["model"], ids)
            fm_ids, fm_sample = cells.obs_names.astype(str).to_numpy(), cells.obs["sample_id"].astype(str).to_numpy()
            assert sorted(zip(fm_sample, fm_ids)) == sorted(zip(stored_sample, stored_ids)), f"{pair} {checked} {key}: cell ids differ from the store's"
            test = drawn(fm_ids, fm_sample, ids, pair, checked)
            assert all(np.array_equal(a, b) for s in ids for a, b in zip(test[s], reference[s], strict=True)), f"{pair} {checked} {key}: bags differ"
            result["checks"].append({"pair": pair, "unit": folds.unit_name(*checked), "configuration": key, "samples": len(ids)})
    # one trained seed re-embedded through the harness
    checkpoint = torch.load(paths.output("06_train", pair, "A", unit) / "cytoset" / "checkpoints" / "s0.pt", map_location="cpu",
                            weights_only=False)
    config = checkpoint["config"]
    cells = training_inputs.unit_cells(pair, unit)
    train, _ = training_inputs.split_cells(cells, training_inputs.encoder_sides(pair, unit, 0))
    inputs = harness.Inputs(config, train, training_inputs.pseudobulk_log(pair, unit))
    net = TRAIN["network"]("cytoset", config, harness.input_width(config, len(cells.genes)))
    net.load_state_dict(checkpoint["state"])
    train_ids = list(folds.training_samples(pair, *unit)["sample_id"])
    three = train_ids[:3]
    embedded = harness.embed_samples(net, training_inputs.subset(cells, three), inputs, three, config["ncell"], pair, unit)
    X_train, _ = S["trained_features"](pair, unit, "cytoset_cells", 0, train_ids, list(S["scored"](pair, unit)["sample_id"]))
    difference = float(np.abs(embedded - X_train[:3]).max())
    assert np.allclose(embedded, X_train[:3], rtol=1e-4, atol=1e-5), f"{pair} {unit}: re-embedded rows differ from stage 06's ({difference})"
    result["checks"].append({"pair": pair, "unit": folds.unit_name(*unit), "configuration": "cytoset_cells", "seed": 0,
                             "max_abs_difference": difference})
result["passed"] = True
out = settings.OUTPUTS / "checks" / "check08_bags.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=1))
print(json.dumps({"passed": True, "checks": len(result["checks"])}), flush=True)
