"""Run-level leakage checks of stage 08 (amendments draft E2-E3).

The bulk-label permutation check that lived here (stage 08 rerun with the same blinding index as the smoke
run, so a determinism check only) is retired: tests/test_permutation.py covers every label-free stage.

- split67: stage 08 writes no Fig 2B or out-of-fold files for it, and its predictions are bulk rows only, so
  none is a split67 held-out donor's single-cell sample (their bulk is split67's scored set).

Needs the smoke outputs (pipeline/run_all.py --smoke through 08_readout). Run with
  SC2BULK_RUN_LEVEL=1 SC2BULK_OUTPUTS=<outputs> python -m pytest tests/run_level
"""
import os

import pandas as pd
import pytest

from sc2bulk.config import settings
from sc2bulk.config.pairs import PAIRS

PAIR = next(p for p, spec in PAIRS.items() if "split67" in spec["fold_sets"])
pytestmark = pytest.mark.skipif(os.environ.get("SC2BULK_RUN_LEVEL") != "1", reason="run-level check")


def smoke_outputs():
    return settings.OUTPUTS if settings.SMOKE else settings.OUTPUTS / "smoke"


def test_split67_reads_no_heldout_donor():
    smoke = smoke_outputs()
    out = smoke / "08_readout" / PAIR / "split67_0"
    assert not [p.name for p in out.iterdir() if "_within_" in p.name or "_out_of_fold_" in p.name or p.name.startswith("out_of_fold")]
    predictions = list(out.glob("*_predictions.parquet"))
    assert predictions
    # bulk rows only (split67 scores its held-out donors' bulk; Sound Life's bulk and single-cell samples share
    # ids, one per blood draw, so the check is on the side, not the sample id)
    for path in predictions:
        assert (pd.read_parquet(path, columns=["side"])["side"] == "bulk").all(), path.name
