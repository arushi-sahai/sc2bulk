"""Stage 04b's helpers (pipeline/04b_deconvolve.py; sc2bulk.deconvolution): merges follow chains,
and a merge adds counts and cells."""
import runpy

import numpy as np
import pandas as pd

from sc2bulk import deconvolution as dc
from sc2bulk.config.settings import REPO

STAGE = runpy.run_path(str(REPO / "pipeline" / "04b_deconvolve.py"), run_name="stage")


def test_merged_follows_chains():
    labels = pd.Series(["a", "b", "c", "d"])
    assert list(STAGE["merged"](labels, {"a": "b", "b": "c"})) == ["c", "c", "c", "d"]


def test_merge_is_additive():
    prof = {"samples": np.array(["s"]), "types": np.array(["a", "b"]), "depth": np.array([1.0]),
            "summed": np.array([[[1.0, 2.0], [3.0, 4.0]]]), "n": np.array([[2, 3]])}
    out = dc.merge(prof, "a", "b")
    assert list(out["types"]) == ["b"] and out["n"].tolist() == [[5]] and out["summed"].tolist() == [[[4.0, 6.0]]]
