"""Stage 04d's statistics (pipeline/04d_characteristics.py): the bias-corrected Cramer's V and the
lupus pools as connected components."""
import runpy

import numpy as np
import pandas as pd

from sc2bulk.config.settings import REPO

STAGE = runpy.run_path(str(REPO / "pipeline" / "04d_characteristics.py"), run_name="stage")


def test_cramers_v():
    v = STAGE["cramers_v"]
    group = pd.Series([0, 1] * 50)
    assert v(group.astype(str), group) > 0.95                          # batch = label
    rng = np.random.default_rng(0)
    assert v(pd.Series(rng.integers(0, 3, 2000)), pd.Series(rng.integers(0, 2, 2000))) < 0.05   # independent
    assert v(pd.Series(["a"] * 10), pd.Series([0, 1] * 5)) == 0.0      # one batch level


def test_pools_are_components_named_stably():
    pools = STAGE["pools"]
    cells = pd.DataFrame({"sample": ["a", "a", "b", "c", "d", "d"], "library": ["L1", "L2", "L2", "L3", "L3", "L3"]})
    samples = pd.DataFrame({"sample_id": ["d", "a", "b", "c"]})
    got = pools(cells, samples)
    assert list(got) == ["pool_1", "pool_0", "pool_0", "pool_1"]          # {a, b} via L2; {c, d} via L3
    shuffled = pools(cells.iloc[::-1].reset_index(drop=True), samples)
    assert list(shuffled) == list(got)                                    # row order doesn't rename pools
