"""Stage 04c's cell selection (pipeline/04c_qc_embed.py): each sample's first n stored cells,
n = total // samples, as a cap."""
import runpy

import pandas as pd

from sc2bulk.config.settings import REPO

STAGE = runpy.run_path(str(REPO / "pipeline" / "04c_qc_embed.py"), run_name="stage")


def test_first_cells_per_sample_with_a_cap():
    obs = pd.DataFrame({"sample_id": ["a"] * 5 + ["b"] * 2 + ["c"] * 4})
    keep = STAGE["first_cells"](obs, 9)                        # n = 3
    assert list(keep) == [True] * 3 + [False] * 2 + [True] * 2 + [True] * 3 + [False]
