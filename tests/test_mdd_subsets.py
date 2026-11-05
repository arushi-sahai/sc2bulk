"""scripts/mdd_subsets.py: reproduces the frozen mdd_table.py's K = all rows,
enumerates every subset, and gives the joint lower bounds."""
import io
import itertools
import subprocess
import sys

import numpy as np
import pandas as pd

from sc2bulk.config import settings

sys.path.insert(0, str(settings.REPO / "scripts"))
import mdd_subsets  # noqa: E402

PAIRS = ["tms", "soundlife", "pineda", "lupus", "crohns"]
COLUMNS = ["rho", "auroc", "mdd", "mdd_holm_first_step", "p_within_margin_at_zero"]


def test_all_pairs_reproduce_mdd_table():
    frozen = subprocess.run([sys.executable, str(settings.REPO / "scripts" / "mdd_table.py"), *PAIRS],
                            capture_output=True, text=True, check=True).stdout
    frozen = pd.read_csv(io.StringIO(frozen))
    frozen = frozen[frozen["pair"] == f"pooled K={len(PAIRS)}"][COLUMNS].reset_index(drop=True)
    rows = mdd_subsets.subset_rows(PAIRS, len(PAIRS))
    assert (rows["pairs"] == " ".join(PAIRS)).all()
    assert np.allclose(rows[COLUMNS].to_numpy(), frozen.to_numpy(), atol=5e-5)   # the frozen CSV is %.4f


def test_subsets_enumerated_completely():
    for k in (3, 2):
        rows = mdd_subsets.subset_rows(PAIRS, k)
        subsets = rows.groupby("pairs").size()
        assert (subsets == 9).all()                                               # every rho x A cell
        assert {frozenset(s.split()) for s in subsets.index} == {frozenset(c) for c in itertools.combinations(PAIRS, k)}
        assert len(subsets) == len(list(itertools.combinations(PAIRS, k)))


def test_joint_lower_bounds():
    rows = mdd_subsets.subset_rows(PAIRS, 3)
    p = rows["p_within_margin_at_zero"]
    assert np.allclose(rows["joint_lower_m5"], p ** 5)
    assert np.allclose(rows["joint_lower_m9"], p ** 9)
    assert (rows["joint_lower_m9"] <= rows["joint_lower_m5"]).all() and (rows["joint_lower_m5"] <= p).all()
