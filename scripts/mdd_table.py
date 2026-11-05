"""The a priori minimum-detectable-difference table: per
non-ceiling pair and pooled, over rho x A, with the Holm first-step column and P(pooled u < X | delta = 0).
Pooled at K = all pairs given (the planning value, the pairs expected below the ceiling) and at every
leave-one-out K - 1 subset (no pair is singled out). Label-free: CLASS_COUNTS comes from the blinded run.

  python scripts/mdd_table.py PAIR [PAIR ...] > mdd_table.csv
The command used (the five pairs expected below the ceiling):
  python scripts/mdd_table.py tms soundlife pineda lupus crohns > mdd_table.csv
"""
import itertools
import sys

import pandas as pd

from sc2bulk import statistics
from sc2bulk.config import settings
from sc2bulk.config import statistics as S

assert S.CLASS_COUNTS is not None, "pin config.statistics.CLASS_COUNTS from the blinded run first"
pairs = sys.argv[1:]
assert len(pairs) >= 2, "the table pools at least 2 pairs (and their leave-one-out subsets)"
tables = [statistics.mdd_table({p: S.CLASS_COUNTS[p] for p in pairs}, S.SEED_SD, settings.N_ENCODER_SEEDS, S.MARGIN, S.ALPHA)]
for subset in itertools.combinations(pairs, len(pairs) - 1):
    t = statistics.mdd_table({p: S.CLASS_COUNTS[p] for p in subset}, S.SEED_SD, settings.N_ENCODER_SEEDS, S.MARGIN, S.ALPHA)
    tables.append(t[t["pair"].str.startswith("pooled")].assign(pair=lambda d: d["pair"] + " without " + next(iter(set(pairs) - set(subset)))))
print(pd.concat(tables, ignore_index=True).to_csv(index=False, float_format="%.4f"), end="")
