"""The two label-free pins from the blinded run: each pair's
scoring units per label (n0, n1) per primary-fold-set unit, for config.statistics.CLASS_COUNTS, and mouse
microglia's litters per label group, for HAMMOND_SCHEME (litters if each group has >= 5). The blinded
permutation moves whole label rows among scoring units, so each pair's totals and Hammond's litters per
group equal the real ones, and (since permuting within the donors scored in the same units)
so does Sound Life's split across its cross-fitting folds. CLASS_COUNTS' Sound Life folds were pinned from
the 2026-10-06 run under the earlier fold-ignoring permutation (only its totals real), so this script no
longer reproduces them. Same scored bulk, scoring units and fold set as stage 11 (pipeline/11_statistics.py
pair_results). Prints counts only.

  SC2BULK_PERMUTE_BULK_LABELS=1 python scripts/blinded_counts.py
"""
import pandas as pd

from sc2bulk import folds, labels
from sc2bulk.config import settings
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

assert settings.BLINDED is not None, "run blinded (SC2BULK_PERMUTE_BULK_LABELS)"
assert not settings.SMOKE, "the counts come from the full data, not the smoke subset"
counts = {}
for pair in PAIR_ORDER:
    primary = next(iter(PAIRS[pair]["fold_sets"]))
    bulk = pd.read_csv(settings.OUTPUTS / "02_preprocess" / pair / "bulk_samples.csv", dtype={"sample_id": str, "donor_id": str})
    by = "donor_id" if PAIRS[pair]["scoring_unit"] == "donor" else "sample_id"
    counts[pair] = []
    for unit in [u for u in folds.units(pair) if u[0] == primary]:
        scored = folds.scored_bulk(pair, *unit, bulk[bulk["drop_reason"].isna()])
        bulk_label = labels.bulk_labels(pair).merge(scored.assign(scoring_unit_id=scored[by]), on="sample_id")
        group = bulk_label.groupby("scoring_unit_id")["group"].first()
        counts[pair].append((int((group == 0).sum()), int((group == 1).sum())))
        if pair == "hammond":   # stage 11's unit_design count, on the same scored bulk
            print(f"hammond litters per label group: {bulk_label.groupby('group')['litter'].nunique().to_dict()}")
print(f"CLASS_COUNTS = {counts}")
