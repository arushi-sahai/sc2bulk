"""Read-only check that stage 06's row E passes its pre-declared assert (training_inputs.learning_curve_subset:
>= 2 validation holdout groups per label at each fraction): for every learning-curve pair, unit with a row E
(not split67), encoder seed and fraction, the validation groups per label row E keeps (06_train.learning_curve_counts,
the counts the smoke run skips on) and whether both are at least 2. Reads stage 03's splits and the training
samples; writes nothing.

  python scripts/check_row_e_groups.py
"""
import runpy

from sc2bulk import folds, training_inputs
from sc2bulk.config import settings
from sc2bulk.config import training as T

G = runpy.run_path(str(settings.REPO / "pipeline" / "06_train.py"), run_name="stage06")   # its functions, not its main
print(f"smoke: {settings.SMOKE}; outputs: {settings.OUTPUTS}")
short = 0
for pair in T.LEARNING_CURVE_PAIRS:
    for unit in [u for u in folds.units(pair) if u[0] != "split67"]:
        for e in training_inputs.encoder_seeds(pair, "E"):
            for point in G["points"](pair, "E"):
                counts = G["learning_curve_counts"](pair, unit, e, int(point[1:]) / 100)
                passes = min(counts.values()) >= 2
                short += not passes
                print(f"{pair} {folds.unit_name(*unit)} s{e} {point}: validation groups label 0 {counts[0]}, label 1 {counts[1]}, "
                      f"{'pass' if passes else 'SHORT'}")
print("every row E split passes" if not short else f"{short} row E splits short of 2 validation groups per label")
