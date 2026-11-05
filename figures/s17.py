"""Fig. S16: every comparison with pseudobulk: one heatmap of the
difference in bulk AUROC from pseudobulk, rows = pair x alternative, columns = the primary analysis (stage 11, ridge and
LASSO) and every stage 11b row with a per-configuration difference (bulk subset all, and the age-adjusted row);
diverging, centered at 0 (the only diverging map). A cell a row doesn't form is hatched n/a. Always the last supplementary figure."""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.transforms import blended_transform_factory

import common
import plots
from sc2bulk.config import figures as F
from sc2bulk.config import secondary as C
from sc2bulk.config import style as S

FIGURE = "Fig. S16"
READS_BULK_LABELS = False
READS_SCORES = True
PAIR_COLUMN = 0.05                     # the pair names' x (figure fraction), just left of the row labels
COLORBAR = (0.55, 0.955, 0.4, 0.008)   # the color bar's axes (figure fraction): top right, above the heatmap
LEFT_OUT = (*C.UNSTREAMED_ROWS, "subgroups", "reference_rows", "within_bulk_reference", "best_candidate", "oracle", "own_recipe")


def column_name(r) -> str:
    return r["row"] + ("" if r["feature_set"] == "panel" else f" ({r['feature_set']})")


def panel_a(fig, cell, pairs):
    stats = common.pick(common.statistics(), neutral=False, pool_set="per_pair", statistic="auroc_difference")
    stats = stats[stats["pair"].isin(pairs)].assign(column=lambda t: "primary (" + t["readout"] + ")")
    sec = common.pick(common.secondary(), neutral=False, pool_set="per_pair", statistic="auroc_difference", row_kind="configuration",
                      bulk_subset=("all", "age_adjusted"))
    sec = sec[sec["pair"].isin(pairs) & ~sec["row"].isin(LEFT_OUT) & sec["key"].isin(common.alternatives())]
    sec = sec.assign(column=sec.apply(column_name, axis=1))
    table = pd.concat([stats[["pair", "key", "column", "delta"]], sec[["pair", "key", "column", "delta"]]], ignore_index=True)
    assert not table.duplicated(["pair", "key", "column"]).any(), "S16: two values for one cell"
    natural = lambda c: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", c)]   # 25% before 100%
    columns = [*sorted(stats["column"].unique()), *sorted(sec["column"].unique(), key=natural)]
    rows = [(p, k) for p in pairs for k in common.alternatives()]
    matrix = table.pivot(index=["pair", "key"], columns="column", values="delta").reindex(index=pd.MultiIndex.from_tuples(rows), columns=columns)
    na = {(r, c) for (r, c), v in np.ndenumerate(matrix.to_numpy()) if not np.isfinite(v)}
    ax = fig.add_subplot(cell)
    limits = F.XLIM_DIFFERENCE[FIGURE]
    keys = common.alternatives()
    plots.heatmap(ax, matrix.to_numpy(), S.DIVERGING_CMAP, limits, [common.label(k) for p in pairs for k in keys],
                  [F.S16_COLUMN_LABELS.get(c, c) for c in columns], na=na, text=False)
    for i, pair in enumerate(pairs):   # pairs as row blocks, named in a rotated column left of the row labels
        if i:
            ax.axhline(i * len(keys) - 0.5, color="black", linewidth=0.5)
        ax.text(PAIR_COLUMN, (i + 0.5) * len(keys) - 0.5, common.pair_title(pair).replace("\n", ", "), rotation=90, ha="center", va="center",
                transform=blended_transform_factory(fig.transFigure, ax.transData), fontsize=F.TEXT_SIZE)
    ax.tick_params(axis="both", labelsize=S.MIN_FONT_SIZE)
    bar = fig.colorbar(ScalarMappable(Normalize(*limits), S.DIVERGING_CMAP), cax=fig.add_axes(COLORBAR), orientation="horizontal",
                       extend="both")   # values past the range take the end colors
    bar.set_ticks(np.linspace(*limits, 5))
    bar.ax.tick_params(labelsize=S.MIN_FONT_SIZE)   # the heatmap's tick size
    bar.set_label(F.TEXT["difference"])
    bar.ax.xaxis.set_label_position("top")
    common.panel_csv(table, FIGURE, "A")   # one panel: no letter


def main(pairs, panels):
    with plt.rc_context(F.TEXT_RC):
        common.draw(FIGURE, {"A": panel_a}, pairs, panels, __file__, top_mm=17, left=0.215, bottom=0.11)
