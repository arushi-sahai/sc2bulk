"""Fig. S12: further changes.

A: the foundation models on their full gene sets, and pseudobulk on the panel genes with one-to-one human orthologs
   (mouse pairs), vs. primary pseudobulk (stage 11b's feature_variants row; drawn from 11b's differences and CIs rather
   than the scores table, so the panel has intervals);
B: UCE, drawn iff config/figures.S12B_UCE (asserted against the scores table both ways);
C: learning curves on training.LEARNING_CURVE_PAIRS: each configuration's difference from pseudobulk at 25, 50 and 100%
   of training donors (stage 11b's learning_curve rows, seeds 0-4); only those pairs are drawn."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
import plots
import style
from sc2bulk.config import figures as F
from sc2bulk.config import style as S

FIGURE = "Fig. S12"
READS_BULK_LABELS = False
READS_SCORES = True
FRACTIONS = {"learning_curve_25": 25, "learning_curve_50": 50, "learning_curve_100": 100}
LETTER_MM = 4                          # the letters' left edge (mm)
DODGE = 10                             # C: the configurations' x spread at each fraction (percentage points)


def panel_a(fig, cell, pairs):
    present = not common.pick(common.scores(), neutral=False, model="uce").empty
    assert present == F.S12B_UCE, f"S12B: UCE rows {present} but S12B_UCE {F.S12B_UCE}"
    fms = [k for k in common.alternatives() if k.startswith(("geneformer", "scgpt"))]
    rows = [*fms, "pseudobulk"]

    values, table, na = common.secondary_columns(pairs, "feature_variants", rows, common.citation(FIGURE, "A"),
                                                 feature_set=("full_genes", "mapped_orthologs"))
    limits = F.XLIM_DIFFERENCE[FIGURE]
    axes = plots.forest_columns(fig, cell, values, rows, limits, na, titles=common.short_column_titles(pairs),
                                labels=[common.label(k) for k in rows], ticks=plots.difference_ticks(limits), wspace=0.45)
    plots.centered_xlabel(axes, F.TEXT["difference"])
    style.panel_label(axes[0], "A", at_left=True, left_mm=LETTER_MM, above_pt=14)   # level with the column titles
    common.panel_csv(table[["pair", "key", "feature_set", "delta", "ci_low", "ci_high"]], FIGURE, "A")


def panel_b(fig, cell, pairs):
    ax = fig.add_subplot(cell)
    rows = []
    for x, pair in enumerate(pairs):
        part = common.pick(common.scores(), pair=pair, model="uce", side="bulk", bulk_subset="all")
        value = part["auroc"].mean()
        ax.bar(x, value, color=S.MODALITY_COLOR, linewidth=0)
        rows.append({"pair": pair, "auroc": value})
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([common.pair_title(p) for p in pairs])
    style.panel_label(ax, "B", at_left=True, left_mm=LETTER_MM)
    common.panel_csv(pd.DataFrame(rows), FIGURE, "B")


def panel_c(fig, cell, pairs):
    """Lettered b when the UCE panel isn't drawn (the registry and file names keep "C"). Only the learning-curve pairs are
    drawn, across the full width (the others are named in the legend)."""
    drawn = [p for p in pairs if (p, "all") not in F.NOT_APPLICABLE["Fig. S12C"]]
    axes = plots.columns(fig, cell, len(drawn), wspace=0.3)
    limits = F.XLIM_DIFFERENCE[FIGURE]
    table = common.pick(common.secondary(), neutral=False, row=tuple(FRACTIONS), statistic="auroc_difference")
    assert set(table["pair"]) <= set(drawn), f"S12C: learning-curve rows outside LEARNING_CURVE_PAIRS ({sorted(set(table['pair']))})"
    keys = common.alternatives()
    rows = []
    for ax, pair in zip(axes, drawn):
        for j, key in enumerate(keys):
            part = table[(table["pair"] == pair) & (table["key"] == key)]
            assert sorted(part["row"]) == sorted(FRACTIONS), f"S12C {pair} {key}: rows {sorted(part['row'])}"
            x = part["row"].map(FRACTIONS).to_numpy()
            order = np.argsort(x)
            shifted = x[order] + plots.dodge(j, len(keys), DODGE)   # configurations side by side at each fraction
            ax.plot(shifted, part["delta"].to_numpy()[order], color=style.color(key), linewidth=0.5)
            for xx, value in zip(shifted, part["delta"].to_numpy()[order]):   # past the range: an arrowhead at the edge
                if np.isfinite(value):   # a member stage 11b could not estimate stays unplotted
                    plots.point(ax, xx, value, limits, style.marker(key), orientation="y")
            rows += [{"pair": pair, "key": key, "training_percent": FRACTIONS[r["row"]], "delta": r["delta"], "ci_low": r["ci_low"],
                      "ci_high": r["ci_high"]} for _, r in part.iterrows()]
        plots.difference_axis(ax, limits, margin=True, orientation="y")
        ax.set_xticks(list(FRACTIONS.values()))
        ax.set_xlim(10, 115)
        ax.set_title(common.pair_title(pair))
    plots.share_y(axes, limits)
    axes[0].set_yticks(plots.difference_ticks(limits))
    axes[0].set_ylabel(F.TEXT["difference_short"])
    plots.centered_xlabel(axes, F.TEXT["fraction"])
    style.panel_label(axes[0], "C" if F.S12B_UCE else "B", at_left=True, left_mm=LETTER_MM)
    common.panel_csv(pd.DataFrame(rows), FIGURE, "C")


def legend(fig):
    style.legend_strip(fig, common.kept(), ("pseudobulk_zero",), ncol=6)


def main(pairs, panels):
    functions = {"A": panel_a, **({"B": panel_b} if F.S12B_UCE else {}), "C": panel_c}
    with plt.rc_context(F.TEXT_RC):
        common.draw(FIGURE, functions, pairs, panels, __file__, legend=legend, top_mm=19, hspace=0.62, left=0.17, bottom=0.1)
