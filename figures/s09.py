"""Fig. S9: alternative read-outs: Fig 2A's format with the margin edge,
one panel per read-out. A LASSO (stage 11's per-pair rows; stage 11 pools ridge only); B-E elastic net, PLS, k-NN, MLP
and F CytoSet's native head (stage 11b's readout_* and native_head rows, per pair and pooled; unadjusted)."""

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

import common
import plots
import style
from sc2bulk.config import figures as F

FIGURE = "Fig. S9"
READS_BULK_LABELS = False
READS_SCORES = True


def lasso(pairs) -> tuple:
    table = common.pick(common.statistics(), neutral=False, pool_set="per_pair", readout="lasso", statistic="auroc_difference")
    table = table[table["pair"].isin(pairs)]
    values = {}
    for pair in pairs:
        part = table[table["pair"] == pair]
        assert sorted(part["key"]) == sorted(common.alternatives()), f"S9A {pair}: LASSO rows {sorted(part['key'])}"
        values[pair] = {r["key"]: (r["delta"], r["ci_low"], r["ci_high"], "") for _, r in part.iterrows()}
    return values, table


def panel(letter):
    readout = F.S9_READOUTS[letter]

    def draw(fig, cell, pairs):
        keys = ["cytoset_cells"] if readout == "native_head" else common.alternatives()
        if readout == "lasso":   # stage 11 pools ridge only: the pooled column is n/a, kept so the columns line up with b-f
            (values, table), na = lasso(pairs), set()
            if common.kept_pairs():
                values["pooled"] = {}
                na = {("pooled", k) for k in keys}
        else:
            row = "native_head" if readout == "native_head" else f"readout_{readout}"
            values, table, na = common.secondary_columns(pairs, row, keys, common.citation(FIGURE, letter))
        titles = common.short_column_titles(pairs) if letter == "A" else {c: "" for c in values}
        limits = F.XLIM_DIFFERENCE[FIGURE]
        axes = plots.forest_columns(fig, cell, values, keys, limits, na, margin=True, titles=titles, labels=[common.label(k) for k in keys],
                                    ticks=plots.difference_ticks(limits), wspace=0.45)
        style.panel_label(axes[0], letter, at_left=True, above_pt=14 if letter == "A" else None)   # A: level with the column titles
        axes[0].text(0.025, 1.02, F.TEXT["readout"][readout], transform=blended_transform_factory(fig.transFigure, axes[0].transAxes),
                     va="bottom", fontsize=F.TEXT_SIZE)
        if letter == list(F.S9_READOUTS)[-1]:
            plots.centered_xlabel(axes, F.TEXT["difference"])
        common.panel_csv(table[["pair", "key", "readout", "delta", "ci_low", "ci_high", "u", "p_better", "p_worse"]].assign(readout_shown=readout),
                         FIGURE, letter)
    return draw


def legend(fig):
    style.legend_strip(fig, common.alternatives(), ("pseudobulk_zero",), ncol=5)


def main(pairs, panels):
    with plt.rc_context(F.TEXT_RC):
        common.draw(FIGURE, {letter: panel(letter) for letter in F.S9_READOUTS}, pairs, panels, __file__, legend=legend, top_mm=24,
                    ratios=(9, 9, 9, 9, 9, 1.6), hspace=0.645, left=0.19, bottom=0.05)
