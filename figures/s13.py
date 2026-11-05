"""Fig. S13: the modality gap in every dataset pair.

A-E: Fig 3A's 2 x 2 for the other pairs, in pair order (fold 0 of each primary fold set: human blood, aging's crossfit_0);
F: Fig 3B and 3C for the pseudobulk-input configurations (and pseudobulk), with the unaligned stacked scaler as a
   fifth, light column (silhouette and bulk AUROC)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
import fig3
import plots
import style
from sc2bulk.config import configurations
from sc2bulk.config import figures as F
from sc2bulk.config import style as S

FIGURE = "Fig. S13"
READS_BULK_LABELS = True
READS_SCORES = True
F_LETTER_MM = 10.2                     # f's letter's left edge (mm), in line with a's and d's
STACKED_OFFSET = 0.42                  # the light unaligned-stacked mark, right of each configuration's four columns


def gap(pair, letter):
    def draw(fig, cell, pairs):
        if pair in pairs:
            fig3.draw_gap(fig, cell, pair, letter, FIGURE, compact=True)
        else:
            common.absent(fig, cell, FIGURE, letter, pair)
    return draw


def panel_f(fig, cell, pairs):
    keys = ["pseudobulk", *[k for k in F.S13F_KEYS if k in common.alternatives()]]
    grid = cell.subgridspec(2, 1, hspace=0.25)
    frames = []
    for row, (values_of, ylabel) in enumerate(((fig3.silhouettes, F.TEXT["silhouette"]), (fig3.aurocs, F.TEXT["bulk_auroc"]))):
        axes = plots.columns(fig, grid[row], len(pairs))
        for ax, pair in zip(axes, pairs):
            frame = values_of(pair, keys, (*configurations.ALIGNMENTS, "unaligned_stacked"))
            values = {(r.key, r.alignment): r.value for r in frame.itertuples()}
            limits = (-np.inf, np.inf) if row == 0 else F.AUROC_RANGE
            plots.offset_columns(ax, values, keys, limits)
            for j, key in enumerate(keys):
                plots.point(ax, j + STACKED_OFFSET, values[(key, "unaligned_stacked")], limits, {**style.marker(key), "alpha": S.SEED_DOT_ALPHA},
                            orientation="y")
            if row == 1:
                fig3.auroc_lines(ax, frame)
            else:
                ax.set_title(common.pair_title(pair))
            frames.append(frame.assign(measure="silhouette" if row == 0 else "bulk_auroc"))
        plots.share_y(axes, None if row == 0 else F.AUROC_RANGE)
        axes[0].set_ylabel(ylabel.replace(" ", "\n", 1))
        if row == 0:
            style.panel_label(axes[0], "F", at_left=True, left_mm=F_LETTER_MM)
            first = frames[0][frames[0]["key"] == "pseudobulk"]   # the first pair's pseudobulk dots, named in place
            plots.alignment_callout(axes[0], [(STACKED_OFFSET if r.alignment == "unaligned_stacked" else S.ALIGNMENT_X_OFFSET[r.alignment], r.value,
                                               F.ALIGNMENT_TICKS[r.alignment]) for r in first.itertuples()], step=0.125)   # five names: more room
    common.panel_csv(pd.concat(frames, ignore_index=True), FIGURE, "F")


def legend(fig):
    """a-e's key (data colors) at the top, as in every figure; f's key (family hues) under f."""
    style.legend_strip(fig, [], ("side_single_cell", "side_bulk", "group_0", "group_1"), ncol=4)
    style.legend_strip(fig, ["pseudobulk", *[k for k in F.S13F_KEYS if k in common.alternatives()]], ("pseudobulk_unaligned", "chance"),
                       y=0.003, ncol=3, loc="lower center")


def main(pairs, panels):
    functions = {letter: gap(pair, letter) for letter, pair in zip("ABCDE", F.FIG13_OTHER_PAIRS)}
    with plt.rc_context(F.TEXT_RC):
        common.draw(FIGURE, {**functions, "F": panel_f}, pairs, panels, __file__, rows=[("A", "B", "C"), (None, "D", "E", None), ("F",)],
                    widths=[(1, 1, 1), (0.5, 1, 1, 0.5), None], wspace=(0.35, 0.35, 0.35), ratios=(1, 1, 1.2), legend=legend, top_mm=22,
                    hspace=(0.5, 0.38), left=0.087, bottom=0.05)
