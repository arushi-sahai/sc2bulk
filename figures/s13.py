"""Fig. S13: the modality gap in every dataset pair.

A-E: Fig 3A's 3 x 4 for the other pairs, in pair order (fold 0 of each primary fold set: human blood, aging's crossfit_0);
F-I: Fig 3B and 3C for every configuration under the five conditions of config/figures.S13_CONDITIONS, split by input
   (pseudobulk the first group in each): F silhouette and G bulk AUROC for the configurations on cells, H and I for the
   pseudobulk-input configurations."""
import matplotlib.pyplot as plt

import common
import fig3
import style
from sc2bulk.config import figures as F

FIGURE = "Fig. S13"
READS_BULK_LABELS = True
READS_SCORES = True
LETTER = {"left_mm": 10.2, "above_pt": 4}   # f-i's letters: in line with a's and d's, clear of the top tick label
KEY_X, KEY_Y = 0.70, 0.71             # the keys' top-left corner (figure fraction), in the empty slot right of d and e


def gap(pair, letter):
    def draw(fig, cell, pairs):
        if pair in pairs:
            fig3.draw_gap(fig, cell, pair, letter, FIGURE, compact=True)
        else:
            common.absent(fig, cell, FIGURE, letter, pair)
    return draw


def keys(group) -> list:
    return [k for k in F.S13_KEYS[group] if k in common.kept()]


def offsets(group, auroc, letter):
    def draw(fig, cell, pairs):
        fig3.offset_panel(fig, cell, pairs, fig3.aurocs if auroc else fig3.silhouettes, keys(group), F.S13_CONDITIONS, letter, FIGURE,
                          auroc, titles=letter == "F", letter=LETTER)
    return draw


def legend(fig):
    """a-e's key (data colors) and f-i's key (family hues), one over the other in the slot right of d and e."""
    style.legend_strip(fig, [], ("side_single_cell", "side_bulk", "group_0", "group_1"), x=KEY_X, y=KEY_Y, ncol=2, loc="upper left")
    style.legend_strip(fig, list(dict.fromkeys(keys("cells") + keys("pseudobulk"))), ("pseudobulk_within_assay",), x=KEY_X,
                       y=KEY_Y - 0.04, ncol=1, loc="upper left")


def main(pairs, panels):
    functions = {letter: gap(pair, letter) for letter, pair in zip("ABCDE", F.FIG13_OTHER_PAIRS)}
    functions.update({"F": offsets("cells", False, "F"), "G": offsets("cells", True, "G"), "H": offsets("pseudobulk", False, "H"),
                      "I": offsets("pseudobulk", True, "I")})
    with plt.rc_context(F.TEXT_RC):
        common.draw(FIGURE, functions, pairs, panels, __file__, rows=[("A", "B", "C"), ("D", "E", None), ("F",), ("G",), ("H",), ("I",)],
                    widths=[(1, 1, 1), (1, 1, 1), None, None, None, None], wspace=(0.35, 0.35, 0, 0, 0, 0),
                    ratios=(1.4, 1.4, 1.0, 0.45, 0.75, 0.45), top_mm=17, hspace=(0.9, 0.6, 0.25, 0.3, 0.25), left=0.087, bottom=0.03,
                    legend=legend)
