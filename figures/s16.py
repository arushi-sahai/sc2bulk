"""Fig. S15: modality-direction genes for each learned cells configuration other than Fig 6B's CytoSet exemplar:
Fig 6B (left) per configuration (one panel each) and pair (columns).
Label-free (stage 10); a configuration without a ranking (Geneformer's deletion triaged) or whose checks failed (Geneformer,
included False, as Fig 5B) is n/a."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import blended_transform_factory

import common
import fig6
import plots
import style
from sc2bulk.config import figures as F
from sc2bulk.config import style as S

FIGURE = "Fig. S15"
READS_BULK_LABELS = False
READS_SCORES = False


def panel(letter, key):
    def draw(fig, cell, pairs):
        axes = plots.columns(fig, cell.subgridspec(1, 2, width_ratios=(1, 0.1), wspace=0)[0, 0], len(pairs), wspace=2.3)   # room for the values
        frames = []
        for ax, pair in zip(axes, pairs):
            dropped, excluded = common.dropped(pair), common.excluded(pair)
            if key not in common.kept() or key in dropped or key in excluded:
                note = F.FIGURE_TEXT["not_applicable"] if key in excluded else F.FIGURE_TEXT[F.DROPPED_TEXT[dropped.get(key, "triaged")]]
                ax.axis("off")
                style.na_cell(ax, 0, 0, 1, 1, note)
                frames.append(pd.DataFrame([{"pair": pair, "key": key, "note": note}]))
            else:
                frames.append(fig6.draw_modality_genes(ax, pair, key, S.MIN_FONT_SIZE))   # 6 pt: thirty narrow columns
                tick = float(f"{0.8 * ax.get_xlim()[1]:.1g}")   # 0 and one round value near the end: tiny ranges' labels collide
                ax.set_xticks([0, tick], ["0", f"{tick:g}"])
            ax.tick_params(axis="x", labelsize=S.MIN_FONT_SIZE)
            if letter == "A":
                ax.set_title(common.pair_title(pair), pad=12)
        style.panel_label(axes[0], letter, at_left=True, above_pt=14 if letter == "A" else None)   # A: level with the pair titles
        axes[0].text(0.025, 1.02, common.label(key), transform=blended_transform_factory(fig.transFigure, axes[0].transAxes), va="bottom",
                     fontsize=F.TEXT_SIZE)
        if letter == list(F.PANELS[FIGURE])[-1]:
            plots.centered_xlabel(axes, F.TEXT["attribution"])
        common.panel_csv(pd.concat(frames, ignore_index=True), FIGURE, letter)
    return draw


def main(pairs, panels):
    with plt.rc_context(F.TEXT_RC):
        common.draw(FIGURE, {letter: panel(letter, key) for letter, key in zip(F.PANELS[FIGURE], F.S15_KEYS)}, pairs, panels, __file__,
                    top_mm=13, hspace=0.44, left=0.06, bottom=0.039)
