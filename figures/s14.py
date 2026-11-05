"""Fig. S14: robustness of the gene lists behind Fig 5.

A: stability of each configuration's top 50 (stage 10's checks: across seeds, seeds 0-9 vs. 10-19, donor halves; ridge,
   phenotype; human blood, aging also across folds), one row per configuration and check;
B: Fig 5B with the top 25 and the top 100 genes; C: with LASSO's genes (top 50) (stage 10c);
D: the attribution checks: for each pair and configuration, the inclusion checks passed and failed (Geneformer's deletion
   checks, with its inclusion decision; triaged = not run), each counted once (random_T_top over both donor halves; human
   blood, aging: the pooled rows that decide, its per-fold rows only in the CSV), the ungated ones (completeness) in the CSV."""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.transforms import blended_transform_factory

import common
import fig5
import plots
import style
from sc2bulk.config import figures as F
from sc2bulk.config import style as S

FIGURE = "Fig. S14"
FULL = (0, 1)                          # the same-sign share axis: the variants here reach 0.06, so the full range
READS_BULK_LABELS = False
READS_SCORES = True                    # B and C read stage 10c's label-bearing concordance
STABILITY = ("stability_seeds", "stability_split_half", "stability_donors", "stability_folds")


def checks(pair) -> pd.DataFrame:
    return common.read_csv("10_attribution", pair, "checks.csv", keep_default_na=False, na_values=[""])


def panel_a(fig, cell, pairs):
    axes = plots.columns(fig, cell, len(pairs), wspace=fig5.WSPACE)
    keys = common.kept()
    frames = []
    for i, (ax, pair) in enumerate(zip(axes, pairs)):
        table = checks(pair)
        table = table[table["check"].isin(STABILITY) & (table["readout"] == "ridge") & (table["target"] == "phenotype")]
        for y, key in enumerate(keys):
            part = table[table["configuration"] == key]
            for j, check in enumerate(STABILITY):
                values = part.loc[part["check"] == check, "value"].astype(float)
                if len(values):
                    ax.plot(values.mean(), y + plots.dodge(j, len(STABILITY), 0.6), **style.marker(key, S.MARKER_SIZE * 0.8))
        fig5.share_axis(ax, FULL)
        plots.rows_axis(ax, [common.label(k) for k in keys], i == 0)
        ax.set_title(common.pair_title(pair))
        frames.append(table)
    order = ", ".join(F.STABILITY_LABELS[c] for c in STABILITY)
    plots.centered_xlabel(axes, f"{F.TEXT['stability']}\n{F.TEXT['stability_order'].format(checks=order)}")
    style.panel_label(axes[0], "A", at_left=True)
    common.panel_csv(pd.concat(frames, ignore_index=True), FIGURE, "A")


def sublabel(fig, ax, text):
    """A panel part's name beside its letter, at the figure's left edge."""
    ax.text(0.025, 1.02, text, transform=blended_transform_factory(fig.transFigure, ax.transAxes), va="bottom", fontsize=F.TEXT_SIZE)


def panel_b(fig, cell, pairs):
    grid = cell.subgridspec(2, 1, hspace=0.3)
    frames = []
    for row, k in enumerate((25, 100)):
        axes, frame = fig5.share_grid(fig, grid[row], pairs, k=k, limits=FULL)
        frames.append(frame)
        sublabel(fig, axes[0], F.TEXT["top_k"].format(k=k))
        if row == 0:
            style.panel_label(axes[0], "B", at_left=True)
    common.panel_csv(pd.concat(frames, ignore_index=True), FIGURE, "B")


def panel_c(fig, cell, pairs):
    axes, frame = fig5.share_grid(fig, cell, pairs, readout="lasso", limits=FULL)
    sublabel(fig, axes[0], F.TEXT["lasso_top"])
    plots.centered_xlabel(axes, F.TEXT["shared_share_k"])
    style.panel_label(axes[0], "C", at_left=True)
    common.panel_csv(frame, FIGURE, "C")


def panel_d(fig, cell, pairs):
    ax = fig.add_subplot(cell)
    keys = common.kept()
    frames = []
    for x, pair in enumerate(pairs):
        table = checks(pair)
        inclusion = common.read_csv("10_attribution", pair, "inclusion.csv", keep_default_na=False).set_index("configuration")
        for y, key in enumerate(keys):
            gated = table[(table["configuration"] == key) & table["passed"].notna()]
            if (gated["fold"] == -1).any():   # human blood, aging: the per-fold rows are reported; the pooled rows decide
                gated = gated[gated["fold"] == -1]
            failed = (gated["passed"].astype(str) == "False").groupby(gated["check"]).any()   # a check fails in either donor half
            if key in inclusion.index and inclusion.loc[key, "failed_checks"] in F.DROPPED_TEXT:
                text = F.FIGURE_TEXT[F.DROPPED_TEXT[inclusion.loc[key, "failed_checks"]]]
            elif len(failed):
                text = f"{F.TEXT['failed']} {failed.sum()}/{len(failed)}" if failed.any() else f"{F.TEXT['passed']} {len(failed)}/{len(failed)}"
            else:
                text = "—"
            ax.text(x, y, text, ha="center", va="center", fontsize=S.MIN_FONT_SIZE)
        frames.append(table.merge(inclusion.reset_index()[["configuration", "included", "failed_checks"]], on="configuration", how="left"))
    ax.set_xlim(-0.5, len(pairs) - 0.5)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([common.pair_title(p) for p in pairs])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0)
    plots.rows_axis(ax, [common.label(k) for k in keys], True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    style.panel_label(ax, "D", at_left=True, y=1.12)   # above the column headers
    common.panel_csv(pd.concat(frames, ignore_index=True), FIGURE, "D")


def legend(fig):
    style.legend_strip(fig, common.kept(), ("base_all", "base_matched"), ncol=5)


def main(pairs, panels):
    ticks = {"xtick.labelsize": S.MIN_FONT_SIZE, "ytick.labelsize": S.MIN_FONT_SIZE}   # ten row labels per panel; one tick size
    with plt.rc_context({**F.TEXT_RC, **ticks}):
        common.draw(FIGURE, {"A": panel_a, "B": panel_b, "C": panel_c, "D": panel_d}, pairs, panels, __file__, legend=legend,
                    ratios=(1, 2.1, 1, 1.05), top_mm=24, hspace=(0.6, 0.39, 0.61), left=0.17, bottom=0.02)
