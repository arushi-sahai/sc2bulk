"""Figs. S2–S7: quality control, one figure per pair, one template.

Single-cell side: A each sample's median genes detected, total counts and mitochondrial percent per cell, by group,
with the per-cell thresholds (stage 02's per-cell QC table; per-sample quantiles in the CSV), B post-QC cells per
sample by group with the MIN_CELLS_PER_SAMPLE line, C the QC UMAP (stage 04c) by annotated cell type (a key, no
in-plot labels), D one small UMAP per group (the group's cells over all cells in gray). Bulk side: E samples per
group, F library size and genes detected, G the bulk PCA by group (bulk groups through sc2bulk.labels). Then the pair's
own panels (config/figures.QC_EXTRA). Data colors only (config/style.py's color rule): sides in SIDE_COLORS, cell
types in the shared cell-type colors, groups open (0) vs. filled (1) and named by config/figures.GROUP_NAMES; one key
at the top right. Sensitive columns plotted here (config/figures.PANEL_SENSITIVE) appear only in
their panel's CSV."""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import common
import plots
import style
from sc2bulk.config import figures as F
from sc2bulk.config import settings
from sc2bulk.config import style as S
from sc2bulk.config.pairs import PAIRS

READS_BULK_LABELS = True
READS_SCORES = False
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
METRICS = ("genes_detected", "total_counts", "percent_mito")
LETTER_GAP_MM = 0.5                    # an extra panel's letter: its right edge this far left of its axes (level with A's)


def groups_axis(ax, ylabel, pair):
    ax.set_xticks([0, 1])
    ax.set_xticklabels(F.GROUP_NAMES[pair])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(ylabel)


def strip(ax, values, groups, side="single_cell"):
    """Values by 0/1 group (open vs. filled, the side's color), jittered by a fixed pattern (no random draw)."""
    for g in (0, 1):
        v = np.asarray(values[groups == g], dtype=float)
        offsets = (np.arange(len(v)) % 7 - 3) * 0.04
        ax.scatter(g + offsets, v, s=4, rasterized=True, **style.side_points(side, g))


def panel_a(fig, cell, pair, item):
    """Each kept sample's median per cell, by group, with the per-cell thresholds. The quantiles are computed for every
    sample in the cell QC table; the CSV keeps them all (group NaN for samples stage 02 dropped), the plot shows only the
    samples with a group (strip draws groups 0 and 1)."""
    ceiling = PAIRS[pair].get("qc_max_percent_mito", settings.QC_MAX_PERCENT_MITO)
    group = common.single_cell_samples(pair).set_index("sample_id")["group"]
    cells = common.cell_qc(pair, ("sample_id", *METRICS))
    quantiles = cells.groupby("sample_id")[list(METRICS)].quantile(list(QUANTILES)).unstack()
    axes = plots.columns(fig, cell, len(METRICS), wspace=0.8)
    for ax, metric, threshold in zip(axes, METRICS, (settings.QC_MIN_GENES_PER_CELL, None, ceiling)):
        median = quantiles[metric][0.5]
        strip(ax, median.to_numpy(), median.index.map(group).to_numpy())
        if threshold is not None:
            ax.axhline(threshold, **S.CHANCE_LINE)
        groups_axis(ax, F.QC_LABELS[metric], pair)
    style.panel_label(axes[0], "A")
    frame = quantiles.stack(level=0, future_stack=True).reset_index().rename(columns={"level_1": "metric"})
    frame.columns = [str(c) for c in frame.columns]
    common.panel_csv(frame.assign(group=frame["sample_id"].map(group), pair=pair), item, "A")


def panel_b(fig, cell, pair, item):
    samples = common.single_cell_samples(pair)
    ax = fig.add_subplot(cell)
    strip(ax, samples["cells_after_qc"].to_numpy(), samples["group"].to_numpy())
    ax.axhline(settings.MIN_CELLS_PER_SAMPLE, **S.CHANCE_LINE)
    groups_axis(ax, F.QC_LABELS["cells_after_qc"], pair)
    style.panel_label(ax, "B")
    common.panel_csv(samples[["sample_id", "group", "cells_after_qc", "cells_stored"]].assign(pair=pair), item, "B")


def umap_axis(ax):
    """A square UMAP axes, equal aspect, no ticks."""
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(F.TEXT["umap1"])


def umap(pair) -> pd.DataFrame:
    return common.read_parquet("04c_qc_embed", pair, "umap.parquet")


def panel_c(fig, cell, pair, item, pairs):
    """The UMAP by annotated cell type, its key at the right (no in-plot labels, so none overlap)."""
    table = umap(pair)
    ax = fig.add_subplot(cell)
    colors = {**common.cell_type_colors(pairs), S.OTHER_TYPE["name"]: S.OTHER_TYPE["color"]}
    kept = common.type_order(pair)[:len(S.CELL_TYPE_PALETTE)]   # the rest are "other"
    types = table["cell_type"].astype(str).where(table["cell_type"].astype(str).isin(kept), S.OTHER_TYPE["name"])
    ax.scatter(table["umap1"], table["umap2"], s=0.5, c=types.map(colors), linewidths=0, rasterized=True)
    umap_axis(ax)
    ax.set_anchor("W")
    ax.set_ylabel(F.TEXT["umap2"])
    shown = [*kept, *([S.OTHER_TYPE["name"]] if (types == S.OTHER_TYPE["name"]).any() else [])]
    ax.legend(handles=[Patch(facecolor=colors[t], linewidth=0, label=style.type_label(t)) for t in shown], loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=S.MIN_FONT_SIZE, handlelength=0.8, handletextpad=0.3, labelspacing=0.2, borderaxespad=0)
    style.panel_label(ax, "C")
    common.panel_csv(table.groupby(table["cell_type"].astype(str)).size().rename("cells").reset_index().assign(pair=pair), item, "C")


def panel_d(fig, cell, pair, item):
    """One small UMAP per group: that group's cells in the single-cell color, the other group's in gray, drawn in one random order."""
    table = umap(pair)
    group = common.single_cell_samples(pair).set_index("sample_id")["group"]
    assert table["sample_id"].isin(group.index).all(), f"{pair}: UMAP cells of samples outside stage 02's kept samples"
    shade = table["sample_id"].map(group)
    axes = plots.columns(fig, cell, 2, wspace=0.1)
    order = np.random.default_rng(UMAP_ORDER_SEED).permutation(len(table))   # one random drawing order: where groups overlap, both show
    shuffled = table.iloc[order]
    for ax, g in zip(axes, (0, 1)):
        mine = (shade.iloc[order] == g).to_numpy()
        ax.scatter(shuffled["umap1"], shuffled["umap2"], s=0.5, c=np.where(mine, S.SIDE_COLORS["single_cell"], S.OTHER_TYPE["color"]),
                   linewidths=0, rasterized=True)
        umap_axis(ax)
        ax.set_title(F.GROUP_NAMES[pair][g])
    style.panel_label(axes[0], "D")
    common.panel_csv(pd.DataFrame({"group": shade}).value_counts().rename("cells").reset_index().assign(pair=pair), item, "D")


def bulk_table(pair) -> pd.DataFrame:
    """The kept bulk samples with their (permuted, in a blinded run) group."""
    groups = common.bulk_labels(pair)[["sample_id", "group"]]
    table = common.bulk_samples(pair).merge(groups, on="sample_id", how="left", validate="one_to_one")
    assert table["group"].notna().all(), f"{pair}: a kept bulk sample without a label"
    return table


def panel_e(fig, cell, pair, item):
    table = bulk_table(pair)
    counts = table["group"].value_counts().reindex([0, 1], fill_value=0)
    ax = fig.add_subplot(cell)
    for g in (0, 1):
        ax.bar(g, counts[g], **style.side_bar("bulk", g))
    groups_axis(ax, F.TEXT["count"], pair)
    ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    style.panel_label(ax, "E")
    common.panel_csv(counts.rename("samples").rename_axis("group").reset_index().assign(pair=pair), item, "E")


def panel_f(fig, cell, pair, item):
    table = common.bulk_samples(pair)
    axes = plots.columns(fig, cell, 2, wspace=0.6)
    for ax, values, label in zip(axes, (table["library_size"] / 1e6, table["genes_detected"] / 1e3), ("library_size", "bulk_genes_detected")):
        ax.hist(values, bins=20, color=S.SIDE_COLORS["bulk"], linewidth=0)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_xlabel(F.QC_LABELS[label])
    axes[0].axvline(settings.BULK_MIN_LIBRARY / 1e6, **S.CHANCE_LINE)
    axes[0].set_ylabel(F.TEXT["count"])
    style.panel_label(axes[0], "F")
    common.panel_csv(table[["sample_id", "library_size", "genes_detected"]].assign(pair=pair), item, "F")


def panel_g(fig, cell, pair, item):
    pca = common.read_parquet("04c_qc_embed", pair, "bulk_pca.parquet")
    table = pca[["sample_id", "pc1", "pc2"]].merge(bulk_table(pair)[["sample_id", "group"]], on="sample_id", validate="one_to_one")
    ax = fig.add_subplot(cell)
    for g, part in table.groupby("group"):
        ax.scatter(part["pc1"], part["pc2"], s=8, rasterized=True, **style.side_points("bulk", g))
    ax.set_xlabel(F.TEXT["pc1"])
    ax.set_ylabel(F.TEXT["pc2"])
    style.panel_label(ax, "G")
    common.panel_csv(table.assign(pair=pair), item, "G")


def extra_panel(panel):
    """A pair-specific panel (config/figures.QC_EXTRA): counts or values of a column by 0/1 group."""
    def draw(fig, cell, pair, item):
        _, source, columns = F.QC_EXTRA[pair][panel]
        columns = (columns,) if isinstance(columns, str) else columns
        if source in ("single_cell", "both"):
            table = common.single_cell_samples(pair).assign(side="single_cell")
        if source == "both":
            table = pd.concat([table, bulk_table(pair).assign(side="bulk")], ignore_index=True)
        elif source == "bulk":
            table = bulk_table(pair).assign(side="bulk")
        elif source == "labels":
            table = common.bulk_labels(pair).assign(side="bulk")
        elif source == "organs_per_donor":
            samples = common.single_cell_samples(pair)
            table = samples.groupby("donor_id").agg(organs=(columns[0], "nunique"), group=("group", "first")).reset_index().assign(side="single_cell")
            columns = ("organs",)
        elif source == "restriction":   # every single-cell sample, kept or dropped, by disease status (group: label_value 0/1)
            samples = common.all_single_cell_samples(pair)
            restricted = samples["drop_reason"].eq(F.LUPUS_RESTRICTION)
            table = samples.assign(restriction=np.where(samples["drop_reason"].isna(), "kept", np.where(restricted, "dropped: cohort restriction", "dropped: other reason")),
                                   group=samples["label_value"].astype(int), side="single_cell")
        elif source == "folds":
            folds_ = common.read_csv("03_splits", pair, "donor_folds.csv", dtype={"donor_id": str})
            donors = common.single_cell_samples(pair).groupby("donor_id")["group"].first()
            table = folds_[(folds_["side"] == "scored") & (folds_["fold_set"] == common.primary_units(pair)[0][0])]
            table = table.assign(group=table["donor_id"].map(donors), side="single_cell")
        side = "bulk" if source in ("bulk", "labels") else "single_cell"
        numeric = {c: pd.api.types.is_float_dtype(table[c]) or (pd.api.types.is_numeric_dtype(table[c]) and table[c].nunique() > 8)
                   for c in columns}   # measurements (float, or many integer values) as strips; codes as bars
        many = [c for c in columns if not numeric[c] and table[c].nunique() > F.MAX_ROTATED_LEVELS]   # lots: one shared axes
        slots = ([tuple(many)] if many else []) + [c for c in columns if c not in many]
        grid = cell.subgridspec(1, len(slots), wspace=0.6,
                                width_ratios=[1.0 * len(c) if isinstance(c, tuple) else 1 if numeric[c] else 2.4 for c in slots])
        axes = [fig.add_subplot(grid[0, i]) for i in range(len(slots))]
        out = []
        previous_counts = False   # "Count" only on the first of adjacent count axes
        for ax, column in zip(axes, slots):
            if isinstance(column, tuple):   # samples per level of each many-level column, by group
                labels = []
                for i, c in enumerate(column):
                    per_level = table.groupby([c, "group"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
                    for g in (0, 1):
                        v = per_level[g].to_numpy()
                        ax.scatter(i + (g - 0.5) * 0.4 + (np.arange(len(v)) % 5 - 2) * 0.03, v, s=4, rasterized=True,
                                   **style.side_points(side, g))
                    labels.append(f"{F.COLUMN_SHORT[c]}\n({len(per_level)})")
                    out.append(per_level.stack().rename("count").reset_index().rename(columns={c: "level"}).assign(column=c, side=side))
                ax.set_xticks(range(len(column)))
                ax.set_xticklabels(labels)
                ax.set_xlim(-0.6, len(column) - 0.4)
                ax.set_xlabel(F.MANY_LEVELS_LABEL)
                ax.set_ylabel(F.QC_LABELS["per_level"])
                ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
                previous_counts = False
            elif numeric[column]:
                strip(ax, table[column].to_numpy(), table["group"].to_numpy(), side)
                groups_axis(ax, style.column_label(column), pair)
                out.append(table[["side", "group", column]].assign(sample_id=table.get("sample_id", table.get("donor_id"))))
                previous_counts = False
            else:
                counts = table.groupby(["side", column, "group"]).size().rename("count").reset_index()
                plots.group_bars(ax, counts, column)
                levels = [v.removeprefix(F.LEVEL_PREFIX.get(column, "")) for v in sorted(counts[column].astype(str).unique())]   # group_bars' order
                plots.level_axis(ax, style.column_label(column), levels)
                ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
                ax.set_ylabel("" if previous_counts else F.TEXT["count"])
                out.append(counts)
                previous_counts = True
        width_mm = axes[0].get_position().width * fig.get_size_inches()[0] * 25.4
        style.panel_label(axes[0], panel, x=-LETTER_GAP_MM / width_mm)   # a wide axes' letter as close as a narrow one's
        common.panel_csv(pd.concat(out, ignore_index=True).assign(pair=pair), item, panel)
    return draw


# Rows after A|B and C|D, with width ratios: small extra panels sit beside E-G (a three-row figure, config/figures.HEIGHT_MM),
# wide ones get a row of their own
BULK_ROW, BULK_WIDTHS = ("E", "F", "G"), (1.0, 2.3, 1.3)
LAYOUT = {
    "hammond": (((*BULK_ROW, "H"),), ((*BULK_WIDTHS, 0.8),)),
    "tms": ((BULK_ROW, ("H", "I")), (BULK_WIDTHS, (2, 1))),
    "soundlife": (((*BULK_ROW, "H", "I"),), ((*BULK_WIDTHS, 1, 1.1),)),
    "lupus": (((*BULK_ROW, "H"),), ((*BULK_WIDTHS, 2.2),)),
    "crohns": ((BULK_ROW, ("H", "I", "J")), (BULK_WIDTHS, (6.6, 1.6, 1.9))),
    "pineda": ((BULK_ROW, ("H",)), (BULK_WIDTHS, None)),
}
# The gaps between rows (fractions of the mean row height; common.draw), set so each is 4-7 mm: three rows, or four
# (pineda's H, with its rotated level names, needs more room above)
HSPACE = {"hammond": (0.45, 0.29), "soundlife": (0.45, 0.29), "lupus": (0.45, 0.29), "tms": (0.5, 0.33, 0.56),
          "crohns": (0.5, 0.33, 0.56), "pineda": (0.53, 0.35, 0.81)}
A_WIDTH = 3 + 2 * 0.8                  # a's three axes and their gaps (wspace 0.8), so b sits on the same grid
UMAP_ORDER_SEED = 0                    # D: the cells' drawing order
ROW_WSPACE = 0.8 / (2 + 0.8)            # the gap between a and b equals a's inner gaps


def key(fig, pair):
    """The figure's one key: the sides' colors and the pair's open/filled groups."""
    sides = [Patch(**style.side_bar(side), label=F.TEXT["side"][side]) for side in S.SIDE_COLORS]
    groups = [Line2D([], [], marker="o", linestyle="none", markersize=S.MARKER_SIZE, markeredgewidth=0.75, markeredgecolor=S.MODALITY_COLOR,
                     markerfacecolor=S.MODALITY_COLOR if S.GROUP_FILL[g] else "none", label=F.GROUP_NAMES[pair][g]) for g in (0, 1)]   # circles: not side swatches
    fig.legend(handles=sides + groups, loc="upper right", bbox_to_anchor=(0.99, 0.999), ncol=F.QC_KEY_COLUMNS, fontsize=S.MIN_FONT_SIZE,
               handlelength=1.0, columnspacing=1.0, borderaxespad=0)


TEMPLATE = {"A": panel_a, "B": panel_b, "C": panel_c, "D": panel_d, "E": panel_e, "F": panel_f, "G": panel_g}


def main(pairs, panels):
    for pair in pairs:
        item = F.QC_FIGURES[pair]
        rows, widths = LAYOUT[pair]
        layout = {"rows": (("A", "B"), ("C", "D"), *rows), "widths": ((A_WIDTH, 1), (1.3, 1.7), *widths), "ratios": (1, 1.2, *[1] * len(rows))}
        functions = {**TEMPLATE, "C": lambda f, c, pair, item: panel_c(f, c, pair, item, pairs),
                     **{panel: extra_panel(panel) for panel in F.QC_EXTRA[pair]}}
        common.draw(item, {p: (lambda f, c, _, fn=fn, pair=pair, item=item: fn(f, c, pair, item)) for p, fn in functions.items()}, pairs,
                    panels, __file__, legend=lambda fig, pair=pair: key(fig, pair), top_mm=12, hspace=HSPACE[pair],
                    wspace=(ROW_WSPACE, 0.3, *[0.6] * len(rows)), bottom=0.12,
                    left=0.1, **layout)
