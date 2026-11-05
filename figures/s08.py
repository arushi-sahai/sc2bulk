"""Fig. S8: characteristics of the dataset pairs (stage 04b and 04d's
tables). ~170 x 200 mm, three rows.

A every Fig 1E characteristic (config/fig1e.FIG1E_COLUMNS) per pair, column names along the top; B cell-type
composition, per pair: the single-cell side and the bulk estimated by deconvolution (stage 04b: the training samples'
annotated fractions and the scored bulk's estimated ones; human blood, aging: mean over its four folds); mouse
microglia has no deconvolution (one cell type), so its single-cell bar is stage 04d's composition and its bulk bar n/a;
C tissue-difference markers (mean log1p CPM per side); D batch x group on the primary unit's single-cell training
samples; E samples per side by group (single-cell groups, and bulk groups through sc2bulk.labels); F samples per organ
and side (mouse organs). Mouse microglia has no markers (one cell type on both sides; the caption says so).

A descriptive figure: data colors only (config/style.py's color rule): sides in SIDE_COLORS, cell types in the shared
cell-type colors (common.cell_type_colors), groups open (0) vs. filled (1)."""
import textwrap

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import common
import plots
import style
from sc2bulk.config import deconvolution, fig1e
from sc2bulk.config import figures as F
from sc2bulk.config import style as S
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIRS

FIGURE = "Fig. S8"
# Panel letters (x, top) in mm: one baseline per row, each at its panel's left edge (as Fig 1)
LETTERS = {"A": (0, 1), "B": (0, 29), "C": (100, 29), "D": (0, 129.5), "E": (93, 129.5), "F": (141, 129.5)}
READS_BULK_LABELS = True               # E's bulk groups
READS_SCORES = False
STATE_GLYPH = {"yes": S.MODALITY_COLOR, "not determinable": "none"}
HEADER_WRAP = 10                       # characters per A header line
A_INDENT = 0.088                       # A starts right of the left margin, room for its full pair names
SIDES = ("single_cell", "bulk")


def characteristics(pair) -> pd.DataFrame:
    return common.read_csv("04d_characteristics", pair, "characteristics.csv", keep_default_na=False)


def panel_a(fig, cell, pairs):
    ax = fig.add_subplot(cell)
    box = ax.get_position()
    ax.set_position([box.x0 + A_INDENT, box.y0, box.width - A_INDENT, box.height])   # A spans to the right edge
    frames = []
    for y, pair in enumerate(pairs):
        table = characteristics(pair).set_index("column").loc[list(fig1e.FIG1E_COLUMNS)]
        for x, (column, r) in enumerate(table.iterrows()):
            if r["state"] != "no":
                ax.plot(x, y, marker="o", markersize=S.MARKER_SIZE, markeredgecolor=S.MODALITY_COLOR,
                        markerfacecolor=STATE_GLYPH[r["state"]], linestyle="none")
        frames.append(table.reset_index())
    ax.set_xlim(-0.5, len(fig1e.FIG1E_COLUMNS) - 0.5)
    ax.set_xticks(range(len(fig1e.FIG1E_COLUMNS)))
    ax.set_xticklabels([textwrap.fill(F.FIG1E_HEADERS.get(c, c), HEADER_WRAP) for c in fig1e.FIG1E_COLUMNS])
    ax.xaxis.tick_top()
    plots.rows_axis(ax, [PAIRS[p]["name"] for p in pairs], True)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    style.panel_label_mm(ax.figure, "A", *LETTERS["A"])
    common.panel_csv(pd.concat(frames, ignore_index=True)[["pair", "column", "state", "value", "annotation"]], FIGURE, "A")


def composition(pair) -> pd.DataFrame:
    """side, type, fraction, source: 04b's deconvolution sides, or (no deconvolution) 04d's single-cell composition."""
    if deconvolution.LEVELS[pair] is not None:
        return common.deconvolution_composition(pair).assign(source="04b_deconvolve")
    table = common.read_csv("04d_characteristics", pair, "composition.csv", dtype={"sample_id": str})
    mean = table.groupby("cell_type")["fraction"].sum() / table["sample_id"].nunique()
    return mean.rename("fraction").rename_axis("type").reset_index().assign(side="single_cell", source="04d_characteristics")


def panel_b(fig, cell, pairs):
    grid = cell.subgridspec(len(pairs), 2, width_ratios=(1, 1.3), hspace=1.33)   # rows far enough apart for the keys
    colors = common.cell_type_colors(pairs)
    frames = []
    for i, pair in enumerate(pairs):
        ax = fig.add_subplot(grid[i, 0])
        table = composition(pair)
        single = table[table["side"] == "single_cell"].set_index("type")["fraction"]
        kept = sorted(single.index, key=lambda t: (-single[t], t))[:len(S.CELL_TYPE_PALETTE)]   # the rest are "other"
        table = table.assign(shown=table["type"].where(table["type"].isin(kept), S.OTHER_TYPE["name"]))
        shown = [*kept, *([S.OTHER_TYPE["name"]] if (table["shown"] == S.OTHER_TYPE["name"]).any() else [])]
        palette = {**colors, S.OTHER_TYPE["name"]: S.OTHER_TYPE["color"]}
        for y, side in enumerate(SIDES):
            part = table[table["side"] == side].groupby("shown")["fraction"].sum()
            if part.empty:
                assert deconvolution.LEVELS[pair] is None and side == "bulk", f"S8B {pair}: no {side} fractions"
                style.na_cell(ax, 0, y - 0.4, 1, 0.8)
                continue
            left = 0.0
            for t in [t for t in shown if t in part.index]:
                ax.barh(y, part[t], left=left, height=0.8, color=palette[t], linewidth=0)
                left += part[t]
        ax.set_xlim(0, 1)
        plots.rows_axis(ax, [F.TEXT["side"][s] for s in SIDES], True)
        ax.set_xticks([0, 0.5, 1] if i == len(pairs) - 1 else [])
        ax.set_title(PAIRS[pair]["name"], fontsize=S.MIN_FONT_SIZE, loc="left")
        handles = [Patch(facecolor=palette[t], linewidth=0, label=style.type_label(t))
                   for t in shown]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.03, 0.5), ncol=2, fontsize=S.MIN_FONT_SIZE, handlelength=0.8,
                  handletextpad=0.3, columnspacing=0.6, labelspacing=0.08, borderaxespad=0)
        if i == 0:
            style.panel_label_mm(ax.figure, "B", *LETTERS["B"])
        frames.append(table.assign(pair=pair))
    ax.set_xlabel(F.TEXT["fraction_of_cells"])
    common.panel_csv(pd.concat(frames, ignore_index=True), FIGURE, "B")


def not_applicable(fig, cell, panel):
    """A panel none of the drawn pairs has (a pair-subset run): n/a, stated in its CSV."""
    ax = fig.add_subplot(cell)
    ax.axis("off")
    style.na_cell(ax, 0, 0, 1, 1)
    style.panel_label(ax, panel)
    common.panel_csv(pd.DataFrame([{"note": F.FIGURE_TEXT["not_applicable"]}]), FIGURE, panel)


def legend_handles() -> list:
    """The figure's one data-color key (drawn in E, which uses both): the sides and the open/filled groups."""
    sides = [Patch(**style.side_bar(side), label=F.TEXT["side"][side]) for side in SIDES]
    groups = [Patch(facecolor=S.MODALITY_COLOR if S.GROUP_FILL[g] else "none", edgecolor=S.MODALITY_COLOR, linewidth=0.75,
                    label=F.GROUP_NAMES_ALL[g]) for g in (0, 1)]
    return sides + groups


def panel_c(fig, cell, pairs):
    with_markers = [p for p in pairs if p in fig1e.TISSUE_MARKERS]
    if not with_markers:
        return not_applicable(fig, cell, "C")
    tables = {}
    for pair in with_markers:
        table = common.read_csv("04d_characteristics", pair, "markers.csv")
        tables[pair] = table[table["present"]]
    columns = 2 if len(with_markers) > 1 else 1   # the pairs row-major in a 2-column grid, rows as tall as their longest list
    rows = [with_markers[i:i + columns] for i in range(0, len(with_markers), columns)]
    grid = cell.subgridspec(len(rows), columns, height_ratios=[max(tables[p]["gene"].nunique() for p in r) for r in rows],
                            hspace=0.6, wspace=0.7)
    limit = 1.05 * max(t["mean_log1p_cpm"].max() for t in tables.values())   # one x range across the pairs
    axes = []
    for i, pair in enumerate(with_markers):
        ax = fig.add_subplot(grid[i // columns, i % columns])
        axes.append(ax)
        ax.set_xlim(0, limit)
        table = tables[pair]
        genes = list(dict.fromkeys(table["gene"]))
        for j, side in enumerate(SIDES):
            values = table[table["side"] == side].set_index("gene").loc[genes, "mean_log1p_cpm"]
            ax.barh(np.arange(len(genes)) + (j - 0.5) * 0.4, values.to_numpy(), height=0.4, **style.side_bar(side))
        plots.rows_axis(ax, genes, True)
        ax.set_title(PAIRS[pair]["name"], fontsize=S.MIN_FONT_SIZE, loc="left")
        ax.tick_params(axis="y", labelsize=S.MIN_FONT_SIZE)
        if i == 0:
            style.panel_label_mm(ax.figure, "C", *LETTERS["C"])
    plots.centered_xlabel(axes[-len(rows[-1]):], F.TEXT["log1p_cpm"])   # under the bottom row's columns
    common.panel_csv(pd.concat([tables[p].assign(pair=p) for p in with_markers], ignore_index=True), FIGURE, "C")


def panel_d(fig, cell, pairs):
    with_batch = [p for p in pairs if fig1e.BATCH[p] is not None]
    axes = plots.columns(fig, cell, len(with_batch), wspace=0.25)
    frames = []
    for ax, pair in zip(axes, with_batch):
        table = common.read_csv("04d_characteristics", pair, "batch_label.csv", dtype={"batch": str})
        table = table[table["level"] == "sample"].assign(side="single_cell")
        plots.group_bars(ax, table, "batch")
        ax.set_title("\n".join([*textwrap.wrap(PAIRS[pair]["title"][0], 9), PAIRS[pair]["title"][1]]), fontsize=S.MIN_FONT_SIZE)
        plots.level_axis(ax, style.column_label(fig1e.BATCH[pair]), sorted(table["batch"].astype(str).unique()), rotate=False)   # group_bars' order
        ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        if ax is not axes[0]:
            ax.sharey(axes[0])
            ax.tick_params(axis="y", labelleft=False)
        frames.append(table.assign(pair=pair))
    axes[0].set_ylabel(F.TEXT["count"])
    style.panel_label_mm(axes[0].figure, "D", *LETTERS["D"])
    common.panel_csv(pd.concat(frames, ignore_index=True), FIGURE, "D")


def panel_e(fig, cell, pairs):
    ax = fig.add_subplot(cell)
    rows = []
    for pair in pairs:
        single = common.single_cell_samples(pair)["group"].value_counts()
        bulk = common.bulk_labels(pair)["group"].value_counts()
        rows += [{"pair": pair, "side": side, "group": g, "count": int(counts.get(g, 0))}
                 for side, counts in (("single_cell", single), ("bulk", bulk)) for g in (0, 1)]
    frame = pd.DataFrame(rows)
    plots.group_bars(ax, frame, "pair", levels=list(pairs))
    ax.set_xticklabels([PAIRS[p]["name"] for p in pairs], rotation=45, ha="right")
    ax.set_ylabel(F.TEXT["count"])
    ax.legend(handles=legend_handles(), loc="lower left", bbox_to_anchor=(0.05, 1.02), ncol=2, fontsize=S.MIN_FONT_SIZE,
              handlelength=1.0, columnspacing=0.8, borderaxespad=0)
    style.panel_label_mm(ax.figure, "E", *LETTERS["E"])
    common.panel_csv(frame.rename(columns={"count": "samples"}), FIGURE, "E")


def panel_f(fig, cell, pairs):
    with_organs = [p for p in pairs if p in T.ORGAN_STRATUM]
    if not with_organs:
        return not_applicable(fig, cell, "F")
    axes = plots.columns(fig, cell, len(with_organs))
    rows = []
    for ax, pair in zip(axes, with_organs):
        single = common.single_cell_samples(pair)[T.ORGAN_STRATUM[pair]].value_counts()
        bulk = common.bulk_samples(pair)[T.ORGAN_STRATUM[pair]].value_counts()
        organs = sorted(set(single.index) | set(bulk.index))
        for j, (side, counts) in enumerate((("single_cell", single), ("bulk", bulk))):
            ax.bar(np.arange(len(organs)) + (j - 0.5) * 0.4, counts.reindex(organs, fill_value=0).to_numpy(), width=0.4,
                   **style.side_bar(side))
            rows += [{"pair": pair, "side": side, "organ": o, "samples": int(counts.get(o, 0))} for o in organs]
        ax.set_xticks(range(len(organs)))
        ax.set_xticklabels([o.replace("_", " ") for o in organs], rotation=45, ha="right")
        ax.set_title(common.pair_title(pair), fontsize=S.MIN_FONT_SIZE)
    style.panel_label_mm(axes[0].figure, "F", *LETTERS["F"])
    common.panel_csv(pd.DataFrame(rows), FIGURE, "F")


def main(pairs, panels):
    common.draw(FIGURE, {"A": panel_a, "B": panel_b, "C": panel_c, "D": panel_d, "E": panel_e, "F": panel_f}, pairs, panels, __file__,
                rows=(("A",), ("B", "C"), ("D", "E", "F")), widths=(None, (1.5, 1), (2.4, 1.2, 0.7)), ratios=(12.8, 78.2, 24.1),
                top_mm=12, hspace=(0.253, 0.678), bottom=0.14, wspace=0.35, left=0.075)
