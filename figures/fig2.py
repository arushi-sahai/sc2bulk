"""Fig 2: the headline.

A: each alternative's difference in bulk AUROC from pseudobulk per pair (stage 11's per-pair rows) and pooled over the
   pairs where pseudobulk's bulk AUROC is below 1 - X (marked; K >= 2 "Average"; K = 1 the single pair; K = 0 no
   pooled column). Above each pair's column, a strip shows that AUROC against the 1 - X line,
   with the CI, the Holm glyphs, the margin band, the within-bulk reference's mark and the faint seed dots (encoder seeds;
   foundation models and pseudobulk: read-out seeds, paired by r), whose mean is asserted equal to stage 11's delta;
   below a thin rule, the gray ✕ reference rows (stage 11b's reference_rows and pseudobulk's label-permuted row).
B: within-single-cell vs. bulk AUROC (Fig 2B's encoder splits; human blood, aging: the folds' held-out donors).
C: the decomposition 0 -> (model, pseudobulk) -> (model, cells), asserted against stage 11's deltas.
The three panels share one column grid (pairs, a gap, the pooled column), each column titled in every panel; the
top legend is B's key (its markers; no end numbers). A and C clip at F.XLIM_DIFFERENCE's pre-declared range
(values past it: arrowheads).
Reads stage 11/11b's labelled tables only (never the labels file)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.transforms import ScaledTranslation

import common
import plots
import style
from sc2bulk.config import configurations
from sc2bulk.config import figures as F
from sc2bulk.config import style as S
from sc2bulk.config.pairs import PAIRS
from sc2bulk.config.statistics import MARGIN

FIGURE = "Fig 2"
READS_BULK_LABELS = False
READS_SCORES = True
TOLERANCE = 1e-9                       # stage 11's delta vs the scores table's seed differences
STRIP_MM = (1.5, 3.0)                  # Fig 2A's pseudobulk strip: gap above a column, and its height (one text line)
WSPACE = 0.2                           # between columns: room for the Holm glyphs just right of each column
GAP_WIDTH = 0.15                       # the gap before the pooled column (a pair column = 1)
REFERENCE_GAP = 1.0                    # A: extra rows' space between the configurations and the reference rows (the rule in it)

# Panel letters (x, top) in mm: at the figure's left edge, each level with the top line of its panel's column titles
LETTERS = {"A": (0.5, 16.9), "B": (0.5, 86.2), "C": (0.5, 127.5)}
WITHIN_LABEL_SHIFT_PT = 3.6             # B: "single-cell" nudged right, so it clears the column before's "bulk" as well as its own

def column_header(ax, column):
    """A column's title: a pair's three-line title (♦ if pooled; common.stacked_title), or the pooled header."""
    ax.set_title(common.stacked_title(column, mark=True) if column in PAIRS else common.pooled_header(), fontsize=F.FIG2_FONT_SIZE)


def strip(fig, axes, columns) -> list:
    """Above each column, its title; above each pair's column also pseudobulk's bulk AUROC, printed (the
    pooled pairs are marked ♦ in the titles). Returns the CSV rows."""
    gap, height = (mm / 25.4 / fig.get_size_inches()[1] for mm in STRIP_MM)
    kept, csv = common.kept_pairs(), []
    for i, (ax, column) in enumerate(zip(axes, columns)):
        box = ax.get_position()
        top = fig.add_axes([box.x0, box.y1 + gap, box.width, height])
        column_header(top, column)
        if column == "pooled":         # no strip: the pooled column's pairs
            top.axis("off")
            continue
        value = common.pseudobulk_auroc(column)
        printed = f"{value:.3f}"
        assert (float(printed) < 1 - MARGIN) == (value < 1 - MARGIN), f"{column}: {printed} reads on the wrong side of {1 - MARGIN:g}"
        top.text(0.5, 0, printed, ha="center", va="center", fontsize=F.FIG2_FONT_SIZE)
        top.set_xlim(0, 1)
        top.set_ylim(-0.5, 0.5)
        top.set_xticks([])
        top.set_yticks([0], [F.FIGURE_TEXT["pooled_strip"]] if i == 0 else [""])
        top.tick_params(length=0)
        for spine in top.spines.values():
            spine.set_visible(False)
        csv.append({"column": column, "row_type": "pseudobulk_auroc", "auroc": value, "pooled": column in kept})
    return csv


def grid(fig, cell, pairs):
    """The pairs' columns, the gap and the pooled column (switched off by the caller where unused): one grid for A-C."""
    n = len(pairs)
    cells = GridSpecFromSubplotSpec(1, n + 2, subplot_spec=cell, wspace=WSPACE, width_ratios=[1] * n + [GAP_WIDTH, 1])
    return [fig.add_subplot(cells[0, i]) for i in (*range(n), n + 1)]


def difference_axis(ax, margin=False):
    limits = F.XLIM_DIFFERENCE["Fig 2A"]
    plots.difference_axis(ax, limits, margin=margin)
    ticks = plots.difference_ticks(limits)
    ax.set_xticks(ticks, [f"{t:g}".replace("-", "−") for t in ticks])
    ax.set_xlim(limits)                # ticks past the limits would widen them
    return limits


def statistics_rows(column) -> pd.DataFrame:
    """Stage 11's primary rows for one column (a pair, or "pooled"), one per alternative (asserted)."""
    pool_set = "nonceiling_equal" if column == "pooled" else "per_pair"
    rows = common.pick(common.statistics(), neutral=False, pair=column, pool_set=pool_set, readout="ridge", statistic="auroc_difference")
    assert sorted(rows["key"]) == sorted(common.alternatives()), f"Fig 2A {column}: statistics rows {sorted(rows['key'])}"
    return rows.set_index("key")


def reference_rows(column) -> dict:
    """The ✕ rows for one column: key -> (delta, ci_low, ci_high, dots), each asserted formed."""
    sec = common.secondary()
    rows = common.pick(sec, neutral=False, pair=column, row="reference_rows", variant="primary", bulk_subset="all").set_index("key")
    permuted = common.pick(sec, neutral=False, pair=column, row="label_permuted", key="pseudobulk")
    kept = common.kept_pairs()
    out = {}
    for key in F.FIG2A_REFERENCE_ROWS:
        if key == F.LABEL_PERMUTED:
            mean = permuted[permuted["statistic"] == "permuted_minus_primary_mean"]
            assert len(mean) == (1 if column != "pooled" or kept else 0), f"Fig 2A {column}: the label-permuted row"
            dots = permuted.loc[permuted["statistic"] == "permuted_minus_primary", "delta"].to_numpy()
            if len(mean):
                out[key] = (float(mean["delta"].iloc[0]), np.nan, np.nan, dots)
            continue
        assert (key in rows.index) == (column != "pooled" or bool(kept)), f"Fig 2A {column} {key}: present {key in rows.index}"
        if key in rows.index:
            r = rows.loc[key]
            out[key] = (float(r["delta"]), float(r["ci_low"]), float(r["ci_high"]), [])
    if column == "pooled" and not kept:   # K = 0: 11b forms no pooled reference or label-permuted row
        assert not out
    return out


def seed_dots(pair) -> dict:
    """key -> the per-seed differences from pseudobulk (scores table), their mean asserted equal to stage 11's delta."""
    rows = statistics_rows(pair)
    pseudobulk = common.aurocs(pair, "pseudobulk", **common.headline_conditions("pseudobulk"))
    out = {}
    for key in common.alternatives():
        dots = common.seed_differences(pair, key, pseudobulk, **common.headline_conditions(key))
        assert abs(dots.mean() - rows.loc[key, "delta"]) < TOLERANCE, f"Fig 2A {pair} {key}: seed dots' mean {dots.mean()} != delta {rows.loc[key, 'delta']}"
        out[key] = dots
    return out


def reference_y(keys) -> list:
    """The reference rows' y positions: below the configurations, past REFERENCE_GAP."""
    return [len(keys) + REFERENCE_GAP + j for j in range(len(F.FIG2A_REFERENCE_ROWS))]


def panel_a(fig, cell, pairs):
    header = common.pooled_header()
    columns = [*pairs, "pooled"] if header else list(pairs)
    axes = grid(fig, cell, pairs)
    keys = common.alternatives()
    labels = [common.label(k) for k in keys] + [common.label(k) for k in F.FIG2A_REFERENCE_ROWS]
    ref_y = reference_y(keys)
    rule = len(keys) - 0.5 + REFERENCE_GAP / 2   # the thin rule between the configurations and the reference rows
    within = common.pick(common.secondary(), neutral=False, row="within_bulk_reference", key="within_bulk_reference").set_index("pair")
    csv = []
    for i, (ax, column) in enumerate(zip(axes, columns)):
        rows = statistics_rows(column)
        dots = seed_dots(column) if column != "pooled" else {}
        limits = difference_axis(ax, margin=True)
        for y, key in enumerate(keys):
            r = rows.loc[key]
            glyph = common.glyph(r["p_better_holm"], r["p_worse_holm"])
            plots.forest_row(ax, y, key, r["delta"], r["ci_low"], r["ci_high"], limits, dots.get(key, []))
            ax.annotate(glyph, (1.0, y), xycoords=("axes fraction", "data"), xytext=(1, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=S.MIN_FONT_SIZE, annotation_clip=False)   # hugging the column's right edge
            csv.append({"column": column, "row_type": "configuration", "key": key, "delta": r["delta"], "ci_low": r["ci_low"],
                        "ci_high": r["ci_high"], "u": r["u"], "holm_better": r["p_better_holm"], "holm_worse": r["p_worse_holm"]})
            csv += [{"column": column, "row_type": "seed", "key": key, "seed": int(s), "delta_seed": d} for s, d in dots.get(key, pd.Series()).items()]
        mark = within.loc[column, "delta"]
        if limits[0] <= mark <= limits[1]:
            ax.plot([mark, mark], [-0.5, len(keys) - 0.5], **S.WITHIN_BULK_LINE)
        else:
            plots.point(ax, mark, -0.5, limits, {**S.WITHIN_BULK_LINE, "markersize": S.MARKER_SIZE, "linestyle": "none"})
        csv.append({"column": column, "row_type": "within_bulk", "key": "within_bulk_reference", "delta": mark})
        references = reference_rows(column)
        ax.axhline(rule, **S.REFERENCE_RULE)
        for key, y in zip(F.FIG2A_REFERENCE_ROWS, ref_y):
            delta, low, high, ref_dots = references[key]
            plots.forest_row(ax, y, "reference", delta, low, high, limits, ref_dots, color=S.REFERENCE_GRAY)
            csv.append({"column": column, "row_type": "permuted" if key == F.LABEL_PERMUTED else "reference", "key": key, "delta": delta,
                        "ci_low": low, "ci_high": high})
            csv += [{"column": column, "row_type": "permuted", "key": key, "seed": j_, "delta_seed": d} for j_, d in enumerate(ref_dots)]
        ax.set_ylim(ref_y[-1] + 0.5, -0.6)
        ax.set_yticks([*range(len(keys)), *ref_y])
        ax.set_yticklabels(labels if i == 0 else [])
        ax.tick_params(axis="y", length=0)
    csv += strip(fig, axes, columns)
    if not header:   # K = 0: the pooled column's place carries the sentence
        axes[-1].axis("off")
        axes[-1].text(0.5, 0.5, F.FIGURE_TEXT["pooled_k0"].format(threshold=common.threshold()), rotation=90, ha="center", va="center", transform=axes[-1].transAxes,
                      fontsize=S.MIN_FONT_SIZE)
    pooled_references = common.pick(common.secondary(), neutral=False, pair="pooled", row=("reference_rows", "label_permuted"))
    assert pooled_references.empty == (not header), f"Fig 2A: 11b's pooled reference rows {len(pooled_references)} with K = {len(common.kept_pairs())}"
    plots.centered_xlabel(axes, F.TEXT["difference"])
    style.panel_label_mm(fig, "A", *LETTERS["A"])
    common.panel_csv(pd.DataFrame(csv), FIGURE, "A")


def panel_b(fig, cell, pairs):
    axes = grid(fig, cell, pairs)
    axes[-1].axis("off")
    csv = []
    for i, (ax, pair) in enumerate(zip(axes, pairs)):
        column_header(ax, pair)
        rows = []
        for key in common.kept():
            within_seeds = common.seeds_of(key) if common.within_side(pair) == "single_cell_heldout" else common.encoder_seeds()
            within = common.aurocs(pair, key, within_seeds, side=common.within_side(pair), **common.headline_conditions(key))
            bulk = common.aurocs(pair, key, **common.headline_conditions(key))
            left, right = common.mean_over_seeds(within), common.mean_over_seeds(bulk)
            left_dots, right_dots = (common.per_seed(t).groupby("encoder_seed")["auroc"].mean() for t in (within, bulk))
            rows.append((key, left, right, left_dots.to_numpy(), right_dots.to_numpy()))
            csv.append({"pair": pair, "key": key, "within_auroc": left, "bulk_auroc": right})
            csv += [{"pair": pair, "key": key, "side": side, "seed": int(s), "auroc": a}
                    for side, d in (("within", left_dots), ("bulk", right_dots)) for s, a in d.items()]
        plots.slope(ax, rows, limits=F.AUROC_RANGE)   # chance to 1: a mean below is an arrowhead, seeds below are left out
        reference = common.mean_over_seeds(common.aurocs(pair, "within_bulk_reference", [-1], row_kind="reference", n_cells=0, pooling="native"))
        ax.axhline(rows[0][2], **S.PSEUDOBULK_LINE, zorder=1)
        ax.axhline(reference, **S.WITHIN_BULK_LINE, zorder=1)
        if i == len(pairs) - 1:   # the two lines named at the right of the last pair (the pooled column is empty in b),
            names = sorted(((rows[0][2], F.TEXT["pseudobulk_bulk"], S.PSEUDOBULK_LINE["color"]),   # the lower below its line,
                            (reference, F.TEXT["within_bulk_fig2"], S.WITHIN_BULK_LINE["color"])))   # the upper above, so they never meet
            for (y, label, color), va in zip(names, ("top", "bottom")):
                ax.annotate(label, (1.0, y), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                            ha="left", va=va, fontsize=F.FIG2_FONT_SIZE, color=color, annotation_clip=False)
        csv.append({"pair": pair, "key": "within_bulk_reference", "bulk_auroc": reference})
        ax.set_ylim(*F.AUROC_RANGE)
        plots.auroc_ticks(ax)
        ax.set_xticklabels([F.TEXT["within"], F.TEXT["bulk"]])
        within_label = ax.get_xticklabels()[0]
        within_label.set_transform(within_label.get_transform() + ScaledTranslation(WITHIN_LABEL_SHIFT_PT / 72, 0, fig.dpi_scale_trans))
        if i == 0:
            ax.set_ylabel(F.TEXT["auroc"])
        else:
            ax.set_yticklabels([])
    style.panel_label_mm(fig, "B", *LETTERS["B"])
    common.panel_csv(pd.DataFrame(csv), FIGURE, "B")


def decomposition(column) -> list:
    """(model, delta(model, pseudobulk), delta(model, cells)) for the models 11b decomposed, asserted."""
    stats = statistics_rows(column)
    rows = common.pick(common.secondary(), neutral=False, pair=column, row="decomposition").set_index("key")
    models = [m for m in F.FIG2C_MODELS if f"{m}_cells" in common.alternatives() and f"{m}_pseudobulk" in common.alternatives()]
    assert sorted(rows.index) == sorted(f"{m}_cells" for m in models), f"Fig 2C {column}: decomposed {sorted(rows.index)}"
    out = []
    for m in models:
        profile, cells = stats.loc[f"{m}_pseudobulk", "delta"], stats.loc[f"{m}_cells", "delta"]
        assert abs(profile + rows.loc[f"{m}_cells", "delta"] - cells) < TOLERANCE, f"Fig 2C {column} {m}: steps don't meet stage 11's delta"
        out.append((m, profile, cells))
    return out


def panel_c(fig, cell, pairs):
    header = common.pooled_header()
    columns = [*pairs, "pooled"] if header else list(pairs)
    axes = grid(fig, cell, pairs)
    if not header:
        axes[-1].axis("off")
    csv = []
    for i, (ax, column) in enumerate(zip(axes, columns)):
        steps = decomposition(column)
        limits = difference_axis(ax)
        for y, (m, profile, cells) in enumerate(steps):
            color = style.color(f"{m}_cells")   # a path from pseudobulk (0) to the model on the summed profile (open) to it on cells (filled)
            ax.plot(np.clip([0, profile, cells], *limits), [y] * 3, color=color, linewidth=S.LINE_WIDTH, zorder=2)
            for value, key in ((profile, f"{m}_pseudobulk"), (cells, f"{m}_cells")):   # past the axis: an arrowhead at its edge
                plots.point(ax, value, y, limits, style.marker(key))
            csv += [{"column": column, "model": m, "step": "pseudobulk_input", "start": 0.0, "end": profile},
                    {"column": column, "model": m, "step": "cells", "start": profile, "end": cells}]
        plots.rows_axis(ax, [configurations.MODELS[m]["label"] for m, _, _ in steps], i == 0)
        column_header(ax, column)
    plots.centered_xlabel(axes, F.TEXT["difference"])
    style.panel_label_mm(fig, "C", *LETTERS["C"])
    common.panel_csv(pd.DataFrame(csv), FIGURE, "C")


def legend(fig):
    style.legend_strip(fig, common.kept(), ("baseline_control", "margin_band", "pseudobulk_fig2", "within_bulk_fig2"), ncol=5)


def main(pairs, panels):
    sizes = {"xtick.labelsize": F.FIG2_FONT_SIZE, "ytick.labelsize": F.FIG2_FONT_SIZE, "legend.fontsize": F.FIG2_FONT_SIZE}
    with plt.rc_context(sizes):
        common.draw(FIGURE, {"A": panel_a, "B": panel_b, "C": panel_c}, pairs, panels, __file__, ratios=(3.1, 1.6, 0.7), legend=legend,
                    top_mm=31, bottom=0.0713, left=0.195, right=0.96, hspace=(0.977, 0.834))
