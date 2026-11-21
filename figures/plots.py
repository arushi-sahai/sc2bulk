"""One plotting function per panel type, reused across figures so the conventions are identical by construction.
Drawing only: values arrive computed."""
import textwrap

import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec

import style
from sc2bulk.config import configurations
from sc2bulk.config import figures as F
from sc2bulk.config import statistics
from sc2bulk.config import style as S


def columns(fig, cell, n, wspace=0.15, gap=None):
    """n axes side by side in one gridspec cell (pairs as columns); gap: index before which a wider gap goes (the
    pooled column)."""
    if gap is None:
        grid = GridSpecFromSubplotSpec(1, n, subplot_spec=cell, wspace=wspace)
        return [fig.add_subplot(grid[0, i]) for i in range(n)]
    widths = [1] * gap + [0.25] + [1] * (n - gap)
    grid = GridSpecFromSubplotSpec(1, n + 1, subplot_spec=cell, wspace=wspace, width_ratios=widths)
    return [fig.add_subplot(grid[0, i if i < gap else i + 1]) for i in range(n)]


def auroc_ticks(ax, step=0.1):
    """An AUROC axis from chance (Figs 2b, 3c and S13's AUROC panels): ticks every step from 0.5 to 1, the bottom one,
    0.5, labeled instead of a chance line."""
    ticks = np.arange(0.5, 1 + step / 2, step)
    decimals = len(f"{step:g}".split(".")[1])
    ax.set_yticks(ticks, [f"{t:.{decimals}f}" for t in ticks])


def share_y(axes, limits=None):
    """Pairs as columns on one y scale (limits, or the widest the axes drew): tick labels on the first column only."""
    low, high = limits or (min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes))
    for i, ax in enumerate(axes):
        ax.set_ylim(low, high)
        if i:
            ax.tick_params(axis="y", labelleft=False)


def centered_xlabel(axes, text):
    """One x label centered under a row of pair columns (on the middle axes, or in the gap between the middle two)."""
    middle = axes[len(axes) // 2]
    box, before = middle.get_position(), axes[len(axes) // 2 - 1].get_position()
    middle.set_xlabel(text, x=-0.5 * (box.x0 - before.x1) / box.width if len(axes) % 2 == 0 else 0.5)


def difference_ticks(limits) -> tuple:
    """Three labelled ticks on a narrow difference axis: 0 and about 80% of each limit, to one decimal."""
    return round(0.8 * limits[0], 1), 0, round(0.8 * limits[1], 1)


def narrow(text, width=F.NARROW_TITLE) -> str:
    """A title wrapped, line by line, for a narrow column."""
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


def dodge(i, n, width):
    """The offset of the i-th of n overlapping points spread evenly over width (centered on 0)."""
    return (i - (n - 1) / 2) * width / max(n - 1, 1)


def difference_axis(ax, limits, margin=False, orientation="x"):
    """A difference-from-pseudobulk axis: fixed limits, the dashed blue 0 (pseudobulk's level), the margin band."""
    (ax.set_xlim if orientation == "x" else ax.set_ylim)(limits)
    (ax.axvline if orientation == "x" else ax.axhline)(0, **S.PSEUDOBULK_LINE, zorder=1)
    if margin:
        (ax.axvspan if orientation == "x" else ax.axhspan)(statistics.MARGIN, limits[1], **S.MARGIN_BAND, zorder=0)


def point(ax, x, y, limits, kwargs, orientation="x"):
    """A marker at (x, y) on a difference (or AUROC) axis with these limits (orientation: the axis it lies on); a value
    past a limit is an arrowhead at that edge instead (ax.plot kwargs give its color and fill). Returns the drawn (x, y)."""
    value = x if orientation == "x" else y
    assert np.isfinite(value), f"point: a non-finite value {value}"
    if limits[0] <= value <= limits[1]:
        ax.plot(x, y, **kwargs, zorder=3)
        return x, y
    past = value > limits[1]
    edge = limits[1] if past else limits[0]
    x, y = (edge, y) if orientation == "x" else (x, edge)
    marker = (">" if past else "<") if orientation == "x" else ("^" if past else "v")
    ax.plot(x, y, **{**kwargs, "marker": marker}, clip_on=False, zorder=3)
    return x, y


def reference_line(ax, value, limits, kwargs, x, zorder=1):
    """A horizontal reference line at value (kwargs: its style); past the y limits, an arrowhead at that edge at x instead."""
    if limits[0] <= value <= limits[1]:
        ax.axhline(value, **kwargs, zorder=zorder)
    else:
        point(ax, x, value, limits, {**kwargs, "markersize": S.MARKER_SIZE, "linestyle": "none"}, orientation="y")


def forest_row(ax, y, key, delta, low, high, limits, dots=(), glyph="", color=None, mark=None, width=S.LINE_WIDTH):
    """One row of a forest: seed dots, the CI (ends past the limits clipped with an arrowhead), the mean marker and
    the Holm glyph (in a marker column right of the axes). A NaN delta is "not estimable". A mean past a limit is an
    arrowhead at that edge and seed dots past it are left out (config/figures.XLIM_DIFFERENCE). color: drawn in this
    color instead of the key's; mark: the mean's ax.plot keywords instead of the key's marker; width: the CI's."""
    if not np.isfinite(delta):
        ax.text(0, y, F.FIGURE_TEXT["not_estimable"], ha="center", va="center", fontsize=S.MIN_FONT_SIZE)
        return
    paint = (lambda kwargs: style.recolored(kwargs, color)) if color else (lambda kwargs: kwargs)
    line = color or style.color(key)
    dots = [d for d in dots if limits[0] <= d <= limits[1]]
    if len(dots):
        ax.plot(dots, np.full(len(dots), y), **paint(style.seed_dots(key)), zorder=2)
    if np.isfinite(low):
        ax.plot([max(low, limits[0]), min(high, limits[1])], [y, y], color=line, linewidth=width, zorder=2)
        for end, edge, sign in ((low, limits[0], -1), (high, limits[1], 1)):
            if sign * (end - edge) > 0:
                ax.annotate("", xy=(edge, y), xytext=(edge - sign * 0.04 * (limits[1] - limits[0]), y),
                            arrowprops={"arrowstyle": "->", "linewidth": width, "color": line})
    point(ax, delta, y, limits, mark or paint(style.marker(key)))
    if glyph:
        ax.text(1.0, y, " " + glyph, transform=ax.get_yaxis_transform(), va="center", fontsize=S.MIN_FONT_SIZE)


def rows_axis(ax, labels, show):
    """Categorical rows top to bottom (labels on the leftmost column only)."""
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels if show else [])
    ax.tick_params(axis="y", length=0)


def slope(ax, rows, limits=None, spread=0.0):
    """Fig 2B's format: per row (key, left, right, left dots, right dots), two points at x = 0 (single-cell) and x = 1
    (bulk) in the legend's marker (hollow = pseudobulk input), joined by a line in the model's color; the figure's legend
    is the key. limits: the y range (Fig 6c), a mean past it an arrowhead at the edge and seed dots past it left out;
    spread: the rows dodged over this width at each end."""
    for i, (key, left, right, left_dots, right_dots) in enumerate(rows):
        shift = dodge(i, len(rows), spread)
        ax.plot([shift, 1 + shift], [left, right], color=style.color(key), linewidth=S.LINE_WIDTH, zorder=2)
        for x, dots in ((shift, left_dots), (1 + shift, right_dots)):
            dots = [d for d in dots if limits is None or limits[0] <= d <= limits[1]]
            ax.plot(np.full(len(dots), x), dots, **style.seed_dots(key))
        for x, y in ((shift, left), (1 + shift, right)):
            point(ax, x, y, limits or (-np.inf, np.inf), style.marker(key), orientation="y")
    ax.set_xlim(-0.35, 1.15)   # room for "single-cell"
    ax.set_xticks([0, 1])
    ax.set_xticklabels([F.TEXT["within"], F.TEXT["bulk"]])


CALLOUT_X, CALLOUT_TOP = 0.45, 0.97      # alignment_callout: the names' left edge (data) and first row (axes fraction)


def condition_x(i, j, n) -> float:
    """The x of the j-th of n conditions in the i-th configuration's group (offset_columns)."""
    return i + dodge(j, n, S.ALIGNMENT_SPREAD)


def offset_columns(ax, values, keys, alignments, limits=(-np.inf, np.inf)):
    """Fig 3b/c's format: each configuration's group (x = its position in keys; its marker names it), the alignments
    as dodged dots within it in their given order (named by alignment_callout), joined by a thin line in the
    configuration's color; dotted dividers between groups. values: {(key, alignment): value}; limits: the y range, a
    value past it an arrowhead at the edge (the line runs to the arrowhead)."""
    for i, key in enumerate(keys):
        drawn = [point(ax, condition_x(i, j, len(alignments)), values[(key, a)], limits, style.marker(key), orientation="y")
                 for j, a in enumerate(alignments)]
        ax.plot(*zip(*drawn), color=style.color(key), linewidth=S.TRACE_WIDTH, zorder=2)
        if i:
            ax.axvline(i - 0.5, **S.DIVIDER, zorder=1)
    ax.set_xticks([])
    ax.set_xlim(-0.5, len(keys) - 0.5)


def alignment_callout(ax, points, size=F.TEXT_SIZE):
    """The alignments named on the first group of an offset_columns row, as genes are on a volcano plot: a leader line from
    each dot (x, y, label) up to its name (size pt), stacked down from the top one line apart in the dots' left-to-right order
    so the lines don't cross."""
    step = (size + 0.5) / (ax.get_window_extent().height * 72 / ax.figure.dpi)
    for depth, (x, y, label) in enumerate(sorted(points)):
        ax.annotate(label, (x, y), xytext=(CALLOUT_X, CALLOUT_TOP - step * depth), textcoords=("data", "axes fraction"),
                    ha="left", va="center", fontsize=size, color="#4d4d4d", zorder=4,
                    bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none"},   # over the dividers
                    arrowprops={"arrowstyle": "-", "connectionstyle": "angle,angleA=0,angleB=90,rad=0", "linewidth": 0.5,
                                "color": "#4d4d4d", "shrinkA": 1, "shrinkB": 2})


def heatmap(ax, values, cmap, limits, row_labels, column_labels, na=(), text=True):
    """A heatmap with values printed; limits: one (low, high) or one per column (Fig 6A's phenotype scale); cells in na
    (row, column) hatched "n/a"; a missing (NaN) cell must be listed in na."""
    values = np.asarray(values, dtype=float)
    limits = np.array([limits] * values.shape[1] if np.ndim(limits) == 1 else limits, dtype=float)
    scaled = np.clip((values - limits[:, 0]) / (limits[:, 1] - limits[:, 0]), 0, 1)
    ax.imshow(np.ma.masked_invalid(scaled), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (r, c), v in np.ndenumerate(values):
        if (r, c) in na:
            style.na_cell(ax, c - 0.5, r - 0.5, 1, 1, F.FIGURE_TEXT["not_applicable"] if text else "")
            continue
        assert np.isfinite(v), f"heatmap cell ({row_labels[r]}, {column_labels[c]}) has no value and isn't listed as n/a"
        if text:
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=S.MIN_FONT_SIZE, color="white" if scaled[r, c] > 0.55 else "black")
    ax.set_xticks(range(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)


def forest_columns(fig, cell, values, rows, limits, na, margin=True, titles=None, labels=None, styles=None, ticks=None, wspace=0.15):
    """A forest per column (pairs, then "pooled" after a wider gap): values maps column -> {row: (delta, ci_low, ci_high,
    glyph)}; a row missing from a column must be in na (column, row), and is drawn "n/a". Values past the limits are
    clipped as in Fig 2 (forest_row). styles: row -> the configuration whose glyph it takes (default the row
    itself); ticks: the labelled x ticks. Returns the axes."""
    names = list(values)
    axes = columns(fig, cell, len(names), wspace=wspace, gap=names.index("pooled") if "pooled" in names else None)
    for i, (ax, column) in enumerate(zip(axes, names)):
        difference_axis(ax, limits, margin=margin)
        for y, row in enumerate(rows):
            if row not in values[column]:
                assert (column, row) in na, f"{column} {row}: missing, and not listed as n/a"
                style.na_cell(ax, limits[0], y - 0.45, limits[1] - limits[0], 0.9)
                continue
            delta, low, high, glyph = values[column][row]
            forest_row(ax, y, (styles or {}).get(row, row), delta, low, high, limits, (), glyph)
        rows_axis(ax, labels or rows, i == 0)
        ax.set_title((titles or {}).get(column, column))
        if ticks is not None:
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}".replace("-", "\N{MINUS SIGN}") for t in ticks])
    return axes


def group_bars(ax, counts, column, levels=None):
    """A descriptive figure's counts of a column's levels (x) per side and 0/1 group, side by side: the side's color,
    group 0 open, group 1 filled. counts: side, <column>, group, count; levels: the x order (default sorted)."""
    levels = [str(v) for v in levels] if levels is not None else sorted(counts[column].astype(str).unique())
    bars = [(side, g) for side in S.SIDE_COLORS if side in set(counts["side"]) for g in (0, 1)]
    width = 0.8 / len(bars)
    for i, (side, g) in enumerate(bars):
        part = counts[(counts["side"] == side) & (counts["group"] == g)]
        values = part.groupby(part[column].astype(str))["count"].sum().reindex(levels, fill_value=0)
        ax.bar(np.arange(len(levels)) + (i - (len(bars) - 1) / 2) * width, values.to_numpy(), width=width, **style.side_bar(side, g))
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, rotation=90)


def level_axis(ax, label, levels, rotate=True):
    """A group_bars axis's level names, flat and wrapped when they fit (measured: neighbours at least
    config/figures.LEVEL_GAP_MM apart), else (rotate, at most config/figures.MAX_ROTATED_LEVELS levels) at 45 degrees;
    otherwise the tick labels go and the x label states the number of levels (config/figures.LEVELS_LABEL)."""
    def fits(width):
        names = [textwrap.fill(str(v).replace("_", " "), width, break_long_words=False) for v in levels]
        ax.set_xticklabels(names, rotation=0)
        boxes = [t.get_window_extent(ax.figure.canvas.get_renderer()) for t in ax.get_xticklabels()]
        gap = F.LEVEL_GAP_MM / 25.4 * ax.figure.dpi
        return names, all(left.x1 + gap < right.x0 for left, right in zip(boxes, boxes[1:]))
    names, flat = fits(F.LEVEL_WRAP)
    if not flat:   # a narrower wrap (at spaces and hyphens) before rotating
        names, flat = fits(F.LEVEL_WRAP // 2)
    label = textwrap.fill(label.replace("_", " "), F.LEVEL_WRAP, break_long_words=False)
    if flat:
        ax.set_xticklabels(names, rotation=0)
    elif rotate and len(names) <= F.MAX_ROTATED_LEVELS:
        ax.set_xticklabels(names, rotation=45, ha="right")
    else:
        ax.set_xticks([])
        label = F.LEVELS_LABEL.format(label=label, n=len(names))
    ax.set_xlabel(label)
