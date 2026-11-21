"""Matplotlib settings and glyphs built from config/style.py and config/configurations.py: one encoding per meaning, the same in every figure."""
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory

from sc2bulk.config import configurations
from sc2bulk.config import figures as F
from sc2bulk.config import style as S


# Blinded figures carry a light corner stamp; pytest's say SYNTHETIC. Kept here, not in
# config/style.py: only the firewall's listed files may name the blinding switch (tests/test_permutation.py).
BLINDED_WATERMARK = {"text": "BLINDED: permuted bulk labels", "color": "#9e9e9e", "alpha": 0.8, "size": S.MIN_FONT_SIZE}
SYNTHETIC_WATERMARK = {"text": "SYNTHETIC", "color": "#9e9e9e", "alpha": 0.8, "size": S.MIN_FONT_SIZE}
# a figure drawn from a gather run without the Geneformer deletion: bottom left, clear of the stamp above
PENDING_WATERMARK = {"text": "Geneformer attribution pending", "color": "#9e9e9e", "alpha": 0.8, "size": S.MIN_FONT_SIZE}


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [S.FONT, *S.FONT_FALLBACKS], "font.size": S.FONT_SIZE,
        "axes.titlesize": S.FONT_SIZE, "axes.labelsize": S.FONT_SIZE, "xtick.labelsize": S.MIN_FONT_SIZE,
        "ytick.labelsize": S.MIN_FONT_SIZE, "legend.fontsize": S.MIN_FONT_SIZE, "pdf.fonttype": S.PDF_FONTTYPE,
        "ps.fonttype": S.PDF_FONTTYPE, "savefig.dpi": S.DPI, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "lines.linewidth": S.LINE_WIDTH,
        "patch.linewidth": 0.5, "hatch.linewidth": 0.3, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "image.interpolation": "nearest",
    })


def marker(key, size=S.MARKER_SIZE, mean=True) -> dict:
    """ax.plot keywords for a configuration's marker: color + marker = model, hollow = pseudobulk input (a mean marker:
    config/style.HOLLOW_MARKER; mean False, for seed dots and samples: open, thin outline); or a reference row's black X."""
    if key in configurations.CONFIGURATIONS:
        config = configurations.CONFIGURATIONS[key]
        hollow = config["input"] == "pseudobulk" and config["model"] != "pseudobulk"
        color = S.MODEL_COLORS[config["model"]]
        if hollow and mean:
            return {"marker": config["marker"], "color": color, "markerfacecolor": S.HOLLOW_MARKER["facecolor"], "markeredgecolor": color,
                    "markeredgewidth": S.HOLLOW_MARKER["linewidth"], "markersize": size * S.HOLLOW_MARKER["size_factor"], "linestyle": "none"}
        return {"marker": config["marker"], "color": color, "markerfacecolor": "none" if hollow else color,
                "markeredgecolor": color, "markeredgewidth": S.MARKER_EDGE, "markersize": size, "linestyle": "none"}
    return {"marker": S.REFERENCE_STYLE["marker"], "color": S.REFERENCE_STYLE["color"], "markerfacecolor": S.REFERENCE_STYLE["color"],
            "markeredgecolor": S.REFERENCE_STYLE["color"], "markeredgewidth": 0.5, "markersize": size, "linestyle": "none"}


def by_construction(key) -> dict:
    """A value that holds by construction (Fig 5D): the row's marker shape, open and gray, so it reads as a reference."""
    return {**marker(key), "color": S.BY_CONSTRUCTION_COLOR, "markerfacecolor": "none", "markeredgecolor": S.BY_CONSTRUCTION_COLOR}


# Fig. S10b's oracle candidate (a reference, never selected) and Fig 4a's published bag sizes (an open mark this much
# larger than the mean marker), both in the key
ORACLE = {"marker": "+", "color": "#000000", "markersize": S.MARKER_SIZE * 2, "markeredgewidth": 0.75, "linestyle": "none"}
# Fig 5b, Fig 6b and Fig. S14b/c's reference shares: all panel genes as a light vertical line behind the points (one per
# pair); genes matched on single-cell shift size (per configuration) as a small open gray triangle under its row's point
BASE_LINE = {"color": "#c8c8c8", "linewidth": 0.75}
BASE_MATCHED = {"marker": "^", "markerfacecolor": "none", "markeredgecolor": "#7a7a7a", "markeredgewidth": 0.5,
                "markersize": S.MARKER_SIZE * 0.8, "linestyle": "none"}


def recolored(kwargs, color) -> dict:
    """ax.plot keywords in another color (Fig 2a's gray reference rows), hollow markers kept hollow."""
    face = kwargs.get("markerfacecolor")
    return {**kwargs, "color": color, "markeredgecolor": color, **({"markerfacecolor": color} if face not in ("none", None, S.HOLLOW_MARKER["facecolor"]) else {})}


def label(key) -> str:
    """A configuration's display label (config/figures.ROW_LABELS over configurations.py's; figures/common.label)."""
    return F.ROW_LABELS.get(key, configurations.CONFIGURATIONS[key]["label"])


def seed_dots(key) -> dict:
    return {**marker(key, S.MARKER_SIZE * S.SEED_DOT_FACTOR, mean=False), "alpha": S.SEED_DOT_ALPHA}


def color(key) -> str:
    return S.MODEL_COLORS[configurations.CONFIGURATIONS[key]["model"]] if key in configurations.CONFIGURATIONS else S.REFERENCE_STYLE["color"]


def ring(ax, x, y, key):
    """The selected configuration: an open ring around its point, in the model's color."""
    ax.plot(x, y, marker=S.SELECTED_RING["marker"], markerfacecolor="none", markeredgecolor=color(key),
            markeredgewidth=S.SELECTED_RING["linewidth"], markersize=S.MARKER_SIZE * S.SELECTED_RING["size_factor"], linestyle="none")


def modality(side, group) -> dict:
    """A sample's glyph (Fig 3a, S13a-e): one circle shape, the side's data color, group 1 filled and group 0 open."""
    return {"marker": S.SAMPLE_MARKER, **side_points(side, int(group))}


def side_bar(side, group=1) -> dict:
    """A descriptive figure's bar (config/style.py's color rule): the side's color, open for group 0, filled for 1."""
    color = S.SIDE_COLORS[side]
    return {"facecolor": color if S.GROUP_FILL[group] else "none", "edgecolor": color, "linewidth": 0.75}


def side_points(side, group) -> dict:
    """A descriptive figure's scatter points: the side's color, open for group 0, filled for 1."""
    color = S.SIDE_COLORS[side]
    return {"facecolors": color if S.GROUP_FILL[group] else "none", "edgecolors": color, "linewidths": 0.5}


def cell_type_colors(type_lists) -> dict:
    """One palette color per cell-type name, the same in every list (each a pair's types, most abundant first). A list
    keeps its first len(CELL_TYPE_PALETTE) types (the rest are "other"); names in the most lists are colored first, each
    with the first color no other name of its lists has."""
    kept = [list(types)[:len(S.CELL_TYPE_PALETTE)] for types in type_lists]
    colors = {}
    for name in sorted({t for k in kept for t in k}, key=lambda t: (-sum(t in k for k in kept), t)):
        used = {colors[t] for k in kept if name in k for t in k if t in colors}
        free = [c for c in S.CELL_TYPE_PALETTE if c not in used]
        assert free, f"cell type {name}: every palette color is taken by a type it shares a pair with"
        colors[name] = free[0]
    return colors


def column_label(column) -> str:
    """A sample-table column's display label (config/figures.COLUMN_LABELS, else sentence case)."""
    return F.COLUMN_LABELS.get(column, column.replace("_", " ").capitalize())


def type_label(name) -> str:
    """A cell type's display name: its short form (config/figures.CELL_TYPE_SHORT), wrapped."""
    return textwrap.fill(F.CELL_TYPE_SHORT.get(name, name), F.TYPE_WRAP)


def panel_label(ax, panel, at_left=False, x=-0.02, y=1.02, left_mm=0, above_pt=None):
    """The panel letter (gid "panel_label": left out of the panel's own PNG, common.save); at_left: left_mm from the
    figure's left edge, clear of row labels and column titles; otherwise right-aligned at x (axes fraction). y: its bottom
    (axes fraction; default level with the axes' top); above_pt: instead, its bottom this many points over the axes' top
    (level with a pair_tag)."""
    at = left_mm / 25.4 / ax.figure.get_size_inches()[0]
    x, transform, ha = (at, blended_transform_factory(ax.figure.transFigure, ax.transAxes), "left") if at_left else (x, ax.transAxes, "right")
    letter = panel.lower() if F.PANEL_LABEL_CASE == "lower" else panel
    font = {"fontsize": S.PANEL_LABEL["size"], "fontweight": S.PANEL_LABEL["weight"], "gid": "panel_label"}
    if above_pt is None:
        text = ax.text(x, y, letter, transform=transform, ha=ha, va="bottom", **font)
    else:
        text = ax.annotate(letter, (x, 1.0), xycoords=transform, xytext=(0, above_pt), textcoords="offset points", ha=ha, va="bottom", **font)
    text.set_in_layout(False)


def panel_label_mm(fig, panel, x_mm, top_mm):
    """The panel letter at a fixed place: its top-left corner x_mm, top_mm from the figure's top-left corner (figures laid
    out in millimetres, so the letters sit on a grid: one baseline per row, each at its panel's left edge)."""
    width, height = fig.get_size_inches() * 25.4
    text = fig.text(x_mm / width, 1 - top_mm / height, panel.lower() if F.PANEL_LABEL_CASE == "lower" else panel, ha="left",
                    va="top", fontsize=S.PANEL_LABEL["size"], fontweight=S.PANEL_LABEL["weight"], gid="panel_label")
    text.set_in_layout(False)


def pair_tag(ax, text, above_pt):
    """The example pair's name beside a single-pair panel's letter, above_pt points over the axes' top."""
    ax.annotate(text, (0.03, 1.0), xycoords=blended_transform_factory(ax.figure.transFigure, ax.transAxes), xytext=(0, above_pt),
                textcoords="offset points", ha="left", va="bottom", fontsize=F.TEXT_SIZE, fontweight="bold")   # part of the panel's own PNG


def na_cell(ax, x, y, width, height, text=F.FIGURE_TEXT["not_applicable"]):
    """A not-applicable (or not-estimable) cell: hatched, with its word."""
    ax.add_patch(Rectangle((x, y), width, height, **S.NA_HATCH))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=S.MIN_FONT_SIZE)


def legend_strip(fig, keys, extras=(), y=0.995, ncol=6, loc="upper center", x=0.5, groups=None):
    """One key per figure, at the top and inside the graphic (Genome Biology): every configuration drawn, and the extra
    glyphs and lines used (the names in glyphs below, e.g. "reference", "ring", "pseudobulk_zero", "within_bulk"). loc: the key's
    point at (x, y) (e.g. "lower center" for a key placed above a row of panels); groups: the 0/1 groups' names for one
    pair (config/figures.GROUP_NAMES), else the generic names."""
    handles = [Line2D([], [], **marker(k), label=label(k)) for k in keys]
    groups = groups or F.TEXT["group"]
    glyphs = {"reference": Line2D([], [], **marker("reference"), label=F.TEXT["reference_row"]),
              "baseline_control": Line2D([], [], **recolored(marker("reference"), S.REFERENCE_GRAY), label=F.TEXT["baseline_control"]),
              "margin_band": Patch(**S.MARGIN_BAND, label=F.TEXT["margin_band"]),
              "ring": Line2D([], [], marker=S.SELECTED_RING["marker"], markerfacecolor="none", markeredgecolor="#4d4d4d",
                             markeredgewidth=S.SELECTED_RING["linewidth"], markersize=S.MARKER_SIZE * S.SELECTED_RING["size_factor"],
                             linestyle="none", label=F.TEXT["ring"]),
              "pseudobulk_zero": Line2D([], [], **S.PSEUDOBULK_LINE, label=F.TEXT["pseudobulk_zero"]),
              "within_bulk": Line2D([], [], **S.WITHIN_BULK_LINE, label=F.TEXT["within_bulk"]),
              "within_bulk_fig2": Line2D([], [], **S.WITHIN_BULK_LINE, label=F.TEXT["within_bulk_fig2"]),
              "chance": Line2D([], [], **S.CHANCE_LINE, label=F.TEXT["chance_line"]),
              "oracle": Line2D([], [], **ORACLE, label=F.FIGURE_TEXT["oracle"]),
              "pseudobulk_within_assay": Line2D([], [], **S.PSEUDOBULK_LINE, label=F.TEXT["pseudobulk_within_assay"]),
              "pseudobulk_bulk": Line2D([], [], **S.PSEUDOBULK_LINE, label=F.TEXT["pseudobulk_bulk"]),
              "pseudobulk_fig2": Line2D([], [], **S.PSEUDOBULK_LINE, label=F.TEXT["pseudobulk_fig2"]),
              "ensemble": Line2D([], [], color="#4d4d4d", linewidth=1.0, label=F.TEXT["ensemble"]),
              "by_construction": Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=S.BY_CONSTRUCTION_COLOR,
                                        markeredgewidth=S.MARKER_EDGE, markersize=S.MARKER_SIZE, label=F.TEXT["by_construction"]),
              "base_all": Line2D([], [], **BASE_LINE, label=F.TEXT["base_all"]),
              "base_matched": Line2D([], [], **BASE_MATCHED, label=F.TEXT["base_matched"]),
              "pseudobulk_n": Line2D([], [], color=color("pseudobulk"), marker="o", markersize=S.MARKER_SIZE * 0.7, label=F.TEXT["pseudobulk_n"]),
              **{f"side_{side}": Line2D([], [], marker=S.SAMPLE_MARKER, linestyle="none", markerfacecolor=c, markeredgecolor=c,
                                        markersize=S.MARKER_SIZE, label=F.TEXT["side"][side]) for side, c in S.SIDE_COLORS.items()},
              **{f"group_{g}": Line2D([], [], marker=S.SAMPLE_MARKER, linestyle="none", markeredgecolor=S.MODALITY_COLOR,
                                      markerfacecolor=S.MODALITY_COLOR if S.GROUP_FILL[g] else "none", markersize=S.MARKER_SIZE,
                                      markeredgewidth=0.5, label=groups[g])
                 for g in (0, 1)}}
    handles += [glyphs[e] for e in extras]
    fig.legend(handles=handles, loc=loc, bbox_to_anchor=(x, y), ncol=min(len(handles), ncol), handlelength=1.5,
               columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0)
