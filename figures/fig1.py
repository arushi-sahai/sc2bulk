"""Fig 1 (label-free). A and B: the schematics (figures/schematics.py; they replaced
the BioRender drafts), A across the top and B beside C.

C: the ten configurations as a model x input grid, drawn from config (the design; no stage output). Rows are the seven
models in the family order (config/figures.MODEL_ORDER), each tinted in its family color and grouped by a family label;
columns are cells (bags of 500) and the pseudobulk profile (hatched); "—" where the model has no such configuration. The
test input and the read-out are stated once under the grid.

D: the dataset pairs, one row per pair (its name gives the species): the single-cell side | connector | the bulk side,
each side's first line its tissue or preparation and assay tag (outlined in the side's data color, config/style.py's
color rule), its second the groups compared (config/figures.FIG1D). The connector comes from 04d's "Independent cohorts" state, so D and E
can't disagree: solid = the same donors (cross-fitted), dotted = the same colony, dashed = independent.

E (beside D, its rows aligned with D's): the five main characteristics (config/fig1e.FIG1E_MAIN): a filled circle =
yes, an open circle = not determinable, blank = no. Thresholds and full header wording are in the legend; the 04d
annotations are in the CSV only (Table S1).

Laid out in millimetres at print size: C ~80 x 55 mm; D | E 170 x 70 mm together, D wider; every
text >= config/figures.FIG1_FONT_SIZE (7 pt). Each panel's own PNG (common.save) is the panel cropped from the
composite, without its letter (the assembly adds the letters)."""
import textwrap

import pandas as pd
from matplotlib.patches import Rectangle

import common
import schematics
import style
from sc2bulk.config import configurations, fig1e
from sc2bulk.config import figures as F
from sc2bulk.config import style as S
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIRS

FIGURE = "Fig 1"
READS_BULK_LABELS = False
READS_SCORES = False
STATES = ("yes", "not determinable", "no")
SIZE = F.FIG1_FONT_SIZE
# Axes rectangles in mm from the composite's top-left corner: (left, top, width, height). Labels sit outside them.
A_AXES = (0, 0, *schematics.SIZES[schematics.fig1a])   # the schematics, in mm at print size
B_AXES = (0, 46, *schematics.SIZES[schematics.fig1b])
C_AXES = (110, 57, 34, 35)              # the grid; model names to its left, family labels to its right, the note below
D_AXES = (17, 109, 94, 45)             # x in mm within the axes; pair names to its left
E_AXES = (111, 109, 58, 45)
# Panel letters (x, top) in mm: one baseline per row, each at its panel's left edge
LETTERS = {"A": (0, 0.5), "B": (0, 47), "C": (92, 47), "D": (0, 103), "E": (106, 103)}
SIDE_X = {"single_cell": 0.5, "bulk": 62.0}   # D's sides: left edge of each side's text (mm)
CONNECTOR_X = 50.4                     # the connector's center (mm)
CONNECTOR_HALF = 7.0                   # mm
LINE_OFFSET = 0.21                     # a side's two lines: tissue and assay tag above, groups below (row units)
TAG_GAP_MM = 1.5                       # between the tissue and its assay tag
FAMILY_WRAP = 16                       # characters per family-label line
PAIR_WRAP = 12                         # characters per pair-name line
HEADER_WRAP = 9                        # characters per E header line
D_HEADER_PAD = 10                      # pt: lifts D's side headers to the top of E's two-line headers (one letter baseline)


def axes_mm(fig, rect):
    width, height = fig.get_size_inches() * 25.4
    left, top, w, h = rect
    return fig.add_axes((left / width, 1 - (top + h) / height, w / width, h / height))


def tidy(ax, n_rows):
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=SIZE)
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_a(fig, cell, pairs):
    schematics.place(fig, A_AXES, schematics.fig1a, FIGURE, "A")
    style.panel_label_mm(fig, "A", *LETTERS["A"])


def panel_b(fig, cell, pairs):
    schematics.place(fig, B_AXES, schematics.fig1b, FIGURE, "B")
    style.panel_label_mm(fig, "B", *LETTERS["B"])


def panel_c(fig, cell, pairs):
    ax = axes_mm(fig, C_AXES)
    rows = []
    for y, model in enumerate(F.MODEL_ORDER):
        family = configurations.MODELS[model]["family"]
        color = S.MODEL_COLORS[model]
        ax.axhspan(y - 0.5, y + 0.5, color=S.FAMILY_COLORS[family], alpha=S.FAMILY_TINT, linewidth=0)
        for x, input_ in enumerate(F.FIG1C_INPUTS):
            keys = [k for k in F.DISPLAY_ORDER if configurations.CONFIGURATIONS[k]["model"] == model
                    and configurations.CONFIGURATIONS[k]["input"] == input_]
            if keys:
                hatched = input_ == "pseudobulk" and model != "pseudobulk"   # pseudobulk's own cell is the profile itself: solid
                ax.add_patch(Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, facecolor="none" if hatched else color, edgecolor=color,
                                       hatch=S.PSEUDOBULK_HATCH if hatched else None, linewidth=0.5))
            else:
                ax.text(x, y, F.TEXT["absent"], ha="center", va="center", fontsize=SIZE)
            rows.append({"family": family, "model": model, "input": input_, "key": keys[0] if keys else None, "exists": bool(keys)})
    assert sum(r["exists"] for r in rows) == len(F.DISPLAY_ORDER)
    right = len(F.FIG1C_INPUTS) - 0.5
    for family, label in F.FAMILY_LABELS.items():
        ys = [y for y, m in enumerate(F.MODEL_ORDER) if configurations.MODELS[m]["family"] == family]
        ax.plot([right + 0.1] * 2, [min(ys) - 0.4, max(ys) + 0.4], color=S.FAMILY_COLORS[family], linewidth=1.0, clip_on=False)
        ax.text(right + 0.2, (min(ys) + max(ys)) / 2, textwrap.fill(label, FAMILY_WRAP), ha="left", va="center",
                color=S.FAMILY_COLORS[family], fontsize=SIZE)
    ax.annotate(F.TEXT["fig1c_header"], (0.5, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 21), textcoords="offset points",
                ha="center", va="bottom", fontsize=SIZE, fontweight="bold")   # over the input columns
    ax.text(0, -0.03, F.TEXT["fig1c_key"], transform=ax.transAxes, ha="left", va="top", fontsize=SIZE)   # test input, readout: caption
    ax.set_xlim(-0.5, right)
    tidy(ax, len(F.MODEL_ORDER))
    ax.set_xticks(range(len(F.FIG1C_INPUTS)))
    ax.set_xticklabels([F.TEXT["fig1c_inputs"][i].format(n=T.FROZEN_NCELL) for i in F.FIG1C_INPUTS])
    ax.set_yticks(range(len(F.MODEL_ORDER)))
    ax.set_yticklabels([configurations.MODELS[m]["label"] for m in F.MODEL_ORDER])
    style.panel_label_mm(fig, "C", *LETTERS["C"])
    common.panel_csv(pd.DataFrame(rows), FIGURE, "C")


def characteristics(pair) -> pd.DataFrame:
    """04d's characteristics of a pair, by column (its states checked)."""
    table = common.read_csv("04d_characteristics", pair, "characteristics.csv", keep_default_na=False).set_index("column")
    assert table["state"].isin(STATES).all(), f"{pair}: states {set(table['state'])}"
    return table


def connector(pair) -> str:
    """The sides' donor relationship, from 04d's "Independent cohorts" state and the pair's cross-fitting."""
    independent = characteristics(pair).loc["Independent cohorts", "state"]
    assert independent in ("yes", "no"), f"{pair}: 04d's Independent cohorts is {independent!r} (D can't draw it)"
    same_donors = "crossfit" in PAIRS[pair]["fold_sets"]
    assert not (same_donors and independent == "yes"), f"{pair}: cross-fitted by donor but 04d records independent cohorts"
    return "independent" if independent == "yes" else ("same_donors" if same_donors else "same_colony")


def panel_d(fig, cell, pairs):
    ax = axes_mm(fig, D_AXES)
    ax.set_xlim(0, D_AXES[2])          # x in mm, so a text's measured width places the tag after it
    renderer = fig.canvas.get_renderer()
    rows = []
    for y, pair in enumerate(pairs):
        kind = connector(pair)
        text = F.TEXT["connector"][kind]
        for side in ("single_cell", "bulk"):
            tissue, assay, groups = F.FIG1D[pair][side]
            x = SIDE_X[side]
            first = ax.text(x, y - LINE_OFFSET, tissue, ha="left", va="center", fontsize=SIZE)
            width_mm = first.get_window_extent(renderer).width / fig.dpi * 25.4
            ax.text(x + width_mm + TAG_GAP_MM, y - LINE_OFFSET, assay, ha="left", va="center", fontsize=SIZE,
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "none", "edgecolor": S.SIDE_COLORS[side], "linewidth": 0.75})
            ax.text(x, y + LINE_OFFSET, groups, ha="left", va="center", fontsize=SIZE)
            rows.append({"pair": pair, "side": side, "species": PAIRS[pair]["species"], "tissue": tissue, "assay": assay,
                         "groups": groups, "connector": kind})
        x = CONNECTOR_X
        ax.plot([x - CONNECTOR_HALF, x + CONNECTOR_HALF], [y, y], color=S.MODALITY_COLOR, linewidth=S.LINE_WIDTH,
                linestyle=F.FIG1D_CONNECTORS[kind])
        ax.text(x, y - 0.06, text, ha="center", va="bottom", fontsize=SIZE)
        if kind == "same_donors":
            ax.text(x, y + 0.06, F.TEXT["crossfit"].format(n=PAIRS[pair]["fold_sets"]["crossfit"]), ha="center", va="top",
                    fontsize=SIZE)
    tidy(ax, len(pairs))
    ax.set_xticks([SIDE_X["single_cell"], SIDE_X["bulk"]])
    ax.set_xticklabels([F.TEXT["sides"]["single_cell"], F.TEXT["sides"]["bulk"]], ha="left")
    ax.tick_params(axis="x", pad=D_HEADER_PAD)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(["\n".join([*textwrap.wrap(PAIRS[p]["title"][0], PAIR_WRAP), PAIRS[p]["title"][1]]) for p in pairs])
    style.panel_label_mm(fig, "D", *LETTERS["D"])
    common.panel_csv(pd.DataFrame(rows), FIGURE, "D")


def panel_e(fig, cell, pairs):
    """Beside D, rows aligned with D's (the pair names are D's)."""
    ax = axes_mm(fig, E_AXES)
    rows = [{"pair": pair, "characteristic": c, "state": r["state"], "value": r["value"], "annotation": r["annotation"]}
            for pair in pairs for c, r in characteristics(pair).loc[list(fig1e.FIG1E_MAIN)].iterrows()]
    frame = pd.DataFrame(rows)
    for _, r in frame[frame["state"] != "no"].iterrows():
        ax.plot(fig1e.FIG1E_MAIN.index(r["characteristic"]), pairs.index(r["pair"]), marker="o", markersize=S.MARKER_SIZE,
                color=S.MODALITY_COLOR, markerfacecolor=S.MODALITY_COLOR if r["state"] == "yes" else "none", linestyle="none")
    ax.set_xlim(-0.5, len(fig1e.FIG1E_MAIN) - 0.5)
    tidy(ax, len(pairs))
    ax.set_xticks(range(len(fig1e.FIG1E_MAIN)))
    ax.set_xticklabels([textwrap.fill(F.FIG1E_HEADERS.get(c, c), HEADER_WRAP, break_long_words=False) for c in fig1e.FIG1E_MAIN])
    ax.set_yticks([])
    style.panel_label_mm(fig, "E", *LETTERS["E"])
    common.panel_csv(frame, FIGURE, "E")


def main(pairs, panels):
    common.draw(FIGURE, {"A": panel_a, "B": panel_b, "C": panel_c, "D": panel_d, "E": panel_e}, pairs, panels, __file__,
                rows=(("A",), ("B", "C"), ("D", "E")))   # cells unused: the panels sit at fixed mm
