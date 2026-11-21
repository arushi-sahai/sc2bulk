"""Fig 1 (label-free). A and B: the schematics (figures/schematics.py; they replaced
the BioRender drafts), A across the top and B beside C.

C: the ten configurations as a model x input grid, drawn from config (the design; no stage output). Rows are the seven
models in the family order (config/figures.MODEL_ORDER), each tinted in its family color and grouped by a family label;
columns are cells and pseudobulk (hatched); "—" where the model has no such configuration.

D: the dataset pairs, one row per pair, each named in Fig 2's three lines (common.stacked_title; last lines on the
cohorts' count line): the single-cell cohort | the bulk cohort, each the phenotype compared
over the units per class (config/figures.FIG1D and table1.group_counts, as Table 1; donors or samples, the pair's
scoring unit).

E (beside D, its rows aligned with D's): config/figures.FIG1E_PANEL's characteristics, 04d's or typed in
(config/figures.FIG1E_TYPED): a filled circle = yes, an open circle = not determinable, blank = no. Thresholds are in the legend; the 04d annotations are in the CSV only (Table S1).

Laid out in millimetres at print size: C ~80 x 55 mm; D | E 170 x 70 mm together, D wider; every
text >= config/figures.FIG1_FONT_SIZE (7 pt). Each panel's own PNG (common.save) is the panel cropped from the
composite, without its letter (the assembly adds the letters)."""
import textwrap

import pandas as pd
from matplotlib.patches import Rectangle

import common
import schematics
import style
import table1
from sc2bulk.config import configurations
from sc2bulk.config import figures as F
from sc2bulk.config import style as S
from sc2bulk.config.pairs import PAIRS

FIGURE = "Fig 1"
READS_BULK_LABELS = False
READS_SCORES = False
STATES = ("yes", "not determinable", "no")
SIDES = ("single_cell", "bulk")
SIZE = F.FIG1_FONT_SIZE
# Axes rectangles in mm from the composite's top-left corner: (left, top, width, height). Labels sit outside them.
A_AXES = (0, 0, *schematics.SIZES[schematics.fig1a])   # the schematics, in mm at print size
B_AXES = (0, 46, *schematics.SIZES[schematics.fig1b])
C_PITCH = schematics.B_ROW_PITCH / 2                     # C's rows within a family block
C_FIRST = B_AXES[1] + B_AXES[3] - schematics.B_ROW_TOP   # B's first row centre, mm from the top


def c_rows() -> dict:
    """C's row positions (row units from the first row): each family's models centred on its B row (B's rows are the
    families in FAMILY_COLORS order), one C_PITCH apart; the gaps between blocks take up the rest."""
    found = {}
    for r, family in enumerate(S.FAMILY_COLORS):
        models = [m for m in F.MODEL_ORDER if configurations.MODELS[m]["family"] == family]
        found |= {m: r * schematics.B_ROW_PITCH / C_PITCH + i - (len(models) - 1) / 2 for i, m in enumerate(models)}
    return found


C_ROWS = c_rows()
C_AXES = (110, C_FIRST - C_PITCH / 2, 46, C_PITCH * (max(C_ROWS.values()) + 1))   # the grid; model names to its left
D_AXES = (17, 106, 72, 49)             # x in mm within the axes; pair names to its left
E_AXES = (93, 106, 74, 49)
# Panel letters (x, top) in mm: one baseline per row, each at its panel's left edge
LETTERS = {"A": (0, 0.5), "B": (0, 47), "C": (92, 47), "D": (0, 100), "E": (89.5, 100)}
SIDE_X = {"single_cell": 0.5, "bulk": 44.0}   # D's cohorts: left edge of each cohort's text (mm)
BASELINES_MM = (-0.4, 2.2)             # D's two lines (phenotype, counts): baselines below each row's centre
NAME_GAP_MM = 1.5                      # D's pair names: right edge left of the axes
NAME_SPACING = 1.05                    # D's pair names: line spacing
HEADER_WRAP = 18                       # characters per E header line
D_HEADER_PAD = 10                      # pt: lifts D's cohort headers to the top of E's two-line headers (one letter baseline)


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
    for model in F.MODEL_ORDER:
        y = C_ROWS[model]
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
    ax.annotate(F.TEXT["fig1c_header"], (0.5, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 21), textcoords="offset points",
                ha="center", va="bottom", fontsize=schematics.BODY, fontweight="bold")   # over the input columns; b's header size
    ax.set_xlim(-0.5, right)
    tidy(ax, len(F.MODEL_ORDER))
    ax.set_ylim(max(C_ROWS.values()) + 0.5, -0.5)
    ax.set_xticks(range(len(F.FIG1C_INPUTS)))
    ax.set_xticklabels([F.TEXT["fig1c_inputs"][i] for i in F.FIG1C_INPUTS])
    ax.set_yticks([C_ROWS[m] for m in F.MODEL_ORDER])
    ax.set_yticklabels([configurations.MODELS[m]["label"] for m in F.MODEL_ORDER])
    style.panel_label_mm(fig, "C", *LETTERS["C"])
    common.panel_csv(pd.DataFrame(rows), FIGURE, "C")


def characteristics(pair) -> pd.DataFrame:
    """04d's characteristics of a pair, by column (its states checked)."""
    table = common.read_csv("04d_characteristics", pair, "characteristics.csv", keep_default_na=False).set_index("column")
    assert table["state"].isin(STATES).all(), f"{pair}: states {set(table['state'])}"
    return table


def panel_d(fig, cell, pairs):
    ax = axes_mm(fig, D_AXES)
    ax.set_xlim(0, D_AXES[2])          # x in mm
    row_mm = D_AXES[3] / len(pairs)
    rows = []
    for y, pair in enumerate(pairs):
        first, second = (y + b / row_mm for b in BASELINES_MM)
        unit = F.TEXT["fig1d_units"][PAIRS[pair]["scoring_unit"]]
        for side, counts in zip(SIDES, table1.group_counts(pair)):
            phenotype = F.FIG1D[pair][side][2]
            n = F.TEXT["fig1d_counts"].format(a=counts[0], b=counts[1], unit=unit)
            ax.text(SIDE_X[side], first, phenotype, ha="left", va="baseline", fontsize=SIZE)
            ax.text(SIDE_X[side], second, n, ha="left", va="baseline", fontsize=SIZE)
            rows.append({"pair": pair, "cohort": side, "phenotype": phenotype, "n_0": counts[0], "n_1": counts[1], "unit": unit})
        ax.text(-NAME_GAP_MM, second, common.stacked_title(pair, mark=False), ha="right", va="baseline", fontsize=SIZE,
                linespacing=NAME_SPACING, clip_on=False)
    tidy(ax, len(pairs))
    for side in SIDES:   # C's header style
        ax.annotate(F.TEXT["fig1d_cohorts"][side], (SIDE_X[side], 1.0), xycoords=("data", "axes fraction"), xytext=(0, D_HEADER_PAD),
                    textcoords="offset points", ha="left", va="bottom", fontsize=schematics.BODY, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    style.panel_label_mm(fig, "D", *LETTERS["D"])
    common.panel_csv(pd.DataFrame(rows), FIGURE, "D")


def state(pair, column, table) -> tuple:
    """A characteristic's (state, value, annotation): typed in (config/figures.FIG1E_TYPED) or 04d's."""
    if column in F.FIG1E_TYPED:
        return ("yes" if pair in F.FIG1E_TYPED[column] else "no"), float("nan"), ""
    r = table.loc[column]
    return r["state"], r["value"], r["annotation"]


def panel_e(fig, cell, pairs):
    """Beside D, rows aligned with D's (the pair names are D's)."""
    ax = axes_mm(fig, E_AXES)
    columns = list(F.FIG1E_PANEL)
    rows = []
    for pair in pairs:
        table = characteristics(pair)
        rows += [dict(zip(("pair", "characteristic", "state", "value", "annotation"), (pair, c, *state(pair, c, table))))
                 for c in columns]
    frame = pd.DataFrame(rows)
    for _, r in frame[frame["state"] != "no"].iterrows():
        ax.plot(columns.index(r["characteristic"]), pairs.index(r["pair"]), marker="o", markersize=S.MARKER_SIZE,
                color=S.MODALITY_COLOR, markerfacecolor=S.MODALITY_COLOR if r["state"] == "yes" else "none", linestyle="none")
    ax.set_xlim(-0.5, len(columns) - 0.5)
    tidy(ax, len(pairs))
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([textwrap.fill(F.FIG1E_PANEL[c], HEADER_WRAP, break_long_words=False) for c in columns])
    ax.set_yticks([])
    style.panel_label_mm(fig, "E", *LETTERS["E"])
    common.panel_csv(frame, FIGURE, "E")


def main(pairs, panels):
    common.draw(FIGURE, {"A": panel_a, "B": panel_b, "C": panel_c, "D": panel_d, "E": panel_e}, pairs, panels, __file__,
                rows=(("A",), ("B", "C"), ("D", "E")))   # cells unused: the panels sit at fixed mm
