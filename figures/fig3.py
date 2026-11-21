"""Fig 3: the modality gap.

A: human ileum, Crohn's, 3 x 4: the configurations of config/figures.FIG3_KEYS as rows (named at the left), at its
   encoder seed (config/alignment), under across-assay standard scaling (one scaler for both sides), within-assay
   standard scaling (per side), ComBat and Harmony as columns, in stage 09c's first two PCs; samples as circles colored
   by side (config/style.SIDE_COLORS), group 1 filled and group 0 open (single-cell groups from stage 02, bulk groups
   through sc2bulk.labels); stage 09's modality silhouette under each plot. A row whose coordinates stage 09c does not
   save (outside config/geometry.CONFIGURATIONS) keeps its silhouettes and says so in its plots. Sides are told apart by
   sample id (stages 09 and 09c assert the two sets are disjoint), never by parsing an id. Its key in one row on its
   pair tag's line, at its right; b and c's key in one row between a and b.
B: modality silhouette per pair for the same configurations, the conditions of config/figures.FIG3_CONDITIONS as
   labeled x-offsets joined by a line: the median over encoder seeds (over folds x seeds for human blood, aging);
   fixed representations' one value.
C: bulk AUROC under the same conditions, the mean over seeds; within-assay is Fig 2B's bulk value (the same rows),
   pseudobulk's the dashed line; the axis from chance, 0.5 (plots.auroc_ticks). The conditions are named on the last
   pair's CytoSet dots (callout_above).
B titles each pair in two lines, species and tissue over phenotype (common.stacked_title), for C as well.
draw_gap(), silhouettes()/aurocs() and offset_panel() are reused by Fig. S13."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import common
import plots
import style
from sc2bulk.config import alignment as A
from sc2bulk.config import configurations
from sc2bulk.config import figures as F
from sc2bulk.config import geometry as G
from sc2bulk.config import style as S

FIGURE = "Fig 3"
READS_BULK_LABELS = True               # A marks bulk samples' groups (open vs. filled)
READS_SCORES = True
COMPACT_TITLE_PAD = 30                 # Fig. S13A-E: points from a block's top to its pair title (and its letter's bottom)
DRAW_ORDER_SEED = 0                    # A, S13A-E: the samples' drawing order
KEY_GAP_PT = 3                         # Fig 3b and c's key: points between its bottom and the top of b's pair titles
TOP_MM = 15                            # Fig 3: room over A for its letter, pair tag and column titles
CALLOUT_KEY = "cytoset_cells"          # Fig 3c: the configuration whose dots (last column) carry the condition names
CALLOUT_TOP_PT, CALLOUT_STEP_PT = 1, 1.5   # ... the first name this far under the frame's top, the names this far apart
CALLOUT_CLEAR_PT = 3                   # ... and at least this far above the dots under them
CALLOUT_RAIL_PT = 3                    # ... each name starting this far right of the one above (its leader's rail between)
CALLOUT_OVERHANG_PT = 6                # ... and reaching at most this far into the gap left of the frame
SILHOUETTE_PAD_PT = 3.5               # Fig 3a, S13a-e: points between a plot's bottom frame and its silhouette (about 1.2 mm)


def geometry(pair, key, unit) -> pd.DataFrame:
    return common.read_kind("alignment_geometry", "09_alignment", pair, f"{key}_geometry.parquet", unit)


def coordinates(pair, key, unit) -> pd.DataFrame:
    return common.read_kind("alignment_geometry", "09c_geometry", pair, f"{key}_geometry.parquet", unit)


def draw_gap(fig, cell, pair, panel, item, compact=False):
    """Fig 3A's PCA plots for one pair (fold 0 of its primary fold set): FIG3_KEYS x FIG3A columns, the modality
    silhouette under each plot, the alignments as column titles and the configurations as row labels. Fig 3: the row
    labels level at the left, its key in one row on the pair tag's line; compact (Fig. S13): under the pair's title,
    the short column titles and the silhouette's value alone (the legend names it). Writes the panel's CSV."""
    keys = [k for k in F.FIG3_KEYS if k in common.kept()]
    columns = F.FIG3A["columns"]
    size, wrap = (S.MIN_FONT_SIZE, 12) if compact else (F.TEXT_SIZE, 20)   # wrap: characters per line of a column title
    unit = common.primary_units(pair)[0]
    single = common.training_samples(pair, unit).set_index("sample_id")["group"]   # stage 09's two sides for the unit
    bulk = common.bulk_labels(pair).set_index("sample_id")["group"].loc[common.scored_bulk(pair, unit)["sample_id"]]
    assert not set(single.index) & set(bulk.index), f"{pair} {unit}: a sample id on both sides"
    if compact:
        grid = cell.subgridspec(len(keys), len(columns), hspace=0.35, wspace=0.45)
    else:
        grid = cell.subgridspec(len(keys), len(columns), hspace=0.3, wspace=0.2)
    csv, axes = [], []
    for r, key in enumerate(keys):
        seed = -1 if key in common.NO_ENCODER_LEVEL else A.GEOMETRY_ENCODER_SEED
        table = common.pick(geometry(pair, key, unit), neutral=False, encoder_seed=seed)
        for c, alignment in enumerate(columns):
            part = table[table["alignment"] == alignment]
            silhouette = float(part.loc[part["measure"] == "silhouette", "value"].iloc[0])
            ax = fig.add_subplot(grid[r, c])
            axes.append(ax)
            if key in G.CONFIGURATIONS:
                points = common.pick(coordinates(pair, key, unit), neutral=False, encoder_seed=seed, alignment=alignment)
                pcs = points[points["measure"].isin(["pc1", "pc2"])].pivot(index="sample_id", columns="measure", values="value")
                side = pd.Series(np.where(pcs.index.isin(single.index), "single_cell", "bulk"), index=pcs.index)
                assert (pcs.index.isin(single.index) | pcs.index.isin(bulk.index)).all(), f"{pair} {key}: a PC row of neither side"
                group = pd.concat([single, bulk]).loc[pcs.index]
                order = np.random.default_rng(DRAW_ORDER_SEED).permutation(len(pcs))   # one random drawing order: where the sides overlap, both show
                glyphs = [style.side_points(side.iloc[k], int(group.iloc[k])) for k in order]
                ax.scatter(pcs["pc1"].iloc[order], pcs["pc2"].iloc[order], s=S.MARKER_SIZE ** 2, marker=S.SAMPLE_MARKER, rasterized=True,
                           facecolors=[gl["facecolors"] for gl in glyphs], edgecolors=[gl["edgecolors"] for gl in glyphs], linewidths=0.5)
                csv += [{"pair": pair, "representation": key, "alignment": alignment, "sample_id": s, "side": side[s], "group": int(group[s]),
                         "pc1": pcs.loc[s, "pc1"], "pc2": pcs.loc[s, "pc2"]} for s in pcs.index]
            else:
                ax.text(0.5, 0.5, plots.narrow(F.FIGURE_TEXT["coordinates_pending"], wrap), ha="center", va="center", fontsize=S.MIN_FONT_SIZE,
                        color="#7a7a7a", transform=ax.transAxes)
            csv.append({"pair": pair, "representation": key, "alignment": alignment, "side": "all", "silhouette": silhouette})
            if compact:
                ax.set_box_aspect(1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"{silhouette:.2f}" if compact else f"{F.TEXT['silhouette_short']} {silhouette:.2f}", fontsize=size, labelpad=SILHOUETTE_PAD_PT)
            if r == 0:
                ax.set_title(F.ALIGNMENT_COLUMN_SHORT[alignment] if compact else plots.narrow(F.ALIGNMENT_COLUMN_LABELS[alignment], wrap),
                             fontsize=size)
            if c == 0 and compact:
                ax.set_ylabel(common.label(key).replace(", ", ",\n"), fontsize=size)
            elif c == 0:
                ax.set_ylabel(common.label(key), fontsize=size, labelpad=2)
    if compact:   # the pair's title over the block, the letter at its top left
        block = fig.add_subplot(cell)
        block.axis("off")
        block.set_title(common.pair_title(pair), pad=COMPACT_TITLE_PAD, fontsize=F.TEXT_SIZE)
        style.panel_label(block, panel, x=-0.06, above_pt=COMPACT_TITLE_PAD)   # level with the pair's title
    else:
        tag = axes[0].get_position().y1 + (F.PAIR_TAG_PT["Fig 3A"] + F.TEXT_SIZE / 2) / 72 / fig.get_figheight()   # the pair tag's middle
        style.legend_strip(fig, [], ("side_single_cell", "side_bulk", "group_0", "group_1"), x=axes[len(columns) - 1].get_position().x1,
                           y=tag, ncol=4, loc="center right", groups=F.GROUP_NAMES[pair])   # the pair's own group names, in its line
        style.panel_label(axes[0], panel, at_left=True, above_pt=F.PAIR_TAG_PT["Fig 3A"])   # level with the pair tag
        style.pair_tag(axes[0], common.pair_title(pair).replace("\n", ", "), F.PAIR_TAG_PT["Fig 3A"])   # Fig 3A: the example pair
    common.panel_csv(pd.DataFrame(csv), item, panel)
    return axes


def silhouettes(pair, keys, alignments=configurations.ALIGNMENTS) -> pd.DataFrame:
    """key, alignment, value: the median over (fold, encoder seed) of the modality silhouette (fixed keys: one value
    per fold, then the median over folds); the expected seeds and folds asserted."""
    out = []
    for key in keys:
        frames = [geometry(pair, key, u) for u in common.primary_units(pair)]
        table = pd.concat(frames, ignore_index=True)
        table = table[(table["measure"] == "silhouette") & table["alignment"].isin(alignments)]
        for alignment, part in table.groupby("alignment"):
            found = sorted(zip(part["fold"], part["encoder_seed"]))
            expected = sorted((f, e) for _, f in common.primary_units(pair) for e in common.seeds_of(key))
            assert found == expected, f"{pair} {key} {alignment}: silhouettes at {found}"
            out.append({"pair": pair, "key": key, "alignment": alignment, "value": float(part["value"].median())})
        assert set(table["alignment"]) == set(alignments), f"{pair} {key}: alignments {set(table['alignment'])}"
    return pd.DataFrame(out)


def aurocs(pair, keys, alignments=configurations.ALIGNMENTS) -> pd.DataFrame:
    """key, alignment, value: bulk AUROC under each alignment, the mean over seeds ("none" = stage 08's rows)."""
    out = []
    for key in keys:
        for alignment in alignments:
            raw = common.aurocs(pair, key, alignment=alignment, **common.headline_conditions(key))
            out.append({"pair": pair, "key": key, "alignment": alignment, "value": common.mean_over_seeds(raw)})
    return pd.DataFrame(out)


def offset_panel(fig, cell, pairs, values_of, keys, alignments, panel, item, auroc, titles=True, letter=None, title=common.pair_title):
    """Fig 3B/C: pairs as columns on one y scale, offset_columns in each; pair titles (title(pair)) iff titles. Silhouettes (auroc
    False): the widest drawn up to config/figures.SILHOUETTE_TOP, in Fig. S13 the conditions named on the first pair's pseudobulk dots
    in the room above the data; bulk AUROC: AUROC_RANGE from chance (plots.auroc_ticks), pseudobulk's within-assay value
    as a dashed line, in Fig 3 the conditions named on the last pair's CALLOUT_KEY dots from the empty space above them (callout_above). letter: style.panel_label's placement keywords (at_left's left_mm, above_pt)."""
    limits = F.AUROC_RANGE if auroc else None
    ylabel = F.TEXT["bulk_auroc"] if auroc else F.TEXT["silhouette"]
    axes = plots.columns(fig, cell, len(pairs))
    frames = []
    for ax, pair in zip(axes, pairs):
        frame = values_of(pair, keys, alignments)
        plots.offset_columns(ax, {(r.key, r.alignment): r.value for r in frame.itertuples()}, keys, alignments, limits or (-np.inf, np.inf))
        if auroc:
            auroc_lines(ax, frame)
        if titles:
            ax.set_title(title(pair))
        frames.append(frame)
    plots.share_y(axes, limits)
    axes[0].set_ylabel(ylabel if item == FIGURE else ylabel.replace(" ", "\n", 1))   # Fig. S13's shorter rows: two lines
    first = frames[0].set_index(["key", "alignment"])["value"]
    if not auroc:
        for ax in axes:
            ax.set_ylim(top=F.SILHOUETTE_TOP)
        if item != FIGURE:
            plots.alignment_callout(axes[0], [(plots.condition_x(0, j, len(alignments)), first[("pseudobulk", a)], F.ALIGNMENT_TICKS[a])
                                              for j, a in enumerate(alignments)], S.MIN_FONT_SIZE)
    else:
        plots.auroc_ticks(axes[0], 0.1 if item == FIGURE else 0.25)   # Fig. S13's shorter rows: three ticks
        if item == FIGURE:
            clip = lambda v: min(max(v, limits[0]), limits[1])
            last = frames[-1].set_index(["key", "alignment"])["value"]
            dots = [(plots.condition_x(i, j, len(alignments)), clip(last[(k, a)])) for i, k in enumerate(keys) for j, a in enumerate(alignments)]
            i = keys.index(CALLOUT_KEY)
            callout_above(axes[-1], [(plots.condition_x(i, j, len(alignments)), clip(last[(CALLOUT_KEY, a)]), F.ALIGNMENT_TICKS[a])
                                     for j, a in enumerate(alignments)], dots)
    style.panel_label(axes[0], panel, at_left=True, **(letter or {}))
    common.panel_csv(pd.concat(frames, ignore_index=True), item, panel)
    return axes


def segments_cross(p, q, r, s) -> bool:
    """Whether segment pq crosses segment rs (a shared end point does not count)."""
    turn = lambda a, b, c: np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return turn(p, q, r) * turn(p, q, s) < 0 and turn(r, s, p) * turn(r, s, q) < 0


def segment_distance(a, b, c) -> float:
    """The distance from point c to segment ab."""
    a, b, c = map(np.asarray, (a, b, c))
    along = np.clip(np.dot(c - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1)
    return float(np.linalg.norm(a + along * (b - a) - c))


def callout_above(ax, points, dots, size=F.TEXT_SIZE):
    """Fig 3c: the conditions named on one configuration's dots (x, y, label) of a column whose upper part is empty: one
    name per line, stacked top to bottom in the dots' left-to-right order from CALLOUT_TOP_PT under the frame's top, each
    as far right as the frame allows while starting right of the names above (by CALLOUT_RAIL_PT). Each leader runs from
    its name's left end a step left, down that rail past the names below, then straight to its dot. Asserted: the names
    inside the frame (up to CALLOUT_OVERHANG_PT into the gap on its left) and CALLOUT_CLEAR_PT above every dot under
    them; no leader crossing a name or another leader, or passing within a marker and a point of another dot (all of
    the column's dots, x, y)."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72   # pixels per point
    to_px, to_data = ax.transData.transform, ax.transData.inverted().transform
    frame = ax.get_window_extent(renderer)
    named = sorted(points)
    texts = [ax.text(0, 0, label, ha="left", va="top", fontsize=size, color="#4d4d4d", zorder=4,
                     bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none"}) for _, _, label in named]
    widths = [t_.get_window_extent(renderer).width for t_ in texts]
    height = texts[0].get_window_extent(renderer).height
    lefts = [frame.x1 - widths[-1]]   # the bottom name flush with the frame's right edge, each above it starting further left
    for w in reversed(widths[:-1]):
        lefts.insert(0, min(frame.x1 - w, lefts[0] - CALLOUT_RAIL_PT * px))
    assert lefts[0] - CALLOUT_RAIL_PT * px >= frame.x0 - CALLOUT_OVERHANG_PT * px, "Fig 3c: the condition names don't fit the column"
    tops = [frame.y1 - CALLOUT_TOP_PT * px - k * (height + CALLOUT_STEP_PT * px) for k in range(len(named))]
    band = tops[-1] - height - CALLOUT_STEP_PT * px   # where the rails turn toward their dots
    radius = (S.MARKER_SIZE / 2 + 1) * px
    boxes, leaders = [], []
    for (x, y, _), text, left, top, width in zip(named, texts, lefts, tops, widths):
        text.set_position(to_data((left, top)))
        boxes.append((left, top - height, left + width, top))
        rail = left - CALLOUT_RAIL_PT * px / 2
        dot = np.asarray(to_px((x, y)))
        turn = np.array([rail, band])
        end = dot + (turn - dot) * radius / np.linalg.norm(turn - dot)
        path = [(left, top - height / 2), (rail, top - height / 2), (rail, band), tuple(end)]
        leaders.append(path)
        ax.plot(*zip(*(to_data(q) for q in path)), color="#4d4d4d", linewidth=0.5, zorder=3, solid_joinstyle="miter")
    for k, ((x0, y0, x1, y1), path) in enumerate(zip(boxes, leaders)):
        under = [to_px(d)[1] for d in dots if x0 - radius <= to_px(d)[0] <= x1 + radius]
        assert not under or y0 - max(under) >= CALLOUT_CLEAR_PT * px + radius, f"Fig 3c: {named[k][2]} meets the dots under it"
        for a, b in zip(path[1:-1], path[2:]):
            for m, (u0, v0, u1, v1) in enumerate(boxes):
                corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
                inside = lambda q: u0 < q[0] < u1 and v0 < q[1] < v1
                assert m == k or not (inside(a) or inside(b) or any(segments_cross(a, b, c, d) for c, d in zip(corners, corners[1:] + corners[:1]))), \
                    f"Fig 3c: the leader of {named[k][2]} crosses {named[m][2]}"
            for other, path_ in enumerate(leaders):
                assert other == k or not any(segments_cross(a, b, c, d) for c, d in zip(path_[1:-1], path_[2:])), \
                    f"Fig 3c: the leaders of {named[k][2]} and {named[other][2]} cross"
            own = np.asarray(to_px(named[k][:2]))
            assert all(segment_distance(a, b, to_px(d)) >= radius for d in dots if np.linalg.norm(np.asarray(to_px(d)) - own) > 1e-6), \
                f"Fig 3c: the leader of {named[k][2]} passes a dot"


def auroc_lines(ax, frame):
    """Pseudobulk's within-assay bulk AUROC (an arrowhead at the left edge if below the axis)."""
    value = frame.loc[(frame["key"] == "pseudobulk") & (frame["alignment"] == "none"), "value"].iloc[0]
    plots.reference_line(ax, value, F.AUROC_RANGE, S.PSEUDOBULK_LINE, -0.4)


def panel_a(fig, cell, pairs):
    if F.FIG3A["pair"] in pairs:
        draw_gap(fig, cell, F.FIG3A["pair"], "A", FIGURE)
    else:
        common.absent(fig, cell, FIGURE, "A", F.FIG3A["pair"])


def stacked(pair) -> str:
    return common.stacked_title(pair, mark=False, lines=2)


def panel_b(fig, cell, pairs):
    """b's pair titles serve c too (the same columns). b and c's key sits over the titles as part of b (an axes legend, so
    the layout checks count it in b), its letter level with the key."""
    keys = [k for k in F.FIG3_KEYS if k in common.kept()]
    axes = offset_panel(fig, cell, pairs, silhouettes, keys, F.FIG3_CONDITIONS, "B", FIGURE, False, title=stacked,
                        letter={"above_pt": 0})
    top = axes[0].title.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).y1
    handles = [Line2D([], [], **style.marker(k), label=common.label(k)) for k in keys]
    handles.append(Line2D([], [], **S.PSEUDOBULK_LINE, label=F.TEXT["pseudobulk_within_assay"]))
    key = axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=((axes[0].get_position().x0 + axes[-1].get_position().x1) / 2,
                         top + KEY_GAP_PT / 72 / fig.get_figheight()), bbox_transform=fig.transFigure, ncol=len(handles),
                         handlelength=1.5, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0)
    letter = next(t for t in axes[0].texts if t.get_gid() == "panel_label")
    fig.draw_without_rendering()   # places the key's entries
    renderer = fig.canvas.get_renderer()
    bottom = key.get_texts()[0].get_window_extent(renderer).y0 - axes[0].get_window_extent(renderer).y1
    letter.xyann = (0, bottom * 72 / fig.dpi)


def panel_c(fig, cell, pairs):
    """Within-assay reads the same scores rows as Fig 2B's bulk values (stage 08's primary files; stage 09 asserts it
    uses them). No pair titles: b's head the same columns."""
    offset_panel(fig, cell, pairs, aurocs, [k for k in F.FIG3_KEYS if k in common.kept()], F.FIG3_CONDITIONS, "C", FIGURE, True,
                 titles=False)


def main(pairs, panels):
    functions = {"A": panel_a, "B": panel_b, "C": panel_c}
    with plt.rc_context(F.TEXT_RC):
        common.draw(FIGURE, functions, pairs, panels, __file__, rows=[("A",), ("B",), ("C",)], ratios=(1.9, 0.66, 0.66),
                    top_mm=TOP_MM, hspace=(0.49, 0.1), left=0.1, bottom=0.03)
