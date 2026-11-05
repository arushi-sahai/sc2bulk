"""Fig 3: the modality gap.

A: human ileum, Crohn's, 2 x 2: gene profiles (pseudobulk) and CytoSet, cells embeddings (config/alignment's geometry
   configurations, at its encoder seed) under the unaligned stacked scaler and ComBat, in stage 09's first two PCs;
   samples as circles colored by side (config/style.SIDE_COLORS), group 1 filled and group 0 open (single-cell groups
   from stage 02, bulk groups through sc2bulk.labels); rows labelled in text ("gene profiles" for pseudobulk); the modality silhouette as an inset. Sides are told apart by sample id (stage 09 asserts the two
   sets are disjoint), never by parsing an id.
B: modality silhouette per pair, pseudobulk and the cells configurations, the four alignments as labeled x-offsets:
   the median over encoder seeds (over folds x seeds for human blood, aging); fixed representations' one value.
C: bulk AUROC under the same alignments, the mean over seeds; "None" is Fig 2B's bulk value (the same rows).
draw_gap() and silhouettes()/aurocs() are reused by Fig. S13."""
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

import common
import plots
import style
from sc2bulk.config import alignment as A
from sc2bulk.config import configurations
from sc2bulk.config import figures as F
from sc2bulk.config import style as S

FIGURE = "Fig 3"
READS_BULK_LABELS = True               # A marks bulk samples' groups (open vs. filled)
READS_SCORES = True
COMPACT_TITLE_PAD = 20                 # Fig. S13A-E: points from a block's top to its pair title (and its letter's bottom)
DRAW_ORDER_SEED = 0                    # A, S13A-E: the samples' drawing order
GROUP_GAP = 0.6                        # Fig 3A: the gap between the two representations' plot pairs (a plot's width = 1)
SUBHEADING_PT = 22                     # Fig 3A: points from the axes' top to a representation's subheading (over its plots' two-line titles)


def geometry(pair, key, unit) -> pd.DataFrame:
    return common.read_kind("alignment_geometry", "09_alignment", pair, f"{key}_geometry.parquet", unit)


def draw_gap(fig, cell, pair, panel, item, compact=False):
    """Fig 3A's four PCA plots for one pair (fold 0 of its primary fold set): GEOMETRY_CONFIGURATIONS x FIG3A columns.
    One row (Fig 3), each plot titled with its representation, alignment and modality silhouette; or compact (Fig. S13):
    2 x 2 under the pair's title, alignments as column titles, representations as row labels, silhouettes below each
    plot. Writes the panel's CSV."""
    ncols, size = (2, S.MIN_FONT_SIZE) if compact else (4, F.TEXT_SIZE)
    unit = common.primary_units(pair)[0]
    single = common.training_samples(pair, unit).set_index("sample_id")["group"]   # stage 09's two sides for the unit
    bulk = common.bulk_labels(pair).set_index("sample_id")["group"].loc[common.scored_bulk(pair, unit)["sample_id"]]
    assert not set(single.index) & set(bulk.index), f"{pair} {unit}: a sample id on both sides"
    n = len(A.GEOMETRY_CONFIGURATIONS) * len(F.FIG3A["columns"])
    if compact:
        grid = cell.subgridspec(n // ncols, ncols, hspace=0.45, wspace=0.12)
        slot = lambda position: grid[position // ncols, position % ncols]
    else:   # one row: two plots per representation, a wider gap between the representations
        grid = cell.subgridspec(1, ncols + 1, wspace=0.25, width_ratios=(1, 1, GROUP_GAP, 1, 1))   # the gap holds a's key
        slot = lambda position: grid[0, position + position // 2]
    csv, axes = [], []
    for r, key in enumerate(A.GEOMETRY_CONFIGURATIONS):
        seed = -1 if key in common.NO_ENCODER_LEVEL else A.GEOMETRY_ENCODER_SEED
        table = common.pick(geometry(pair, key, unit), neutral=False, encoder_seed=seed)
        row = F.TEXT["gene_profiles"].capitalize() if key == "pseudobulk" else common.label(key)
        for c, alignment in enumerate(F.FIG3A["columns"]):
            part = table[table["alignment"] == alignment]
            pcs = part[part["measure"].isin(["pc1", "pc2"])].pivot(index="sample_id", columns="measure", values="value")
            side = pd.Series(np.where(pcs.index.isin(single.index), "single_cell", "bulk"), index=pcs.index)
            assert (pcs.index.isin(single.index) | pcs.index.isin(bulk.index)).all(), f"{pair} {key}: a PC row of neither side"
            group = pd.concat([single, bulk]).loc[pcs.index]
            silhouette = float(part.loc[part["measure"] == "silhouette", "value"].iloc[0])
            position = r * len(F.FIG3A["columns"]) + c
            ax = fig.add_subplot(slot(position))
            axes.append(ax)
            order = np.random.default_rng(DRAW_ORDER_SEED).permutation(len(pcs))   # one random drawing order: where the sides overlap, both show
            glyphs = [style.side_points(side.iloc[k], int(group.iloc[k])) for k in order]
            ax.scatter(pcs["pc1"].iloc[order], pcs["pc2"].iloc[order], s=S.MARKER_SIZE ** 2, marker=S.SAMPLE_MARKER, rasterized=True,
                       facecolors=[gl["facecolors"] for gl in glyphs], edgecolors=[gl["edgecolors"] for gl in glyphs], linewidths=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if compact:
                if r == 0:
                    ax.set_title(plots.narrow(F.ALIGNMENT_COLUMN_LABELS[alignment], 16), fontsize=size)
                if c == 0:
                    ax.set_ylabel(row, fontsize=size)
                ax.set_xlabel(f"{F.TEXT['silhouette_short']} {silhouette:.2f}", fontsize=size)
            else:
                ax.set_box_aspect(1)
                ax.set_title(f"{F.ALIGNMENT_COLUMN_LABELS[alignment]}\n{F.TEXT['silhouette']}: {silhouette:.2f}", fontsize=size)
                if c == 1:   # the representation's subheading, centered over its two plots
                    left = axes[-2].get_position()
                    ax.annotate(row, ((left.x0 + ax.get_position().x1) / 2, 1.0), xycoords=blended_transform_factory(fig.transFigure, ax.transAxes),
                                xytext=(0, SUBHEADING_PT), textcoords="offset points", ha="center", va="bottom", fontsize=size)
                if position % ncols == 0:
                    ax.set_ylabel(F.TEXT["pc2"], fontsize=size)
                if position >= n - ncols:
                    ax.set_xlabel(F.TEXT["pc1"], fontsize=size)
            csv += [{"pair": pair, "representation": key, "alignment": alignment, "sample_id": s, "side": side[s], "group": int(group[s]),
                     "pc1": pcs.loc[s, "pc1"], "pc2": pcs.loc[s, "pc2"]} for s in pcs.index]
            csv.append({"pair": pair, "representation": key, "alignment": alignment, "side": "all", "silhouette": silhouette})
    if compact:   # the pair's title over the block, the letter at its top left
        block = fig.add_subplot(cell)
        block.axis("off")
        block.set_title(common.pair_title(pair), pad=COMPACT_TITLE_PAD, fontsize=F.TEXT_SIZE)
        style.panel_label(block, panel, x=-0.06, above_pt=COMPACT_TITLE_PAD)   # level with the pair's title
    else:
        left, right = axes[1].get_position(), axes[2].get_position()   # a's key (data colors) between the two representations
        style.legend_strip(fig, [], ("side_single_cell", "side_bulk", "group_0", "group_1"), x=(left.x1 + right.x0) / 2,
                           y=(left.y0 + left.y1) / 2, ncol=1, loc="center", groups=F.GROUP_NAMES[pair])   # the pair's own group names
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


def offset_panel(fig, cell, pairs, values_of, keys, panel, item, ylabel, reference=None, titles=True, limits=None):
    """Fig 3B/C: pairs as columns on one y scale (limits, or the widest drawn), offset_columns in each; pair titles iff titles."""
    axes = plots.columns(fig, cell, len(pairs))
    frames = []
    for ax, pair in zip(axes, pairs):
        frame = values_of(pair, keys)
        plots.offset_columns(ax, {(r.key, r.alignment): r.value for r in frame.itertuples()}, keys, limits or (-np.inf, np.inf))
        if reference is not None:
            reference(ax, frame)
        if titles:
            ax.set_title(common.pair_title(pair))
        frames.append(frame)
    plots.share_y(axes, limits)
    axes[0].set_ylabel(ylabel)
    style.panel_label(axes[0], panel, at_left=True)
    common.panel_csv(pd.concat(frames, ignore_index=True), item, panel)
    return axes


def auroc_lines(ax, frame):
    """Pseudobulk's unaligned bulk AUROC (an arrowhead at the left edge if below the axis) and chance."""
    value = frame.loc[(frame["key"] == "pseudobulk") & (frame["alignment"] == "none"), "value"].iloc[0]
    plots.reference_line(ax, value, F.AUROC_RANGE, S.PSEUDOBULK_LINE, -0.4)
    ax.axhline(0.5, **S.CHANCE_LINE, zorder=1)


def panel_a(fig, cell, pairs):
    if F.FIG3A["pair"] in pairs:
        draw_gap(fig, cell, F.FIG3A["pair"], "A", FIGURE)
    else:
        common.absent(fig, cell, FIGURE, "A", F.FIG3A["pair"])


def panel_b(fig, cell, pairs, bottom):
    """bottom: c isn't drawn (a blinded run), so b carries the alignment-order x label."""
    axes = offset_panel(fig, cell, pairs, silhouettes, [k for k in F.FIG3B_KEYS if k in common.kept()], "B", FIGURE, F.TEXT["silhouette"])
    first = silhouettes(pairs[0], ["pseudobulk"])   # the first pair's pseudobulk dots, named in place (b has room above them)
    plots.alignment_callout(axes[0], [(S.ALIGNMENT_X_OFFSET[r.alignment], r.value, F.ALIGNMENT_TICKS[r.alignment]) for r in first.itertuples()])


def panel_c(fig, cell, pairs):
    """"None" reads the same scores rows as Fig 2B's bulk values (stage 08's primary files; stage 09 asserts it uses
    them)."""
    axes = offset_panel(fig, cell, pairs, aurocs, [k for k in F.FIG3B_KEYS if k in common.kept()], "C", FIGURE, F.TEXT["bulk_auroc"],
                        auroc_lines, titles=False, limits=F.AUROC_RANGE)



def legend(fig):
    """b and c's key (family hues) at the top, where every figure keeps its configuration key; a's key sits beside a (panel_a)."""
    style.legend_strip(fig, [k for k in F.FIG3B_KEYS if k in common.kept()], ("pseudobulk_unaligned", "chance"), ncol=5)


def main(pairs, panels):
    functions = {"A": panel_a, "B": lambda f, c, p: panel_b(f, c, p, "C" not in panels[FIGURE]), "C": panel_c}
    with plt.rc_context(F.TEXT_RC):
        common.draw(FIGURE, functions, pairs, panels, __file__, ratios=(1.15, 0.85, 0.85), legend=legend, top_mm=24, hspace=(0.27, 0.15), left=0.08,
                    bottom=0.06)
