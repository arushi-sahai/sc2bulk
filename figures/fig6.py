"""Fig 6: what learned representations track.

A: one grayscale heatmap per pair: rows pseudobulk and PCA-64, a separator, then the nine others (minus cuts); columns
   depth, mitochondrial fraction, composition, phenotype (stage 12a): the split-mean value (12a's own aggregation,
   properties.split_mean), printed, negative R² shown as 0 (raw in the CSV), the phenotype column on its own 0.5-1
   scale, n/a cells hatched (config/figures.NOT_APPLICABLE); each pair's chance level as a footnote under its column:
   the largest null_q95 over the representations (each row's, asserted constant over its splits, in the CSV). With every pair present, the licence counts are recomputed by
   properties.licence and asserted equal to licence.json's.
B: left, CytoSet, cells on FIG6B_PAIR: the FIG6B_TOP (10) genes with the largest modality-direction attribution (stage 10),
   annotated with the bulk-minus-single-cell level difference; right, each learned cells configuration's share of top
   modality genes that are large-difference genes or markers (stage 10's modality_summary; human blood, aging: the
   folds' mean), with the panel's base rate; Geneformer "n/a" where it fails its checks (included False, as Fig 5B),
   "n/a (not affordable)" where its deletion was triaged, "n/a (pending)" before it has run.
C: Fig 2B's format for the one-type rows (stage 08b's one_type family; trained models at seeds 0-4) on the pairs with
   a one-type store (asserted equal to the pairs with one-type scores); mouse microglia is n/a.
draw_modality_genes() is Fig. S15's panel function."""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

import common
import plots
import style
from sc2bulk import properties
from sc2bulk.config import settings
from sc2bulk.config import figures as F
from sc2bulk.config import properties as P
from sc2bulk.config import style as S
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

FIGURE = "Fig 6"
READS_BULK_LABELS = False
READS_SCORES = True
SPREAD = 0.3                           # C: the configurations dodged over this width at each end
LINE_EDGE_X = -0.35                    # C: where a reference line past the y range is drawn as an arrowhead
TAG_PT = 16                            # B: the exemplar pair's name over the exemplar (points above its axes), as Fig 5D
TITLE_WIDTH = 15                       # B: a right-hand column title rewrapped as one text, so every pair takes two lines
EXEMPLAR_TOP = 8                       # B: the exemplar's top genes (S15 shows 10 per configuration); 8 leaves c its height
SCALE_X = (92, 150)                    # A: the two scale bars' left edges (mm), labels to their left
SCALE_WIDTH_MM, SCALE_BAR_MM, SCALE_TOP_MM = 17, 1.6, 15.5   # A: each scale bar's size and its top (mm from the figure's top)


def property_values(pair) -> pd.DataFrame:
    """pair, key, property, r2 (the split mean) and null_q95, every row and unit read."""
    keys = [*F.FIG6A_ROWS, *[k for k in common.kept() if k not in F.FIG6A_ROWS]]
    frames = [common.read_kind("property_r2", "12a_properties", pair, f"{k}_property_r2.parquet", u) for k in keys for u in common.primary_units(pair)]
    table = pd.concat(frames, ignore_index=True)
    values = properties.split_mean(table)
    assert table.groupby(["key", "property"])["null_q95"].nunique().le(1).all(), f"{pair}: null_q95 varies within a row (12a writes one per pair)"
    nulls = table.groupby(["pair", "key", "property"])["null_q95"].first().rename("null_q95").reset_index()
    return values.merge(nulls, on=["pair", "key", "property"])


def check_licence(values: pd.DataFrame, pairs):
    """licence.json's per-pair counts, recomputed from the same split means (only when every pair is drawn)."""
    licence = common.read_json("12a_properties", "", "licence.json")
    assert licence["title"] == (P.LICENSED_TITLE if licence["licensed"] else P.DESCRIPTIVE_TITLE)
    if list(pairs) == list(PAIR_ORDER):
        mine = properties.licence(values, licence["present"])
        assert mine["per_pair"] == licence["per_pair"] and mine["licensed"] == licence["licensed"], "Fig 6A disagrees with licence.json"


def panel_a(fig, cell, pairs):
    axes = plots.columns(fig, cell, len(pairs), wspace=0.1)
    keys = [*F.FIG6A_ROWS, *[k for k in common.kept() if k not in F.FIG6A_ROWS]]
    frames = []
    for i, (ax, pair) in enumerate(zip(axes, pairs)):
        values = property_values(pair)
        frames.append(values)
        table = values.pivot(index="key", columns="property", values="r2")
        assert sorted(table.index) == sorted(keys), f"Fig 6A {pair}: rows {sorted(table.index)}"
        na = {(r, c) for r, k in enumerate(keys) for c, p in enumerate(P.PROPERTIES) if (pair, p) in F.NOT_APPLICABLE["Fig 6A"]}
        for (r, c) in na:
            assert P.PROPERTIES[c] not in table.columns or not np.isfinite(table.loc[keys[r], P.PROPERTIES[c]]), f"Fig 6A {pair}: an n/a cell has a value"
        shown = table.reindex(index=keys, columns=list(P.PROPERTIES)).clip(lower=0)   # negative R² shown as 0 (raw in the CSV)
        chance = values.groupby("property")["null_q95"].max()
        limits = [F.PHENOTYPE_RANGE if p == "phenotype" else (0, 1) for p in P.PROPERTIES]
        plots.heatmap(ax, shown.to_numpy(), S.HEATMAP_CMAP, limits, [common.label(k) for k in keys], [F.PROPERTY_SHORT[p] for p in P.PROPERTIES],
                      na=na)
        for c, p in enumerate(P.PROPERTIES):   # each column's chance level, a text row under the heatmap
            if p in chance and np.isfinite(chance[p]):
                ax.text(c, len(keys), f"{chance[p]:.2f}", ha="center", va="center", fontsize=S.MIN_FONT_SIZE)
        ax.set_ylim(len(keys) + 0.5, -0.5)
        ax.set_yticks(range(len(keys) + 1))
        ax.set_yticklabels([common.label(k) for k in keys] + [F.TEXT["chance_row"]] if i == 0 else [])
        ax.axhline(len(F.FIG6A_ROWS) - 0.5, color="black", linewidth=1.0)
        ax.set_title(common.pair_title(pair))
    check_licence(pd.concat(frames, ignore_index=True), pairs)
    style.panel_label(axes[0], "A", at_left=True)
    common.panel_csv(pd.concat(frames, ignore_index=True), FIGURE, "A")


def scale_bars(fig):
    """A's two gray scales, in the band between the key and a's titles: R² (0-1) for the technical columns, AUROC (0.5-1)
    for the phenotype column."""
    width, height = fig.get_size_inches() * 25.4
    for (x, label, limits) in ((SCALE_X[0], F.TEXT["scale_r2"], (0, 1)), (SCALE_X[1], F.TEXT["scale_phenotype"], F.PHENOTYPE_RANGE)):
        cax = fig.add_axes((x / width, 1 - (SCALE_TOP_MM + SCALE_BAR_MM) / height, SCALE_WIDTH_MM / width, SCALE_BAR_MM / height))
        fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(*limits), cmap=S.HEATMAP_CMAP), cax=cax, orientation="horizontal",
                     ticks=list(limits))
        cax.set_xticklabels([f"{v:g}" for v in limits])
        cax.tick_params(length=1.5, pad=1)
        cax.text(-0.04, 0.5, label, transform=cax.transAxes, ha="right", va="center", fontsize=F.TEXT_SIZE)


def draw_modality_genes(ax, pair, key, size=F.TEXT_SIZE, n=F.FIG6B_TOP) -> pd.DataFrame:
    """The FIG6B_TOP genes with the largest modality-direction attribution, annotated with the level difference (the
    primary units' mean) in a number column under its own header row; gene names and numbers at size pt."""
    ranking = common.read_kind("attributions", "10_attribution", pair, f"{key}_ranking.parquet")
    top = common.pick(ranking, neutral=False, target="modality_direction").sort_values("rank").head(n)
    assert len(top) == n, f"{pair} {key}: {len(top)} modality-direction genes"
    level = pd.concat([common.read_csv("10_attribution", pair, "level_difference.csv", u) for u in common.primary_units(pair)])
    delta = level.groupby("gene")["delta"].mean().loc[top["gene"]]
    y = np.arange(len(top))
    ax.barh(y, top["attribution"], color=style.color(key), height=0.7, linewidth=0)
    for yy, d in zip(y, delta):   # the level difference in a text column right of the axes
        ax.text(1.03, yy, f"{d:+.2f}".replace("-", "\N{MINUS SIGN}"), va="center", fontsize=size, transform=ax.get_yaxis_transform())
    plots.rows_axis(ax, list(top["gene"]), True)
    ax.set_ylim(len(top) - 0.5, -2.2)   # a header row above the bars for the number column
    ax.text(1.03, -0.7, F.TEXT["level_difference"], va="bottom", fontsize=S.MIN_FONT_SIZE, transform=ax.get_yaxis_transform())
    ax.tick_params(axis="y", labelsize=size)
    ax.xaxis.set_major_locator(MaxNLocator(2))
    return pd.DataFrame({"pair": pair, "key": key, "gene": top["gene"].to_numpy(), "attribution": top["attribution"].to_numpy(),
                         "rank": top["rank"].to_numpy(), "level_difference": delta.to_numpy()})


def panel_b(fig, cell, pairs):
    grid = cell.subgridspec(1, 2, width_ratios=(0.68, 4.32), wspace=0.55)   # room for the exemplar's number column
    assert F.FIG6B_PAIR in pairs, f"Fig 6B's exemplar pair {F.FIG6B_PAIR} isn't drawn"
    left = fig.add_subplot(grid[0, 0])
    csv = [draw_modality_genes(left, F.FIG6B_PAIR, "cytoset_cells", n=EXEMPLAR_TOP).assign(part="exemplar")]
    exemplar = common.pair_title(F.FIG6B_PAIR).replace("\n", ", ")
    left.set_title(common.label("cytoset_cells"), fontsize=S.MIN_FONT_SIZE)
    style.pair_tag(left, exemplar, TAG_PT)   # the exemplar pair, as in Figs 3A and 5D
    left.set_xlabel(F.TEXT["attribution"])
    axes = plots.columns(fig, grid[0, 1], len(pairs), wspace=0.3)
    keys = [k for k in common.kept() if k.endswith("_cells")]
    for i, (ax, pair) in enumerate(zip(axes, pairs)):
        summary = common.read_csv("10_attribution", pair, "modality_summary.csv")
        summary = summary[summary["fold"] == (-1 if len(common.primary_units(pair)) > 1 else common.primary_units(pair)[0][1])].set_index("configuration")
        dropped, excluded = common.dropped(pair), common.excluded(pair)
        assert summary["base_either"].nunique() == 1, f"Fig 6B {pair}: the base rate varies across configurations"
        ax.axvline(summary["base_either"].iloc[0], **style.BASE_LINE, zorder=0)   # the pair's base rate
        for y, key in enumerate(keys):
            if key not in summary.index:
                assert key in dropped, f"Fig 6B {pair} {key}: no modality summary"
                style.na_cell(ax, -0.05, y - 0.4, 1.1, 0.8, F.FIGURE_TEXT[F.DROPPED_TEXT[dropped[key]]])
                continue
            if key in excluded:   # Geneformer's checks failed: its genes aren't read (as Fig 5B)
                style.na_cell(ax, -0.05, y - 0.4, 1.1, 0.8)
                csv.append(pd.DataFrame([{"pair": pair, "key": key, "note": F.FIGURE_TEXT["not_applicable"], "part": "summary"}]))
                continue
            r = summary.loc[key]
            ax.plot(r["share_either"], y, **style.marker(key), zorder=3)
            csv.append(pd.DataFrame([{"pair": pair, "key": key, "share_either": r["share_either"], "base_either": r["base_either"],
                                      "share_large": r["share_large"], "share_marker": r["share_marker"], "part": "summary"}]))
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 1])
        plots.rows_axis(ax, [common.label(k) for k in keys], False)   # rows named by their markers (the key): the six columns
        ax.set_title(textwrap.fill(common.pair_title(pair).replace("\n", " "), TITLE_WIDTH), fontsize=S.MIN_FONT_SIZE)   # two lines each
    plots.centered_xlabel(axes, F.TEXT["modality_share"])
    style.panel_label(left, "B", at_left=True, above_pt=TAG_PT)   # level with the pair tag
    common.panel_csv(pd.concat(csv, ignore_index=True), FIGURE, "B")


def cell_type(pair) -> str:
    """Fig 6C's column subtitle: the pair's one-type annotation as displayed."""
    one_type = PAIRS[pair]["one_type"]
    return F.ONE_TYPE_PER_STRATUM if one_type["by"] else F.FIG6C_TYPE_NAMES[one_type["types"][None][0]]


def panel_c(fig, cell, pairs):
    one_type = [p for p in pairs if (p, "all") not in F.NOT_APPLICABLE["Fig 6C"]]
    assert one_type == [p for p in pairs if "one_type" in PAIRS[p]], "Fig 6C's pairs differ from the pairs with a one-type store"
    with_scores = sorted(set(common.pick(common.scores(), neutral=False, cell_subset="one_type")["pair"]) & set(pairs))
    if settings.SMOKE:   # 08b skips a unit's one-type rows where its store falls short (one_type_absent): the pair isn't drawn
        one_type = [p for p in one_type if p in with_scores]
    assert with_scores == sorted(one_type), f"Fig 6C: one-type scores for {with_scores}"
    axes = plots.columns(fig, cell, len(one_type))
    csv = []
    for i, (ax, pair) in enumerate(zip(axes, one_type)):
        rows = []
        for key in common.kept():
            seeds = common.seeds_of(key, "H")
            within_seeds = seeds if common.within_side(pair) == "single_cell_heldout" else common.encoder_seeds("H")
            conditions = {**common.headline_conditions(key), "cell_subset": "one_type"}
            conditions.pop("n_cells")   # the bag is min(500, available): the rows' own n_cells
            within = common.aurocs(pair, key, within_seeds, side=common.within_side(pair), **conditions)
            bulk = common.aurocs(pair, key, seeds, **conditions)
            left, right = common.mean_over_seeds(within), common.mean_over_seeds(bulk)
            rows.append((key, left, right, *(common.per_seed(t).groupby("encoder_seed")["auroc"].mean().to_numpy() for t in (within, bulk))))
            csv.append({"pair": pair, "cell_type": cell_type(pair), "key": key, "within_auroc": left, "bulk_auroc": right})
        plots.slope(ax, rows, limits=F.AUROC_RANGE, spread=SPREAD)
        reference = common.mean_over_seeds(common.aurocs(pair, "within_bulk_reference", [-1], row_kind="reference", n_cells=0, pooling="native"))
        plots.reference_line(ax, rows[0][2], F.AUROC_RANGE, S.PSEUDOBULK_LINE, LINE_EDGE_X, zorder=1.5)
        plots.reference_line(ax, reference, F.AUROC_RANGE, {**S.WITHIN_BULK_LINE, "linewidth": 1.6}, LINE_EDGE_X)   # wider and behind, so
        # the dashed blue line stays visible where the two coincide (Fig 2B's format)
        csv.append({"pair": pair, "cell_type": cell_type(pair), "key": "within_bulk_reference", "bulk_auroc": reference})
        ax.set_xlim(-0.45, 1.3)
        ax.set_title(f"{common.pair_title(pair)}\n{cell_type(pair)}")   # the one-type cell type
    plots.share_y(axes, F.AUROC_RANGE)
    axes[0].set_ylabel(F.TEXT["auroc"])
    style.panel_label(axes[0], "C", at_left=True, above_pt=14)
    common.panel_csv(pd.DataFrame(csv), FIGURE, "C")


def legend(fig):
    style.legend_strip(fig, common.kept(), ("base_all", "pseudobulk_bulk", "within_bulk"), ncol=5)
    scale_bars(fig)   # a's gray scales sit with the key


def main(pairs, panels):
    with plt.rc_context(F.TEXT_RC):
        common.draw(FIGURE, {"A": panel_a, "B": panel_b, "C": panel_c}, pairs, panels, __file__, ratios=(1.2, 0.92, 0.82), legend=legend,
                    top_mm=30, hspace=(0.88, 0.97), left=0.19, bottom=0.06)
