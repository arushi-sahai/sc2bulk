"""Fig 5: what transfer relies on.

A: per pair, each panel gene's single-cell vs. bulk phenotype shift (stage 10c), rasterized gray; pseudobulk's top 50 in
   blue; the signature genes named in figures/signature_labels.tsv marked by their pre-declared direction (up / down
   triangles; checked against 10c's signature_direction), the NAMED_PER_PAIR with the largest absolute single-cell shift named and
   the rest listed in the CSV; names sit in the empty corners (up genes upper left, down genes lower right; every rho is
   positive) with leader lines; Spearman rho in the title (one per pair, asserted constant). Each pair has its own
   symmetric scale, 1.05 x its EDGE_QUANTILE absolute shift over both axes (the shifts are standardized per pair, and a few genes with very
   large shifts would flatten the rest); genes beyond it are drawn at the edge (clipped; CSV values unclipped).
B: the same-sign share of each configuration's top 50 (ridge), with the two reference shares as short ticks
   (all panel genes; genes matched on single-cell shift size); Geneformer "n/a" where it fails its
   checks (included False) or where its deletion was triaged (absent from 10c, stage 10's inclusion.csv "triaged").
C: same-sign share vs. bulk AUROC (the mean over seeds), one point per configuration; no test.
D: the simulation (stage 12, human ileum, Crohn's): per scenario, within-single-cell vs. simulated-bulk AUROC for
   pseudobulk, the mean of log-normalized cells (✕) and the set encoders kept after triage; seeds as dots (the mean over
   read-out seeds per split e), the mean as the marker. Values that hold by construction (config/figures.BY_CONSTRUCTION)
   are drawn as gray references, never results, and flagged in the CSV. The criterion's pass flag and readings go to figure_text.json
   (figures/titles.py), not into the image."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.ticker import MaxNLocator

import common
import plots
import style
from sc2bulk.config import figures as F
from sc2bulk.config import simulation as SIM
from sc2bulk.config import style as S
from sc2bulk.config import settings

FIGURE = "Fig 5"
READS_BULK_LABELS = False
READS_SCORES = True
LABELS_FILE = settings.REPO / "figures" / "signature_labels.tsv"
K = 50
MATCHED_BELOW = 0.33                   # B: the matched-genes reference's offset below its row (rows are 1 apart)
LABEL_SLOTS = (0.95, 0.85)             # A: label heights from the corner (axes fraction), one per named gene per side
EDGE_QUANTILE = 0.99                   # A: each pair's scale
SHARE_RANGE = (0.3, 1.0)               # B, C: the same-sign share axis, zoomed to where the shares lie (S14: 0-1)
SIDE_LABEL_Y = (0.8, 0.64)              # D: the heights of the bulk and single-cell names on the first pair of points
ARROW = {"up": "\u2191", "down": "\u2193"}   # A: a named gene's signature direction, after its name
NAMED_PER_PAIR = 2                     # A: signature genes named per pair (three crowded the data; Sound Life lists 50)
SIGNATURE_SIZE = 14                    # A: the signature triangles' area (points^2); the cloud's dots are 1 and 3
WSPACE = 0.25                          # A-C share one column grid
WSPACE_A = 0.6                         # A: wider, since each pair has its own y tick labels


def concordance(pair) -> pd.DataFrame:
    return common.read_kind("concordance", "10c_concordance", pair, f"{common.LABELLED}concordance.parquet")


def signature_labels(pair) -> pd.DataFrame:
    table = common.read_tsv(LABELS_FILE)
    return table[table["pair"] == pair]


def panel_genes(pair, genes) -> pd.Series:
    """Panel gene -> the symbol signature_labels.tsv uses (10c's rule: mouse microglia uppercased, mouse organs through
    the panel's one-to-one human orthologs, human pairs as is)."""
    if pair == "hammond":
        return pd.Series([g.upper() for g in genes], index=genes)
    if pair == "tms":
        mapped = pd.concat([common.read_csv("04_panel", pair, "orthologs.csv", u, keep_default_na=False) for u in common.primary_units(pair)])
        human = dict(zip(mapped["gene"], mapped["human_symbol"]))
        return pd.Series([human.get(g, "") for g in genes], index=genes)
    return pd.Series(list(genes), index=genes)


def shares(pair, readout="ridge", k=K) -> pd.DataFrame:
    """One row per kept configuration (absent: a triaged or pending Geneformer deletion), with included."""
    table = common.pick(concordance(pair), neutral=False, readout=readout, k=k).set_index("key")
    expected = [c for c in common.kept() if c not in common.dropped(pair)]
    assert sorted(table.index) == sorted(expected), f"{pair}: concordance rows {sorted(table.index)} != {sorted(expected)}"
    return table


def panel_a(fig, cell, pairs):
    axes = plots.columns(fig, cell, len(pairs), wspace=WSPACE_A)
    csv = []
    for i, (ax, pair) in enumerate(zip(axes, pairs)):
        shifts = common.read_kind("gene_shifts", "10c_concordance", pair, f"{common.LABELLED}gene_shifts.parquet")
        rho = concordance(pair)["rho"]
        assert rho.nunique() == 1, f"{pair}: rho varies across concordance rows"
        named = signature_labels(pair).set_index("gene")["direction"]
        symbol = panel_genes(pair, shifts["gene"])
        labelled = shifts[symbol.loc[shifts["gene"]].isin(named.index).to_numpy()]
        for _, g in labelled.iterrows():
            assert g["signature_direction"] == named[symbol[g["gene"]]], f"{pair} {g['gene']}: direction differs from the signature"
        edge = 1.05 * float(np.quantile(shifts[["single_cell_shift", "bulk_shift"]].abs().to_numpy(), EDGE_QUANTILE))
        x, y = (shifts[c].clip(-edge, edge) for c in ("single_cell_shift", "bulk_shift"))
        ax.scatter(x, y, s=1, color="#b0b0b0", linewidths=0, rasterized=True)
        top = shifts["pseudobulk_top50"]
        ax.scatter(x[top], y[top], s=3, color=style.color("pseudobulk"), linewidths=0, rasterized=True)
        for direction, glyph in (("up", "^"), ("down", "v")):
            marked = labelled.index[labelled["signature_direction"] == direction]
            ax.scatter(x[marked], y[marked], s=SIGNATURE_SIZE, marker=glyph, facecolors="white", edgecolors=S.MODALITY_COLOR, linewidths=0.6,
                       zorder=3)   # white-filled, over the gene cloud
        shown = labelled.sort_values("single_cell_shift", key=abs, ascending=False).head(NAMED_PER_PAIR)
        for up, group in shown.groupby(shown["signature_direction"] == "up"):   # up genes upper left, down genes lower right
            for slot, (row, g) in zip(LABEL_SLOTS, group.sort_values("bulk_shift", ascending=not up).iterrows()):
                ax.annotate(f"{g['gene']} {ARROW[g['signature_direction']]}", (x[row], y[row]),
                            xytext=(0.03, slot) if up else (0.97, 1 - slot), textcoords="axes fraction",
                            ha="left" if up else "right", va="center", fontsize=S.MIN_FONT_SIZE, zorder=4,
                            path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")],
                            arrowprops={"arrowstyle": "-", "linewidth": 0.3, "color": S.MODALITY_COLOR, "shrinkA": 1, "shrinkB": 1})
        ax.set_xlim(-edge, edge)
        ax.set_ylim(-1.15 * edge, 1.15 * edge)   # room above and below the cloud for the names
        ax.xaxis.set_major_locator(MaxNLocator(3, symmetric=True))
        ax.yaxis.set_major_locator(MaxNLocator(3, symmetric=True))
        ax.set_title(f"{common.pair_title(pair)}\n{F.TEXT['rho'].format(rho=rho.iloc[0])}")
        if i == 0:
            ax.set_ylabel(F.TEXT["bulk_shift"])
        csv.append(shifts[["pair", "gene", "single_cell_shift", "bulk_shift", "signature_direction", "pseudobulk_top50"]].assign(
            signature=shifts["gene"].isin(labelled["gene"]), named=shifts["gene"].isin(shown["gene"]), rho=rho.iloc[0]))
    plots.centered_xlabel(axes, F.TEXT["single_cell_shift"])
    style.panel_label(axes[0], "A", at_left=True, above_pt=14)   # level with the column titles
    common.panel_csv(pd.concat(csv, ignore_index=True), FIGURE, "A")


def share_axis(ax, limits=SHARE_RANGE):
    """A same-sign share axis over limits (every drawn share inside them, asserted), with room for markers at the ends."""
    values = [x for line in ax.lines for x in np.atleast_1d(line.get_xdata()) if np.isfinite(x)]
    assert all(limits[0] <= v <= limits[1] for v in values), f"a share outside {limits}: {min(values):.3f}"
    ax.set_xlim(limits[0] - 0.03, limits[1] + 0.03)
    ticks = [0.4, 0.6, 0.8, 1] if limits[0] > 0 else [0, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])


def share_grid(fig, cell, pairs, readout="ridge", k=K, titles=False, limits=SHARE_RANGE) -> tuple:
    """Fig 5B's dot grid (configurations x pairs), reused by S14B/C: (axes, the plotted values); pair titles iff titles."""
    axes = plots.columns(fig, cell, len(pairs), wspace=WSPACE)
    keys = common.kept()
    csv = []
    for i, (ax, pair) in enumerate(zip(axes, pairs)):
        table = shares(pair, readout, k)
        alls = []
        for y, key in enumerate(keys):
            if key not in table.index or not table.loc[key, "included"]:
                style.na_cell(ax, limits[0] - 0.03, y - 0.4, limits[1] - limits[0] + 0.06, 0.8)
                csv.append({"pair": pair, "key": key, "readout": readout, "k": k, "note": F.FIGURE_TEXT["not_applicable"]})
                continue
            r = table.loc[key]
            ax.plot(r["base_matched"], y + MATCHED_BELOW, **style.BASE_MATCHED)   # under its row's point, never on it
            ax.plot(r["shared_share"], y, **style.marker(key), zorder=3)
            alls.append(r["base_all"])
            csv.append({"pair": pair, "key": key, "readout": readout, "k": k, "n_genes": r["n_genes"], "shared_share": r["shared_share"],
                        "base_all": r["base_all"], "base_matched": r["base_matched"], "included": r["included"]})
        assert len(set(alls)) == 1, f"{pair}: the all-panel-genes share varies across configurations"
        ax.axvline(alls[0], **style.BASE_LINE, zorder=0)
        share_axis(ax, limits)
        plots.rows_axis(ax, [common.label(k_) for k_ in keys], i == 0)
        if titles:
            ax.set_title(common.pair_title(pair))
    return axes, pd.DataFrame(csv)


def panel_b(fig, cell, pairs):
    axes, frame = share_grid(fig, cell, pairs)
    style.panel_label(axes[0], "B", at_left=True)
    common.panel_csv(frame, FIGURE, "B")


def panel_c(fig, cell, pairs):
    axes = plots.columns(fig, cell, len(pairs), wspace=WSPACE)
    csv = []
    for ax, pair in zip(axes, pairs):
        table = shares(pair)
        for key in [k for k in common.kept() if k in table.index and table.loc[k, "included"]]:
            bulk = common.mean_over_seeds(common.aurocs(pair, key, **common.headline_conditions(key)))
            plots.point(ax, table.loc[key, "shared_share"], bulk, F.AUROC_RANGE, style.marker(key), orientation="y")
            csv.append({"pair": pair, "key": key, "shared_share": table.loc[key, "shared_share"], "bulk_auroc": bulk})
        share_axis(ax)
    plots.share_y(axes, F.AUROC_RANGE)
    axes[0].set_ylabel(F.TEXT["bulk_auroc"])
    plots.centered_xlabel(axes, F.TEXT["shared_share"])
    style.panel_label(axes[0], "C", at_left=True)
    common.panel_csv(pd.DataFrame(csv), FIGURE, "C")


def simulation_rows() -> list:
    """Pseudobulk, the mean of log-normalized cells and the set encoders kept after triage (asserted equal to the table's rows)."""
    keys = ["pseudobulk", "mean_log_cells", *(f"{m}_cells" for m in SIM.ENCODERS)]
    criterion = common.read_json("12_simulation", SIM.PAIR, "criterion.json", SIM.UNIT)
    assert criterion["set_encoders"] == list(SIM.ENCODERS), f"criterion.json's set encoders {criterion['set_encoders']}"
    return keys


def by_construction(scenario, key, column) -> bool:
    return (scenario, key, column) in F.BY_CONSTRUCTION or (scenario, None, column) in F.BY_CONSTRUCTION


def draw_simulation(fig, cell, panel, item):
    table = common.read_kind("simulation", "12_simulation", SIM.PAIR, "simulation.parquet", SIM.UNIT)
    keys = simulation_rows()
    assert sorted(table["key"].unique()) == sorted(keys), f"simulation rows {sorted(table['key'].unique())}"
    axes = plots.columns(fig, cell, len(SIM.SCENARIOS), wspace=0.12)   # room for the three-line "Mean of log-normalized cells"
    csv = []
    for i, (ax, scenario) in enumerate(zip(axes, SIM.SCENARIOS)):
        part = table[table["scenario"] == scenario]
        for j, key in enumerate(keys):
            rows = part[part["key"] == key]
            assert sorted(rows["encoder_seed"].unique()) == list(SIM.SPLITS), f"simulation {scenario} {key}: splits"
            assert rows.groupby("encoder_seed")["readout_seed"].nunique().eq(len(common.readout_seeds())).all(), f"simulation {scenario} {key}"
            per_e = rows.groupby("encoder_seed")[["within_sc_auroc", "simulated_bulk_auroc"]].mean()
            for x, column in ((j - 0.15, "within_sc_auroc"), (j + 0.15, "simulated_bulk_auroc")):
                reference = by_construction(scenario, key, column)   # a reference, never a result
                glyph = style.by_construction(key) if reference else style.marker(key)
                ax.plot(np.full(len(per_e), x), per_e[column], **{**glyph, "alpha": S.SEED_DOT_ALPHA,
                                                                    "markersize": S.MARKER_SIZE * S.SEED_DOT_FACTOR})
                ax.plot(x, per_e[column].mean(), **glyph)
                csv.append({"scenario": scenario, "key": key, "measure": column, "auroc": per_e[column].mean(), "by_construction": reference})
                csv += [{"scenario": scenario, "key": key, "measure": column, "seed": int(e), "auroc": v, "by_construction": reference}
                        for e, v in per_e[column].items()]
            ax.plot([j - 0.15, j + 0.15], per_e.mean().to_numpy(), color=style.color(key), linewidth=0.5)
            if i == 0 and j == 0:   # the first pair of points named in place: which is single-cell, which is simulated bulk
                means = per_e.mean()
                for x, column, text_y in ((-0.15, "within_sc_auroc", SIDE_LABEL_Y[1]), (0.15, "simulated_bulk_auroc", SIDE_LABEL_Y[0])):
                    ax.annotate(F.TEXT["simulation_sides"][column], (x, means[column]), xytext=(0.3, text_y), ha="left", va="center",
                                fontsize=F.TEXT_SIZE, color="#4d4d4d", zorder=4,
                                arrowprops={"arrowstyle": "-", "connectionstyle": "angle,angleA=0,angleB=90,rad=0", "linewidth": 0.5,
                                            "color": "#4d4d4d", "shrinkA": 1, "shrinkB": 3})
        ax.axhline(0.5, **S.CHANCE_LINE)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([F.SHORT_LABELS[k] for k in keys])
        ax.set_xlim(-0.4, len(keys) - 0.6)
        ax.set_title(F.TEXT["scenario"][scenario])
    plots.share_y(axes, (0, 1.02))
    axes[0].set_ylabel(F.TEXT["auroc"])
    style.panel_label(axes[0], panel, at_left=True, above_pt=F.PAIR_TAG_PT["Fig 5D"])   # level with the pair tag, as in Fig 3A
    style.pair_tag(axes[0], common.pair_title(SIM.PAIR).replace("\n", ", "), F.PAIR_TAG_PT["Fig 5D"])   # the simulated pair
    common.panel_csv(pd.DataFrame(csv), item, panel)


def panel_d(fig, cell, pairs):
    if SIM.PAIR in pairs:
        draw_simulation(fig, cell, "D", FIGURE)
    else:
        common.absent(fig, cell, FIGURE, "D", SIM.PAIR)


def legend(fig):
    style.legend_strip(fig, common.kept(), ("base_all", "base_matched", "reference", "by_construction", "chance"), ncol=5)


def main(pairs, panels):
    with plt.rc_context(F.TEXT_RC):
        common.draw(FIGURE, {"A": panel_a, "B": panel_b, "C": panel_c, "D": panel_d}, pairs, panels, __file__, ratios=(1, 1.25, 0.7, 0.75),
                    legend=legend, top_mm=25, hspace=(0.68, 0.53, 1.12), left=0.19, bottom=0.08)
