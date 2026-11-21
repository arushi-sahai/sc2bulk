"""Fig. S11: robustness.

A: bulk predictions by group: each configuration's per-sample bulk score (stage 08's primary predictions, ridge, the
   mean over encoder and read-out seeds), by group (sc2bulk.labels); per-sample values in the CSV.
B: seed variability and ensembling: each trained configuration's bulk AUROC per encoder seed (the mean over read-out
   seeds; stage 11b's scores), its mean, and the seed ensemble's AUROC (predictions averaged over encoder seeds).
C: pseudobulk under TMM and VST, D: on smaller panels and every shared gene (stage 11b's feature_variants row);
E: within bulk subgroups (stage 11b's subgroups row, one column per pair and subset); F: Crohn's covariate-adjusted bulk
   AUROC (row 15) and ALS's covariate-adjusted within-single-cell AUROC (stage 11b's scores, bulk_subset age_adjusted)."""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

import common
import plots
import style
from sc2bulk.config import figures as F
from sc2bulk.config import style as S

FIGURE = "Fig. S11"
READS_BULK_LABELS = True               # A splits bulk predictions by group
READS_SCORES = True
NORMALIZATIONS = ("tmm", "vst")
PANELS_D = ("panel_1000", "panel_2000", "all_shared")
WSPACE = 0.55                         # A and B share one column grid (A's tick labels clear the column before)
E_PAIR_WSPACE = 0.15                  # E: between pairs' column blocks (of a block: wider than a block's 0.15 of a column)
E_HEADER_PAD = 24                     # E: points from the columns' tops to a pair's title, clear of two-line subset titles
E_HEADER_ROOM = 0.5                  # E: the top of its row left for the pair titles (of the columns' height)
F_TITLE_ROOM = 0.55                    # F: likewise for its titles, clear of e's tick labels
LETTER_MM = {p: 14 for p in "ABCDEF"}   # the letters' left edge (mm): one line, each level with its panel's top


def panel_a(fig, cell, pairs):
    axes = plots.columns(fig, cell, len(pairs), wspace=WSPACE)
    frames = []
    for ax, pair in zip(axes, pairs):
        groups = common.bulk_labels(pair).set_index("sample_id")["group"]
        for x, key in enumerate(common.kept()):
            parts = [common.read_kind("predictions", "08_readout", pair, f"{key}_predictions.parquet", u) for u in common.primary_units(pair)]
            table = common.pick(pd.concat(parts), side="bulk", encoder_seed=tuple(common.seeds_of(key)), **common.headline_conditions(key))
            score = table.groupby("sample_id")["score"].mean()
            group = groups.loc[score.index]
            for g in (0, 1):   # group by x-position (left 0, right 1), never by fill (hollow = pseudobulk input)
                v = score[group == g].to_numpy()
                ax.plot(x + (g - 0.5) * S.GROUP_X_OFFSET + (np.arange(len(v)) % 3 - 1) * 0.06, v,
                        **{**style.marker(key, S.MARKER_SIZE * S.SEED_DOT_FACTOR * 1.3, mean=False), "alpha": 0.6}, rasterized=True)
            frames.append(pd.DataFrame({"pair": pair, "key": key, "sample_id": score.index, "group": group.to_numpy(), "score": score.to_numpy()}))
        ax.set_xticks([])   # configurations by marker (the key), in key order
        ax.set_xlim(-0.6, len(common.kept()) - 0.4)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_title(common.pair_title(pair))
    axes[0].set_ylabel(F.TEXT["score"])
    plots.centered_xlabel(axes, F.TEXT["groups_by_x"])
    style.panel_label(axes[0], "A", at_left=True, left_mm=LETTER_MM["A"])
    common.panel_csv(pd.concat(frames, ignore_index=True), FIGURE, "A")


def panel_b(fig, cell, pairs):
    trained = [k for k in common.alternatives() if k not in common.NO_ENCODER_LEVEL]
    axes = plots.columns(fig, cell, len(pairs), wspace=WSPACE)
    rows = []
    for ax, pair in zip(axes, pairs):
        for x, key in enumerate(trained):
            per_seed = common.per_seed(common.aurocs(pair, key, **common.headline_conditions(key)))
            dots = per_seed.groupby("encoder_seed")["auroc"].mean()
            ensemble = common.mean_over_seeds(common.aurocs(pair, key, [-1], variant="seed_ensemble", **common.headline_conditions(key)))
            ax.plot(np.full(len(dots), x), dots.to_numpy(), **style.seed_dots(key))
            ax.plot(x, dots.mean(), **style.marker(key))
            ax.plot([x - 0.3, x + 0.3], [ensemble, ensemble], color=style.color(key), linewidth=1.0)
            rows += [{"pair": pair, "key": key, "kind": "seed", "seed": int(s), "auroc": a} for s, a in dots.items()]
            rows.append({"pair": pair, "key": key, "kind": "ensemble", "auroc": ensemble})
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(trained) - 0.4)
    plots.share_y(axes, (0, 1.02))
    axes[0].set_ylabel(F.TEXT["bulk_auroc"])
    style.panel_label(axes[0], "B", at_left=True, left_mm=LETTER_MM["B"])
    common.panel_csv(pd.DataFrame(rows), FIGURE, "B")


def feature_panel(letter, column, levels, titles):
    def draw(fig, cell, pairs):
        name = lambda r: r[column]
        values, table, na = common.secondary_columns(pairs, "feature_variants", list(levels), common.citation(FIGURE, letter), name,
                                                     key="pseudobulk", **{column: tuple(levels)})
        limits = F.XLIM_DIFFERENCE[FIGURE]
        shown = common.short_column_titles(pairs) if titles else {c: "" for c in values}
        axes = plots.forest_columns(fig, cell, values, list(levels), limits, na, titles=shown, labels=[F.FEATURE_LABELS[v] for v in levels],
                                    styles={level: "pseudobulk" for level in levels}, ticks=plots.difference_ticks(limits), wspace=0.45)
        style.panel_label(axes[0], letter, at_left=True, left_mm=LETTER_MM[letter],
                          above_pt=14 if titles else None)   # level with the column titles
        common.panel_csv(table[["pair", "key", column, "delta", "ci_low", "ci_high"]], FIGURE, letter)
    return draw


def subgroup_title(subset) -> str:
    """A subgroup column's title: the subset's level (Crohn's: inflamed or uninflamed, each vs. every control), its first
    letter capitalized, shortened by config/figures.SUBGROUP_SHORT, wrapped as a level name."""
    level = subset.split("=")[-1].split("_vs_")[0].replace("_", " ")
    level = level[0].upper() + level[1:]
    return textwrap.fill(F.SUBGROUP_SHORT.get(level, level), F.LEVEL_WRAP, break_long_words=False)


def panel_e(fig, cell, pairs):
    """One forest column per pair and subset (stage 11b's order), a pair's columns side by side under its title, wider
    gaps between pairs; the subsets' titles at 6 pt and two ticks per column (twelve narrow columns)."""
    table = common.pick(common.secondary(), neutral=False, row="subgroups", statistic="auroc_difference")
    table = table[table["pair"].isin(pairs)]
    cell = cell.subgridspec(2, 1, height_ratios=(E_HEADER_ROOM, 1), hspace=0)[1]   # the pair titles' room above
    shown = [p for p in pairs if p in set(table["pair"])]
    grid = cell.subgridspec(1, len(shown), width_ratios=[table[table["pair"] == p]["bulk_subset"].nunique() for p in shown],
                            wspace=E_PAIR_WSPACE)
    keys = common.alternatives()
    limits = F.XLIM_DIFFERENCE[FIGURE]
    axes = []
    for i, pair in enumerate(shown):
        part = table[table["pair"] == pair]
        values = {s: {} for s in part["bulk_subset"]}
        for _, r in part.iterrows():
            values[r["bulk_subset"]][r["key"]] = (r["delta"], r["ci_low"], r["ci_high"], "")
        labels = [common.label(k) for k in keys] if i == 0 else [""] * len(keys)
        titles = {s: subgroup_title(s) for s in values}
        columns = plots.forest_columns(fig, grid[0, i], values, keys, limits, set(), titles=titles, labels=labels,
                                       ticks=plots.difference_ticks(limits)[:2], wspace=0.15)
        for ax in columns:
            ax.title.set_fontsize(S.MIN_FONT_SIZE)
        block = fig.add_subplot(grid[0, i])   # the pair's title over its columns
        block.axis("off")
        block.set_title(plots.narrow(common.pair_title(pair)), pad=E_HEADER_PAD)
        axes += columns
    plots.centered_xlabel(axes, F.TEXT["difference"])
    style.panel_label(axes[0], "E", at_left=True, left_mm=LETTER_MM["E"], above_pt=E_HEADER_PAD + 6)   # level with the pair titles
    common.panel_csv(table[["pair", "bulk_subset", "key", "delta", "ci_low", "ci_high", "n0", "n1", "estimable"]], FIGURE, "E")


def panel_f(fig, cell, pairs):
    """Crohn's covariate-adjusted bulk rows (11b row 15) and ALS's covariate-adjusted within-single-cell AUROC (11b's
    scores, the mean over its estimable fits), each asserted present exactly for its pair."""
    cell = cell.subgridspec(2, 1, height_ratios=(F_TITLE_ROOM, 1), hspace=0)[1]   # the three-line titles' room above
    grid = cell.subgridspec(1, 3, wspace=0.45, width_ratios=(1, 1.1, 0.7))   # Crohn's in line with e's labels, ALS beside it
    keys = common.alternatives()
    bulk = common.pick(common.secondary(), neutral=False, row="age_adjusted", bulk_subset="age_adjusted")
    within = common.pick(common.scores(), bulk_subset="age_adjusted", side="single_cell_validation", row_kind="configuration")
    assert set(bulk["pair"]) == set(F.AGE_ADJUSTED["bulk"]) and set(within["pair"]) == set(F.AGE_ADJUSTED["within"]), \
        f"S11F: age-adjusted rows for {sorted(set(bulk['pair']))} and {sorted(set(within['pair']))}"
    bulk_pairs = [p for p in pairs if p in F.AGE_ADJUSTED["bulk"]]
    values = {p: {r["key"]: (r["delta"], r["ci_low"], r["ci_high"], "") for _, r in bulk[bulk["pair"] == p].iterrows()} for p in bulk_pairs}
    frames = [bulk[bulk["pair"].isin(bulk_pairs)][["pair", "key", "delta", "ci_low", "ci_high"]].assign(part="bulk, age-adjusted")]
    if values:
        limits = F.XLIM_DIFFERENCE[FIGURE]
        axes = plots.forest_columns(fig, grid[0, 0], values, keys, limits, set(),
                                    titles={p: f"{common.pair_title(p)}\n{F.TEXT['age_adjusted']}" for p in values},
                                    labels=[common.label(k) for k in keys], ticks=plots.difference_ticks(limits))
        axes[0].set_xlabel(F.TEXT["difference_short"])
    ax = fig.add_subplot(grid[0, 1])
    rows = []
    for pair in [p for p in pairs if p in F.AGE_ADJUSTED["within"]]:
        for x, key in enumerate(common.kept()):
            fits = within[(within["pair"] == pair) & (within["key"] == key)]
            assert len(fits), f"S11F {pair} {key}: no estimable age-adjusted fit"
            adjusted = fits["auroc"].mean()
            raw = common.mean_over_seeds(common.aurocs(pair, key, common.encoder_seeds(), side="single_cell_validation", **common.headline_conditions(key)))
            ax.plot([x - 0.15, x + 0.15], [raw, adjusted], color=style.color(key), linewidth=0.5)
            ax.plot(x + 0.15, adjusted, **style.marker(key))
            rows.append({"pair": pair, "key": key, "within_auroc": raw, "within_auroc_age_adjusted": adjusted, "fits": len(fits)})
        ax.set_title(f"{common.pair_title(pair)}\n{F.TEXT['age_adjusted']}")
    ax.set_xticks([])   # configurations by marker (the key), in key order
    ax.set_xlim(-0.6, len(common.kept()) - 0.4)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_ylabel(F.TEXT["within_auroc"])
    ax.set_xlabel(F.TEXT["raw_to_adjusted"])
    block = fig.add_subplot(cell)   # the letter over both parts of f
    block.axis("off")
    style.panel_label(block, "F", at_left=True, left_mm=LETTER_MM["F"], above_pt=14)   # level with the titles
    frames.append(pd.DataFrame(rows).assign(part="within single-cell, age-adjusted"))
    common.panel_csv(pd.concat(frames, ignore_index=True), FIGURE, "F")


def legend(fig):
    style.legend_strip(fig, common.kept(), ("ensemble", "pseudobulk_zero"), ncol=6)


def main(pairs, panels):
    with plt.rc_context(F.TEXT_RC):
        common.draw(FIGURE, {"A": panel_a, "B": panel_b, "C": feature_panel("C", "variant", NORMALIZATIONS, True),
                             "D": feature_panel("D", "feature_set", PANELS_D, False), "E": panel_e, "F": panel_f}, pairs, panels, __file__,
                    legend=legend, ratios=(1, 0.85, 0.3, 0.42, 1.45, 1.45), top_mm=19, hspace=(0.5, 0.77, 0.5, 0.71, 0.5), left=0.17,
                    bottom=0.062)
