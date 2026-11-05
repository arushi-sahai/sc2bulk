"""Fig. S10: configuration searches and training checks.

A: each search candidate's single-cell validation BCE (stage 05; the mean over search seeds), per encoder, per fold for
   human blood, aging (the merged configurations included, per fold: they differ between folds); log scale, chance (ln 2)
   dotted;
B: bulk AUROC vs. validation BCE for every candidate and search seed (stage 11b's scores, row_kind candidate), the
   oracle (stage 11b's row 13, selected on bulk) marked "for reference only, never selected";
C: outside the search: each encoder's own published training procedure vs. pseudobulk (stage 11b's own_recipe row), and CytoSet's MSE
   objective (stage 11b's best-candidate member objective_mse; aging pairs only; its unadjusted CI, no Holm
   glyph; never pooled);
D: training checks per encoder seed (stage 06 row A's records: d* against the collapse floor, failing seeds gray);
E: scVI's search (stage 05v: each candidate's validation ELBO, the mean over search seeds, minus the base candidate's).
A, D and E are label-free (drawn in blinded runs)."""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

import common
import plots
import style
from sc2bulk.config import configurations
from sc2bulk.config import figures as F
from sc2bulk.config import style as S
from sc2bulk.config import training as T

FIGURE = "Fig. S10"
READS_BULK_LABELS = False
READS_SCORES = True
ENCODERS = ("cytoset", "set_transformer", "scrat")
WSPACE = 0.45                          # A, B, D, E share one column grid
SPREAD = 0.6                           # A: the width each encoder's candidates spread over
MSE = "objective_mse"                  # C: CytoSet's MSE objective (stage 11b's key)
BCE_FLOOR = 0.05                       # B: single-seed BCEs below it (down to ~1e-58) drawn at the left edge
B_ROOM = 0.3                           # B: the share of its row (vs. 1 for its plots) left below for C's column titles
C_LETTER_Y = 1.3                       # C: its letter level with the pair titles, clear of the row labels
LETTER_MM = {p: 5 for p in "ABCDE"}   # each letter's left edge (mm), just left of its panel's labels


def encoders() -> list:
    return [e for e in ENCODERS if f"{e}_cells" in common.alternatives()]


def candidates(pair) -> pd.DataFrame:
    """Every candidate row of every encoder, unit and search seed (stage 05's candidates files, merged ones included)."""
    frames = []
    for unit in common.primary_units(pair):
        for encoder in encoders():
            files = common.names("05_search", pair, f"{encoder}/candidates*_s*.parquet", unit)
            assert {f"{encoder}/candidates_s{i}.parquet" for i in common.search_seeds()} <= set(files), f"S10 {pair} {unit} {encoder}: {files}"
            frames += [common.read_kind("candidates", "05_search", pair, f, unit) for f in files]
    return pd.concat(frames, ignore_index=True)


def offsets(table) -> pd.Series:
    """x per row: the encoder's position, its candidates (every fold's) spread across it by value."""
    position = table["model"].map({e: i for i, e in enumerate(encoders())})
    rank = table.groupby("model")["validation_bce"].rank(method="first") - 1
    size = table.groupby("model")["validation_bce"].transform("size")
    return position + (rank - (size - 1) / 2) * SPREAD / (size - 1).clip(lower=1)


def panel_a(fig, cell, pairs):
    axes = plots.columns(fig, cell, len(pairs), wspace=WSPACE)
    frames = []
    for ax, pair in zip(axes, pairs):
        table = candidates(pair)
        mean = table.groupby(["model", "fold", "key", "change_group", "merged"], as_index=False)["validation_bce"].mean()
        x = offsets(mean)
        for encoder, part in mean.groupby("model"):
            ax.plot(x[part.index], part["validation_bce"], **style.marker(f"{encoder}_cells"))
        ax.axhline(math.log(2), **S.CHANCE_LINE, zorder=1)
        ax.set_yscale("log")   # one diverged candidate (mouse microglia) reaches ~10^4
        ax.yaxis.set_minor_formatter(NullFormatter())   # decades labeled only
        ax.set_xticks(range(len(encoders())))
        ax.set_xticklabels([])
        ax.set_xlim(-0.6, len(encoders()) - 0.4)
        ax.set_title(common.pair_title(pair))
        frames.append(table[["pair", "fold", "model", "key", "change_group", "change_value", "encoder_seed", "validation_bce", "did_train",
                             "merged"]])
    plots.share_y(axes)   # fitted to the candidates and chance (ln 2)
    axes[0].set_ylabel(F.TEXT["validation_bce"])
    plots.centered_xlabel(axes, F.TEXT["encoder_axis"])
    style.panel_label(axes[0], "A", at_left=True, left_mm=LETTER_MM["A"])
    common.panel_csv(pd.concat(frames, ignore_index=True), FIGURE, "A")


def panel_b(fig, cell, pairs):
    cell = cell.subgridspec(2, 1, height_ratios=(1, B_ROOM), hspace=0)[0]
    axes = plots.columns(fig, cell, len(pairs), wspace=WSPACE)
    oracle = common.pick(common.secondary(), neutral=False, row="oracle", pair=tuple(pairs))
    frames = []
    for ax, pair in zip(axes, pairs):
        table = candidates(pair)
        scores = common.pick(common.scores(), pair=pair, row_kind="candidate", side="bulk", bulk_subset="all", n_cells=T.FROZEN_NCELL,
                             fold_set=common.primary_units(pair)[0][0], model=tuple(encoders()))
        auroc = scores.groupby(["model", "fold", "key", "encoder_seed"], as_index=False)["auroc"].mean()
        merged = table.merge(auroc, on=["model", "fold", "key", "encoder_seed"], how="left", validate="one_to_one")
        assert merged["auroc"].notna().all(), f"S10B {pair}: candidates without bulk scores {sorted(set(merged.loc[merged['auroc'].isna(), 'key']))}"
        for encoder, part in merged.groupby("model"):
            ax.plot(part["validation_bce"].clip(lower=BCE_FLOOR), part["auroc"], **style.seed_dots(f"{encoder}_cells"))
        chosen = oracle[oracle["pair"] == pair]
        for _, r in chosen.iterrows():
            point = merged[(merged["model"] == r["model"]) & (merged["key"] == r["key"])][["validation_bce", "auroc"]].mean()
            x = max(point["validation_bce"], BCE_FLOOR)
            ax.plot(x, point["auroc"], **style.marker(f"{r['model']}_cells"))
            ax.plot(x, point["auroc"], **style.ORACLE)
        ax.axvline(math.log(2), **S.CHANCE_LINE, zorder=1)
        ax.set_xscale("log")   # each column's own range (up to ~10^4 in mouse microglia)
        ax.set_xlim(left=BCE_FLOOR / 1.2)
        ax.xaxis.set_major_locator(LogLocator(numticks=4))
        ax.xaxis.set_minor_formatter(NullFormatter())   # decades labeled only (every pair spans ln 2 to above 1)
        frames.append(merged[["pair", "fold", "model", "key", "encoder_seed", "validation_bce", "auroc"]].assign(
            oracle=pd.Series(list(zip(merged["model"], merged["key"]))).isin(set(zip(chosen["model"], chosen["key"]))).to_numpy()))
    plots.share_y(axes, (0, 1.02))
    axes[0].set_ylabel(F.TEXT["bulk_auroc"])
    plots.centered_xlabel(axes, F.TEXT["validation_bce"])
    style.panel_label(axes[0], "B", at_left=True, left_mm=LETTER_MM["B"])
    common.panel_csv(pd.concat(frames, ignore_index=True), FIGURE, "B")


def mse_rows(values, na) -> pd.DataFrame:
    """CytoSet's MSE objective, added to panel C's values in place: stage 11b's best-candidate member, one
    row per aging pair (the others n/a, config/figures.NOT_APPLICABLE); the pooled cell is always n/a; no Holm glyph."""
    rows = common.pick(common.secondary(), neutral=False, row="best_candidate", key=MSE, model="cytoset",
                       statistic="auroc_difference")
    for column in values:
        part = rows[rows["pair"] == column]
        if column == "pooled" or (column, MSE) in F.NOT_APPLICABLE[common.citation(FIGURE, "C")]:
            assert part.empty, f"S10C {column}: an MSE-objective row where none is expected"
            na.add((column, MSE))
        else:
            assert len(part) == 1, f"S10C {column}: MSE-objective rows {len(part)}"
            r = part.iloc[0]
            values[column][MSE] = (r["delta"], r["ci_low"], r["ci_high"], "")
    return rows[rows["pair"].isin(values)]


def panel_c(fig, cell, pairs):
    names = [f"{e}_own_recipe" for e in encoders()]
    values, table, na = common.secondary_columns(pairs, "own_recipe", names, common.citation(FIGURE, "C"))
    labels = [configurations.MODELS[e]["label"] for e in encoders()]
    if "cytoset" in encoders():
        table = pd.concat([table, mse_rows(values, na)], ignore_index=True)
        names, labels = [*names, MSE], [*labels, F.TEXT[MSE]]
    limits = F.XLIM_DIFFERENCE[FIGURE]
    axes = plots.forest_columns(fig, cell, values, names, limits, na, titles=common.short_column_titles(pairs), labels=labels,
                                styles={**{f"{e}_own_recipe": f"{e}_cells" for e in encoders()}, MSE: "cytoset_cells"},
                                ticks=plots.difference_ticks(limits), wspace=0.45)
    plots.centered_xlabel(axes, F.TEXT["difference"])
    style.panel_label(axes[0], "C", at_left=True, left_mm=LETTER_MM["C"], y=C_LETTER_Y)
    common.panel_csv(table[["pair", "key", "model", "delta", "ci_low", "ci_high"]], FIGURE, "C")


def panel_d(fig, cell, pairs):
    axes = plots.columns(fig, cell, len(pairs), wspace=WSPACE)
    rows = []
    for ax, pair in zip(axes, pairs):
        for unit in common.primary_units(pair):
            for j, encoder in enumerate(encoders()):
                for e in common.encoder_seeds():
                    record = common.read_json("06_train", pair, f"A/{encoder}/record_s{e}.json", unit)
                    rows.append({"pair": pair, "fold": unit[1], "model": encoder, "encoder_seed": e, "d_star": record["margin_d_star"],
                                 "p": record["margin_p"], "did_train": record["did_train"], "reason": record["did_train_reason"]})
        frame = pd.DataFrame([r for r in rows if r["pair"] == pair])
        for j, encoder in enumerate(encoders()):   # encoders side by side at each seed
            part = frame[frame["model"] == encoder]
            x = part["encoder_seed"] + plots.dodge(j, len(encoders()), 0.5)
            passed = part["did_train"].to_numpy(bool)
            ax.plot(x[passed], part.loc[passed, "d_star"], **style.marker(f"{encoder}_cells"))
            ax.plot(x[~passed], part.loc[~passed, "d_star"], **{**style.marker(f"{encoder}_cells"), "color": S.BY_CONSTRUCTION_COLOR,
                                                                 "markerfacecolor": S.BY_CONSTRUCTION_COLOR, "markeredgecolor": S.BY_CONSTRUCTION_COLOR})
        ax.axhline(T.MARGIN_FLOOR * math.log(2), **S.CHANCE_LINE)
        ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    plots.share_y(axes)
    axes[0].yaxis.set_major_locator(MaxNLocator(4))
    axes[0].set_ylabel(F.TEXT["d_star"])
    plots.centered_xlabel(axes, F.TEXT["encoder_seed"])
    style.panel_label(axes[0], "D", at_left=True, left_mm=LETTER_MM["D"])
    common.panel_csv(pd.DataFrame(rows), FIGURE, "D")


def panel_e(fig, cell, pairs):
    axes = plots.columns(fig, cell, len(pairs), wspace=WSPACE)
    frames = []
    for ax, pair in zip(axes, pairs):
        units = common.primary_units(pair)
        for k, unit in enumerate(units):   # folds side by side
            table = pd.concat([common.read_csv("05v_scvi_search", pair, f"elbo_s{i}.csv", unit).assign(seed=i) for i in common.search_seeds()])
            mean = table.groupby("candidate", sort=True)["validation_elbo"].mean()
            delta = mean - mean.loc["base"]   # relative to the base candidate, so the pairs share one axis
            ax.plot(np.arange(len(mean)) + plots.dodge(k, len(units), 0.4), delta.to_numpy(), **style.marker("scvi_cells", S.MARKER_SIZE * 1.5))
            ax.set_xticks(range(len(mean)))
            ax.set_xticklabels([F.S10E_CANDIDATE_LABELS[c] for c in mean.index], rotation=45, ha="right", rotation_mode="anchor")
            frames.append(table.rename(columns={"group": "change_group"}).assign(pair=pair, fold=unit[1]))
        ax.set_xlim(-0.6, len(mean) - 0.4)
        ax.axhline(0, **S.DIVIDER, zorder=1)
    plots.share_y(axes)
    axes[0].yaxis.set_major_locator(MaxNLocator(4))
    axes[0].set_ylabel(F.TEXT["delta_elbo"])
    style.panel_label(axes[0], "E", at_left=True, left_mm=LETTER_MM["E"])
    common.panel_csv(pd.concat(frames, ignore_index=True), FIGURE, "E")


def legend(fig):
    style.legend_strip(fig, [*(f"{e}_cells" for e in encoders()), "scvi_cells"], ("pseudobulk_zero", "chance", "oracle"), ncol=4)


def main(pairs, panels):
    with plt.rc_context(F.TEXT_RC):
        common.draw(FIGURE, {"A": panel_a, "B": panel_b, "C": panel_c, "D": panel_d, "E": panel_e}, pairs, panels, __file__, legend=legend,
                    top_mm=16, ratios=(0.95, 0.85 * (1 + B_ROOM), 0.85, 0.9, 1), hspace=(0.55, 0.89, 0.67, 0.71), left=0.12, bottom=0.07)
