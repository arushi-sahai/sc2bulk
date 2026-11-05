"""Fig 4: shared design choices of the set encoders.

Every value is AUROC minus full pseudobulk's (per fold, then the mean over folds), the mean over seeds; dots are seeds.
A: cells per bag at each set encoder's frozen configuration (stage 06 row F; n = 500 is row A at seeds 0-4) on a log
   axis, the foundation models' and scVI's mean of n cells, the pseudobulk of the same cells (solid, in pseudobulk's color, circles);
   published bag sizes as open marks; the ring around each set encoder's n = 500 point. The foundation models' "all"
   point iff config/figures.FIG4A_FM_ALL_AVAILABLE names them, and CytoSet's "all" n.e. iff its frozen configuration
   has the age-similarity loss in a unit, both asserted against the scores table both ways.
B, C: per set encoder the base and the one-change candidates of the group (stage 05's change_group; B: CytoSet's output
   pooling, Set Transformer and ScRAT their native pooling; the foundation models mean / max, scVI mean; C: input
   normalization), then a dotted divider and the ring point (the frozen configuration's 5-seed value, = A's n = 500
   point; human blood, aging: the mean over folds of each fold's own selection); "merged" is never drawn here.
The per-pair best-candidate test (stage 11b's row 18) is printed above A's columns, with the Holm value across pairs
beside it when any pair is significant. The difference axes use Fig 2's pre-declared range
(config/figures.XLIM_DIFFERENCE) and clip as Fig 2 does: a value past it is an arrowhead at the edge, seed dots past it
are left out (the CSVs keep every value)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
import plots
import style
from sc2bulk.config import configurations, representations
from sc2bulk.config import figures as F
from sc2bulk.config import style as S
from sc2bulk.config import training as T

FIGURE = "Fig 4"
READS_BULK_LABELS = False
READS_SCORES = True
ALL = representations.ALL_AVAILABLE_CAP   # the "all" point's x position (the store's cap)
FMS = ("geneformer", "scgpt")
DODGE_DECADES = 0.2                    # Fig 4A: the lines' x spread at each bag size (log10 units)


def base_by_fold(pair) -> pd.Series:
    """Full pseudobulk's AUROC per fold (the mean over read-out seeds)."""
    return common.aurocs(pair, "pseudobulk", **common.headline_conditions("pseudobulk")).groupby("fold")["auroc"].mean()


def difference(pair, key, seeds, row_kind="configuration", **conditions) -> tuple:
    """(the mean over seeds of AUROC - full pseudobulk's, the per-seed values): per fold, then the mean over folds."""
    raw = common.aurocs(pair, key, seeds, row_kind=row_kind, **conditions)
    per = common.per_seed(raw)
    per["diff"] = per["auroc"] - per["fold"].map(base_by_fold(pair))
    dots = per.groupby(["encoder_seed", "fold"])["diff"].mean().groupby("encoder_seed").mean()
    return float(per.groupby("fold")["diff"].mean().mean()), dots


def has_rows(pair, key, unit=None, **conditions) -> bool:
    part = common.pick(common.scores(), pair=pair, fold_set=common.primary_units(pair)[0][0], key=key, side="bulk", bulk_subset="all", **conditions)
    return not (part if unit is None else part[part["fold"] == unit[1]]).empty


def age_loss_units(pair) -> list:
    """The units whose frozen CytoSet configuration has the age-similarity loss."""
    out = []
    for unit in common.primary_units(pair):
        selection = common.read_json("05_search", pair, "cytoset/selection.json", unit)
        out += [unit] if selection["frozen"].get("age_similarity_loss") else []
    return out


# The cell configurations without a set encoder's design choices (scVI, the frozen foundation models), in display order
UNTUNED = tuple(k for k in F.DISPLAY_ORDER if k.endswith("_cells") and configurations.CONFIGURATIONS[k]["model"] not in F.FIG4_SET_ENCODERS)

def encoders() -> list:
    return [e for e in F.FIG4_SET_ENCODERS if f"{e}_cells" in common.alternatives()]


def ring_value(pair, encoder) -> tuple:
    """The ring: each unit's frozen configuration (row A) at the curve seeds, as A's n = 500 point."""
    return difference(pair, f"{encoder}_cells", common.encoder_seeds("F"), **common.headline_conditions(f"{encoder}_cells"))


def points_a(pair) -> list:
    """(model key, n or ALL, value, dots) for every Fig 4A point, with the both-ways checks of the conditional points."""
    curve = common.encoder_seeds("F")
    out = []
    lost = age_loss_units(pair)
    for unit in common.primary_units(pair):
        present = has_rows(pair, "cytoset_cells", unit, cell_subset="all_available")
        assert present == (unit not in lost), f"Fig 4A {pair} {unit}: CytoSet 'all' rows {present}, age-similarity loss {unit in lost} (frozen §60)"
    for fm in FMS:
        if f"{fm}_cells" in common.alternatives():
            present = has_rows(pair, f"{fm}_cells", cell_subset="all_available")
            assert present == (fm in F.FIG4A_FM_ALL_AVAILABLE and pair in T.FIG4A_ALL_PAIRS), \
                f"Fig 4A {pair} {fm}: 'all' rows {present} but FIG4A_FM_ALL_AVAILABLE {F.FIG4A_FM_ALL_AVAILABLE}"
    lines = [*(f"{e}_cells" for e in encoders()), *(k for k in UNTUNED if k in common.alternatives()), "pseudobulk"]
    for key in lines:
        seeds = [-1] if key in common.NO_ENCODER_LEVEL else curve
        pooling = "mean" if configurations.CONFIGURATIONS[key]["model"] in ("geneformer", "scgpt", "scvi") else "native"
        for n in representations.BAG_SIZES:
            conditions = common.headline_conditions(key) if n == T.FROZEN_NCELL and key != "pseudobulk" else {"n_cells": n, "pooling": pooling}
            out.append((key, n, *difference(pair, key, seeds, **conditions)))
        drawn = pair in T.FIG4A_ALL_PAIRS and not (key == "cytoset_cells" and lost) and \
            (configurations.CONFIGURATIONS[key]["model"] not in FMS or configurations.CONFIGURATIONS[key]["model"] in F.FIG4A_FM_ALL_AVAILABLE)
        if drawn:
            out.append((key, ALL, *difference(pair, key, seeds, cell_subset="all_available", n_cells=0, pooling=pooling)))
    return out


def test_label(pair) -> str:
    """The per-pair best-candidate p (unadjusted across pairs), and its Holm value across pairs when any pair is
    significant; "n.e." where the test isn't formed."""
    tests = common.pick(common.secondary(), neutral=False, row="best_candidate", key="best_candidate")
    row = tests[tests["pair"] == pair]
    assert len(row) == 1, f"Fig 4 {pair}: best-candidate test rows {len(row)}"
    p, holm = row["p_better"].iloc[0], row["p_better_holm"].iloc[0]
    if not np.isfinite(p):
        return F.FIGURE_TEXT["not_estimable"]
    if (tests["p_better"] <= F.HOLM_ALPHA).any():
        return F.TEXT["p_value_holm"].format(p=p, holm=holm)
    return F.TEXT["p_value"].format(p=p)


def panel_a(fig, cell, pairs):
    axes = plots.columns(fig, cell, len(pairs))
    limits = F.XLIM_DIFFERENCE["Fig 4"]
    csv = []
    for ax, pair in zip(axes, pairs):
        shown_keys = {"pseudobulk", *(f"{e}_cells" for e in encoders())}   # the set encoders and the pseudobulk of the same cells
        points = [p for p in points_a(pair) if p[0] in shown_keys]   # (scVI's and the foundation models' rows: Table S8)
        frame = pd.DataFrame([(k, n, v) for k, n, v, _ in points], columns=["key", "n", "value"])
        keys = list(dict.fromkeys(frame["key"]))
        shift = {k: 10 ** plots.dodge(j, len(keys), DODGE_DECADES) for j, k in enumerate(keys)}   # overlapping lines spread in x
        for key, part in frame.groupby("key", sort=False):
            ax.plot(part["n"] * shift[key], part["value"], color=style.color(key), linewidth=S.LINE_WIDTH)
            for n, value in zip(part["n"], part["value"]):
                plots.point(ax, n * shift[key], value, limits, style.marker(key, S.MARKER_SIZE * (0.7 if key == "pseudobulk" else 1)), orientation="y")
        for key, n, value, dots in points:   # seeds in the source data only: drawn, they hid the lines
            csv.append({"pair": pair, "key": key, "n_cells": "all" if n == ALL else n, "delta": value})
            csv += [{"pair": pair, "key": key, "n_cells": "all" if n == ALL else n, "seed": int(s), "delta_seed": d} for s, d in dots.items()]
        for encoder in encoders():
            key = f"{encoder}_cells"
            value = frame.loc[(frame["key"] == key) & (frame["n"] == T.FROZEN_NCELL), "value"].iloc[0]
            style.ring(ax, T.FROZEN_NCELL * shift[key], min(max(value, limits[0]), limits[1]), key)
        if age_loss_units(pair):
            ax.text(ALL, 0, F.FIGURE_TEXT["not_estimable"], ha="center", va="bottom", fontsize=S.MIN_FONT_SIZE, color=style.color("cytoset_cells"))
            csv.append({"pair": pair, "key": "cytoset_cells", "n_cells": "all", "note": F.FIGURE_TEXT["not_estimable"]})
        plots.difference_axis(ax, limits, margin=True, orientation="y")
        ax.set_yticks(plots.difference_ticks(limits))
        ax.set_xscale("log")
        ax.set_xlim(0.5, ALL * 2)
        ax.set_xticks([1, 10, 100, ALL])
        ax.set_xticklabels(["1", "10", "100", F.TEXT["all_point"]])
        ax.minorticks_off()
        ax.set_title(f"{common.pair_title(pair)}\n{test_label(pair)}")
    plots.share_y(axes, limits)
    axes[0].set_ylabel(F.TEXT["difference"])
    plots.centered_xlabel(axes, F.TEXT["cells_per_bag"])
    style.panel_label(axes[0], "A", at_left=True, above_pt=14)   # level with the column titles
    common.panel_csv(pd.DataFrame(csv), FIGURE, "A")


def candidate_groups(pair, encoder) -> pd.DataFrame:
    """key, change_group, change_value of an encoder's search candidates (stage 05; the same grid in every fold, asserted)."""
    tables = []
    for unit in common.primary_units(pair):
        frames = [common.read_kind("candidates", "05_search", pair, f"{encoder}/candidates_s{i}.parquet", unit) for i in common.search_seeds()]
        tables.append(pd.concat(frames)[["key", "change_group", "change_value"]].drop_duplicates().sort_values("key").reset_index(drop=True))
    assert all(t.equals(tables[0]) for t in tables), f"Fig 4 {pair} {encoder}: the folds searched different grids"
    return tables[0]


def option(change_value, group) -> str:
    return str(json.loads(change_value)[group])


def points_bc(pair, panel) -> tuple:
    """[(group label, [(x label, key, value, dots, kind)])]: the panel's row groups. Every candidate is scored in every fold
    and at every search seed (stage 08b), so a missing one fails in common.checked, naming the fold."""
    groups = []
    for encoder in encoders():
        table = candidate_groups(pair, encoder)
        group = F.FIG4_GROUPS[panel].get(encoder)
        names = [("base", F.FIG4_BASE_VALUE[panel][encoder])]
        names += [(r.key, option(r.change_value, group)) for r in table.itertuples() if group and r.change_group == group]
        points = []
        for name, text in names:
            value, dots = difference(pair, name, common.search_seeds(), row_kind="candidate", model=encoder, n_cells=T.FROZEN_NCELL, pooling="native")
            points.append((text, f"{encoder}_cells", value, dots, "candidate"))
        points.append((F.FIGURE_TEXT["selected"], f"{encoder}_cells", *ring_value(pair, encoder), "ring"))
        groups.append((configurations.MODELS[encoder]["label"], points))
    if panel == "B":
        if "scvi_cells" in common.alternatives():
            groups.append((common.label("scvi_cells"),
                           [("mean", "scvi_cells", *difference(pair, "scvi_cells", common.encoder_seeds("F"), **common.headline_conditions("scvi_cells")), "candidate")]))
        for key in [k for k in ("geneformer_cells", "scgpt_cells") if k in common.alternatives()]:
            points = [(pooling, key, *difference(pair, key, [-1], n_cells=T.FROZEN_NCELL, pooling=pooling), "candidate") for pooling in ("mean", "max")]
            groups.append((common.label(key), points))
    return groups


def panel_bc(fig, cell, pairs, panel):
    """Rows (top to bottom): per model its options, a dotted divider and the ring point; pairs as columns, the x axis the
    difference from pseudobulk; row labels on the first column ("model: option" on a model's first row, then the option)."""
    axes = plots.columns(fig, cell, len(pairs))
    limits = F.XLIM_DIFFERENCE["Fig 4"]
    csv = []
    for i, (ax, pair) in enumerate(zip(axes, pairs)):
        groups = points_bc(pair, panel)
        y, ticks = 0, []
        for name, points in groups:
            for text, key, value, dots, kind in points:
                if kind == "ring":
                    ax.axhline(y - 0.5, **S.DIVIDER)
                shown = dots[(dots >= limits[0]) & (dots <= limits[1])].to_numpy()   # seed dots past the range left out
                ax.plot(shown, np.full(len(shown), y), **style.seed_dots(key))
                drawn = plots.point(ax, value, y, limits, style.marker(key))
                if kind == "ring":
                    style.ring(ax, drawn[0], y, key)
                model = configurations.CONFIGURATIONS[key]["model"]
                text_shown = F.FIG4_NATIVE_POOLING[model] if text == "native" else F.FIG4_OPTION_LABELS.get(text, text)
                first = not ticks or ticks[-1][2] != name
                ticks.append((y, f"{configurations.MODELS[model]['label']}: {text_shown}" if first else text_shown, name))
                csv.append({"pair": pair, "block": name, "key": key, "option": text, "kind": kind, "delta": value})
                csv += [{"pair": pair, "block": name, "key": key, "option": text, "kind": kind, "seed": int(s), "delta_seed": d} for s, d in dots.items()]
                y += 1
            y += 0.5
        plots.difference_axis(ax, limits, margin=True)
        ax.set_xticks(plots.difference_ticks(limits))
        ax.set_ylim(y - 0.9, -0.6)
        ax.set_yticks([t for t, _, _ in ticks])
        ax.set_yticklabels([t for _, t, _ in ticks] if i == 0 else [])
        ax.tick_params(axis="y", length=0)
    if panel == "C":
        plots.centered_xlabel(axes, F.TEXT["difference"])
    style.panel_label(axes[0], panel, at_left=True)
    common.panel_csv(pd.DataFrame(csv), FIGURE, panel)


def legend(fig):
    keys = [k for k in F.DISPLAY_ORDER if k in common.kept() and k.endswith("_cells")]
    style.legend_strip(fig, keys, ("pseudobulk_n", "ring", "pseudobulk_zero"), ncol=6)


def main(pairs, panels):
    with plt.rc_context(F.TEXT_RC):
        common.draw(FIGURE, {"A": panel_a, "B": lambda f, c, p: panel_bc(f, c, p, "B"), "C": lambda f, c, p: panel_bc(f, c, p, "C")},
                    pairs, panels, __file__, ratios=(1.45, 1.35, 1.05), legend=legend, top_mm=21, hspace=(0.43, 0.32), left=0.165, bottom=0.065)
