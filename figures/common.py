"""What every figure script shares: the readers (the only code in figures/ that opens a stage output), the display aggregations, the
row-count checks, saving in Genome Biology's format, one CSV per panel, and the provenance sidecar.

Figures are end products and feed no stage, so they don't go through manifest.finish (whose label-read rule would
force labelled_ names onto the paper's files): each figure directory gets sources.json instead (every input, its
content hash and its record's commit, tag and blinded flag). Before a figure is saved its inputs are checked:
unblinded in a real run (manifest.assert_unblinded) or blinded with this run's index (label-free inputs: either), made
from a clean tree (tagged in full-run mode), and current (their recorded code hash equals the tree's).

Outputs, in <labelled outputs>/figures/<item>/: <item>.pdf (the submission file), <item>.png (300-dpi preview), <item>.svg (rasterized artists in the PDF and SVG at 600 dpi), one
<item><panel>.csv per panel, one <item><panel>.png per drawn panel (the panel cropped from the composite, for review),
sources.json. In a blinded run that is outputs/blinded_<seed>/figures/, and nothing there
leaves the server except the PNG previews.
"""
import functools
import hashlib
import itertools
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, ft2font
from matplotlib.axis import Axis
from matplotlib.collections import Collection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.spines import Spine
from matplotlib.text import Text
from matplotlib.transforms import Bbox

import plots
import style
from sc2bulk import codehash, folds, labels, manifest, paths, schemas
from sc2bulk.config import configurations, deconvolution, settings, smoke, triage
from sc2bulk.config.statistics import MARGIN
from sc2bulk.statistics import at_ceiling
from sc2bulk.config import figures as F
from sc2bulk.config import properties as P
from sc2bulk.config import style as S
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

LABELLED = settings.LABELLED + "_"
INPUTS = []          # the current figure's stage-output inputs, in reading order (reset by start())
REPOSITORY = []      # the current figure's repository inputs (figures/signature_labels.tsv): recorded by blob hash
CACHE = {}           # path -> table: each file is read once per make_all process
WRITTEN = set()      # (item, panel) CSVs written
WATERMARK = None     # set by make_all: style.BLINDED_WATERMARK in a blinded run; pytest sets style.SYNTHETIC_WATERMARK
NO_ENCODER_LEVEL = {k for k in F.DISPLAY_ORDER
                    if k == "pseudobulk" or configurations.CONFIGURATIONS[k]["model"] in ("geneformer", "scgpt")}
PANEL_PAD_IN = 0.02                     # margin around a panel's own PNG (inches)
NEUTRAL = {"readout": "ridge", "alignment": "none", "variant": "primary", "feature_set": "panel", "cell_subset": "all",
           "permutation": -1}


# ---- reading ----

def start():
    INPUTS.clear()
    REPOSITORY.clear()


def _read(path, reader, key=None):
    """Read a stage output once per process (key: when one file is read two ways), recording it as an input."""
    INPUTS.append(path)
    key = path if key is None else key
    if key not in CACHE:
        CACHE[key] = reader(path)
    return CACHE[key]


def read_kind(kind, stage, pair, name, unit=None) -> pd.DataFrame:
    """A schema kind's table (schemas.read checks it whole: no column projection, so each file is read once and kept)."""
    return _read(paths.output(stage, pair, name, unit), lambda p: schemas.read(kind, p))


def read_csv(stage, pair, name, unit=None, **kwargs) -> pd.DataFrame:
    return _read(paths.output(stage, pair, name, unit), lambda p: pd.read_csv(p, **kwargs))


def dropped(pair) -> dict:
    """Configuration -> "triaged" or "pending": stage 10's Geneformer rows left out."""
    inclusion = read_csv("10_attribution", pair, "inclusion.csv", keep_default_na=False)
    rows = inclusion[inclusion["failed_checks"].isin(list(F.DROPPED_TEXT))]
    return dict(zip(rows["configuration"], rows["failed_checks"]))


def excluded(pair) -> set:
    """The configurations whose Geneformer checks failed in pair (stage 10's inclusion.csv: included False, not dropped)."""
    inclusion = read_csv("10_attribution", pair, "inclusion.csv", keep_default_na=False)
    return set(inclusion.loc[~inclusion["included"] & ~inclusion["failed_checks"].isin(list(F.DROPPED_TEXT)), "configuration"])


def geneformer_pending() -> bool:
    """Whether the current figure read a stage 10 inclusion.csv from a gather run without Geneformer."""
    return any((CACHE[p]["failed_checks"] == "pending").any() for p in set(INPUTS)
               if p.name == "inclusion.csv" and p.parent.parent.name == "10_attribution")


def read_parquet(stage, pair, name, unit=None) -> pd.DataFrame:
    """A plain Parquet output (stage 04c's embeddings)."""
    return _read(paths.output(stage, pair, name, unit), pd.read_parquet)


def read_tsv(path) -> pd.DataFrame:
    """A repository table a figure reads (figures/signature_labels.tsv; "#" comment lines)."""
    REPOSITORY.append(path)
    if path not in CACHE:
        CACHE[path] = pd.read_csv(path, sep="\t", comment="#")
    return CACHE[path]


def read_note(path) -> str:
    """A repository text file a table reads (the hand-written tables and legends in figures/text; recorded by blob hash)."""
    REPOSITORY.append(path)
    if path not in CACHE:
        CACHE[path] = path.read_text()
    return CACHE[path]


def costs() -> pd.DataFrame:
    """Every stage output's recorded cost (Table S11): one row per output of a pipeline stage (pipeline/<stage>.py) in
    this run's outputs and labelled outputs (smoke, figures, tables, and blinded rehearsal trees left out), with its
    manifest's run record: the settings hash (one per work unit: a unit's outputs share it, wherever they are written),
    seed and cost, and the record's configuration, row, model and job name where it has them (None otherwise). Columns
    path (from the stage directory), stage, pair ("" for a pair-less output), settings_hash, seed, seconds, device, gpu,
    configuration, row, model, name."""
    stages = {p.stem for p in (settings.REPO / "pipeline").glob("[0-9]*.py")}
    columns = ["path", "stage", "pair", "settings_hash", "seed", "seconds", "device", "gpu", "configuration", "row", "model",
               "name"]
    rows = []
    for path in sorted(settings.OUTPUTS.rglob("*.manifest.json")):
        root = settings.LABELLED_OUTPUTS if path.is_relative_to(settings.LABELLED_OUTPUTS) else settings.OUTPUTS
        parts = path.relative_to(root).parts
        if parts[0] not in stages:
            continue
        record = json.loads(path.read_text())
        rows.append({"path": "/".join(parts).removesuffix(".manifest.json"), "stage": parts[0],
                     "pair": parts[1] if len(parts) > 2 else "", "settings_hash": record["settings_hash"], "seed": record["seed"],
                     "seconds": record["cost"]["seconds"], "device": record["cost"]["device"], "gpu": record["cost"]["gpu"],
                     **{k: record["settings"].get(k) for k in ("configuration", "row", "model", "name")}})
    return pd.DataFrame(rows, columns=columns)


def read_json(stage, pair, name, unit=None) -> dict:
    """A JSON output; pair "" for pair-less files (12a's licence.json)."""
    return _read(paths.output(stage, pair, name, unit), lambda p: json.loads(p.read_text()))


def names(stage, pair, pattern, unit=None) -> list:
    """The names of a stage's outputs matching a pattern (stage 05's merged-candidate files exist only when made)."""
    return sorted(str(p.relative_to(paths.output(stage, pair, "", unit))) for p in paths.output(stage, pair, "", unit).glob(pattern))


def scores() -> pd.DataFrame:
    return read_kind("scores", "11b_secondary", "pooled", f"{LABELLED}scores.parquet")


def statistics() -> pd.DataFrame:
    return read_kind("statistics", "11_statistics", "pooled", f"{LABELLED}statistics.parquet")


def secondary() -> pd.DataFrame:
    return read_kind("secondary_statistics", "11b_secondary", "pooled", f"{LABELLED}secondary_statistics.parquet")


def claims() -> dict:
    return read_json("11_statistics", "pooled", f"{LABELLED}claims.json")


def single_cell_samples(pair) -> pd.DataFrame:
    """Stage 02's kept single-cell samples, through sc2bulk.folds (the single-cell reader)."""
    return _read(folds.inputs(pair)[0], lambda p: folds.single_cell_samples(pair))


def training_samples(pair, unit) -> pd.DataFrame:
    """A unit's single-cell training samples (stage 09's single-cell side), through sc2bulk.folds."""
    INPUTS.append(folds.inputs(pair)[0])
    path = paths.output("03_splits", pair, "donor_folds.csv")
    return _read(path, lambda p: folds.training_samples(pair, *unit), key=(path, "training", unit))


def all_single_cell_samples(pair) -> pd.DataFrame:
    """Stage 02's whole single-cell sample table, dropped samples included (S5H: the lupus batch-4 restriction)."""
    return _read(folds.inputs(pair)[0], lambda p: pd.read_csv(p, dtype={"holdout_group": str, "donor_id": str, "sample_id": str}),
                 key=("all", folds.inputs(pair)[0]))


def cell_qc(pair, columns) -> pd.DataFrame:
    """Stage 02's per-cell QC table (Figs. S2–S7 panel A; descriptive, never a model feature)."""
    return _read(settings.OUTPUTS / "02_preprocess" / pair / "cell_qc.csv.gz",
                 lambda p: pd.read_csv(p, usecols=list(columns), dtype={"sample_id": str}))


def bulk_samples(pair) -> pd.DataFrame:
    """Stage 02's kept bulk samples (label-free)."""
    table = read_csv("02_preprocess", pair, "bulk_samples.csv", dtype={"sample_id": str, "donor_id": str})
    return table[table["drop_reason"].isna()].reset_index(drop=True)


def scored_bulk(pair, unit) -> pd.DataFrame:
    """A unit's scored bulk samples (stage 09's bulk side; label-free), through sc2bulk.folds."""
    kept = bulk_samples(pair)
    path = paths.output("03_splits", pair, "donor_folds.csv")
    return _read(path, lambda p: folds.scored_bulk(pair, *unit, kept), key=(path, "scored", unit))


def bulk_labels(pair) -> pd.DataFrame:
    """The kept bulk samples' labels (permuted in a blinded run), through sc2bulk.labels after require_access: only the
    scripts with READS_BULK_LABELS call this."""
    labels.require_access()
    path = settings.OUTPUTS / "02_preprocess" / pair / "bulk_labels.csv"
    return _read(path, lambda p: labels.bulk_labels(pair))


def deconvolution_composition(pair) -> pd.DataFrame:
    """Mean cell-type fractions per side (S8B; stage 04b, label-free): the single-cell training samples' annotated
    fractions ("true") and the scored bulk's deconvolved ones ("estimated"), each the mean over samples (a type absent
    from a sample's stratum counts 0), then over the primary units (human blood, aging: its four folds). Columns side,
    type, fraction."""
    parts = []
    for unit in primary_units(pair):
        table = read_parquet("04b_deconvolve", pair, "fractions.parquet", unit)
        for side, kind in (("single_cell", "true"), ("bulk", "estimated")):
            part = table[(table["side"] == side) & (table["kind"] == kind)]
            wide = part.pivot_table(index="sample_id", columns="type", values="fraction", aggfunc="sum", fill_value=0)
            parts.append(wide.mean().rename("fraction").rename_axis("type").reset_index().assign(side=side, unit=folds.unit_name(*unit)))
    units = pd.concat(parts).pivot_table(index=["side", "type"], columns="unit", values="fraction", fill_value=0)
    return units.mean(axis=1).rename("fraction").reset_index()


def type_order(pair) -> list:
    """A pair's annotated cell types (stage 04c's UMAP cells), most cells first."""
    types = read_parquet("04c_qc_embed", pair, "umap.parquet")["cell_type"].astype(str)
    counts = types.value_counts()
    return sorted(counts.index, key=lambda t: (-counts[t], t))


def cell_type_colors(pairs) -> dict:
    """The cell-type colors of every descriptive figure (style.cell_type_colors): over each pair's annotated types and,
    where it has a deconvolution, its deconvolution types by mean single-cell fraction."""
    lists = [type_order(p) for p in pairs]
    for p in pairs:
        if deconvolution.LEVELS[p] is not None:
            single = deconvolution_composition(p)
            single = single[single["side"] == "single_cell"].set_index("type")["fraction"]
            lists.append(sorted(single.index, key=lambda t: (-single[t], t)))
    return style.cell_type_colors(lists)


# ---- selecting and aggregating ----

def pick(table, neutral=True, **conditions) -> pd.DataFrame:
    """Rows matching every condition (a tuple: any of its values), with the neutral key values unless overridden."""
    keep = pd.Series(True, index=table.index)
    for column, value in {**(NEUTRAL if neutral else {}), **conditions}.items():
        keep &= table[column].isin(value) if isinstance(value, (tuple, list, set)) else table[column] == value
    return table[keep]


def alternatives() -> list:
    return [k for k in F.DISPLAY_ORDER if k != "pseudobulk" and k not in triage.CUT_CONFIGURATIONS]


def kept() -> list:
    return ["pseudobulk", *alternatives()]


def encoder_seeds(row="A") -> list:
    """A row's headline encoder seeds (training_inputs.encoder_seeds' rule without the attribution seeds 10-19, which no
    figure reads): 0-9 (the smoke run's count), rows E, F, H 0-4."""
    n = smoke.N_ENCODER_SEEDS if settings.SMOKE else settings.N_ENCODER_SEEDS
    return list(range(min(n, T.CURVE_SEEDS) if row in ("E", "F", "H") else n))


def readout_seeds() -> list:
    return list(range(smoke.N_READOUT_SEEDS if settings.SMOKE else settings.N_READOUT_SEEDS))


def search_seeds() -> list:
    return list(range(smoke.N_SEARCH_SEEDS if settings.SMOKE else settings.N_SEARCH_SEEDS))


def seeds_of(key, row="A") -> list:
    return [-1] if key in NO_ENCODER_LEVEL else encoder_seeds(row)


def primary_units(pair) -> list:
    return [u for u in folds.units(pair) if u[0] == next(iter(PAIRS[pair]["fold_sets"]))]


def checked(part, pair, seeds, what) -> pd.DataFrame:
    """One configuration's rows of one pair (fold, encoder_seed, readout_seed, auroc), asserting every primary fold,
    every expected encoder seed and the same read-out seeds under each (a missing row fails,
    naming it)."""
    part = part[part["pair"] == pair]
    readouts = sorted(part["readout_seed"].unique())
    expected = {(f, e) for _, f in primary_units(pair) for e in seeds}
    found = set(zip(part["fold"], part["encoder_seed"]))
    assert found == expected, f"{pair} {what}: (fold, encoder seed) {sorted(found)} != {sorted(expected)}"
    assert all(sorted(r) == readouts for _, r in part.groupby(["fold", "encoder_seed"])["readout_seed"]), \
        f"{pair} {what}: read-out seeds differ between encoder seeds"
    assert readouts in ([-1], readout_seeds()), f"{pair} {what}: read-out seeds {readouts}"
    assert not part.duplicated(["fold", "encoder_seed", "readout_seed"]).any(), f"{pair} {what}: duplicate rows"
    return part[["fold", "encoder_seed", "readout_seed", "auroc"]].reset_index(drop=True)


def aurocs(pair, key, seeds=None, side="bulk", row_kind="configuration", **conditions) -> pd.DataFrame:
    """A configuration's (or reference's, or candidate's) AUROC per (fold, encoder seed, read-out seed) from the
    scores table (the pair's primary fold set, bulk subset all, the neutral keys unless overridden)."""
    seeds = seeds_of(key) if seeds is None else seeds
    part = pick(scores(), pair=pair, fold_set=primary_units(pair)[0][0], key=key, side=side, row_kind=row_kind, bulk_subset="all",
                encoder_seed=tuple(seeds), **conditions)
    return checked(part, pair, seeds, f"{key} {side} {conditions}")


def per_seed(raw: pd.DataFrame) -> pd.DataFrame:
    """fold, encoder_seed, auroc: the mean over read-out seeds."""
    return raw.groupby(["fold", "encoder_seed"])["auroc"].mean().rename("auroc").reset_index()


def mean_over_seeds(raw: pd.DataFrame) -> float:
    """Stage 11's order: the mean over read-out seeds, then over encoder seeds within each fold, then over the folds."""
    return float(per_seed(raw).groupby("fold")["auroc"].mean().mean())


def seed_differences(pair, key, comparator, **conditions) -> pd.Series:
    """Per-seed differences from the comparator (the faint dots of a difference axis): per encoder seed e, the mean over
    folds of [the mean over read-out seeds of key's AUROC at e - the comparator's]; for keys without an encoder level,
    per read-out seed r, paired by r. Their mean is the stage's point difference (asserted by the callers)."""
    raw = aurocs(pair, key, **conditions)
    base = comparator.groupby(["fold", "readout_seed"])["auroc"].mean()
    if raw["encoder_seed"].eq(-1).all():
        merged = raw.assign(diff=raw["auroc"].to_numpy() - base.loc[list(zip(raw["fold"], raw["readout_seed"]))].to_numpy())
        return merged.groupby(["readout_seed", "fold"])["diff"].mean().groupby("readout_seed").mean()
    fold_base = base.groupby("fold").mean()
    seeds_ = per_seed(raw)
    seeds_["diff"] = seeds_["auroc"] - seeds_["fold"].map(fold_base)
    return seeds_.groupby(["encoder_seed", "fold"])["diff"].mean().groupby("encoder_seed").mean()


def headline_conditions(key) -> dict:
    """The primary rows' bag size and pooling (stage 11's unit_scores)."""
    config = configurations.CONFIGURATIONS[key]
    return {"n_cells": 500 if config["input"] == "cells" else 0, "pooling": "mean" if config["model"] in schemas.EMBED_ONLY else "native"}


def within_side(pair) -> str:
    """Fig 2B's single-cell side: held-out donors' samples for the cross-fitted pair, else the encoder
    split's validation side."""
    return "single_cell_heldout" if primary_units(pair)[0][0] == "crossfit" else "single_cell_validation"


# ---- labels and text ----

def label(key) -> str:
    if key in F.ROW_LABELS:
        return F.ROW_LABELS[key]
    if key in configurations.CONFIGURATIONS:
        return configurations.CONFIGURATIONS[key]["label"]
    if key in configurations.REFERENCE_ROWS:
        return configurations.REFERENCE_ROWS[key]["label"]
    return key


def pair_title(pair) -> str:
    return "\n".join(PAIRS[pair]["title"])


def stacked_title(pair, mark, lines=3) -> str:
    """Figs 1d, 2 and 3's column title: species, tissue and phenotype, one per line, or (lines=2, Fig 3) species and
    tissue on the first line (♦ after a pooled pair's phenotype iff mark). Titles' last lines share a baseline, so each
    attribute forms one row across the columns."""
    species, tissue = PAIRS[pair]["title"][0].split(" ", 1)
    pooled = f" {F.FIGURE_TEXT['pooled_mark']}" if mark and pair in kept_pairs() else ""
    sep = " " if lines == 2 else "\n"
    return f"{species}{sep}{tissue}\n{PAIRS[pair]['title'][1]}{pooled}"


def citation(item, panel) -> str:
    """A panel's citation form: Fig 2A, Fig. S9A."""
    return f"{item}{panel}"


def stem(item) -> str:
    """The item's file stem: Fig 2 -> Fig2; Fig. S9 -> S9Fig (the internal file names keep their earlier form)."""
    return item.removeprefix("Fig. ") + "Fig" if item.startswith("Fig. S") else item.replace(" ", "")


def panel_name(item, panel) -> str:
    """The panel's file name (its CSV's and PNG's): Fig2A, S9AFig."""
    return stem(citation(item, panel))


def panel_letter(panel) -> str:
    return panel.lower() if F.PANEL_LABEL_CASE == "lower" else panel


# ---- writing ----

def root() -> Path:
    """<labelled outputs>/figures: blinded_<seed>/figures in a blinded run."""
    out = settings.LABELLED_OUTPUTS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def directory(item) -> Path:
    out = root() / stem(item)
    out.mkdir(exist_ok=True)
    return out


def write_bytes(path, write):
    """Write through a temporary name, then rename (as manifest does), so a crash never leaves a half-written file."""
    temporary = manifest.temporary(path)
    write(temporary)
    os.replace(temporary, path)


def panel_csv(frame: pd.DataFrame, item, panel):
    """The panel's plotted values (long format, with the config labels): Additional file 3."""
    assert panel in F.PANELS[item], f"{item} {panel}: not a registered panel"
    assert len(frame), f"{item} {panel}: an empty panel"
    frame = frame.copy()
    if "key" in frame and "label" not in frame:
        frame.insert(frame.columns.get_loc("key") + 1, "label", frame["key"].map(label))
    if "pair" in frame and "pair_label" not in frame:
        frame.insert(frame.columns.get_loc("pair") + 1, "pair_label", frame["pair"].map(lambda p: PAIRS[p]["name"] if p in PAIRS else p))
    write_bytes(directory(item) / f"{panel_name(item, panel)}.csv", lambda p: frame.to_csv(p, index=False))
    WRITTEN.add((item, panel))


def read_written(item, panel) -> pd.DataFrame:
    """A panel CSV this process wrote (Additional file 3's members)."""
    return pd.read_csv(directory(item) / f"{panel_name(item, panel)}.csv", low_memory=False, keep_default_na=False, na_values=[""])   # "n/a" notes kept


FORBIDDEN = re.compile(r"^(Fig\.? ?S?\d|Figure \d|Table S?\d|Additional file \d)")


def forbidden_texts() -> set:
    """Every registered title (with every conditional variant's fixed part): never drawn (Genome Biology: titles and
    legends go in the manuscript)."""
    titles = {*F.TITLES.values(), F.FIG6_TITLE, P.LICENSED_TITLE, P.DESCRIPTIVE_TITLE}
    variants = [F.FIG2_TITLES["not_evaluable"], *(t for forms in (F.FIG2_TITLES["pooled"], F.FIG2_TITLES["single"]) for t in forms.values())]
    return titles | {t.split("{")[0] for t in variants if len(t.split("{")[0]) > 20}


def legend_mm(item) -> float:
    """The typeset height of a figure's title and legend (figures/text/legends.md: the lines under its bold title up to the next
    heading, without comments or ">" notes), estimated at style.LEGEND_CHARS_PER_LINE and LEGEND_LINE_MM."""
    lines = read_note(settings.REPO / "figures" / "text" / "legends.md").split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(f"**{item}."))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith(("**Fig", "**Table", "## ", "---")))
    text = [line for line in lines[start:end] if line.strip() and not line.startswith(("<!--", ">"))]
    return sum(-(-len(line) // S.LEGEND_CHARS_PER_LINE) for line in text) * S.LEGEND_LINE_MM


def check_figure(fig, item):
    """Genome Biology's artwork rules, asserted: width, height, fonts (allowed families, >= MIN_FONT_SIZE), lines >
    0.25 pt, main figures' figure + legend height, no figure number, title or legend text in the image."""
    width, height = fig.get_size_inches() * 25.4
    assert abs(width - S.WIDTH_MM[F.ITEMS[item]["width"]]) < 0.01, f"{item}: {width:.2f} mm wide"
    assert height <= S.MAX_HEIGHT_MM + 1e-9, f"{item}: {height:.1f} mm tall"
    if not item.startswith("Fig. S"):   # a main figure: the limit covers figure and legend (supplementary figures sit in Additional file 1)
        total = height + legend_mm(item)
        print(f"{item}: figure {height:.0f} mm + legend ~{total - height:.0f} mm = ~{total:.0f} mm (limit {S.MAX_HEIGHT_MM})")
        assert total <= S.MAX_HEIGHT_MM or item in F.LEGEND_FIT_PENDING, f"{item}: figure and legend ~{total:.0f} mm tall"
    allowed = {S.FONT, *S.FONT_FALLBACKS}
    titles = forbidden_texts()
    for text in fig.findobj(Text):
        content = text.get_text()
        if not content.strip() or not text.get_visible():
            continue
        path = font_manager.findfont(text.get_fontproperties(), fallback_to_default=False)
        family = font_manager.FontProperties(fname=path).get_name()
        assert family in allowed, f"{item}: {content!r} in {family}"
        font = ft2font.FT2Font(path)
        missing = [c for c in content if not c.isspace() and not font.get_char_index(ord(c))]
        assert not missing, f"{item}: {content!r} has characters {family} lacks: {missing}"   # they'd print as empty boxes
        assert text.get_fontsize() >= S.MIN_FONT_SIZE, f"{item}: {content!r} at {text.get_fontsize()} pt"
        assert not FORBIDDEN.match(content) and not any(content.startswith(t) for t in titles), f"{item}: {content!r} is caption text"
    for artist in fig.findobj(lambda a: isinstance(a, (Line2D, Patch, Collection)) and a.get_visible()):
        if isinstance(artist, Line2D):
            widths = [artist.get_linewidth()] if artist.get_linestyle() not in ("None", "", " ") else []
            widths += [artist.get_markeredgewidth()] if artist.get_marker() not in ("None", "", " ", None) else []
        elif isinstance(artist, Collection) and not len(artist.get_paths()):
            widths = []   # an empty collection draws nothing (e.g. a color bar's unused dividers)
        else:
            widths = list(np.atleast_1d(artist.get_linewidth()))
        assert all(w == 0 or w > S.MIN_LINE_WIDTH_PT for w in widths), f"{item}: a line of {min(widths)} pt ({artist})"


# ---- layout checks ----

def mm_box(box, fig) -> Bbox:
    """A display-unit box in mm from the figure's bottom-left corner."""
    return Bbox(box.get_points() * 25.4 / fig.dpi)


def drawn_box(ax, renderer):
    """An axes' drawn extent (display units): its tight box with its axis labels or, with its axis off, the union of what
    it draws (its frame isn't drawn). Panel letters are left out (not in layout)."""
    if ax.axison:
        boxes = [ax.get_tightbbox(renderer), *(Text.get_window_extent(t, renderer) for t in (ax.xaxis.label, ax.yaxis.label)
                                               if t.get_visible() and t.get_text().strip())]
    else:
        boxes = [a.get_tightbbox(renderer) for a in ax.get_children()
                 if a.get_visible() and a.get_in_layout() and a is not ax.patch and not isinstance(a, (Spine, Axis))]
    boxes = [b for b in boxes if b is not None and np.isfinite(b.get_points()).all() and (b.width > 0 or b.height > 0)]
    return Bbox.union(boxes) if boxes else None


def drawn_texts(fig) -> list:
    """Every drawn, non-empty text: (text, role, its axes or None); role letter (gid "panel_label"), xtick, ytick, xlabel,
    ylabel, title or other. Tick labels are those of the ticks drawn (inside the view limits)."""
    found = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        if ax.axison:
            for name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
                if axis.get_visible():
                    found += [(t, f"{name}tick", ax) for tick in axis._update_ticks() for t in (tick.label1, tick.label2)]
                    found += [(axis.label, f"{name}label", ax), (axis.offsetText, "other", ax)]
        found += [(t, "title", ax) for t in (ax.title, ax._left_title, ax._right_title)]
        found += [(t, "other", ax) for t in ax.texts]
        legend = ax.get_legend()
        if legend is not None and legend.get_visible():
            found += [(t, "other", ax) for t in (legend.get_title(), *legend.get_texts())]
    for legend in fig.legends:
        if legend.get_visible():
            found += [(t, "other", None) for t in (legend.get_title(), *legend.get_texts())]
    found += [(t, "other", None) for t in fig.texts]
    return [(t, "letter" if t.get_gid() == "panel_label" else role, ax) for t, role, ax in found if t.get_visible() and t.get_text().strip()]


def overlaps(boxes_a, boxes_b) -> np.ndarray:
    """Whether each box of a overlaps each box of b by more than LAYOUT_OVERLAP_MM both ways (len(a) x len(b))."""
    a = np.array([b.extents for b in boxes_a]).reshape(-1, 4)
    b = np.array([b.extents for b in boxes_b]).reshape(-1, 4)
    width = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    height = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    return (width > S.LAYOUT_OVERLAP_MM) & (height > S.LAYOUT_OVERLAP_MM)


def sibling_groups(panel_axes) -> list:
    """(panel, axes left to right) per gridspec row holding two or more of a panel's axes side by side (plots.columns'
    pair columns, among others)."""
    groups = {}
    for panel, axes in panel_axes.items():
        for ax in axes:
            spec = ax.get_subplotspec()
            if spec is not None and ax.get_visible():
                groups.setdefault((panel, spec.get_gridspec(), spec.rowspan.start), []).append(ax)
    return [(key[0], sorted(axes, key=lambda a: a.get_subplotspec().colspan.start)) for key, axes in groups.items() if len(axes) > 1]


def pooled_split(axes) -> tuple:
    """A sibling group's (pair columns, pooled columns): the pooled column sits after an empty gridspec column (plots.columns' gap)."""
    starts = [ax.get_subplotspec().colspan.start for ax in axes]
    k = next((i for i, start in enumerate(starts) if start != i), len(axes))
    return axes[:k], axes[k:]


def name_tokens(text) -> list:
    """A name's words and punctuation marks, lower case (X7)."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def within(short, long) -> bool:
    """Whether short's tokens appear as one run in long's."""
    return any(long[i:i + len(short)] == short for i in range(len(long) - len(short) + 1))


def interleaved(short, long) -> bool:
    """Whether short's tokens appear in order in long's but not as one run: words or punctuation inserted inside a name."""
    rest = iter(long)
    return all(t in rest for t in short) and not within(short, long)


def layout_problems(fig, panel_axes, rows) -> list:
    """The layout checks (check_layout), measured in mm on the rendered figure: one string per problem, report-only ones
    prefixed "warn:". panel_axes: panel -> its axes (draw); rows: draw's rows of panels."""
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    width, height = fig.get_size_inches() * 25.4
    down = lambda y: f"{height - y:.1f}"   # a height as mm from the figure's top
    texts = [(t, role, ax, mm_box(Text.get_window_extent(t, renderer), fig)) for t, role, ax in drawn_texts(fig)]
    box_of = {t: box for t, _, _, box in texts}
    axes = [ax for ax in fig.axes if ax.get_visible()]
    frame = {ax: mm_box(ax.get_window_extent(renderer), fig) for ax in axes}
    drawn = {ax: mm_box(box, fig) for ax in axes if (box := drawn_box(ax, renderer)) is not None}
    panel_of = {ax: panel for panel, group in panel_axes.items() for ax in group}
    panel_box = {panel: Bbox.union([drawn[ax] for ax in group if ax in drawn]) for panel, group in panel_axes.items()
                 if any(ax in drawn for ax in group)}
    legends = [legend for legend in [*fig.legends, *(ax.get_legend() for ax in axes)] if legend is not None and legend.get_visible()]
    problems = []

    # L1, L2: panel letters (a letter's panel: the one owning its axes, else the panel it names)
    letters = [(t, box, panel_of.get(t.axes) or next((p for p in panel_axes if panel_letter(p) == t.get_text()), None))
               for t, role, _, box in texts if role == "letter"]
    left = [(t, box) for t, box, panel in letters if panel in [row[0] for row in rows]]
    if left and max(b.x0 for _, b in left) - min(b.x0 for _, b in left) > S.LAYOUT_LETTER_ALIGN_MM:
        problems.append("L1: the left column's letters' left edges differ: " + ", ".join(f"{t.get_text()} {b.x0:.1f}" for t, b in left)
                        + f" mm (within {S.LAYOUT_LETTER_ALIGN_MM})")
    for t, box, panel in letters:
        if panel not in panel_box:
            problems.append(f"L2: letter {t.get_text()!r} belongs to no drawn panel")
        elif not -S.LAYOUT_LETTER_TOP_MM[1] <= box.y1 - panel_box[panel].y1 <= S.LAYOUT_LETTER_TOP_MM[0]:
            problems.append(f"L2: letter {t.get_text()}'s top {down(box.y1)} mm, its panel's top {down(panel_box[panel].y1)} mm "
                            f"(at most {S.LAYOUT_LETTER_TOP_MM[0]} above, {S.LAYOUT_LETTER_TOP_MM[1]} below)")

    # W1: the gaps between consecutive rows
    drawn_rows = [row for row in rows if any(p in panel_box for p in row)]
    row_box = [Bbox.union([panel_box[p] for p in row if p in panel_box]) for row in drawn_rows]
    low, high = S.LAYOUT_ROW_GAP_MM
    for i in range(len(drawn_rows) - 1):
        gap = row_box[i].y0 - row_box[i + 1].y1
        if not low <= gap <= high:
            names = ["+".join(p for p in row if p) for row in drawn_rows[i:i + 2]]
            problems.append(f"W1: rows {names[0]} | {names[1]}: gap {gap:.1f} mm ({low}-{high})")

    # W2: empty bands and side margins
    content = [*drawn.values(), *(box for _, role, ax, box in texts if ax is None or role == "letter"),
               *(mm_box(a.get_window_extent(renderer), fig) for a in [*fig.legends, *fig.lines, *fig.patches] if a.get_visible())]
    spans = sorted((b.y0, b.y1) for b in content)
    reach = spans[0][1]
    for y0, y1 in spans[1:]:
        if y0 - reach > S.LAYOUT_EMPTY_MM:
            problems.append(f"W2: an empty band {y0 - reach:.1f} mm tall, {down(y0)}-{down(reach)} mm from the top")
        reach = max(reach, y1)
    margins = {"left": min(b.x0 for b in content), "right": width - max(b.x1 for b in content)}
    problems += [f"W2: the {side} margin is {value:.1f} mm" for side, value in margins.items() if value > S.LAYOUT_EMPTY_MM]

    # W3 (report-only): empty space inside a panel's row, not at its right end nor before a pooled column (plots.columns' gap)
    groups = sibling_groups(panel_axes)
    pooled = {ax for _, group in groups for ax in pooled_split(group)[1]}
    for panel, group in panel_axes.items():
        spans = sorted((drawn[ax].x0, drawn[ax].x1, ax in pooled) for ax in group if ax in drawn)
        reach = spans[0][1] if spans else 0
        for x0, x1, after_gap in spans[1:]:
            if x0 - reach > S.LAYOUT_EMPTY_MM and not after_gap:
                problems.append(f"warn: W3: panel {panel}: an empty gap {x0 - reach:.1f} mm wide, {reach:.1f}-{x0:.1f} mm from the left")
            reach = max(reach, x1)

    # X1: text outside the figure
    problems += [f"X1: {t.get_text()!r} extends outside the figure (x {b.x0:.1f}-{b.x1:.1f}, y {down(b.y1)}-{down(b.y0)} mm)"
                 for t, _, _, b in texts if b.x0 < 0 or b.y0 < 0 or b.x1 > width or b.y1 > height]

    # X2: overlaps. Two rotated texts are not compared (their boxes overlap where the texts don't).
    plain = [(t, box) for t, role, _, box in texts if role != "letter"]
    hits = overlaps([b for _, b in plain], [b for _, b in plain])
    problems += [f"X2: {plain[i][0].get_text()!r} overlaps {plain[j][0].get_text()!r}" for i, j in zip(*np.nonzero(np.triu(hits, 1)))
                 if not (plain[i][0].get_rotation() and plain[j][0].get_rotation())]
    owned = [(t, role, ax, box) for t, role, ax, box in texts if role in ("xtick", "ytick", "xlabel", "ylabel", "title")]
    framed = [ax for ax in axes if ax.axison]
    hits = overlaps([b for *_, b in owned], [frame[ax] for ax in framed])
    foreign = {}   # (role, its panel, the other axes' panel) -> texts; twins (one frame) aren't foreign
    for i, j in zip(*np.nonzero(hits)):
        if framed[j] is not owned[i][2] and frame[framed[j]].bounds != frame[owned[i][2]].bounds:
            foreign.setdefault((owned[i][1], panel_of.get(owned[i][2]), panel_of.get(framed[j])), []).append(owned[i][0].get_text())
    problems += [f"X2: {len(found)} {role} text of panel {mine} overlap another axes of panel {theirs} (e.g. "
                 f"{', '.join(repr(x) for x in sorted(set(found))[:3])})" for (role, mine, theirs), found in foreign.items()]
    things = [*((repr(t.get_text()), b) for t, _, _, b in texts), *((f"an axes (panel {panel_of.get(ax)})", frame[ax]) for ax in framed),
              *(("a key", mm_box(legend.get_window_extent(renderer), fig)) for legend in legends)]
    for t, box, _ in letters:
        others = [(name, b) for name, b in things if b is not box]
        hit = overlaps([box], [b for _, b in others])[0]
        if hit.any():
            problems.append(f"X2: letter {t.get_text()} overlaps " + ", ".join(sorted({name for (name, _), h in zip(others, hit) if h})))

    # X3 (report-only): matching pair columns aligned across rows; X4: siblings' tops and heights
    for (p1, g1), (p2, g2) in itertools.combinations(groups, 2):
        u1, u2 = Bbox.union([frame[ax] for ax in g1]), Bbox.union([frame[ax] for ax in g2])
        (pairs1, pooled1), (pairs2, pooled2) = pooled_split(g1), pooled_split(g2)
        if p1 == p2 or (u1.y0 < u2.y1 and u2.y0 < u1.y1) or len(pairs1) != len(pairs2):
            continue
        matched = [*zip(pairs1, pairs2), *(zip(pooled1, pooled2) if len(pooled1) == len(pooled2) == 1 else [])]
        off = max(abs(frame[a].x0 - frame[b].x0) for a, b in matched)
        if off > S.LAYOUT_COLUMN_ALIGN_MM:
            problems.append(f"warn: X3: panels {p1} and {p2}: matching columns' left edges differ by up to {off:.1f} mm")
    for panel, group in groups:
        tops, heights = [frame[ax].y1 for ax in group], [frame[ax].height for ax in group]
        if max(tops) - min(tops) > S.LAYOUT_SIBLING_MM or max(heights) - min(heights) > S.LAYOUT_SIBLING_MM:
            problems.append(f"X4: panel {panel}: sibling axes' tops differ by {max(tops) - min(tops):.1f} mm, heights by "
                            f"{max(heights) - min(heights):.1f} mm")

    # X5: font roles: tick labels one size, at least MIN_FONT_SIZE; letters PANEL_LABEL; other text in LAYOUT_TEXT_PT (report-only)
    ticks = {}
    for t, role, _, _ in texts:
        if role in ("xtick", "ytick"):
            ticks.setdefault(t.get_fontsize(), []).append(t.get_text())
    if len(ticks) > 1 or min(ticks, default=S.MIN_FONT_SIZE) < S.MIN_FONT_SIZE:
        problems.append("X5: tick labels at " + ", ".join(f"{size:g} pt ({len(found)}, e.g. {sorted(set(found))[0]!r})" for size, found in sorted(ticks.items()))
                        + f" (one size, at least {S.MIN_FONT_SIZE} pt)")
    sizes = {}
    for t, role, _, _ in texts:
        size = t.get_fontsize()
        if role == "letter" and (size != S.PANEL_LABEL["size"] or t.get_fontweight() != S.PANEL_LABEL["weight"]):
            problems.append(f"X5: letter {t.get_text()} at {size:g} pt {t.get_fontweight()}, not {S.PANEL_LABEL['size']} pt {S.PANEL_LABEL['weight']}")
        elif role not in ("xtick", "ytick", "letter") and not S.LAYOUT_TEXT_PT[0] <= size <= S.LAYOUT_TEXT_PT[1]:
            sizes.setdefault(("warn: ", f"{role} text at {size:g} pt, outside {S.LAYOUT_TEXT_PT[0]}-{S.LAYOUT_TEXT_PT[1]}"), []).append(t.get_text())
    problems += [f"{prefix}X5: {len(found)} {what} (e.g. {', '.join(repr(x) for x in sorted(set(found))[:3])})"
                 for (prefix, what), found in sizes.items()]

    # X6: a y label near its tick labels; a shared x label centered under its columns
    for ax in framed:
        ticks = [box for t, role, a, box in texts if a is ax and role == "ytick"]
        text = box_of.get(ax.yaxis.label)
        if text is not None and ticks:
            tick = Bbox.union(ticks)
            gap = tick.x0 - text.x1 if text.x0 < tick.x0 else text.x0 - tick.x1
            if gap > S.LAYOUT_YLABEL_GAP_MM:
                problems.append(f"X6: y label {ax.yaxis.label.get_text()!r} (panel {panel_of.get(ax)}) {gap:.1f} mm from its tick labels")
    for panel, group in groups:
        labelled = [ax for ax in group if ax.axison and ax.xaxis.label in box_of]
        if len(labelled) == 1:
            text = box_of[labelled[0].xaxis.label]
            middle = group[len(group) // 2]   # plots.centered_xlabel's center: the middle column, or the gap between the middle two
            center = (frame[middle].x0 + frame[middle].x1) / 2 if len(group) % 2 else (frame[group[len(group) // 2 - 1]].x1 + frame[middle].x0) / 2
            off = (text.x0 + text.x1) / 2 - center
            row = [frame[ax] for ax in panel_axes[panel] if ax in frame and ax.axison
                   and abs(frame[ax].y0 - frame[labelled[0]].y0) < 1]   # a label shared by several grids: all of the panel's columns on its row
            across = (text.x0 + text.x1) / 2 - (min(b.x0 for b in row) + max(b.x1 for b in row)) / 2
            if abs(off) > S.LAYOUT_XLABEL_CENTER_MM and abs(across) > S.LAYOUT_XLABEL_CENTER_MM:
                problems.append(f"X6: panel {panel}: shared x label {labelled[0].xaxis.label.get_text()!r} {off:+.1f} mm off its columns' center")

    # X7: pair titles and configuration labels in their canonical words and punctuation (case, line breaks, spacing and a
    # pooled pair's mark aside): a text whose words and punctuation hold a name's, or are held in it, with others inside
    # (not a name with words before or after it, as "Human blood aging p = 0.96"); registered forms (the pairs' names,
    # config/figures.SHORT_LABELS) are names too, and a text holding one in a run is taken as that name with more words
    canonical = {" ".join(name.split()) for name in [*(pair_title(p) for p in PAIRS), *(label(k) for k in
                 [*configurations.CONFIGURATIONS, *configurations.REFERENCE_ROWS, *F.ROW_LABELS])]}
    registered = {tuple(name_tokens(n)) for n in [*canonical, *(PAIRS[p]["name"] for p in PAIRS), *F.SHORT_LABELS.values()]}
    shown = {t.get_text().removesuffix(f" {F.FIGURE_TEXT['pooled_mark']}") for t, *_ in texts}
    problems += [f"X7: {' '.join(text.split())!r} differs from {name!r}" for text in sorted(shown)
                 if not any(within(list(r), name_tokens(text)) for r in registered) for name in sorted(canonical)
                 if interleaved(name_tokens(text), name_tokens(name)) or interleaved(name_tokens(name), name_tokens(text))]
    return problems


def check_layout(fig, item, panel_axes, rows) -> list:
    """The layout checks (layout_problems: L1-L2 letters, W1-W3 white space, X1-X7 overlaps, alignment, fonts, names),
    printed; asserted none (report-only "warn:" lines aside) unless the item is in config/figures.LAYOUT_PENDING or the run
    is blinded."""
    problems = layout_problems(fig, panel_axes, rows)
    failures = [p for p in problems if not p.startswith("warn:")]
    print(f"{item}: layout: {len(failures)} problems, {len(problems) - len(failures)} warnings" + "".join(f"\n  {item}: {p}" for p in problems))
    # a blinded run leaves panels undrawn (a stated note in their place), so its layout is reported only
    assert not failures or item in F.LAYOUT_PENDING or settings.BLINDED is not None, f"{item}: layout problems:\n" + "\n".join(failures)
    return problems


def blob_hash(path) -> str:
    """The file's git blob hash (what `git hash-object` prints)."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


@functools.cache
def current_code(stage, pair) -> str:
    return codehash.stage_hash(settings.REPO / "pipeline" / f"{stage}.py", pair, codehash.extra(stage, pair))


@functools.cache
def content_hash(path) -> str:
    return manifest.content_hash(path)


def sources(paths_, repository=()) -> dict:
    """The inputs' provenance, asserting the real/blinded rule, a clean tree (a tag in full-run mode) and that every
    input is current (its recorded code hash equals the tree's: the run_all dry run's comparison). Repository files
    (no manifest) are recorded by their git blob hash."""
    paths_ = sorted(set(paths_))
    if settings.BLINDED is None:
        manifest.assert_unblinded(paths_)
    out = []
    for path in paths_:
        record = manifest.read(path)
        labelled = path.name.startswith(LABELLED)
        root = settings.LABELLED_OUTPUTS if labelled else settings.OUTPUTS
        stage, pair = path.relative_to(root).parts[:2]
        if settings.BLINDED is not None:
            assert record["blinded"] == settings.BLINDED or (not labelled and record["blinded"] is None), \
                f"{path}: blinded {record['blinded']} in a run blinded with {settings.BLINDED}"
        assert not record["git"]["dirty"], f"{path}: made from a dirty tree"
        assert record["tag"] or not settings.FULL_RUN, f"{path}: untagged in full-run mode"
        code = current_code(stage, pair if pair in PAIR_ORDER else None)
        assert record["settings"]["code"] == code, f"{path}: stale (code changed since)"
        out.append({"path": str(path.relative_to(settings.OUTPUTS)), "content_hash": content_hash(path), "commit": record["git"]["commit"],
                    "tag": record["tag"], "blinded": record["blinded"]})
    files = [{"path": str(Path(p).relative_to(settings.REPO)), "blob": blob_hash(p)} for p in sorted(set(repository))]
    return {"inputs": out, "repository": files, "commits": sorted({i["commit"] for i in out})}


def save(fig, item, script, panel_axes, rows):
    """Check and write the figure (PDF and PNG, byte-reproducible), each panel's own PNG (panel_axes: panel -> its axes,
    cropped closely to their extent at the same dpi, without the panel letter; the composite is unchanged) and its
    sources.json. The layout checks (check_layout; rows: draw's rows of panels) run on the finished figure, before the
    corner stamps."""
    check_layout(fig, item, panel_axes, rows)
    if WATERMARK:
        fig.text(0.998, 0.002, WATERMARK["text"], ha="right", va="bottom", color=WATERMARK["color"], alpha=WATERMARK["alpha"],
                 fontsize=WATERMARK["size"])
    pending = geneformer_pending()
    if pending:
        fig.text(0.002, 0.002, style.PENDING_WATERMARK["text"], ha="left", va="bottom", color=style.PENDING_WATERMARK["color"],
                 alpha=style.PENDING_WATERMARK["alpha"], fontsize=style.PENDING_WATERMARK["size"])
    check_figure(fig, item)
    out = directory(item)
    name = stem(item)
    with plt.rc_context({"svg.hashsalt": "sc2bulk"}):   # fixed SVG ids: byte-reproducible, as the PDF and PNG
        for suffix, metadata, dpi in (("pdf", {"CreationDate": None, "Producer": None}, S.RASTER_DPI), ("png", {"Software": None}, S.DPI),
                                      ("svg", {"Date": None}, S.RASTER_DPI)):
            write_bytes(out / f"{name}.{suffix}", lambda p: fig.savefig(p, format=suffix, dpi=dpi, metadata=metadata))
    renderer = fig.canvas.get_renderer()
    letters = [t for t in fig.findobj(Text) if t.get_gid() == "panel_label"]
    for letter in letters:
        letter.set_visible(False)
    for panel, axes in panel_axes.items():
        box = Bbox.union([ax.get_tightbbox(renderer) for ax in axes]).transformed(fig.dpi_scale_trans.inverted()).padded(PANEL_PAD_IN)
        write_bytes(out / f"{panel_name(item, panel)}.png",
                    lambda p: fig.savefig(p, format="png", dpi=S.DPI, metadata={"Software": None}, bbox_inches=box))
    for letter in letters:
        letter.set_visible(True)
    plt.close(fig)
    assert (out / f"{name}.pdf").stat().st_size <= S.MAX_FIGURE_BYTES, f"{item}: over 10 MB"
    record = {"item": item, "script": blob_hash(script), "blinded": settings.BLINDED, "geneformer_pending": pending,
              **sources(INPUTS, REPOSITORY)}
    write_bytes(out / "sources.json", lambda p: p.write_text(json.dumps(record, indent=1, sort_keys=True)))


def figure(item, height_mm=None):
    """A new figure at the item's width and height (mm)."""
    width = S.WIDTH_MM[F.ITEMS[item]["width"]]
    return plt.figure(figsize=(width / 25.4, (height_mm or F.HEIGHT_MM[item]) / 25.4))


def draw(item, functions, pairs, panels, script, ratios=None, legend=None, top_mm=12, hspace=0.55, rows=None, widths=None,
         wspace=0.3, bottom=0.06, left=0.17, right=0.97):
    """Lay out an item: its registered panels as rows (or rows: tuples of panels side by side, widths: their width
    ratios), each drawn by its function(fig, cell, pairs). panels: item -> the panels to draw (make_all: every panel, or
    in a blinded run config/figures.BLINDED_PANELS, whose others are left with a stated note). An item with no panel to
    draw is neither drawn nor saved, so earlier files stay."""
    assert list(functions) == list(F.PANELS[item]), f"{item}: panel functions {list(functions)} != registry {list(F.PANELS[item])}"
    rows = rows or [(panel,) for panel in functions]
    assert [p for row in rows for p in row if p is not None] == list(functions), f"{item}: rows {rows}"   # None: an empty slot
    if not panels[item]:
        return
    start()
    fig = figure(item)
    height = F.HEIGHT_MM.get(item)
    if isinstance(hspace, tuple):   # one gap per pair of rows (as a fraction of the mean row height): spacer rows between them
        assert len(hspace) == len(rows) - 1, f"{item}: {len(hspace)} gaps for {len(rows)} rows"
        heights = list(ratios or [1] * len(rows))
        mean = sum(heights) / len(heights)
        heights = [h for i, r in enumerate(heights) for h in ([r] if i == 0 else [hspace[i - 1] * mean, r])]
        grid = fig.add_gridspec(len(heights), 1, height_ratios=heights, top=1 - top_mm / height, bottom=bottom, left=left,
                                right=right, hspace=0)
        row_cells = [grid[2 * i] for i in range(len(rows))]
    else:
        grid = fig.add_gridspec(len(rows), 1, height_ratios=ratios, top=1 - top_mm / height, bottom=bottom, left=left, right=right,
                                hspace=hspace)
        row_cells = [grid[i] for i in range(len(rows))]
    cells, panel_axes = {}, {}
    for i, row in enumerate(rows):
        if len(row) == 1:
            cells[row[0]] = row_cells[i]
        else:
            sub = row_cells[i].subgridspec(1, len(row), width_ratios=widths[i] if widths else None,
                                      wspace=wspace[i] if isinstance(wspace, tuple) else wspace)   # one, or one per row
            cells.update({panel: sub[j] for j, panel in enumerate(row) if panel is not None})
    for panel, function in functions.items():
        if panel in panels[item]:
            before = set(fig.axes)
            function(fig, cells[panel], pairs)
            panel_axes[panel] = [ax for ax in fig.axes if ax not in before]
        else:
            assert settings.BLINDED is not None, f"{item} {panel}: only a blinded run leaves a panel undrawn"
            ax = fig.add_subplot(cells[panel])
            ax.axis("off")
            ax.text(0.5, 0.5, F.FIGURE_TEXT["not_drawn"], ha="center", va="center", transform=ax.transAxes)
    if legend is not None:
        legend(fig)
    save(fig, item, script, panel_axes, rows)


# ---- stage 11/11b rows as forest columns ----

def kept_pairs() -> list:
    """The pooled pairs: pseudobulk below 1 - X (stage 11's claims)."""
    return [p for p in PAIR_ORDER if p not in claims()["excluded_at_ceiling"]]


def threshold() -> str:
    """The pooling rule's bound, 1 - X, as printed ("0.95"; FIG2_TITLES' not-evaluable title carries it literally)."""
    bound = f"{1 - MARGIN:.2f}"
    assert bound in F.FIG2_TITLES["not_evaluable"], f"FIG2_TITLES' not-evaluable title doesn't name {bound}"
    return bound


def pseudobulk_auroc(pair) -> float:
    """Pseudobulk's bulk AUROC (stage 11's order; Fig 2B's line) and the pooling rule asserted against stage 11's claims:
    a pair enters the pooled average iff it is below 1 - X."""
    value = mean_over_seeds(aurocs(pair, "pseudobulk", **headline_conditions("pseudobulk")))
    assert at_ceiling(value, MARGIN) == (pair in claims()["excluded_at_ceiling"]), f"{pair}: pseudobulk AUROC {value} vs stage 11's pooling"
    return value


def pooled_header():
    """The pooled column's header by K, or None for K = 0."""
    kept = kept_pairs()
    if not kept:
        return None
    return F.FIGURE_TEXT["pooled_header" if len(kept) > 1 else "pooled_header_k1"].format(mark=F.FIGURE_TEXT["pooled_mark"])


def column_titles(pairs) -> dict:
    """Two-line pair titles, a pooled pair's marked, and the pooled header."""
    kept = kept_pairs()
    titles = {p: pair_title(p) + (f" {F.FIGURE_TEXT['pooled_mark']}" if p in kept else "") for p in pairs if p in PAIRS}
    return {**titles, "pooled": pooled_header()}


def short_column_titles(pairs) -> dict:
    """Figs. S9-S12's column titles: the pair title wrapped for a narrow column (as Fig 2's, so neighbours keep a gap), a
    pooled pair's marked (explained in the legend), and the pooled column's short header (None for K = 0)."""
    kept = kept_pairs()
    mark = F.FIGURE_TEXT["pooled_mark"]
    titles = {p: plots.narrow(pair_title(p)) + (f" {mark}" if p in kept else "") for p in pairs if p in PAIRS}
    pooled = None if not kept else F.FIGURE_TEXT["pooled_short" if len(kept) > 1 else "pooled_short_k1"].format(mark=mark)
    return {**titles, "pooled": pooled}


def secondary_columns(pairs, row, rows, item, name=lambda r: r["key"], **conditions) -> tuple:
    """(column -> {name: (delta, ci_low, ci_high, Holm glyph)}, the rows as a frame, the cells allowed to be n/a) of one
    pooled stage 11b row, per pair and pooled. 11b writes every member per pair (not estimable: NaN), so a missing per-pair
    row fails unless config/figures.NOT_APPLICABLE names it; the pooled column exists iff some pair is below the ceiling,
    and a pooled row may be missing only by 11b's own rule (not estimable, or not formed, in a non-ceiling pair)."""
    full = pick(secondary(), neutral=False, row=row, statistic="auroc_difference", **conditions)
    kept = kept_pairs()
    assert (full["pair"] == "pooled").any() == bool(kept), f"{row}: pooled rows {sorted(set(full['pair']))} with non-ceiling pairs {kept}"
    named = lambda part: {name(r): (r["delta"], r["ci_low"], r["ci_high"], glyph(r["p_better_holm"], r["p_worse_holm"])) for _, r in part.iterrows()}
    every = {p: named(full[full["pair"] == p]) for p in [*PAIR_ORDER, "pooled"]}
    applicable = F.NOT_APPLICABLE.get(item, set())
    na = set()
    for p in pairs:
        for r in rows:
            if r not in every[p]:
                assert (p, r) in applicable, f"{item} {row}: no row for {p} {r}, and it isn't listed as not applicable"
                na.add((p, r))
    if kept:
        for r in rows:
            if r not in every["pooled"]:
                reason = [p for p in kept if r not in every[p] or not np.isfinite(every[p][r][0])]
                assert reason, f"{item} {row}: no pooled row for {r} although every non-ceiling pair has one"
                na.add(("pooled", r))
    values = {p: every[p] for p in [*pairs, *(["pooled"] if kept else [])]}
    table = full[full["pair"].isin(values)]
    assert all(len(values[p]) == len(table[table["pair"] == p]) for p in values), f"{row}: two rows share a name"
    return values, table, na


def glyph(p_better_holm, p_worse_holm) -> str:
    return "".join(g for g, p in ((F.HOLM_GLYPH["better"], p_better_holm), (F.HOLM_GLYPH["worse"], p_worse_holm))
                   if np.isfinite(p) and p <= F.HOLM_ALPHA)


def absent(fig, cell, item, panel, pair):
    """A panel fixed to one pair that a pair-subset run doesn't include: n/a, stated in its CSV."""
    ax = fig.add_subplot(cell)
    ax.axis("off")
    ax.text(0.5, 0.5, F.FIGURE_TEXT["not_applicable"], ha="center", va="center", transform=ax.transAxes)
    style.panel_label(ax, panel)
    panel_csv(pd.DataFrame([{"pair": pair, "note": F.FIGURE_TEXT["not_applicable"]}]), item, panel)
