"""Table 1: the dataset pairs.

Two rows per pair (single-cell, then bulk), eight columns. Study, tissue, assay and groups from config (F.TABLE1,
F.FIG1D); single-cell counts and cells per sample (after QC) from stage 02's kept single-cell samples; bulk sample counts from
stage 02's label-free bulk sample table; bulk group counts from statistics.CLASS_COUNTS (pinned; group totals don't
depend on which samples carry which label), so no bulk label is read.

Outputs, in <outputs>/tables/Table1/: Table1.csv (one row per table row), Table1.md (title, pipe table, footnotes:
converted to a Word table with pandoc), sources.json."""
import json

import numpy as np
import pandas as pd

import common
from sc2bulk.config import settings, statistics
from sc2bulk.config import figures as F
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

ITEM = "Table 1"
SIDES = ("single_cell", "bulk")
DASH = "—"


def number(x) -> str:
    """A whole number without thousands separator; a half-integer median rounds up."""
    return str(int(np.floor(x + 0.5)))


def units(samples, pair) -> pd.Series:
    """Each counted unit's id: donors where the pair is scored per donor, else samples."""
    return samples["donor_id"] if PAIRS[pair]["scoring_unit"] == "donor" else samples["sample_id"]


def samples_cell(samples) -> str:
    n, donors = len(samples), samples["donor_id"].nunique()
    return number(n) if donors == n else f"{number(n)} ({number(donors)})"


def row(pair, side, groups, samples, cells) -> dict:
    citation, accession = F.TABLE1["studies"][pair][side]
    tissue, assay, compared = F.FIG1D[pair][side]
    values = [PAIRS[pair]["name"], F.TABLE1["sides"][side], f"{citation} ({accession})", tissue, assay,
              f"{compared}{F.TABLE1['group_marks'].get(pair, '')} ({number(groups[0])} / {number(groups[1])})", samples, cells]
    return dict(zip(F.TABLE1["columns"], values))


def group_counts(pair) -> tuple:
    """Units per class (0, 1) on each side: single-cell from stage 02's kept samples, bulk from statistics.CLASS_COUNTS
    (Fig 1D reads them here too)."""
    single = common.single_cell_samples(pair)
    per_unit = single.assign(unit=units(single, pair)).groupby("unit")["group"]
    assert per_unit.nunique().eq(1).all(), f"{pair}: a unit in both groups"
    counts = per_unit.first().value_counts()
    return (int(counts[0]), int(counts[1])), tuple(int(n) for n in np.sum(statistics.CLASS_COUNTS[pair], axis=0))


def rows(pair) -> list:
    single = common.single_cell_samples(pair)
    single_groups, bulk_groups = group_counts(pair)
    cells = single["cells_after_qc"]
    single_row = row(pair, "single_cell", single_groups, samples_cell(single),
                     f"{number(cells.median())} ({number(cells.min())}–{number(cells.max())})")
    bulk = common.bulk_samples(pair)
    assert units(bulk, pair).nunique() == sum(bulk_groups), f"{pair}: bulk units differ from CLASS_COUNTS"
    return [single_row, row(pair, "bulk", bulk_groups, samples_cell(bulk), DASH)]


def markdown(table) -> str:
    shown = table.copy()
    shown.loc[shown["Side"] == F.TABLE1["sides"]["bulk"], "Pair"] = ""   # the pair named once, on its single-cell row
    header = [c + F.TABLE1["header_marks"].get(c, "") for c in shown.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(shown.columns)]
    lines += ["| " + " | ".join(r) + " |" for r in shown.itertuples(index=False)]
    names = [PAIRS[p]["name"][0].lower() + PAIRS[p]["name"][1:] for p in PAIR_ORDER if PAIRS[p]["scoring_unit"] == "donor"]   # mid-sentence
    donor_pairs = "; ".join(names[:-1]) + "; and " + names[-1]   # serial "and"; pair names contain commas
    notes = [n.format(donor_pairs=donor_pairs, max_cells=settings.MAX_STORED_CELLS) for n in F.TABLE1["footnotes"]]
    return "\n".join([f"**{ITEM}. {F.TITLES[ITEM]}**", "", *lines, "", *notes, ""])


def main():
    common.start()
    table = pd.DataFrame([r for pair in PAIR_ORDER for r in rows(pair)], columns=F.TABLE1["columns"])
    assert len(F.TITLES[ITEM].split()) <= F.TITLE_MAX_WORDS
    out = settings.OUTPUTS / "tables" / ITEM.replace(" ", "")
    out.mkdir(parents=True, exist_ok=True)
    common.write_bytes(out / "Table1.csv", lambda p: table.to_csv(p, index=False))
    common.write_bytes(out / "Table1.md", lambda p: p.write_text(markdown(table)))
    record = {"item": ITEM, "script": common.blob_hash(__file__), **common.sources(common.INPUTS)}
    common.write_bytes(out / "sources.json", lambda p: p.write_text(json.dumps(record, indent=1, sort_keys=True)))
    return table


if __name__ == "__main__":
    main()
