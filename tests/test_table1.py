"""Table 1 (figures/table1.py) on a synthetic stage-02 tree, plus its config and static rules."""
import re
import sys

import numpy as np
import pandas as pd

from sc2bulk import folds
from sc2bulk.config import settings, statistics
from sc2bulk.config import figures as F
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

sys.path.insert(0, str(settings.REPO / "figures"))
import common  # noqa: E402
import table1  # noqa: E402
from test_firewall import REFERENCE  # noqa: E402  the labels file and label-bearing outputs
from test_schedule import SWITCH  # noqa: E402  the blinding switch

SCRIPT = settings.REPO / "figures" / "table1.py"
SOURCES = (settings.REPO / "src" / "sc2bulk" / "config" / "sources.py").read_text()


def build(root, pair):
    """Six kept single-cell samples (three per group; Sound Life's donors give two samples each) and one dropped;
    five kept bulk samples (the ALS pair's first donor gives two) and one dropped."""
    out = root / "02_preprocess" / pair
    out.mkdir(parents=True)
    donors = [f"d{i // 2}" for i in range(6)] if pair == "soundlife" else [f"d{i}" for i in range(6)]
    group = [0, 0, 1, 1, 1, 1] if pair == "soundlife" else [0, 0, 0, 1, 1, 1]
    single = pd.DataFrame({"sample_id": [f"s{i}" for i in range(7)], "donor_id": [*donors, "x"], "holdout_group": [*donors, "x"],
                           "group": [*group, 0], "cells_after_qc": [100, 200, 300, 400, 500, 2586, 50],
                           "drop_reason": [*[np.nan] * 6, "fewer than 100 cells"]})
    single.to_csv(out / "single_cell_samples.csv", index=False)
    bulk_donors = ["b0", "b0", "b1", "b2", "b3"] if pair == "pineda" else [f"b{i}" for i in range(5)]
    bulk = pd.DataFrame({"sample_id": [f"r{i}" for i in range(6)], "donor_id": [*bulk_donors, "y"],
                         "drop_reason": [*[np.nan] * 5, "outlier"]})
    bulk.to_csv(out / "bulk_samples.csv", index=False)


def test_table1(tmp_path, monkeypatch):
    for pair in PAIR_ORDER:
        build(tmp_path, pair)
    monkeypatch.setattr(settings, "OUTPUTS", tmp_path)
    monkeypatch.setattr(folds, "SAMPLES", tmp_path / "02_preprocess")
    monkeypatch.setattr(common, "CACHE", {})
    monkeypatch.setattr(common, "sources", lambda paths_: {"inputs": [str(p) for p in paths_]})
    counts = {p: [(2, 2)] for p in PAIR_ORDER}
    counts["soundlife"] = [(1, 1), (1, 0), (0, 1), (0, 1)]   # summed over folds: (2, 3)
    counts["hammond"] = counts["tms"] = [(2, 3)]             # scored per sample: 5 bulk samples
    counts["lupus"] = counts["crohns"] = [(4, 1)]            # scored per donor: 5 bulk donors
    monkeypatch.setattr(statistics, "CLASS_COUNTS", counts)
    table = table1.main()

    assert list(table.columns) == list(F.TABLE1["columns"]) and len(table) == 2 * len(PAIR_ORDER)
    assert list(table["Pair"]) == [PAIRS[p]["name"] for p in PAIR_ORDER for _ in (0, 1)]
    assert list(table["Side"]) == ["Single-cell", "Bulk"] * len(PAIR_ORDER)
    by = table.set_index(["Pair", "Side"])
    name = {p: PAIRS[p]["name"] for p in PAIR_ORDER}
    assert by.loc[(name["hammond"], "Single-cell"), "Samples (donors)"] == "6"
    assert by.loc[(name["soundlife"], "Single-cell"), "Samples (donors)"] == "6 (3)"
    assert by.loc[(name["soundlife"], "Single-cell"), "Groups compared (n per group)"].endswith("(1 / 2)")   # donors
    assert by.loc[(name["tms"], "Single-cell"), "Groups compared (n per group)"].endswith("(3 / 3)")         # samples
    assert by.loc[(name["soundlife"], "Bulk"), "Groups compared (n per group)"].endswith("(2 / 3)")
    assert by.loc[(name["pineda"], "Bulk"), "Samples (donors)"] == "5 (4)"
    assert by.loc[(name["hammond"], "Single-cell"), "Cells per sample, median (range)"] == "350 (100–2586)"
    assert (table.loc[table["Side"] == "Bulk", "Cells per sample, median (range)"] == "—").all()
    assert by.loc[(name["soundlife"], "Bulk"), "Study (accession)"] == "Gong et al. 2025; Smithmyer et al. 2025 (doi:10.57785/bn17-3h16)"   # no access mark
    assert not table["Study (accession)"].str.contains("[ᵃᵇᶜᵈ]").any()   # footnote marks only on the column headers

    out = tmp_path / "tables" / "Table1"
    assert pd.read_csv(out / "Table1.csv", dtype=str, keep_default_na=False).equals(table)
    text = (out / "Table1.md").read_text()
    assert text.startswith(f"**Table 1. {F.TITLES['Table 1']}**")
    assert "| Samples (donors)ᵇ |" in text and "\nᶜCells per sample after quality control" in text and "*" not in text.replace("**", "")
    assert not re.search(r"\d,\d", text)                           # no thousands separators
    assert len(text.split("\n\n")[-1].split()) <= 300              # the footnotes fit the legend's limit


def test_number():
    assert table1.number(2586999) == "2586999" and table1.number(350.0) == "350" and table1.number(4114.5) == "4115"


def test_studies_match_pair_config():
    """Each short accession's id (before any comma) appears in config/pairs' accession, and a "version v" in
    config/sources; every side has a study."""
    for pair in PAIR_ORDER:
        for side in table1.SIDES:
            citation, accession = F.TABLE1["studies"][pair][side]
            assert accession.split(",")[0].split()[-1] in PAIRS[pair][side]["accession"], (pair, side)
            version = re.search(r"version (\w+)", accession)
            assert version is None or version.group(1) in SOURCES, (pair, side)


def test_names_no_labels_or_switch():
    text = SCRIPT.read_text()
    assert not SWITCH.search(text) and not REFERENCE.search(text)
