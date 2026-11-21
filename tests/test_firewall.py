"""The bulk-label firewall (sc2bulk.labels).

Bulk labels exist only in stage 02's bulk_labels.csv, read only through
sc2bulk.labels.bulk_labels by the files in ALLOWED. Every other stage sees bulk expression and
label-free metadata only. The allowed unlabeled-bulk uses are:
  - gene names for the shared-gene axis (stage 04);
  - the bulk side of alignment (stage 09);
  - transfer predictions on bulk (stage 08, written label-free, apart from the scores);
  - deconvolution inputs (stage 04b).

These static checks stop honest mistakes, not deliberate circumvention (code could build the
marked prefix at run time, as schemas.py does); the guarantee is the bulk-label permutation
test (tests/test_permutation.py), which reruns every stage that run_all marks as reading no bulk labels
under two different synthetic labels files and compares every output; the stages marked as reading them
write only labelled_ outputs (sc2bulk.manifest.LABELS_READ) and never run in its arms.

Two checks:
  - static: only the ALLOWED files name the labels file or its reader;
  - outputs (when they exist): the label-free bulk files carry only the columns listed here.
    The list is an allowlist, so a new column fails until it is reviewed.

Columns that encode the label (title, litter and treatment for Hammond/Reemst; nygc_group for
Pineda/NYGC; age_group for Sound Life) were found by testing every bulk column against the
0/1 group on kept samples (2026-10-04). Sample and donor ids are kept as join keys, though some
encode the label (Reemst GSM numbers run in age blocks; Sound Life donor prefixes BR1/BR2 are
young/old; Pineda's MSS donors are all controls, its JHU donors all ALS). No code may parse an
id. The bulk-label permutation test cannot catch code that does, so this rule is stated here.
Human blood, aging (Sound Life) has the same donors on both sides, so every bulk sample's
label can be inferred from the single-cell side by a join on donor_id; this is inherent. No
code outside the whitelist may make that join, and each cross-fitting fold's models are
built from its training donors only (sc2bulk.folds.training_samples; the fold-leak test).
"""
import ast
import re
from pathlib import Path

import anndata
import pandas as pd
import pytest

from sc2bulk import preprocess
from sc2bulk.config import settings
from sc2bulk.config.pairs import PAIR_ORDER

REPO = settings.REPO
ALLOWED = {
    "src/sc2bulk/labels.py",            # the reader
    "pipeline/02_preprocess.py",        # the writer
    "src/sc2bulk/scoring.py",           # bulk scores (AUROC)
    "src/sc2bulk/reference.py",         # the within-bulk reference and its folds
    "src/sc2bulk/concordance.py",       # Fig 5A/B bulk gene shifts and shared-gene shares
    "pipeline/10c_concordance.py",      # stage 10c: reads bulk labels for concordance.py
    "tests/test_concordance.py",        # 10c's tests (synthetic labels; names the labelled_ prefix)
    "src/sc2bulk/evaluate.py",          # the public entry point (scores via the scorer)
    "pipeline/11_statistics.py",        # statistics and bootstrap
    "pipeline/11b_secondary.py",        # the secondary statistics and the scores table
    "tests/test_statistics_secondary.py",   # 11b's tests (synthetic labels, patched in; names the labelled_ prefix)
    "scripts/blinded_counts.py",        # the class and litter counts (blinded run only)
    "tests/test_firewall.py",
    "tests/test_schedule.py",           # asserts the scheduler names neither the labels nor the switch
    "tests/test_permutation.py",        # the bulk-label permutation test
    "tests/test_labels.py",             # the blinded-run permutation
    "tests/test_statistics_driver.py",  # stage 11's fixed-input test (synthetic labels, patched in)
    "tests/test_figures_firewall.py",     # the figure scripts' static rules (names the reader and the labelled_ prefix)
    "tests/test_figures_end_to_end.py",   # every figure on a synthetic tree (synthetic labels patched in; names the labelled_ prefix)
}
ALLOWED_DIRECTORIES = ("figures/",)     # descriptive and scored panels
REFERENCE = re.compile(r"bulk_labels|labelled_")   # the labels file, and label-bearing outputs (schemas.LABELLED)

BULK_COLUMNS = {
    "hammond": ["condition", "donor_id", "library_size", "genes_detected"],
    "tms": ["donor_id", "organ", "sex", "library_size", "genes_detected"],
    "pineda": ["donor_id", "tissue", "project", "library_prep", "platform",
               "fraction_counts_outside_annotation", "library_size", "genes_detected"],
    "soundlife": ["donor_id", "sex", "cmv", "visit", "libraries", "library_size", "genes_detected"],
    "lupus": ["donor_id", "library_size", "genes_detected"],
    "crohns": ["donor_id", "age", "sex", "library_size", "genes_detected"],
}


def code_files():
    for directory in ("src", "pipeline", "figures", "tests"):
        yield from (REPO / directory).rglob("*.py")


def test_only_allowed_files_name_the_bulk_labels():
    offenders = [str(p.relative_to(REPO)) for p in [*code_files(), *(REPO / "scripts").rglob("*.py")]
                 if REFERENCE.search(p.read_text())
                 and str(p.relative_to(REPO)) not in ALLOWED
                 and not str(p.relative_to(REPO)).startswith(ALLOWED_DIRECTORIES)]
    assert not offenders, f"files outside the firewall whitelist name the bulk labels: {offenders}"


@pytest.mark.parametrize("pair", PAIR_ORDER)
def test_label_free_bulk_outputs(pair):
    out = settings.OUTPUTS / "02_preprocess" / pair
    if not (out / "bulk_samples.csv").exists():
        pytest.skip("stage 02 outputs not present")
    samples = pd.read_csv(out / "bulk_samples.csv", dtype=str)
    assert sorted(samples.columns) == sorted(["sample_id", "drop_reason", *BULK_COLUMNS[pair]]), list(samples.columns)
    bulk = anndata.read_h5ad(out / "bulk.h5ad", backed="r")
    assert sorted(bulk.obs.columns) == sorted(BULK_COLUMNS[pair]), list(bulk.obs.columns)
    # rows in the seeded order, so row order carries no label
    seeds = [preprocess.sample_seed(pair, s) for s in bulk.obs_names]
    assert seeds == sorted(seeds)
    labels = pd.read_csv(out / "bulk_labels.csv", dtype={"sample_id": str})
    assert list(labels["sample_id"]) == list(bulk.obs_names)
    single_cell = pd.read_csv(out / "single_cell_samples.csv", dtype=str)
    assert "split" not in single_cell, "fold membership lives only in 03_splits/<pair>/donor_folds.csv"


RESTRICTED_MODULES = {"scoring", "reference", "concordance", "labels", "evaluate"}
# Files that load modules by name, none label-bearing: 02_preprocess and 02b load the pair's data reader
# (sc2bulk.data.<pair>); config/pairs loads the pair configurations (config/pairs/<pair>.py).
# The same list is the code hash's dynamic-import exceptions (tests/test_codehash.py).
DYNAMIC_IMPORTS = {"pipeline/02_preprocess.py", "pipeline/02b_all_available.py", "src/sc2bulk/config/pairs/__init__.py",
                   "pipeline/07d_embed_all_available.py",   # 07d runs stage 07's functions (runpy; codehash.extra)
                   "pipeline/03b_attribution_splits.py",   # 03b runs stage 03's split() (runpy; codehash.extra)
                   "pipeline/09_alignment.py",   # 09 reads stage 08's features (runpy; codehash.extra)
                   "pipeline/09c_geometry.py",   # 09c runs stages 09 and 08's functions (runpy; codehash.extra)
                   "pipeline/12_simulation.py",   # 12 runs stage 06's helpers (runpy; codehash.extra)
                   "pipeline/12a_properties.py",   # 12a reads stage 08's features (runpy; codehash.extra)
                   "pipeline/08b_secondary.py",   # 08b runs stage 08's functions (runpy; codehash.extra)
                   "pipeline/11b_secondary.py",   # 11b runs stage 11's functions (runpy; codehash.extra)
                   "pipeline/10_attribution.py"}   # 10 runs stages 06, 07 and 08's functions (runpy; codehash.extra)


def imported_modules(tree):
    """sc2bulk modules a file imports, in any form (absolute or relative); True if it uses importlib."""
    names, uses_importlib = set(), False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "__import__":
            uses_importlib = True
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                uses_importlib |= parts[0] in ("importlib", "runpy")
                if parts[0] == "sc2bulk" and len(parts) > 1:
                    names.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            parts = node.module.split(".") if node.module else []
            if node.level > 0:   # relative: from . import X / from .X import y
                names |= {parts[0]} if parts else {a.name for a in node.names}
                continue
            uses_importlib |= bool(parts) and parts[0] in ("importlib", "runpy")
            if parts and parts[0] == "sc2bulk":
                names |= {parts[1]} if len(parts) > 1 else {a.name for a in node.names}
    return names, uses_importlib


def test_only_allowed_files_import_label_modules():
    offenders = []
    for path in code_files():
        name = str(path.relative_to(REPO))
        if name in ALLOWED or name.startswith(ALLOWED_DIRECTORIES) or name.startswith("src/sc2bulk/labels.py"):
            continue
        imported, uses_importlib = imported_modules(ast.parse(path.read_text()))
        # tests may load pipeline scripts with runpy to test their functions
        if imported & RESTRICTED_MODULES or (uses_importlib and name not in DYNAMIC_IMPORTS and not name.startswith("tests/")):
            offenders.append(name)
    assert not offenders, f"files outside the whitelist import label-bearing modules (or use importlib): {offenders}"


def test_import_check_catches_every_form():
    for source in ("import sc2bulk.scoring", "from sc2bulk import scoring", "from sc2bulk.scoring import score",
                   "from . import scoring", "from .scoring import score"):
        assert imported_modules(ast.parse(source))[0] & RESTRICTED_MODULES, source
    assert imported_modules(ast.parse("import importlib"))[1]
    assert imported_modules(ast.parse("__import__('sc2bulk.scoring')"))[1]
    assert imported_modules(ast.parse("import runpy"))[1]
