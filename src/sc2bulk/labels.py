"""The bulk-label firewall and the blinded-run switch.

Bulk labels (label_value, the 0/1 group, and the pair's bulk_label_columns, which encode the
label) exist only in stage 02's bulk_labels.csv, and this is the only reader. Only the
modules listed in tests/test_firewall.py may call it: scoring, the within-bulk reference,
statistics and figures. Everything else sees bulk expression and label-free metadata only
(bulk.h5ad, bulk_samples.csv), so its outputs are the same whatever the bulk labels are.

Blinded runs (settings.BLINDED, from SC2BULK_PERMUTE_BULK_LABELS): bulk_labels() permutes whole
label rows (group, label_value and every label-encoding column, e.g. Reemst's litter) among
scoring units within strata (donors where the scoring unit is the donor; organs within mouse
organs; for a pair with cross-fitted donors, the donors scored in the same units, so each fold
keeps its real class counts), at read time; stage 02's labels file is never rewritten. label_value can differ within
a unit (Sound Life's age at each visit); a permuted unit takes its source unit's earliest
label_value (the donor's age at first visit) for all its samples. Label-bearing stages call
require_access() first: it refuses unless the labels are blinded or the analysis choices have been fixed, marked
by ANALYSIS_FIXED: an annotated git tag (analysis-fixed, with its tagger date) and its commit, which must match the
repository and be an ancestor of the checkout. So no real result exists before the analysis is
fixed. Once a process has read bulk labels it may write only label-bearing outputs (sc2bulk.manifest.LABELS_READ).
"""
import json
import subprocess

import numpy as np
import pandas as pd

from sc2bulk import folds, manifest, seeds
from sc2bulk.config import settings
from sc2bulk.config.pairs import PAIRS

STRATA = {"tms": "organ"}   # pairs whose labels are permuted within a stratum (bulk_samples column)
ANALYSIS_FIXED = settings.RAW.parent / "analysis_fixed.json"   # {"tag", "commit"}, written when the tag is pushed


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=settings.REPO, capture_output=True, text=True)


def require_access():
    """Assert that bulk labels may be read now: blinded, or once the analysis choices are fixed (ANALYSIS_FIXED)."""
    if settings.BLINDED is not None:
        return
    assert ANALYSIS_FIXED.exists(), \
        "bulk labels are locked until the analysis choices are fixed (or set SC2BULK_PERMUTE_BULK_LABELS for a blinded run)"
    fixed = json.loads(ANALYSIS_FIXED.read_text())
    assert git("cat-file", "-t", f"refs/tags/{fixed['tag']}").stdout.strip() == "tag", f"{fixed['tag']} isn't an annotated tag here"
    commit = git("rev-parse", "--verify", f"refs/tags/{fixed['tag']}^{{commit}}").stdout.strip()
    assert commit == fixed["commit"], f"the tag {fixed['tag']} is at {commit}, not the recorded {fixed['commit']}"
    assert git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0, f"this checkout predates the tag {fixed['tag']}"


def permute(labels: pd.DataFrame, unit: pd.Series, stratum: pd.Series, seed: int) -> pd.DataFrame:
    """Permute whole label rows among scoring units within strata. labels, unit and stratum are
    indexed by sample_id; every sample of a unit gets the same new row (label_value: the source
    unit's earliest)."""
    columns = [c for c in labels.columns if c != "sample_id"]
    per_unit = labels.assign(_unit=unit.loc[labels["sample_id"]].to_numpy(),
                             _stratum=stratum.loc[labels["sample_id"]].to_numpy())
    constant = [c for c in columns if c != "label_value"] + ["_stratum"]
    assert (per_unit.groupby("_unit")[constant].nunique() <= 1).all().all(), "labels vary within a scoring unit"
    rows = per_unit.sort_values(["_unit", "label_value"]).drop_duplicates("_unit")
    rng = np.random.default_rng(seed)
    new = {}
    for _, block in rows.groupby("_stratum", sort=True):
        block = block.sort_values("_unit")
        shuffled = block[columns].to_numpy()[rng.permutation(len(block))]
        new.update({u: values for u, values in zip(block["_unit"], shuffled)})
    out = pd.DataFrame([new[u] for u in per_unit["_unit"]], columns=columns).astype(labels[columns].dtypes.to_dict())
    return pd.concat([labels[["sample_id"]].reset_index(drop=True), out], axis=1)


def bulk_labels(pair: str) -> pd.DataFrame:
    """Kept bulk samples' labels: sample_id, label_value, group and the label columns (permuted
    by scoring unit when the run is blinded)."""
    require_access()
    manifest.LABELS_READ = True
    labels = pd.read_csv(settings.OUTPUTS / "02_preprocess" / pair / "bulk_labels.csv", dtype={"sample_id": str})
    if settings.BLINDED is None:
        return labels
    samples = pd.read_csv(settings.OUTPUTS / "02_preprocess" / pair / "bulk_samples.csv",
                          dtype={"sample_id": str, "donor_id": str}).set_index("sample_id")
    unit = samples["donor_id"] if PAIRS[pair]["scoring_unit"] == "donor" else pd.Series(samples.index, index=samples.index)
    stratum = samples[STRATA[pair]] if pair in STRATA else pd.Series("all", index=samples.index)
    if PAIRS[pair]["scoring_unit"] == "donor":   # within the units each donor is scored in
        assert pair not in STRATA
        donor_folds = pd.read_csv(folds.FOLDS / pair / "donor_folds.csv", dtype={"donor_id": str})
        scored = donor_folds[donor_folds["side"] == "scored"]
        cell = (scored["fold_set"] + "_" + scored["fold"].astype(str)).groupby(scored["donor_id"]).agg(",".join)
        stratum = samples["donor_id"].map(cell).fillna("all")
        assert scored.empty or labels["sample_id"].map(stratum).ne("all").all(), f"{pair}: a kept bulk donor outside the folds"
    return permute(labels, unit, stratum, seeds.derived("permutation", pair, settings.BLINDED))
