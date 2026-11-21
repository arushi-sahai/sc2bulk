"""The bulk-label permutation test.

The claim: with stage 02's label-free outputs and the code fixed, every output of every label-free stage after
stage 02 is a function of those inputs and the code alone, whatever the bulk labels are and however they are read.

Two parts:
  - static and unit checks (pytest, standard suite): the stage table matches the firewall's whitelist; only the
    allowed files name the blinding switch; every pipeline script is in run_all, on ADDED_IN_SCOPE or on EXCLUDED;
    the synthetic labels and the comparator on fixtures; and (SC2BULK_RUN_LEVEL=1, beside the real outputs) the
    run-level record <outputs>/checks/permutation.json matches the current code (the go/no-go before the first real
    bulk stage);
  - the run-level test (this file's main(), hours on one GPU, on the server only). Two arms rerun
    every in-scope stage on the smoke data from a copy of the smoke run's stage 02 outputs. They differ only in
    the bulk labels: arm B has a synthetic labels file S and SC2BULK_PERMUTE_BULK_LABELS unset; arm C has S',
    which flips every scoring unit but one, and the switch set to 2. Every output is then compared between the
    arms byte for byte, apart from a closed list of wall-clock fields, and the result is written to a record.

  python tests/test_permutation.py --run                    build or resume both arms, compare, write the record
                                                            (<root>/permutation.json and <outputs>/checks/)
  python tests/test_permutation.py --run --until 07d_embed_all_available   stop after a stage (the early prefix)
  python tests/test_permutation.py --run --from 08_readout  remove a stage and every later one from both arms first
  python tests/test_permutation.py --triage 08_readout      rerun a stage onward in a copy of arm B (B') and
                                                            compare it with B: nondeterminism (B' differs) or a
                                                            label dependence (B' equals B)

The driver needs SC2BULK_OUTPUTS (the real outputs, whose smoke/ subdirectory is the smoke run A), SC2BULK_SERVER_ROOT
(the default root: $SC2BULK_SERVER_ROOT/permutation_test) unless --root is given, one GPU in
CUDA_VISIBLE_DEVICES, A's SC2BULK_THREADS and the FM paths. A is only read: its stage 02 outputs (never the labels
file) and its FM pilot files. The arms run one at a time at the same path, <root>/R, and are renamed arm_B and
arm_C, so every absolute path a file records is the same in both. This file writes the synthetic files named
bulk_labels.csv, so it is on the firewall's whitelist (tests/test_firewall.py).
"""
import argparse
import filecmp
import json
import os
import re
import runpy
import shutil
import subprocess
import time
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

import anndata
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from sc2bulk import codehash, manifest
from sc2bulk.config import genes, settings, sources
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS
from sc2bulk.data import soundlife

REPO = settings.REPO
PIPELINE = REPO / "pipeline"
RUN_ALL = runpy.run_path(str(PIPELINE / "run_all.py"))
FIREWALL_ALLOWED = runpy.run_path(str(REPO / "tests" / "test_firewall.py"))["ALLOWED"]

A = settings.RAW.parent / "smoke"   # the smoke run, read only
A_BLINDED = 1   # the smoke run's blinding index
# The record the go/no-go reads: beside the real outputs (as tests/run_level/check09.py's), on the machine that runs
# the real bulk stages; never in the repository.
RECORD = settings.RAW.parent / "checks" / "permutation.json"
ARMS = {"B": None, "C": 2}   # arm -> SC2BULK_PERMUTE_BULK_LABELS (C's is below PERMUTATION_OFFSET)
LABEL_SEED = 20261007   # the synthetic labels' draw
LABELLED_PREFIX = settings.LABELLED + "_"

# Scope: run_all's entries that read no bulk labels, after the inputs
INPUT_STAGES = ("01_download", "02_preprocess")   # the arms' shared inputs
ADDED_IN_SCOPE = {"05s_eligibility_simulation.py"}   # in scope, outside run_all: the driver runs them
ENTRY_05S = ("05s_eligibility_simulation", "05s_eligibility_simulation.py", "main", ["--processes", "8"])
EXCLUDED = {"00_soundlife_slims.py": "writes RAW before stage 01",
            "05t_timing.py": "wall times by design; nothing downstream reads them",
            "run_all.py": "the driver"}
# Outside src/pipeline/scripts' firewall whitelist, only these may name the blinding switch
NAMERS = {"src/sc2bulk/config/settings.py", "src/sc2bulk/manifest.py", "src/sc2bulk/paths.py",
          "src/sc2bulk/config/training.py"}
SWITCH = re.compile(r"\bBLINDED\b|\bLABELLED_OUTPUTS\b|SC2BULK_PERMUTE_BULK_LABELS")
# The practitioner comparison (Methods) runs after stage 11b: it reads the label-bearing outputs and
# writes its own, with a manifest, under the label-bearing outputs folder
POST_HOC = {"scripts/practitioner_comparison.py"}
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

# The wall-clock fields: TrainingRecord's measurements, wherever a record is
# serialized (wall_clock_kind). Adding one needs a note on why it is a measurement.
NORMALIZED = ("seconds", "peak_gpu_bytes")

# The FM pilots, copied from A: run by hand, their decisions depend on measured speed
PILOTS = [f"10_attribution/{genes.PILOT['pair']}/pilot/pilot_{model}.json{suffix}"
          for model in ("scgpt", "geneformer") for suffix in ("", ".manifest.json")]

# Every raw file an in-scope stage opens: 02b's readers (per pair), 04's ortholog inputs, 04d's lupus
# single-cell file, and the GTF that stages 04, 07, 07d and 10 read. Any other raw file is absent from the view.
REEMST_FIRST = ("reemst", sorted(n for d, n in sources.DOWNLOADS if d == "reemst" and fnmatch(n, "GSM*.count.txt.gz"))[0])
STRIPPED = ("soundlife", sources.SOUNDLIFE_BULK)   # read for .var only; the arms get a var-only copy (no ages)
RAW_ALLOWLIST = sorted({
    *[("hammond", n) for d, n in sources.DOWNLOADS if d == "hammond" and fnmatch(n, "GSM*.dge.txt.gz")],
    REEMST_FIRST,
    *[("soundlife", soundlife.slim_name(n)) for n in sources.SOUNDLIFE_SINGLE_CELL],
    STRIPPED,
    *sources.SYNAPSE,   # Pineda: the S01 sample table and the library files
    *[("tms", f"tms_droplet_{organ.lower()}.h5ad") for organ in sources.TMS_SINGLE_CELL],
    ("lupus", sources.LUPUS_SINGLE_CELL),
    ("crohns", sources.CROHNS_SINGLE_CELL),
    *[("ensembl", n) for n in (*sources.ORTHOLOG_FILES.values(), sources.HUMAN_GTF)],
})

# What the arms cannot catch, stated in the record
CHANNELS = [
    "1. Sample and donor ids that encode the label (Reemst GSM blocks, incl. the first Reemst count file "
    f"{REEMST_FIRST[1]} that the raw view keeps; Sound Life BR1/BR2; Pineda MSS/JHU): no code may parse an id "
    "(firewall docstring).",
    "2. Label-encoding bulk metadata columns: the column allowlist (test_firewall.test_label_free_bulk_outputs).",
    "3. Human blood, aging's donor join to the single-cell side: inherent; the firewall's join rule and the "
    "fold-leak test.",
    "4. Raw files with bulk metadata: closed by the raw view (RAW_ALLOWLIST); Sound Life's bulk h5ad is a var-only "
    "copy, so no allowlisted file holds bulk labels.",
    "5. Stage 02 itself: the arms' shared input; the firewall's output test covers its columns and row order.",
    "6. Code paths not exercised on smoke data (testable-unit branches, 08a's real-run decision, scale-down and FM "
    "fallback steps): the import firewall, the name regex and the blinding-switch check.",
    "7. Reads that bypass settings.OUTPUTS (an absolute path into the real tree): coding_plan rule 25 (paths.output).",
]


def in_scope_entries() -> list:
    """(stage, script, environment, arguments) of every in-scope entry, in run order: run_all's label-free entries
    after the inputs, with 05s after 05_search."""
    table = [(stage, script, environment, list(args)) for stage, script, environment, args, labels, _ in RUN_ALL["STAGES"]
             if not labels and stage not in INPUT_STAGES]
    after = max(i for i, entry in enumerate(table) if entry[0] == "05_search") + 1
    return table[:after] + [ENTRY_05S] + table[after:]


ENTRIES = in_scope_entries()
STAGE_ORDER = list(dict.fromkeys(entry[0] for entry in ENTRIES))


def pairs_of(stage) -> list:
    return [None] if stage == ENTRY_05S[0] else PAIR_ORDER   # 05s runs once, without --pair


def other_code(smoke: Path, until=None) -> dict:
    """{"<stage>/<pair>": [outputs, outputs whose recorded code hash isn't the current tree's]} in a smoke tree, as
    run_all.dry_run counts them (the seed pilots left out)."""
    pilots = {smoke / n for n in PILOTS}
    out = {}
    for key, code in code_hashes(until).items():
        stage, pair = key.split("/")
        directory = smoke / stage if pair == "None" else smoke / stage / pair
        records = [json.loads(p.read_text()) for p in directory.rglob("*.manifest.json") if p not in pilots]
        out[key] = [len(records), sum(r["settings"].get("code") != code for r in records)]
    return out


def code_hashes(until=None) -> dict:
    """{"<stage>/<pair>": the current scoped code hash} of every written in-scope stage up to `until`, computed as
    run_all.dry_run computes it."""
    scripts = dict((entry[0], entry[1]) for entry in ENTRIES)
    stages = STAGE_ORDER[:STAGE_ORDER.index(until) + 1] if until else STAGE_ORDER
    return {f"{stage}/{pair}": codehash.stage_hash(PIPELINE / scripts[stage], pair, codehash.extra(stage, pair))
            for stage in stages if (PIPELINE / scripts[stage]).exists() for pair in pairs_of(stage)}


# --- The synthetic labels ---

def synthetic_labels(pair, units: pd.Series, arm) -> pd.DataFrame:
    """Arm B's labels S, or arm C's S', for one pair. units: each kept bulk sample's scoring unit, indexed by
    sample_id in bulk.h5ad's order. S puts a seeded third of the units in group 1; S' is S's complement on every
    unit but the first in sorted order, so the unordered class counts change too. Ages (pairs whose label_value
    is an age) are drawn per unit: 20-40 in group 0 and 60-80 in group 1 under S, one more under S'."""
    ordered = sorted(set(units))
    assert len(ordered) >= 3, f"{pair}: fewer than 3 scoring units"
    rng = np.random.default_rng([LABEL_SEED, settings.SEED_PAIRS[pair]])
    chosen = set(rng.permutation(ordered)[:len(ordered) // 3])
    ages = dict(zip(ordered, rng.integers(20, 41, len(ordered))))
    group = {u: int(u in chosen) for u in ordered}
    if arm == "C":
        group = {u: g if u == ordered[0] else 1 - g for u, g in group.items()}
    groups = units.map(group)
    value = units.map(ages) + 40 * groups + int(arm == "C") if PAIRS[pair]["label_value"] == "age" else groups
    labels = pd.DataFrame({"sample_id": units.index, "label_value": value.to_numpy(), "group": groups.to_numpy()})
    for column in PAIRS[pair]["bulk_label_columns"]:
        labels[column] = [f"synthetic_{arm}_{g}" for g in groups]
    return labels


def label_properties(pair, s, s_prime, units: pd.Series) -> dict:
    """Assert the properties of a pair's S and S' (and audit point 1's), and return them for the record."""
    first = []
    for labels in (s, s_prime):
        assert list(labels["sample_id"]) == list(units.index), f"{pair}: rows not in bulk.h5ad's order"
        by_unit = labels.groupby(units.to_numpy())["group"]
        assert (by_unit.nunique() == 1).all(), f"{pair}: a group varies within a scoring unit"
        first.append(by_unit.first())
    assert (first[0] != first[1]).sum() == len(first[0]) - 1, f"{pair}: S and S' must differ on every unit but one"
    samples = [[int((labels["group"] == g).sum()) for g in (0, 1)] for labels in (s, s_prime)]
    unit_counts = [[int((groups == g).sum()) for g in (0, 1)] for groups in first]
    for counts in (*samples, *unit_counts):
        assert min(counts) > 0, f"{pair}: a group is empty"
    assert sorted(samples[0]) != sorted(samples[1]), f"{pair}: S and S' have the same unordered sample counts"
    assert sorted(unit_counts[0]) != sorted(unit_counts[1]), f"{pair}: S and S' have the same unordered unit counts"
    return {"samples": len(units), "units": len(first[0]),
            "changed_sample_fraction": float((s["group"] != s_prime["group"]).mean()),
            "sample_counts": dict(zip(ARMS, samples)), "unit_counts": dict(zip(ARMS, unit_counts))}


def scoring_units(pair, stage02: Path) -> pd.Series:
    """Each kept bulk sample's scoring unit (donor or sample), indexed by sample_id in bulk.h5ad's order, from a
    pair's label-free stage 02 outputs."""
    names = list(anndata.read_h5ad(stage02 / "bulk.h5ad", backed="r").obs_names)
    samples = pd.read_csv(stage02 / "bulk_samples.csv", dtype={"sample_id": str, "donor_id": str}).set_index("sample_id")
    units = samples.loc[names, "donor_id"].to_numpy() if PAIRS[pair]["scoring_unit"] == "donor" else names
    return pd.Series(units, index=pd.Index(names, name="sample_id"))


# --- The comparator ---

def wall_clock_kind(name):
    """'parquet', 'json' or 'checkpoint' for an outputs-relative path whose file serializes a TrainingRecord (it
    holds NORMALIZED fields), else None: stage 05's candidates_s<i>.parquet (and the merge's
    candidates_<name>_s<i>.parquet), stage 06's and 12's record_*.json, stage 05's and 06's checkpoints/*.pt (not row
    J's scvi-tools checkpoints/s<e>/model.pt)."""
    path = PurePosixPath(name)
    stage = path.parts[0]
    if stage == "05_search" and fnmatch(path.name, "candidates*_s*.parquet"):
        return "parquet"
    if stage in ("06_train", "12_simulation") and fnmatch(path.name, "record_*.json"):
        return "json"
    if stage in ("05_search", "06_train") and path.parent.name == "checkpoints" and path.suffix == ".pt":
        return "checkpoint"
    return None


def without_wall_clock(record: dict):
    """The record without NORMALIZED fields (only those present), and the removed values."""
    return ({k: v for k, v in record.items() if k not in NORMALIZED},
            {k: record[k] for k in NORMALIZED if k in record})


def load_normalized(path, kind):
    """A wall-clock file's contents without its NORMALIZED fields, and the removed values."""
    if kind == "parquet":
        table = pq.read_table(path)
        frame = table.to_pandas()
        removed = {c: frame[c].tolist() for c in NORMALIZED if c in frame}
        return (frame.drop(columns=list(removed)), table.schema.metadata), removed
    if kind == "json":
        return without_wall_clock(json.loads(Path(path).read_text()))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    record, removed = without_wall_clock(checkpoint["record"])
    return {**checkpoint, "record": record}, removed


def same(a, b) -> bool:
    """Exact equality of nested values: tensors and numpy arrays and scalars by dtype, shape and value, frames by
    DataFrame.equals, NaN equal to NaN (outside tensors)."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, (np.ndarray, np.generic)):
        return (type(b) is type(a) and a.dtype == b.dtype and a.shape == b.shape
                and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    if isinstance(a, pd.DataFrame):
        return isinstance(b, pd.DataFrame) and a.equals(b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and a != a:
        return isinstance(b, float) and b != b
    return type(a) is type(b) and a == b


def placeholder(value):
    """settings with the content hash of every wall-clock file replaced, wherever it sits ({outputs-relative path:
    hash} entries: 05 select's candidates, 10's checkpoints, 12a's and 12 --readout's records)."""
    if isinstance(value, dict):
        return {k: "<wall-clock file>" if isinstance(v, str) and wall_clock_kind(k) else placeholder(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [placeholder(v) for v in value]
    return value


def normalized_manifest(path, name, blinded, seen, rename=None) -> dict:
    """A manifest without cost, blinded, git and settings_hash, with wall-clock files' hashes replaced (and a
    wall-clock output's own content_hash and size dropped). In an arm (rename None), first asserted: settings_hash
    matches the manifest's own settings, blinded is the arm's, git.dirty is False. rename: (old, new) applied to the
    manifest's text (A, whose paths are under the real outputs; information only, nothing asserted)."""
    record = json.loads(Path(path).read_text())
    if rename is None:
        assert record["settings_hash"] == manifest.settings_hash(record["settings"]), f"{path}: settings_hash doesn't match"
        assert record["blinded"] == blinded, f"{path}: blinded is {record['blinded']}, not {blinded}"
        assert record["git"]["dirty"] is False, f"{path}: made from a dirty tree"
    else:
        record = json.loads(Path(path).read_text().replace(*rename))
    seen["commits"].add(record["git"]["commit"])
    seen["environments"].add((record["environment"]["name"], record["environment"]["fingerprint"]))
    seen["gpus"].add(record["cost"]["gpu"])
    out = {k: v for k, v in record.items() if k not in ("cost", "blinded", "git", "settings_hash")}
    out["settings"] = placeholder(record["settings"])
    if wall_clock_kind(name.removesuffix(".manifest.json")):
        del out["content_hash"], out["size"]
    return out


def compared_files(smoke: Path) -> set:
    """Outputs-relative names of every file in an arm's smoke tree but the seed files; asserted: no label-bearing
    output or blinded_ directory."""
    names = {str(p.relative_to(smoke)) for p in smoke.rglob("*") if p.is_file()}
    marked = [n for n in names if any(part.startswith((LABELLED_PREFIX, "blinded_")) for part in n.split("/"))]
    assert not marked, f"{smoke}: label-bearing outputs: {marked}"
    return {n for n in names if not n.startswith("02_preprocess/") and n not in PILOTS}


def compare_file(a, b, name, blinded, seen, removed, rename):
    """'identical', 'normalized', or why the two copies of one output differ."""
    if name.endswith(".manifest.json"):
        x = normalized_manifest(a, name, blinded[0], seen, rename)
        y = normalized_manifest(b, name, blinded[1], seen)
        keys = sorted(k for k in x.keys() | y.keys() if not same(x.get(k), y.get(k)))
        return f"manifest fields {keys}" if keys else "normalized"
    if filecmp.cmp(a, b, shallow=False):
        return "identical"
    kind = wall_clock_kind(name)
    if kind is None:
        return "bytes"
    (x, removed_a), (y, removed_b) = load_normalized(a, kind), load_normalized(b, kind)
    removed.append({"file": name, "a": removed_a, "b": removed_b})
    return "normalized" if same(x, y) else "contents after normalization"


def compare(a: Path, b: Path, blinded=(None, ARMS["C"]), rename=None) -> dict:
    """Compare two smoke trees output by output. blinded: each side's expected manifest `blinded`. With rename
    (A against arm B), only arm B's files are looked up in a, which is never listed."""
    names_b = compared_files(b)
    names_a = compared_files(a) if rename is None else {n for n in names_b if (a / n).is_file()}
    seen = {"commits": set(), "environments": set(), "gpus": set()}
    stages, removed = {}, []
    for name in sorted(names_a & names_b):
        tally = stages.setdefault(name.split("/")[0], {"compared": 0, "identical": 0, "normalized": 0, "differing": {}})
        tally["compared"] += 1
        outcome = compare_file(a / name, b / name, name, blinded, seen, removed, rename)
        if outcome in ("identical", "normalized"):
            tally[outcome] += 1
        else:
            tally["differing"][name] = outcome
    only = sorted(names_a ^ names_b)
    failing = {n.split("/")[0] for n in only} | {stage for stage, tally in stages.items() if tally["differing"]}
    order = {stage: i for i, stage in enumerate(STAGE_ORDER)}
    environments = {}
    for env_name, fingerprint in seen["environments"]:
        environments.setdefault(env_name, []).append(fingerprint)
    return {"stages": stages, "only_in_a": sorted(names_a - names_b), "only_in_b": sorted(names_b - names_a),
            "first_difference": min(failing, key=lambda s: order.get(s, len(order))) if failing else None,
            "normalized_values": removed, "commits": sorted(seen["commits"]),
            "environments": {k: sorted(v) for k, v in environments.items()},
            "gpus": sorted(str(g) for g in seen["gpus"]), "passed": not failing}


# --- The arms ---

def read_only(directory: Path):
    for path in directory.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    directory.chmod(0o555)


def copy_pilots(R: Path):
    """A's pilot files, where the arm lacks them (A's are unchanged: seed_state)."""
    for name in PILOTS:
        if not (R / "smoke" / name).exists():
            (R / "smoke" / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(A / name, R / "smoke" / name)


def build_seed(R: Path, arm):
    """The arm's starting tree at R: the raw view (symlinks to RAW_ALLOWLIST, Sound Life's bulk h5ad as a
    var-only copy), a read-only copy of A's stage 02 outputs without the labels file, and the arm's synthetic labels
    (no manifest), whose properties are asserted first. No analysis-fixed marker. arm.txt, written last, names the arm.
    A's FM pilots are copied by run_arm when stage 10 is in range."""
    for directory, name in RAW_ALLOWLIST:
        source, path = settings.RAW / directory / name, R / "01_download" / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if (directory, name) == STRIPPED:
            with h5py.File(source, "r") as f:
                var = anndata.io.read_elem(f["var"])
            anndata.AnnData(var=var).write_h5ad(path)
        else:
            assert source.is_file(), f"{source}: an allowlisted raw file is missing"
            path.symlink_to(source)
    for pair in PAIR_ORDER:
        out = R / "smoke" / "02_preprocess" / pair
        shutil.copytree(A / "02_preprocess" / pair, out,
                        ignore=lambda _, names: [n for n in names if n.startswith(("bulk_labels", LABELLED_PREFIX))])
        units = scoring_units(pair, out)
        label_properties(pair, synthetic_labels(pair, units, "B"), synthetic_labels(pair, units, "C"), units)   # before any compute
        synthetic_labels(pair, units, arm).to_csv(out / "bulk_labels.csv", index=False)
        read_only(out)
    (R / "arm.txt").write_text(arm)


def arm_env(R: Path, blinded) -> dict:
    """run_all's environment for the smoke run at R, with the arm's blinding switch."""
    env = dict(os.environ)
    env.update({name: str(settings.N_THREADS) for name in THREAD_VARIABLES})
    env.update(SC2BULK_SMOKE="1", SC2BULK_OUTPUTS=str(R), SC2BULK_GENEFORMER_GO="1")
    env.pop("SC2BULK_PERMUTE_BULK_LABELS", None)
    env.pop("SC2BULK_FULL_RUN", None)
    if blinded is not None:
        env["SC2BULK_PERMUTE_BULK_LABELS"] = str(blinded)
    assert re.fullmatch(r"\d+", env.get("CUDA_VISIBLE_DEVICES", "")), "pin one GPU with CUDA_VISIBLE_DEVICES"
    return env


def recorded_env(env: dict) -> dict:
    """The variables the record names (none is a secret)."""
    return {k: v for k, v in sorted(env.items()) if k.startswith("SC2BULK_") or k == "CUDA_VISIBLE_DEVICES" or k in THREAD_VARIABLES}


def clean_head() -> str:
    state = manifest.git_state()
    assert not state["dirty"], "the arms run from a clean tree"
    return state["commit"]


def raw_state(R: Path) -> dict:
    """Every file in the raw view: its link target, and the target's size and modification time."""
    return {str(p.relative_to(R)): [os.readlink(p) if p.is_symlink() else None, p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted((R / "01_download").rglob("*")) if not p.is_dir()}


def seed_state() -> dict:
    """Size and SHA-256 of every seed source in A, and size and modification time of every raw file the view
    holds or is built from."""
    files = [p for pair in PAIR_ORDER for p in sorted((A / "02_preprocess" / pair).rglob("*"))
             if p.is_file() and not p.name.startswith(("bulk_labels", LABELLED_PREFIX))] + [A / n for n in PILOTS if (A / n).exists()]
    state = {str(p): [p.stat().st_size, manifest.content_hash(p)] for p in files}
    for directory, name in RAW_ALLOWLIST:
        stat = (settings.RAW / directory / name).stat()
        state[str(settings.RAW / directory / name)] = [stat.st_size, stat.st_mtime_ns]
    return state


def log_name(stage, environment, args, pair) -> str:
    return "_".join([stage, environment, *(a.lstrip("-") for a in args), pair or "all"])


def in_range(stage, start, until) -> bool:
    i = STAGE_ORDER.index(stage)
    return (start is None or i >= STAGE_ORDER.index(start)) and (until is None or i <= STAGE_ORDER.index(until))


def run_arm(root: Path, arm, blinded, start=None, until=None):
    """Build or resume one arm at root/R and run its in-scope entries in run_all's order, with run_all's commands
    (each skips its finished units), then move it to root/arm_<arm>. start: remove that stage and every later one
    first."""
    R, done = root / "R", root / f"arm_{arm}"
    if done.exists():
        done.rename(R)
    elif not R.exists():
        build_seed(R, arm)
    assert (R / "arm.txt").read_text() == arm, f"{R} holds another arm"
    if start:
        for stage in STAGE_ORDER[STAGE_ORDER.index(start):]:
            if (R / "smoke" / stage).exists():
                shutil.rmtree(R / "smoke" / stage)
    if in_range("10_attribution", start, until):   # the prefix (--until before 10) runs without A's pilots
        copy_pilots(R)
    env = arm_env(R, blinded)
    head, raw = clean_head(), raw_state(R)
    (R / "logs").mkdir(exist_ok=True)
    started = time.perf_counter()
    for stage, script, environment, args in ENTRIES:
        if not in_range(stage, start, until):
            continue
        if not (PIPELINE / script).exists():
            print(f"== {stage}: {script} not written yet; skipped", flush=True)
            continue
        for pair in pairs_of(stage):
            command = ["conda", "run", "--no-capture-output", "-n", RUN_ALL["ENVIRONMENTS"][environment],
                       "python", str(PIPELINE / script), *args, *(["--pair", pair] if pair else [])]
            log = R / "logs" / log_name(stage, environment, args, pair)
            print(f"== arm {arm}: {' '.join(command[5:])}", flush=True)
            with open(f"{log}.out", "w") as out, open(f"{log}.err", "w") as err:
                subprocess.run(command, env=env, stdout=out, stderr=err, check=True)
    assert clean_head() == head, "the commit changed during the arm"
    assert raw_state(R) == raw, "the raw view changed during the arm"
    assert not (R / "analysis_fixed.json").exists(), "an analysis-fixed marker appeared in the arm"
    compared_files(R / "smoke")   # no label-bearing output
    with open(R / "wall_seconds.txt", "a") as f:
        f.write(f"{head} {start} {until} {time.perf_counter() - started:.0f}\n")
    R.rename(done)
    return recorded_env(env)


def stdout_08a(a: Path, b: Path) -> dict:
    """Per pair: stage 08a's stdout equal in both arms (it writes no file in smoke); None where it didn't run."""
    names = {pair: f"logs/{log_name('08a_alpha_edges', 'main', [], pair)}.out" for pair in PAIR_ORDER}
    return {pair: (a / n).read_text() == (b / n).read_text() if (a / n).exists() and (b / n).exists() else None
            for pair, n in names.items()}


def a_information(root: Path) -> dict:
    """A against arm B (information only), and A's recorded code against the current tree: per in-scope stage
    and pair, how many of A's outputs carry another code hash; the pilots' and A's stage 02 code."""
    pilot = codehash.stage_hash(PIPELINE / "10_attribution.py", genes.PILOT["pair"], codehash.extra("10_attribution", genes.PILOT["pair"]))
    stage02 = {pair: json.loads((A / "02_preprocess" / pair / "bulk_samples.csv.manifest.json").read_text())["settings"]["code"]
               == codehash.stage_hash(PIPELINE / "02_preprocess.py", pair, codehash.extra("02_preprocess", pair))
               for pair in PAIR_ORDER}
    return {"comparison": compare(A, root / "arm_B" / "smoke", (A_BLINDED, None), rename=(str(settings.RAW.parent), str(root / "R"))),
            "outputs_with_other_code": other_code(A),
            "pilots_at_current_code": {n: json.loads((A / n).read_text())["settings"]["code"] == pilot
                                       for n in PILOTS if n.endswith(".manifest.json") and (A / n).exists()},
            "stage02_at_current_code": stage02}


def excluded() -> dict:
    """Every entry or script outside the arms, and why."""
    flagged = {stage: "reads bulk labels (run_all flag; firewall whitelist): never run in an arm"
               for stage, _, _, _, labels, _ in RUN_ALL["STAGES"] if labels}
    return {**flagged, "01_download": "the raw files (the arms see a view)", "02_preprocess": "the arms' shared input",
            "10_attribution --pilot": "run by hand; its decisions depend on measured speed (A's files are seeds)",
            **EXCLUDED}


def run(root: Path, start, until):
    """Both arms, the comparison and the record."""
    real = settings.RAW.parent.resolve()
    assert not root.resolve().is_relative_to(real) and not real.is_relative_to(root.resolve()), \
        f"the arm root {root} and the real outputs {real} must be apart"
    root.mkdir(parents=True, exist_ok=True)
    before = seed_state()
    held = (root / "R" / "arm.txt").read_text() if (root / "R" / "arm.txt").exists() else None
    environments = {arm: run_arm(root, arm, ARMS[arm], start, until) for arm in sorted(ARMS, key=lambda arm: arm != held)}
    after = seed_state()
    assert after == before, "a seed source changed"
    B, C = root / "arm_B", root / "arm_C"
    comparison = compare(B / "smoke", C / "smoke")
    stdout = stdout_08a(B, C)
    current = other_code(B / "smoke", until)   # a run with --from leaves earlier stages as they were
    labels = {}
    for pair in PAIR_ORDER:
        units = scoring_units(pair, B / "smoke" / "02_preprocess" / pair)
        labels[pair] = label_properties(pair, synthetic_labels(pair, units, "B"), synthetic_labels(pair, units, "C"), units)
    record = {
        "commit": clean_head(), "clean": True, "from": start, "until": until,
        "environment_variables": environments,
        "code": code_hashes(until),
        "outputs_with_other_code": current,
        "not_written": sorted({entry[1] for entry in ENTRIES if not (PIPELINE / entry[1]).exists()}),
        "excluded": excluded(),
        "raw_allowlist": [{"file": f"{d}/{n}", "size": (settings.RAW / d / n).stat().st_size,
                           "note": "stripped: var only" if (d, n) == STRIPPED else
                           "channel 1: its id encodes the label" if (d, n) == REEMST_FIRST else None}
                          for d, n in RAW_ALLOWLIST],
        "channels": CHANNELS,
        "labels": labels,
        "seed_sources": {"before": before, "after": after},
        "comparison": comparison,
        "stdout_08a": stdout,
        "a_vs_b": a_information(root),
        "wall_seconds": {arm: (root / f"arm_{arm}" / "wall_seconds.txt").read_text().splitlines() for arm in ARMS},
        "passed": comparison["passed"] and False not in stdout.values() and not any(n for _, n in current.values()),
    }
    text = json.dumps(record, indent=1, default=str)
    (root / "permutation.json").write_text(text)
    RECORD.parent.mkdir(exist_ok=True)
    RECORD.write_text(text)
    print(json.dumps({"passed": record["passed"], "first_difference": comparison["first_difference"]}), flush=True)


def triage(root: Path, stage, until):
    """Triage (i): rerun `stage` onward in a copy of arm B (B'; arm B stays as evidence) and compare it with B.
    B' differing from B means the stage is nondeterministic; B' equal to B means it depends on the bulk labels."""
    rerun = root / "arm_B_rerun"
    if not rerun.exists() and not (root / "R").exists():
        shutil.copytree(root / "arm_B", rerun, symlinks=True)
        (rerun / "arm.txt").write_text("B_rerun")
    run_arm(root, "B_rerun", None, stage, until)
    report = compare(root / "arm_B" / "smoke", rerun / "smoke", (None, None))
    (root / f"triage_{stage}.json").write_text(json.dumps(report, indent=1, default=str))
    verdict = (f"B' differs from B at {stage}: nondeterminism (the determinism check takes it over)"
               if report["first_difference"] == stage else
               f"B' equals B at {stage}: its B/C difference depends on the bulk labels, a firewall breach (stop; report to Arushi)")
    print(verdict, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--from", dest="start", choices=STAGE_ORDER)
    parser.add_argument("--until", choices=STAGE_ORDER)
    parser.add_argument("--triage", choices=STAGE_ORDER)
    parser.add_argument("--root", type=Path, help="the arms (R, arm_B, arm_C) and the record; default "
                        "$SC2BULK_SERVER_ROOT/permutation_test")
    args = parser.parse_args()
    args.root = args.root or Path(os.environ["SC2BULK_SERVER_ROOT"]) / "permutation_test"
    if args.triage:
        triage(args.root, args.triage, args.until)
    elif args.run:
        run(args.root, args.start, args.until)
    else:
        parser.error("give --run or --triage")


# --- Static and unit checks ---

def test_stage_table_matches_firewall():
    """Only run_all entries flagged as reading bulk labels are out of scope, and they are exactly the firewall's
    whitelisted pipeline scripts (other than stage 02, the writer)."""
    flagged = {script for _, script, _, _, labels, _ in RUN_ALL["STAGES"] if labels}
    allowed = {name.removeprefix("pipeline/") for name in FIREWALL_ALLOWED if name.startswith("pipeline/")}
    assert flagged == allowed - {"02_preprocess.py"}
    assert {stage for stage, *_ in RUN_ALL["STAGES"]} - set(STAGE_ORDER) == \
        set(INPUT_STAGES) | {stage for stage, _, _, _, labels, _ in RUN_ALL["STAGES"] if labels}


def test_only_allowed_files_name_the_blinding_switch():
    files = [p for directory in ("src", "pipeline", "scripts") for p in (REPO / directory).rglob("*.py")]
    offenders = [str(p.relative_to(REPO)) for p in files if SWITCH.search(p.read_text())
                 and str(p.relative_to(REPO)) not in FIREWALL_ALLOWED | NAMERS | POST_HOC]
    assert not offenders, f"files that may not branch on the blinding switch name it: {offenders}"


def test_every_pipeline_script_is_placed():
    staged = {script for _, script, *_ in RUN_ALL["STAGES"]}
    lists = (staged, ADDED_IN_SCOPE, set(EXCLUDED))
    assert not (staged & ADDED_IN_SCOPE or staged & set(EXCLUDED) or ADDED_IN_SCOPE & set(EXCLUDED))
    scripts = {p.name for p in PIPELINE.glob("*.py")}
    assert scripts <= set.union(*lists), f"pipeline scripts in neither run_all nor a list: {scripts - set.union(*lists)}"
    assert ADDED_IN_SCOPE | set(EXCLUDED) <= scripts
    assert ENTRY_05S[1] in ADDED_IN_SCOPE


def test_no_script_writes_an_output():
    writers = [p.name for p in (REPO / "scripts").rglob("*.py") if re.search(r"manifest\.finish|import finish", p.read_text())
               and str(p.relative_to(REPO)) not in POST_HOC]
    assert not writers, f"scripts that write pipeline outputs: {writers}"


def test_synthetic_labels():
    for pair in PAIR_ORDER:
        names = [f"s{i}" for i in range(12)]
        units = pd.Series([f"d{i // 2}" for i in range(12)] if PAIRS[pair]["scoring_unit"] == "donor" else names,
                          index=pd.Index(names, name="sample_id"))
        s, s_prime = synthetic_labels(pair, units, "B"), synthetic_labels(pair, units, "C")
        for labels in (s, s_prime):
            assert list(labels.columns) == ["sample_id", "label_value", "group", *PAIRS[pair]["bulk_label_columns"]]
        properties = label_properties(pair, s, s_prime, units)
        assert 0.5 < properties["changed_sample_fraction"] < 1
        assert s.equals(synthetic_labels(pair, units, "B")), "the draw is fixed"
        if PAIRS[pair]["label_value"] == "age":
            assert (s["label_value"] != s_prime["label_value"]).all()
            assert s.loc[s["group"] == 0, "label_value"].between(20, 40).all()
            assert s.loc[s["group"] == 1, "label_value"].between(60, 80).all()
        else:
            assert (s["label_value"] == s["group"]).all()
        for column in PAIRS[pair]["bulk_label_columns"]:
            assert (s[column] != s_prime[column]).all()


CHECKPOINT = "06_train/hammond/crossfit_0/A/pseudobulk/checkpoints/s0.pt"


def plant(smoke: Path, arm):
    """A small arm tree: a plain file, an h5ad, a parquet, and one file of each wall-clock kind, whose
    NORMALIZED fields differ between arms (a, b), each with a manifest; a downstream manifest records the
    checkpoint's hash. Manifests differ in cost, blinded and git.commit."""
    seconds, blinded, commit = (1.0, None, "c1") if arm == "a" else (2.5, ARMS["C"], "c2")
    record = {"seconds": seconds, "peak_gpu_bytes": int(seconds * 100), "steps": 3, "did_train": True}
    files = {
        "03_splits/hammond/splits.csv": lambda p: p.write_text("sample_id,split\ns1,train\n"),
        "04_panel/hammond/crossfit_0/cells.h5ad": lambda p: anndata.AnnData(np.full((2, 3), 0.1, dtype=np.float32)).write_h5ad(p),
        "08_readout/hammond/crossfit_0/pseudobulk_predictions.parquet":
            lambda p: pq.write_table(pa.Table.from_pandas(pd.DataFrame({"sample_id": ["s1", "s2"], "score": [0.1, 0.2]})), p),
        "05_search/hammond/crossfit_0/set/candidates_s0.parquet":
            lambda p: pq.write_table(pa.Table.from_pandas(pd.DataFrame({"name": ["c0"], "loss": [0.3], "seconds": [seconds],
                                                                           "peak_gpu_bytes": [int(seconds * 100)]})), p),
        "06_train/hammond/crossfit_0/A/pseudobulk/record_s0.json": lambda p: p.write_text(json.dumps(record)),
        CHECKPOINT: lambda p: torch.save({"state": {"w": torch.full((2,), 0.1)}, "config": {"lr": 0.01}, "record": record}, p),
        "10_attribution/hammond/crossfit_0/set.csv": lambda p: p.write_text("gene,score\nA,1\n"),
    }
    for name, write in files.items():
        (smoke / name).parent.mkdir(parents=True, exist_ok=True)
        write(smoke / name)
    for name in files:
        stage_settings = {"stage": name.split("/")[0], "code": "abc"}
        if name.startswith("10_attribution"):
            stage_settings["inputs"] = {CHECKPOINT: manifest.content_hash(smoke / CHECKPOINT)}
        write_manifest(smoke / name, stage_settings, blinded, commit, seconds)


def write_manifest(path: Path, stage_settings, blinded, commit, seconds):
    record = {"settings_hash": manifest.settings_hash(stage_settings), "settings": stage_settings,
              "content_hash": manifest.content_hash(path), "size": path.stat().st_size, "seed": 0, "inputs": ["03_splits/hammond/splits.csv"],
              "git": {"commit": commit, "dirty": False}, "tag": None, "host": "h",
              "environment": {"name": "sc2bulk", "python": "3.10", "fingerprint": "f", "conda": True},
              "n_threads": 16, "bookkeeping": "k", "blinded": blinded, "cost": {"seconds": seconds, "gpu": None},
              "torch": "2.8.0", "deterministic_algorithms": True}
    manifest.manifest_path(path).write_text(json.dumps(record, indent=1, sort_keys=True))


def planted_pair(directory: Path):
    a, b = directory / "a", directory / "b"
    plant(a, "a")
    plant(b, "b")
    return a, b


def test_comparator_passes_wall_clock_and_bookkeeping_differences(tmp_path):
    a, b = planted_pair(tmp_path)
    report = compare(a, b)
    assert report["passed"], report
    assert {f["file"] for f in report["normalized_values"]} == {
        "05_search/hammond/crossfit_0/set/candidates_s0.parquet", "06_train/hammond/crossfit_0/A/pseudobulk/record_s0.json", CHECKPOINT}
    assert report["normalized_values"][0]["a"]["seconds"] != report["normalized_values"][0]["b"]["seconds"]
    assert report["commits"] == ["c1", "c2"]
    assert report["stages"]["03_splits"]["identical"] == 1 and report["stages"]["10_attribution"]["normalized"] == 1
    # the seed files are not compared
    for smoke in (a, b):
        (smoke / "02_preprocess/hammond").mkdir(parents=True)
        (smoke / "02_preprocess/hammond/bulk_labels.csv").write_text(f"{smoke.name}\n")
    assert compare(a, b)["passed"]


def change_float(path: Path):
    """Rewrite a file with its first float moved by one ulp (h5ad matrix, parquet cell, checkpoint tensor)."""
    if path.suffix == ".h5ad":
        data = anndata.read_h5ad(path)
        data.X[0, 0] = np.nextafter(data.X[0, 0], np.float32(1))
        data.write_h5ad(path)
    elif path.suffix == ".parquet":
        frame = pq.read_table(path).to_pandas()
        column = [c for c in frame if frame[c].dtype.kind == "f" and c not in NORMALIZED][0]
        frame.loc[0, column] = np.nextafter(frame.loc[0, column], 1.0)
        pq.write_table(pa.Table.from_pandas(frame), path)
    else:
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["state"]["w"][0] = torch.nextafter(checkpoint["state"]["w"][0], torch.tensor(1.0))
        torch.save(checkpoint, path)


def test_comparator_fails_on_any_other_difference(tmp_path):
    def manifest_seed(b):
        path = b / "03_splits/hammond/splits.csv.manifest.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "seed": 1}))

    def record_key(b):
        path = b / "06_train/hammond/crossfit_0/A/pseudobulk/record_s0.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), "steps": 4}))

    changes = {   # label -> (the stage that must differ, the change made in arm b)
        "tensor": ("06_train", lambda b: change_float(b / CHECKPOINT)),
        "h5ad": ("04_panel", lambda b: change_float(b / "04_panel/hammond/crossfit_0/cells.h5ad")),
        "parquet": ("08_readout", lambda b: change_float(b / "08_readout/hammond/crossfit_0/pseudobulk_predictions.parquet")),
        "wall-clock parquet": ("05_search", lambda b: change_float(b / "05_search/hammond/crossfit_0/set/candidates_s0.parquet")),
        "bytes": ("03_splits", lambda b: (b / "03_splits/hammond/splits.csv").write_text("sample_id,split\ns1,test\n")),
        "one arm only": ("03_splits", lambda b: (b / "03_splits/hammond/extra.csv").write_text("x\n")),
        "manifest field": ("03_splits", manifest_seed),
        "record key": ("06_train", record_key),
    }
    for label, (stage, change) in changes.items():
        a, b = planted_pair(tmp_path / label.replace(" ", "_"))
        change(b)
        report = compare(a, b)
        assert not report["passed"] and report["first_difference"] == stage, (label, report["first_difference"])


def test_comparator_asserts_each_arms_manifests(tmp_path):
    a, b = planted_pair(tmp_path / "hash")
    path = b / "03_splits/hammond/splits.csv.manifest.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "settings": {"stage": "03_splits", "code": "other"}}))
    with pytest.raises(AssertionError, match="settings_hash"):
        compare(a, b)
    a, b = planted_pair(tmp_path / "blinded")
    with pytest.raises(AssertionError, match="blinded"):
        compare(a, b, blinded=(None, None))
    a, b = planted_pair(tmp_path / "labelled")
    (b / "11_statistics/hammond").mkdir(parents=True)
    (b / "11_statistics/hammond" / f"{LABELLED_PREFIX}scores.parquet").write_text("x")
    with pytest.raises(AssertionError, match="label-bearing"):
        compare(a, b)


def test_wall_clock_files():
    assert wall_clock_kind("05_search/tms/crossfit_0/set/candidates_s3.parquet") == "parquet"
    assert wall_clock_kind("05_search/tms/crossfit_0/set/candidates_lr_s1.parquet") == "parquet"   # the merge's rows
    assert wall_clock_kind("05_search/tms/crossfit_0/set/checkpoints/c1_s0.pt") == "checkpoint"
    assert wall_clock_kind("06_train/tms/crossfit_0/A/set/record_search_s0.json") == "json"
    assert wall_clock_kind("12_simulation/tms/crossfit_0/s1/set/record_s0.json") == "json"
    assert wall_clock_kind("06_train/tms/crossfit_0/J/scvi/checkpoints/s0/model.pt") is None   # scvi-tools' own file
    assert wall_clock_kind("08_readout/tms/crossfit_0/record_s0.json") is None
    assert placeholder({"candidates": {"05_search/t/u/candidates_s0.parquet": "h"}, "inputs": {"03_splits/t/s.csv": "h"}}) == \
        {"candidates": {"05_search/t/u/candidates_s0.parquet": "<wall-clock file>"}, "inputs": {"03_splits/t/s.csv": "h"}}


@pytest.mark.skipif(os.environ.get("SC2BULK_RUN_LEVEL") != "1", reason="run-level check")
def test_record_matches_current_code():
    """The go/no-go before the first real bulk stage, on the machine that runs it
    (SC2BULK_RUN_LEVEL=1, SC2BULK_OUTPUTS=<real outputs>): the record exists and passed, and its code hash for
    every in-scope stage and pair equals the current tree's (a --until prefix record lacks the later stages)."""
    assert RECORD.exists(), f"no run-level record at {RECORD} (python tests/test_permutation.py --run)"
    record = json.loads(RECORD.read_text())
    assert record["passed"], f"the run-level test failed at {record['comparison']['first_difference']}"
    stale = sorted(key for key, code in code_hashes().items() if record["code"].get(key) != code)
    assert not stale, f"in-scope code changed since the record, or never covered (rerun the driver): {stale}"


if __name__ == "__main__":
    main()
