"""The figure scripts' static rules and provenance:
a script that doesn't read bulk labels doesn't name their reader; a script that doesn't read scored tables names no
label-bearing output; no figure module imports a scoring module; only common.py opens files; and sources.json's checks
on synthetic manifests (real run: every input unblinded; blinded run: this run's index, label-free inputs either; a
clean tree; a current code hash)."""
import ast
import json
import re
import sys

import pytest

from sc2bulk import manifest
from sc2bulk.config import settings

sys.path.insert(0, str(settings.REPO / "figures"))
import common  # noqa: E402
import make_all  # noqa: E402

FIGURES = settings.REPO / "figures"
SCORING = {"scoring", "statistics", "statistics_secondary", "reference", "concordance"}
PREFIX = settings.LABELLED + "_"


def imported(path) -> set:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module == "sc2bulk":
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.Import):
            names |= {a.name.split(".")[-1] for a in node.names if a.name.startswith("sc2bulk.")}
    return names


def test_static_rules():
    for name, module in make_all.SCRIPTS.items():
        path = FIGURES / f"{name}.py"
        text = path.read_text()
        assert not imported(path) & SCORING, name
        if not module.READS_BULK_LABELS:
            assert "bulk_labels(" not in text and "labels" not in imported(path), name
        if not module.READS_SCORES:
            assert not re.search(r"LABELLED|common\.(scores|statistics|secondary|claims)\(", text) and PREFIX not in text, name
    for path in FIGURES.glob("*.py"):
        if path.name not in ("common.py", "citation_order.py"):   # citation_order reads the manuscript files it is given
            assert not re.search(r"\bopen\(|pd\.read_|schemas\.read\(|\.read_text\(|\.read_bytes\(", path.read_text()), path.name
        assert not imported(path) & SCORING, path.name


def record(blinded=None, dirty=False, tag="v1", code="current"):
    return {"blinded": blinded, "git": {"commit": "abc", "dirty": dirty}, "tag": tag, "settings": {"code": code}, "content_hash": "h"}


def tree(tmp_path, monkeypatch, records):
    """Outputs with synthetic manifests: name -> record."""
    monkeypatch.setattr(settings, "OUTPUTS", tmp_path)
    monkeypatch.setattr(settings, "LABELLED_OUTPUTS", tmp_path / "blinded_1" if settings.BLINDED is not None else tmp_path)
    monkeypatch.setattr(common, "current_code", lambda stage, pair: "current")
    paths = []
    for name, made in records.items():
        root = settings.LABELLED_OUTPUTS if name.startswith(PREFIX) else settings.OUTPUTS
        path = root / "08_readout" / "lupus" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
        manifest.manifest_path(path).write_text(json.dumps(made))
        paths.append(path)
    return paths


def test_sources_real_run(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "BLINDED", None)
    out = common.sources(tree(tmp_path, monkeypatch, {"a.parquet": record(), f"{PREFIX}b.parquet": record()}))
    assert out["commits"] == ["abc"] and len(out["inputs"]) == 2
    with pytest.raises(AssertionError, match="permuted"):
        common.sources(tree(tmp_path / "1", monkeypatch, {f"{PREFIX}b.parquet": record(blinded=1)}))
    with pytest.raises(AssertionError, match="dirty"):
        common.sources(tree(tmp_path / "2", monkeypatch, {"a.parquet": record(dirty=True)}))
    with pytest.raises(AssertionError, match="stale"):
        common.sources(tree(tmp_path / "3", monkeypatch, {"a.parquet": record(code="old")}))
    monkeypatch.setattr(settings, "FULL_RUN", True)
    with pytest.raises(AssertionError, match="untagged"):
        common.sources(tree(tmp_path / "4", monkeypatch, {"a.parquet": record(tag=None)}))


def test_sources_blinded_run(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "BLINDED", 1)
    paths = tree(tmp_path, monkeypatch, {"a.parquet": record(), "b.parquet": record(blinded=1), f"{PREFIX}c.parquet": record(blinded=1)})
    assert len(common.sources(paths)["inputs"]) == 3   # label-free inputs: unblinded or this index
    with pytest.raises(AssertionError, match="blinded"):
        common.sources(tree(tmp_path / "1", monkeypatch, {f"{PREFIX}c.parquet": record(blinded=None)}))
    with pytest.raises(AssertionError, match="blinded"):
        common.sources(tree(tmp_path / "2", monkeypatch, {"a.parquet": record(blinded=2)}))


def test_sources_records_repository_files_by_blob(tmp_path, monkeypatch):
    """A repository input (figures/signature_labels.tsv) has no manifest: recorded by its git blob hash."""
    monkeypatch.setattr(settings, "BLINDED", None)
    path = settings.REPO / "figures" / "signature_labels.tsv"
    out = common.sources(tree(tmp_path, monkeypatch, {"a.parquet": record()}), [path])
    assert out["repository"] == [{"path": "figures/signature_labels.tsv", "blob": common.blob_hash(path)}]
