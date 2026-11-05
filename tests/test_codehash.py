"""The scoped code hash (sc2bulk.codehash): a stage's hash
covers what it imports and nothing else, and every dynamic import is a listed exception."""
import ast
import shutil

import pytest

from sc2bulk import codehash
from sc2bulk.config.settings import REPO

from test_firewall import DYNAMIC_IMPORTS, code_files, imported_modules

PIPELINE = REPO / "pipeline"


def relative(paths, root=REPO):
    return {str(p.relative_to(root)) for p in paths}


def test_walk_finds_dependencies():
    found = relative(codehash.files(PIPELINE / "04_panel.py", "tms"))
    for name in ("pipeline/04_panel.py", "src/sc2bulk/folds.py", "src/sc2bulk/preprocess.py",
                 "src/sc2bulk/orthologs.py", "src/sc2bulk/determinism.py",
                 "src/sc2bulk/__init__.py", "src/sc2bulk/config/__init__.py", "src/sc2bulk/config/settings.py",
                 "src/sc2bulk/config/pairs/__init__.py", "src/sc2bulk/config/pairs/tms.py"):
        assert name in found, name
    # only the pair being run, and nothing the stage doesn't import
    assert "src/sc2bulk/config/pairs/hammond.py" not in found
    assert "src/sc2bulk/labels.py" not in found and "src/sc2bulk/config/style.py" not in found
    assert "src/sc2bulk/config/fig1e.py" not in found and "src/sc2bulk/config/readout.py" not in found
    assert "src/sc2bulk/manifest.py" not in found and "src/sc2bulk/codehash.py" not in found   # bookkeeping


def test_dynamic_imports_and_data_files():
    reader = REPO / "src/sc2bulk/data/soundlife.py"
    found = relative(codehash.files(PIPELINE / "02_preprocess.py", "soundlife", [reader]))
    assert "src/sc2bulk/data/soundlife.py" in found
    assert any(name.startswith("src/sc2bulk/data/slim_barcodes/") for name in found)
    assert "src/sc2bulk/data/hammond.py" not in found


def test_relative_and_from_imports():
    # 'from sc2bulk import a' yields the submodule; relative imports resolve within the package
    names = codehash.imported(REPO / "src/sc2bulk/folds.py")
    assert any(name.startswith("sc2bulk.") for name in names)
    assert codehash.module_file("sc2bulk.config.settings") == REPO / "src/sc2bulk/config/settings.py"
    assert codehash.module_file("cytoset.model") == REPO / "third_party/cytoset/model.py"
    assert codehash.module_file("numpy") is None


@pytest.fixture
def repo_copy(tmp_path, monkeypatch):
    """A copy of the code, so editing it never touches the tree a stage might be running on."""
    for directory in ("src", "pipeline", "third_party"):
        shutil.copytree(REPO / directory, tmp_path / directory, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(codehash, "REPO", tmp_path)
    monkeypatch.setattr(codehash, "SRC", tmp_path / "src")
    monkeypatch.setattr(codehash, "VENDORED", tmp_path / "third_party")
    monkeypatch.setattr(codehash, "BOOKKEEPING", {tmp_path / "src/sc2bulk" / name for name in ("manifest.py", "codehash.py")})
    return tmp_path


def test_unrelated_module_leaves_hash_unchanged(repo_copy):
    before = codehash.stage_hash(repo_copy / "pipeline/04_panel.py", "tms")
    (repo_copy / "src/sc2bulk/unrelated.py").write_text("X = 1\n")
    assert codehash.stage_hash(repo_copy / "pipeline/04_panel.py", "tms") == before


def test_edit_to_imported_module_changes_hash(repo_copy):
    stages = ("02_preprocess.py", "04_panel.py")
    before = {stage: codehash.stage_hash(repo_copy / "pipeline" / stage, "tms") for stage in stages}
    with open(repo_copy / "src/sc2bulk/preprocess.py", "a") as f:
        f.write("\nEDITED = 1\n")
    for stage in stages:
        assert codehash.stage_hash(repo_copy / "pipeline" / stage, "tms") != before[stage], stage


def test_bookkeeping_and_other_pairs_leave_hash_unchanged(repo_copy):
    before = codehash.stage_hash(repo_copy / "pipeline/04_panel.py", "tms")
    for name in ("manifest.py", "codehash.py", "config/pairs/hammond.py"):
        with open(repo_copy / "src/sc2bulk" / name, "a") as f:
            f.write("\nEDITED = 1\n")
    assert codehash.stage_hash(repo_copy / "pipeline/04_panel.py", "tms") == before


def test_no_pair_means_every_pair(repo_copy):
    found = relative(codehash.files(repo_copy / "pipeline/04_panel.py"), repo_copy)
    assert {f"src/sc2bulk/config/pairs/{p}.py" for p in ("hammond", "tms", "soundlife", "lupus", "crohns", "pineda")} <= found


def test_no_unlisted_dynamic_imports():
    # the walk can't see importlib/__import__; each use must be listed and handled
    offenders = [str(p.relative_to(REPO)) for p in code_files()
                 if imported_modules(ast.parse(p.read_text()))[1]
                 and str(p.relative_to(REPO)) not in DYNAMIC_IMPORTS
                 and not str(p.relative_to(REPO)).startswith("tests/")]
    assert not offenders, offenders


def test_comments_docstrings_and_assert_messages_leave_hash_unchanged(repo_copy):
    # what the public export strips can't make an output look stale
    module = repo_copy / "src/sc2bulk/preprocess.py"
    with open(module, "a") as f:
        f.write('\n\nclass Edited:\n    """A docstring."""\n    def f(self, x):\n        """Another."""\n'
                '        assert x, "a message"   # a comment\n        return x\n')
    before = codehash.stage_hash(repo_copy / "pipeline/04_panel.py", "tms")
    text = module.read_text()
    assert text.startswith('"""')
    for old, new in (('"""', '"""Reworded. '), ('"""A docstring."""', '"""Reworded."""'), ('"""Another."""', '"""Reworded too."""'),
                     ('assert x, "a message"', 'assert x, "reworded"'), ("# a comment", "# reworded"), ("import", "# a new comment\nimport")):
        text = text.replace(old, new, 1)
        module.write_text(text)
        assert codehash.stage_hash(repo_copy / "pipeline/04_panel.py", "tms") == before, new


def test_code_and_string_literal_edits_change_hash(repo_copy):
    module = repo_copy / "src/sc2bulk/preprocess.py"
    with open(module, "a") as f:
        f.write('\n\ndef edited(x):\n    """A docstring."""\n    assert x, "a message"\n    return "a literal"\n')
    before = codehash.stage_hash(repo_copy / "pipeline/04_panel.py", "tms")
    text = module.read_text()
    for old, new in (('return "a literal"', 'return "another literal"'), ("assert x,", "assert not x,"),
                     ('"""A docstring."""', '"""A docstring."""\n    x = 1')):
        text = text.replace(old, new)
        module.write_text(text)
        after = codehash.stage_hash(repo_copy / "pipeline/04_panel.py", "tms")
        assert after != before, new
        before = after


def test_data_files_hash_their_bytes(repo_copy):
    # non-Python files (tables, barcode lists, signatures) are hashed byte for byte
    script, extra = repo_copy / "pipeline/02_preprocess.py", codehash.extra("02_preprocess", "pineda")
    before = codehash.stage_hash(script, "pineda", extra)
    with open(repo_copy / "src/sc2bulk/config/pineda_synapse_files.tsv", "a") as f:
        f.write("# edited\n")
    assert codehash.stage_hash(script, "pineda", extra) != before
