"""Which source files determine a stage's outputs: the stage script, every sc2bulk module it imports (transitively,
found by walking the import statements; imports inside functions included), their packages'
__init__ files, the vendored modules they import (third_party/cytoset), the package data those
modules read, and the configuration of the pair being run (config/pairs/<pair>.py), not the
other pairs' (all of them when no pair is given). A new module nobody imports, or an edit to a
module a stage doesn't import, leaves that stage's hash unchanged; an edit to anything it imports
changes it.

The bookkeeping modules (BOOKKEEPING: manifest records and this file) never change an output's
bytes, so they are left out; each record carries their hash as an observation instead
(bookkeeping_hash).

Dynamic imports are invisible to the walk, so they are listed: stage 02 loads data/<pair>.py by
name (extra(), passed as `extra`), and config/pairs loads its pair modules by name (added for the pair).
tests/test_codehash.py fails on any other importlib/__import__ use.

What is hashed: each file's path within the repo and, for a .py file, its syntax tree without docstrings
(module, class and function) or assert messages (ast.dump; comments are not in the tree), so an edit to
those leaves the hash unchanged; for any other file (tables, barcode lists, signatures), its bytes. The
dump is stable within a Python minor version, and every environment runs Python 3.10.
"""
import ast
import sys
import hashlib
from pathlib import Path

from sc2bulk.config.pairs import PAIR_ORDER
from sc2bulk.config.settings import REPO

SRC = REPO / "src"
VENDORED = REPO / "third_party"
# modules that read package data files at run time -> those files
DATA_FILES = {
    "src/sc2bulk/data/soundlife.py": "src/sc2bulk/data/slim_barcodes/*.txt.gz",
    "src/sc2bulk/config/sources.py": "src/sc2bulk/config/pineda_synapse_files.tsv",
    "src/sc2bulk/signatures.py": "src/sc2bulk/config/signatures/*",
    "pipeline/10_attribution.py": "src/sc2bulk/config/attribution/*",   # the HRT Atlas housekeeping list (config/genes.py)
}
PAIR_MODULES = "sc2bulk.config.pairs."
BOOKKEEPING = {SRC / "sc2bulk" / "manifest.py", SRC / "sc2bulk" / "codehash.py"}


def module_file(name: str):
    """The file of a module name inside sc2bulk or the vendored packages, or None (stdlib,
    third-party)."""
    parts = name.split(".")
    root = SRC if parts[0] == "sc2bulk" else VENDORED if (VENDORED / parts[0]).is_dir() else None
    if root is None:
        return None
    path = root.joinpath(*parts)
    if (path / "__init__.py").exists():
        return path / "__init__.py"
    if path.with_suffix(".py").exists():
        return path.with_suffix(".py")
    return None


def module_name(path: Path) -> str:
    root = SRC if SRC in path.parents else VENDORED
    parts = path.relative_to(root).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imported(path: Path) -> set:
    """Module names imported by a file (absolute and relative; 'from pkg import a' also yields
    pkg.a, kept if it is a module)."""
    names = set()
    package = module_name(path) if (SRC in path.parents or VENDORED in path.parents) else ""
    if path.name != "__init__.py":
        package = package.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:   # relative: level 1 = the file's package, 2 = its parent, ...
                parts = package.split(".")
                base = parts[:len(parts) - (node.level - 1)]
                module = ".".join([*base, *(node.module.split(".") if node.module else [])])
            else:
                module = node.module
            names.add(module)
            names |= {f"{module}.{alias.name}" for alias in node.names}
    return names


def files(script: Path, pair: str | None = None, extra=()) -> set:
    """Every source file a stage depends on (see the module docstring)."""
    found, queue = set(), [Path(script).resolve(), *[Path(e).resolve() for e in extra]]
    queue += [module_file(PAIR_MODULES + p) for p in ([pair] if pair is not None else PAIR_ORDER)]
    while queue:
        path = queue.pop()
        if path in found or path in BOOKKEEPING:
            continue
        found.add(path)
        relative = str(path.relative_to(REPO))
        if relative in DATA_FILES:
            found |= {p.resolve() for p in REPO.glob(DATA_FILES[relative])}
        for name in imported(path):
            if name.startswith(PAIR_MODULES):   # only the pair being run (added above)
                continue
            target = module_file(name)
            if target is not None:
                queue.append(target)
                # the packages' __init__ files run on import too
                parts = name.split(".")
                for i in range(1, len(parts)):
                    parent = module_file(".".join(parts[:i]))
                    if parent is not None:
                        queue.append(parent)
    return found


def code(path: Path) -> bytes:
    """What is hashed of a file: a .py file's syntax tree without docstrings or assert messages,
    any other file's bytes (see the module docstring)."""
    if path.suffix != ".py":
        return path.read_bytes()
    assert sys.version_info[:2] == (3, 10), "ast.dump differs across Python versions: hash under 3.10 only"
    tree = ast.parse(path.read_bytes())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            node.msg = None
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False).encode()


def digest(paths) -> str:
    """Hash of files (each file's path within the repo and what is hashed of it, length-prefixed)."""
    out = hashlib.sha256()
    for path in sorted(Path(p).resolve() for p in paths):
        for part in (str(path.relative_to(REPO)).encode(), code(path)):
            out.update(len(part).to_bytes(8, "little") + part)
    return out.hexdigest()[:16]


def extra(stage: str, pair: str) -> list:
    """Files a stage loads by name, added to its hash: the pair's data reader (stages 02, 02b); stage
    07's script, whose functions stage 07d runs (runpy); stage 03's, whose split() stage 03b runs; stage
    08's, whose features stages 09 and 12a read; stages 09 and 08's, whose functions stage 09c runs; stage
    06's, whose helpers stage 12 runs."""
    if stage in ("02_preprocess", "02b_all_available"):
        return [SRC / "sc2bulk" / "data" / f"{pair}.py"]
    return {"07d_embed_all_available": [REPO / "pipeline" / "07_embed.py"],
            "03b_attribution_splits": [REPO / "pipeline" / "03_splits.py"],
            "09_alignment": [REPO / "pipeline" / "08_readout.py"],
            "09c_geometry": [REPO / "pipeline" / n for n in ("09_alignment.py", "08_readout.py")],   # 09c runs 09's and 08's (runpy)
            "12_simulation": [REPO / "pipeline" / "06_train.py"],     # 12 runs stage 06's helpers (runpy)
            "12a_properties": [REPO / "pipeline" / "08_readout.py"],   # 12a reads stage 08's features
            "08b_secondary": [REPO / "pipeline" / "08_readout.py"],    # 08b runs stage 08's functions (runpy)
            "11b_secondary": [REPO / "pipeline" / "11_statistics.py"],   # 11b runs stage 11's functions
            "10_attribution": [REPO / "pipeline" / n for n in ("06_train.py", "07_embed.py", "08_readout.py")]}.get(stage, [])   # 10 runs 06/07/08's (runpy)


def stage_hash(script, pair: str | None = None, extra=()) -> str:
    return digest(files(script, pair, extra))


def bookkeeping_hash() -> str:
    return digest(BOOKKEEPING)
