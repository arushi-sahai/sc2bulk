"""One-time migration of the manifests' code hashes to the comment-insensitive hash (sc2bulk.codehash
now hashes a .py file's syntax tree without docstrings or assert messages; it used to hash its bytes).

For every manifest under the outputs root (SC2BULK_OUTPUTS, the smoke and blinded subtrees included), the
record's stage and pair come from its path (as figures/common.py's stale check reads them). A record is
restamped only if its code hash (settings["code"]) equals the old byte hash of the current files of that
stage and pair, and its settings hash is reproducible from its settings: then settings["code"] becomes the
new hash, settings_hash is recomputed, and the old hash is kept in the record (restamp). Records whose code
hash differs are already stale and are left untouched and listed; records without a code hash, of no
pipeline script, or already restamped are skipped.

Run it on the tree the outputs were made from plus only the commit that added this script: any later edit
to a hashed file's bytes, even a comment, makes the old hash differ and every record of that stage look
stale. Each rewritten manifest's original is kept beside it as <name>.pre_restamp; nothing is deleted.

  SC2BULK_OUTPUTS=/path/to/outputs python scripts/restamp_hashes.py           # dry run: counts only
  SC2BULK_OUTPUTS=/path/to/outputs python scripts/restamp_hashes.py --write   # rewrite the manifests
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path

from sc2bulk import codehash, manifest
from sc2bulk.config.pairs import PAIR_ORDER


def old_digest(paths) -> str:
    """The old code hash: each file's path within the repo and its bytes, length-prefixed."""
    out = hashlib.sha256()
    for path in sorted(Path(p).resolve() for p in paths):
        for part in (str(path.relative_to(codehash.REPO)).encode(), path.read_bytes()):
            out.update(len(part).to_bytes(8, "little") + part)
    return out.hexdigest()[:16]


@functools.cache
def hashes(stage, pair) -> tuple:
    """(old, new) code hash of the current files of a stage and pair."""
    script = codehash.REPO / "pipeline" / f"{stage}.py"
    files = codehash.files(script, pair, codehash.extra(stage, pair))
    return old_digest(files), codehash.digest(files)


def stage_and_pair(root, path) -> tuple:
    """A manifest's stage and pair (None for a pooled output), below the smoke and blinded subdirectories."""
    parts = list(path.relative_to(root).parts)
    if parts[0] == "smoke":
        parts = parts[1:]
    if parts[0].startswith("blinded_"):
        parts = parts[1:]
    return parts[0], parts[1] if parts[1] in PAIR_ORDER else None


def restamp(root, write: bool) -> dict:
    """{"restamped", "stale", "skipped": [(manifest path, reason)]} for every manifest under root; with
    write, the restamped ones are rewritten (their originals kept as <name>.pre_restamp)."""
    out = {"restamped": [], "stale": [], "skipped": []}
    for path in sorted(Path(root).rglob("*.manifest.json")):
        if path.name.startswith(".tmp."):   # left by a crashed write
            out["skipped"].append((path, "temporary file"))
            continue
        text = path.read_text()
        record = json.loads(text)
        stage, pair = stage_and_pair(root, path)
        if "code" not in record["settings"]:
            out["skipped"].append((path, "no code hash"))
            continue
        if not (codehash.REPO / "pipeline" / f"{stage}.py").exists():
            out["skipped"].append((path, f"no pipeline script {stage}.py"))
            continue
        old, new = hashes(stage, pair)
        code = record["settings"]["code"]
        if code == new:
            out["skipped"].append((path, "already restamped"))
            continue
        if code != old:
            out["stale"].append((path, f"{stage} {pair}: recorded {code}, current files {old}"))
            continue
        if manifest.settings_hash(record["settings"]) != record["settings_hash"]:
            out["skipped"].append((path, "settings hash not reproducible from the recorded settings"))
            continue
        record["settings"]["code"] = new
        record["settings_hash"] = manifest.settings_hash(record["settings"])
        record["restamp"] = {"code": old}
        out["restamped"].append((path, f"{stage} {pair}: {old} -> {new}"))
        if write:
            backup = path.with_name(path.name + ".pre_restamp")
            assert not backup.exists(), f"{backup} exists"
            backup.write_text(text)
            written = manifest.temporary(path)
            written.write_text(json.dumps(record, indent=1, sort_keys=True, default=str))
            os.replace(written, path)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="count only (the default)")
    mode.add_argument("--write", action="store_true", help="rewrite the manifests")
    args = parser.parse_args()
    root = Path(os.environ["SC2BULK_OUTPUTS"])
    out = restamp(root, args.write)
    for path, why in out["stale"]:
        print(f"already stale: {path.relative_to(root)} ({why})")
    for path, why in out["skipped"]:
        print(f"skipped: {path.relative_to(root)} ({why})")
    print(f"{'restamped' if args.write else 'would restamp'} {len(out['restamped'])}, already stale {len(out['stale'])}, "
          f"skipped {len(out['skipped'])}")


if __name__ == "__main__":
    main()
