"""The vendored Set Transformer modules are the original, unchanged (checksums at the pinned commit)."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "third_party" / "set_transformer"


def test_files_match_upstream():
    for line in (ROOT / "UPSTREAM_SHA256.txt").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
