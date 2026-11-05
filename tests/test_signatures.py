"""The known-signature lists (sc2bulk.signatures): files match their pinned checksums and the
counts transcribed from the sources (Holtman's file has 209 down, the paper's text 205)."""
import hashlib

import pytest

from sc2bulk import signatures

COUNTS = {"hammond": (295, 209), "tms": (449, 162), "soundlife": (600, 897), "lupus": (21, 0),
          "crohns": (704, 507), "pineda": (17, 40)}


def test_checksums():
    listed = {line.split()[1] for line in (signatures.DIRECTORY / "SHA256SUMS").read_text().splitlines()}
    assert listed == {p.name for p in signatures.DIRECTORY.glob("*.tsv")}
    for line in (signatures.DIRECTORY / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((signatures.DIRECTORY / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("pair", COUNTS)
def test_counts(pair):
    table = signatures.load(pair)
    assert table["gene"].is_unique
    assert ((table["direction"] == "up").sum(), (table["direction"] == "down").sum()) == COUNTS[pair]


def test_aliases():
    assert "SPATS2L" in set(signatures.load("lupus")["gene"])          # Yao's DNAPTP6, renamed
    assert "DNAPTP6" in set(signatures.load("lupus")["printed"])
