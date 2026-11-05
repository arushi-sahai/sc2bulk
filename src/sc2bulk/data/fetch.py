"""Downloading and verifying raw input files."""
import hashlib
import os

import requests


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path, expected: str):
    actual = sha256(path)
    assert actual == expected, (f"{path}: checksum {actual} does not match the pinned {expected}. "
                                "Delete the file to download it again, or check whether the source changed.")


def fetch_synapse(synapse, synapse_id: str, version: int, path, expected: str):
    """Download a pinned version of a Synapse file to path (skipped if path already exists)
    and verify its checksum. `synapse` is a logged-in synapseclient.Synapse (its token is
    read from ~/.synapseConfig, never from code). Synapse saves under the file's own name,
    which must equal path's name."""
    if not path.exists():
        partial = path.parent / ".partial"   # downloaded here, then renamed, so a cut download never has the final name
        partial.mkdir(parents=True, exist_ok=True)
        entity = synapse.get(synapse_id, version=version, downloadLocation=str(partial), ifcollision="overwrite.local")
        assert entity.path == str(partial / path.name), f"{synapse_id}: saved as {entity.path}, expected {path.name}"
        os.replace(entity.path, path)
    verify(path, expected)


def fetch(url: str, path, expected: str):
    """Download url to path (skipped if path already exists) and verify its checksum. The
    download is written to a temporary file and renamed, so an interrupted download never
    leaves a partial file under the final name."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        with requests.get(url, stream=True, timeout=600) as response:
            response.raise_for_status()
            with open(partial, "wb") as f:
                for chunk in response.iter_content(1 << 22):
                    f.write(chunk)
        os.replace(partial, path)
    verify(path, expected)
