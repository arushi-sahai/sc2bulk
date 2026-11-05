"""Stage 01: download every raw input file and verify it against its pinned checksum
(sc2bulk.config.sources): GEO and CELLxGENE over HTTPS, Pineda's single-nucleus data from
Synapse (needs a Synapse personal access token in ~/.synapseConfig). Files that can't be
fetched by script must already be in place; if one is missing, this stops with the
instructions for getting it.

  python pipeline/01_download.py [--pair PAIR]
"""
import argparse

import synapseclient

from sc2bulk import determinism
from sc2bulk.config import sources
from sc2bulk.config.pairs import PAIR_ORDER
from sc2bulk.config.settings import RAW
from sc2bulk.data.fetch import fetch, fetch_synapse, sha256, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER)
    args = parser.parse_args()
    determinism.configure()
    directories = {d for pair in ([args.pair] if args.pair else PAIR_ORDER)
                   for d in sources.PAIR_DIRECTORIES[pair] + sources.ORTHOLOG_DIRECTORIES[pair]}

    for (directory, name), (url, expected) in sources.DOWNLOADS.items():
        if directory in directories:
            print(f"{directory}/{name}", flush=True)
            fetch(url, RAW / directory / name, expected)

    synapse_files = {key: spec for key, spec in sources.SYNAPSE.items() if key[0] in directories}
    if any(not (RAW / d / n).exists() for d, n in synapse_files):   # log in only to download
        synapse = synapseclient.Synapse(cache_root_dir=str(RAW.parent / "synapse_cache"), silent=True)
        synapse.login()   # reads the personal access token from ~/.synapseConfig
        for (directory, name), (synapse_id, version, expected) in synapse_files.items():
            if not (RAW / directory / name).exists():
                print(f"{directory}/{name} (Synapse {synapse_id}.{version})", flush=True)
                fetch_synapse(synapse, synapse_id, version, RAW / directory / name, expected)
    for (directory, name), (_, _, expected) in synapse_files.items():
        verify(RAW / directory / name, expected)

    manual = {key: spec for key, spec in sources.MANUAL.items() if key[0] in directories}
    missing = [f"{RAW / d / n}: {instructions}" for (d, n), (instructions, _) in manual.items() if not (RAW / d / n).exists()]
    assert not missing, "missing manual downloads:\n" + "\n".join(missing)
    unpinned = [f"{d}/{n}: {sha256(RAW / d / n)}" for (d, n), (_, expected) in manual.items() if expected is None]
    assert not unpinned, "no pinned checksum yet (add these to sources.py):\n" + "\n".join(unpinned)
    for (directory, name), (_, expected) in manual.items():
        print(f"{directory}/{name} (manual)", flush=True)
        verify(RAW / directory / name, expected)


if __name__ == "__main__":
    main()
