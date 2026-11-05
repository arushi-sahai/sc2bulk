"""Optional step before stage 01: rebuild the SoundLife slim files from the AIFI portal files.

The pipeline's SoundLife single-cell input is one slim file per portal file: a uniform
random 3,000 cells per sample (all cells of a sample with fewer), drawn before QC, raw
counts. They were made once from the portal files; the kept barcodes of each are shipped in
src/sc2bulk/data/slim_barcodes/. This step selects exactly those barcodes from the portal
files and checks the result's content checksum against the pinned one, so anyone with the
portal files can verify the slims. (Selecting by barcode, rather than redrawing, keeps the
result independent of the random-number library's version.) This rebuild has not been run
end to end since the portal files were deleted; the content checksum is what certifies it.

How the barcodes were first drawn: per portal file, numpy.random.default_rng(42 +
int(md5(<file stem>)[:8], 16)); per sample in order of first appearance, 3,000 cells chosen
without replacement where the sample had more, all cells otherwise; rows kept in file order.

  python pipeline/00_soundlife_slims.py <directory with the 8 portal files>

Writes the slims into <outputs>/01_download/soundlife/.
"""
import argparse
import os
from pathlib import Path

from sc2bulk.config import sources
from sc2bulk.config.settings import RAW
from sc2bulk.data import soundlife


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("portal_directory", type=Path)
    args = parser.parse_args()
    out = RAW / "soundlife"
    out.mkdir(parents=True, exist_ok=True)
    for portal in sources.SOUNDLIFE_SINGLE_CELL:
        slim = soundlife.build_slim(args.portal_directory / portal, soundlife.slim_barcodes(portal))
        checksum = soundlife.content_checksum(slim)
        expected = sources.SOUNDLIFE_SLIM_CONTENT[soundlife.slim_name(portal)]
        assert checksum == expected, f"{portal}: rebuilt slim content {checksum} does not match the pinned {expected}"
        path = out / soundlife.slim_name(portal)
        partial = path.with_name(".tmp." + path.name)
        slim.write_h5ad(partial)
        os.replace(partial, path)   # replaces a symlink rather than writing through it
        print(f"{soundlife.slim_name(portal)}: {slim.n_obs} cells, content verified", flush=True)


if __name__ == "__main__":
    main()
