"""The known-signature gene lists: one file per pair in
config/signatures/, copied byte for byte from the transcriptions (each file's header names its
source, table and counts), with outdated human symbols renamed by aliases.tsv (built from HGNC),
all pinned by SHA256SUMS. Scoring comes with stage 08.
"""
from pathlib import Path

import pandas as pd

DIRECTORY = Path(__file__).resolve().parent / "config" / "signatures"


def load(pair: str) -> pd.DataFrame:
    """gene (current symbol: outdated printed symbols renamed by aliases.tsv), direction ("up" /
    "down"), source_symbol, printed (the symbol as listed)."""
    table = pd.read_csv(DIRECTORY / f"{pair}.tsv", sep="\t", comment="#", keep_default_na=False)
    aliases = pd.read_csv(DIRECTORY / "aliases.tsv", sep="\t", comment="#", keep_default_na=False)
    rename = dict(aliases.loc[aliases["pair"] == pair, ["printed", "current"]].to_numpy())
    table.insert(len(table.columns), "printed", table["gene"])
    table["gene"] = table["gene"].replace(rename)
    assert table["gene"].is_unique, f"{pair}: two printed symbols map to one current symbol"
    return table
