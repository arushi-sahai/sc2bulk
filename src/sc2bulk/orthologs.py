"""Mouse -> human genes for the foundation models (they read human genes only).

One-to-one orthologs from Ensembl Compara (sources.ENSEMBL_RELEASE), with no confidence
filter (the usual choice for foundation-model input; the low-confidence count is reported).
A mouse gene is mapped by its Ensembl id(s) from the pair's annotation (stage 02's var): it
is "mapped" if its ids reach exactly one human gene, "no one2one ortholog" if none does
(only one-to-many or many-to-many orthologs, or an id Compara doesn't cover, such as a
non-coding or retired gene), "conflicting ids" if its ids reach different human genes, and
"no Ensembl id" if the pair's annotation has none for it (unreadable by the FMs; counted).
A gene with several ids, some mapped and some not, is mapped. In orthologs.csv,
high_confidence is True/False for mapped genes and empty otherwise.
"""
import gzip
import re

import pandas as pd

from sc2bulk.config import sources

STATUSES = ("mapped", "no one2one ortholog", "conflicting ids", "no Ensembl id")


def one2one(raw) -> pd.DataFrame:
    """mouse_ensembl, human_ensembl, high_confidence: every mouse-human one-to-one pair,
    from both species' files (each pair is in only one of them)."""
    columns = ["gene_stable_id", "species", "homology_type", "homology_gene_stable_id",
               "homology_species", "is_high_confidence"]
    tables = []
    for species, other in (("mouse", "homo_sapiens"), ("human", "mus_musculus")):
        table = pd.read_csv(raw / "ensembl" / sources.ORTHOLOG_FILES[species], sep="\t", usecols=columns,
                            dtype=str, keep_default_na=False)
        table = table[(table["homology_type"] == "ortholog_one2one") & (table["homology_species"] == other)]
        mouse, human = ("gene_stable_id", "homology_gene_stable_id") if species == "mouse" \
            else ("homology_gene_stable_id", "gene_stable_id")
        tables.append(pd.DataFrame({"mouse_ensembl": table[mouse].to_numpy(), "human_ensembl": table[human].to_numpy(),
                                    "high_confidence": table["is_high_confidence"].eq("1").to_numpy()}))
    pairs = pd.concat(tables, ignore_index=True)
    assert pairs["mouse_ensembl"].is_unique and pairs["human_ensembl"].is_unique, "a one-to-one id appears twice"
    return pairs


def human_names(raw) -> pd.Series:
    """Human Ensembl id -> gene name, from the release's GTF (genes without a name omitted)."""
    names = {}
    with gzip.open(raw / "ensembl" / sources.HUMAN_GTF, "rt") as f:
        for line in f:
            if "\tgene\t" in line:
                name = re.search(r'gene_name "([^"]+)"', line)
                if name:
                    names[re.search(r'gene_id "([^"]+)"', line).group(1)] = name.group(1)
    return pd.Series(names, name="human_symbol")


def map_genes(genes: pd.DataFrame, pairs: pd.DataFrame, names: pd.Series) -> pd.DataFrame:
    """For each mouse gene (columns gene, ensembl: ';'-joined ids), its human Ensembl id and
    symbol, the status, and whether the pair is high-confidence."""
    human_of = pairs.set_index("mouse_ensembl")["human_ensembl"]
    confident = pairs.set_index("human_ensembl")["high_confidence"]
    rows = []
    for gene, ids in zip(genes["gene"], genes["ensembl"]):
        human = sorted({human_of[i] for i in ids.split(";") if i in human_of.index}) if ids else []
        status = (STATUSES[3] if not ids else STATUSES[0] if len(human) == 1
                  else STATUSES[1] if not human else STATUSES[2])
        target = human[0] if status == STATUSES[0] else ""
        rows.append({"gene": gene, "mouse_ensembl": ids, "human_ensembl": target,
                     "human_symbol": names.get(target, ""), "status": status,
                     "high_confidence": bool(confident[target]) if target else None})
    return pd.DataFrame(rows)
