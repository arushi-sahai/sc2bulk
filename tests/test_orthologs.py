"""Mapping mouse panel genes to one-to-one human orthologs (sc2bulk.orthologs.map_genes)."""
import pandas as pd

from sc2bulk.orthologs import map_genes


def test_map_genes():
    pairs = pd.DataFrame({"mouse_ensembl": ["M1", "M2", "M3", "M4"], "human_ensembl": ["H1", "H2", "H3", "H4"],
                          "high_confidence": [True, False, True, True]})
    names = pd.Series({"H1": "A", "H2": "B", "H3": "C"})   # H4 has no gene name
    genes = pd.DataFrame({"gene": ["a", "b", "c", "d", "e"], "ensembl": ["M1", "M2;M9", "M9", "M1;M3", "M4"]})
    mapped = map_genes(genes, pairs, names).set_index("gene")
    assert list(mapped["status"]) == ["mapped", "mapped", "no one2one ortholog", "conflicting ids", "mapped"]
    assert list(mapped["human_ensembl"]) == ["H1", "H2", "", "", "H4"]
    assert list(mapped["human_symbol"]) == ["A", "B", "", "", ""]
    assert mapped.loc["b", "high_confidence"] == False and mapped.loc["c", "high_confidence"] is None
