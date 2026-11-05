"""crohns: the pair's configuration (see sc2bulk.config.pairs for the fields). Its own file, so
editing it reruns only this pair's stages."""

PAIR = {
    "name": "Human ileum, Crohn's",
    "title": ("Human ileum", "Crohn's"),
    "studies": "IBDverse → MSCCR",
    "species": "human",
    "single_cell": {"source": "IBDverse terminal-ileum scRNA-seq (CELLxGENE)",
                    "accession": "CELLxGENE dataset 0e5c75da"},
    "bulk": {"source": "MSCCR ileal-biopsy RNA-seq", "accession": "GEO GSE193677"},
    "label_value": "cd",
    # ileal epithelium runs high in mitochondrial reads: the uniform 20% ceiling removed 39% of
    # epithelial cells and 0% of immune/stromal cells, so this pair uses the provider's own
    # 50% ceiling (label-blind, chosen from metadata after seeing pass rates; disclosed;
    # Osorio & Cai 2021)
    "qc_max_percent_mito": 50.0,
    # Crohn's (inflamed and uninflamed pooled) vs. non-IBD control
    "groups": {"single_cell": {0: (0, 0), 1: (1, 1)},
               "bulk": {0: (0, 0), 1: (1, 1)}},
    "bulk_label_columns": ("inflammation",),   # defined for Crohn's biopsies only
    "fold_sets": {"all": 1},
    "scoring_unit": "donor",
    "holdout_group": "donor",
    "expected": {"single_cell": 359, "single_cell_groups": {0: 242, 1: 117}, "bulk": 427, "bulk_groups": {0: 113, 1: 314}},
    "one_type": {"column": "cell_type", "by": None, "types": {None: ("enterocyte of epithelium proper of ileum",)}},
}
