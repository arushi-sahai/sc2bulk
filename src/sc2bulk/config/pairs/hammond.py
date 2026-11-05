"""hammond: the pair's configuration (see sc2bulk.config.pairs for the fields). Its own file, so
editing it reruns only this pair's stages."""

PAIR = {
    "name": "Mouse microglia, aging",
    "title": ('Mouse microglia', 'aging'),
    "studies": "Hammond et al. → Reemst et al.",
    "species": "mouse",
    "single_cell": {"source": "Hammond et al. 2019, mouse microglia scRNA-seq", "accession": "GEO GSE121654"},
    "bulk": {"source": "Reemst et al. 2022, hippocampal microglia bulk RNA-seq (LPS-challenged animals excluded)",
             "accession": "GEO GSE207067"},
    # age in days (E14.5 coded as -4.5). Single-cell ages: -4.5, 4, 5, 30, 100, 540
    # (P30 young); bulk P9 vs. P200
    "label_value": "age",
    "groups": {"single_cell": {0: (-4.5, 30), 1: (100, 540)},
               "bulk": {0: (9, 9), 1: (200, 200)}},
    "bulk_label_columns": ("title", "litter", "treatment"),
    "fold_sets": {"all": 1},
    "scoring_unit": "sample",
    "holdout_group": "sample; the P5 Percoll pairs (GSM3442047-52) together",
    "expected": {"single_cell": 41, "bulk": 38, "bulk_groups": {0: 25, 1: 13}},
}
