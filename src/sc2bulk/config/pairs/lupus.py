"""lupus: the pair's configuration (see sc2bulk.config.pairs for the fields). Its own file, so
editing it reruns only this pair's stages."""

PAIR = {
    "name": "Human blood, lupus",
    "title": ("Human blood", "lupus"),
    "studies": "Perez et al. → Hung et al.",
    "species": "human",
    "single_cell": {"source": "Perez et al. 2022, PBMC scRNA-seq, processing cohort 4 (CELLxGENE)",
                    "accession": "CELLxGENE dataset 218acb0f (doi:10.1126/science.abf1970)"},
    "bulk": {"source": "Hung et al. 2015, whole-blood RNA-seq (recount3 coverage / 50 bp)",
             "accession": "SRA SRP062966 / GEO GSE72509 (doi:10.1126/science.aac7442)"},
    "label_value": "sle",
    "groups": {"single_cell": {0: (0, 0), 1: (1, 1)},
               "bulk": {0: (0, 0), 1: (1, 1)}},
    "bulk_label_columns": (),   # only the disease status is read; titles, anti-Ro and the interferon score are never stored
    "fold_sets": {"all": 1},
    "scoring_unit": "donor",
    "holdout_group": "donor",
    "expected": {"single_cell": 84, "single_cell_groups": {0: 44, 1: 40}, "bulk": 117, "bulk_groups": {0: 18, 1: 99}},
    "one_type": {"column": "cell_type", "by": None, "types": {None: ("CD4-positive, alpha-beta T cell",)}},
}
