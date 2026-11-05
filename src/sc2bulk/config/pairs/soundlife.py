"""soundlife: the pair's configuration (see sc2bulk.config.pairs for the fields). Its own file, so
editing it reruns only this pair's stages."""

PAIR = {
    "name": "Human blood, aging",
    "title": ('Human blood', 'aging'),
    "studies": "Sound Life (PBMC → whole blood)",
    "species": "human",
    "single_cell": {"source": "AIFI Sound Life PBMC scRNA-seq (Gustafson et al. 2025, Nature)",
                    "accession": "doi:10.57785/bn17-3h16"},
    "bulk": {"source": "Sound Life whole-blood bulk RNA-seq, unstimulated, same donors",
             "accession": "doi:10.57785/bn17-3h16"},
    # age in years; no donor between 37 and 55
    "label_value": "age",
    "groups": {"single_cell": {0: (25, 37), 1: (55, 67)},
               "bulk": {0: (25, 37), 1: (55, 67)}},
    "bulk_label_columns": ("age_group",),
    "scoring_unit": "donor",
    "holdout_group": "donor",
    "expected": {"single_cell": 868, "bulk": 836, "donors": 96},
    "one_type": {"column": "AIFI_L1", "by": None, "types": {None: ("T cell",)}},
    # the same 96 donors on both sides: 4-fold donor cross-fitting, stratified by age
    # group (primary), and the 2026-09-30 67/29 split as a sensitivity fold set, which
    # sc2bulk.splits.donor_split reproduces from the single-cell age groups
    "fold_sets": {"crossfit": 4, "split67": 1},
    "donor_split": {"test_fraction": 0.3, "stratify": "age group", "seed": 20260930,
                    "test_donors": ("BR1002", "BR1006", "BR1007", "BR1016", "BR1018", "BR1026", "BR1028",
                                    "BR1031", "BR1035", "BR1037", "BR1040", "BR1044", "BR1047", "BR1056",
                                    "BR1058", "BR2002", "BR2003", "BR2005", "BR2019", "BR2022", "BR2027",
                                    "BR2030", "BR2035", "BR2037", "BR2038", "BR2042", "BR2045", "BR2052",
                                    "BR2053")},
}
