"""pineda: the pair's configuration (see sc2bulk.config.pairs for the fields). Its own file, so
editing it reruns only this pair's stages."""

PAIR = {
    "name": "Human motor cortex, ALS",
    "title": ('Human motor cortex', 'ALS'),
    "studies": "Pineda et al. → NYGC ALS Consortium",
    "species": "human",
    "single_cell": {"source": "Pineda et al. 2024, motor cortex (BA4) snRNA-seq",
                    "accession": "Synapse syn51105515 (doi:10.1016/j.cell.2024.02.031)"},
    "bulk": {"source": "NYGC ALS Consortium + Target ALS motor cortex bulk RNA-seq",
             "accession": "GEO GSE153960"},
    "label_value": "als",
    "groups": {"single_cell": {0: (0, 0), 1: (1, 1)},
               "bulk": {0: (0, 0), 1: (1, 1)}},
    "bulk_label_columns": ("nygc_group",),
    "fold_sets": {"all": 1},
    "scoring_unit": "donor",    # NYGC subjects contribute up to two samples
    "holdout_group": "donor",   # Fig 2B is marked "confounded with sequencing run"
    "expected": {"single_cell": 48, "bulk": 260, "bulk_donors": 170, "bulk_without_counts": 7,
                 "genes_without_symbol": 11894},
    "one_type": {"column": "cell_type", "by": None, "types": {None: ("Oligo",)}},
    # named exclusions, decided before any result
    "excluded_samples": {"201019_ALS_102_snRNA-B2": "failed library: 195 nuclei, 55% vascular, no "
                                                   "oligodendrocytes or astrocytes; not a usable motor-cortex sample"},
}
