"""tms: the pair's configuration (see sc2bulk.config.pairs for the fields). Its own file, so
editing it reruns only this pair's stages."""

PAIR = {
    "name": "Mouse organs, aging",
    "title": ('Mouse organs', 'aging'),
    "studies": "Tabula Muris Senis (single-cell → bulk)",
    "species": "mouse",
    "single_cell": {"source": "Tabula Muris Senis 10x droplet, marrow, lung, kidney, limb muscle",
                    "accession": "CELLxGENE collection 0b9d8a04-bb9d-44da-aa27-705bb65b54eb"},
    "bulk": {"source": "Schaum et al. 2020 whole-organ bulk RNA-seq, the same four organs",
             "accession": "GEO GSE132040"},
    # age in months; single-cell ages are 1, 3, 18, 21, 24, 30; bulk 9-15 months excluded
    "label_value": "age",
    "groups": {"single_cell": {0: (0, 6), 1: (18, 30)},
               "bulk": {0: (0, 6), 1: (18, 30)}},
    "bulk_label_columns": (),
    "fold_sets": {"all": 1},
    "scoring_unit": "sample",   # bulk mice can't be identified across organs
    "holdout_group": "mouse",   # a mouse's organs stay together
    "strata": "organ",
    "expected": {"single_cell": 65, "bulk": 145},
    # kidney: proximal tubule under either of the provider's two labels (CL:1000838 and
    # its parent CL:0002306), annotated inconsistently across mice
    "one_type": {"column": "cell_type", "by": "organ",
                 "types": {"Kidney": ("kidney proximal convoluted tubule epithelial cell",
                                      "epithelial cell of proximal tubule"),
                           "Limb_Muscle": ("mesenchymal stem cell",),
                           "Lung": ("classical monocyte",),
                           "Marrow": ("granulocyte",)}},
}
