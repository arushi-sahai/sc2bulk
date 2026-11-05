"""Raw-data readers, one module per pair. Each module provides:

  single_cell_genes(raw)    the gene symbols of the single-cell side (all files)
  single_cell_samples(raw)  yields (sample, X, symbols, cells) per single-cell sample: a dict
                            of sample metadata (sample_id, donor_id, holdout_group = what a
                            split must keep together, label_value = the raw label, and a drop reason or
                            None), raw counts (cells x features), the feature
                            symbols, and a per-cell table (cell_id, cell_type)
  bulk(raw)                 (X, symbols, samples): raw counts (samples x features), the
                            feature symbols, and the sample metadata (sample_id, donor_id,
                            label_value, and a drop reason or None)
  ensembl(raw)              a table of gene symbol -> Ensembl id for the pair's annotation

Readers only read and label. Every rule that changes the data (QC, cell subsampling,
library-size cuts) is applied in sc2bulk.preprocess, the same way for every pair.
"""
