"""QC embeddings (stage 04c_qc_embed; S2-S7 Fig panels C, D, G).

Hashed only into the stages that import it (sc2bulk.codehash).
"""

QC_EMBED_CELLS = 50_000     # cells per pair: each sample's first n stored cells, n = this // samples (a cap)
N_PCS = 50                  # single-cell PCA before the neighbor graph
N_NEIGHBORS = 15
BULK_PCS = 10
RANDOM_STATE = 0            # PCA, neighbors and UMAP (UMAP runs single-threaded with a fixed state)
