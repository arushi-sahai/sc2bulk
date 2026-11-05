"""scVI (scvi-tools, pinned in its own `scvi` environment: envs/scvi.txt), as a sample encoder.

- Trained unsupervised on the split's training-side cells (counts on the unit's panel), with the
  scvi-tools defaults (n_hidden 128, n_latent 10, n_layers 1, dropout 0.1, ZINB; observed library
  size); its grid varies n_latent and n_layers. Early stopping on the ELBO of whole held-out validation
  samples (external indexing), not scvi's random cell split (scvi's default has no early stopping;
  disclosed); selection by the returned model's (last epoch's) validation ELBO (label-free).
- A sample's embedding is the mean of its cells' latent means.
- Bulk and pseudobulk profiles are binomially thinned to the median training-cell library size,
  THIN_DRAWS times, and the latent means averaged (D3); the observed-library-size version is a
  sensitivity row (thin=False).
"""
import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

THIN_DRAWS = 10


def fit(train_counts, validation_counts, config: dict, seed: int, accelerator="auto"):
    """(model, record) for counts matrices (cells x panel genes, integers). record: epochs, the
    validation ELBO curve and its minimum (the selection criterion), the median training library size."""
    import scvi
    scvi.settings.seed = seed
    X = sp.vstack([sp.csr_matrix(train_counts), sp.csr_matrix(validation_counts)]).tocsr().astype(np.float32)
    adata = anndata.AnnData(X)
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata, n_hidden=config.get("n_hidden", 128), n_latent=config["n_latent"],
                            n_layers=config["n_layers"], dropout_rate=config.get("dropout_rate", 0.1),
                            gene_likelihood=config.get("gene_likelihood", "zinb"))
    n_train = train_counts.shape[0]
    indices = [np.arange(n_train), np.arange(n_train, X.shape[0]), np.array([], dtype=int)]
    model.train(early_stopping=True, accelerator=accelerator, devices=1, datasplitter_kwargs={"external_indexing": indices})
    curve = model.history["elbo_validation"]["elbo_validation"].to_numpy()
    # selection uses the returned model's ELBO: scvi's early stopping keeps the last epoch's weights
    record = {"epochs": len(curve), "validation_elbo_curve": curve.tolist(), "validation_elbo": float(curve[-1]),
              "median_library_size": float(np.median(np.asarray(sp.csr_matrix(train_counts).sum(axis=1)).ravel()))}
    return model, record


def load(directory, counts):
    """A saved model (scvi-tools' model directory), set up on these counts (cells x panel genes)."""
    import scvi
    adata = anndata.AnnData(sp.csr_matrix(counts).astype(np.float32))
    scvi.model.SCVI.setup_anndata(adata)
    return scvi.model.SCVI.load(str(directory), adata=adata)


def latent(model, counts) -> np.ndarray:
    adata = anndata.AnnData(sp.csr_matrix(counts).astype(np.float32))
    model.setup_anndata(adata)
    return model.get_latent_representation(adata, give_mean=True)


def embed_samples(model, counts, sample_ids: np.ndarray, samples) -> np.ndarray:
    """Each sample's embedding: the mean latent of its cells (rows of counts, labelled sample_ids)."""
    z = pd.DataFrame(latent(model, counts))
    return z.groupby(sample_ids).mean().loc[list(samples)].to_numpy()


def thin(counts: np.ndarray, target: float, rng) -> np.ndarray:
    """Binomial thinning of one profile to an expected total of `target` counts."""
    total = counts.sum()
    return rng.binomial(counts.astype(np.int64), min(1.0, target / total)) if total > target else counts


def embed_profiles(model, counts, median_library_size: float, seed: int, thinned=True) -> np.ndarray:
    """One-profile inputs: latent means averaged over THIN_DRAWS thinnings to the median training-cell
    library size (D3), or of the profile at its observed size (thinned=False, the sensitivity row)."""
    X = sp.csr_matrix(counts).toarray()
    if not thinned:
        return latent(model, X)
    rng = np.random.default_rng(seed)
    draws = [latent(model, np.vstack([thin(row, median_library_size, rng) for row in X])) for _ in range(THIN_DRAWS)]
    return np.mean(draws, axis=0)
