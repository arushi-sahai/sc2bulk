"""training_inputs.unit_cells reads the stored cells' panel columns without holding the shared-gene matrix:
its output equals the former computation, preprocess.log_normalize
over the shared genes then the panel columns, exactly (values, dtype, sparsity structure, obs and cell order)."""
import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk import folds, preprocess, training_inputs

UNIT = ("crossfit", 0)


def store(n_cells=300, seed=0):
    """Toy stored cells: a gene axis wider than the shared genes, in an order that is neither the shared genes'
    (sorted) nor the panel's; int32 counts. Some cells have counts only off the panel (on shared genes) and
    some have many counts on genes outside the shared set, so the library totals must come from every shared
    gene and from no other. One explicit stored zero."""
    rng = np.random.default_rng(seed)
    stored = [f"g{i:03d}" for i in rng.permutation(120)]
    shared = sorted(stored[:90])
    panel = list(rng.permutation(shared[:40]))
    off_panel, outside = [stored.index(g) for g in shared[40:]], [stored.index(g) for g in stored[90:]]
    X = sp.random(n_cells, len(stored), density=0.3, format="csr", random_state=seed,
                  data_rvs=lambda k: rng.negative_binomial(2, 0.05, k) + 1).astype(np.int32).tolil()
    for i in range(0, n_cells, 10):   # only off-panel shared genes, plus large non-shared counts
        X[i, :] = 0
        X[i, off_panel[i % len(off_panel)]] = 1 + i
        X[i, outside[i % len(outside)]] = 50_000
    for i in range(5, n_cells, 10):   # most of the library outside the shared genes
        X[i, outside[: 10]] = 10_000
    X = X.tocsr()
    X.data[np.flatnonzero(X.data)[3]] = 0   # an explicit zero, kept as stored
    obs = pd.DataFrame({"sample_id": [f"s{i % 6}" for i in range(n_cells)], "cell_type": "T",
                        "total_counts": np.asarray(X.sum(axis=1)).ravel()},
                       index=[f"c{i}" for i in rng.permutation(n_cells)])
    return anndata.AnnData(X, obs=obs, var=pd.DataFrame(index=stored)), shared, panel


def patch(monkeypatch, data, shared, panel):
    samples = pd.DataFrame({"sample_id": [f"s{i}" for i in range(6)], "group": [0, 1] * 3, "holdout_group": [f"s{i}" for i in range(6)]})
    monkeypatch.setattr(folds, "training_cells", lambda pair, fold_set, fold, cell_subset="all": data.copy())
    monkeypatch.setattr(folds, "training_samples", lambda pair, fold_set, fold: samples)
    monkeypatch.setattr(training_inputs, "shared", lambda pair: shared)
    monkeypatch.setattr(training_inputs, "panel", lambda pair, unit: panel)


def former(data, shared, panel, counts):
    """unit_cells' X before the change (the reference)."""
    X = data[:, shared].X
    X = sp.csr_matrix(X, dtype=np.float32) if counts else preprocess.log_normalize(X)
    return X[:, pd.Index(shared).get_indexer(panel)].tocsr()


def identical(A, B):
    assert type(A) is type(B) and A.dtype == B.dtype == np.float32 and A.shape == B.shape
    assert np.array_equal(A.indptr, B.indptr) and np.array_equal(A.indices, B.indices)
    assert A.data.tobytes() == B.data.tobytes()


def test_totals_need_every_shared_gene():
    """The toy store is a real test: totals over the panel alone, or over every stored gene, differ."""
    data, shared, panel = store()
    over = lambda genes: np.asarray(data[:, genes].X.sum(axis=1)).ravel()
    assert (over(shared) != over(panel)).any() and (over(shared) != over(list(data.var_names))).any()


def test_unit_cells_equals_former(monkeypatch):
    data, shared, panel = store()
    patch(monkeypatch, data, shared, panel)
    for counts in (False, True):
        cells = training_inputs.unit_cells("toy", UNIT, counts=counts)
        identical(cells.X, former(data, shared, panel, counts))
        assert cells.genes == panel
        assert np.array_equal(cells.cell_id, data.obs_names.astype(str).to_numpy())
        assert np.array_equal(cells.sample_id, data.obs["sample_id"].astype(str).to_numpy())
        assert np.array_equal(cells.library_size, data.obs["total_counts"].to_numpy(dtype=float))


def test_chunks_do_not_matter():
    """Row chunks smaller than the store, not dividing it, give the same values as one chunk."""
    data, shared, panel = store(n_cells=203, seed=1)
    columns = data.var_names.get_indexer(shared), data.var_names.get_indexer(panel)
    for chunk in (1, 7, 64, 10_000):
        identical(training_inputs.log_normalized_panel(data.X, *columns, chunk=chunk), former(data, shared, panel, False))


def test_float32_row_sums():
    """Non-integer values, whose float32 row sums differ from float64 ones: the totals keep log_normalize's
    float32 accumulation (integer counts sum exactly either way, so the tests above can't tell)."""
    data, shared, panel = store(seed=2)
    X = sp.csr_matrix(data.X, dtype=np.float32)
    X.data *= np.exp(np.random.default_rng(2).normal(0, 3, X.nnz)).astype(np.float32)
    data = anndata.AnnData(X, obs=data.obs, var=data.var)
    counts = X[:, data.var_names.get_indexer(shared)]
    assert (np.asarray(counts.sum(axis=1, dtype=np.float64)).ravel() != np.asarray(counts.astype(np.float64).sum(axis=1)).ravel()).any()
    columns = data.var_names.get_indexer(shared), data.var_names.get_indexer(panel)
    identical(training_inputs.log_normalized_panel(data.X, *columns, chunk=7), former(data, shared, panel, False))
