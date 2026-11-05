"""The per-cell and per-sample preprocessing rules."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk import preprocess
from sc2bulk.config import settings
from sc2bulk.data.hammond import restore_names
from sc2bulk.splits import donor_split


def test_gene_axis_sums_duplicates_drops_non_genes_and_zero_fills():
    symbols = ["A", "B", "A", "__no_feature", "ERCC-00002"]
    genes = pd.Index(["A", "B", "C"])
    X = np.array([[1, 2, 3, 4, 5], [0, 1, 0, 9, 9]])
    out = preprocess.to_gene_axis(X, symbols, genes).toarray()
    assert out.tolist() == [[4, 2, 0], [0, 1, 0]]
    assert list(preprocess.gene_axis(symbols)) == ["A", "B"]
    assert preprocess.merged_symbols(symbols).to_dict() == {"A": 2}


def test_cell_qc_thresholds():
    genes = pd.Index(["mt-Co1"] + [f"g{i}" for i in range(300)])
    healthy = np.r_[10, np.ones(300)]           # 300 genes, 10 of 310 counts mitochondrial
    dying = np.r_[100, np.ones(300)]            # 100 of 400 counts mitochondrial (25%)
    sparse = np.r_[0, np.ones(150), np.zeros(150)]   # 150 genes
    qc = preprocess.cell_qc(sp.csr_matrix(np.vstack([healthy, dying, sparse])), genes, "mouse")
    assert qc["genes_detected"].tolist() == [301, 301, 150]
    assert np.allclose(qc["percent_mito"], [100 * 10 / 310, 25.0, 0.0])
    assert qc["passed_qc"].tolist() == [True, False, False]


def test_stored_cells_seeded_random_and_capped():
    a = preprocess.stored_cells("tms", "Lung_1", 5000)
    assert len(a) == settings.MAX_STORED_CELLS and len(set(a)) == len(a)
    assert (a == preprocess.stored_cells("tms", "Lung_1", 5000)).all()       # same every run
    assert not (a == preprocess.stored_cells("tms", "Lung_2", 5000)).all()   # differs by sample
    assert sorted(preprocess.stored_cells("tms", "Lung_1", 331)) == list(range(331))   # small samples keep every cell


def test_group_of_inclusive_ranges():
    groups = {0: (-4.5, 30), 1: (100, 540)}
    assert [preprocess.group_of(v, groups) for v in (-4.5, 30, 31, 100, 540)] == [0, 0, None, 1, 1]


def test_restore_r_mangled_names():
    known = ["mt-Nd1", "H2-K1", "0610007P14Rik", "Actb"]
    assert restore_names(["mt.Nd1", "H2.K1", "X0610007P14Rik", "Actb", "Gt.ROSA.26Sor"], known) == \
        ["mt-Nd1", "H2-K1", "0610007P14Rik", "Actb", "Gt.ROSA.26Sor"]


def test_donor_split_is_stratified_and_seeded():
    donors = pd.DataFrame({"donor_id": [f"d{i:02d}" for i in range(20)], "age_group": ["young"] * 10 + ["old"] * 10})
    test = donor_split(donors, 0.3, 1)
    assert len(test) == 6 and len(test & set(donors.donor_id[:10])) == 3
    assert test == donor_split(donors.sample(frac=1, random_state=0), 0.3, 1)   # row order doesn't matter


def test_type_cells_nest_inside_the_main_stored_cells():
    rng = np.random.default_rng(0)
    is_type = rng.random(3000) < 0.3
    main = preprocess.stored_cells("soundlife", "KT1", 3000)
    typed = preprocess.stored_type_cells("soundlife", "KT1", is_type)
    assert is_type[typed].all() and len(typed) == min(settings.MAX_STORED_CELLS, is_type.sum())
    in_main = main[is_type[main]]                      # the type's cells among the main stored cells
    assert (typed[:len(in_main)] == in_main).all()     # come first, in the same order


def test_missing_symbols_fail_loudly():
    import pytest
    for bad in (["A", None], ["A", ""], ["A", "nan"], ["A", "N/A"]):
        with pytest.raises(AssertionError):
            preprocess.gene_axis(bad)


def test_log_normalize_and_seurat_dispersion_match_scanpy():
    import anndata
    import scanpy as sc
    rng = np.random.default_rng(0)
    counts = sp.csr_matrix(rng.negative_binomial(2, 0.3, size=(500, 300)).astype(float))
    ours = preprocess.log_normalize(counts)
    adata = anndata.AnnData(counts.copy())
    sc.pp.normalize_total(adata, target_sum=settings.TARGET_SUM)
    sc.pp.log1p(adata)
    assert np.allclose(ours.toarray(), adata.X.toarray(), atol=1e-5)
    table = preprocess.seurat_dispersion(ours, chunk=37)   # chunking must not change the result
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_bins=settings.N_DISPERSION_BINS)
    assert np.allclose(table["dispersions_norm"], adata.var["dispersions_norm"], atol=1e-4, equal_nan=True)
    assert np.allclose(table["means"], adata.var["means"], atol=1e-5)
