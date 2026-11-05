"""Every output is hashed by its file bytes (sc2bulk.manifest.content_hash), so each format is
written byte-reproducibly: writing the same object
twice, or reading and rewriting it, gives identical bytes."""
import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk import manifest


def test_h5ad_bytes(tmp_path):
    rng = np.random.default_rng(0)
    adata = anndata.AnnData(sp.random(50, 30, density=0.2, format="csr", random_state=0, dtype=np.float32),
                            obs=pd.DataFrame({"sample_id": [f"s{i % 5}" for i in range(50)],
                                              "cell_type": pd.Categorical(rng.choice(["a", "b"], 50))},
                                             index=[f"c{i}" for i in range(50)]),
                            var=pd.DataFrame(index=[f"g{i}" for i in range(30)]))
    adata.uns["note"] = "x"
    adata.write_h5ad(tmp_path / "a.h5ad")
    adata.write_h5ad(tmp_path / "b.h5ad")
    anndata.read_h5ad(tmp_path / "a.h5ad").write_h5ad(tmp_path / "c.h5ad")
    hashes = {manifest.content_hash(tmp_path / f"{name}.h5ad") for name in "abc"}
    assert len(hashes) == 1


def test_gzip_csv_and_parquet_bytes(tmp_path):
    # gzip's header holds the file name, so each file is written twice under the same name
    # (the pipeline always writes the same temporary name, sc2bulk.manifest.temporary)
    assert manifest.temporary(tmp_path / "t.csv.gz") == tmp_path / ".tmp.t.csv.gz"
    table = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    writers = {"t.csv.gz": lambda path: table.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0}),
               "t.parquet": lambda path: table.to_parquet(path, index=False)}
    for name, write in writers.items():
        write(tmp_path / name)
        first = manifest.content_hash(tmp_path / name)
        write(tmp_path / name)
        assert manifest.content_hash(tmp_path / name) == first, name
