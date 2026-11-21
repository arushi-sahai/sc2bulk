"""Stage 09c: the Fig 3A / S13A-E coordinates on a toy unit: stage 09's pc rows
exactly, at fold 0, at the configuration's encoder seed, every sample once per alignment and measure; the inputs it
records; determinism. The agreement with stage 09's saved rows is a run-level check (tests/run_level/check09c.py)."""
import runpy

import numpy as np
import pandas as pd
import pytest

from sc2bulk import codehash, folds, manifest, schemas
from sc2bulk.config import alignment as A
from sc2bulk.config import geometry, settings

C = runpy.run_path(str(settings.REPO / "pipeline" / "09c_geometry.py"), run_name="stage09c")
S08, S09 = C["S08"], C["S09"]


def toy_unit(monkeypatch):
    """Stage 08's readers replaced by a toy unit of 40 training and 24 bulk samples in two organs; encoder seeds 0, 1,
    10 and 11 offered, each its own matrix."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"sample_id": [f"sc{i}" for i in range(40)], "group": [0, 1] * 20, "organ": np.repeat(["a", "b"], 20)})
    bulk = pd.DataFrame({"sample_id": [f"b{i}" for i in range(24)], "organ": np.repeat(["a", "b"], 12)})
    bulk["scoring_unit_id"] = bulk["sample_id"]
    noise = {e: (rng.normal(size=(40, 6)), rng.normal(2.0, 1.5, size=(24, 6))) for e in (-1, 0, 1, 10, 11)}
    monkeypatch.setitem(S08, "N_READOUT", 1)
    monkeypatch.setitem(S08, "scored", lambda pair, unit: bulk)
    monkeypatch.setitem(S08, "readout_folds", lambda pair, unit, ids, r: np.arange(len(ids)) % 5)
    monkeypatch.setitem(S08, "inputs", lambda pair, unit, key: [])
    monkeypatch.setitem(S08, "encoder_seeds", lambda pair, key: [0, 1, 10, 11])
    monkeypatch.setitem(S08, "trained_features", lambda pair, unit, key, e, t, b: noise[e])
    monkeypatch.setitem(S08, "features", lambda pair, unit, key, t, b: [(-1, *noise[-1])])
    monkeypatch.setitem(S08, "feature_names", lambda pair, unit, key, X: [f"f{i}" for i in range(X.shape[1])])
    monkeypatch.setattr(folds, "training_samples", lambda pair, fold_set, fold: train)
    rows = manifest.settings_hash({"training": list(train["sample_id"]), "bulk": list(bulk["sample_id"])})
    monkeypatch.setattr(manifest, "read", lambda path: {"settings": {"code": "c", "inputs": {}, "rows": rows}})
    monkeypatch.setattr(manifest, "stage_code_hash", lambda *args: "c")
    monkeypatch.setattr(manifest, "input_hashes", lambda paths: {})
    return train, bulk


def coordinates(pair, unit, key) -> pd.DataFrame:
    return schemas.check("alignment_geometry", schemas.with_neutral_keys(C["table"](pair, unit, key)))


def stage09_coordinates(pair, unit, key) -> pd.DataFrame:
    made = S09["tables"](pair, unit, key)["alignment_geometry"]
    table = schemas.check("alignment_geometry", schemas.with_neutral_keys(pd.concat(made, ignore_index=True)))
    return table[table["measure"].isin(["pc1", "pc2"])].reset_index(drop=True)


@pytest.mark.parametrize("pair, key", [("tms", "cytoset_cells"), ("hammond", "pseudobulk"), ("lupus", "scgpt_cells")])
def test_coordinates_are_stage_09s(monkeypatch, pair, key):
    toy_unit(monkeypatch)
    monkeypatch.setattr(A, "GEOMETRY_CONFIGURATIONS", geometry.CONFIGURATIONS)   # stage 09's rows for scGPT too
    mine, theirs = coordinates(pair, ("all", 0), key), stage09_coordinates(pair, ("all", 0), key)
    pd.testing.assert_frame_equal(mine, theirs, check_exact=True)


def test_rows_seeds_alignments_and_measures(monkeypatch):
    train, bulk = toy_unit(monkeypatch)
    ids = [*train["sample_id"], *bulk["sample_id"]]
    for key, seed in (("cytoset_cells", A.GEOMETRY_ENCODER_SEED), ("scgpt_cells", -1), ("pseudobulk", -1)):
        table = coordinates("tms", ("all", 0), key)
        assert set(table["encoder_seed"]) == {seed}
        assert set(table["alignment"]) == {"none", *A.ALIGNED} and set(table["measure"]) == {"pc1", "pc2"}
        assert set(table["readout"]) == {"none"} and set(table["readout_seed"]) == {-1}
        for _, group in table.groupby(["alignment", "measure"]):
            assert list(group["sample_id"]) == ids


def test_harmony_column_moves_the_rows(monkeypatch):
    toy_unit(monkeypatch)
    table = coordinates("hammond", ("all", 0), "scgpt_cells")
    values = {a: part.sort_values(["measure", "sample_id"])["value"].to_numpy() for a, part in table.groupby("alignment")}
    assert not np.allclose(np.abs(values["harmony"]), np.abs(values["unaligned_stacked"]))


def test_fold_0_of_the_primary_fold_set_only():
    assert C["units"]("soundlife") == [("crossfit", 0)]
    assert C["units"]("crohns") == [("all", 0)]


def test_inputs_leave_out_the_seeds_not_read(monkeypatch):
    toy_unit(monkeypatch)
    files = {e: S08["trained_file"]("tms", ("all", 0), "cytoset_cells", e) for e in (0, 1, 10, 11)}
    monkeypatch.setitem(S08, "inputs", lambda pair, unit, key: ["panel.csv", *files.values()])
    assert C["inputs"]("tms", ("all", 0), "cytoset_cells") == ["panel.csv", files[A.GEOMETRY_ENCODER_SEED]]
    monkeypatch.setitem(S08, "inputs", lambda pair, unit, key: ["panel.csv", "scgpt_cells.h5ad"])
    assert C["inputs"]("tms", ("all", 0), "scgpt_cells") == ["panel.csv", "scgpt_cells.h5ad"]


def test_deterministic(monkeypatch):
    toy_unit(monkeypatch)
    pd.testing.assert_frame_equal(coordinates("tms", ("all", 0), "pseudobulk"), coordinates("tms", ("all", 0), "pseudobulk"),
                                  check_exact=True)


def test_hash_covers_stages_09_and_08_and_the_list():
    found = {str(p.relative_to(settings.REPO)) for p in codehash.files(settings.REPO / "pipeline" / "09c_geometry.py", "tms",
                                                                       codehash.extra("09c_geometry", "tms"))}
    assert {"pipeline/09_alignment.py", "pipeline/08_readout.py", "src/sc2bulk/config/geometry.py", "src/sc2bulk/config/alignment.py",
            "src/sc2bulk/alignment.py"} <= found
    stage09 = codehash.files(settings.REPO / "pipeline" / "09_alignment.py", "tms", codehash.extra("09_alignment", "tms"))
    assert settings.REPO / "src" / "sc2bulk" / "config" / "geometry.py" not in stage09
    assert set(geometry.CONFIGURATIONS) >= set(A.GEOMETRY_CONFIGURATIONS)
