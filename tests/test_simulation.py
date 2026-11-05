"""The planted-signal simulation (sc2bulk.simulation and pipeline/12_simulation.py): the scenarios, labels, planted genes, calibration, halves
and criterion on fixed inputs; the seed streams of stages 12 and 12a against every other use; and stage 12 end to end on
a CPU toy unit (two encoders, small CytoSets), with its leakage and determinism checks."""
import importlib.util
import json
import re
import types

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from sc2bulk import harness, manifest, paths, preprocess, schemas, seeds, simulation, training_inputs
from sc2bulk.config import properties as P
from sc2bulk.config import settings
from sc2bulk.config import simulation as SIM
from sc2bulk.config import training as T
from sc2bulk.encoders import cytoset
from test_harness import CONFIGS


def test_structure_only_keeps_sums_and_one_subset():
    rng = np.random.default_rng(0)
    block = rng.poisson(2.0, (100, 6))
    weight = rng.uniform(500, 1500, 100)
    subset = simulation.subset_of(rng.permutation(100), 0.2)
    out = simulation.structure_only(block, weight, subset, np.random.default_rng(1))
    assert np.array_equal(out.sum(axis=0), block.sum(axis=0))                      # every per-gene sum preserved
    outside = np.setdiff1d(np.arange(100), subset)
    assert (out[outside] == 0).all() and len(subset) == 20                          # one subset, shared by every gene
    everyone = simulation.structure_only(block, weight, np.arange(100), np.random.default_rng(1))   # label 0: all cells
    assert np.array_equal(everyone.sum(axis=0), block.sum(axis=0)) and (everyone > 0).any(axis=1).mean() > 0.5
    order = rng.permutation(50)
    assert set(simulation.subset_of(order, 0.2)) <= set(simulation.subset_of(order, 0.6))   # subsets nest across levels


def test_mean_shift_thins_the_named_genes_by_rho():
    block = np.full((2000, 4), 10)
    thinned = np.array([True, False, True, False])
    u = simulation.uniforms(block, np.random.default_rng(2))
    out = simulation.mean_shift(block, thinned, 0.7, u)
    assert np.array_equal(out[:, ~thinned], block[:, ~thinned])                     # other genes untouched
    assert out[:, thinned].sum() / block[:, thinned].sum() == pytest.approx(0.7, abs=0.01)
    weaker = simulation.mean_shift(block, thinned, 0.9, u)
    assert (out <= weaker).all()                                                    # stronger levels keep subsets


def test_simulated_labels_balanced_within_strata():
    stratum = pd.Series(["0"] * 7 + ["1"] * 10, index=[f"g{i:02d}" for i in range(17)])
    labels = simulation.simulated_labels(stratum, 3)
    for key in ("0", "1"):
        counts = labels[stratum == key].value_counts()
        assert abs(counts.get(0, 0) - counts.get(1, 0)) <= 1
    assert labels.equals(simulation.simulated_labels(stratum, 3))
    odd = {simulation.simulated_labels(stratum, s)[stratum == "0"].sum() for s in range(20)}
    assert odd == {3, 4}                                                            # the odd group's extra by a coin


def test_planted_genes_from_the_filter():
    panel = [f"g{k}" for k in range(80)] + ["MT-CO1", "mt-Nd1"]
    detected = {g: 0.5 if int(re.sub(r"\D", "", g) or 0) < 60 else 0.05 for g in panel}
    planted = simulation.planted_genes(panel, detected, 5)
    assert len(planted) == SIM.N_PLANTED and (planted == "plus").sum() == SIM.N_PLANTED // 2
    assert all(detected[g] >= SIM.DETECTION_MIN_FRACTION and not g.startswith(SIM.MITO_PREFIXES) for g in planted.index)
    assert planted.equals(simulation.planted_genes(list(reversed(panel)), detected, 5))   # order-free, seeded
    with pytest.raises(AssertionError):
        simulation.planted_genes(panel[:40], detected, 5)


def test_calibration_takes_the_weakest_level_reaching_the_target():
    assert simulation.calibrate([0.6, 0.85, 0.91, 0.95]) == (2, True)
    assert simulation.calibrate([0.6, 0.92, 0.89, 0.95]) == (1, True)               # first in grid order
    assert simulation.calibrate([0.6, 0.7]) == (1, False)                           # none: the strongest, reported


def test_halves_stratified_and_disjoint():
    labels = pd.Series([0] * 9 + [1] * 8, index=[f"g{i:02d}" for i in range(17)])
    half = simulation.half_split(labels, 7, 2)
    for label in (0, 1):
        counts = half[labels == label].value_counts()
        assert counts["A"] >= 2 and counts["B"] >= 2 and counts["A"] - counts["B"] in (0, 1)
    with pytest.raises(AssertionError):
        simulation.half_split(pd.Series([0, 0, 0, 1, 1, 1], index=list("abcdef")), 7, 2)


def test_log_normalize_rows_matches_preprocess_and_the_structure_oracle():
    rng = np.random.default_rng(8)
    counts = rng.poisson(1.5, (30, 40))
    expected = preprocess.log_normalize(counts)[:, :25].toarray()
    ours = simulation.log_normalize_rows(sp.csr_matrix(counts[:, :25]), counts.sum(axis=1)).toarray()
    assert np.array_equal(ours, expected)
    assert np.array_equal(simulation.log_cp10k(counts[:, :25], counts.sum(axis=1)), expected)
    block, totals = np.full((10, 3), 4), np.full(10, 1000.0)
    concentrated = np.zeros_like(block)
    concentrated[:2] = 20                                                           # same sums in 2 of 10 cells
    assert simulation.oracle_structure(concentrated, totals) > simulation.oracle_structure(block, totals)


def criterion_table(values: dict) -> pd.DataFrame:
    """values: key -> list of within AUROCs over (e, r) cells (structure only); the mean scenario is filler."""
    rows = [{"key": key, "encoder_seed": i // 2, "readout_seed": i % 2, "scenario": scenario, "within_sc_auroc": v}
            for scenario in SIM.SCENARIOS for key, vs in values.items() for i, v in enumerate(vs)]
    return pd.DataFrame(rows)


def test_criterion_on_fixed_numbers():
    routes = {"cytoset": "none", "set_transformer": "per_sample", "scrat": "population"}
    values = {"pseudobulk": [0.0, 0.0], "mean_log_cells": [0.05, 0.05], "cytoset_cells": [0.1, 0.1],     # exactly 0.10: passes
              "set_transformer_cells": [0.3, 0.3], "scrat_cells": [0.15, 0.0]}                          # scrat: a collapsed seed
    result = simulation.criterion(criterion_table(values), routes)
    assert result["passes"] == {"cytoset": True, "set_transformer": True, "scrat": False, "mean_log_cells": False}
    assert result["delta"]["scrat"] == pytest.approx(0.075)                       # did-it-train failures count (all seeds)
    assert not result["control_passes"] and result["failing"] == ["scrat"]
    assert result["structure_readings"]["cytoset"] == simulation.READINGS[(True, False)]
    assert result["structure_readings"]["scrat"] == simulation.READINGS[(False, False)]
    assert "inconclusive about set encoders specifically" in result["structure_readings"]["scrat"]
    assert result["mean_shift_readings"]["set_transformer"] == simulation.PER_SAMPLE_MEAN_READING
    assert result["mean_shift_readings"]["cytoset"] == simulation.FORMAT_READING
    values = {**values, "mean_log_cells": [0.2, 0.2], "scrat_cells": [0.2, 0.2]}
    result = simulation.criterion(criterion_table(values), routes)
    assert result["control_passes"] and result["structure_readings"]["scrat"] == simulation.READINGS[(True, True)]
    assert result["by_construction"] == simulation.BY_CONSTRUCTION and "never change" in result["readings_note"]
    # an exact 0.10 gap in floating point (0.6 - 0.5 < 0.1) passes: inclusive, with the 1e-12 tolerance
    tie = {"pseudobulk": [0.5, 0.5], "mean_log_cells": [0.5, 0.5], "cytoset_cells": [0.6, 0.6]}
    assert 0.6 - 0.5 < SIM.SUCCESS_MARGIN and simulation.criterion(criterion_table(tie), {"cytoset": "none"})["passes"]["cytoset"]
    below = {**tie, "cytoset_cells": [0.6 - 1e-9, 0.6 - 1e-9]}
    assert not simulation.criterion(criterion_table(below), {"cytoset": "none"})["passes"]["cytoset"]


def test_triage_cut_leaves_scrat_out_of_the_criterion(monkeypatch):
    """A set encoder cut before unblinding (config/triage) is neither trained nor judged: the criterion is over the kept
    ones, and criterion.json lists the cut one."""
    import importlib
    from sc2bulk.config import triage
    monkeypatch.setattr(triage, "CUT_CONFIGURATIONS", ("scrat_cells",))
    kept = importlib.reload(SIM)
    try:
        assert kept.ENCODERS == ("cytoset", "set_transformer")
        values = {"pseudobulk": [0.5, 0.5], "mean_log_cells": [0.5, 0.5], "cytoset_cells": [0.7, 0.7], "set_transformer_cells": [0.7, 0.7]}
        result = simulation.criterion(criterion_table(values), {m: "none" for m in kept.ENCODERS})
        assert result["control_passes"] and result["set_encoders"] == ["cytoset", "set_transformer"] and result["cut"] == ["scrat"]
        monkeypatch.setattr(triage, "CUT_CONFIGURATIONS", ("cytoset_cells", "set_transformer_cells", "scrat_cells"))
        with pytest.raises(AssertionError):
            importlib.reload(SIM)                                                       # no set encoder left: refused
    finally:
        monkeypatch.setattr(triage, "CUT_CONFIGURATIONS", ())
        importlib.reload(SIM)


def code_files():
    for directory in ("src", "pipeline"):
        yield from (settings.REPO / directory).rglob("*.py")


def test_streams_of_stages_12_and_12a_are_their_own():
    """`simulation` (id 18) is drawn only by stage 12, at distinct indices; the 12a nulls' `permutation` indices are clear
    of the blinded run's (< PERMUTATION_OFFSET) and row D's (PERMUTATION_OFFSET + j, j < N_PERMUTED); every other
    `permutation` call site is one of those."""
    users = {str(p.relative_to(settings.REPO)) for p in code_files() if re.search(r"""derived\(\s*["']simulation["']""", p.read_text())}
    assert users == {"pipeline/12_simulation.py"}
    indices = [SIM.GENES_INDEX, SIM.LABELS_INDEX, *SIM.DRAWS_INDEX.values(), *(SIM.HALF_OFFSET + e for e in range(10))]
    assert len(set(indices)) == len(indices)
    null = {P.PROPERTY_NULL_OFFSET, P.PROPERTY_NULL_OFFSET + 1}
    assert not null & set(range(T.PERMUTATION_OFFSET + T.N_PERMUTED))
    permutation_users = {str(p.relative_to(settings.REPO)) for p in code_files() if re.search(r"""derived\(\s*["']permutation["']""", p.read_text())}
    assert permutation_users == {"src/sc2bulk/labels.py", "src/sc2bulk/training_inputs.py", "pipeline/12a_properties.py"}


# ---- stage 12 end to end on a toy unit ----

GENES = [f"g{k}" for k in range(80)]
PANEL = GENES[:70]
N_SAMPLES, N_CELLS = 40, 40
IDS = [f"s{i:02d}" for i in range(N_SAMPLES)]
SIDES = {s: "validation" if i < 16 else "train" for i, s in enumerate(IDS)}   # 8 + 8 validation groups per real label


def driver():
    spec = importlib.util.spec_from_file_location("stage12", settings.REPO / "pipeline" / "12_simulation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def toy_source(M):
    rng = np.random.default_rng(0)
    counts = rng.poisson(rng.uniform(0.5, 3.0, len(GENES)), (N_SAMPLES * N_CELLS, len(GENES)))
    sample_id = np.repeat(IDS, N_CELLS)
    samples = pd.DataFrame({"group": [i % 2 for i in range(N_SAMPLES)], "holdout_group": IDS}, index=IDS)
    return M.Source(counts=sp.csr_matrix(counts), genes=GENES, panel=PANEL, cell_id=np.array([f"c{i}" for i in range(len(sample_id))]),
                    sample_id=sample_id, cell_type=np.array(["t"] * len(sample_id)), total_counts=counts.sum(axis=1) + 50.0, samples=samples)


def patched(tmp_path, monkeypatch):
    """Stage 12 on the toy unit: its data, stage 05's selection, stage 03's splits and folds, outputs and manifests stubbed;
    two encoders, both small CytoSets; one split, two read-out seeds; strong levels; every seed drawn recorded."""
    M = driver()
    source = toy_source(M)
    config = {**CONFIGS["cytoset"], "ncell": 500, "n_epochs": 12, "patience": 3}
    monkeypatch.setattr(T, "FROZEN_NCELL", 500)
    monkeypatch.setattr(SIM, "ENCODERS", ("cytoset", "set_transformer"))
    monkeypatch.setattr(SIM, "SPLITS", (0,))
    monkeypatch.setattr(SIM, "RHO_GRID", (0.3, 0.2))
    monkeypatch.setattr(SIM, "Q_GRID", (0.15, 0.1))
    monkeypatch.setattr(SIM, "GRIDS", {"mean_shift": SIM.RHO_GRID, "structure_only": SIM.Q_GRID})
    monkeypatch.setattr(M, "N_READOUT", 2)
    monkeypatch.setattr(M, "load", lambda pair, unit: source)
    monkeypatch.setattr(M, "readout_folds", lambda pair, unit, ids, r, context: (np.arange(len(ids)) + r) % 5)
    M.TRAIN["frozen"] = lambda pair, unit, encoder, key="frozen": dict(config)
    M.TRAIN["network"] = lambda encoder, c, in_dim: cytoset.network(c, in_dim)
    M.TRAIN["selection_path"] = lambda pair, unit, encoder: tmp_path / f"{encoder}_selection.json"
    M.TRAIN["scoring_units"] = lambda pair: {s: s for s in IDS}
    monkeypatch.setattr(harness, "micro_batch_for", lambda encoder, c: c["batch_size"])
    monkeypatch.setattr(training_inputs, "encoder_sides", lambda pair, unit, e: dict(SIDES))
    columns = pd.Index(GENES).get_indexer(PANEL)
    pseudobulk = {s: preprocess.log_normalize(preprocess.summed(source.counts[source.sample_id == s])[None, :])[:, columns].toarray().ravel()
                  for s in IDS}
    monkeypatch.setattr(training_inputs, "pseudobulk_log", lambda pair, unit: pseudobulk)
    monkeypatch.setattr(paths, "output", lambda stage, pair, name, unit=None: tmp_path / stage / name)
    monkeypatch.setattr(manifest, "finish", lambda p, r, c, seed=None, inputs=(): manifest.temporary(p).replace(p))
    monkeypatch.setattr(manifest, "begin", lambda device="cpu": {})
    monkeypatch.setattr(manifest, "is_done", lambda outputs, record: False)
    monkeypatch.setattr(manifest, "input_hashes", lambda sources: {})
    monkeypatch.setattr(manifest, "stage_code_hash", lambda *a: "code")
    drawn = []

    def derived(purpose, pair, index, fold_set="all", fold=0):
        drawn.append((purpose, index))
        return seeds.derived(purpose, pair, index, fold_set, fold)
    monkeypatch.setattr(M, "seeds", types.SimpleNamespace(derived=derived))
    return M, source, drawn


def run_stage(M, source):
    pair, unit = SIM.PAIR, SIM.UNIT
    d = M.design(pair, unit, source)
    M.write_design(pair, unit, source, d)
    M.train_all(pair, unit, source, d, "main", None)
    M.readout_all(pair, unit, source, d)
    return d


def test_stage12_end_to_end(tmp_path, monkeypatch):
    M, source, drawn = patched(tmp_path, monkeypatch)
    d = run_stage(M, source)
    out = tmp_path / M.STAGE
    allowed = {("simulation", i) for i in (SIM.GENES_INDEX, SIM.LABELS_INDEX, *SIM.DRAWS_INDEX.values(), SIM.HALF_OFFSET)}
    assert set(drawn) <= allowed and ("simulation", SIM.HALF_OFFSET) in drawn                # stage 12's own streams only
    # the design: planted genes, labels balanced within the real groups, halves of the validation side
    planted = pd.read_csv(out / "planted_genes.csv")
    assert len(planted) == SIM.N_PLANTED and set(planted["gene"]) <= set(PANEL)
    labels = pd.read_csv(out / "simulated_labels.csv")
    assert labels.groupby("sim_group").size().tolist() == [20, 20]
    assert (labels.assign(real=source.samples["group"].to_numpy()).groupby("real")["sim_group"].mean() == 0.5).all()
    a_ids, b_ids = d["halves"][0]
    assert set(a_ids) | set(b_ids) == {s for s, side in SIDES.items() if side == "validation"} and not set(a_ids) & set(b_ids)
    # each training: half B is in no training input (leakage), every split-0 sample embedded, half B's simulated bulk
    for scenario in SIM.SCENARIOS:
        for encoder in SIM.ENCODERS:
            data = anndata.read_h5ad(out / scenario / encoder / "embeddings" / "s0.h5ad")
            side = data.obs.groupby("side", observed=True)["sample_id"].apply(set)
            assert side["half_b"] == set(b_ids) == side["simulated_bulk"] and side["half_a"] == set(a_ids)
            assert side["train"] == {s for s, v in SIDES.items() if v == "train"}
            record = json.loads((out / scenario / encoder / "record_s0.json").read_text())
            assert record["stop_epoch"] >= T.BURN_IN_EPOCHS
    # the read-outs: every row, split, read-out seed and scenario; the strong planted signals are detected where expected
    table = schemas.read("simulation", out / "simulation.parquet")
    assert len(table) == 2 * 2 * (2 + len(SIM.ENCODERS)) and set(table["n_scored"]) == {len(b_ids)}
    mean = table.groupby(["scenario", "key"])["within_sc_auroc"].mean()
    assert mean[("structure_only", "mean_log_cells")] >= 0.8       # a fixed mean of logs sees concentration ...
    assert mean[("mean_shift", "pseudobulk")] >= 0.8               # ... and pseudobulk sees a mean shift
    criterion = json.loads((out / "criterion.json").read_text())
    assert set(criterion["delta"]) == {*SIM.ENCODERS, "mean_log_cells"}
    assert criterion["normalization"] == {"cytoset": "none", "set_transformer": "none"}
    # determinism: a rerun gives the same design and read-outs
    first = {name: (out / name).read_bytes() for name in ("planted_genes.csv", "simulated_labels.csv", "calibration.csv", "half_split.csv")}
    run_stage(M, source)
    assert all((out / name).read_bytes() == content for name, content in first.items())
    pd.testing.assert_frame_equal(schemas.read("simulation", out / "simulation.parquet"), table)


def test_structure_only_leaves_every_pseudobulk_unchanged(tmp_path, monkeypatch):
    M, source, _ = patched(tmp_path, monkeypatch)
    d = M.design(SIM.PAIR, SIM.UNIT, source)
    cells, pseudobulk = M.simulated(SIM.PAIR, SIM.UNIT, "structure_only", source, d)     # asserts bit-identity inside
    assert all(np.array_equal(pseudobulk[s], d["pseudobulk"][s]) for s in IDS)
    assert not np.array_equal(cells.X.toarray(), M.simulated(SIM.PAIR, SIM.UNIT, "mean_shift", source, d)[0].X.toarray())
    assert not np.array_equal(cells.library_size, source.total_counts)                  # G's counts moved between cells ...
    assert cells.library_size.sum() == source.total_counts.sum()                        # ... with every sum kept
    assert dict(cells.samples["group"]) == dict(d["label"])                              # the simulated label is the group
    mean_cells, mean_profiles = M.simulated(SIM.PAIR, SIM.UNIT, "mean_shift", source, d)
    assert not all(np.array_equal(mean_profiles[s], d["pseudobulk"][s]) for s in IDS)    # the mean shift moves the sum
