"""Stage 05's select phase end to end on hand-built candidate tables, training stubbed (the smoke run, with
one search seed, never reaches winners, merges or the per-sample rerun)."""
import importlib.util
import json
import types

import numpy as np

from sc2bulk.config import search_space, settings


def driver():
    spec = importlib.util.spec_from_file_location("stage05", settings.REPO / "pipeline" / "05_search.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BCE = {"base": 0.60, "h128": 0.50, "h256": 0.58, "norm_per_sample": 0.45, "norm_population": 0.52, "bs32": 0.61}


def fake_trials(encoder, pair):
    base = search_space.base("cytoset")
    groups = {"h128": "h_dim", "h256": "h_dim", "norm_per_sample": "cell_normalization",
              "norm_population": "cell_normalization", "bs32": "batch_size"}
    changes = {"h128": {"h_dim": 128}, "h256": {"h_dim": 256}, "norm_per_sample": {"cell_normalization": "per_sample"},
               "norm_population": {"cell_normalization": "population"}, "bs32": {"batch_size": 32}}
    return {"base": (None, base), **{k: (groups[k], {**base, **changes[k]}) for k in changes}}


def row(name, i, bce, curve=None):
    return {"key": name, "change_group": "x", "change_value": "{}", "validation_bce": bce, "did_train": True,
            "did_train_reason": "", "d": 0.2, "d_star": 0.2, "t_star": 1.0, "p": 0.01, "testable": True,
            "epochs": 20, "kept_epoch": 15, "seconds": 1.0, "peak_gpu_bytes": 0, "validation_curve": curve or f"[{bce}]"}


def test_select_merges_and_reruns_without_per_sample(tmp_path, monkeypatch):
    D = driver()
    stub = types.SimpleNamespace(temporary=lambda p: p, finish=lambda *a, **k: None)
    monkeypatch.setattr(D, "manifest", stub)
    monkeypatch.setattr(D, "K", 5)
    monkeypatch.setattr(D, "trials", fake_trials)
    monkeypatch.setattr(D, "unit_dir", lambda pair, unit, encoder: tmp_path)
    trained = []

    def fake_train(pair, unit, encoder, i, configs, record, cost):
        (name, (_, config)), = configs.items()
        trained.append((name, i, config))
        return [row(name, i, 0.40 + 0.001 * i)]
    monkeypatch.setattr(D, "train_seed", fake_train)
    rng = np.random.default_rng(0)
    for i in range(5):
        rows = [row(k, i, v + rng.normal(0, 0.002)) for k, v in BCE.items()]
        rows[-1]["validation_curve"] = rows[0]["validation_curve"]           # bs32's curve is the base's on every seed
        D.write_candidates("hammond", ("all", 0), "cytoset", i, rows, {}, {})
    decision = D.select("hammond", ("all", 0), "cytoset", {}, {})
    assert decision["how"] == "merged" and decision["frozen"]["cell_normalization"] == "per_sample"
    assert decision["frozen"]["h_dim"] == 128 and decision["frozen"]["batch_size"] == search_space.base("cytoset")["batch_size"]
    assert decision["per_sample_route"]["cell_normalization"] == "population" and decision["per_sample_route"]["h_dim"] == 128
    assert sorted({n for n, _, _ in trained}) == ["merged", "merged_no_per_sample"] and len(trained) == 10
    assert decision["winners"]["merged_result"] == [True, True] or decision["winners"]["merged_result"] == (True, True)
    assert "batch_size" not in decision["winners"]["winners"]                     # bs32 is worse than the base
    assert decision["identical_to_base"] == ["bs32"]                              # the run-level no-op check
    assert json.loads((tmp_path / "selection.json").read_text())["how"] == "merged"
