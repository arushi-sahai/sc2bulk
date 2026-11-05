"""Stage 05v's select phase on hand-built ELBO tables, fits stubbed: merge, single best, base, K < 2."""
import importlib.util
import types

import pandas as pd

from sc2bulk.config import settings


def driver():
    spec = importlib.util.spec_from_file_location("stage05v", settings.REPO / "pipeline" / "05v_scvi_search.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(tmp_path, monkeypatch, elbo, merged_elbo=None, k=5):
    D = driver()
    monkeypatch.setattr(D, "K", k)
    monkeypatch.setattr(D, "manifest", types.SimpleNamespace(temporary=lambda p: p, finish=lambda *a, **k: None))
    monkeypatch.setattr(D, "fit_seed", lambda pair, unit, i, configs: [{"candidate": "merged", "group": "base", "config": "{}",
                                                                        "validation_elbo": merged_elbo[i], "epochs": 10}])
    for i in range(k):
        pd.DataFrame([{"candidate": c, "group": "g", "config": "{}", "validation_elbo": v[i], "epochs": 10}
                      for c, v in elbo.items()]).to_csv(tmp_path / f"elbo_s{i}.csv", index=False)
    return D.select("hammond", ("all", 0), tmp_path, {}, {})


def series(level, step=0.01):
    return [level + step * i for i in range(5)]


def test_merge_single_best_and_base(tmp_path, monkeypatch):
    base = series(100.0)
    both = {"base": base, "latent20": series(95.0), "latent30": series(99.9, 0.0), "layers2": series(96.0)}
    out = run(tmp_path, monkeypatch, both, merged_elbo=series(90.0))
    assert out["how"] == "merged" and out["frozen"] == {"n_latent": 20, "n_layers": 2}
    out = run(tmp_path, monkeypatch, both, merged_elbo=series(101.0))                 # the merge doesn't beat the base
    assert out["how"] == "best_single" and out["frozen"] == {"n_latent": 20, "n_layers": 1}
    none = {"base": base, "latent20": series(100.5), "latent30": series(101.0), "layers2": series(100.2)}
    assert run(tmp_path, monkeypatch, none)["how"] == "base"
    assert run(tmp_path, monkeypatch, none, k=1)["how"] == "base (one search seed)"
