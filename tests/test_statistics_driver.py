"""Stage 11's fixed-input end-to-end test: 11_statistics.main() on a fixed synthetic outputs
tree (mouse microglia at the ceiling, mouse organs with organ strata, human blood aging cross-fitted over
4 folds, lupus), through the real helpers (folds, schemas, configurations), with exact values pinned at the
analysis-fixed tag, plus a digest of the whole table and claims. Also the K = 1 and K = 0 rules.
labels.bulk_labels is patched with synthetic labels, so the label reader itself isn't covered here (its own
tests are tests/test_labels.py)."""
import hashlib
import importlib.util
import json
import types

import numpy as np
import pandas as pd

from sc2bulk import folds, labels, schemas
from sc2bulk.config import configurations, settings
from sc2bulk.config import statistics as S
from sc2bulk.config.pairs import PAIRS as PAIR_CONFIG

PAIRS = ("hammond", "tms", "soundlife", "lupus")
WRITE = schemas.write   # the real writer, for the fixture (the tests capture the stage's own writes)


def driver():
    spec = importlib.util.spec_from_file_location("stage11", settings.REPO / "pipeline" / "11_statistics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def design(pair):
    """bulk_samples rows and labels: (sample_id, donor_id, organ, group)."""
    if pair == "hammond":
        return [(f"h{i}", f"h{i}", "", i % 2) for i in range(12)]
    if pair == "tms":
        return [(f"{o}_{i}", f"{o}_{i}", o, int(i >= 5)) for o in ("Kidney", "Lung") for i in range(11)]
    if pair == "soundlife":
        return [(f"d{d}_v{v}", f"d{d}", "", d % 2) for d in range(32) for v in range(2)]
    return [(f"l{i}", f"l{i}", "", int(i >= 10)) for i in range(24)]


def build(tmp_path, at_ceiling):
    """The synthetic outputs tree: bulk_samples.csv, crossfit donor folds, stage 08 predictions."""
    rng = np.random.default_rng(2026)
    for pair in PAIRS:
        rows = design(pair)
        bulk = pd.DataFrame(rows, columns=["sample_id", "donor_id", "organ", "group"]).assign(drop_reason=np.nan)
        (tmp_path / "02_preprocess" / pair).mkdir(parents=True)
        bulk.drop(columns="group").to_csv(tmp_path / "02_preprocess" / pair / "bulk_samples.csv", index=False)
        if pair == "soundlife":
            donors = sorted(bulk["donor_id"].unique(), key=lambda d: int(d[1:]))
            (tmp_path / "03_splits" / pair).mkdir(parents=True)
            pd.DataFrame([{"fold_set": "crossfit", "fold": f, "donor_id": d, "side": "scored" if i // 8 == f else "train"}
                          for f in range(4) for i, d in enumerate(donors)]).to_csv(
                tmp_path / "03_splits" / pair / "donor_folds.csv", index=False)
        units = [u for u in folds.units(pair) if u[0] == next(iter(PAIR_CONFIG[pair]["fold_sets"]))]
        for unit in units:
            scored = folds.scored_bulk(pair, *unit, bulk)
            y = scored["group"].to_numpy()
            unit_id = scored["donor_id" if pair in ("soundlife", "lupus") else "sample_id"].to_numpy()
            for key in configurations.CONFIGURATION_ORDER:
                config = configurations.CONFIGURATIONS[key]
                fixed = key == "pseudobulk" or config["model"] in ("geneformer", "scgpt")
                strength = 9.0 if (key == "pseudobulk" and pair in at_ceiling) else 1.0 + 0.1 * len(key) % 0.5
                frames = []
                for e in ([-1] if fixed else range(10)):
                    shift = rng.normal(0, 0.2)
                    for readout in ("ridge", "lasso"):
                        for r in range(10):
                            score = strength * y + shift + rng.normal(0, 1.0, len(y))
                            frames.append(pd.DataFrame({
                                "pair": pair, "fold_set": unit[0], "fold": unit[1], "row_kind": "configuration", "key": key,
                                "model": config["model"], "input": config["input"], "encoder_seed": e, "readout": readout,
                                "readout_seed": r, "n_cells": 500 if config["input"] == "cells" else 0,
                                "pooling": "mean" if config["model"] in schemas.EMBED_ONLY else "native", "side": "bulk",
                                "sample_id": scored["sample_id"].to_numpy(), "scoring_unit_id": unit_id,
                                "score": score, "probability": np.clip(score, 0, 1)}))
                path = tmp_path / "08_readout" / pair / folds.unit_name(*unit) / f"{key}_predictions.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                WRITE("predictions", schemas.with_neutral_keys(pd.concat(frames, ignore_index=True)), path)
    return {pair: pd.DataFrame(design(pair), columns=["sample_id", "donor_id", "organ", "group"]).assign(
        label_value=lambda t: t["group"].astype(float), litter=lambda t: [f"L{i % 6}" for i in range(len(t))]
    )[["sample_id", "label_value", "group", "litter"]] for pair in PAIRS}


def run(tmp_path, monkeypatch, at_ceiling):
    monkeypatch.setattr(settings, "OUTPUTS", tmp_path)
    monkeypatch.setattr(settings, "LABELLED_OUTPUTS", tmp_path / "labelled")
    monkeypatch.setattr(settings, "BLINDED", None)
    monkeypatch.setattr(folds, "FOLDS", tmp_path / "03_splits")
    truth = build(tmp_path, at_ceiling)
    D = driver()
    written = {}
    monkeypatch.setattr(labels, "require_access", lambda: None)
    monkeypatch.setattr(labels, "bulk_labels", lambda pair: truth[pair])
    monkeypatch.setattr(S, "HAMMOND_SCHEME", "samples")
    monkeypatch.setattr(D, "PAIR_ORDER", PAIRS)
    monkeypatch.setattr(D, "assert_frozen", lambda: None)
    monkeypatch.setattr(D, "determinism", types.SimpleNamespace(configure=lambda: None))   # other tests touched CUDA already
    monkeypatch.setattr(D, "manifest", types.SimpleNamespace(
        input_hashes=lambda paths: {}, stage_code_hash=lambda *a, **k: "x", begin=lambda *a, **k: {},
        finish=lambda *a, **k: None, temporary=lambda p: p))
    monkeypatch.setattr(D.schemas, "write", lambda kind, frame, path: written.update({kind: schemas.check(kind, frame)}))
    D.main()
    claims = json.loads((settings.LABELLED_OUTPUTS / "11_statistics" / "pooled" / f"{settings.LABELLED}_claims.json").read_text())
    return written["statistics"], claims


# exact values pinned at the analysis-fixed tag: the pooled primary rows' delta and u per configuration
GOLDEN = {"cytoset_cells": [0.0642098214, 0.0930381944], "cytoset_pseudobulk": [0.0615124008, 0.0881076389],
          "geneformer_cells": [-0.005203373, 0.0329910714], "geneformer_pseudobulk": [0.0619196429, 0.0903075397],
          "scgpt_cells": [0.0317212302, 0.0659325397], "scgpt_pseudobulk": [0.0158035714, 0.0536656746],
          "scrat_cells": [0.0300560516, 0.0577425595], "scvi_cells": [0.0104424603, 0.0377385913],
          "set_transformer_cells": [0.0375649802, 0.0640823413]}
GOLDEN_DIGEST = "9a89408bf842749984f2c3abc910d2ad31e4121d7d8add12b74492983405f4e4"


def test_end_to_end_fixed_input(tmp_path, monkeypatch):
    table, claims = run(tmp_path, monkeypatch, at_ceiling={"hammond"})
    pooled = table[table["pool_set"] == "nonceiling_equal"].set_index("key")
    assert set(pooled.index) == set(configurations.CONFIGURATION_ORDER) - {"pseudobulk"}
    assert claims["excluded_at_ceiling"] == ["hammond"] and pooled["excluded_pairs"].iloc[0] == "hammond"
    assert set(table["pool_set"]) == {"per_pair", "nonceiling_equal", "nonceiling_logit", "nonceiling_inverse_variance",
                                       "nonceiling_hartung_knapp", "lupus_crohns_equal"}
    values = {k: [round(float(pooled.loc[k, "delta"]), 10), round(float(pooled.loc[k, "u"]), 10)] for k in sorted(pooled.index)}
    assert values == GOLDEN
    # the whole table (every row kind, p-values, Holm, the logit and LASSO rows) and the claims, by digest
    numeric = table.select_dtypes("number").columns
    canonical = table.sort_values(["pool_set", "pair", "readout", "statistic", "key"]).reset_index(drop=True)
    canonical[numeric] = canonical[numeric].round(10)
    digest = hashlib.sha256((canonical.to_csv(index=False) + json.dumps(claims, sort_keys=True, default=str)).encode()).hexdigest()
    assert digest == GOLDEN_DIGEST, digest
    assert claims["wording"].startswith(("No configuration", "pooled better"))


def test_single_and_no_nonceiling_pair(tmp_path, monkeypatch):
    table, claims = run(tmp_path / "one", monkeypatch, at_ceiling={"hammond", "tms", "soundlife"})
    pooled = table[table["pool_set"] == "nonceiling_equal"]
    assert "single non-ceiling pair: lupus" in pooled["excluded_pairs"].iloc[0]
    assert not table["pool_set"].isin(["nonceiling_inverse_variance", "nonceiling_hartung_knapp"]).any()
    lupus = table[(table["pool_set"] == "per_pair") & (table["pair"] == "lupus") & (table["readout"] == "ridge")].set_index("key")
    assert np.allclose(pooled.set_index("key")["delta"], lupus.loc[pooled["key"], "delta"])   # K = 1: the pair itself
    table, claims = run(tmp_path / "none", monkeypatch, at_ceiling=set(PAIRS))
    assert not table["pool_set"].isin(["nonceiling_equal", "nonceiling_logit", "lupus_crohns_equal"]).any()
    assert claims["wording"].startswith("Pooled claims not evaluable") and claims["within_margin_all"] is None
