"""Stage 11b: the numerical
primitives, the scores, the within-bulk reference, and stage 11b's machinery on stage 11's fixed-input fixture
(tests/test_statistics_driver.py's synthetic tree; labels.bulk_labels patched with synthetic labels)."""
import hashlib
import importlib.util
import json
import types

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from sc2bulk import folds, labels, readout, reference, schemas, scoring, statistics, statistics_secondary
from sc2bulk.config import configurations, settings
from sc2bulk.config import deconvolution as deconvolution_config
from sc2bulk.config import secondary as C
from sc2bulk.config import statistics as S
from sc2bulk.config import training as T

import test_statistics_driver as fixture


def stage11b():
    spec = importlib.util.spec_from_file_location("stage11b", settings.REPO / "pipeline" / "11b_secondary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- streams ----

def test_row_streams_pinned():
    assert S.ROW_STREAMS == {
        "primary": 0, "logit": 1, "passing_seeds": 2, "lasso": 3, "out_of_fold": 4, "thinned_bulk": 5, "scvi_observed_library": 6,
        "numeric_age": 7, "alignment_combat": 8, "alignment_coral": 9, "alignment_harmony": 10, "alignment_unaligned_stacked": 11,
        "own_recipe": 12, "oracle": 13, "subgroups": 14, "age_adjusted": 15, "split67": 16, "pooled_predictions": 17,
        "best_candidate": 18, "outside_moments": 19, "female_only": 20, "readout_elasticnet": 21, "readout_pls": 22, "readout_knn": 23,
        "readout_mlp": 24, "native_head": 25, "per_sample_route": 26, "learning_curve_25": 27, "learning_curve_50": 28, "one_type": 29,
        "feature_variants": 30, "within_bulk_reference": 31, "reference_rows": 32, "learning_curve_100": 33,
        "single_cell_moments": 34}


def toy_scores(seed=0, n=24):
    rng = np.random.default_rng(seed)
    label = np.repeat([0, 1], n // 2)
    scores = {"pseudobulk": label + rng.normal(0, 1, (1, 3, n)), "a": label + rng.normal(0, 1, (4, 3, n)),
              "fm": label + rng.normal(0, 1, (1, 3, n))}
    return scores, label


def test_adding_a_fixed_configuration_keeps_the_draws():
    """Within a key, a fixed configuration (an FM line, a reference row) shifts no other configuration's draws."""
    scores, label = toy_scores()
    strata = label.astype(str)
    one = statistics.fold_bootstrap({k: scores[k] for k in ("pseudobulk", "a")}, label, strata, None, 40, np.random.default_rng(5), {"pseudobulk"})
    two = statistics.fold_bootstrap(scores, label, strata, None, 40, np.random.default_rng(5), {"pseudobulk", "fm"})
    assert np.array_equal(one["a"][1], two["a"][1])


def test_streams_differ_across_keys_and_subsets():
    from sc2bulk import seeds
    a, b = (seeds.derived("bootstrap", "tms", S.ROW_STREAMS[k], "all", 0) for k in ("thinned_bulk", "own_recipe"))
    assert a != b
    first = np.random.default_rng([a, 0]).random(5)
    assert np.array_equal(first, np.random.default_rng([a, 0]).random(5)) and not np.array_equal(first, np.random.default_rng([a, 1]).random(5))


# ---- primitives ----

def test_max_statistic_by_hand():
    draws = {"x": np.array([0.1, 0.3, 0.2, 0.0]), "y": np.array([0.0, 0.0, 0.1, 0.1]), "z": np.zeros(4)}
    results = {"x": (0.15, draws["x"]), "y": (0.05, draws["y"]), "z": (0.0, draws["z"])}
    out = statistics_secondary.max_statistic(results, ["x", "y", "z"])
    se = {k: np.std(draws[k], ddof=1) for k in ("x", "y")}
    t = max(0.15 / se["x"], 0.05 / se["y"])
    null = np.maximum((draws["x"] - 0.15) / se["x"], (draws["y"] - 0.05) / se["y"])
    assert out["excluded"] == ["z"] and np.isclose(out["t"], t) and np.isclose(out["p"], (1 + (null >= t).sum()) / 5)
    with_fm = {**results, "fm": (0.9, np.array([1.0, 0.8, 0.9, 1.0]))}   # a non-candidate never enters the max
    assert statistics_secondary.max_statistic(with_fm, ["x", "y", "z"])["p"] == out["p"]
    assert np.isnan(statistics_secondary.max_statistic(results, ["z"])["p"])   # every candidate at se 0: no test


def test_age_residuals_and_the_adjusted_auroc():
    rng = np.random.default_rng(0)
    age = rng.uniform(20, 70, 40)
    label = (np.arange(40) >= 20).astype(int)
    scores = 0.05 * age + rng.normal(0, 0.1, 40)
    residual = statistics_secondary.age_residuals(scores[None, None, :], age, label)[0, 0]
    controls = label == 0
    assert np.isclose(residual[controls].mean(), 0, atol=1e-10)
    assert np.isclose(np.polyfit(age[controls], residual[controls], 1)[0], 0, atol=1e-10)
    pure_age = 0.05 * age   # a score that is age alone has no adjusted signal; a label shift on top keeps all of it
    assert np.allclose(statistics_secondary.age_residuals(pure_age, age, label), 0, atol=1e-10)
    assert statistics_secondary.residual_auroc(pure_age + 0.5 * label, age, label, 3) == 1.0
    assert statistics_secondary.residual_auroc(pure_age, np.where(controls, 50.0, age), label, 3) is None   # one distinct control age


def test_permuted_summary_rank_decomposition():
    out = statistics_secondary.permuted_summary(np.arange(21) / 100)
    assert np.isclose(out["permuted_mean"], 0.1) and np.isclose(out["permuted_q95"], 0.19)
    table = statistics_secondary.summary_rank({"p1": {"pseudobulk": 0.0, "a": 0.1, "b": 0.1}, "p2": {"pseudobulk": 0.0, "a": -0.1, "b": 0.2}})
    assert table.loc["a", "p1"] == 1.5 and table.loc["pseudobulk", "p1"] == 3 and table.loc["a", "mean_rank"] == 2.25
    results = {"m_cells": (0.3, np.array([0.3, 0.4])), "m_pseudobulk": (0.1, np.array([0.2, 0.0]))}
    step = statistics_secondary.decomposition(results, ["m"])["m_cells"]
    assert np.isclose(step[0], 0.2) and np.allclose(step[1], [0.1, 0.4])


def test_aggregate_scores():
    table = pd.DataFrame({"key": "a", "fold": [0, 0, 0, 0, 1, 1, 1, 1], "encoder_seed": [0, 0, 1, 1] * 2, "readout_seed": [0, 1] * 4,
                          "auroc": [0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.7, 0.7]})
    out = statistics_secondary.aggregate_scores(table, ["key"]).iloc[0]
    assert np.isclose(out["auroc"], np.mean([0.75, 0.6]))   # per fold: mean over r then e; then over folds
    assert np.isclose(out["sd_encoder_seeds"], np.mean([np.std([0.65, 0.85], ddof=1), np.std([0.5, 0.7], ddof=1)]))
    assert np.isclose(out["sd_readout_seeds"], np.mean([np.std([0.7, 0.8], ddof=1), 0.0]))


def test_scores():
    rng = np.random.default_rng(1)
    label = np.repeat([0, 1], 10)
    score = label + rng.normal(0, 1, 20)
    probability = np.clip(score, 0, 1)
    out = scoring.score(score, probability, label)
    assert out["auroc"] == statistics.auroc(score, label) and out["n0"] == 10 and out["n1"] == 10
    assert np.isclose(out["auprc"], average_precision_score(label, score))
    assert np.isclose(out["brier"], np.mean((probability - label) ** 2))
    from scipy import stats
    p = stats.mannwhitneyu(score[label == 1], score[label == 0], method="asymptotic").pvalue   # two-sided, tie-corrected
    assert np.isclose(out["signed_log10_p"], np.sign(out["auroc"] - 0.5) * -np.log10(p))
    tied = np.round(score)
    assert np.isclose(scoring.average_precision(tied, label), average_precision_score(label, tied))
    p_tied = stats.mannwhitneyu(tied[label == 1], tied[label == 0], method="asymptotic").pvalue   # the tie correction
    assert np.isclose(scoring.score(tied, np.clip(tied, 0, 1), label)["signed_log10_p"],
                      np.sign(statistics.auroc(tied, label) - 0.5) * -np.log10(p_tied))
    table = pd.DataFrame({"scoring_unit_id": ["b", "a", "a"], "score": [1.0, 2.0, 4.0], "probability": [0.5, 0.0, 1.0]})
    assert scoring.unit_means(table).to_dict("list") == {"score": [3.0, 1.0], "probability": [0.5, 0.5]}


# ---- the within-bulk reference ----

def bulk_toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    donor = np.repeat([f"d{i}" for i in range(n // 2)], 2)
    label = np.repeat((np.arange(n // 2) % 2), 2)
    organ = np.where(np.arange(n) % 8 < 4, "kidney", "lung")
    X = rng.normal(size=(n, 5)) + label[:, None] * 0.8
    return X, label, donor, organ


def test_within_bulk_reference_folds_and_no_leak():
    X, label, donor, organ = bulk_toy()
    outer = reference.folds_of(donor, label, np.full(len(donor), "all"), seed=3)
    assert (pd.Series(outer).groupby(donor).nunique() == 1).all()                      # a donor never splits across folds
    by_stratum = reference.folds_of(np.arange(60).astype(str), label, organ, seed=3)    # mouse organs: samples, dealt within organ
    for o in ("kidney", "lung"):
        for g in (0, 1):
            counts = pd.Series(by_stratum[(organ == o) & (label == g)]).value_counts()
            assert counts.max() - counts.min() <= 1
    stratum = np.full(60, "all")
    score = reference.cross_fit(X, label, donor, stratum, outer, seed=3)
    flipped = label.copy()
    test = np.flatnonzero(outer == 2)
    flipped[test[donor[test] == donor[test[0]]]] ^= 1                                    # one test-fold donor's label flipped
    again = reference.cross_fit(X, flipped, donor, stratum, outer, seed=3)
    assert np.allclose(score[outer == 2], again[outer == 2])                            # its fold's predictions don't move


def test_within_bulk_reference_skips_an_empty_outer_fold_in_the_smoke_run_only(monkeypatch):
    # a smoke Sound Life unit scores 4 donors, so one of the 5 outer folds is empty
    X, label, donor, _ = bulk_toy(n=8)
    stratum = np.full(8, "all")
    outer = reference.folds_of(donor, label, stratum, seed=3)
    assert len(set(outer)) == 4
    monkeypatch.setattr(settings, "SMOKE", True)
    score = reference.cross_fit(X, label, donor, stratum, outer, seed=3)
    for f in set(outer):                                                                  # every row scored by its own fold's fit
        test = outer == f
        inner = reference.folds_of(donor[~test], label[~test], stratum[~test], seed=3)
        direct = readout.run(X[~test], label[~test], inner, None, strata_train=stratum[~test], X_scored=X[test], strata_scored=stratum[test])
        assert np.allclose(score[test], direct["scored_score"])
    monkeypatch.setattr(settings, "SMOKE", False)
    with pytest.raises(AssertionError, match="an empty outer fold"):
        reference.cross_fit(X, label, donor, stratum, outer, seed=3)


# ---- stage 11b's machinery on stage 11's fixture ----

def setup(tmp_path, monkeypatch, at_ceiling=("hammond",)):
    monkeypatch.setattr(settings, "OUTPUTS", tmp_path)
    monkeypatch.setattr(settings, "LABELLED_OUTPUTS", tmp_path / "labelled")
    monkeypatch.setattr(settings, "BLINDED", None)
    monkeypatch.setattr(folds, "FOLDS", tmp_path / "03_splits")
    monkeypatch.setattr(folds, "SAMPLES", tmp_path / "02_preprocess")
    truth = fixture.build(tmp_path, set(at_ceiling))
    monkeypatch.setattr(labels, "require_access", lambda: None)
    monkeypatch.setattr(labels, "bulk_labels", lambda pair: truth[pair])
    monkeypatch.setattr(S, "HAMMOND_SCHEME", "samples")
    M = stage11b()
    monkeypatch.setattr(M, "PAIR_ORDER", fixture.PAIRS)
    return M, truth


def test_generic_call_reproduces_stage11(tmp_path, monkeypatch):
    """The 11b machinery on stream 'primary' gives stage 11's results exactly; seeds 10-19 in a file change nothing (B2)."""
    M, _ = setup(tmp_path, monkeypatch)
    for pair in ("tms", "soundlife"):
        members = {k: M.primary(pair, k) for k in M.KEYS}
        results, n, missing = M.bootstrap(pair, M.primary_units(pair), members, "primary")
        expected = M.D11["pair_results"](pair)[0]
        assert not missing and set(results) == set(expected)
        for k in expected:
            assert results[k][0] == expected[k][0] and np.array_equal(results[k][1], expected[k][1])
    path = tmp_path / "08_readout" / "tms" / "all_0" / "cytoset_cells_predictions.parquet"
    table = schemas.read("predictions", path)
    schemas.write("predictions", pd.concat([table, table[table["encoder_seed"] < 2].assign(encoder_seed=lambda t: t["encoder_seed"] + 10)]), path)
    M.CACHE.clear()
    again = M.bootstrap("tms", M.primary_units("tms"), {k: M.primary("tms", k) for k in M.KEYS}, "primary")[0]
    assert np.array_equal(again["cytoset_cells"][1], M.D11["pair_results"]("tms")[0]["cytoset_cells"][1])


def test_stage11_recompute_matches_its_table(tmp_path, monkeypatch):
    """B3: the blinded recompute equals what a real stage 11 writes, and the real-run assert holds."""
    M, truth = setup(tmp_path, monkeypatch)
    D = fixture.driver()
    written = {}
    monkeypatch.setattr(D, "PAIR_ORDER", fixture.PAIRS)
    monkeypatch.setattr(D, "assert_frozen", lambda: None)
    monkeypatch.setattr(D, "determinism", types.SimpleNamespace(configure=lambda: None))
    monkeypatch.setattr(D, "manifest", types.SimpleNamespace(input_hashes=lambda paths: {}, stage_code_hash=lambda *a, **k: "x",
                                                             begin=lambda *a, **k: {}, finish=lambda *a, **k: None, temporary=lambda p: p))
    monkeypatch.setattr(D.schemas, "write", lambda kind, frame, path: (path.parent.mkdir(parents=True, exist_ok=True),
                                                                        fixture.WRITE(kind, frame, path), written.update({kind: frame})))
    D.main()
    notes = {}
    results, tables, pooled, kept, excluded = M.stage11(notes)
    assert excluded == ["hammond"] and kept == ["tms", "soundlife", "lupus"]
    M.assert_stage11_equal(tables, pooled, excluded)
    tables["tms"].loc[0, "delta"] += 1e-15
    with pytest.raises(AssertionError):
        M.assert_stage11_equal(tables, pooled, excluded)


def records(tmp_path, pair, unit, failing=()):
    for key, (row, model) in {"cytoset_cells": ("A", "cytoset"), "set_transformer_cells": ("A", "set_transformer"),
                              "scrat_cells": ("A", "scrat"), "cytoset_pseudobulk": ("B", "cytoset")}.items():
        directory = tmp_path / "06_train" / pair / folds.unit_name(*unit) / row / model
        directory.mkdir(parents=True, exist_ok=True)
        for e in range(10):
            (directory / f"record_s{e}.json").write_text(json.dumps({"did_train": (key, e) not in failing}))


def test_trained_label():
    M = stage11b()   # trained if at least ceil(k/2) of k seeds pass
    assert M.trained_label(5, 10) == "trained (5/10 seeds)" and M.trained_label(4, 10) == "did not train (4/10 seeds)"
    assert M.trained_label(2, 3) == "trained (2/3 seeds)" and M.trained_label(1, 3) == "did not train (1/3 seeds)"


def test_passing_seeds(tmp_path, monkeypatch):
    M, _ = setup(tmp_path, monkeypatch)
    records(tmp_path, "tms", ("all", 0), failing={("scrat_cells", e) for e in range(9)} | {("cytoset_cells", 3)})
    notes = {"seeds_passed": {}}
    frame, (results, members) = M.passing_seeds("tms", notes)
    assert notes["seeds_passed"]["tms"]["cytoset_cells"] == {"all_0": "trained (9/10 seeds)"}
    assert notes["seeds_passed"]["tms"]["scrat_cells"] == {"all_0": "did not train (1/10 seeds)"}
    assert notes["seeds_passed"]["tms"]["scvi_cells"].startswith("n/a")
    scrat = frame[frame["key"] == "scrat_cells"].iloc[0]
    assert not scrat["estimable"] and np.isnan(scrat["delta"])   # one passing seed: not estimable
    stage11 = M.D11["pair_results"]("tms")[0]
    assert results["set_transformer_cells"][0] == stage11["set_transformer_cells"][0]   # all passing: stage 11's point delta
    assert results["cytoset_cells"][0] != stage11["cytoset_cells"][0]
    assert frame["p_better_holm"].isna().all() and set(frame["row"]) == {"passing_seeds"}


def test_out_of_fold_not_estimable_and_pooling(tmp_path, monkeypatch):
    M, _ = setup(tmp_path, monkeypatch)
    for unit in M.primary_units("soundlife"):
        path = tmp_path / "08_readout" / "soundlife" / folds.unit_name(*unit) / "out_of_fold_guard.csv"
        pd.DataFrame({"encoder_seed": range(10), "passes": [unit[1] != 2 or e < 1 for e in range(10)]}).to_csv(path, index=False)
    notes = {"not_estimable": [], "not_pooled": []}
    frame, made = M.out_of_fold("soundlife", notes)
    assert made is None and not frame["estimable"].any() and len(frame) == len(M.ALTERNATIVES)   # fold 2 has one passing split
    per_pair = {"tms": ({"a": (0.1, np.zeros(40)), "b": (0.2, np.ones(40))}, {"a": {"keys": M.config_keys("cytoset_cells")},
                                                                           "b": {"keys": M.config_keys("scrat_cells")}}),
                "lupus": ({"a": (0.3, np.ones(40))}, {"a": {"keys": M.config_keys("cytoset_cells")}})}
    rows_, results = M.pooled(per_pair, ["tms", "lupus"], "thinned_bulk", False, notes)
    assert set(results) == {"a"} and np.isclose(results["a"][0], 0.2) and notes["not_pooled"]   # b missing in lupus: listed
    rows_, results = M.pooled(per_pair, ["lupus"], "thinned_bulk", False, notes)
    assert "single non-ceiling pair: lupus" in rows_[0]["excluded_pairs"].iloc[0]                # K = 1
    assert M.pooled(per_pair, [], "thinned_bulk", False, notes) == []                            # K = 0


def test_subgroups_and_age_adjusted_calls(tmp_path, monkeypatch):
    M, truth = setup(tmp_path, monkeypatch)
    notes = {"not_estimable": []}
    monkeypatch.setitem(M.C.SUBGROUP_COLUMNS, "tms", "organ")
    frames = M.subgroups("tms", notes)
    assert sorted(f["bulk_subset"].iloc[0] for f in frames) == ["organ=Kidney", "organ=Lung"] and not notes["not_estimable"]
    kidney = frames[0][frames[0]["key"] == "cytoset_cells"].iloc[0]
    assert kidney["n0"] + kidney["n1"] == 11 and kidney["p_better_holm"] != kidney["p_better_holm"]   # unadjusted (NaN)
    # the covariate adjustment through the bootstrap: an age column that is pure noise leaves the call well formed
    bulk = pd.read_csv(tmp_path / "02_preprocess" / "lupus" / "bulk_samples.csv")
    bulk.assign(age=np.linspace(20, 80, len(bulk))).to_csv(tmp_path / "02_preprocess" / "lupus" / "bulk_samples.csv", index=False)
    M.CACHE.clear()
    frame = M.age_adjusted("lupus", {"age_adjusted": {}})
    assert set(frame["bulk_subset"]) == {"age_adjusted"} and frame["estimable"].all() and np.isfinite(frame["delta"]).all()


def test_rows_schema_and_main_blinded_writes_nothing(tmp_path, monkeypatch):
    """main() end to end with the row builders reduced to the stage 11 fixture's files: a blinded run computes and
    checks, and writes nothing; a real run writes the labelled tables."""
    M, truth = setup(tmp_path, monkeypatch)
    def pair_rows(pair, references, notes):
        members = {k: M.primary(pair, k, list(range(5))) for k in M.KEYS}
        results, n, missing = M.bootstrap(pair, M.primary_units(pair), members, "learning_curve_100")
        return [M.rows(results, members, "learning_curve_100", pair, M.primary_units(pair)[0][0], n=n)], {"thinned_bulk": (results, members)}
    def scores_table(pair, references, notes):
        frame = M.bulk_frame(pair, M.primary_units(pair)[0])
        table = M.predictions(M.stage08(pair, "pseudobulk_predictions.parquet")(M.primary_units(pair)[0]))
        label = frame.groupby("scoring_unit_id")["group"].first()
        return pd.DataFrame(M.score_rows(table[table["readout_seed"] < 2], label, label))
    monkeypatch.setattr(M, "pair_rows", pair_rows)
    monkeypatch.setattr(M, "scores_table", scores_table)
    monkeypatch.setattr(M, "within_bulk", lambda pair: {})
    monkeypatch.setattr(M, "determinism", types.SimpleNamespace(configure=lambda: None))
    monkeypatch.setattr(M, "assert_stage11_equal", lambda *a: None)
    monkeypatch.setattr(M, "preflight", lambda: ["src/sc2bulk/config/triage.py"])
    finished = []
    monkeypatch.setattr(M, "manifest", types.SimpleNamespace(input_hashes=lambda paths: {}, stage_code_hash=lambda *a, **k: "x",
                                                             begin=lambda *a, **k: {}, temporary=lambda p: p,
                                                             finish=lambda path, *a, **k: finished.append(path.name)))
    monkeypatch.setattr(settings, "BLINDED", 1)
    M.main()
    assert not finished and not (tmp_path / "labelled").exists()
    monkeypatch.setattr(settings, "BLINDED", None)
    M.main()
    assert finished == [f"{settings.LABELLED}_secondary_statistics.parquet", f"{settings.LABELLED}_scores.parquet",
                        f"{settings.LABELLED}_reading_rule.csv", f"{settings.LABELLED}_secondary_notes.json"]
    table = schemas.read("secondary_statistics", tmp_path / "labelled" / "11b_secondary" / "pooled" / finished[0])
    assert {"learning_curve_100", "thinned_bulk", "decomposition", "summary_rank"} <= set(table["row"])
    assert set(table.loc[table["pair"] == "pooled", "pool_set"]) == {"nonceiling_equal", "pooled"} | \
        {f"nonceiling_equal_without_{p}" for p in ("tms", "soundlife", "lupus")}                     # D6's rows
    ranks = table[(table["row"] == "summary_rank") & (table["pair"] == "pooled")]
    assert len(ranks) == len(configurations.CONFIGURATION_ORDER) and np.isclose(ranks["delta"].sum(), sum(range(1, 11)))


# ---- the whole stage on a synthetic tree: every 08b and 09 input derived from the fixture's stage 08 files ----

def test_one_type_smoke_skipped_unit(tmp_path, monkeypatch):
    """Row 29 equals same_variant at row H's seeds; under smoke, a unit whose one_type family 08b skipped (its
    one_type marker: stage 06 skipped row H, or the one-type store falls short) leaves every configuration not
    estimable, trained or fixed (noted); stage 06's row H marker alone changes nothing here; the real run asserts."""
    from sc2bulk import training_inputs
    M, _ = setup(tmp_path, monkeypatch)
    fabricate(tmp_path, M)
    notes = {"not_estimable": []}
    expected = M.same_variant("tms", "one_type", "one_type", {"cell_subset": "one_type"}, seeds_=training_inputs.encoder_seeds("tms", "H"))[0]
    pd.testing.assert_frame_equal(M.one_type("tms", notes), expected)
    assert not notes["not_estimable"]
    stage06 = tmp_path / "06_train" / "tms" / "all_0" / "H" / "smoke_skipped.json"
    stage06.parent.mkdir(parents=True)
    stage06.write_text("{}")
    pd.testing.assert_frame_equal(M.one_type("tms", notes), expected)                               # 08b's marker decides
    marker = tmp_path / "08b_secondary" / "tms" / "all_0" / "one_type" / "smoke_skipped.json"
    marker.write_text("{}")
    with pytest.raises(AssertionError, match="smoke row one_type marker"):
        M.one_type("tms", notes)
    monkeypatch.setattr(M, "SMOKE", True)
    estimable = M.one_type("tms", notes).set_index("key")["estimable"]
    assert set(estimable.index) == set(M.ALTERNATIVES) and not estimable.any()
    assert notes["not_estimable"] == ["one_type tms: smoke skipped the one-type rows in ['all_0']"]


def test_learning_curve_smoke_skipped_unit(tmp_path, monkeypatch):
    """Rows 27, 28 bootstrap 08b's learning_curve rows at row E's seeds, row 33 stage 08's primary; under smoke, a unit
    whose row E stage 06 skipped (its marker) leaves rows 27 and 28 not estimable for every configuration (noted)
    and row 33 as it was; the real run asserts."""
    from sc2bulk import training_inputs
    M, _ = setup(tmp_path, monkeypatch)
    fabricate(tmp_path, M)
    notes = {"not_estimable": []}
    curve = training_inputs.encoder_seeds("soundlife", "E")
    units = M.primary_units("soundlife")
    expected = []
    for f, row in ((0.25, "learning_curve_25"), (0.5, "learning_curve_50")):
        members = {k: M.member(M.stage08b("soundlife", "learning_curve", k), M.config_keys(k, variant=f"training_fraction_{f}"), M.const(curve))
                   for k in M.KEYS}
        results, n, missing = M.bootstrap("soundlife", units, members, row)
        expected.append(M.rows(results, members, row, "soundlife", units[0][0], n=n, missing=missing))
    members = {k: M.primary("soundlife", k, curve) for k in M.KEYS}
    results, n, missing = M.bootstrap("soundlife", units, members, "learning_curve_100")
    expected.append(M.rows(results, members, "learning_curve_100", "soundlife", units[0][0], n=n, missing=missing))
    made = M.learning_curve("soundlife", notes)
    for got, want in zip(made, expected, strict=True):
        pd.testing.assert_frame_equal(got, want)
    assert not notes["not_estimable"] and all(m["estimable"].all() for m in made)
    marker = tmp_path / "06_train" / "soundlife" / "crossfit_0" / "E" / "smoke_skipped.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    with pytest.raises(AssertionError, match="smoke row E marker"):
        M.learning_curve("soundlife", notes)
    monkeypatch.setattr(M, "SMOKE", True)
    quarter, half, full = M.learning_curve("soundlife", notes)
    for frame, row in ((quarter, "learning_curve_25"), (half, "learning_curve_50")):
        assert set(frame["key"]) == set(M.ALTERNATIVES) and not frame["estimable"].any() and (frame["row"] == row).all()
    assert full["estimable"].all() and (full["row"] == "learning_curve_100").all()
    assert notes["not_estimable"] == ["learning_curve soundlife: smoke skipped row E in ['crossfit_0']"]


def derived(table, seeds_=None, readouts=None, **change):
    """A copy of predictions with other key columns (a fabricated 08b / 09 file), optionally restricted."""
    out = table if seeds_ is None else table[table["encoder_seed"].isin(seeds_)]
    out = out if readouts is None else out[out["readout"].isin(readouts)]
    return out.assign(**change)


def fabricate(tmp_path, M):
    """Every file stage 11b reads beyond stage 11's: 08b's families, stage 09's alignments, stage 06's records, stage
    08's guard, split67, the single-cell sample tables."""
    write = lambda frame, path: (path.parent.mkdir(parents=True, exist_ok=True), schemas.write("predictions", frame.reset_index(drop=True), path))
    for pair in fixture.PAIRS:
        samples = pd.DataFrame({"sample_id": [f"sc{i}" for i in range(8)], "donor_id": [f"sc{i}" for i in range(8)],
                                "holdout_group": [f"sc{i}" for i in range(8)], "group": [0, 1] * 4, "drop_reason": np.nan})
        samples.to_csv(tmp_path / "02_preprocess" / pair / "single_cell_samples.csv", index=False)
        for unit in M.primary_units(pair):
            name = folds.unit_name(*unit)
            primary = {k: schemas.read("predictions", tmp_path / "08_readout" / pair / name / f"{k}_predictions.parquet") for k in M.KEYS}
            out = lambda family, file: tmp_path / "08b_secondary" / pair / name / family / f"{file.replace('/', '__')}_predictions.parquet"
            trained = [k for k in M.KEYS if k not in M.NO_ENCODER_LEVEL]
            records(tmp_path, pair, unit)
            pd.DataFrame({"encoder_seed": range(10), "passes": True}).to_csv(tmp_path / "08_readout" / pair / name / "out_of_fold_guard.csv", index=False)
            for k in M.KEYS:
                t = primary[k]
                fixed = k in M.NO_ENCODER_LEVEL
                write(derived(t, variant="out_of_fold_readout") if not fixed else
                      pd.concat([t.assign(encoder_seed=e, variant="out_of_fold_readout") for e in range(10)]),
                      tmp_path / "08_readout" / pair / name / f"{k}_out_of_fold_predictions.parquet")
                write(derived(t, variant="outside_moments_scaling"), out("outside_moments", k))
                write(derived(t), out("single_cell_moments", k))                                     # variant "primary"
                for which in ("elasticnet", "pls", "knn", "mlp"):
                    write(derived(t, readout=which, readouts=["ridge"]), out("alternative_readouts", f"{k}/{which}"))
                if pair in ("hammond", "tms", "soundlife"):
                    write(derived(t, readouts=["ridge"], variant="numeric_age_readout"), out("numeric_age", k))
                if pair in ("soundlife", "lupus"):   # every configuration at encoder seeds 0-4 (the subset is indexed by e)
                    curve = pd.concat([t.assign(encoder_seed=e) for e in range(5)]) if fixed else derived(t, range(5))
                    write(pd.concat([curve.assign(variant=f"training_fraction_{f}") for f in (0.25, 0.5)]), out("learning_curve", k))
                write(derived(t, None if fixed else range(5), cell_subset="one_type"), out("one_type", k))
                perm = []
                for j in range(T.N_PERMUTED):
                    e = -1 if fixed else j % 10 if k == "scvi_cells" else j
                    perm.append(derived(t[t["encoder_seed"] == (-1 if fixed else j % 10)], readouts=["ridge"], encoder_seed=e, permutation=j))
                write(pd.concat(perm), out("permuted", k))
                write(pd.concat([derived(t, alignment=a) for a in ("combat", "coral", "harmony", "unaligned_stacked")]),
                      tmp_path / "09_alignment" / pair / name / f"{k}_predictions.parquet")
            for k in ("cytoset_cells", "set_transformer_cells", "scrat_cells", "scvi_cells"):
                write(derived(primary[k], variant="thinned_pseudocells"), out("thinned", k))
            write(derived(primary["scvi_cells"], variant="observed_library"), out("observed_library", "scvi_cells"))
            write(derived(primary["cytoset_cells"], readouts=["ridge"], readout="native_head")
                  .query("readout_seed == 0").assign(readout_seed=-1), out("native_head", "cytoset_cells"))
            write(derived(primary["cytoset_cells"], variant="per_sample_route_sensitivity"), out("per_sample_route", "cytoset"))
            for encoder in ("cytoset", "set_transformer", "scrat"):   # stage 05 chose a per-sample route for CytoSet only
                selection = tmp_path / "05_search" / pair / name / encoder / "selection.json"
                selection.parent.mkdir(parents=True, exist_ok=True)
                selection.write_text(json.dumps({"frozen": {}, **({"per_sample_route": {}} if encoder == "cytoset" else {})}))
            base = primary["cytoset_cells"]
            for encoder in ("cytoset", "set_transformer", "scrat"):
                recipe = M.candidate_keys(encoder, f"{encoder}_own_recipe", T.OWN_RECIPES[encoder].get("ncell", 500))
                write(derived(base, **{c: recipe[c] for c in ("row_kind", "key", "model", "n_cells")}), out("own_recipe", encoder))
                for i, candidate in enumerate(("base", "merged")):
                    shifted = derived(base, range(5), row_kind="candidate", key=candidate, model=encoder)
                    write(shifted.assign(score=shifted["score"] + 0.1 * i), out("search_candidates", f"{encoder}/{candidate}"))
            if pair in ("hammond", "tms", "soundlife"):
                write(derived(base, range(5), row_kind="candidate", key="objective_mse"), out("mse", "cytoset"))
            for k in (*M.SET_ENCODER_CELLS, "scvi_cells", *M.FM_CELLS, "pseudobulk"):
                t = primary[k]
                points = M.bag_points(pair) + ([(500, "all")] if k == "pseudobulk" else [])
                write(pd.concat([derived(t, None if k in M.NO_ENCODER_LEVEL else range(5), n_cells=n, cell_subset=s) for n, s in points]), out("bag_size", k))
            for k in M.FM_CELLS:
                write(derived(primary[k], pooling="max"), out("bag_size", f"{k}_max"))
            for v in ("tmm", "vst"):
                write(derived(primary["pseudobulk"], variant=v), out("feature_variants", v))
            for fs in ("panel_1000", "panel_2000", "all_shared", "mapped_orthologs"):
                write(derived(primary["pseudobulk"], feature_set=fs), out("feature_variants", fs))
            for k in ("geneformer_cells", "geneformer_pseudobulk", "scgpt_cells", "scgpt_pseudobulk"):
                write(derived(primary[k], feature_set="full_genes"), out("feature_variants", f"{k}/full_genes"))
            fabricate_reference_rows(pair, primary, out, write)
    folds_table = pd.read_csv(tmp_path / "03_splits" / "soundlife" / "donor_folds.csv")
    split = folds_table[(folds_table["fold"] == 0)].assign(fold_set="split67")
    pd.concat([folds_table, split]).to_csv(tmp_path / "03_splits" / "soundlife" / "donor_folds.csv", index=False)
    for k in M.KEYS:
        t = schemas.read("predictions", tmp_path / "08_readout" / "soundlife" / "crossfit_0" / f"{k}_predictions.parquet")
        write(t.assign(fold_set="split67", fold=0), tmp_path / "08_readout" / "soundlife" / "split67_0" / f"{k}_predictions.parquet")


def fabricate_reference_rows(pair, primary, out, write):
    """08b's reference_rows files: each row from the unit's pseudobulk (untrained CytoSet from CytoSet's
    seeds 0-9), its scores shifted by seeded noise; the known signature at readout none with probability NaN."""
    noise = np.random.default_rng(len(pair))
    def ref(table, key, seeds_=None, readouts=None, **change):
        row = configurations.REFERENCE_ROWS[key]
        made = derived(table, seeds_, readouts, **{"row_kind": "reference", "key": key, "model": row["model"], "input": row["input"],
                                                   "n_cells": 0, "pooling": "native", **change})
        return made.assign(score=made["score"] + noise.normal(0, 0.7, len(made)))
    pb = primary["pseudobulk"]
    write(ref(pb, "pseudobulk_pca64"), out("reference_rows", "pseudobulk_pca64"))
    write(ref(pb, "mean_log_cells"), out("reference_rows", "mean_log_cells"))
    write(ref(pb, "per_cell_classifier", readouts=["ridge"], n_cells=C.PER_CELL_CAP), out("reference_rows", "per_cell_classifier"))
    write(ref(primary["cytoset_cells"], "cytoset_untrained", range(10), n_cells=T.FROZEN_NCELL), out("reference_rows", "cytoset_untrained"))
    signature = pb[(pb["readout"] == "ridge") & (pb["readout_seed"] == 0)]
    write(ref(signature, "known_signature", readout="none", readout_seed=-1, probability=np.nan), out("reference_rows", "known_signature"))
    if deconvolution_config.LEVELS[pair] is not None:
        for key in ("deconvolution_clr", "deconvolution_true"):
            write(ref(pb, key), out("reference_rows", key))
    for (key, variant), pairs in C.REFERENCE_VARIANTS.items():
        if (pairs is None or pair in pairs) and (deconvolution_config.LEVELS[pair] is not None or key == "known_signature"):
            base = signature if key == "known_signature" else pb
            extra = {"readout": "none", "readout_seed": -1, "probability": np.nan} if key == "known_signature" else {}
            write(ref(base, key, variant=variant, **extra), out("reference_rows", f"{key}/{variant}"))


def pooled_digest(table) -> str:
    """The pooled rows of every row but reference_rows, label_permuted and the two rows, canonical (audit F7:
    byte-identical)."""
    frame = table[(table["pair"] == "pooled") & ~table["row"].isin(["reference_rows", "label_permuted", "single_cell_moments", "leave_one_pair_out"])]
    canonical = frame.sort_values([*schemas.KEYS, "row", "pool_set", "statistic", "bulk_subset"]).reset_index(drop=True)
    numeric = canonical.select_dtypes("number").columns
    canonical[numeric] = canonical[numeric].round(10)
    return hashlib.sha256(canonical.to_csv(index=False).encode()).hexdigest()


# the other rows' pooled rows, from stage 11b at b6e0f3a (before reference_rows was formed) on this fixture; unchanged by
# the single_cell_moments and leave_one_pair_out rows
OTHER_POOLED_DIGEST = "6d9538812aa33f1b7f32eeae0b5f2a1d380e38d25c0c40408711224d36716b28"


def test_main_on_a_synthetic_tree(tmp_path, monkeypatch):
    """Every row on the fixture's pairs: a blinded run checks everything and writes nothing; a real run writes the
    four tables and the within-bulk reference, every row in its vocabulary, the best-candidate test formed."""
    from sc2bulk import training_inputs
    M, truth = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(T, "N_PERMUTED", 3)
    fabricate(tmp_path, M)
    monkeypatch.setattr(M, "B", 200)
    monkeypatch.setattr(M.C, "SUBGROUP_COLUMNS", {"tms": "organ"})
    rng = np.random.default_rng(0)
    monkeypatch.setattr(training_inputs, "bulk_profiles", lambda pair, unit, ids: types.SimpleNamespace(
        X=__import__("scipy.sparse", fromlist=["csr_matrix"]).csr_matrix(rng.normal(size=(len(ids), 4)) + np.array(
            [truth[pair].set_index("sample_id").loc[s, "group"] for s in ids])[:, None])))
    monkeypatch.setattr(M, "determinism", types.SimpleNamespace(configure=lambda: None))
    monkeypatch.setattr(M, "assert_stage11_equal", lambda *a: None)
    monkeypatch.setattr(M, "preflight", lambda: [])
    finished = []
    monkeypatch.setattr(M, "manifest", types.SimpleNamespace(input_hashes=lambda paths: {}, stage_code_hash=lambda *a, **k: "x",
                                                             begin=lambda *a, **k: {}, temporary=lambda p: p,
                                                             finish=lambda path, *a, **k: finished.append(path.name)))
    monkeypatch.setattr(settings, "BLINDED", 1)
    M.main()
    assert not finished
    monkeypatch.setattr(settings, "BLINDED", None)
    M.main()
    root = tmp_path / "labelled" / "11b_secondary"
    table = schemas.read("secondary_statistics", root / "pooled" / f"{settings.LABELLED}_secondary_statistics.parquet")
    expected = set(S.ROW_STREAMS) - {"primary", "logit", "lasso", "female_only", "age_adjusted"} | set(M.C.UNSTREAMED_ROWS)
    assert set(table["row"]) == expected
    unadjusted = table.assign(**{c: table[c].where(table["row"] != "thinned_bulk") for c in ("p_better_holm", "p_worse_holm")})
    assert pooled_digest(unadjusted) == OTHER_POOLED_DIGEST, pooled_digest(unadjusted)          # F7: every other pooled row unchanged
                                                                                                # (thinned_bulk: only its Holm columns)
    reference = table[table["row"] == "reference_rows"]
    assert set(reference.loc[reference["pair"] == "lupus", "key"]) == {"pseudobulk_pca64", "mean_log_cells", "per_cell_classifier", "cytoset_untrained",
                                                                       "deconvolution_clr", "deconvolution_true", "known_signature"}
    assert reference["p_better_holm"].isna().all() and reference["estimable"].all()               # unadjusted
    pooled_reference = reference[reference["pair"] == "pooled"]
    assert set(pooled_reference["variant"]) == {"primary", "raw_fractions"}                      # pair-specific variants aren't pooled
    assert (pooled_reference["excluded_pairs"] == "hammond").all()                               # the ceiling pair (also no deconvolution)
    cross = table[(table["row"] == "label_permuted") & (table["pair"] == "pooled")]
    assert len(cross) == 1 and cross["key"].iloc[0] == "pseudobulk" and cross["statistic"].iloc[0] == "permuted_minus_primary_mean"
    per_pair = table[(table["row"] == "label_permuted") & (table["statistic"] == "permuted_minus_primary_mean") & (table["pair"] != "pooled")]
    assert np.isclose(cross["delta"].iloc[0], per_pair.set_index("pair").loc[["tms", "soundlife", "lupus"], "delta"].mean())
    tests_ = table[(table["row"] == "best_candidate") & (table["key"] == "best_candidate")]
    assert len(tests_) == 4 and tests_["p_better"].between(0, 1).all() and tests_["p_better_holm"].notna().all()
    assert set(table.loc[table["row"] == "oracle", "key"]) <= {"base", "merged"}
    oof = table[(table["row"] == "out_of_fold") & (table["pair"] != "pooled")]
    assert oof["p_better_holm"].notna().all()
    thinned = table[(table["row"] == "thinned_bulk") & table["estimable"]]                       # its own families, per pair and pooled
    assert "pooled" in set(thinned["pair"]) and set(thinned["key"]) <= {"cytoset_cells", "set_transformer_cells", "scrat_cells", "scvi_cells"}
    for _, family in thinned.groupby("pair"):
        assert len(family) > 1 and np.allclose(family["p_better_holm"], statistics.holm(family["p_better"]))
        assert np.allclose(family["p_worse_holm"], statistics.holm(family["p_worse"]))
    assert table.loc[table["row"] == "scvi_observed_library", "p_better_holm"].isna().all()       # the other rows stay unadjusted
    permuted = table[(table["row"] == "label_permuted") & (table["permutation"] == -1)]
    assert set(permuted.loc[permuted["key"] != "pseudobulk", "statistic"]) == {"permuted_mean", "permuted_q95"}
    assert set(permuted.loc[permuted["key"] == "pseudobulk", "statistic"]) == {"permuted_minus_primary_mean", "permuted_minus_primary_q95"}   # F12
    scores = schemas.read("scores", root / "pooled" / f"{settings.LABELLED}_scores.parquet")
    assert (scores.loc[scores["permutation"] == -1, "encoder_seed"] < 10).all() and "seed_ensemble" in set(scores["variant"])
    assert set(scores.loc[scores["row_kind"] == "reference", "key"]) == {"within_bulk_reference", *configurations.REFERENCE_ROWS}
    assert scores.loc[scores["key"] == "known_signature", "brier"].isna().all()                  # no probability scale
    rule = pd.read_csv(root / "pooled" / f"{settings.LABELLED}_reading_rule.csv")
    assert set(rule["pair"]) == {*fixture.PAIRS, "pooled"}
    # per pair and pooled, every configuration, unadjusted; its 08b files are a copy of stage 08's, so each
    # point difference is stage 11's (its own stream changes only the draws)
    moments = table[table["row"] == "single_cell_moments"]
    assert set(moments["pair"]) == {*fixture.PAIRS, "pooled"} and set(moments["key"]) == set(M.ALTERNATIVES)
    assert moments["p_better_holm"].isna().all() and (moments["variant"] == "primary").all()
    stage11 = {p: M.D11["pair_results"](p)[0] for p in fixture.PAIRS}
    for (pair, key), delta in moments[moments["pair"] != "pooled"].set_index(["pair", "key"])["delta"].items():
        assert np.isclose(delta, stage11[pair][key][0])
    # K = 3 non-ceiling pairs here, so three rows, each the equal-weight pool of the other two
    left_out = table[table["row"] == "leave_one_pair_out"]
    assert set(left_out["pool_set"]) == {f"nonceiling_equal_without_{p}" for p in ("tms", "soundlife", "lupus")}
    without_tms = left_out[left_out["pool_set"] == "nonceiling_equal_without_tms"].set_index("key")
    assert (without_tms["excluded_pairs"] == "hammond,tms").all() and without_tms["p_better_holm"].isna().all()
    for key in M.ALTERNATIVES:
        assert np.isclose(without_tms.loc[key, "delta"], (stage11["soundlife"][key][0] + stage11["lupus"][key][0]) / 2)
    assert f"{settings.LABELLED}_within_bulk_reference_predictions.parquet" in finished



def test_scores_table_skips_the_single_cell_moments_family(tmp_path, monkeypatch):
    """The scores table never reads 08b's single_cell_moments files (variant "primary", stage 08's keys), so
    its rows stay unique and stage 08's primary scores appear once."""
    M, truth = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(T, "N_PERMUTED", 3)
    fabricate(tmp_path, M)
    read = []
    original = M.predictions
    monkeypatch.setattr(M, "predictions", lambda path: read.append(path) or original(path))
    frame = M.scores_table("tms", {}, {"als_age_adjusted": {}})
    assert read and not [p for p in read if p.parent.name == "single_cell_moments"]
    assert any(p.parent.name == "outside_moments" for p in read)                                    # the other 08b families are scored
    schemas.check("scores", schemas.with_neutral_keys(frame))                                     # no duplicate keys


def test_leave_one_pair_out_rows():
    """With K >= 3 non-ceiling pairs, one row per left-out pair, each stage 11's equal-weight pool of the
    other K - 1 pairs' draws; the ceiling pairs never enter and are listed with the left-out pair; K = 2 forms none."""
    M = stage11b()
    rng = np.random.default_rng(0)
    pairs = list(M.PAIR_ORDER)
    results = {p: {k: (float(rng.normal()), rng.normal(size=40)) for k in M.ALTERNATIVES} for p in pairs}
    ceiling = [pairs[0], pairs[3]]
    for p in ceiling:   # a ceiling pair's draws would move every pooled value if they entered
        results[p] = {k: (100.0, np.full(40, 100.0)) for k in M.ALTERNATIVES}
    kept = [p for p in pairs if p not in ceiling]
    for K in (4, 3):
        notes = {"not_formed": []}
        frames = M.leave_one_pair_out_rows(results, kept[:K], notes)
        assert len(frames) == K and not notes["not_formed"]
        for left, frame in zip(kept[:K], frames):
            others = [p for p in kept[:K] if p != left]
            assert len(others) == K - 1
            expected = statistics.summarize(statistics.pool_equal({p: results[p] for p in others}), False).set_index("key")
            frame = frame.set_index("key")
            assert set(frame.index) == set(M.ALTERNATIVES) and (frame["row"] == "leave_one_pair_out").all()
            assert (frame["pair"] == "pooled").all() and (frame["pool_set"] == f"nonceiling_equal_without_{left}").all()
            assert (frame["excluded_pairs"] == ",".join(p for p in pairs if p not in others)).all()   # PAIR_ORDER order
            for c in ("delta", "ci_low", "ci_high", "u", "p_better", "p_worse"):
                assert np.allclose(frame.loc[expected.index, c], expected[c])
            assert frame["p_better_holm"].isna().all() and (frame["delta"].abs() < 50).all()        # unadjusted; no ceiling pair in it
        table = schemas.check("secondary_statistics", schemas.with_neutral_keys(pd.concat(frames, ignore_index=True)))
        assert len(table) == K * len(M.ALTERNATIVES)
    notes = {"not_formed": []}
    assert M.leave_one_pair_out_rows(results, kept[:2], notes) == [] and notes["not_formed"] == ["leave_one_pair_out: 2 non-ceiling pairs (needs 3)"]


def test_best_candidate_age_loss_cytoset_all_not_estimable(tmp_path, monkeypatch):
    """With the age loss in one Sound Life fold's frozen CytoSet configuration, CytoSet's "all" member has
    no seeds there, so the point is not estimable (NaN statistics), noted, and the max statistic runs without it."""
    M, _ = setup(tmp_path, monkeypatch)
    fabricate(tmp_path, M)
    monkeypatch.setattr(M, "B", 200)
    member = "cytoset_cells@0all_available"
    notes = {"not_estimable": [], "best_candidate": {}}
    _, results, _ = M.best_candidate("soundlife", notes)
    before = notes["best_candidate"]["soundlife"]
    assert member in results and member not in before["excluded_zero_se"] and not notes["not_estimable"]
    unit = M.primary_units("soundlife")[1]
    selection = tmp_path / "05_search" / "soundlife" / folds.unit_name(*unit) / "cytoset" / "selection.json"
    selection.write_text(json.dumps({"frozen": {"age_similarity_loss": True}}))
    frame, results, _ = M.best_candidate("soundlife", notes)
    row = frame[(frame["key"] == "cytoset_cells") & (frame["cell_subset"] == "all_available")]
    assert len(row) == 1 and not row["estimable"].iloc[0] and np.isnan(row["delta"].iloc[0])
    assert notes["not_estimable"] == ["bag_size soundlife cytoset_cells@all: age-similarity loss; whole batch above the 3,000-cell "
                                      "micro-batch cap"]
    assert member not in results and notes["best_candidate"]["soundlife"]["candidates"] == before["candidates"] - 1
    others = frame[(frame["key"] == "set_transformer_cells") & (frame["cell_subset"] == "all_available")]
    assert others["estimable"].all() and frame.loc[frame["key"] == "best_candidate", "estimable"].iloc[0]


def test_reference_rows_call(tmp_path, monkeypatch):
    """Stream 32: one call against primary pseudobulk; a fixed member added or removed leaves the
    others' draws byte-identical (N8); the readout-none reader; the known signature's point delta by hand."""
    from sc2bulk import seeds
    M, truth = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(T, "N_PERMUTED", 3)
    fabricate(tmp_path, M)
    members = M.reference_members("tms")
    assert set(members) == {"pseudobulk_pca64", "mean_log_cells", "per_cell_classifier", "deconvolution_clr", "deconvolution_true", "known_signature",
                            "cytoset_untrained", "deconvolution_clr/raw_fractions", "deconvolution_true/raw_fractions", "known_signature/pooled_z"}
    assert "deconvolution_clr" not in M.reference_members("hammond") and "known_signature/median_centered" in M.reference_members("lupus")
    frame, (results, used) = M.against_primary("tms", "reference_rows", members)
    assert set(results) == set(members) and set(frame["row"]) == {"reference_rows"} and frame["p_better_holm"].isna().all()
    unit = ("all", 0)
    array, order = M.unit_array(members["known_signature"], unit)
    assert array.shape == (1, 1, len(order))                                                    # readout none: read-out seed -1 only
    label = M.bulk_frame("tms", unit).groupby("scoring_unit_id")["group"].first().loc[order].to_numpy()
    pb, _ = M.unit_array(M.primary("tms", "pseudobulk"), unit)
    by_hand = statistics.auroc(array[0, 0], label) - np.mean([statistics.auroc(pb[0, r], label) for r in range(pb.shape[1])])
    assert np.isclose(results["known_signature"][0], by_hand, rtol=0, atol=1e-12)
    scores = {k: M.unit_array(m, unit)[0] for k, m in used.items()}
    label_, strata, clusters = M.D11["unit_design"]("tms", unit, order, M.bulk_frame("tms", unit))
    rng = np.random.default_rng(seeds.derived("bootstrap", "tms", 32, "all", 0))                 # stream 32
    direct = statistics.fold_bootstrap(scores, label_, strata, clusters, M.B, rng, {k for k in used if k != "cytoset_untrained"}, "pseudobulk")
    assert all(np.array_equal(direct[k][1], results[k][1]) for k in results)
    fewer = {k: m for k, m in members.items() if k not in ("mean_log_cells", "known_signature/pooled_z")}
    again = M.against_primary("tms", "reference_rows", fewer)[1][0]
    assert all(np.array_equal(again[k][1], results[k][1]) for k in again)                       # N8


def test_reference_rows_pooled_per_member():
    """D7 / F7: each member pooled over the non-ceiling pairs where it is defined (deconvolution: not mouse
    microglia), its own excluded pairs and K = 1 label; pair-specific variants never pooled; other rows unchanged."""
    M = stage11b()
    keys = lambda k, **change: {"keys": M.reference_keys(k.split("/")[0], **change)}
    draw = lambda v: (v, np.full(40, v))
    def made(pair, deconvolved, extra=()):
        names = ["pseudobulk_pca64", *(["deconvolution_clr", "deconvolution_clr/raw_fractions"] if deconvolved else []), *extra]
        members = {k: keys(k, **({"variant": k.split("/")[1]} if "/" in k else {})) for k in names}
        return {k: draw(0.1 * (i + 1) + len(pair) / 100) for i, k in enumerate(names)}, {**members, "pseudobulk": {"keys": M.config_keys("pseudobulk")}}
    per_pair = {"hammond": made("hammond", False), "tms": made("tms", True, ["known_signature/pooled_z"]), "lupus": made("lupus", True)}
    notes = {"not_pooled": []}
    frames, results = M.pooled(per_pair, ["hammond", "tms", "lupus"], "reference_rows", False, notes)
    table = pd.concat(frames).set_index("key")
    assert set(results) == {"pseudobulk_pca64", "deconvolution_clr", "deconvolution_clr/raw_fractions"}
    assert np.isclose(results["pseudobulk_pca64"][0], np.mean([per_pair[p][0]["pseudobulk_pca64"][0] for p in ("hammond", "tms", "lupus")]))
    assert np.isclose(results["deconvolution_clr"][0], np.mean([per_pair[p][0]["deconvolution_clr"][0] for p in ("tms", "lupus")]))
    deconvolution = table[table["variant"] == "primary"].loc["deconvolution_clr"]
    assert "hammond" in deconvolution["excluded_pairs"].split(",") and table.loc["pseudobulk_pca64", "excluded_pairs"] == ",".join(
        p for p in M.PAIR_ORDER if p not in ("hammond", "tms", "lupus"))
    assert any("known_signature/pooled_z: pair-specific" in n for n in notes["not_pooled"])
    frames, results = M.pooled(per_pair, ["hammond", "lupus"], "reference_rows", False, notes)
    one = pd.concat(frames).set_index("key")
    assert "single non-ceiling pair: lupus" in one[one["variant"] == "primary"].loc["deconvolution_clr", "excluded_pairs"]   # K = 1
    lost = {**per_pair, "lupus": ({k: v for k, v in per_pair["lupus"][0].items() if k != "deconvolution_clr"}, per_pair["lupus"][1])}
    frames, results = M.pooled(lost, ["tms", "lupus"], "reference_rows", False, notes)   # defined but not estimable in a pair
    assert "deconvolution_clr" not in results and any("deconvolution_clr not estimable" in n for n in notes["not_pooled"])
    assert M.pooled(per_pair, [], "reference_rows", False, notes) == []                     # K = 0
    other = {p: ({"a": draw(0.1 * i)}, {"a": {"keys": M.config_keys("cytoset_cells")}}) for i, p in enumerate(("tms", "lupus"))}
    frames, results = M.pooled(other, ["tms", "lupus"], "thinned_bulk", False, notes)      # another row: the common rule, one frame
    assert len(frames) == 1 and frames[0]["excluded_pairs"].iloc[0] == ",".join(p for p in M.PAIR_ORDER if p not in ("tms", "lupus"))


def test_label_permuted_pseudobulk(tmp_path, monkeypatch):
    """Fig 2A's ✕: per j, pseudobulk's permuted AUROC minus its primary AUROC (stage 11's)."""
    M, truth = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(T, "N_PERMUTED", 3)
    fabricate(tmp_path, M)
    notes = {}
    M.stage11(notes)
    primary_auroc = notes["pseudobulk_primary_auroc"]["soundlife"]
    assert primary_auroc == M.D11["pair_results"]("soundlife")[1]
    frame = M.label_permuted("soundlife", primary_auroc)
    mine = frame[(frame["key"] == "pseudobulk")].set_index(["statistic", "permutation"])["delta"]
    for j in range(3):
        per_unit = []
        for unit in M.primary_units("soundlife"):
            array, order = M.unit_array(M.member(M.stage08b("soundlife", "permuted", "pseudobulk"), M.config_keys("pseudobulk", permutation=j),
                                                 M.const([-1])), unit)
            label = M.bulk_frame("soundlife", unit).groupby("scoring_unit_id")["group"].first().loc[order].to_numpy()
            per_unit.append(np.mean([statistics.auroc(array[0, r], label) for r in range(array.shape[1])]))
        assert np.isclose(mine[("permuted_minus_primary", j)], np.mean(per_unit) - primary_auroc, rtol=0, atol=1e-12)
    deltas = [mine[("permuted_minus_primary", j)] for j in range(3)]
    assert np.isclose(mine[("permuted_minus_primary_mean", -1)], np.mean(deltas)) and np.isclose(mine[("permuted_minus_primary_q95", -1)], np.quantile(deltas, 0.95))
    assert not ({"permuted_mean", "permuted_q95"} & set(frame.loc[frame["key"] == "pseudobulk", "statistic"]))   # one meaning per name
    schemas.check("secondary_statistics", schemas.with_neutral_keys(frame))


class Started(Exception):
    pass


def test_runs_once_under_run_all(monkeypatch):
    """run_all passes --pair to every stage; 11b pools every pair, so only the last pair's call (or none) runs."""
    M = stage11b()

    def configure():
        raise Started

    monkeypatch.setattr(M, "determinism", types.SimpleNamespace(configure=configure))
    for pair in M.PAIR_ORDER[:-1]:
        M.main(["--pair", pair])   # returns without starting
    for argv in (["--pair", M.PAIR_ORDER[-1]], []):
        with pytest.raises(Started):
            M.main(argv)


def test_best_candidate_records_its_carrier(tmp_path, monkeypatch):
    """Row 18 records the candidate carrying T (Fig 4's caption: whether the selected configuration exceeded pseudobulk)."""
    M, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(T, "N_PERMUTED", 3)
    fabricate(tmp_path, M)
    monkeypatch.setattr(M, "B", 200)
    notes = {"best_candidate": {}, "not_estimable": []}
    frame, results, candidates = M.best_candidate("tms", notes)
    test = statistics_secondary.max_statistic(results, [c for c in candidates if c in results])
    carrier = notes["best_candidate"]["tms"]["carrier"]
    assert carrier in candidates and np.isclose(results[carrier][0] / test["se"][carrier], test["t"])
