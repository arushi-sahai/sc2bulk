"""Stage 11's primitives against closed forms and reference values."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from sc2bulk import statistics


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = rng.integers(4, 40)
        label = rng.permutation(np.r_[[0, 1], rng.integers(0, 2, n - 2)])
        score = rng.integers(0, 5, n).astype(float)          # many ties
        assert abs(statistics.auroc(score, label) - roc_auc_score(label, score)) < 1e-12


def test_auroc_weights_equal_duplication():
    rng = np.random.default_rng(1)
    score, label = rng.normal(size=12), np.r_[[0] * 6, [1] * 6]
    weight = rng.integers(0, 4, 12)
    weight[[0, 6]] = np.maximum(weight[[0, 6]], 1)            # both classes present
    expanded = statistics.auroc(np.repeat(score, weight), np.repeat(label, weight))
    assert abs(statistics.auroc(score, label, weight) - expanded) < 1e-12


def test_holm():
    adjusted = statistics.holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06, 0.02])   # sorted: .005*4, .01*3, .03*2, max(.04*1, .06)
    assert np.allclose(statistics.holm([0.5, 0.9]), [1.0, 1.0])


def test_bootstrap_p():
    better, worse = statistics.bootstrap_p([0.1, 0.2, -0.1, 0.0])
    assert better == (1 + 2) / 5 and worse == (1 + 3) / 5


def test_pooling_reference_values():
    # pinned from an external implementation: statsmodels 0.15 combine_effects(method_re="dl", use_t=True),
    # "random effect wls" (Hartung-Knapp; here q > 1, so the modified and unmodified forms agree)
    y, v = np.array([0.02, -0.06, 0.09, 0.00, 0.05]), np.array([0.001, 0.002, 0.0015, 0.0025, 0.0012])
    hk = statistics.hartung_knapp(y, v)
    assert np.isclose(hk["tau2"], 0.0012118988902589402) and np.isclose(hk["estimate"], 0.02438004058077267)
    assert np.isclose(hk["se"], 0.024174239010946577) and hk["q"] > 1
    assert np.isclose(statistics.heterogeneity(y, v)["tau2"], 0.0012118988902589402)
    # q < 1: the modified floor binds, se = (1 / sum w)^(1/2)
    v4 = np.array([0.001, 0.002, 0.0015, 0.0025])
    calm = statistics.hartung_knapp(np.array([0.01, 0.0101, 0.0099, 0.01]), v4)
    assert calm["q"] < 1 and np.isclose(calm["se"], (1 / (1 / (v4 + calm["tau2"])).sum()) ** 0.5)


FIXED = {"pseudobulk", "fm"}


def toy_scores(rng, n0=12, n1=9, seeds=10, readouts=3, shift=0.5):
    label = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    base = rng.normal(size=n0 + n1) + shift * label
    def configuration(e, extra):
        return np.stack([np.stack([base + extra * label + rng.normal(0, 0.3, n0 + n1) for _ in range(readouts)])
                         for _ in range(e)])
    return label, {"pseudobulk": configuration(1, 0.0), "a": configuration(seeds, 0.2), "b": configuration(seeds, -0.2),
                   "fm": configuration(1, 0.1)}


def test_bilinear_auroc_matches_weighted_auroc():
    rng = np.random.default_rng(0)
    label, scores = toy_scores(rng)
    s = scores["a"][3, 1]
    cases, controls = statistics.unit_weights(label, label, None, 5, rng)
    w = np.zeros((5, len(label)))
    w[:, label == 1], w[:, label == 0] = cases, controls
    M = statistics.comparison_matrix(s, label)
    for b in range(5):
        assert np.isclose(statistics.auroc_resamples(M, cases[b:b + 1], controls[b:b + 1])[0], statistics.auroc(s, label, w[b]))


def test_unit_weights_strata_and_clusters():
    rng = np.random.default_rng(1)
    label = np.array([0] * 6 + [1] * 6)
    organ = np.array(["k", "k", "k", "m", "m", "m"] * 2)
    strata = np.char.add(label.astype(str), organ)
    cases, controls = statistics.unit_weights(label, strata, None, 50, rng)
    assert (cases[:, :3].sum(axis=1) == 3).all() and (controls[:, 3:].sum(axis=1) == 3).all()   # per stratum
    litter = np.array(["l1", "l1", "l2", "l2", "l2", "l3", "l4", "l4", "l5", "l5", "l5", "l5"])
    cases, controls = statistics.unit_weights(label, label, litter, 50, rng)
    assert (controls[:, 0] == controls[:, 1]).all() and (controls[:, 2] == controls[:, 4]).all()  # litters whole
    assert set(np.unique(controls[:, 5])) <= {0, 1, 2, 3}


def test_fold_bootstrap_point_and_sharing():
    rng = np.random.default_rng(2)
    label, scores = toy_scores(rng)
    scores["a_copy"] = scores["a"].copy()
    out = statistics.fold_bootstrap(scores, label, label, None, 300, np.random.default_rng(5), FIXED)
    A = lambda s: np.mean([np.mean([statistics.auroc(s[e, r], label) for r in range(s.shape[1])]) for e in range(s.shape[0])])
    for key in ("a", "b", "fm"):
        assert np.isclose(out[key][0], A(scores[key]) - A(scores["pseudobulk"]))
    assert np.array_equal(out["a"][1], out["a_copy"][1])                       # shared seed and unit draws
    again = statistics.fold_bootstrap(scores, label, label, None, 300, np.random.default_rng(5), FIXED)
    assert np.array_equal(again["b"][1], out["b"][1])                           # reproducible
    wider = np.concatenate([scores["a"], scores["a"]])                     # 20 seeds: more uniforms drawn
    other_seeds = statistics.fold_bootstrap({**scores, "a20": wider}, label, label, None, 300, np.random.default_rng(5), FIXED)
    assert np.array_equal(other_seeds["fm"][1], out["fm"][1])               # fixed rows don't depend on seed draws
    with pytest.raises(AssertionError):                                      # one seed: not estimable
        statistics.fold_bootstrap({**scores, "a": scores["a"][:1]}, label, label, None, 10, np.random.default_rng(0), FIXED)


def test_unequal_seed_counts_share_the_coupling():
    rng = np.random.default_rng(8)
    label, scores = toy_scores(rng)
    out = statistics.fold_bootstrap({**scores, "a7": scores["a"][:7]}, label, label, None, 200, np.random.default_rng(9), FIXED)
    # replay the draws: the unit multiplicities, then one uniform per (resample, seed slot)
    rng = np.random.default_rng(9)
    cases, controls = statistics.unit_weights(label, label, None, 200, rng)
    uniform = rng.random(size=(200, 10))
    def replay(s, k):
        M = [np.mean([statistics.comparison_matrix(s[e, r], label) for r in range(s.shape[1])], axis=0) for e in range(k)]
        per_seed = np.stack([statistics.auroc_resamples(m, cases, controls) for m in M])
        return np.take_along_axis(per_seed, np.floor(uniform[:, :k] * k).astype(int).T, axis=0).mean(axis=0)
    reference = statistics.auroc_resamples(np.mean([statistics.comparison_matrix(scores["pseudobulk"][0, r], label)
                                                     for r in range(scores["pseudobulk"].shape[1])], axis=0), cases, controls)
    assert np.allclose(out["a7"][1], replay(scores["a"][:7], 7) - reference)
    assert np.allclose(out["a"][1], replay(scores["a"], 10) - reference)


def test_summarize_holm_only_for_families():
    rng = np.random.default_rng(10)
    label, scores = toy_scores(rng)
    out = statistics.fold_bootstrap(scores, label, label, None, 200, rng, FIXED)
    assert statistics.summarize(out, family=True)["p_better_holm"].notna().all()
    assert statistics.summarize(out, family=False)["p_better_holm"].isna().all()


def test_clusters_must_nest_in_strata():
    label = np.array([0, 0, 1, 1])
    with pytest.raises(AssertionError):
        statistics.unit_weights(label, label.astype(str), np.array(["l1", "l1", "l1", "l2"]), 5, np.random.default_rng(0))


def test_bounds_and_p_agree():
    rng = np.random.default_rng(3)
    for n in (500, 2000):
        for _ in range(2000):
            draws = np.round(rng.normal(rng.normal(0, 0.05), 0.03, size=n), 2)   # rounding makes ties at 0
            better, worse = statistics.bootstrap_p(draws)
            assert (better <= 0.05) == (statistics.lower(draws, 0.05) > 0)
            assert (worse <= 0.05) == (statistics.upper(draws, 0.05) < 0)


def test_mdd_reproduces_the_record_values():
    # the binormal AUROC variance at A = 0.8, rho = 0.5, one-sided 0.05, power 0.8
    v = statistics.difference_variance(n0=18, n1=99, auroc=0.8, rho=0.5)
    assert abs(statistics.mdd(v) - 0.139) < 0.002                            # lupus (binormal; Hanley-McNeil gave 0.118)
    assert abs(statistics.binormal_variance(0.8, 12, 12) - 0.008315) < 1e-5


def test_logit_row_and_pooling():
    rng = np.random.default_rng(6)
    label, scores = toy_scores(rng)
    out = statistics.fold_bootstrap(scores, label, label, None, 200, np.random.default_rng(1), FIXED, logit=True)
    logit = lambda s: np.mean([np.mean([statistics.logit_auroc(statistics.comparison_matrix(s[e, r], label).sum(),
                                                               (label == 1).sum() * (label == 0).sum())
                                        for r in range(s.shape[1])]) for e in range(s.shape[0])])
    assert np.isclose(out["a"][0], logit(scores["a"]) - logit(scores["pseudobulk"]))
    assert np.isfinite(statistics.logit_auroc(np.array([0.0, 100.0]), 100)).all()      # AUROC 0 and 1
    p1 = statistics.fold_bootstrap(scores, label, label, None, 200, np.random.default_rng(2), FIXED)
    p2 = statistics.fold_bootstrap(scores, label, label, None, 200, np.random.default_rng(3), FIXED)
    pooled = statistics.pool_equal({"x": p1, "y": p2})
    assert np.isclose(pooled["a"][0], (p1["a"][0] + p2["a"][0]) / 2)
    assert np.allclose(pooled["a"][1], (p1["a"][1] + p2["a"][1]) / 2)
    rows = statistics.pool_rows({"x": p1, "y": p2, "z": p1})
    assert set(rows["pool_set"]) == {"nonceiling_inverse_variance", "nonceiling_hartung_knapp"}
    assert statistics.at_ceiling(0.95, 0.05) and not statistics.at_ceiling(0.9499, 0.05)
    crossfit = statistics.folds_mean([p1, p2])
    assert np.isclose(crossfit["b"][0], (p1["b"][0] + p2["b"][0]) / 2)


def test_pool_rows_inverse_variance_by_hand():
    rng = np.random.default_rng(11)
    per_pair = {p: {"a": (m, rng.normal(m, s, 400))} for p, m, s in (("x", 0.02, 0.03), ("y", -0.01, 0.05), ("z", 0.04, 0.02))}
    rows = statistics.pool_rows(per_pair).set_index("pool_set")
    y = np.array([per_pair[p]["a"][0] for p in per_pair])
    w = 1 / np.array([np.var(per_pair[p]["a"][1], ddof=1) for p in per_pair])
    estimate = (w * y).sum() / w.sum()
    assert np.isclose(rows.loc["nonceiling_inverse_variance", "delta"], estimate)
    assert np.isclose(rows.loc["nonceiling_inverse_variance", "u"], estimate + 1.6448536269514722 / w.sum() ** 0.5)


def test_claims_branches():
    import importlib.util
    import pandas as pd
    from sc2bulk.config import settings
    spec = importlib.util.spec_from_file_location("stage11", settings.REPO / "pipeline" / "11_statistics.py")
    D = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(D)
    def table(u, better, per_pair_better=1.0):
        pooled = pd.DataFrame({"key": ["a", "b"], "pool_set": "nonceiling_equal", "readout": "ridge", "pair": "pooled",
                               "u": u, "p_better_holm": better, "p_worse_holm": [1.0, 1.0]})
        per = pd.DataFrame({"key": ["a"], "pool_set": "per_pair", "readout": "ridge", "pair": "lupus", "u": [0.1],
                            "p_better_holm": [per_pair_better], "p_worse_holm": [1.0]})
        return pd.concat([pooled, per])
    assert D.claims(table([0.01, 0.02], [1.0, 1.0]))["wording"].startswith("No configuration outperformed pseudobulk; pooled gains")
    at_margin = D.claims(table([0.01, 0.05], [1.0, 1.0]))                    # u = X exactly: inconclusive
    assert at_margin["inconclusive_at_margin"] == ["b"] and not at_margin["within_margin_all"]
    assert D.claims(table([0.01, 0.02], [0.01, 1.0]))["wording"].startswith("pooled better")
    named = D.claims(table([0.01, 0.02], [1.0, 1.0], per_pair_better=0.01))
    assert named["wording"].endswith("a outperformed pseudobulk in lupus")


def test_mdd_table_pooled_by_hand():
    counts = {"p": [(18, 99)], "q": [(12, 12)] * 4}
    t = statistics.mdd_table(counts, seed_sd=0.03, seeds=10).set_index(["pair", "rho", "auroc"])
    vp = statistics.difference_variance(18, 99, 0.8, 0.5, 0.03, 10)
    vq = statistics.difference_variance(12, 12, 0.8, 0.5, 0.03, 10) / 4      # the cross-fitted pair: 4 folds
    assert np.isclose(t.loc[("p", 0.5, 0.8), "mdd"], statistics.mdd(vp))
    assert np.isclose(t.loc[("q", 0.5, 0.8), "mdd"], statistics.mdd(vq))
    assert np.isclose(t.loc[("pooled K=2", 0.5, 0.8), "mdd"], statistics.mdd((vp + vq) / 4))
    assert t.loc[("p", 0.5, 0.8), "mdd_holm_first_step"] > t.loc[("p", 0.5, 0.8), "mdd"]
