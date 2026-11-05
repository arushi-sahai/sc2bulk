"""Stage 11's simulation tests, fixed with the analysis: the two-level paired bootstrap's calibration at the real designs, with sc2bulk.statistics
itself (not a re-implementation). Slow (minutes): run with SC2BULK_SIMULATION=1; the results table is
written to $SC2BULK_SIMULATION_OUT (default: stdout only).

Generative model: a scoring unit's latent z = mu * label + N(0, 1); a set encoder's
encoder seed e scores lam_e * z + a configuration-level unit effect + a seed x unit interaction, plus
read-out-seed noise; pseudobulk scores z + its own noise. The true difference is computed in closed form
(binormal AUROCs; lam_e averaged by Monte Carlo)."""
import os

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.stats import norm

from sc2bulk import statistics

pytestmark = pytest.mark.skipif(os.environ.get("SC2BULK_SIMULATION") != "1", reason="slow: SC2BULK_SIMULATION=1")

P = dict(mu=1.2, lam_sd=0.15, s_c=0.6, s_int=0.4, s_r=0.15, s_pb=0.5)
SIMS, B = 1000, 1000   # B: 0.025 x B must be an integer (statistics.lower)
RESULTS = []


def truth(p):
    lam = np.random.default_rng(0).normal(1, p["lam_sd"], 400_000)
    a = norm.cdf(lam * p["mu"] / np.sqrt(2 * (lam ** 2 + p["s_c"] ** 2 + p["s_int"] ** 2 + p["s_r"] ** 2))).mean()
    return a - norm.cdf(p["mu"] / np.sqrt(2 * (1 + p["s_pb"] ** 2 + p["s_r"] ** 2)))


def scores(rng, z, p, seeds=10, readouts=10):
    n = len(z)
    lam = rng.normal(1, p["lam_sd"], seeds)
    unit = rng.normal(0, p["s_c"], n)
    alt = np.stack([lam[e] * z + unit + rng.normal(0, p["s_int"], n) for e in range(seeds)])
    alt = alt[:, None, :] + rng.normal(0, p["s_r"], (seeds, readouts, n))
    pb = (z + rng.normal(0, p["s_pb"], n))[None, None, :] + rng.normal(0, p["s_r"], (1, readouts, n))
    return {"pseudobulk": pb, "set_encoder": alt}


def iid(n0, n1):
    def design(rng):
        label = np.r_[np.zeros(n0, int), np.ones(n1, int)]
        return [(label, label.astype(str), None, np.zeros(len(label)))]
    return design


def crossfit(n0=12, n1=12, folds=4):
    def design(rng):
        return [iid(n0, n1)(rng)[0] for _ in range(folds)]
    return design


def tms_strata(m0=15, m1=21, icc=0.3):
    def design(rng):
        label = np.tile(np.r_[np.zeros(m0, int), np.ones(m1, int)], 4)
        organ = np.repeat(np.arange(4), m0 + m1)
        mouse = np.tile(np.arange(m0 + m1), 4)
        shared = rng.normal(0, np.sqrt(icc), m0 + m1)[mouse]            # a mouse's organs share an effect
        return [(label, np.char.add(label.astype(str), organ.astype(str)), None, shared)]
    return design


def hammond(scheme, sizes0=(5,) * 5, sizes1=(5, 4, 4), icc=0.3):
    def design(rng):
        label = np.r_[np.zeros(sum(sizes0), int), np.ones(sum(sizes1), int)]
        litter = np.r_[np.repeat(np.arange(len(sizes0)), sizes0), len(sizes0) + np.repeat(np.arange(len(sizes1)), sizes1)]
        shared = rng.normal(0, np.sqrt(icc), litter.max() + 1)[litter]
        return [(label, label.astype(str), litter if scheme == "litters" else None, shared)]
    return design


def simulate(design, p=P, sims=SIMS, seed=0):
    """Per simulated dataset: (point, lower 2.5%, upper 2.5%, u, p_better) of the pair's difference."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(sims):
        per_fold = []
        for label, strata, clusters, shared in design(rng):
            z = p["mu"] * label + rng.normal(0, 1, len(label)) * np.sqrt(1 - p.get("icc", 0)) + shared
            per_fold.append(statistics.fold_bootstrap(scores(rng, z, p), label, strata, clusters, B, rng, {"pseudobulk"}))
        point, draws = statistics.folds_mean(per_fold)["set_encoder"]
        out.append((point, statistics.lower(draws, 0.025), statistics.upper(draws, 0.025), statistics.upper(draws, 0.05),
                    statistics.bootstrap_p(draws)[0], draws))
    return out


def write(row):
    """Append a results row and rewrite the table (every test writes, whatever order they run in)."""
    RESULTS.append(row)
    pd.DataFrame(RESULTS).to_csv(os.environ.get("SC2BULK_SIMULATION_OUT", os.devnull), index=False)
    print(row, flush=True)


def record(name, true, sims, check=True):
    est = np.array([s[0] for s in sims])
    cov = np.mean([(s[1] <= true <= s[2]) for s in sims])
    cov_u = np.mean([true <= s[3] for s in sims])
    cov_lower = np.mean([statistics.lower(s[5], 0.05) <= true for s in sims])   # the one-sided lower bound (p_better)
    write({"design": name, "true_delta": round(true, 4), "mean_estimate": round(est.mean(), 4),
           "coverage_95": round(cov, 3), "coverage_u": round(cov_u, 3), "coverage_lower": round(cov_lower, 3),
           "simulations": len(sims), "resamples": B, "calibration_checked": check})
    return cov, cov_u, cov_lower


@pytest.mark.parametrize("name,design", [("lupus 18 v 99", iid(18, 99)), ("human blood aging, 4 folds of 12 v 12", crossfit()),
                                         ("mouse organs, organ x age strata", tms_strata())])
def test_coverage(name, design):
    p = dict(P, icc=0.3) if "organs" in name else P
    cov, cov_u, cov_lower = record(name, truth(p), simulate(design, p))
    assert abs(cov - 0.95) <= 0.025 and abs(cov_u - 0.95) <= 0.025 and abs(cov_lower - 0.95) <= 0.025


def test_null_rejection():
    # two exchangeable set-encoder configurations: the true difference is exactly 0
    rng = np.random.default_rng(1)
    rejections = []
    for _ in range(SIMS):
        label = np.r_[np.zeros(18, int), np.ones(99, int)]
        z = P["mu"] * label + rng.normal(0, 1, len(label))
        a, b = scores(rng, z, P)["set_encoder"], scores(rng, z, P)["set_encoder"]
        draws = statistics.fold_bootstrap({"reference": b, "x": a}, label, label.astype(str), None, B, rng, set(),
                                          reference="reference")["x"][1]
        rejections.append(statistics.bootstrap_p(draws)[0] <= 0.05)
    rate = float(np.mean(rejections))
    write({"design": "null, two exchangeable encoders (lupus sizes)", "rejection_rate": rate, "simulations": SIMS,
           "resamples": B, "calibration_checked": True})
    assert abs(rate - 0.05) <= 0.02


def test_pooled_u_covers_the_headline_claim():
    # P(pooled u < true pooled delta) = 1 - u coverage: the intersection-union claim's error rate per configuration
    designs = [iid(18, 99), crossfit(), tms_strata()]
    params = [P, P, dict(P, icc=0.3)]
    runs = [simulate(d, p, seed=10 + i) for i, (d, p) in enumerate(zip(designs, params))]
    true = np.mean([truth(p) for p in params])
    pooled = [statistics.upper(np.mean([r[s][5] for r in runs], axis=0), 0.05) for s in range(SIMS)]
    cov_u = float(np.mean([true <= u for u in pooled]))
    write({"design": "pooled equal weight (3 designs)", "true_delta": round(true, 4), "coverage_u": cov_u,
           "simulations": SIMS, "resamples": B, "calibration_checked": True})
    assert cov_u >= 0.95 - 0.025


def test_headline_claim_error_rate():
    # "gains of X or more are ruled out" for a configuration whose true pooled gain is X must
    # be claimed (pooled u < X) at most 5% of the time, at the five non-ceiling pairs' real sizes
    x = 0.05
    s_pb = brentq(lambda s: truth(dict(P, s_pb=s)) - x, 0.5, 5.0)              # pseudobulk noisier: Delta = X
    p = dict(P, s_pb=s_pb)
    assert abs(truth(p) - x) < 1e-4
    designs = [iid(18, 99), crossfit(), tms_strata(), iid(113, 314), iid(30, 140)]   # lupus, blood, organs, Crohn's, ALS
    params = [p, p, dict(p, icc=0.3), p, p]
    runs = [simulate(d, q, seed=20 + i) for i, (d, q) in enumerate(zip(designs, params))]
    pooled = [statistics.upper(np.mean([r[s][5] for r in runs], axis=0), 0.05) for s in range(SIMS)]
    rate = float(np.mean([u < x for u in pooled]))
    write({"design": "headline claim at true pooled Delta = X (5 pair sizes)", "true_delta": x, "claim_rate": rate,
           "simulations": SIMS, "resamples": B, "calibration_checked": True})
    assert rate <= 0.05 + 0.02


def test_mdd_variance_matches_simulation():
    # the a priori MDD's core (amended to the binormal variance): the AUROC variance at A = 0.8
    # matches the binormal simulation's at lupus sizes (within 10%); Hanley-McNeil's approximation failed this
    rng = np.random.default_rng(3)
    a = norm.ppf(0.8) * np.sqrt(2)
    label = np.r_[np.zeros(18, int), np.ones(99, int)]
    aurocs = [statistics.auroc(rng.normal(size=len(label)) + a * label, label) for _ in range(4000)]
    simulated, formula = float(np.var(aurocs)), statistics.binormal_variance(0.8, 18, 99)
    write({"design": "MDD: binormal AUROC variance vs simulation (lupus sizes, A = 0.8)", "simulated_variance": simulated,
           "hanley_mcneil_variance": statistics.hanley_mcneil_variance(0.8, 18, 99),
           "formula_variance": formula, "calibration_checked": True})
    assert abs(simulated / formula - 1) <= 0.10


@pytest.mark.parametrize("name,change", [("seed-dominant (lam_sd 0.5)", {"lam_sd": 0.5}), ("pseudobulk near 0.9 (mu 2.2)", {"mu": 2.2})])
def test_robustness_reported(name, change):
    p = dict(P, **change)
    record(f"lupus 18 v 99, {name}", truth(p), simulate(iid(18, 99), p, seed=30), check=False)


@pytest.mark.parametrize("scheme", ["litters", "samples"])
def test_hammond_schemes_reported(scheme):
    # reported, not checked: at 5 + 3 litters neither scheme is calibrated
    p = dict(P, icc=0.3)
    record(f"mouse microglia 5 + 3 litters, {scheme}", truth(p), simulate(hammond(scheme), p), check=False)


@pytest.mark.parametrize("scheme", ["litters", "samples"])
def test_hammond_real_litters_reported(scheme):
    # reported: the blinded run's litter sizes per label group (9 and 10 litters; HAMMOND_SCHEME = litters)
    p = dict(P, icc=0.3)
    design = hammond(scheme, sizes0=(4, 3, 3, 3, 3, 3, 2, 2, 2), sizes1=(2, 2, 2, 1, 1, 1, 1, 1, 1, 1))
    record(f"mouse microglia 9 + 10 litters (blinded run), {scheme}", truth(p), simulate(design, p), check=False)
