"""Stage 11b's simulation tests for Fig 4's best-candidate max-statistic, with sc2bulk.statistics.fold_bootstrap and
sc2bulk.statistics_secondary.max_statistic themselves, called as pipeline/11b_secondary.py's best_candidate row
calls them. Slow: run with SC2BULK_SIMULATION=1; the results table is written to $SC2BULK_SIMULATION_OUT (default:
stdout only). Reported in the 11b tag notes; a failure is reported, and nothing switches.

Generative model: a scoring unit's latent z = mu * label + N(0, 1), shared by every configuration (the same bulk
samples). 30 candidates from 3 set encoders, 10 per encoder, at the search seeds. Candidate v of encoder m at seed e
scores z + an encoder unit effect (shared by m's candidates at every seed) + an encoder x seed x unit effect (shared
by m's candidates at seed e: candidates sharing seeds) + its own seed x unit noise, plus read-out-seed noise.
Pseudobulk (one fixed row) scores z + noise of the same total variance. Every AUROC's expectation is then
Phi(mu / sqrt(2 V)), V = 1 + the non-z variance, so each candidate's true difference vs pseudobulk is exactly 0.
mu is set so that this AUROC is 0.8, the a priori MDD table's middle row.

The two-level bootstrap counts the unit x seed interaction twice (Owen 2007), so in this model the test is
conservative; the size check is control only, size <= alpha + 0.02.
The seed-effect scenario adds a global candidate x encoder-seed shift, where the test could be liberal."""
import os

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from sc2bulk import statistics, statistics_secondary
from sc2bulk.config import settings
from sc2bulk.config import statistics as S

pytestmark = pytest.mark.skipif(os.environ.get("SC2BULK_SIMULATION") != "1", reason="slow: SC2BULK_SIMULATION=1")

P = dict(s_c=0.4, s_int=0.4, s_v=0.4, s_r=0.15)
V = 1 + P["s_c"] ** 2 + P["s_int"] ** 2 + P["s_v"] ** 2 + P["s_r"] ** 2
AUROC = 0.8
MU = np.sqrt(2 * V) * norm.ppf(AUROC)
ENCODERS, PER_ENCODER = 3, 10                  # 30 candidates
SEEDS, READOUTS = settings.N_SEARCH_SEEDS, settings.N_READOUT_SEEDS
TAU = 0.15                                     # the global seed effect: encoder-seed SD of the AUROC ~ S.SEED_SD
SIMS, B = 1000, 1000   # B: 0.025 x B must be an integer (statistics.lower)
PAIRS = ["lupus", "soundlife", "tms", "crohns"]   # lupus 18 v 99, human blood aging's 4 folds, mouse organs, Crohn's
RESULTS = []


def scores(rng, label, shift=0.0, tau=0.0):
    """One unit's scores: pseudobulk (1, readouts, units) and the candidates (seeds, readouts, units); candidate
    c0_0 gets `shift` added to its cases' mean. tau > 0: every candidate x encoder seed also shifts its cases' mean
    by c + N(0, tau), c = MU (sqrt(1 + tau^2 / 2V) - 1), so the expected AUROC stays at 0.8."""
    n = len(label)
    z = MU * label + rng.normal(0, 1, n)
    s_pb = np.sqrt(P["s_c"] ** 2 + P["s_int"] ** 2 + P["s_v"] ** 2)
    out = {"pseudobulk": (z + rng.normal(0, s_pb, n))[None, None, :] + rng.normal(0, P["s_r"], (1, READOUTS, n))}
    for m in range(ENCODERS):
        encoder = z + rng.normal(0, P["s_c"], n)
        seed = encoder + rng.normal(0, P["s_int"], (SEEDS, n))
        for v in range(PER_ENCODER):
            own = seed + rng.normal(0, P["s_v"], (SEEDS, n)) + (shift * label if (m, v) == (0, 0) else 0)
            if tau > 0:
                c = MU * (np.sqrt(1 + tau ** 2 / (2 * V)) - 1)
                own = own + (c + rng.normal(0, tau, (SEEDS, 1))) * label
            out[f"c{m}_{v}"] = own[:, None, :] + rng.normal(0, P["s_r"], (SEEDS, READOUTS, n))
    return out


def simulate(rng, counts, shift=0.0, tau=0.0):
    """One simulated pair: 11b's best_candidate call (one fold_bootstrap per fold, then the folds mean) and its
    max-statistic over the candidates."""
    per_fold = []
    for n0, n1 in counts:
        label = np.r_[np.zeros(n0, int), np.ones(n1, int)]
        per_fold.append(statistics.fold_bootstrap(scores(rng, label, shift, tau), label, label.astype(str), None, B, rng, {"pseudobulk"}))
    results = statistics.folds_mean(per_fold)
    return statistics_secondary.max_statistic(results, [k for k in results])


def a_priori_mdd(pair):
    """The pair's MDD from the a priori table (statistics.mdd_table) at rho 0.5, AUROC 0.8, the search seeds."""
    table = statistics.mdd_table({pair: S.CLASS_COUNTS[pair]}, S.SEED_SD, SEEDS)
    return float(table.query("pair == @pair and rho == 0.5 and auroc == 0.8")["mdd"].iloc[0])


def write(row):
    """Append a results row and rewrite the table (every test writes, whatever order they run in)."""
    RESULTS.append(row)
    pd.DataFrame(RESULTS).to_csv(os.environ.get("SC2BULK_SIMULATION_OUT", os.devnull), index=False)
    print(row, flush=True)


@pytest.mark.parametrize("pair", PAIRS)
def test_null_rejection(pair):
    rng = np.random.default_rng(PAIRS.index(pair))
    tests = [simulate(rng, S.CLASS_COUNTS[pair]) for _ in range(SIMS)]
    rate = float(np.mean([t["p"] <= S.ALPHA for t in tests]))
    write({"design": f"global null, {pair} {S.CLASS_COUNTS[pair]}", "candidates": ENCODERS * PER_ENCODER, "rejection_rate": rate,
           "excluded_zero_se": sum(len(t["excluded"]) for t in tests), "simulations": SIMS, "resamples": B, "calibration_checked": True})
    assert rate <= S.ALPHA + 0.02


def test_null_rejection_seed_effect():
    # the global null plus a candidate x encoder-seed shift (tau = 0.15) at mouse-organ sizes: the scenario where
    # the test could be liberal; the row reports the encoder-seed SD of the per-seed AUROC
    rng = np.random.default_rng(30)
    counts = S.CLASS_COUNTS["tms"]
    tests = [simulate(rng, counts, tau=TAU) for _ in range(SIMS)]
    rate = float(np.mean([t["p"] <= S.ALPHA for t in tests]))
    label = np.r_[np.zeros(counts[0][0], int), np.ones(counts[0][1], int)]
    seed_sd = np.mean([np.std([np.mean([statistics.auroc(s[e, r], label) for r in range(READOUTS)]) for e in range(SEEDS)], ddof=1)
                       for s in [scores(rng, label, tau=TAU)["c0_0"] for _ in range(50)]])
    write({"design": f"global null with a seed effect (tau {TAU}), tms {counts}", "candidates": ENCODERS * PER_ENCODER,
           "rejection_rate": rate, "seed_sd_auroc": round(float(seed_sd), 4), "excluded_zero_se": sum(len(t["excluded"]) for t in tests),
           "simulations": SIMS, "resamples": B, "calibration_checked": True})
    assert rate <= S.ALPHA + 0.02


@pytest.mark.parametrize("pair", PAIRS)
def test_power_at_mdd(pair):
    # one candidate's expected AUROC is 0.8 + the pair's a priori MDD (the others stay at the null)
    rng = np.random.default_rng(10 + PAIRS.index(pair))
    mdd = a_priori_mdd(pair)
    shift = np.sqrt(2 * V) * norm.ppf(AUROC + mdd) - MU
    rate = float(np.mean([simulate(rng, S.CLASS_COUNTS[pair], shift)["p"] <= S.ALPHA for _ in range(SIMS)]))
    write({"design": f"one candidate at +MDD, {pair} {S.CLASS_COUNTS[pair]}", "mdd": round(mdd, 4), "power": rate,
           "simulations": SIMS, "resamples": B, "calibration_checked": True})
    assert rate >= 0.8


def test_zero_se_candidates_are_left_out():
    # a ceiling pair: pseudobulk and candidates c0_* separate perfectly in every seed and read-out, so their draws
    # are all 0 (se 0) and they are listed; the rest are tested as if c0_* were never candidates. If every
    # candidate is at se 0, the test is not formed
    rng = np.random.default_rng(20)
    label = np.r_[np.zeros(18, int), np.ones(99, int)]
    for _ in range(20):
        s = scores(rng, label)
        s["pseudobulk"] = s["pseudobulk"] + 100 * label
        perfect = [f"c0_{v}" for v in range(PER_ENCODER)]
        for k in perfect:
            s[k] = s[k] + 100 * label
        results = statistics.fold_bootstrap(s, label, label.astype(str), None, B, rng, {"pseudobulk"})
        out = statistics_secondary.max_statistic(results, list(results))
        rest = statistics_secondary.max_statistic(results, [k for k in results if k not in perfect])
        assert out["excluded"] == sorted(perfect) and rest["excluded"] == []
        assert out["t"] == rest["t"] and out["p"] == rest["p"]
        assert np.isnan(statistics_secondary.max_statistic(results, perfect)["p"])
    write({"design": "zero-SE exclusion (lupus sizes, 10 of 30 candidates at the ceiling)", "simulations": 20, "resamples": B,
           "calibration_checked": True})
