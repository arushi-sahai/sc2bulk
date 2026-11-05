"""Stage 05s: the search-eligibility confirmation simulation. No switch: the pre-declared criteria are checked and
reported; a failure is reported before the run.

At each unit's real search-split structure (stage 03's search_split 0-4: the validation holdout groups,
their 0/1 group and their samples; single-cell labels only), simulated learners produce validation
logits, and the real rules decide: sc2bulk.harness.did_it_train per seed (d, d*, p) and
sc2bulk.search.eligible over the k = 5 seeds and over the first 3 (scale-down).

Learner model, per replicate: a holdout group's latent mean is +-m (class 1 / 0) plus a persistent
effect sqrt(rho) a_g (the same group in every seed's split) and a per-seed effect sqrt(1 - rho) e_gs;
each validation sample adds sqrt(rho) b + sqrt(1 - rho) c (sd 0.5), each bag noise (sd 0.5). Logits =
scale x the Bayes per-bag log-odds factor 2m / 1.5 x latent, so scale 1 is calibrated. m sets the
nominal group-level AUROC Phi(sqrt(2) m) (the sample effect, ignored by that formula, lowers a
single-sample group's AUROC slightly: 0.91 nominal is about 0.885). Learners: AUROC 0.91 (scale 1 and
3, on the same draws, so criterion (a) compares like with like), 0.75 (scale 1), a null (m = 0, at the
0.91 learner's scale: d* doesn't depend on scale and the permutation p is exact under the null), a
constant (each seed's logit N(0, 0.5)) and a near-constant (that plus label-independent noise, sd 0.05
per group and per bag, which the grid's T = 0.05 amplifies). rho in {0, 0.5, 0.8}; the persistent effects
are shared by every learner at a given rho. Sound Life uses crossfit fold 0: its four folds have the
same validation structure (11 v 11); split67 (11 v 10, its own search) is treated as equivalent.

Outputs, in <outputs>/05s_eligibility_simulation/:
  table.csv      unit, learner, rho, k, replicates, rule1 (the d* majority screen), rule2 (testable units:
                 the p majority), eligible, seed_pass (per-seed pass rate), temperature (median over
                 replicates of the seeds' geometric-mean T*)
  criteria.csv   the pre-declared criteria, each over every unit, rho and k (the worst
                 cell decides), with its value and pass/fail

  python pipeline/05s_eligibility_simulation.py [--replicates N] [--processes N]
"""
import argparse
import math
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import norm

from sc2bulk import determinism, folds, harness, manifest, search, seeds
from sc2bulk.config import settings
from sc2bulk.config import training as T

STAGE = "05s_eligibility_simulation"
UNITS = (("tms", ("all", 0)), ("hammond", ("all", 0)), ("pineda", ("all", 0)), ("soundlife", ("crossfit", 0)))
LEARNERS = {"auroc_0.91": (0.91, 1.0), "auroc_0.91_x3": (0.91, 3.0), "auroc_0.75": (0.75, 1.0),
            "null_learner": (0.5, 1.0), "constant": None, "near_constant": None}
# per-learner noise streams, numbered from 1: numpy seeds [a, b] and [a, b, 0] identically, and [a, b] is
# the persistent effects' stream
STREAMS = {"auroc_0.91": 1, "auroc_0.91_x3": 1, "auroc_0.75": 2, "null_learner": 3, "constant": 4, "near_constant": 5}
RHOS = (0.0, 0.5, 0.8)
SEARCH_SEEDS = 5
PURPOSE = "spare_2"   # the simulation's seed stream (spare_3: H3's thinned bulk)
SAMPLE_SD, BAG_SD = 0.5, 0.5


def structure(pair, unit) -> list:
    """Per search split 0-4: the validation side's samples (sample_id, holdout_group, group)."""
    table = pd.read_csv(settings.OUTPUTS / "03_splits" / pair / "splits.csv", dtype={"sample_id": str, "holdout_group": str})
    rows = table[(table["fold_set"] == unit[0]) & (table["fold"] == unit[1]) & (table["purpose"] == "search_split")
                 & (table["side"] == "validation")]
    return [rows[rows["index"] == i][["sample_id", "holdout_group", "group"]].reset_index(drop=True) for i in range(SEARCH_SEEDS)]


def logits_for(rng, split, learner, rho, persistent):
    """One seed's per-bag validation logits (VALIDATION_BAGS_PER_SAMPLE per sample, sample order)."""
    bags = T.VALIDATION_BAGS_PER_SAMPLE
    groups = split["holdout_group"].to_numpy()
    if learner in ("constant", "near_constant"):
        constant = np.full(len(split) * bags, rng.normal(0, 0.5))
        if learner == "constant":
            return constant
        names = sorted(set(groups))
        per_group = dict(zip(names, rng.normal(0, 0.05, len(names))))
        return constant + np.repeat([per_group[h] for h in groups], bags) + rng.normal(0, 0.05, len(constant))
    auroc, scale = LEARNERS[learner]
    m = norm.ppf(auroc) / math.sqrt(2)
    m_scale = norm.ppf(0.91) / math.sqrt(2) if learner == "null_learner" else m   # the null keeps the 0.91 learner's scale
    factor = scale * 2 * m_scale / (1 + SAMPLE_SD ** 2 + BAG_SD ** 2)
    names = sorted(set(groups))
    seed_effect = dict(zip(names, rng.normal(size=len(names))))
    group = np.array([(2 * g - 1) * m + math.sqrt(rho) * persistent["group"][h] + math.sqrt(1 - rho) * seed_effect[h]
                      for h, g in zip(groups, split["group"])])
    sample = SAMPLE_SD * np.array([math.sqrt(rho) * persistent["sample"][s] for s in split["sample_id"]]) \
        + SAMPLE_SD * math.sqrt(1 - rho) * rng.normal(size=len(split))
    latent = np.repeat(group + sample, bags) + rng.normal(0, BAG_SD, len(split) * bags)
    return factor * latent


def replicate(args):
    """One (unit, learner, rho) cell: per replicate, the rules over k = 5 and k = 3 seeds."""
    pair, unit, learner, rho, replicates = args
    splits = structure(pair, unit)
    testable = (pair, *unit) not in T.UNTESTABLE_UNITS and not settings.SMOKE   # smoke splits have < MIN_TESTABLE labellings
    all_groups = sorted(set().union(*[set(s["holdout_group"]) for s in splits]))
    all_samples = sorted(set().union(*[set(s["sample_id"]) for s in splits]))
    rows = []
    for r in range(replicates):
        base = seeds.derived(PURPOSE, pair, r, *unit)
        shared = np.random.default_rng([base, RHOS.index(rho)])                    # the same for every learner
        persistent = {"group": dict(zip(all_groups, shared.normal(size=len(all_groups)))),
                      "sample": dict(zip(all_samples, shared.normal(size=len(all_samples))))}
        rng = np.random.default_rng([base, RHOS.index(rho), STREAMS[learner]])   # x1 and x3 share it
        per_seed = []
        for i, split in enumerate(splits):
            samples = split.set_index("sample_id")[["group", "holdout_group"]]
            val_bags = [(s, None) for s in split["sample_id"] for _ in range(T.VALIDATION_BAGS_PER_SAMPLE)]
            passed, d, d_star, t_star, p, is_testable, _ = harness.did_it_train(
                logits_for(rng, split, learner, rho, persistent), val_bags, samples, seed=int(rng.integers(2 ** 31)))
            assert is_testable == testable
            per_seed.append((d_star, p, passed, t_star))
        for k in (5, 3):
            d_star, p, passed, t_star = map(np.array, zip(*per_seed[:k]))
            majority = math.ceil(k / 2)
            rows.append({"unit": f"{pair} {folds.unit_name(*unit)}", "learner": learner, "rho": rho, "k": k,
                         "rule1": (d_star > T.MARGIN_FLOOR * math.log(2)).sum() >= majority,
                         "rule2": (p <= T.NULL_ALPHA).sum() >= majority if testable else np.nan,
                         "eligible": search.eligible(d_star, p, testable), "seed_pass": passed.mean(),
                         "temperature": np.exp(np.log(t_star).mean())})
    return rows


def criteria(table: pd.DataFrame) -> pd.DataFrame:
    """The pre-declared criteria, over every unit, rho and k."""
    t = table.set_index(["unit", "learner", "rho", "k"])
    testable = [u for u in table["unit"].unique() if not table.loc[table["unit"] == u, "rule2"].isna().all()]
    gap = (t.xs("auroc_0.91", level="learner")["rule1"] - t.xs("auroc_0.91_x3", level="learner")["rule1"]).abs().max()
    good = pd.concat([t.xs("auroc_0.91", level="learner"), t.xs("auroc_0.91_x3", level="learner")])   # "the same learner"
    good_testable = good[good.index.get_level_values("unit").isin(testable)]["eligible"].min()
    good_untestable = good[~good.index.get_level_values("unit").isin(testable)]["rule1"].min()
    constant = pd.concat([t.xs(c, level="learner")["rule1"] for c in ("constant", "near_constant")]).max()
    null = t.xs("null_learner", level="learner")
    null_testable = null[null.index.get_level_values("unit").isin(testable)]["eligible"].max()
    out = [("(a) |rule 1, calibrated - x3| <= 0.05", gap, gap <= 0.05),
           ("(b) 0.91 learner eligible >= 0.80 on testable units", good_testable, good_testable >= 0.80),
           ("(b) 0.91 learner rule 1 >= 0.80 on mouse organs", good_untestable, good_untestable >= 0.80),
           ("(c) constant and near-constant learners rule 1 <= 0.01", constant, constant <= 0.01),
           ("(d) null eligible <= 0.10 on testable units", null_testable, null_testable <= 0.10)]
    return pd.DataFrame(out, columns=["criterion", "value", "passed"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    determinism.configure()
    out_dir = settings.OUTPUTS / STAGE
    record = {"stage": STAGE, "smoke": settings.SMOKE, "replicates": args.replicates, "units": UNITS, "rhos": RHOS,
              "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py"),
              "inputs": manifest.input_hashes([settings.OUTPUTS / "03_splits" / pair / "splits.csv" for pair, _ in UNITS])}
    paths_ = [out_dir / "table.csv", out_dir / "criteria.csv"]
    if manifest.is_done(paths_, record):
        print("done", flush=True)
        return
    cost = manifest.begin()
    cells = [(pair, unit, learner, rho, args.replicates) for pair, unit in UNITS for learner in LEARNERS
             for rho in (RHOS if LEARNERS[learner] else (0.0,))]
    with Pool(args.processes) as pool:
        rows = [row for cell in pool.map(replicate, cells) for row in cell]
    table = pd.DataFrame(rows).groupby(["unit", "learner", "rho", "k"], sort=True).agg(
        replicates=("eligible", "size"), rule1=("rule1", "mean"), rule2=("rule2", "mean"), eligible=("eligible", "mean"),
        seed_pass=("seed_pass", "mean"), temperature=("temperature", "median")).reset_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, frame in zip(paths_, (table, criteria(table))):
        frame.to_csv(manifest.temporary(path), index=False, float_format="%.4f")
        manifest.finish(path, record, cost, inputs=[settings.OUTPUTS / "03_splits" / pair / "splits.csv" for pair, _ in UNITS])
    print(criteria(table).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
