"""Stage 11: the primary analysis.
Its code is fixed at the analysis-fixed tag together with its simulation tests (tests/simulation/test_calibration.py);
it reads bulk labels, so it runs only blinded or after that tag (sc2bulk.labels.require_access), and a blinded run
writes nothing.

Per pair: each configuration's difference in bulk AUROC from pseudobulk (the mean over encoder seeds of
the per-seed AUROC, itself the mean over read-out seeds; ridge; scores averaged within scoring units),
with the two-level paired bootstrap (encoder seeds x scoring units, stratified by label; mouse organs
within organ x age-group strata; mouse microglia by the pinned scheme; human blood, aging per fold, then
the mean over the four folds). Two-sided percentile CI, one-sided upper bound u, one-sided p-values,
Holm within the "better" and "worse" families. Pooled over the non-ceiling pairs: equal weights
(primary), inverse variance, modified Hartung-Knapp, logit scale; the lupus + Crohn's row (its own
families). The headline claims follow from the pooled rows.

Outputs, in <labelled outputs>/11_statistics/pooled/:
  labelled_statistics.parquet   statistics kind: one row per (pool set, configuration)
  labelled_claims.json          the intersection-union claim, pooled and per-pair "better" results,
                                the pairs excluded at the ceiling

  python pipeline/11_statistics.py
"""
import json
import subprocess

import numpy as np
import pandas as pd

from sc2bulk import determinism, folds, labels, manifest, paths, schemas, seeds, statistics
from sc2bulk.config import configurations, settings, smoke, triage
from sc2bulk.config import statistics as S
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

STAGE = "11_statistics"
REFERENCE = "pseudobulk"
B = S.SMOKE_BOOTSTRAP_RESAMPLES if settings.SMOKE else S.BOOTSTRAP_RESAMPLES
ALTERNATIVES = [k for k in configurations.CONFIGURATION_ORDER if k != REFERENCE and k not in triage.CUT_CONFIGURATIONS]
# the primary-analysis files the analysis-fixed tag fixes: a real run asserts they equal the tag's copies
FROZEN_FILES = ("pipeline/11_statistics.py", "src/sc2bulk/statistics.py", "src/sc2bulk/config/statistics.py",
                "tests/test_statistics.py", "tests/simulation/test_calibration.py", "tests/test_statistics_driver.py",
                "scripts/mdd_table.py")
ENCODER_SEEDS = range(smoke.N_ENCODER_SEEDS if settings.SMOKE else settings.N_ENCODER_SEEDS)   # 0-9 (10-19: attribution)
N_READOUT = smoke.N_READOUT_SEEDS if settings.SMOKE else settings.N_READOUT_SEEDS
NO_ENCODER_LEVEL = {k for k in configurations.CONFIGURATION_ORDER     # one row, encoder_seed -1
                    if k == REFERENCE or configurations.CONFIGURATIONS[k]["model"] in {"geneformer", "scgpt"}}
# the smoke run has one encoder seed: every configuration is treated as fixed there (it exercises the
# path; its numbers mean nothing and are never written: smoke runs are blinded)
FIXED = set(configurations.CONFIGURATION_ORDER) if len(ENCODER_SEEDS) < 2 else NO_ENCODER_LEVEL
PRIMARY = {"alignment": "none", "cell_subset": "all", "feature_set": "panel", "permutation": -1, "variant": "primary"}


def unit_scores(pair, unit, key, readout):
    """(scores (encoder seeds, read-out seeds, scoring units), scoring-unit order) of one configuration:
    the given read-out, every other key column at its primary value, bulk side; each scoring unit's
    score is the mean over its bulk samples."""
    table = schemas.read("predictions", paths.output("08_readout", pair, f"{key}_predictions.parquet", unit))
    config = configurations.CONFIGURATIONS[key]
    keep = (table["readout"] == readout) & (table["side"] == "bulk")
    for column, value in {**PRIMARY, "n_cells": T.FROZEN_NCELL if config["input"] == "cells" else 0,
                          "pooling": "mean" if config["model"] in schemas.EMBED_ONLY else "native"}.items():
        keep &= table[column] == value
    table = table[keep]
    seeds_used = [-1] if key in NO_ENCODER_LEVEL else list(ENCODER_SEEDS)   # encoder_seed order (the coupling is by position)
    assert set(seeds_used) <= set(table["encoder_seed"]), f"{pair} {unit} {key}: missing encoder seeds"
    table = table[table["encoder_seed"].isin(seeds_used)]
    assert not table.duplicated(["encoder_seed", "readout_seed", "sample_id"]).any(), f"{pair} {unit} {key}: duplicate rows"
    assert sorted(table["readout_seed"].unique()) == list(range(N_READOUT)), f"{pair} {unit} {key}: read-out seeds"
    mean = table.groupby(["encoder_seed", "readout_seed", "scoring_unit_id"])["score"].mean()
    units = sorted(table["scoring_unit_id"].unique())
    readouts = sorted(table["readout_seed"].unique())
    array = np.stack([np.stack([mean.loc[(e, r)].loc[units].to_numpy() for r in readouts]) for e in seeds_used])
    return array, units


def unit_design(pair, unit, units, bulk_label):
    """(0/1 label, resampling stratum, clusters) per scoring unit."""
    by_unit = bulk_label.groupby("scoring_unit_id")["group"].agg(["min", "max"])
    assert (by_unit["min"] == by_unit["max"]).all(), f"{pair}: a scoring unit with mixed labels"
    label = by_unit.loc[units, "min"].to_numpy().astype(int)
    strata = label.astype(str)
    clusters = None
    if pair == "tms":                                   # within organ x age group
        organ = bulk_label.groupby("scoring_unit_id")["organ"].first().loc[units].to_numpy()
        strata = np.char.add(strata, organ.astype(str))
    if pair == "hammond":                               # pinned before the analysis was fixed
        litters = bulk_label.groupby("group")["litter"].nunique().to_dict()   # kept exactly by the blinded permutation
        print(f"hammond: litters per label group {litters} (the T2 rule: litters if each >= 5)", flush=True)
        assert S.HAMMOND_SCHEME in ("litters", "samples"), "pin config.statistics.HAMMOND_SCHEME from the blinded run"
    if pair == "hammond" and S.HAMMOND_SCHEME == "litters":
        clusters = bulk_label.groupby("scoring_unit_id")["litter"].first().loc[units].to_numpy()
    return label, strata, clusters


def pair_results(pair, logit=False, readout="ridge", stream="primary"):
    """configuration -> (point difference, resampled differences) for the pair's primary fold set, and
    pseudobulk's AUROC (the ceiling rule's)."""
    primary = next(iter(PAIRS[pair]["fold_sets"]))
    bulk = pd.read_csv(settings.OUTPUTS / "02_preprocess" / pair / "bulk_samples.csv", dtype={"sample_id": str, "donor_id": str})
    per_fold, pb, counts = [], [], []
    for unit in [u for u in folds.units(pair) if u[0] == primary]:
        scored = folds.scored_bulk(pair, *unit, bulk[bulk["drop_reason"].isna()])
        by = "donor_id" if PAIRS[pair]["scoring_unit"] == "donor" else "sample_id"
        bulk_label = labels.bulk_labels(pair).merge(scored.assign(scoring_unit_id=scored[by]), on="sample_id")
        scores, order = {}, None
        for key in [REFERENCE, *ALTERNATIVES]:
            scores[key], units = unit_scores(pair, unit, key, readout)
            assert order is None or units == order, f"{pair} {unit}: configurations score different units"
            order = units
        assert set(order) == set(bulk_label["scoring_unit_id"]), f"{pair} {unit}: predictions don't cover the scored bulk"
        if len(ENCODER_SEEDS) < 2:   # the smoke run: one encoder seed (see FIXED)
            scores = {k: s[:1] for k, s in scores.items()}
        label, strata, clusters = unit_design(pair, unit, order, bulk_label)
        rng = np.random.default_rng(seeds.derived("bootstrap", pair, S.ROW_STREAMS[stream], *unit))
        per_fold.append(statistics.fold_bootstrap(scores, label, strata, clusters, B, rng, FIXED, REFERENCE, logit=logit))
        pb.append(np.mean([statistics.auroc(scores[REFERENCE][0, r], label) for r in range(scores[REFERENCE].shape[1])]))
        counts.append((int((label == 0).sum()), int((label == 1).sum())))
    return statistics.folds_mean(per_fold), float(np.mean(pb)), primary, tuple(np.sum(counts, axis=0))


def configuration_keys(table):
    return table.merge(pd.DataFrame([{"key": k, "model": configurations.CONFIGURATIONS[k]["model"], "input": configurations.CONFIGURATIONS[k]["input"],
                          "n_cells": T.FROZEN_NCELL if configurations.CONFIGURATIONS[k]["input"] == "cells" else 0,
                          "pooling": "mean" if configurations.CONFIGURATIONS[k]["model"] in schemas.EMBED_ONLY else "native"}
                         for k in table["key"].unique()]), on="key")


def rows(results, pool_set, pair, fold_set, statistic, family, excluded=(), readout="ridge", n=(np.nan, np.nan)):
    if family:   # the family is every configuration that produced predictions
        assert set(results) == set(ALTERNATIVES), "a Holm family must hold every configuration run"
    table = statistics.summarize(results, family)
    return configuration_keys(table).assign(pair=pair, fold_set=fold_set, fold=-1 if fold_set != "all" or pair == "pooled" else 0,
                                               row_kind="configuration", encoder_seed=-1, readout_seed=-1,
                                               pool_set=pool_set, statistic=statistic, excluded_pairs=",".join(excluded),
                                               readout=readout, n0=n[0], n1=n[1])


def inputs(pair) -> list:
    out = [*folds.inputs(pair), settings.OUTPUTS / "02_preprocess" / pair / "bulk_samples.csv",
           settings.OUTPUTS / "02_preprocess" / pair / "bulk_labels.csv", settings.OUTPUTS / "03_splits" / pair / "donor_folds.csv"]
    units = [u for u in folds.units(pair) if u[0] == next(iter(PAIRS[pair]["fold_sets"]))]
    return out + [paths.output("08_readout", pair, f"{k}_predictions.parquet", u) for u in units for k in [REFERENCE, *ALTERNATIVES]]


def claims(table) -> dict:
    """The headline rule, the intersection-union claim and the pre-declared wording; with
    no non-ceiling pair, the pooled claims are not evaluable."""
    pooled = table[table["pool_set"] == "nonceiling_equal"]
    per = table[(table["pool_set"] == "per_pair") & (table["readout"] == "ridge")]
    if pooled.empty:
        named = sorted(map(list, per.loc[per["p_better_holm"] <= S.ALPHA, ["pair", "key"]].to_numpy()))
        wording = "Pooled claims not evaluable: pseudobulk at the ceiling in every pair" + "".join(
            f"; {k} outperformed pseudobulk in {p}" for p, k in named)
        return {"no_configuration_outperformed": None, "within_margin_all": None, "per_pair_better": named,
                "wording": wording, "configurations": ALTERNATIVES, "cut": list(triage.CUT_CONFIGURATIONS), "resamples": B}
    better = sorted(pooled.loc[pooled["p_better_holm"] <= S.ALPHA, "key"])
    inconclusive = sorted(pooled.loc[pooled["u"] >= S.MARGIN, "key"])
    per_pair_better = sorted(map(list, per.loc[per["p_better_holm"] <= S.ALPHA, ["pair", "key"]].to_numpy()))
    if better:
        wording = "pooled better: '[configuration] outperformed pseudobulk by delta (95% CI)'"
    elif not inconclusive:
        wording = "No configuration outperformed pseudobulk; pooled gains of X or more are ruled out for every configuration"
    else:
        wording = "No configuration outperformed pseudobulk; gains above u are ruled out (inconclusive at X: listed)"
    if per_pair_better:   # named in the headline sentence itself
        wording += "; " + "; ".join(f"{k} outperformed pseudobulk in {p}" for p, k in per_pair_better)
    return {"no_configuration_outperformed": not better, "within_margin_all": not inconclusive,
            "inconclusive_at_margin": inconclusive, "pooled_u": dict(zip(pooled["key"], pooled["u"])),
            "pooled_better": better, "pooled_worse": sorted(pooled.loc[pooled["p_worse_holm"] <= S.ALPHA, "key"]),
            "per_pair_better": per_pair_better,
            "wording": wording, "configurations": ALTERNATIVES, "cut": list(triage.CUT_CONFIGURATIONS), "resamples": B}


def assert_frozen():
    """On a real (unblinded) run, the primary-analysis files equal the analysis-fixed tag's."""
    tag = json.loads(labels.ANALYSIS_FIXED.read_text())["tag"]
    changed = subprocess.run(["git", "diff", "--quiet", tag, "--", *FROZEN_FILES], cwd=settings.REPO).returncode
    assert changed == 0, f"the primary-analysis files differ from the analysis-fixed tag {tag}"


def main():
    determinism.configure()
    labels.require_access()
    if settings.BLINDED is None:
        assert_frozen()
    record = {"stage": STAGE, "smoke": settings.SMOKE, "resamples": B,
              "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py"),
              "inputs": manifest.input_hashes([f for pair in PAIR_ORDER for f in inputs(pair)])}
    cost = manifest.begin()
    per_pair, logit_pair, lasso_pair, ceiling, out = {}, {}, {}, {}, []
    for pair in PAIR_ORDER:
        per_pair[pair], pb, fold_set, n = pair_results(pair)
        logit_pair[pair] = pair_results(pair, logit=True, stream="logit")[0]
        lasso_pair[pair] = pair_results(pair, readout="lasso", stream="lasso")[0]
        ceiling[pair] = statistics.at_ceiling(pb, S.MARGIN)
        out.append(rows(per_pair[pair], "per_pair", pair, fold_set, "auroc_difference", True, n=n))
        out.append(rows(lasso_pair[pair], "per_pair", pair, fold_set, "auroc_difference", False, readout="lasso", n=n))
    kept = [p for p in PAIR_ORDER if not ceiling[p]]
    excluded = [p for p in PAIR_ORDER if ceiling[p]]
    extra = pd.DataFrame()
    if kept:   # K = 1 -> the pooled rows are that pair (labelled); K = 0 -> no pooled rows
        label = excluded + ([f"single non-ceiling pair: {kept[0]}"] if len(kept) == 1 else [])
        out.append(rows(statistics.pool_equal({p: per_pair[p] for p in kept}), "nonceiling_equal", "pooled", "all",
                        "auroc_difference", True, label))
        out.append(rows(statistics.pool_equal({p: logit_pair[p] for p in kept}), "nonceiling_logit", "pooled", "all",
                        "logit_auroc_difference", False, label))
    if len(kept) >= 2:   # inverse variance and Hartung-Knapp need K >= 2
        extra = configuration_keys(statistics.pool_rows({p: per_pair[p] for p in kept})).assign(
            pair="pooled", fold_set="all", fold=-1, row_kind="configuration", encoder_seed=-1, readout_seed=-1,
            statistic="auroc_difference", excluded_pairs=",".join(excluded), readout="ridge",
            **{c: np.nan for c in ("ci_low", "ci_high", "p_better", "p_worse", "p_better_holm", "p_worse_holm", "mdd_obs",
                                   "resamples", "n0", "n1")})   # model-based rows: estimate and u only
    disease = [p for p in ("lupus", "crohns") if p in ceiling and not ceiling[p]]
    if disease:
        out.append(rows(statistics.pool_equal({p: per_pair[p] for p in disease}), "lupus_crohns_equal", "pooled", "all",
                        "auroc_difference", True, [p for p in ("lupus", "crohns") if ceiling.get(p)]))
    table = pd.concat(out, ignore_index=True)
    claim = {**claims(table), "excluded_at_ceiling": excluded, "lupus_crohns_formed": bool(disease),
             "lupus_crohns_note": "" if disease else "not formed: lupus and Crohn's both at the ceiling (frozen §42 T7)"}
    full = schemas.with_neutral_keys(pd.concat([table, extra], ignore_index=True))
    schemas.check("statistics", full)   # the write path's check, exercised by blinded runs too
    if settings.BLINDED is not None:   # a blinded run produces no statistics
        print("blinded run: stage 11 computed and checked, outputs suppressed", flush=True)
        return
    path = paths.output(STAGE, "pooled", f"{settings.LABELLED}_statistics.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    schemas.write("statistics", full, manifest.temporary(path))
    manifest.finish(path, record, cost, inputs=[f for pair in PAIR_ORDER for f in inputs(pair)])
    claims_path = path.with_name(f"{settings.LABELLED}_claims.json")
    manifest.temporary(claims_path).write_text(json.dumps(claim, indent=1, default=str))
    manifest.finish(claims_path, record, cost, inputs=[f for pair in PAIR_ORDER for f in inputs(pair)])


if __name__ == "__main__":
    main()
