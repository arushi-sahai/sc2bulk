"""Stage 11b: the secondary statistics. It reads bulk labels, so it runs only blinded
or after the analysis-fixed tag (labels.require_access); a blinded run computes and checks everything and writes nothing.

Stage 11 is frozen: sc2bulk.statistics and config/statistics.py are imported unedited, and stage 11's functions
are loaded from pipeline/11_statistics.py (runpy; codehash.extra adds that file to this stage's hash). A blinded
stage 11 writes nothing, so 11b recomputes stage 11's per-pair rows and ceiling set with stage 11's own functions
and stream; a real run asserts they equal stage 11's written table exactly. A real run also asserts
stage 11's frozen files (assert_frozen), allows only config/triage.py to differ from the analysis-fixed tag across
stage 11's import closure (every difference is written to the run record), and asserts 11b's own files equal that
tag's (config/secondary.SECONDARY_FILES).

Bootstrapped rows: one ROW_STREAMS key = one statistics.fold_bootstrap call per (pair, unit), from
seeds.derived("bootstrap", pair, ROW_STREAMS[key], fold_set, fold); the configurations of a call share its draws
(subgroups: subset k takes default_rng([seed, k]); pooled predictions: one call over Sound Life's folds at
("crossfit", 0)). Human blood, aging: per fold, then statistics.folds_mean. Encoder seeds 0-9 only, or the row's own rule (E, F, H: 0-4; the search: the search seeds); fixed representations carry -1. A
configuration with fewer than 2 seeds in a row is not estimable (written with NaN statistics); a cross-fitted pair
needs every fold estimable. Only the out-of-fold row and the thinned-bulk row (a key
secondary) are Holm-adjusted, each within its own "better" and "worse" families, per pair and pooled;
the best-candidate test carries its own max-statistic adjustment (per pair, unadjusted across pairs; the Holm
value across pairs is given beside it in p_better_holm); every other row is unadjusted. Rows defined on every
pair also get an equal-weight pooled row over the non-ceiling pairs (stage 11's ceiling set and
K = 1 / K = 0 rules); a configuration missing in a pooled pair is not pooled, and is listed. reference_rows pools each
member over the non-ceiling pairs where it is defined (deconvolution: not mouse microglia), with its own excluded
pairs; a pair-specific sensitivity variant is not pooled.

  passing_seeds (2), out_of_fold (4), thinned_bulk (5), scvi_observed_library (6), numeric_age (7), alignment_*
  (8-11), own_recipe (12), oracle (13), subgroups (14), age_adjusted (15: Crohn's covariate-adjusted AUROC), split67 (16), pooled_predictions (17), best_candidate (18), outside_moments (19), readout_* (21-24),
  native_head (25), per_sample_route (26), learning_curve_25/50/100 (27, 28, 33), one_type (29), feature_variants
  (30), within_bulk_reference (31), reference_rows (32: Fig 2A's reference rows and their sensitivity variants against
  primary pseudobulk, unadjusted),
  single_cell_moments (34: 08b's single_cell_moments family against pseudobulk under the same scaling, unadjusted, per
  pair and pooled). female_only (20) is reserved and never drawn.

Rows without a stream: label_permuted (delta_j per permutation, their mean and 95th percentile; for pseudobulk, Fig
2A's ✕: its permuted AUROC minus its primary AUROC, per pair and a descriptive pooled mean), summary_rank,
decomposition (Fig 2C: stage 11's primary draws replayed), leave_one_pair_out (with K >= 3 non-ceiling pairs, stage
11's equal-weight pooled row without each non-ceiling pair in turn, from its primary draws; descriptive, unadjusted), the reading rule (reading_rule.changes, per pair and pooled), the scores table (scoring.score: AUROC,
AUPRC, Brier, signed -log10 p per unit, configuration, seed, side; seed ensembles; ALS's age-adjusted within-single-cell
AUROC; not the single_cell_moments family, reported as differences only, whose variant "primary" would repeat stage 08's
keys) and the within-bulk reference predictions.

Outputs, in <labelled outputs>/11b_secondary/:
  pooled/labelled_secondary_statistics.parquet   secondary_statistics kind
  pooled/labelled_scores.parquet                  scores kind
  pooled/labelled_reading_rule.csv                the out-of-fold reading rule, per pair and pooled
  pooled/labelled_secondary_notes.json            listed and skipped rows, "trained (k/10 seeds)" or "did not train (k/10 seeds)", kept counts
  <pair>/<unit>/labelled_within_bulk_reference_predictions.parquet   reference_predictions kind

  python pipeline/11b_secondary.py [--pair PAIR]   (11b pools every pair, so it runs once: without --pair, or with
                                                 the last pair in PAIR_ORDER; run_all passes each pair in turn and the
                                                 others are skipped)
"""
import argparse
import json
import runpy
import subprocess

import numpy as np
import pandas as pd

from sc2bulk import (codehash, determinism, folds, labels, manifest, paths, reading_rule, reference, schemas, scoring, seeds,
                     statistics, statistics_secondary, training_inputs)
from sc2bulk.config import configurations, representations, search_space, settings
from sc2bulk.config import deconvolution as deconvolution_config
from sc2bulk.config import secondary as C
from sc2bulk.config import statistics as S
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

STAGE = "11b_secondary"
D11 = runpy.run_path(str(settings.REPO / "pipeline" / "11_statistics.py"), run_name="stage11")   # its functions, not its main
REFERENCE = D11["REFERENCE"]
B = D11["B"]
ENCODER_SEEDS = list(D11["ENCODER_SEEDS"])     # 0-9
N_READOUT = D11["N_READOUT"]
ALTERNATIVES = D11["ALTERNATIVES"]              # without the triage cuts (removed everywhere)
KEYS = [REFERENCE, *ALTERNATIVES]
NO_ENCODER_LEVEL = D11["NO_ENCODER_LEVEL"]
SMOKE = len(ENCODER_SEEDS) < 2                  # stage 11's rule: one encoder seed, every configuration fixed
MIN_SEEDS = 1 if SMOKE else 2
STATS = ("delta", "ci_low", "ci_high", "u", "p_better", "p_worse", "p_better_holm", "p_worse_holm")
NEUTRAL = {c: v for c, (_, v, _) in schemas.KEYS.items() if v is not None and c not in ("encoder_seed", "readout_seed")}
SET_ENCODER_CELLS = ("cytoset_cells", "set_transformer_cells", "scrat_cells")
FM_CELLS = ("geneformer_cells", "scgpt_cells")
HOLM_ROWS = ("out_of_fold", "thinned_bulk")         # each Holm within its own families
POOLED_ROWS = ("passing_seeds", "out_of_fold", "thinned_bulk", "scvi_observed_library", "alignment_combat", "alignment_coral",
               "alignment_harmony", "alignment_unaligned_stacked", "own_recipe", "outside_moments", "readout_elasticnet", "readout_pls",
               "readout_knn", "readout_mlp", "native_head", "feature_variants", "within_bulk_reference",
               "reference_rows", "single_cell_moments")   # per_sample_route: per pair only
CACHE, READ = {}, set()


# ---- members: one configuration's predictions in a row ----

def config_keys(key, **change) -> dict:
    config = configurations.CONFIGURATIONS[key]
    return {**NEUTRAL, "row_kind": "configuration", "key": key, "model": config["model"], "input": config["input"],
            "n_cells": T.FROZEN_NCELL if config["input"] == "cells" else 0,
            "pooling": "mean" if config["model"] in schemas.EMBED_ONLY else "native", **change}


def reference_keys(key, **change) -> dict:
    """A reference row's key columns: model and input from configurations.REFERENCE_ROWS, n_cells 0 unless
    changed, pooling native."""
    row = configurations.REFERENCE_ROWS[key]
    return {**NEUTRAL, "row_kind": "reference", "key": key, "model": row["model"], "input": row["input"], "n_cells": 0, "pooling": "native", **change}


def candidate_keys(encoder, name, n_cells=T.FROZEN_NCELL) -> dict:
    return {**NEUTRAL, "row_kind": "candidate", "key": name, "model": encoder, "input": "cells", "n_cells": n_cells, "pooling": "native"}


def predictions(path) -> pd.DataFrame:
    if path not in CACHE:
        READ.add(path)
        CACHE[path] = schemas.read("predictions", path)
    return CACHE[path]


def member(path_of, keys, seeds_of) -> dict:
    """A row's configuration: its file per unit, the key columns it is filtered on (and written with), and its
    encoder seeds per unit ([-1]: fixed)."""
    return {"load": lambda unit: predictions(path_of(unit)), "keys": keys, "seeds": seeds_of}


def const(value):
    return lambda unit: value


def seeds_for(key, seeds_) -> list:
    return [-1] if key in NO_ENCODER_LEVEL else list(seeds_)


def stage08(pair, name):
    return lambda unit: paths.output("08_readout", pair, name, unit)


def stage08b(pair, family, name):
    return lambda unit: paths.output("08b_secondary", pair, family, unit) / f"{name.replace('/', '__')}_predictions.parquet"


def primary(pair, key, seeds_=None) -> dict:
    """Stage 08's primary predictions of a configuration (encoder seeds 0-9, or the given seeds)."""
    return member(stage08(pair, f"{key}_predictions.parquet"), config_keys(key), const(seeds_for(key, ENCODER_SEEDS if seeds_ is None else seeds_)))


def secondary(pair, family, key, seeds_=ENCODER_SEEDS, **change) -> dict:
    return member(stage08b(pair, family, key), config_keys(key, **change), const(seeds_for(key, seeds_)))


def unit_array(m, unit, samples=None) -> tuple:
    """(scores (encoder seeds, read-out seeds, scoring units), scoring-unit order) of one member in one unit, as
    stage 11's unit_scores: bulk side, the member's key columns, its seeds in order, scores averaged within
    scoring units (only the given samples' rows, when given)."""
    table = m["load"](unit)
    keep = table["side"] == "bulk"
    for column, value in m["keys"].items():
        keep &= table[column] == value
    table = table[keep]
    seeds_used = m["seeds"](unit)
    assert set(seeds_used) <= set(table["encoder_seed"]), f"{unit} {m['keys']['key']}: missing encoder seeds"
    table = table[table["encoder_seed"].isin(seeds_used)]
    if samples is not None:
        table = table[table["sample_id"].isin(samples)]
    assert not table.duplicated(["encoder_seed", "readout_seed", "sample_id"]).any(), f"{unit} {m['keys']['key']}: duplicate rows"
    readouts = [-1] if m["keys"]["readout"] in ("native_head", "none") else list(range(N_READOUT))
    assert sorted(table["readout_seed"].unique()) == readouts, f"{unit} {m['keys']['key']}: read-out seeds"
    mean = table.groupby(["encoder_seed", "readout_seed", "scoring_unit_id"])["score"].mean()
    units = sorted(table["scoring_unit_id"].unique())
    return np.stack([np.stack([mean.loc[(e, r)].loc[units].to_numpy() for r in readouts]) for e in seeds_used]), units


def primary_units(pair) -> list:
    return [u for u in folds.units(pair) if u[0] == next(iter(PAIRS[pair]["fold_sets"]))]


def bulk_frame(pair, unit) -> pd.DataFrame:
    """The unit's scored bulk samples with their labels (and the pair's label columns), label-free metadata and
    scoring unit, as stage 11's pair_results builds them."""
    if ("bulk", pair, unit) not in CACHE:
        bulk = pd.read_csv(settings.OUTPUTS / "02_preprocess" / pair / "bulk_samples.csv", dtype={"sample_id": str, "donor_id": str})
        scored = folds.scored_bulk(pair, *unit, bulk[bulk["drop_reason"].isna()])
        by = "donor_id" if PAIRS[pair]["scoring_unit"] == "donor" else "sample_id"
        CACHE[("bulk", pair, unit)] = labels.bulk_labels(pair).merge(scored.assign(scoring_unit_id=scored[by]), on="sample_id")
    return CACHE[("bulk", pair, unit)]


def counts(frame) -> tuple:
    by_unit = frame.groupby("scoring_unit_id")["group"].first()
    return int((by_unit == 0).sum()), int((by_unit == 1).sum())


# ---- the bootstrap ----

def bootstrap(pair, units, members, stream, select=None, sub=None, transform=None, reference_name=REFERENCE) -> tuple:
    """One fold_bootstrap call per unit over the members estimable in every unit (folds mean over cross-fitting
    folds). select: frame -> the bulk samples kept (a subset row); transform: (scores, frame, order) -> scores (the
    age adjustment). Returns (results, (n0, n1), members not estimable)."""
    estimable = [k for k, m in members.items() if all(m["seeds"](u) == [-1] or len(m["seeds"](u)) >= MIN_SEEDS for u in units)]
    assert reference_name in estimable, f"{pair} {stream}: the comparator isn't estimable"
    per_fold, n = [], []
    for unit in units:
        frame = bulk_frame(pair, unit)
        frame = frame[select(frame)] if select is not None else frame
        scores, order = {}, None
        for k in estimable:
            scores[k], got = unit_array(members[k], unit, set(frame["sample_id"]))
            assert order is None or got == order, f"{pair} {unit} {stream}: configurations score different units"
            order = got
        assert set(order) == set(frame["scoring_unit_id"]), f"{pair} {unit} {stream}: predictions don't cover the scored bulk"
        if transform is not None:
            scores = transform(scores, frame, order)
        if SMOKE:
            scores = {k: s[:1] for k, s in scores.items()}
        label, strata, clusters = D11["unit_design"](pair, unit, order, frame)
        seed = seeds.derived("bootstrap", pair, S.ROW_STREAMS[stream], *unit)
        rng = np.random.default_rng(seed if sub is None else [seed, sub])
        fixed = set(scores) if SMOKE else {k for k in scores if members[k]["seeds"](unit) == [-1]}
        per_fold.append(statistics.fold_bootstrap(scores, label, strata, clusters, B, rng, fixed, reference_name))
        n.append(counts(frame))
    return statistics.folds_mean(per_fold), tuple(int(x) for x in np.sum(n, axis=0)), [k for k in members if k not in estimable]


def rows(results, members, row, pair, fold_set, family=False, bulk_subset="all", n=(np.nan, np.nan), excluded="", missing=(),
         pool_set=None, statistic="auroc_difference", change=None) -> pd.DataFrame:
    """The secondary_statistics rows of one call: summarize()'s statistics with each member's key columns; members
    not estimable carry NaN statistics and estimable False."""
    table = statistics.summarize(results, family) if results else pd.DataFrame(columns=["key", *STATS, "mdd_obs", "resamples"])
    out = [{**members[r["key"]]["keys"], **{c: r[c] for c in (*STATS, "mdd_obs", "resamples")}, "estimable": True}
           for r in table.to_dict("records")]
    out += [{**members[k]["keys"], **{c: np.nan for c in (*STATS, "mdd_obs")}, "resamples": B, "estimable": False} for k in missing]
    frame = pd.DataFrame(out)
    if frame.empty:
        return frame
    return frame.assign(**(change or {}), pair=pair, fold_set=fold_set, fold=-1 if fold_set == "crossfit" or pair == "pooled" else 0,
                        encoder_seed=-1, readout_seed=-1, row=row, pool_set=pool_set or ("per_pair" if pair != "pooled" else "nonceiling_equal"),
                        statistic=statistic, bulk_subset=bulk_subset, n0=n[0], n1=n[1], excluded_pairs=excluded)


# ---- the rows ----

def record_file(pair, unit, key, e):
    row, model = ("B", "cytoset") if key == "cytoset_pseudobulk" else ("A", configurations.CONFIGURATIONS[key]["model"])
    return paths.output("06_train", pair, row, unit) / model / f"record_s{e}.json"


def passing(pair, unit, key) -> list:
    """Encoder seeds 0-9 that passed "did it train?" (stage 06's TrainingRecord)."""
    out = []
    for e in ENCODER_SEEDS:
        path = record_file(pair, unit, key, e)
        READ.add(path)
        out += [e] if json.loads(path.read_text())["did_train"] else []
    return out


def trained_label(passed: int, k: int) -> str:
    """The configuration-level label: "trained" if at least ceil(k/2) of its k seeds pass."""
    return f"trained ({passed}/{k} seeds)" if passed >= (k + 1) // 2 else f"did not train ({passed}/{k} seeds)"


def passing_seeds(pair, notes) -> tuple:
    """Row 2: stage 08's primary predictions, trained configurations restricted to their passing seeds; scVI (no
    did-it-train rule: "n/a (unsupervised; ELBO finite)") and the fixed representations keep theirs.
    The notes carry each configuration's label per unit (trained_label)."""
    members = {k: primary(pair, k) for k in KEYS}
    counted = {}
    for k in KEYS:
        if k not in NO_ENCODER_LEVEL and k != "scvi_cells":
            kept = {u: passing(pair, u, k) for u in primary_units(pair)}
            members[k] = {**members[k], "seeds": lambda unit, kept=kept: kept[unit]}
            counted[k] = {folds.unit_name(*u): trained_label(len(s), len(ENCODER_SEEDS)) for u, s in kept.items()}
    counted["scvi_cells"] = "n/a (unsupervised; ELBO finite)"
    notes["seeds_passed"][pair] = counted
    results, n, missing = bootstrap(pair, primary_units(pair), members, "passing_seeds")
    return rows(results, members, "passing_seeds", pair, primary_units(pair)[0][0], n=n, missing=missing), (results, members)


def guard_passes(pair, unit) -> list:
    path = paths.output("08_readout", pair, "out_of_fold_guard.csv", unit)
    READ.add(path)
    table = pd.read_csv(path)
    return sorted(int(e) for e in table.loc[table["passes"] & (table["encoder_seed"] < len(ENCODER_SEEDS)), "encoder_seed"])


def out_of_fold(pair, notes) -> tuple:
    """Row 4: stage 08's out-of-fold predictions; every configuration (pseudobulk too) at the
    unit's passing splits e, so the seeds pair by position; any unit with fewer than 2 makes the pair not
    estimable. Holm within the row's own families."""
    units = primary_units(pair)
    kept = {u: guard_passes(pair, u) for u in units}
    members = {k: member(stage08(pair, f"{k}_out_of_fold_predictions.parquet"), config_keys(k, variant="out_of_fold_readout"),
                         lambda unit: kept[unit]) for k in KEYS}
    assert all(m["seeds"](u) == members[REFERENCE]["seeds"](u) for m in members.values() for u in units)   # same e
    if any(len(kept[u]) < MIN_SEEDS for u in units):
        notes["not_estimable"].append(f"out_of_fold {pair}: a unit with fewer than {MIN_SEEDS} passing splits")
        return rows({}, members, "out_of_fold", pair, units[0][0], missing=ALTERNATIVES), None
    results, n, missing = bootstrap(pair, units, members, "out_of_fold")
    return rows(results, members, "out_of_fold", pair, units[0][0], family=True, n=n), (results, members)


def against_primary(pair, row, members, family=False) -> tuple:
    """A call whose comparator is stage 08's primary pseudobulk (schemas_protocol_the exception, C9)."""
    members = {**members, REFERENCE: primary(pair, REFERENCE)}
    results, n, missing = bootstrap(pair, primary_units(pair), members, row)
    return rows(results, members, row, pair, primary_units(pair)[0][0], family=family, n=n, missing=missing), (results, members)


def same_variant(pair, row, family, change, keys=None, seeds_=ENCODER_SEEDS) -> tuple:
    """A call over configurations and their pseudobulk comparator under the same change (one 08b family)."""
    members = {k: secondary(pair, family, k, seeds_, **change) for k in (keys or KEYS)}
    results, n, missing = bootstrap(pair, primary_units(pair), members, row)
    return rows(results, members, row, pair, primary_units(pair)[0][0], n=n, missing=missing), (results, members)


def alignment(pair, which) -> tuple:
    """Rows 8-11: stage 09's predictions under one alignment, pseudobulk aligned the same way."""
    members = {k: member(lambda unit, k=k: paths.output("09_alignment", pair, f"{k}_predictions.parquet", unit), config_keys(k, alignment=which),
                         const(seeds_for(k, ENCODER_SEEDS))) for k in KEYS}
    results, n, missing = bootstrap(pair, primary_units(pair), members, f"alignment_{which}")
    return rows(results, members, f"alignment_{which}", pair, primary_units(pair)[0][0], n=n, missing=missing), (results, members)


def per_sample_route_triggered(pair, unit, encoder) -> bool:
    """Whether stage 05 adopted per_sample and so chose a per-sample route (08b reads the same)."""
    path = paths.output("05_search", pair, encoder, unit) / "selection.json"
    READ.add(path)
    return "per_sample_route" in json.loads(path.read_text())


def search_members(pair, notes) -> dict:
    """Every stage 05 search candidate 08b read out ("<encoder>/<candidate>"), at the search seeds; for a cross-fitted
    pair, the candidates present in every fold (others listed)."""
    found = None
    for unit in primary_units(pair):
        names = {p.name.removesuffix("_predictions.parquet").replace("__", "/")
                 for p in (paths.output("08b_secondary", pair, "search_candidates", unit)).glob("*_predictions.parquet")}
        if found is not None and names != found:
            notes["not_estimable"].append(f"search candidates {pair}: not in every fold: {sorted(found ^ names)}")
        found = names if found is None else found & names
    search = training_inputs.encoder_seeds(pair, "MSE")
    kept = {k.removesuffix("_cells") for k in SET_ENCODER_CELLS if k in ALTERNATIVES}   # a triage-cut encoder leaves the set
    return {name: member(stage08b(pair, "search_candidates", name), candidate_keys(*name.split("/")), const(search))
            for name in sorted(found) if name.split("/")[0] in kept}


def bag_points(pair) -> list:
    """Fig 4A's points besides 500: (n_cells key, cell_subset key)."""
    out = [(n, "all") for n in representations.BAG_SIZES if n != T.FROZEN_NCELL]
    return out + ([(0, "all_available")] if pair in T.FIG4A_ALL_PAIRS else [])


def age_loss_cytoset(pair, unit) -> bool:
    """Whether the unit's frozen CytoSet configuration has the age-similarity loss: stages 06 and 08b then have no
    CytoSet row F "all"."""
    path = paths.output("05_search", pair, "cytoset", unit) / "selection.json"
    READ.add(path)
    return bool(json.loads(path.read_text())["frozen"].get("age_similarity_loss"))


def best_candidate(pair, notes) -> tuple:
    """Row 18: one call over every Fig 4 / S10 point. The candidate set: every search
    candidate of the three set encoders and their row F points (every n, 500 = row A at seeds 0-4, and "all");
    the MSE point, the FM mean and max lines, scVI's line and the pseudobulk-n line get CIs only. The test row
    (key best_candidate) holds T in delta and the max-statistic p in p_better. Returns the rows and the results
    (the oracle selects from their point estimates)."""
    curve = training_inputs.encoder_seeds(pair, "F")
    candidates = search_members(pair, notes)
    for key in [k for k in SET_ENCODER_CELLS if k in ALTERNATIVES]:
        candidates[f"{key}@{T.FROZEN_NCELL}"] = primary(pair, key, curve)
        for n, subset in bag_points(pair):
            candidates[f"{key}@{n}{subset}"] = secondary(pair, "bag_size", key, curve, n_cells=n, cell_subset=subset)
        if key == "cytoset_cells" and pair in T.FIG4A_ALL_PAIRS:   # no seeds in an age-loss unit: not estimable
            candidates[f"{key}@0all_available"]["seeds"] = lambda unit: [] if age_loss_cytoset(pair, unit) else list(curve)
            if any(age_loss_cytoset(pair, unit) for unit in primary_units(pair)):
                notes["not_estimable"].append(f"bag_size {pair} cytoset_cells@all: age-similarity loss; whole batch above the 3,000-cell "
                                              "micro-batch cap")
    others = {REFERENCE: primary(pair, REFERENCE)}
    if pair in T.AGE_LOSS_PAIRS:
        others["cytoset/objective_mse"] = member(stage08b(pair, "mse", "cytoset"), candidate_keys("cytoset", "objective_mse"),
                                                 const(training_inputs.encoder_seeds(pair, "MSE")))
    for key in [k for k in (*FM_CELLS, "scvi_cells") if k in ALTERNATIVES]:
        others[f"{key}@{T.FROZEN_NCELL}"] = primary(pair, key, curve)
        for n, subset in bag_points(pair):
            others[f"{key}@{n}{subset}"] = secondary(pair, "bag_size", key, curve, n_cells=n, cell_subset=subset)
        if key in FM_CELLS:
            others[f"{key}@max"] = member(stage08b(pair, "bag_size", f"{key}_max"), config_keys(key, pooling="max"), const([-1]))
    for n, subset in [(n, "all") for n in representations.BAG_SIZES] + [p for p in bag_points(pair) if p[1] == "all_available"]:
        others[f"pseudobulk@{n}{subset}"] = secondary(pair, "bag_size", REFERENCE, n_cells=n, cell_subset=subset)
    members = {**candidates, **others}
    units = primary_units(pair)
    results, n, missing = bootstrap(pair, units, members, "best_candidate")
    test = statistics_secondary.max_statistic(results, [c for c in candidates if c in results])
    carrier = max(test["kept"], key=lambda v: results[v][0] / test["se"][v]) if test["kept"] else None   # the candidate carrying T (Fig 4's caption)
    notes["best_candidate"][pair] = {"excluded_zero_se": test["excluded"], "candidates": len(test["kept"]), "carrier": carrier}
    head = pd.DataFrame([{**NEUTRAL, "row_kind": "candidate", "key": "best_candidate", "model": "none", "input": "none", "n_cells": 0,
                          "pooling": "native", **{c: np.nan for c in (*STATS, "mdd_obs")}, "delta": test["t"], "p_better": test["p"],
                          "resamples": B, "estimable": bool(np.isfinite(test["t"]))}]).assign(
        pair=pair, fold_set=units[0][0], fold=-1 if units[0][0] == "crossfit" else 0, encoder_seed=-1, readout_seed=-1, row="best_candidate",
        pool_set="per_pair", statistic="max_statistic", bulk_subset="all", n0=n[0], n1=n[1], excluded_pairs="")
    return pd.concat([head, rows(results, members, "best_candidate", pair, units[0][0], n=n, missing=missing)], ignore_index=True), results, candidates


def oracle(pair, chosen_from, candidates) -> pd.DataFrame:
    """Row 13: per encoder, the search candidate with the largest folds-mean point difference
    (selected once, on the point estimates; ties: the first in sorted order), its CI from stream 13 without selection
    inside the resamples. Labelled "selected on bulk; for reference only"."""
    chosen = {}
    for encoder in ("cytoset", "set_transformer", "scrat"):
        names = sorted(k for k in candidates if k.startswith(f"{encoder}/") and k in chosen_from)
        if names:
            best = max(names, key=lambda k: chosen_from[k][0])
            chosen[best] = candidates[best]
    return against_primary(pair, "oracle", chosen)[0]


def subgroup_subsets(pair) -> list:
    """[(sub-stream k, bulk_subset name, frame -> kept samples)] of the pair's subgroup row: Crohn's two fixed
    subsets, else the column's levels sorted (over every primary unit's scored bulk)."""
    column = C.SUBGROUP_COLUMNS[pair]
    if pair == "crohns":
        return [(k, name, lambda f, level=name.split("_")[0]: (f["group"] == 0) | (f["inflammation"] == level))
                for k, name in enumerate(C.CROHNS_SUBSETS)]
    levels = sorted({str(v) for u in primary_units(pair) for v in bulk_frame(pair, u)[column].dropna()})
    return [(k, f"{column}={level}", lambda f, level=level: f[column].astype(str) == level) for k, level in enumerate(levels)]


def subgroup_estimable(pair, select) -> bool:
    """Both labels in every unit's subset; for litter clusters, >= MIN_SUBGROUP_LITTERS litters per label group."""
    for unit in primary_units(pair):
        frame = bulk_frame(pair, unit)
        frame = frame[select(frame)]
        if frame["group"].nunique() < 2:
            return False
        if pair == "hammond" and S.HAMMOND_SCHEME == "litters" and frame.groupby("group")["litter"].nunique().min() < C.MIN_SUBGROUP_LITTERS:
            return False
    return True


def subgroups(pair, notes) -> list:
    """Row 14: stage 08's primary predictions restricted to each bulk subset, sub-stream k per subset."""
    members = {k: primary(pair, k) for k in KEYS}
    out = []
    for k, name, select in subgroup_subsets(pair):
        if not subgroup_estimable(pair, select):
            notes["not_estimable"].append(f"subgroups {pair} {name}: a label group (or < {C.MIN_SUBGROUP_LITTERS} litters) missing")
            continue
        results, n, missing = bootstrap(pair, primary_units(pair), members, "subgroups", select=select, sub=k)
        out.append(rows(results, members, "subgroups", pair, primary_units(pair)[0][0], bulk_subset=name, n=n, missing=missing))
    return out


def age_adjusted(pair, notes) -> pd.DataFrame:
    """Row 15, Crohn's: the covariate-adjusted AUROC (Janes & Pepe 2008). Per configuration and
    (encoder seed, read-out seed), the scoring-unit scores residualized on age by a linear fit among the controls,
    then the row's bootstrap on the residuals (the fit isn't redone inside resamples; stated). Donors without an
    age are left out (counted)."""
    members = {k: primary(pair, k) for k in KEYS}
    def transform(scores, frame, order):
        age = frame.groupby("scoring_unit_id")["age"].mean().loc[order].to_numpy()
        label = frame.groupby("scoring_unit_id")["group"].first().loc[order].to_numpy()
        return {k: statistics_secondary.age_residuals(s, age, label) for k, s in scores.items()}
    for unit in primary_units(pair):
        frame = bulk_frame(pair, unit)
        notes["age_adjusted"][pair] = {"units_without_age": int(frame.groupby("scoring_unit_id")["age"].mean().isna().sum())}
    results, n, missing = bootstrap(pair, primary_units(pair), members, "age_adjusted", select=lambda f: f["age"].notna(), transform=transform)
    return rows(results, members, "age_adjusted", pair, primary_units(pair)[0][0], bulk_subset="age_adjusted", n=n, missing=missing)


def split67(pair) -> pd.DataFrame:
    """Row 16: human blood, aging's 67/29 split."""
    units = [("split67", 0)]
    members = {k: primary(pair, k) for k in KEYS}
    results, n, missing = bootstrap(pair, units, members, "split67")
    return rows(results, members, "split67", pair, "split67", n=n, missing=missing)


def pooled_predictions(pair) -> pd.DataFrame:
    """Row 17: the four folds' predictions concatenated per (encoder seed index e, r), one AUROC over
    every donor; donors resampled within label across folds; one call at ("crossfit", 0)."""
    members = {k: primary(pair, k) for k in KEYS}
    units = primary_units(pair)
    frames = [bulk_frame(pair, u) for u in units]
    scores, order = {}, []
    for k, m in members.items():
        parts = [unit_array(m, u, set(f["sample_id"])) for u, f in zip(units, frames)]
        scores[k] = np.concatenate([a for a, _ in parts], axis=2)
        order = [x for _, units_ in parts for x in units_]
    if SMOKE:
        scores = {k: s[:1] for k, s in scores.items()}
    frame = pd.concat(frames, ignore_index=True)
    label, strata, clusters = D11["unit_design"](pair, ("crossfit", 0), order, frame)
    rng = np.random.default_rng(seeds.derived("bootstrap", pair, S.ROW_STREAMS["pooled_predictions"], "crossfit", 0))
    fixed = set(scores) if SMOKE else {k for k in scores if k in NO_ENCODER_LEVEL}
    results = statistics.fold_bootstrap(scores, label, strata, clusters, B, rng, fixed, REFERENCE)
    return rows(results, members, "pooled_predictions", pair, "crossfit", n=counts(frame), change={"variant": "pooled_predictions"})


def feature_variants(pair) -> tuple:
    """Row 30: pseudobulk TMM, VST, the smaller panels, every shared gene, the mapped orthologs (mouse pairs) and
    the FMs on the full gene set, all fixed, against primary pseudobulk."""
    members = {f"pseudobulk/{v}": member(stage08b(pair, "feature_variants", v), config_keys(REFERENCE, variant=v), const([-1])) for v in ("tmm", "vst")}
    names = [*(f"panel_{n}" for n in settings.PANEL_SIZE_SENSITIVITY), "all_shared"] + (["mapped_orthologs"] if PAIRS[pair]["species"] == "mouse" else [])
    for name in names:
        members[f"pseudobulk/{name}"] = member(stage08b(pair, "feature_variants", name), config_keys(REFERENCE, feature_set=name), const([-1]))
    for k in [k for k in ALTERNATIVES if configurations.CONFIGURATIONS[k]["model"] in ("geneformer", "scgpt")]:
        members[f"{k}/full_genes"] = member(stage08b(pair, "feature_variants", f"{k}/full_genes"), config_keys(k, feature_set="full_genes"), const([-1]))
    return against_primary(pair, "feature_variants", members)


def within_bulk(pair) -> dict:
    """unit -> the within-bulk reference's predictions (reference_predictions kind; sc2bulk.reference): ridge
    cross-fitted within the unit's scored bulk, read-out seeds 0-9 (human blood, aging: per fold)."""
    out = {}
    for unit in primary_units(pair):
        frame = bulk_frame(pair, unit)
        X = training_inputs.bulk_profiles(pair, unit, list(frame["sample_id"])).X.toarray().astype(np.float64)
        READ.update([settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad", paths.output("04_panel", pair, "panel.csv", unit),
                     paths.output("04_panel", pair, "shared_genes.csv")])
        group = frame["litter"] if pair == "hammond" else frame["donor_id"] if PAIRS[pair]["scoring_unit"] == "donor" else frame["sample_id"]
        stratum = frame["organ"].astype(str) if pair == "tms" else pd.Series("all", index=frame.index)
        parts = []
        for r in range(N_READOUT):
            seed = seeds.derived("within_bulk", pair, r, *unit)
            outer = reference.folds_of(group, frame["group"], stratum, seed)
            score = reference.cross_fit(X, frame["group"].to_numpy(), group, stratum, outer, seed)
            parts.append(pd.DataFrame({**NEUTRAL, "row_kind": "reference", "key": "within_bulk_reference", "model": "none", "input": "none",
                                       "n_cells": 0, "pooling": "native", "pair": pair, "fold_set": unit[0], "fold": unit[1],
                                       "encoder_seed": -1, "readout_seed": r, "side": "bulk", "sample_id": frame["sample_id"].to_numpy(),
                                       "scoring_unit_id": frame["scoring_unit_id"].to_numpy(), "score": score,
                                       "probability": np.clip(score, 0.0, 1.0)}))
        out[unit] = schemas.check("reference_predictions", pd.concat(parts, ignore_index=True))
    return out


def reference_members(pair) -> dict:
    """Stream 32's members: every reference row 08b formed for the pair, fixed but untrained
    CytoSet (encoder seeds 0-9), and the pair's sensitivity variants "<key>/<variant>" (fixed)."""
    fixed = const([-1])
    members = {k: member(stage08b(pair, "reference_rows", k), reference_keys(k), fixed) for k in ("pseudobulk_pca64", "mean_log_cells")}
    members["per_cell_classifier"] = member(stage08b(pair, "reference_rows", "per_cell_classifier"),
                                            reference_keys("per_cell_classifier", n_cells=C.PER_CELL_CAP), fixed)
    if deconvolution_config.LEVELS[pair] is not None:   # not mouse microglia (one cell type)
        for k in ("deconvolution_clr", "deconvolution_true"):
            members[k] = member(stage08b(pair, "reference_rows", k), reference_keys(k), fixed)
    members["known_signature"] = member(stage08b(pair, "reference_rows", "known_signature"), reference_keys("known_signature", readout="none"), fixed)
    members["cytoset_untrained"] = member(stage08b(pair, "reference_rows", "cytoset_untrained"),
                                          reference_keys("cytoset_untrained", n_cells=T.FROZEN_NCELL), const(ENCODER_SEEDS))
    for (key, variant), pairs in C.REFERENCE_VARIANTS.items():
        if key in members and (pairs is None or pair in pairs):
            members[f"{key}/{variant}"] = member(stage08b(pair, "reference_rows", f"{key}/{variant}"), {**members[key]["keys"], "variant": variant}, fixed)
    return members


def pair_specific(name) -> bool:
    """Whether a member is a sensitivity variant formed for one pair only (never pooled)."""
    key, _, variant = name.partition("/")
    return bool(variant) and C.REFERENCE_VARIANTS[(key, variant)] is not None


def label_permuted(pair, primary_auroc) -> pd.DataFrame:
    """The label-permuted control (no bootstrap): per permutation j, delta_j = AUROC(j) -
    pseudobulk's AUROC(j), each the mean over read-out seeds (and over folds); per configuration the delta_j rows
    and their mean and 95th percentile (permutation -1). Pseudobulk (Fig 2A's ✕): per j its
    AUROC(j) minus its primary AUROC (stage 11's pair_results), statistic permuted_minus_primary, and their mean
    and 95th percentile (permuted_minus_primary_mean, _q95)."""
    permutations = training_inputs.encoder_seeds(pair, "D")
    auroc = {}
    for k in KEYS:
        for j in permutations:
            e = -1 if k in NO_ENCODER_LEVEL else j % settings.N_ENCODER_SEEDS if k == "scvi_cells" else j
            m = member(stage08b(pair, "permuted", k), config_keys(k, permutation=j), const([e]))
            per_unit = []
            for unit in primary_units(pair):
                array, order = unit_array(m, unit)
                label = bulk_frame(pair, unit).groupby("scoring_unit_id")["group"].first().loc[order].to_numpy()
                per_unit.append(np.mean([statistics.auroc(array[0, r], label) for r in range(array.shape[1])]))
            auroc[k, j] = float(np.mean(per_unit))
    out = []
    for k in ALTERNATIVES:
        deltas = [auroc[k, j] - auroc[REFERENCE, j] for j in permutations]
        out += [{**config_keys(k, permutation=j), "statistic": "auroc_difference", "delta": d} for j, d in zip(permutations, deltas)]
        out += [{**config_keys(k), "statistic": s, "delta": v} for s, v in statistics_secondary.permuted_summary(deltas).items()]
    deltas = [auroc[REFERENCE, j] - primary_auroc for j in permutations]
    out += [{**config_keys(REFERENCE, permutation=j), "statistic": "permuted_minus_primary", "delta": d} for j, d in zip(permutations, deltas)]
    out += [{**config_keys(REFERENCE), "statistic": f"permuted_minus_primary_{s.removeprefix('permuted_')}", "delta": v}
            for s, v in statistics_secondary.permuted_summary(deltas).items()]
    fold_set = primary_units(pair)[0][0]
    return pd.DataFrame(out).assign(pair=pair, fold_set=fold_set, fold=-1 if fold_set == "crossfit" else 0, encoder_seed=-1, readout_seed=-1,
                                    row="label_permuted", pool_set="per_pair", bulk_subset="all", excluded_pairs="", estimable=True,
                                    resamples=0, **{c: np.nan for c in (*STATS[1:], "mdd_obs", "n0", "n1")})


def stage11(notes) -> tuple:
    """Stage 11's per-pair primary results, rows and pseudobulk AUROC, recomputed with its own functions and stream,
    and the ceiling set. Pseudobulk's AUROC per pair goes to notes["pseudobulk_primary_auroc"] (the
    label-permuted ✕'s comparator)."""
    results, tables, ceiling = {}, {}, {}
    notes["pseudobulk_primary_auroc"] = {}
    for pair in PAIR_ORDER:
        results[pair], pb, fold_set, n = D11["pair_results"](pair)
        notes["pseudobulk_primary_auroc"][pair] = pb
        tables[pair] = D11["rows"](results[pair], "per_pair", pair, fold_set, "auroc_difference", True, n=n)
        ceiling[pair] = statistics.at_ceiling(pb, S.MARGIN)
    kept = [p for p in PAIR_ORDER if not ceiling[p]]
    excluded = [p for p in PAIR_ORDER if ceiling[p]]
    pooled = None
    if kept:
        label = excluded + ([f"single non-ceiling pair: {kept[0]}"] if len(kept) == 1 else [])
        pooled = D11["rows"](statistics.pool_equal({p: results[p] for p in kept}), "nonceiling_equal", "pooled", "all", "auroc_difference", True, label)
    notes["excluded_at_ceiling"] = excluded
    return results, tables, pooled, kept, excluded


def assert_stage11_equal(tables, pooled, excluded):
    """A real run: the recomputed rows equal stage 11's written table exactly, and so does the ceiling set."""
    path = paths.output("11_statistics", "pooled", f"{settings.LABELLED}_statistics.parquet")
    written = schemas.read("statistics", path)
    claims = json.loads(path.with_name(f"{settings.LABELLED}_claims.json").read_text())
    assert claims["excluded_at_ceiling"] == excluded, "the ceiling set differs from stage 11's"
    mine = pd.concat([*tables.values(), *([pooled] if pooled is not None else [])], ignore_index=True)
    theirs = written[(written["readout"] == "ridge") & written["pool_set"].isin(["per_pair", "nonceiling_equal"])]
    merged = mine.merge(theirs, on=["pair", "pool_set", "key"], suffixes=("", "_11"), validate="one_to_one")
    assert len(merged) == len(mine) == len(theirs), "the recomputed rows don't match stage 11's rows one to one"
    for c in STATS:
        assert np.array_equal(merged[c].to_numpy(dtype=float), merged[f"{c}_11"].to_numpy(dtype=float), equal_nan=True), f"{c} differs from stage 11's"


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=settings.REPO, capture_output=True, text=True, check=check)


def preflight() -> list:
    """A real run: stage 11's frozen files (its assert_frozen), stage 11's import closure against the analysis-fixed
    tag with only config/triage.py allowed to differ (the differences are returned for the run record),
    and 11b's own files against the same tag."""
    D11["assert_frozen"]()
    tag = json.loads(labels.ANALYSIS_FIXED.read_text())["tag"]
    closure = sorted(str(p.relative_to(settings.REPO)) for p in codehash.files(settings.REPO / "pipeline" / "11_statistics.py"))
    changed = git("diff", "--name-only", tag, "--", *closure).stdout.split()
    assert set(changed) <= set(C.CLOSURE_ALLOWLIST), f"stage 11's import closure differs from the analysis-fixed tag {tag}: {changed}"
    assert git("diff", "--quiet", tag, "--", *C.SECONDARY_FILES, check=False).returncode == 0, \
        f"11b's files differ from the analysis-fixed tag {tag} (a row added after the real-label run goes in a new, separately named file)"
    return changed


# ---- the scores table ----

def score_rows(table, bulk_label, single_label, bulk_subset="all") -> list:
    """scoring.score per (key columns, side) group of a predictions-like table."""
    columns = [*schemas.KEYS, "side"]
    means = table.groupby([*columns, "scoring_unit_id"], sort=True)[["score", "probability"]].mean().reset_index()   # one pass
    group = means.groupby(columns, sort=False).ngroup().to_numpy()
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])   # sorted by the key columns: each group is contiguous
    bulk = means["side"].to_numpy() == "bulk"
    label = np.where(bulk, bulk_label.reindex(means["scoring_unit_id"]).to_numpy(), single_label.reindex(means["scoring_unit_id"]).to_numpy())
    assert not pd.isna(label).any(), "a scoring unit without a label"
    score, probability, label = means["score"].to_numpy(), means["probability"].to_numpy(), label.astype(float)
    keys = means.iloc[starts][columns].to_dict("records")
    return [{**k, "bulk_subset": bulk_subset, **scoring.score(score[a:b], probability[a:b], label[a:b])}
            for k, a, b in zip(keys, starts, [*starts[1:], len(means)])]


def scores_table(pair, references, notes) -> pd.DataFrame:
    """Absolute scores of every predictions file of stages 08, 08b (but its single_cell_moments family) and 09
    (stage 08's at encoder seeds 0-9: 10-19 are attribution only), the within-bulk reference, the seed ensembles
    (S11B) and ALS's age-adjusted within-single-cell AUROC."""
    bulk_label = pd.concat([bulk_frame(pair, u) for u in folds.units(pair)]).groupby("scoring_unit_id")["group"].first()
    samples = folds.single_cell_samples(pair)
    single_label = samples.groupby(samples["donor_id" if PAIRS[pair]["scoring_unit"] == "donor" else "sample_id"])["group"].first()
    out = []
    for unit in folds.units(pair):
        for path in sorted(paths.output("08_readout", pair, "", unit).glob("*_predictions.parquet")):
            table = predictions(path)
            table = table[table["encoder_seed"] < len(ENCODER_SEEDS)]
            out += score_rows(table, bulk_label, single_label)
            if not path.name.endswith(("_within_predictions.parquet", "_out_of_fold_predictions.parquet")) and table["encoder_seed"].max() >= 0:
                ensemble = table[table["side"] == "bulk"].groupby([c for c in table.columns if c not in ("encoder_seed", "score", "probability")],
                                                                  sort=False)[["score", "probability"]].mean().reset_index()
                out += score_rows(ensemble.assign(encoder_seed=-1, variant="seed_ensemble"), bulk_label, single_label)
            if pair == "pineda" and path.name.endswith("_within_predictions.parquet"):
                out += age_adjusted_within(table, samples, notes)
        for path in sorted(paths.output("08b_secondary", pair, "", unit).glob("*/*_predictions.parquet")):
            if path.parent.name == "single_cell_moments":   # differences only; its keys are stage 08's
                continue
            out += score_rows(predictions(path), bulk_label, single_label)
        for path in sorted(paths.output("09_alignment", pair, "", unit).glob("*_predictions.parquet")):
            out += score_rows(predictions(path), bulk_label, single_label)
        if unit in references:
            out += score_rows(references[unit], bulk_label, single_label)
        CACHE.clear()
    return pd.DataFrame(out)


def age_adjusted_within(table, samples, notes) -> list:
    """ALS's covariate-adjusted within-single-cell AUROC: per split e (and read-out, read-out seed),
    Fig 2B's validation scores residualized on the donor's age at death among that side's controls; a split with
    fewer than MIN_AGE_CONTROLS controls of distinct ages is not estimable (counted)."""
    age = samples.groupby("donor_id")["age_at_death"].mean()
    label = samples.groupby("donor_id")["group"].first()
    out, skipped = [], 0
    columns = [*schemas.KEYS, "side"]
    for values, part in table[table["side"] == "single_cell_validation"].groupby(columns, sort=False):
        means = scoring.unit_means(part)
        y = label.loc[means.index].to_numpy()
        auroc = statistics_secondary.residual_auroc(means["score"].to_numpy(), age.loc[means.index].to_numpy(), y, C.MIN_AGE_CONTROLS)
        if auroc is None:
            skipped += 1
            continue
        out.append({**dict(zip(columns, values)), "bulk_subset": "age_adjusted", "n0": int((y == 0).sum()), "n1": int((y == 1).sum()),
                    "auroc": auroc, "auprc": np.nan, "brier": np.nan, "signed_log10_p": np.nan})
    notes["als_age_adjusted"]["not_estimable_fits"] = notes["als_age_adjusted"].get("not_estimable_fits", 0) + skipped
    notes["als_age_adjusted"]["estimable_fits"] = notes["als_age_adjusted"].get("estimable_fits", 0) + len(out)
    return out


# ---- the run ----

def pooled(per_pair, kept, row, family, notes) -> list:
    """The equal-weight pooled row over the non-ceiling pairs where the row was formed (K = 1 is that
    pair, labelled; K = 0 forms nothing); a configuration missing in one of them is not pooled (listed).
    reference_rows: per member (pooled_by_member)."""
    pairs = [p for p in kept if per_pair.get(p) is not None]
    if not pairs:
        return []
    if row == "reference_rows":
        return pooled_by_member(per_pair, pairs, row, notes)
    common = set.intersection(*(set(per_pair[p][0]) for p in pairs))
    dropped = sorted(set().union(*(set(per_pair[p][0]) for p in pairs)) - common)
    if dropped:
        notes["not_pooled"].append(f"{row}: {dropped} missing in a non-ceiling pair")
    excluded = [p for p in PAIR_ORDER if p not in pairs] + ([f"single non-ceiling pair: {pairs[0]}"] if len(pairs) == 1 else [])
    members = per_pair[pairs[0]][1]
    results = statistics.pool_equal({p: {k: per_pair[p][0][k] for k in common} for p in pairs})
    return [rows(results, members, row, "pooled", "all", family=family, excluded=",".join(excluded))], results


def pooled_by_member(per_pair, pairs, row, notes) -> tuple:
    """reference_rows' pooled rows: each member (the union over the pairs) pooled over the
    non-ceiling pairs where it is defined, with the K = 1 rule on that set and its own excluded pairs; a member
    defined but not estimable in one of them is not pooled, and a pair-specific sensitivity variant is never pooled
    (both listed)."""
    members = {k: m for p in pairs for k, m in per_pair[p][1].items() if k != REFERENCE}
    frames, results = [], {}
    for k, m in members.items():
        defined = [p for p in pairs if k in per_pair[p][1]]
        if pair_specific(k):
            notes["not_pooled"].append(f"{row}: {k}: pair-specific sensitivity row")
            continue
        if any(k not in per_pair[p][0] for p in defined):
            notes["not_pooled"].append(f"{row}: {k} not estimable in a non-ceiling pair")
            continue
        excluded = [p for p in PAIR_ORDER if p not in defined] + ([f"single non-ceiling pair: {defined[0]}"] if len(defined) == 1 else [])
        one = statistics.pool_equal({p: {k: per_pair[p][0][k]} for p in defined})
        frames.append(rows(one, {k: m}, row, "pooled", "all", excluded=",".join(excluded)))
        results.update(one)
    return frames, results


def label_permuted_pooled(frames, kept) -> list:
    """Fig 2A's pooled ✕: the equal-weight mean over the non-ceiling pairs where the row was formed of
    pseudobulk's per-pair permuted_minus_primary_mean, descriptive (no interval); excluded_pairs the other pairs (the
    ceiling pairs; K = 1: that pair, labelled; K = 0: none)."""
    table = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    means = table[(table["row"] == "label_permuted") & (table["statistic"] == "permuted_minus_primary_mean")].set_index("pair")["delta"]
    pairs = [p for p in kept if p in means.index]
    if not pairs:
        return []
    label = [p for p in PAIR_ORDER if p not in pairs] + ([f"single non-ceiling pair: {pairs[0]}"] if len(pairs) == 1 else [])
    return [pd.DataFrame([{**config_keys(REFERENCE), "statistic": "permuted_minus_primary_mean", "delta": float(means.loc[pairs].mean())}]).assign(
        pair="pooled", fold_set="all", fold=-1, encoder_seed=-1, readout_seed=-1, row="label_permuted", pool_set="nonceiling_equal",
        bulk_subset="all", excluded_pairs=",".join(label), estimable=True, resamples=0, **{c: np.nan for c in (*STATS[1:], "mdd_obs", "n0", "n1")})]


def smoke_absent(pair, row, stage="06_train") -> list:
    """Smoke only: the primary units whose row stage 06 skipped (row H or E: 06_train.smoke_marker; 08b skips the rows
    that read it), or (stage 08b_secondary, row one_type) whose one_type family 08b skipped (its one_type_marker). The
    real run has every row in every unit."""
    absent = [u for u in primary_units(pair) if (paths.output(stage, pair, row, u) / "smoke_skipped.json").exists()]
    assert SMOKE or not absent, f"{pair}: {stage}'s smoke row {row} marker in the real run"
    return absent


def one_type(pair, notes) -> pd.DataFrame:
    """Row 29 (row H): as same_variant, except that under smoke, where 08b skipped a unit's one_type family (stage 06
    skipped its row H, or the one-type store falls short of a fit; smoke_absent), no configuration is estimable."""
    absent = smoke_absent(pair, "one_type", "08b_secondary")
    if absent:
        notes["not_estimable"].append(f"one_type {pair}: smoke skipped the one-type rows in {[folds.unit_name(*u) for u in absent]}")
    H = training_inputs.encoder_seeds(pair, "H")
    members = {k: secondary(pair, "one_type", k, H, cell_subset="one_type") for k in KEYS}
    results, n, missing = ({}, (np.nan, np.nan), ALTERNATIVES) if absent else bootstrap(pair, primary_units(pair), members, "one_type")
    return rows(results, members, "one_type", pair, primary_units(pair)[0][0], n=n, missing=missing)


def learning_curve(pair, notes) -> list:
    """Rows 27, 28, 33 (row E; S12b): each configuration at 25% and 50% (08b's learning_curve) and at 100% (stage 08's
    primary) at row E's seeds. Under smoke, a unit whose row E stage 06 skipped (smoke_absent) has no 25% or 50%
    rows for any configuration (08b skips them), so those two rows are not estimable."""
    curve = training_inputs.encoder_seeds(pair, "E")
    absent = smoke_absent(pair, "E")
    if absent:
        notes["not_estimable"].append(f"learning_curve {pair}: smoke skipped row E in {[folds.unit_name(*u) for u in absent]}")
    out = []
    for f, row in zip(T.LEARNING_CURVE_FRACTIONS, ("learning_curve_25", "learning_curve_50")):
        members = {k: member(stage08b(pair, "learning_curve", k), config_keys(k, variant=f"training_fraction_{f}"), const(curve)) for k in KEYS}
        results, n, missing = ({}, (np.nan, np.nan), ALTERNATIVES) if absent else bootstrap(pair, primary_units(pair), members, row)
        out.append(rows(results, members, row, pair, primary_units(pair)[0][0], n=n, missing=missing))
    members = {k: primary(pair, k, curve) for k in KEYS}
    results, n, missing = bootstrap(pair, primary_units(pair), members, "learning_curve_100")
    return out + [rows(results, members, "learning_curve_100", pair, primary_units(pair)[0][0], n=n, missing=missing)]


def pair_rows(pair, references, notes) -> tuple:
    """Every bootstrapped row of one pair: (frames, {row: (results, members)} for pooling)."""
    out, pool = [], {}
    def keep(row, made):
        frame, pooled_input = made
        out.append(frame)
        pool[row] = pooled_input
    keep("passing_seeds", passing_seeds(pair, notes))
    keep("out_of_fold", out_of_fold(pair, notes))
    trained = [k for k in ("cytoset_cells", "set_transformer_cells", "scrat_cells", "scvi_cells") if k in ALTERNATIVES]
    keep("thinned_bulk", against_primary(pair, "thinned_bulk", {k: secondary(pair, "thinned", k, variant="thinned_pseudocells") for k in trained},
                                         family=True))
    if "scvi_cells" in ALTERNATIVES:
        keep("scvi_observed_library", against_primary(pair, "scvi_observed_library",
                                                      {"scvi_cells": secondary(pair, "observed_library", "scvi_cells", variant="observed_library")}))
    if pair in T.AGE_LOSS_PAIRS:
        out.append(same_variant(pair, "numeric_age", "numeric_age", {"variant": "numeric_age_readout"})[0])
    for which in ("combat", "coral", "harmony", "unaligned_stacked"):
        keep(f"alignment_{which}", alignment(pair, which))
    recipes = {f"{e}_own_recipe": member(stage08b(pair, "own_recipe", e),
                                         candidate_keys(e, f"{e}_own_recipe", T.OWN_RECIPES[e].get("ncell", search_space.base(e)["ncell"])),
                                         const(ENCODER_SEEDS)) for e in ("cytoset", "set_transformer", "scrat")}
    keep("own_recipe", against_primary(pair, "own_recipe", recipes))
    frame, results, candidates = best_candidate(pair, notes)
    out.append(frame)
    out.append(oracle(pair, results, candidates))
    out += subgroups(pair, notes) if pair in C.SUBGROUP_COLUMNS else []
    if pair == "crohns":
        out.append(age_adjusted(pair, notes))
    if "split67" in PAIRS[pair]["fold_sets"]:
        out.append(split67(pair))
        out.append(pooled_predictions(pair))
    keep("outside_moments", same_variant(pair, "outside_moments", "outside_moments", {"variant": "outside_moments_scaling"}))
    keep("single_cell_moments", same_variant(pair, "single_cell_moments", "single_cell_moments", {}))   # variant "primary"
    for which in C.ALTERNATIVE_READOUTS:
        members = {k: member(stage08b(pair, "alternative_readouts", f"{k}/{which}"), config_keys(k, readout=which),
                             const(seeds_for(k, ENCODER_SEEDS))) for k in KEYS}
        results, n, missing = bootstrap(pair, primary_units(pair), members, f"readout_{which}")
        keep(f"readout_{which}", (rows(results, members, f"readout_{which}", pair, primary_units(pair)[0][0], n=n, missing=missing), (results, members)))
    if "cytoset_cells" in ALTERNATIVES:
        keep("native_head", against_primary(pair, "native_head", {"cytoset_cells": secondary(pair, "native_head", "cytoset_cells", readout="native_head")}))
    routes = {}
    for e in [e for e in ("cytoset", "set_transformer", "scrat") if f"{e}_cells" in ALTERNATIVES]:
        triggered = [per_sample_route_triggered(pair, u, e) for u in primary_units(pair)]
        if all(triggered):
            routes[f"{e}_cells"] = member(stage08b(pair, "per_sample_route", e), config_keys(f"{e}_cells", variant="per_sample_route_sensitivity"),
                                          const(ENCODER_SEEDS))
        elif any(triggered):
            notes["not_estimable"].append(f"per_sample_route {pair} {e}: triggered in {sum(triggered)} of {len(triggered)} folds")
    if routes:
        keep("per_sample_route", against_primary(pair, "per_sample_route", routes))
    if pair in T.LEARNING_CURVE_PAIRS:
        out += learning_curve(pair, notes)
    if "one_type" in PAIRS[pair]:
        out.append(one_type(pair, notes))
    keep("feature_variants", feature_variants(pair))
    within = {"within_bulk_reference": {"load": lambda unit: references[unit], "seeds": const([-1]),
                                        "keys": {**NEUTRAL, "row_kind": "reference", "key": "within_bulk_reference", "model": "none",
                                                 "input": "none", "n_cells": 0, "pooling": "native"}}}
    keep("within_bulk_reference", against_primary(pair, "within_bulk_reference", within))
    keep("reference_rows", against_primary(pair, "reference_rows", reference_members(pair)))
    out.append(label_permuted(pair, notes["pseudobulk_primary_auroc"][pair]))
    return out, pool


def decomposition_rows(results, kept, excluded) -> list:
    """Fig 2C's decomposition: per pair and pooled, delta(model, cells) - delta(model, pseudobulk) from
    stage 11's primary draws (unadjusted)."""
    models = [m for m in ("cytoset", "geneformer", "scgpt") if f"{m}_cells" in ALTERNATIVES and f"{m}_pseudobulk" in ALTERNATIVES]
    members = {f"{m}_cells": {"keys": config_keys(f"{m}_cells")} for m in models}
    out = []
    for pair in PAIR_ORDER:
        fold_set = next(iter(PAIRS[pair]["fold_sets"]))
        out.append(rows(statistics_secondary.decomposition(results[pair], models), members, "decomposition", pair, fold_set,
                        statistic="cells_minus_pseudobulk_input"))
    if kept and models:
        label = excluded + ([f"single non-ceiling pair: {kept[0]}"] if len(kept) == 1 else [])
        pooled_ = statistics_secondary.decomposition(statistics.pool_equal({p: results[p] for p in kept}), models)
        out.append(rows(pooled_, members, "decomposition", "pooled", "all", excluded=",".join(label), statistic="cells_minus_pseudobulk_input"))
    return out


def leave_one_pair_out_rows(results, kept, notes) -> list:
    """Descriptive: with K >= 3 non-ceiling pairs, stage 11's equal-weight pooled row without each
    non-ceiling pair in turn, from stage 11's primary draws combined index by index; unadjusted, triggers nothing.
    excluded_pairs: the ceiling pairs and the pair left out."""
    if len(kept) < 3:
        notes["not_formed"].append(f"leave_one_pair_out: {len(kept)} non-ceiling pairs (needs 3)")
        return []
    members = {k: {"keys": config_keys(k)} for k in ALTERNATIVES}
    out = []
    for left in kept:
        others = [p for p in kept if p != left]
        excluded = [p for p in PAIR_ORDER if p not in others]
        out.append(rows(statistics.pool_equal({p: results[p] for p in others}), members, "leave_one_pair_out", "pooled", "all",
                        excluded=",".join(excluded), pool_set=f"nonceiling_equal_without_{left}"))
    return out


def rank_rows(results) -> pd.DataFrame:
    """The summary rank: per pair, each configuration's rank of stage 11's point difference
    (pseudobulk 0 included; 1 = the largest), and the mean over the six pairs."""
    table = statistics_secondary.summary_rank({p: {REFERENCE: 0.0, **{k: results[p][k][0] for k in ALTERNATIVES}} for p in PAIR_ORDER})
    out = [{**config_keys(k), "pair": p, "fold_set": next(iter(PAIRS[p]["fold_sets"])), "pool_set": "per_pair", "delta": table.loc[k, p]}
           for p in PAIR_ORDER for k in table.index]
    out += [{**config_keys(k), "pair": "pooled", "fold_set": "all", "pool_set": "pooled", "delta": table.loc[k, "mean_rank"]} for k in table.index]
    frame = pd.DataFrame(out)
    return frame.assign(fold=np.where(frame["fold_set"].eq("crossfit") | frame["pair"].eq("pooled"), -1, 0), encoder_seed=-1, readout_seed=-1,
                        row="summary_rank", statistic="summary_rank", bulk_subset="all", excluded_pairs="", estimable=True, resamples=0,
                        **{c: np.nan for c in (*STATS[1:], "mdd_obs", "n0", "n1")})


def reading_rule_table(results, per_pair_oof, pooled_oof, kept) -> pd.DataFrame:
    """reading_rule.changes per pair (stage 11's per-pair rows against the out-of-fold row) and pooled (reported in
    the Discussion; triggers nothing). A pair not estimable in the out-of-fold row has every configuration
    reported as not estimable."""
    empty = pd.DataFrame(columns=["key", "delta", "p_better_holm", "p_worse_holm", "u"])
    out = []
    for pair in PAIR_ORDER:
        oof = statistics.summarize(per_pair_oof[pair][0], True) if per_pair_oof.get(pair) is not None else empty
        out.append(reading_rule.changes(statistics.summarize(results[pair], True), oof).assign(pair=pair))
    if kept:
        primary_pooled = statistics.summarize(statistics.pool_equal({p: results[p] for p in kept}), True)
        out.append(reading_rule.changes(primary_pooled, statistics.summarize(pooled_oof, True) if pooled_oof is not None else empty).assign(pair="pooled"))
    return pd.concat(out, ignore_index=True)


def write(kind, frame, path, record, cost):
    path.parent.mkdir(parents=True, exist_ok=True)
    schemas.write(kind, frame, manifest.temporary(path))
    manifest.finish(path, record, cost, inputs=sorted(READ))


def main(argv=()):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER)
    pair = parser.parse_args(argv).pair
    if pair not in (None, PAIR_ORDER[-1]):   # run_all calls every stage once per pair; 11b runs once, after the last
        print(f"11b pools every pair: it runs with --pair {PAIR_ORDER[-1]} (or none); skipped for {pair}", flush=True)
        return
    determinism.configure()
    labels.require_access()
    deviations = preflight() if settings.BLINDED is None else []
    CACHE.clear()
    READ.clear()
    cost = manifest.begin()
    notes = {"not_estimable": [], "not_pooled": [], "seeds_passed": {}, "best_candidate": {}, "age_adjusted": {}, "als_age_adjusted": {},
             "not_formed": ["reference_rows: deconvolution for mouse microglia (one cell type)", "female_only (stream 20): reserved, never drawn"]}
    results, tables, pooled_primary, kept, excluded = stage11(notes)
    if settings.BLINDED is None:
        assert_stage11_equal(tables, pooled_primary, excluded)
    frames, per_row, references, scores = [], {}, {}, []
    for pair in PAIR_ORDER:
        CACHE.clear()
        references[pair] = within_bulk(pair)
        out, pool = pair_rows(pair, references[pair], notes)
        frames += out
        for row, made in pool.items():
            per_row.setdefault(row, {})[pair] = made
        scores.append(scores_table(pair, references[pair], notes))
    pooled_oof = None
    for row in POOLED_ROWS:
        made = pooled(per_row.get(row, {}), kept, row, row in HOLM_ROWS, notes)
        if made:
            frames += made[0]
            pooled_oof = made[1] if row == "out_of_fold" else pooled_oof
    frames += label_permuted_pooled(frames, kept)
    frames += decomposition_rows(results, kept, excluded)
    frames += leave_one_pair_out_rows(results, kept, notes)
    frames.append(rank_rows(results))
    table = schemas.with_neutral_keys(pd.concat([f for f in frames if not f.empty], ignore_index=True))
    tests = (table["row"] == "best_candidate") & (table["key"] == "best_candidate") & table["p_better"].notna()
    table.loc[tests, "p_better_holm"] = statistics.holm(table.loc[tests, "p_better"])   # beside the per-pair p
    assert set(table["row"]) <= set(S.ROW_STREAMS) | set(C.UNSTREAMED_ROWS), "a row outside ROW_STREAMS and the unstreamed rows"
    schemas.check("secondary_statistics", table)
    scores_frame = schemas.check("scores", schemas.with_neutral_keys(pd.concat(scores, ignore_index=True)))
    rule = reading_rule_table(results, per_row.get("out_of_fold", {}), pooled_oof, kept)
    for pair, by_unit in references.items():
        for frame in by_unit.values():
            schemas.check("reference_predictions", frame)
    if settings.BLINDED is not None:   # a blinded run produces nothing
        print("blinded run: stage 11b computed and checked, outputs suppressed", flush=True)
        return
    for pair in PAIR_ORDER:
        for path in [paths.output("02_preprocess", pair, name) for name in ("bulk_labels.csv", "bulk_samples.csv")]:
            READ.add(path)
        READ.add(paths.output("03_splits", pair, "donor_folds.csv"))   # the blinded permutation's strata
    record = {"stage": STAGE, "smoke": settings.SMOKE, "resamples": B, "deviations": deviations,
              "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", None, codehash.extra(STAGE, None)),
              "inputs": manifest.input_hashes(sorted(READ))}
    out = paths.output(STAGE, "pooled", f"{settings.LABELLED}_secondary_statistics.parquet")
    write("secondary_statistics", table, out, record, cost)
    write("scores", scores_frame, out.with_name(f"{settings.LABELLED}_scores.parquet"), record, cost)
    for name, text in ((f"{settings.LABELLED}_reading_rule.csv", rule.to_csv(index=False)),
                       (f"{settings.LABELLED}_secondary_notes.json", json.dumps(notes, indent=1, default=str))):
        manifest.temporary(out.with_name(name)).write_text(text)
        manifest.finish(out.with_name(name), record, cost, inputs=sorted(READ))
    for pair, by_unit in references.items():
        for unit, frame in by_unit.items():
            write("reference_predictions", frame, paths.output(STAGE, pair, f"{settings.LABELLED}_within_bulk_reference_predictions.parquet", unit),
                  record, cost)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
