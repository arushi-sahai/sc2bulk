"""Stage 04b: the deconvolution baseline (reference rows deconvolution_clr / deconvolution_true),
label-free (computations in sc2bulk.deconvolution, numbers and maps in config/deconvolution.py).

Per unit (and within organ for mouse organs), from the unit's training single-cell data only:
  1. types: the pre-declared annotation (mapped where a map is declared); reference types (>= 0.2%
     and >= 50 cells) and feature types (>= 1%); the rest dropped;
  2. V1 and merging: on stage 03's encoder splits, each split's reference (signature and markers
     from its training samples) deconvolves its validation samples' pseudobulk; per feature type,
     the Pearson r with the true (annotated) fractions, pooled over splits. While a feature type has
     r < 0.5 it is merged into the feature type whose log signature it correlates with most;
  3. the final reference deconvolves the training samples' pseudobulk (row A's single-cell
     features) and the unit's scored bulk (row A's bulk features); true fractions of the training
     samples (row B's single-cell features) from the annotated stored cells;
  4. checks: V2 (one feature type removed), V4 (residuals; the neutrophil-leak indicator for the
     blood pairs); for the cross-fitted pair, V3 (each fold's held-out donors' bulk vs their own
     single-cell fractions, matched by donor and visit) in a unit-free diagnostic file.
Marker candidates: the unit's panel, minus technical genes, minus genes failing the label-free
mismatch filter against the unit's scored bulk. Lupus also writes a sensitivity version with Yao's
21 interferon genes removed from the candidates. Mouse microglia: excluded (one type).

The type lists written here are frozen as outputs (pinned in the deposit). Nothing reads a label.
Outputs, in <outputs>/04b_deconvolve/<pair>/<unit>/: types.csv, markers.csv, fractions.parquet,
check_v1.csv, check_v2.csv, check_v4.csv (lupus: also fractions_no_ifn.parquet); and
<pair>/diagnostic/<unit>_v3.csv, <unit>_scored_true.parquet (cross-fitting units).

  python pipeline/04b_deconvolve.py [--pair PAIR]
"""
import argparse

import anndata
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from sc2bulk import deconvolution as dc
from sc2bulk import determinism, folds, manifest, paths, signatures
from sc2bulk.config import deconvolution as config
from sc2bulk.config import settings
from sc2bulk.config.pairs import PAIR_ORDER

STAGE = "04b_deconvolve"


def input_paths(pair) -> list:
    return folds.inputs(pair) + [settings.OUTPUTS / "02_preprocess" / pair / name for name in ("bulk.h5ad", "bulk_samples.csv")] \
        + [settings.OUTPUTS / "03_splits" / pair / name for name in ("donor_folds.csv", "splits.csv")] \
        + [paths.output("04_panel", pair, "shared_genes.csv")] + [paths.output("04_panel", pair, "panel.csv", u) for u in folds.units(pair)]


def record_for(pair) -> dict:
    return {"stage": STAGE, "pair": pair, "smoke": settings.SMOKE,
            "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair),
            "inputs": manifest.input_hashes(input_paths(pair))}


def cell_types(obs: pd.DataFrame, level: dict) -> pd.Series:
    labels = obs[level["column"]].astype(str)
    if level["map"]:
        missing = set(labels) - set(level["map"])
        assert not missing, f"labels missing from the declared map: {sorted(missing)}"
        return labels.map(level["map"])
    return labels


def merged(types: pd.Series, merges: dict) -> pd.Series:
    """Labels after the V1 merges, following chains (a -> b, then b -> c)."""
    def final(t):
        while t in merges:
            t = merges[t]
        return t
    return types.map(final)


def true_fractions(prof: dict, samples, feature) -> pd.DataFrame:
    rows = np.isin(prof["samples"], samples)
    counts = pd.DataFrame(prof["n"][rows], index=prof["samples"][rows], columns=prof["types"])[list(feature)]
    return counts.div(counts.sum(axis=1), axis=0)


def recovery(prof, splits, pb, allowed, feature) -> tuple:
    """V1: per feature type, Pearson r and mean bias between estimated and true fractions of the
    validation samples, pooled over the splits. Also returns every split's estimates for its own
    training and validation samples (each with that split's reference: the inputs of the
    within-single-cell score) and the validation residuals (V4's out-of-sample baseline)."""
    est, true, per_split, residuals = [], [], [], []
    for _, split in splits.groupby("index"):
        train = split.loc[split["side"] == "train", "sample_id"]
        val = split.loc[split["side"] == "validation", "sample_id"]
        if val.empty:   # a stratum (organ) can have no validation sample in a split
            continue
        S, depth = dc.signature(prof, train)
        mk, _ = dc.markers(S, depth, allowed)
        both = pd.concat([train, val])
        f, res = dc.fit(S[mk], pb.loc[both].to_numpy()[:, mk])
        closed = dc.close(f, prof["types"], feature).set_axis(both)
        est.append(closed.loc[val])
        true.append(true_fractions(prof, val, feature).loc[val])
        per_split.append(closed.assign(split=split["index"].iloc[0],
                                       side=np.where(both.isin(val), "validation", "train")))
        residuals.append(res[len(train):])
    est, true = pd.concat(est), pd.concat(true)
    rows = []
    for t in feature:
        r = pearsonr(est[t], true[t])[0] if true[t].std() > 0 and est[t].std() > 0 else np.nan
        rows.append({"type": t, "r": r, "bias": float((est[t] - true[t]).mean()), "n": len(est)})
    return pd.DataFrame(rows), pd.concat(per_split), np.concatenate(residuals)


def stratum_run(prof, splits, pb, bulk, allowed) -> dict:
    """Types, V1 with merging, the final reference and fractions, V2 and V4 for one stratum."""
    reference, feature = dc.type_lists(prof)
    cells = dict(zip(prof["types"], prof["n"].sum(axis=0)))
    roles = {t: ("feature" if t in feature else "nuisance" if t in reference else "dropped") for t in prof["types"]}
    prof = {**prof, "types": prof["types"][np.isin(prof["types"], reference)],
            "summed": prof["summed"][:, np.isin(prof["types"], reference)], "n": prof["n"][:, np.isin(prof["types"], reference)]}
    feature, merges = list(feature), {}
    while True:
        v1, per_split, val_residuals = recovery(prof, splits, pb, allowed, feature)
        failing = v1[~(v1["r"] >= config.V1_MIN_R)]
        if failing.empty or len(feature) <= 2:
            break
        worst = failing.sort_values("r", na_position="first")["type"].iloc[0]
        S, depth = dc.signature(prof)
        mk, _ = dc.markers(S, depth, allowed)
        partner = dc.most_similar(S[mk], prof["types"], worst, feature)
        merges[worst] = partner
        prof = dc.merge(prof, worst, partner)
        feature.remove(worst)
    v1["gate_passed"] = bool((v1["r"] >= config.V1_MIN_R).all() and v1["r"].median() >= config.V1_MEDIAN_R)
    v1["stopped_at_two"] = len(feature) <= 2 and not (v1["r"] >= config.V1_MIN_R).all()
    S, depth = dc.signature(prof)
    mk, strong = dc.markers(S, depth, allowed)
    samples = list(pb.index)
    sc_est, sc_res = dc.fit(S[mk], pb.to_numpy()[:, mk])
    bulk_est, bulk_res = dc.fit(S[mk], bulk.to_numpy()[:, mk])
    v2 = []
    for i, k in enumerate(prof["types"]):
        if k not in feature:
            continue
        others = [t for t in feature if t != k]
        without = np.delete(S[mk], i, axis=1)
        f, _ = dc.fit(without, pb.to_numpy()[:, mk])
        f = pd.DataFrame(f, columns=np.delete(prof["types"], i))[others]
        before = dc.close(sc_est, prof["types"], prof["types"])[others]
        true_k = true_fractions(prof, samples, feature).loc[samples, k].to_numpy()
        for t in others:
            v2.append({"removed": k, "type": t, "mean_shift": float((f[t].to_numpy() - before[t].to_numpy()).mean()),
                       "shift_vs_removed_r": pearsonr(f[t].to_numpy() - before[t].to_numpy(), true_k)[0]
                       if true_k.std() > 0 else np.nan})
    types = pd.DataFrame({"type": list(cells), "cells": list(cells.values()), "role": [roles[t] for t in cells]})
    types["share"] = types["cells"] / types["cells"].sum()
    types["merged_into"] = types["type"].map(merges)
    strong_by = dict(zip(prof["types"], strong))
    types["strong_markers"] = types["type"].map(strong_by)
    return {"types": types, "feature": feature, "prof": prof, "S": S, "markers": mk, "v1": v1, "v2": pd.DataFrame(v2), "merges": merges,
            "splits": per_split.rename_axis("sample_id").reset_index(), "validation_residuals": val_residuals,
            "sc": (dc.close(sc_est, prof["types"], feature).set_axis(samples), sc_res),
            "bulk": (dc.close(bulk_est, prof["types"], feature).set_axis(bulk.index), bulk_res),
            "true": true_fractions(prof, samples, feature).loc[samples]}


def fractions_table(stratum, run) -> pd.DataFrame:
    parts = []
    for side, kind, (table, residual) in (("single_cell", "estimated", run["sc"]), ("bulk", "estimated", run["bulk"]),
                                          ("single_cell", "true", (run["true"], None))):
        long = table.rename_axis("sample_id").reset_index().melt(id_vars="sample_id", var_name="type", value_name="fraction")
        if residual is not None:
            long["residual"] = long["sample_id"].map(dict(zip(table.index, residual)))
        parts.append(long.assign(stratum=stratum, side=side, kind=kind))
    return pd.concat(parts, ignore_index=True)


def write(frame, path, record, unit_cost, inputs):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(manifest.temporary(path), index=False)
    else:
        frame.to_csv(manifest.temporary(path), index=False)
    manifest.finish(path, record, unit_cost, inputs=inputs)


def run_unit(pair, unit, level, shared, bulk_all, bulk_samples, splits_all, record, inputs):
    fold_set, fold = unit
    cost = manifest.begin()
    panel = pd.Index(pd.read_csv(paths.output("04_panel", pair, "panel.csv", unit), keep_default_na=False)["gene"])
    cells = folds.training_cells(pair, fold_set, fold)
    types = cell_types(cells.obs, level)
    keep = types.notna().to_numpy()
    X = cells[keep][:, panel].X
    obs = cells.obs[keep]
    strata = obs[level["within"]].astype(str) if level["within"] else pd.Series("all", index=obs.index)
    pb_data = folds.training_pseudobulk(pair, fold_set, fold)[:, shared]
    pb_cpm = pd.DataFrame(dc.cpm(pb_data.X), index=pb_data.obs_names.astype(str), columns=shared)
    scored = folds.scored_bulk(pair, fold_set, fold, bulk_samples)["sample_id"]
    bulk = bulk_all[scored.to_numpy()]
    bulk_cpm = pd.DataFrame(dc.cpm(bulk.X), index=bulk.obs_names.astype(str), columns=shared)
    sample_stratum = pb_data.obs[level["within"]].astype(str) if level["within"] else pd.Series("all", index=pb_data.obs_names)
    bulk_stratum = bulk.obs[level["within"]].astype(str) if level["within"] else pd.Series("all", index=bulk.obs_names)
    splits = splits_all[(splits_all["fold_set"] == fold_set) & (splits_all["fold"] == fold) & (splits_all["purpose"] == config.V1_SPLITS)]
    excluded_sets = {"main": ()} | ({"no_ifn": tuple(signatures.load("lupus")["gene"])} if pair == "lupus" else {})
    out = {name: {"types": [], "markers": [], "fractions": [], "splits": [], "v1": [], "v2": [], "v4": []} for name in excluded_sets}
    for stratum in sorted(strata.unique()):
        rows = (strata == stratum).to_numpy()
        prof = dc.profiles(X[rows], obs["sample_id"].astype(str).to_numpy()[rows], types[keep].to_numpy()[rows],
                           obs["total_counts"].to_numpy()[rows])
        pb = pb_cpm.loc[sample_stratum.index[sample_stratum == stratum], panel]
        b = bulk_cpm.loc[bulk_stratum.index[bulk_stratum == stratum], panel]
        st_splits = splits[splits["sample_id"].isin(pb.index)]
        for name, excluded in excluded_sets.items():
            allowed = dc.candidates(panel, b.to_numpy(), pb.to_numpy(), excluded)
            run = stratum_run(prof, st_splits, pb, b, allowed)
            o = out[name]
            o["types"].append(run["types"].assign(stratum=stratum))
            o["markers"].append(pd.DataFrame({"stratum": stratum, "gene": panel[run["markers"]]}))
            o["fractions"].append(fractions_table(stratum, run))
            o["splits"].append(run["splits"].assign(stratum=stratum))
            o["v1"].append(run["v1"].assign(stratum=stratum))
            o["v2"].append(run["v2"].assign(stratum=stratum))
            sc_res, bulk_res = run["sc"][1], run["bulk"][1]
            v4 = {"stratum": stratum, "single_cell_median_residual": float(np.median(run["validation_residuals"])),
                  "single_cell_in_sample_median_residual": float(np.median(sc_res)),
                  "bulk_median_residual": float(np.median(bulk_res)),
                  "bulk_residual_iqr": float(np.subtract(*np.percentile(bulk_res, [75, 25])))}
            v4["reference_mismatch"] = v4["bulk_median_residual"] > config.V4_RESIDUAL_RATIO * v4["single_cell_median_residual"]
            v4["markers_in_yao"] = int(panel[run["markers"]].isin(signatures.load("lupus")["gene"]).sum()) if pair == "lupus" else None
            mono = [t for t in run["feature"] if config.MONOCYTE in t.lower()]
            if pair in ("soundlife", "lupus") and mono:
                neut = np.log1p(bulk_cpm.loc[b.index, list(config.NEUTROPHIL_SCORE)]).mean(axis=1)
                v4["neutrophil_leak_spearman"] = spearmanr(run["bulk"][0][mono].sum(axis=1).loc[b.index], neut)[0]
            o["v4"].append(pd.DataFrame([v4]))
            if name == "main" and fold_set == "crossfit":
                v3_check(pair, unit, run, level, bulk_samples, shared, panel, record, cost, inputs)
    for name, o in out.items():
        suffix = "" if name == "main" else f"_{name}"
        if name == "main":
            for key in ("types", "markers", "v1", "v2", "v4"):
                write(pd.concat(o[key], ignore_index=True), paths.output(STAGE, pair, f"{'check_' if key.startswith('v') else ''}{key}.csv", unit),
                      record, cost, inputs)
        write(pd.concat(o["fractions"], ignore_index=True), paths.output(STAGE, pair, f"fractions{suffix}.parquet", unit), record, cost, inputs)
        splits_long = pd.concat(o["splits"], ignore_index=True).melt(id_vars=["sample_id", "split", "side", "stratum"],
                                                                      var_name="type", value_name="estimated")
        write(splits_long, paths.output(STAGE, pair, f"fractions_splits{suffix}.parquet", unit), record, cost, inputs)
    v1 = pd.concat(out["main"]["v1"])
    print(f"{pair} {folds.unit_name(*unit)}: {len(v1)} feature types, median V1 r {v1['r'].median():.2f}, "
          f"min {v1['r'].min():.2f}", flush=True)


def v3_check(pair, unit, run, level, bulk_samples, shared, panel, record, cost, inputs):
    """V3 (cross-fitting units): the fold's held-out donors' bulk, deconvolved with this fold's
    reference, against their own single-cell fractions (annotated stored cells), matched by donor
    and visit. Unit-free diagnostic outputs."""
    fold_set, fold = unit
    held = folds.scored_cells(pair, fold_set, fold)
    types = cell_types(held.obs, level)
    mapped = types.notna().to_numpy()
    obs = held.obs[mapped].assign(type=merged(types[mapped], run["merges"]).to_numpy())
    counts = pd.crosstab(obs["sample_id"].astype(str), obs["type"]).reindex(columns=run["feature"], fill_value=0)
    true = counts.div(counts.sum(axis=1), axis=0)
    write(true.rename_axis("sample_id").reset_index(), paths.output(STAGE, pair, f"diagnostic/{folds.unit_name(*unit)}_scored_true.parquet"),
          record, cost, inputs)
    # their estimated fractions (Fig 2B for the cross-fitted pair): pseudobulk summed from their stored
    # cells (all of them, as stage 02's pseudobulk), CPM over the shared genes, this fold's reference
    sample_of = held.obs["sample_id"].astype(str).to_numpy()
    ids = np.unique(sample_of)
    summed = np.vstack([np.asarray(held[:, shared][sample_of == i].X.sum(axis=0)).ravel() for i in ids])
    held_cpm = pd.DataFrame(dc.cpm(summed), index=ids, columns=shared)[panel]
    f, res = dc.fit(run["S"][run["markers"]], held_cpm.to_numpy()[:, run["markers"]])
    estimated = dc.close(f, run["prof"]["types"], run["feature"]).set_axis(ids).assign(residual=res)
    write(estimated.rename_axis("sample_id").reset_index(),
          paths.output(STAGE, pair, f"diagnostic/{folds.unit_name(*unit)}_scored_estimated.parquet"), record, cost, inputs)
    sc = folds.single_cell_samples(pair).set_index("sample_id")
    bulk_key = bulk_samples.set_index("sample_id")[["donor_id", "visit"]].astype(str).agg("|".join, axis=1)
    sc_key = sc.loc[true.index, ["donor_id", "visit"]].astype(str).agg("|".join, axis=1)
    est = run["bulk"][0]
    pairs_ = [(s, b) for s, key in sc_key.items() for b in est.index if bulk_key[b] == key]
    assert pairs_, f"{pair} {unit}: no held-out draw has both a single-cell and a bulk sample"
    rows = []
    for t in run["feature"]:
        x = np.array([est.loc[b, t] for _, b in pairs_]); y = np.array([true.loc[s, t] for s, _ in pairs_])
        rows.append({"type": t, "r": pearsonr(x, y)[0] if x.std() > 0 and y.std() > 0 else np.nan,
                     "mean_shift": float((x - y).mean()), "matched_draws": len(pairs_)})
    write(pd.DataFrame(rows), paths.output(STAGE, pair, f"diagnostic/{folds.unit_name(*unit)}_v3.csv"), record, cost, inputs)


def main():
    determinism.configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER)
    args = parser.parse_args()
    for pair in [args.pair] if args.pair else PAIR_ORDER:
        level = config.LEVELS[pair]
        if level is None:
            print(f"{pair}: excluded (one cell type)", flush=True)
            continue
        record = record_for(pair)
        shared = pd.Index(pd.read_csv(paths.output("04_panel", pair, "shared_genes.csv"), keep_default_na=False)["gene"])
        bulk_all = anndata.read_h5ad(settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad")[:, shared]   # label-free
        bulk_samples = pd.read_csv(settings.OUTPUTS / "02_preprocess" / pair / "bulk_samples.csv", dtype={"sample_id": str, "donor_id": str})
        bulk_samples = bulk_samples[bulk_samples["drop_reason"].isna()]
        splits_all = pd.read_csv(settings.OUTPUTS / "03_splits" / pair / "splits.csv", dtype={"sample_id": str})
        for unit in folds.units(pair):
            names = ["types.csv", "markers.csv", "fractions.parquet", "fractions_splits.parquet", "check_v1.csv",
                     "check_v2.csv", "check_v4.csv"] + (["fractions_no_ifn.parquet", "fractions_splits_no_ifn.parquet"] if pair == "lupus" else [])
            if manifest.is_done([paths.output(STAGE, pair, n, unit) for n in names], record):
                print(f"{pair} {folds.unit_name(*unit)}: done", flush=True)
                continue
            run_unit(pair, unit, level, shared, bulk_all, bulk_samples, splits_all, record, input_paths(pair))


if __name__ == "__main__":
    main()
