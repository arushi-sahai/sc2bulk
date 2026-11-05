"""Stage 10c: concordance of stage 10's gene rankings with the bulk phenotype (Fig 5A-B, S14B-C, S9 Table). Label-bearing: reads bulk labels through sc2bulk.labels.bulk_labels (require_access first: blinded
or after the analysis-fixed tag) and writes only labelled_ outputs. Computations in sc2bulk.concordance.

Per pair, on the primary fold set: the single-cell shift on the training samples' log-CP10k pseudobulk
(single-cell labels) and the bulk shift on the scored bulk's log-CP10k (bulk labels), per scoring unit, within
organ for mouse organs; human blood, aging once over every single-cell training sample and every scored bulk
sample (each once) on the union of the fold panels. Then, per configuration, read-out (ridge; LASSO
for S14C) and k in TOP_K, the share of stage 10's top-k genes (phenotype ranking) whose shifts share a nonzero
sign, with base_all (all panel genes) and the matched reference (the expected share among genes
matched on single-cell shift size; column base_matched). Geneformer rows carry stage
10's inclusion flag; figures filter by it. Fig 5A's named labels live in figures/signature_labels.tsv, read by the figure script, not here.

Outputs, in <labelled outputs>/10c_concordance/<pair>/:
  labelled_gene_shifts.parquet   gene_shifts kind (key pseudobulk): gene, single_cell_shift, bulk_shift,
                                 signature_direction (up / down / ""), pseudobulk_top50
  labelled_concordance.parquet   concordance kind: configuration x readout x k; n_genes (the list's length:
                                 genes with attribution > 0 only), shared_share, base_all, base_matched, rho, included

  python pipeline/10c_concordance.py --pair PAIR
"""
import argparse

import anndata
import numpy as np
import pandas as pd

from sc2bulk import concordance, determinism, folds, labels, manifest, paths, preprocess, schemas, signatures, training_inputs
from sc2bulk.config import configurations, settings
from sc2bulk.config import genes as G
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

STAGE = "10c_concordance"
ATTRIBUTION = "10_attribution"


def units(pair) -> list:
    primary = next(iter(PAIRS[pair]["fold_sets"]))
    return [u for u in folds.units(pair) if u[0] == primary]


def side_table(pair, table: pd.DataFrame, ids_column="sample_id") -> pd.DataFrame:
    """group, scoring_unit_id and stratum per row."""
    by = "donor_id" if PAIRS[pair]["scoring_unit"] == "donor" else ids_column
    column = PAIRS[pair].get("strata")
    return pd.DataFrame({"group": table["group"].astype(int).to_numpy(), "scoring_unit_id": table[by].astype(str).to_numpy(),
                         "stratum": table[column].astype(str).to_numpy() if column else np.full(len(table), "all")})


def log_profiles(X, shared, genes) -> np.ndarray:
    """Counts over the shared genes -> log-CP10k (normalized over the shared genes) on the given genes."""
    return preprocess.log_normalize(X)[:, pd.Index(shared).get_indexer(genes)].toarray()


def single_cell_side(pair, genes, shared) -> tuple:
    """Every training sample of the primary fold set's units (each once): log-CP10k pseudobulk and its table."""
    tables, blocks, seen = [], [], set()
    for unit in units(pair):
        samples = folds.training_samples(pair, *unit)
        samples = samples[~samples["sample_id"].isin(seen)]
        seen |= set(samples["sample_id"])
        data = folds.training_pseudobulk(pair, *unit)[samples["sample_id"].to_numpy()]
        blocks.append(log_profiles(data[:, shared].X, shared, genes))
        tables.append(samples)
    samples = pd.concat(tables, ignore_index=True)
    return np.vstack(blocks), side_table(pair, samples)


def bulk_side(pair, genes, shared) -> tuple:
    """Every unit's scored bulk (each sample scored once) with its bulk label: log-CP10k and its table."""
    scored = pd.concat([training_inputs.scored_bulk(pair, u) for u in units(pair)], ignore_index=True)
    assert scored["sample_id"].is_unique, "a bulk sample is scored by two units"
    labelled = scored.merge(labels.bulk_labels(pair)[["sample_id", "group"]], on="sample_id", how="left", validate="one_to_one")
    assert labelled["group"].notna().all(), "a scored bulk sample has no label"
    data = anndata.read_h5ad(settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad")[labelled["sample_id"].to_numpy()]
    return log_profiles(data[:, shared].X, shared, genes), side_table(pair, labelled)


def signature_direction(pair, genes) -> list:
    """Each panel gene's signature direction: human pairs by symbol; mouse microglia by
    symbol, uppercased (Holtman's file prints mouse symbols in capitals); mouse organs (human symbols) through
    the panel's one-to-one orthologs."""
    direction = dict(signatures.load(pair)[["gene", "direction"]].to_numpy())
    if pair == "hammond":
        return [direction.get(g.upper(), "") for g in genes]
    if pair != "tms":
        return [direction.get(g, "") for g in genes]
    human = {}
    for unit in units(pair):
        mapped = pd.read_csv(paths.output("04_panel", pair, "orthologs.csv", unit), keep_default_na=False)
        human.update(dict(zip(mapped["gene"], mapped["human_symbol"])))
    return [direction.get(human.get(g, ""), "") for g in genes]


def key_columns(pair, key) -> dict:
    config = configurations.CONFIGURATIONS[key]
    unit = units(pair)[0]
    return {"pair": pair, "fold_set": unit[0], "fold": -1 if len(units(pair)) > 1 else unit[1], "row_kind": "configuration", "key": key,
            "model": config["model"], "input": config["input"], "n_cells": T.FROZEN_NCELL if config["input"] == "cells" else 0,
            "pooling": "mean" if config["model"] in schemas.EMBED_ONLY else "native", "encoder_seed": -1, "readout_seed": -1}


def write(kind, frame, path, record, cost, sources):
    path.parent.mkdir(parents=True, exist_ok=True)
    schemas.write(kind, schemas.with_neutral_keys(frame), manifest.temporary(path))
    manifest.finish(path, record, cost, inputs=sources)


def run(pair):
    labels.require_access()
    rankings = {key: paths.output(ATTRIBUTION, pair, f"{key}_ranking.parquet") for key in configurations.CONFIGURATION_ORDER}
    inclusion_path = paths.output(ATTRIBUTION, pair, "inclusion.csv")
    inclusion = pd.read_csv(inclusion_path, keep_default_na=False).set_index("configuration")
    dropped = set(inclusion.index[inclusion["failed_checks"].isin(["triaged", "pending"])])   # Geneformer triaged or pending
    present = {k: p for k, p in rankings.items() if k not in dropped}
    sources = [*folds.inputs(pair), settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad",
               settings.OUTPUTS / "02_preprocess" / pair / "bulk_samples.csv", settings.OUTPUTS / "02_preprocess" / pair / "bulk_labels.csv",
               settings.OUTPUTS / "03_splits" / pair / "donor_folds.csv", paths.output("04_panel", pair, "shared_genes.csv"),
               *[paths.output("04_panel", pair, "panel.csv", u) for u in units(pair)], inclusion_path, *present.values()]
    outputs = [paths.output(STAGE, pair, f"{settings.LABELLED}_{name}.parquet") for name in ("gene_shifts", "concordance")]
    record = {"stage": STAGE, "pair": pair, "smoke": settings.SMOKE, "blinded": settings.BLINDED,
              "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair), "inputs": manifest.input_hashes(sources)}
    if manifest.is_done(outputs, record):
        print(f"{pair}: done", flush=True)
        return
    cost = manifest.begin()
    shared = training_inputs.shared(pair)
    genes = sorted(set().union(*(training_inputs.panel(pair, u) for u in units(pair))))
    X_sc, sc_table = single_cell_side(pair, genes, shared)
    X_bulk, bulk_table = bulk_side(pair, genes, shared)
    sc = pd.Series(concordance.side_shift(X_sc, sc_table, skip_short=settings.SMOKE), index=genes)   # smoke: thin strata skipped
    bulk = pd.Series(concordance.side_shift(X_bulk, bulk_table, skip_short=settings.SMOKE), index=genes)
    rho = concordance.rho(sc, bulk)
    tables = {key: schemas.read("attributions", path) for key, path in present.items()}
    pseudobulk = tables["pseudobulk"]
    top50 = set(pseudobulk[(pseudobulk["readout"] == "ridge") & (pseudobulk["target"] == "phenotype") & pseudobulk["top50"]]["gene"])
    shifts = pd.DataFrame({"gene": genes, "single_cell_shift": sc.to_numpy(), "bulk_shift": bulk.to_numpy(),
                           "signature_direction": signature_direction(pair, genes), "pseudobulk_top50": [g in top50 for g in genes]})
    write("gene_shifts", shifts.assign(**key_columns(pair, "pseudobulk"), readout="none"), outputs[0], record, cost, sources)
    rows = []
    for key, table in tables.items():
        for readout_name in ("ridge", "lasso"):
            ranking = table[(table["readout"] == readout_name) & (table["target"] == "phenotype")].sort_values("rank")
            assert set(ranking["gene"]) == set(genes), f"{pair} {key}: the ranking's genes aren't the panel union"
            for k in G.TOP_K:   # stage 10's top-k list (genes with attribution > 0 only, so possibly shorter than k)
                chosen = list(ranking.loc[ranking[f"top{k}"], "gene"])
                rows.append({**key_columns(pair, key), "readout": readout_name, "k": k, "n_genes": len(chosen), "rho": rho,
                             **concordance.shares(chosen, sc, bulk), "included": bool(inclusion["included"].get(key, True))})
    write("concordance", pd.DataFrame(rows), outputs[1], record, cost, sources)
    print(f"{pair}: concordance written (rho {rho:.3f})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER, required=True)
    args = parser.parse_args()
    determinism.configure()
    run(args.pair)


if __name__ == "__main__":
    main()
