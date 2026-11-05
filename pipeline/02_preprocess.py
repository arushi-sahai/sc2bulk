"""Stage 02: per-pair preprocessing of the raw counts (rules in sc2bulk.preprocess).

Single-cell side, per sample: features summed onto one gene axis (the union of the side's
gene symbols); per-cell QC recorded for every cell; cells failing QC removed; a seeded
random subsample of at most MAX_STORED_CELLS post-QC cells stored; pseudobulk = the summed
counts of the stored cells, the same cells every model sees.
Bulk side: features summed onto its gene axis; library size and genes detected recorded;
libraries below BULK_MIN_LIBRARY and samples outside both 0/1 groups dropped; samples put in
a seeded order (SHA-256 of pair:sample_id), so row order carries no label.

Bulk-label firewall: the bulk labels (label_value, the 0/1 group and
the pair's bulk_label_columns) are written only to bulk_labels.csv, read only through
sc2bulk.labels by whitelisted code. bulk.h5ad and bulk_samples.csv carry none of them.

Nothing is fit across samples and no gene is filtered here. Outputs (raw counts only), in
<outputs>/02_preprocess/<pair>/:
  cells.h5ad               stored single-cell cells x genes, with per-cell QC and cell types
  cell_qc.csv.gz           every single-cell cell before QC: sample, cell type, QC values
  pseudobulk.h5ad          single-cell samples x genes: summed counts of the stored cells
  bulk.h5ad                kept bulk samples x genes (no label columns)
  bulk_labels.csv          kept bulk samples' labels: behind the firewall
  single_cell_samples.csv  every single-cell sample, kept or dropped (with the reason)
  bulk_samples.csv         every bulk sample, kept or dropped (with the reason; no label columns)
  merged_symbols.csv       gene symbols carried by several features (counts summed)
  cells_one_type.h5ad      Fig 6C: up to 1,000 post-QC cells of the pair's single cell type per
                           sample (samples with fewer than ONE_TYPE_MIN_CELLS of it not stored)
  pseudobulk_one_type.h5ad the summed counts of those cells

  python pipeline/02_preprocess.py [--pair PAIR]
"""
import argparse
import importlib

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk import codehash, determinism, manifest, preprocess
from sc2bulk.config import settings, smoke, sources
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

RAW = settings.RAW
OUT = settings.OUTPUTS / "02_preprocess"
OUTPUTS = ("cells.h5ad", "cell_qc.csv.gz", "pseudobulk.h5ad", "bulk.h5ad", "merged_symbols.csv",
           "single_cell_samples.csv", "bulk_labels.csv", "bulk_samples.csv")
ONE_TYPE_OUTPUTS = ("cells_one_type.h5ad", "pseudobulk_one_type.h5ad")


def outputs_for(pair):
    return OUTPUTS + (ONE_TYPE_OUTPUTS if "one_type" in PAIRS[pair] else ())


def record_for(pair) -> dict:
    """Everything that determines a pair's stage-02 outputs: settings, the pinned input
    checksums and a hash of the code."""
    directories = sources.PAIR_DIRECTORIES[pair]
    inputs = {f"{d}/{n}": spec[-1] for (d, n), spec in {**sources.DOWNLOADS, **sources.MANUAL, **sources.SYNAPSE}.items()
              if d in directories}
    # the pair's data reader is loaded by name (importlib), so it is added to the hash explicitly
    return {"stage": "02_preprocess", "pair": pair, "smoke": settings.SMOKE, "inputs": inputs,
            "code": manifest.stage_code_hash(settings.REPO / "pipeline" / "02_preprocess.py", pair,
                                             codehash.extra("02_preprocess", pair))}


def gene_table(genes, annotation: pd.DataFrame) -> pd.DataFrame:
    """var for a gene axis: each symbol with its Ensembl id(s) in the pair's annotation."""
    ids = annotation.groupby("symbol")["ensembl"].agg(lambda x: ";".join(sorted(set(x))))
    return pd.DataFrame({"ensembl": ids.reindex(genes).fillna("").to_numpy()}, index=genes)


def single_cell(pair, reader, genes):
    groups = PAIRS[pair]["groups"]["single_cell"]
    samples, qc_tables, stored, pseudobulk, merged = [], [], [], [], {}
    one_type = PAIRS[pair].get("one_type")
    stored_type = []
    smoke_groups, smoke_donors = {}, set()
    for sample, X, symbols, cells in reader.single_cell_samples(RAW):
        sample["group"] = None
        if sample["drop_reason"] is None:
            sample["group"] = preprocess.group_of(sample["label_value"], groups)
            if sample["group"] is None:
                sample["drop_reason"] = "label outside both groups"
        if settings.SMOKE and sample["drop_reason"] is None:   # the smoke subset is by holdout group
            key = sample["holdout_group"]
            if key not in smoke_groups:
                if list(smoke_groups.values()).count(sample["group"]) >= smoke.HOLDOUT_GROUPS_PER_LABEL:
                    sample["drop_reason"] = "smoke run"
                else:
                    smoke_groups[key] = sample["group"]
            if sample["drop_reason"] is None:
                smoke_donors.add(sample["donor_id"])
        if sample["drop_reason"] is not None:
            samples.append(sample)
            continue
        preprocess.assert_counts(X, f"{pair} {sample['sample_id']}")
        merged.update(preprocess.merged_symbols(symbols).to_dict())
        X = preprocess.to_gene_axis(X, symbols, genes)
        qc = preprocess.cell_qc(X, genes, PAIRS[pair]["species"],
                                PAIRS[pair].get("qc_max_percent_mito", settings.QC_MAX_PERCENT_MITO))
        qc.insert(0, "sample_id", sample["sample_id"])
        qc = pd.concat([cells.reset_index(drop=True), qc], axis=1)
        qc_tables.append(qc.astype({c: "category" for c in qc.columns if qc[c].dtype == object and c != "cell_id"}))
        passed = np.flatnonzero(qc["passed_qc"].to_numpy())
        assert len(passed) >= settings.MIN_CELLS_PER_SAMPLE, (
            f"{pair} {sample['sample_id']}: {len(passed)} cells pass QC, below the floor of "
            f"{settings.MIN_CELLS_PER_SAMPLE}; the floor must drop nothing (decided 2026-10-04)")
        if pair == "soundlife":
            # the slim's cells stand in for all cells only if the full <= 1,000 can be drawn from
            # them, or the slim is the whole sample (one sample, KT00294, has 984 cells in all)
            assert len(passed) >= settings.MAX_STORED_CELLS or len(qc) == sample["cells_in_portal_file"], \
                f"soundlife {sample['sample_id']}: {len(passed)} slim cells pass QC"
        keep = passed[preprocess.stored_cells(pair, sample["sample_id"], len(passed))]
        cell_obs = qc.iloc[keep].reset_index(drop=True)
        for column in ("donor_id", "holdout_group", "organ", "release_cohort"):
            if column in sample:
                cell_obs[column] = sample[column]
        stored.append((X[keep].astype(np.int32), cell_obs))
        pseudobulk.append(preprocess.summed(X[keep]))
        sample.update(cells_before_qc=len(qc), cells_after_qc=len(passed), cells_stored=len(keep))
        if one_type:
            types = one_type["types"][sample[one_type["by"]] if one_type["by"] else None]
            is_type = qc[one_type["column"]].iloc[passed].isin(types).to_numpy()
            type_keep = passed[preprocess.stored_type_cells(pair, sample["sample_id"], is_type)]
            sample.update(one_type_cells=int(is_type.sum()), one_type_stored=bool(is_type.sum() >= settings.ONE_TYPE_MIN_CELLS))
            if sample["one_type_stored"]:
                type_obs = qc.iloc[type_keep].reset_index(drop=True)
                for column in ("donor_id", "holdout_group", "organ", "release_cohort"):
                    if column in sample:
                        type_obs[column] = sample[column]
                stored_type.append((sample["sample_id"], X[type_keep].astype(np.int32), type_obs))
        samples.append(sample)
    samples = pd.DataFrame(samples)
    for column in ("group", "cells_before_qc", "cells_after_qc", "cells_stored", "one_type_cells"):
        if column in samples:
            samples[column] = samples[column].astype("Int64")
    if "one_type_stored" in samples:
        samples["one_type_stored"] = samples["one_type_stored"].eq(True)   # False for dropped samples
    qc = pd.concat(qc_tables, ignore_index=True)
    if one_type and not settings.SMOKE:
        assert_most_abundant(qc, samples, one_type)
    return samples, qc, stored, pseudobulk, merged, stored_type, smoke_donors


def assert_most_abundant(qc, samples, one_type):
    """Fig 6C's configured type must be the most abundant post-QC type (its labels counted
    together, against every other single label), by pooled cells and by mean per-sample
    fraction, within each value of `by` (e.g. each TMS organ)."""
    passed = qc[qc["passed_qc"]]
    if one_type["by"]:
        passed = passed.merge(samples[["sample_id", one_type["by"]]], on="sample_id")
    for key, types in one_type["types"].items():
        cells = passed if key is None else passed[passed[one_type["by"]] == key]
        label = cells[one_type["column"]].astype(str).where(~cells[one_type["column"]].isin(types), "TYPE")
        pooled = label.value_counts()
        fractions = pd.crosstab(cells["sample_id"], label, normalize="index").mean()
        assert pooled.idxmax() == "TYPE" and fractions.idxmax() == "TYPE", \
            f"{key}: {types} is not the most abundant type ({pooled.idxmax()}, {fractions.idxmax()})"


def bulk(pair, reader, smoke_donors):
    X, symbols, samples = reader.bulk(RAW)
    preprocess.assert_counts(X, f"{pair} bulk")
    assert samples["sample_id"].is_unique, f"{pair}: bulk sample ids are not unique"
    genes = preprocess.gene_axis(symbols)
    X = preprocess.to_gene_axis(X, symbols, genes)
    order = np.argsort([preprocess.sample_seed(pair, s) for s in samples["sample_id"]])
    samples, X = samples.iloc[order].reset_index(drop=True), X[order]
    samples["library_size"] = np.asarray(X.sum(axis=1)).ravel()
    samples["genes_detected"] = np.diff(X.indptr)
    samples["group"] = pd.array([preprocess.group_of(v, PAIRS[pair]["groups"]["bulk"]) for v in samples["label_value"]], dtype="Int64")
    reason = samples["drop_reason"].astype(object)
    reason = reason.where(reason.notna(), np.where(samples["library_size"] < settings.BULK_MIN_LIBRARY,
                                                   "library below 1M counts", None))
    reason = reason.where(reason.notna(), np.where(samples["group"].isna(), "label outside both groups", None))
    samples["drop_reason"] = reason
    if settings.SMOKE and "donors" in PAIRS[pair]["expected"]:   # the smoke donors' bulk samples
        outside = samples["drop_reason"].isna() & ~samples["donor_id"].isin(smoke_donors)
        samples.loc[outside, "drop_reason"] = "smoke run"
    elif settings.SMOKE:
        kept = samples["drop_reason"].isna()
        rank = samples[kept].groupby("group").cumcount()
        samples.loc[rank.index[rank >= smoke.BULK_SAMPLES_PER_GROUP], "drop_reason"] = "smoke run"
    return samples, X, genes, preprocess.merged_symbols(symbols).to_dict()


def write_h5ad(adata, path, settings_record, unit):
    adata.write_h5ad(manifest.temporary(path))
    manifest.finish(path, settings_record, unit)


def write_csv(table, path, settings_record, unit):
    # gzip without a timestamp in its header, so a rerun writes identical bytes
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    table.to_csv(manifest.temporary(path), index=False, compression=compression)
    manifest.finish(path, settings_record, unit)


def main():
    determinism.configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER)
    args = parser.parse_args()
    for pair in [args.pair] if args.pair else PAIR_ORDER:
        out = OUT / pair
        record = record_for(pair)
        if manifest.is_done([out / name for name in outputs_for(pair)], record):
            print(f"{pair}: done", flush=True)
            continue
        unit = manifest.begin()
        out.mkdir(parents=True, exist_ok=True)
        reader = importlib.import_module(f"sc2bulk.data.{pair}")
        annotation = reader.ensembl(RAW)

        print(f"{pair}: single-cell", flush=True)
        genes = preprocess.gene_axis(reader.single_cell_genes(RAW))
        if pair != "tms":   # TMS's provider removed mitochondrial genes
            assert genes.str.startswith(preprocess.MITO_PREFIX[PAIRS[pair]["species"]]).sum() > 0, f"{pair}: no mitochondrial genes"
        samples, qc, stored, pseudobulk, merged_sc, stored_type, smoke_donors = single_cell(pair, reader, genes)
        kept = samples[samples["drop_reason"].isna()].reset_index(drop=True)
        assert kept["sample_id"].is_unique, f"{pair}: single-cell sample ids are not unique"
        if not settings.SMOKE:
            assert len(kept) == PAIRS[pair]["expected"]["single_cell"], f"{pair}: {len(kept)} single-cell samples kept"
            for group, n in PAIRS[pair]["expected"].get("single_cell_groups", {}).items():
                assert (kept["group"] == group).sum() == n, f"{pair}: single-cell group {group} count"
        assert set(kept["group"]) == {0, 1}
        var = gene_table(genes, annotation)
        cells = anndata.AnnData(sp.vstack([X for X, _ in stored], format="csr"),
                                obs=pd.concat([obs for _, obs in stored], ignore_index=True).set_index("cell_id"),
                                var=var)
        assert cells.obs_names.is_unique and cells.obs["passed_qc"].all()
        sample_obs = kept.drop(columns="drop_reason").set_index("sample_id")
        sample_obs = sample_obs.astype({c: str for c in sample_obs.columns if sample_obs[c].dtype == object})
        pb = anndata.AnnData(np.vstack(pseudobulk).astype(np.int64), obs=sample_obs, var=var)

        print(f"{pair}: bulk", flush=True)
        bulk_samples, Xb, bulk_genes, merged_bulk = bulk(pair, reader, smoke_donors)
        keep = bulk_samples["drop_reason"].isna().to_numpy()
        kept_bulk = bulk_samples[keep].reset_index(drop=True)
        if not settings.SMOKE:
            assert len(kept_bulk) == PAIRS[pair]["expected"]["bulk"], f"{pair}: {len(kept_bulk)} bulk samples kept"
            if "bulk_donors" in PAIRS[pair]["expected"]:
                assert kept_bulk["donor_id"].nunique() == PAIRS[pair]["expected"]["bulk_donors"], f"{pair}: bulk donor count"
            for group, n in PAIRS[pair]["expected"].get("bulk_groups", {}).items():
                assert (kept_bulk["group"] == group).sum() == n, f"{pair}: bulk group {group} count"
        label_columns = ["label_value", "group", *PAIRS[pair]["bulk_label_columns"]]
        labels = kept_bulk[["sample_id", *label_columns]]
        bulk_obs = kept_bulk.drop(columns=["drop_reason", *label_columns]).set_index("sample_id")
        bulk_obs = bulk_obs.astype({c: str for c in bulk_obs.columns if bulk_obs[c].dtype == object})
        bulk_adata = anndata.AnnData(Xb[keep].astype(np.int64).toarray(), obs=bulk_obs, var=gene_table(bulk_genes, annotation))
        if "donors" in PAIRS[pair]["expected"] and not settings.SMOKE:   # the same donors on both sides
            assert kept["donor_id"].nunique() == kept_bulk["donor_id"].nunique() == PAIRS[pair]["expected"]["donors"]
            assert set(kept["donor_id"]) == set(kept_bulk["donor_id"]), f"{pair}: donors differ between sides"

        merged = pd.DataFrame([{"side": side, "symbol": s, "features": n}
                               for side, m in (("single_cell", merged_sc), ("bulk", merged_bulk)) for s, n in m.items()],
                              columns=["side", "symbol", "features"])
        if "one_type" in PAIRS[pair]:
            type_ids = [sid for sid, _, _ in stored_type]
            cells_type = anndata.AnnData(sp.vstack([X for _, X, _ in stored_type], format="csr"),
                                         obs=pd.concat([obs for _, _, obs in stored_type], ignore_index=True).set_index("cell_id"),
                                         var=var)
            pb_type = anndata.AnnData(np.vstack([preprocess.summed(X) for _, X, _ in stored_type]).astype(np.int64),
                                      obs=sample_obs.loc[type_ids], var=var)
            assert cells_type.obs_names.is_unique and set(cells_type.obs["sample_id"]) == set(type_ids)
            write_h5ad(cells_type, out / "cells_one_type.h5ad", record, unit)
            write_h5ad(pb_type, out / "pseudobulk_one_type.h5ad", record, unit)
        write_h5ad(cells, out / "cells.h5ad", record, unit)
        write_csv(qc, out / "cell_qc.csv.gz", record, unit)
        write_h5ad(pb, out / "pseudobulk.h5ad", record, unit)
        write_h5ad(bulk_adata, out / "bulk.h5ad", record, unit)
        write_csv(merged, out / "merged_symbols.csv", record, unit)
        write_csv(samples, out / "single_cell_samples.csv", record, unit)
        write_csv(labels, out / "bulk_labels.csv", record, unit)
        write_csv(bulk_samples.drop(columns=label_columns), out / "bulk_samples.csv", record, unit)   # last: marks the pair done
        print(f"{pair}: {len(kept)} single-cell samples, {cells.n_obs} stored cells, "
              f"{len(kept_bulk)} bulk samples", flush=True)


if __name__ == "__main__":
    main()
