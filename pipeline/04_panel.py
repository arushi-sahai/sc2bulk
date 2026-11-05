"""Stage 04: the gene panel, per unit (fold set, fold), from that unit's training single-cell
data only.

Per pair:
  shared genes  the single-cell gene axis intersected with the bulk gene axis (fold-independent)
Per unit:
  panel         among the shared genes detected in >= PANEL_MIN_CELL_FRACTION of the unit's
                training cells, the top PANEL_SIZE by scanpy's seurat-flavor normalized
                dispersion of the log-normalized training cells (normalized over all shared
                genes, the one normalization every stage uses); ties broken by gene symbol

Training cells are the stored cells of stage 02 (<= 1,000 per sample) of the unit's training
samples (sc2bulk.folds.training_samples), the cells every model of that unit sees, including
the samples later held out for validation (label-free). No bulk expression or label, and no
scored donor's single-cell data, is used; bulk contributes only its gene names, for the
intersection. Ranking follows scanpy exactly, including its small-bin behaviour (a gene alone
in its mean bin gets a normalized dispersion of 1; disclosed). Outputs, in
<outputs>/04_panel/<pair>/:
  shared_genes.csv          the shared genes, with their Ensembl ids
  <unit>/panel.csv          the panel: gene, rank, fraction_detected (of the unit's training
                            cells), ensembl, and scanpy's columns: means (log1p of the mean
                            CP10k), dispersions (log), dispersions_norm
  <unit>/orthologs.csv      mouse pairs only: each panel gene's one-to-one human ortholog for
                            the foundation models (sc2bulk.orthologs): gene, mouse_ensembl,
                            human_ensembl, human_symbol, status, high_confidence
  orthologs_shared.csv      mouse pairs only: the same for every shared gene (the FMs'
                            full-gene-set embeddings, S12A)
where <unit> is <fold_set>_<fold> (e.g. all_0, crossfit_2).

  python pipeline/04_panel.py [--pair PAIR]
"""
import argparse

import anndata
import numpy as np
import pandas as pd

from sc2bulk import determinism, folds, manifest, orthologs, preprocess
from sc2bulk.config import settings, sources
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

IN = settings.OUTPUTS / "02_preprocess"
SPLITS = settings.OUTPUTS / "03_splits"
RAW = settings.RAW
OUT = settings.OUTPUTS / "04_panel"
ORTHOLOG_INPUTS = [("ensembl", name) for name in (*sources.ORTHOLOG_FILES.values(), sources.HUMAN_GTF)]


def unit_outputs(pair):
    return ("panel.csv",) + (("orthologs.csv",) if PAIRS[pair]["species"] == "mouse" else ())


def outputs_for(pair):
    shared = ["shared_genes.csv"] + (["orthologs_shared.csv"] if PAIRS[pair]["species"] == "mouse" else [])
    return [OUT / pair / name for name in shared] + [OUT / pair / folds.unit_name(*u) / name
                                                    for u in folds.units(pair) for name in unit_outputs(pair)]


def input_paths(pair):
    return folds.inputs(pair) + [IN / pair / "bulk.h5ad", SPLITS / pair / "donor_folds.csv"]


def record_for(pair) -> dict:
    return {"stage": "04_panel", "pair": pair, "smoke": settings.SMOKE,
            "code": manifest.stage_code_hash(settings.REPO / "pipeline" / "04_panel.py", pair),
            "inputs": manifest.input_hashes(input_paths(pair)),   # by content: stale only if 02/03 outputs change
            "panel_size": settings.PANEL_SIZE, "min_cell_fraction": settings.PANEL_MIN_CELL_FRACTION,
            "target_sum": settings.TARGET_SUM, "bins": settings.N_DISPERSION_BINS,
            "orthologs": {name: sources.DOWNLOADS[(d, name)][1] for d, name in ORTHOLOG_INPUTS}
            if PAIRS[pair]["species"] == "mouse" else None}


def write_csv(table, path, record, unit, inputs):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(manifest.temporary(path), index=False)
    manifest.finish(path, record, unit, inputs=inputs)


def panel_of(X, genes):
    """The panel from training cells X (cells x shared genes, raw counts)."""
    detected = np.asarray((X > 0).sum(axis=0)).ravel() / X.shape[0]
    table = genes.assign(fraction_detected=detected)
    keep = detected >= settings.PANEL_MIN_CELL_FRACTION
    stats = preprocess.seurat_dispersion(preprocess.log_normalize(X)[:, np.flatnonzero(keep)])
    ranked = table[keep].reset_index(drop=True).join(stats)
    ranked = ranked.sort_values(["dispersions_norm", "gene"], ascending=[False, True], na_position="last")
    panel = ranked.head(settings.PANEL_SIZE).reset_index(drop=True)
    panel.insert(1, "rank", np.arange(1, len(panel) + 1))
    assert len(panel) == settings.PANEL_SIZE and panel["dispersions_norm"].notna().all()
    assert (panel["fraction_detected"] >= settings.PANEL_MIN_CELL_FRACTION).all()
    return panel, int(keep.sum())


def main():
    determinism.configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER)
    args = parser.parse_args()
    for pair in [args.pair] if args.pair else PAIR_ORDER:
        out = OUT / pair
        record = record_for(pair)
        if manifest.is_done(outputs_for(pair), record):
            print(f"{pair}: done", flush=True)
            continue
        unit = manifest.begin()
        inputs = input_paths(pair)
        var = folds.single_cell_genes(pair)
        bulk_genes = anndata.read_h5ad(IN / pair / "bulk.h5ad", backed="r").var_names
        shared = var.index.intersection(bulk_genes).sort_values()
        genes = pd.DataFrame({"gene": shared, "ensembl": var.loc[shared, "ensembl"].to_numpy()})
        write_csv(genes, out / "shared_genes.csv", record, unit, inputs)
        if PAIRS[pair]["species"] == "mouse":
            pairs_table, names = orthologs.one2one(RAW), orthologs.human_names(RAW)
            write_csv(orthologs.map_genes(genes, pairs_table, names), out / "orthologs_shared.csv", record, unit,
                      inputs + [RAW / d / n for d, n in ORTHOLOG_INPUTS])
        for fold_set, fold in folds.units(pair):
            cells = folds.training_cells(pair, fold_set, fold)
            X = cells[:, shared].X.tocsr()
            n_cells = cells.n_obs
            del cells   # keep one copy of the counts in memory
            panel, n_filtered = panel_of(X, genes)
            del X
            unit_dir = out / folds.unit_name(fold_set, fold)
            if PAIRS[pair]["species"] == "mouse":
                mapped = orthologs.map_genes(panel, pairs_table, names)
                write_csv(mapped, unit_dir / "orthologs.csv", record, unit, inputs + [RAW / d / n for d, n in ORTHOLOG_INPUTS])
                print(f"{pair} {fold_set} {fold}: orthologs {mapped['status'].value_counts().to_dict()}; low-confidence "
                      f"{(mapped['high_confidence'] == False).sum()}; no human symbol "
                      f"{(mapped['status'].eq('mapped') & mapped['human_symbol'].eq('')).sum()}", flush=True)
            write_csv(panel, unit_dir / "panel.csv", record, unit, inputs)
            print(f"{pair} {fold_set} {fold}: {len(shared)} shared genes, {n_filtered} detected in >= "
                  f"{settings.PANEL_MIN_CELL_FRACTION:.0%} of {n_cells} training cells, panel {len(panel)}", flush=True)


if __name__ == "__main__":
    main()
