"""Stage 10: gene attribution. Label-free: reads stage 08's heads files only (single-cell side; never its
predictions), stage 06's checkpoints and embeddings, stage 07's embeddings and readable-gene maps (a gene sharing
its token with another is credited to none), and the single-cell stores through sc2bulk.folds; no
bulk label (manifest.LABELS_READ is asserted False).

Per unit of the pair's primary fold set (human blood, aging: its four folds; split67 isn't used) and
configuration, two targets on every single-cell training sample (bulk is never attributed):
  phenotype           the read-out-averaged head's prediction (ridge, and LASSO for S14C)
  modality_direction  u . e, u = scored bulk's mean embedding - training samples' (learned cells configurations)
Methods (sc2bulk.attribution): pseudobulk coefficient x value (gradient x input of a linear head); the set
encoders and CytoSet on pseudobulk gradient x input of their log-CP10k input through TorchInputs, on stage 06's
10 test bags (reproduction asserted per sample); scVI gradient x input of log1p(counts) through its encoder's
mean; scGPT gradient x input of the bin values on stage 07's tokens, 50 cells per sample; Geneformer
in-silico deletion of every gene in each of 50 cells' (or the profile's) input, re-encoded by stage 07's
encoder, delta e stored per gene and the heads applied at gather (so it is head-free and can run before
stage 08). Human blood, aging: the cell and sample draws are pair-level, everything else per fold.

Model runs (`--model`; environment in parentheses):
  set (main)          pseudobulk, CytoSet on cells and on pseudobulk, Set Transformer; per unit and encoder seed
  scrat (scrat), scvi (scvi)
  scgpt (scgpt)       both configurations, per unit (needs pilot_scgpt.json: `--model scgpt --pilot`)
  geneformer (geneformer)  the deletion, per unit and sample (`--shard i --shards n`: every n-th sample); needs
                      pilot_geneformer.json (`--model geneformer --pilot`) and SC2BULK_GENEFORMER_GO=1, set only
                      last of all.
                      `--screen1`: the two-stage screen's stage 1 (only when the pilot selects the screen);
                      `--joint`: the checks' joint deletions (after stage 08: needs the heads)
  --gather (main)     Geneformer's attributions from delta e, the rankings over seeds and folds, the checks,
                      inclusion, the level difference and the modality summary. `--geneformer-pending`: without
                      Geneformer, whose deletion runs last; its inclusion rows read "pending", and the
                      gather is rerun without the flag once the deletion is done

Outputs, in <outputs>/10_attribution/<pair>/:
  <unit>/<configuration>/s<e>.parquet         attributions kind, one encoder seed (-1: fixed representations):
                                              readout x target x gene; attribution (mean |A|), rank, mean_signed,
                                              mean_abs_half_a/b (donor halves), n_present (GF cells also
                                              mean_abs_cells_a/b)
  <unit>/<configuration>/completeness_s<e>.csv  sample_id, readout, target, sum_attribution, delta_prediction,
                                              reproduction (max |delta| vs stage 06, or cosine vs stage 07)
  <unit>/geneformer/<configuration>/delta/<sample>.npz, joint/<sample>_<half>.npz, screen1/...   (delta, screen1: credited tokens
                                              only; <sample> URL-encoded)
  <configuration>_ranking.parquet             the ranking over seeds (and folds): encoder_seed -1, human blood,
                                              aging fold -1; top25/top50/top100 (genes with attribution > 0 only, so
                                              a list can be shorter than k; its length is a checks row)
  checks.csv, inclusion.csv, curve.csv, modality_summary.csv, <unit>/level_difference.csv
                                              (Geneformer's inclusion, human blood, aging: the gate on the folds'
                                              pooled checks, fold -1; the per-fold values reported beside it)
  pilot/pilot_geneformer.json, pilot/pilot_scgpt.json (mouse microglia)

  python pipeline/10_attribution.py --pair PAIR --model {set,scrat,scvi,scgpt,geneformer} [--set UNIT] [--seed E]
  python pipeline/10_attribution.py --pair PAIR --gather
"""
import argparse
import hashlib
import json
import os
import runpy
import time
from pathlib import Path
from urllib.parse import quote

import anndata
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from sc2bulk import attribution, bags, codehash, determinism, folds, harness, manifest, paths, readout, schemas, seeds, training_inputs
from sc2bulk.config import configurations, deconvolution, representations, settings
from sc2bulk.config import genes as G
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

STAGE = "10_attribution"
# stages 06, 07 and 08's functions, not their mains (their scripts are in this stage's code hash: codehash.extra)
S06 = runpy.run_path(str(settings.REPO / "pipeline" / "06_train.py"), run_name="stage06")   # network
S07 = runpy.run_path(str(settings.REPO / "pipeline" / "07_embed.py"), run_name="stage07")   # load_model
S08 = runpy.run_path(str(settings.REPO / "pipeline" / "08_readout.py"), run_name="stage08")   # fm_cells, bag_means
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAINED = {"cytoset_cells": ("A", "cytoset"), "cytoset_pseudobulk": ("B", "cytoset"), "set_transformer_cells": ("A", "set_transformer"),
           "scrat_cells": ("A", "scrat"), "scvi_cells": ("J", "scvi")}
MODELS = {"set": ("pseudobulk", "cytoset_cells", "cytoset_pseudobulk", "set_transformer_cells"), "scrat": ("scrat_cells",),
          "scvi": ("scvi_cells",), "scgpt": ("scgpt_cells", "scgpt_pseudobulk"), "geneformer": ("geneformer_cells", "geneformer_pseudobulk")}
MODALITY = {"cytoset_cells", "set_transformer_cells", "scrat_cells", "scvi_cells", "scgpt_cells", "geneformer_cells"}
READOUTS = ("ridge", "lasso")
GENEFORMER = ("geneformer_cells", "geneformer_pseudobulk")


# ---------------------------------------------------------------------------------------------- shared readers


def units(pair) -> list:
    """The primary fold set's units (human blood, aging: crossfit 0-3)."""
    primary = next(iter(PAIRS[pair]["fold_sets"]))
    return [u for u in folds.units(pair) if u[0] == primary]


def out_dir(pair, unit=None):
    return paths.output(STAGE, pair, "x", unit).parent


def strata(pair, table: pd.DataFrame) -> np.ndarray:
    column = PAIRS[pair].get("strata")
    return table[column].astype(str).to_numpy() if column else np.full(len(table), "all")


def heads_file(pair, unit, key) -> Path:
    """Stage 08's primary heads file (never its predictions), asserted current."""
    path = paths.output("08_readout", pair, f"{key}_heads.parquet", unit)
    assert path.name.endswith("_heads.parquet")
    assert manifest.read(path)["settings"]["code"] == manifest.stage_code_hash(settings.REPO / "pipeline" / "08_readout.py", pair), \
        f"{path}: stale stage 08 heads"
    return path


def read_heads(pair, unit, key, per_seed=False) -> tuple:
    return attribution.heads(schemas.read("heads", heads_file(pair, unit, key)), per_seed)


def key_columns(pair, unit, key) -> dict:
    config = configurations.CONFIGURATIONS[key]
    return {"pair": pair, "fold_set": unit[0], "fold": unit[1], "row_kind": "configuration", "key": key, "model": config["model"],
            "input": config["input"], "n_cells": T.FROZEN_NCELL if config["input"] == "cells" else 0,
            "pooling": "mean" if config["model"] in schemas.EMBED_ONLY else "native"}


def method(key) -> str:
    return "deletion" if key in GENEFORMER else "gradient_x_input"


def record_for(pair, unit, what, sources, **extra) -> dict:
    return {"stage": STAGE, "pair": pair, "unit": unit, "what": what, "smoke": settings.SMOKE, **extra,
            "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair, codehash.extra(STAGE, pair)),
            "inputs": manifest.input_hashes(sources)}


def write_attributions(path, keys, e, key, results, record, cost, sources):
    """results: [(readout, target, attribution.summarize table)] -> one attributions table, ranked per (readout, target)."""
    frames = []
    for readout_name, target, table in results:
        frames.append(table.assign(**keys, encoder_seed=e, readout=readout_name, readout_seed=-1, target=target, method=method(key),
                                   rank=attribution.ranks(table.set_index("gene")["attribution"]).to_numpy()))
    path.parent.mkdir(parents=True, exist_ok=True)
    schemas.write("attributions", schemas.with_neutral_keys(pd.concat(frames, ignore_index=True)), manifest.temporary(path))
    manifest.finish(path, record, cost, inputs=sources)


def write_csv(path, table, record, cost, sources=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(manifest.temporary(path), index=False)
    manifest.finish(path, record, cost, inputs=sources)


def targets_of(heads, e, stratum, modality=None) -> dict:
    """{(target, readout): head vector} for one sample's stratum."""
    out = {("phenotype", r): heads[(e, r)][stratum][0] for r in READOUTS}
    if modality is not None:
        out[("modality_direction", "none")] = modality[stratum][0]
    return out


def results_of(per_sample: list, ids, present, halves, genes) -> list:
    """per_sample: [{(target, readout): per-gene A}] in ids order -> [(readout, target, summary table)]."""
    keys = list(per_sample[0])
    return [(r, t, attribution.summarize(np.vstack([a[(t, r)] for a in per_sample]), present, halves.loc[ids].to_numpy(), genes))
            for t, r in keys]


def completeness_rows(ids, per_sample_sum, per_sample_delta, reproduction) -> pd.DataFrame:
    rows = []
    for s, sums, deltas, rep in zip(ids, per_sample_sum, per_sample_delta, reproduction):
        rows += [{"sample_id": s, "target": t, "readout": r, "sum_attribution": float(sums[(t, r)]), "delta_prediction": deltas[(t, r)],
                  "reproduction": rep} for t, r in sums]
    return pd.DataFrame(rows)


def stage06_embedding(pair, unit, key, e) -> tuple:
    """Stage 06's embeddings of encoder seed e: (training samples' rows by sample id, scored bulk's primary rows
    by sample id), asserted current."""
    row, encoder = TRAINED[key]
    path = paths.output("06_train", pair, row, unit) / encoder / "embeddings" / f"s{e}.h5ad"
    assert manifest.read(path)["settings"]["code"] == manifest.stage_code_hash(settings.REPO / "pipeline" / "06_train.py", pair), \
        f"{path}: stale stage 06 embedding"
    data = anndata.read_h5ad(path)
    obs = data.obs.assign(row=np.arange(data.n_obs))
    single = obs[obs["side"].isin(["single_cell_train", "single_cell_validation"])].set_index("sample_id")["row"]
    bulk = obs[(obs["side"] == "bulk") & (obs["variant"] == "primary")].set_index("sample_id")["row"]
    X = np.asarray(data.X, dtype=np.float64)
    return pd.DataFrame(X[single.to_numpy()], index=single.index), pd.DataFrame(X[bulk.to_numpy()], index=bulk.index)


def fm_embedding(pair, unit, model, what, ids) -> pd.DataFrame:
    """Stage 07's embeddings of the unit's panel (cells, pseudobulk or bulk) for the given ids (rows by id)."""
    path = paths.output("07_embed", pair, f"{model}_panel_{what}_all.h5ad", unit)
    assert tuple(manifest.read(path)["settings"]["unit"]) == tuple(unit), f"{path}: not the unit's own panel"
    data = anndata.read_h5ad(path, backed="r")
    keep = data.obs_names.astype(str).isin(set(ids))
    part = data[keep].to_memory()
    return pd.DataFrame(np.asarray(part.X, dtype=np.float64), index=part.obs_names.astype(str)).loc[list(ids)]


def stage08_bag_means(pair, unit, model, ids) -> np.ndarray:
    """Each training sample's stage 08 feature (stage 08's bag_means of its fm_cells: the mean over the shared
    test bags of FROZEN_NCELL cells of the mean cell embedding)."""
    return S08["bag_means"](S08["fm_cells"](pair, unit, model, ids), ids, pair, unit)


def bulk_table(pair, unit) -> pd.DataFrame:
    return training_inputs.scored_bulk(pair, unit)


# ---------------------------------------------------------------------------------------------- pseudobulk and the set encoders


def pseudobulk_job(pair, unit):
    """Coefficient x value on the log-CP10k profile: A = w x, exact completeness."""
    key = "pseudobulk"
    path = out_dir(pair, unit) / key / "s-1.parquet"
    samples = folds.training_samples(pair, *unit)
    ids = list(samples["sample_id"])
    sources = [*folds.inputs(pair), paths.output("04_panel", pair, "panel.csv", unit), heads_file(pair, unit, key)]
    record = record_for(pair, unit, key, sources)
    if manifest.is_done([path], record):
        return
    cost = manifest.begin()
    profiles = training_inputs.pseudobulk_log(pair, unit)
    X = np.vstack([profiles[s] for s in ids])
    heads, dims = read_heads(pair, unit, key)
    assert dims == training_inputs.panel(pair, unit), "stage 08's pseudobulk head isn't on the unit's panel"
    sample_strata = strata(pair, samples)
    per_sample = [{("phenotype", r): heads[(-1, r)][st][0] * x for r in READOUTS} for x, st in zip(X, sample_strata)]
    sums = [{k: a.sum() for k, a in A.items()} for A in per_sample]
    deltas = [{("phenotype", r): float(heads[(-1, r)][st][0] @ x) for r in READOUTS} for x, st in zip(X, sample_strata)]
    halves = attribution.donor_halves(samples, pair, unit)
    write_csv(out_dir(pair, unit) / key / "completeness_s-1.csv", completeness_rows(ids, sums, deltas, [0.0] * len(ids)), record, cost, sources)
    write_attributions(path, key_columns(pair, unit, key), -1, key, results_of(per_sample, ids, X > 0, halves, dims), record, cost, sources)


def set_encoder_job(pair, unit, key, e, data):
    """Gradient x input through TorchInputs on stage 06's test bags."""
    row, encoder = TRAINED[key]
    directory = paths.output("06_train", pair, row, unit) / encoder
    path = out_dir(pair, unit) / key / f"s{e}.parquet"
    sources = [*folds.inputs(pair), settings.OUTPUTS / "03_splits" / pair / "splits.csv", paths.output("04_panel", pair, "panel.csv", unit),
               directory / "checkpoints" / f"s{e}.pt", directory / "embeddings" / f"s{e}.h5ad", heads_file(pair, unit, key)]
    if e >= 10:
        sources.append(settings.OUTPUTS / "03b_attribution_splits" / pair / "splits.csv")
    record = record_for(pair, unit, key, sources, seed=e)
    if manifest.is_done([path], record):
        return
    cost = manifest.begin(device=DEVICE)
    cells, pseudobulk, samples = data
    ids = list(samples["sample_id"])
    checkpoint = torch.load(directory / "checkpoints" / f"s{e}.pt", map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    train, _ = training_inputs.split_cells(cells, training_inputs.encoder_sides(pair, unit, e))
    inputs = harness.Inputs(config, train, pseudobulk)   # recomputed deterministically from the same split (stage 06)
    inputs.prepare(cells)
    if inputs.pca:   # the trained network's own basis, as stage 06 stored it (a refit eigh can differ in its last bits)
        inputs.pca_mean, inputs.components = checkpoint["pca_mean"], checkpoint["components"]
    net = S06["network"](encoder, config, harness.input_width(config, len(cells.genes))).to(DEVICE)
    net.load_state_dict(checkpoint["state"])
    net.eval()
    torch_inputs = attribution.TorchInputs(inputs, DEVICE)
    E_train, E_bulk = stage06_embedding(pair, unit, key, e)
    assert set(E_train.index) == set(ids), f"{pair} {unit} {key} s{e}: stage 06's training rows differ"
    heads, dims = read_heads(pair, unit, key)
    assert dims == [str(i) for i in range(E_train.shape[1])]
    sample_strata = dict(zip(ids, strata(pair, samples)))
    bulk = bulk_table(pair, unit)
    modality = (attribution.modality_direction(E_train.loc[ids].to_numpy(), E_bulk.loc[bulk["sample_id"]].to_numpy(),
                                               strata(pair, samples), strata(pair, bulk)) if key in MODALITY else None)
    one_profile = row == "B"
    if not one_profile:
        assert config["ncell"] == T.FROZEN_NCELL
        test = bags.test_bags(cells.cell_id, cells.sample_id, ids, config["ncell"], representations.TEST_TIME_BAGS, pair, unit)
    per_sample, sums, deltas, reproduction, present = [], [], [], [], []
    for s in ids:
        if one_profile:
            x = pseudobulk[s].astype(np.float64)[None, None, :]
            ones = torch.ones((1, 1), device=DEVICE)
            embed = lambda xt: net.embed(torch_inputs.profile(xt[:, 0, :]), ones)
        else:
            rows = np.stack([cells.row_of(b) for b in test[s]])
            x = cells.X[rows.ravel()].toarray().astype(np.float64).reshape(*rows.shape, -1)
            weights = attribution.depth_weights(cells.library_size[rows], DEVICE)
            embed = lambda xt, s=s, weights=weights: net.embed(torch_inputs.cells(xt, s), weights)
        A, delta, mean_embedding = attribution.gradient_x_input(embed, x, targets_of(heads, e, sample_strata[s], modality), DEVICE)
        expected = E_train.loc[s].to_numpy()
        error = float(np.abs(mean_embedding - expected).max())
        assert error <= G.REPRODUCTION_TOLERANCE * max(1.0, float(np.abs(expected).max())), \
            f"{pair} {unit} {key} s{e} {s}: TorchInputs doesn't reproduce stage 06's embedding (max |delta| {error:.2e})"
        per_sample.append(A)
        sums.append({k: a.sum() for k, a in A.items()})
        deltas.append(delta)
        reproduction.append(error)
        present.append((x > 0).any(axis=(0, 1)))
    halves = attribution.donor_halves(samples, pair, unit)
    write_csv(out_dir(pair, unit) / key / f"completeness_s{e}.csv", completeness_rows(ids, sums, deltas, reproduction), record, cost, sources)
    write_attributions(path, key_columns(pair, unit, key), e, key, results_of(per_sample, ids, np.vstack(present), halves, cells.genes),
                       record, cost, sources)


def scvi_job(pair, unit, e, cells, samples):
    """scVI: gradient x input of log1p(counts) through the encoder's mean, every cell
    of the test bags weighted as in the sample embedding."""
    from sc2bulk.encoders import scvi as adapter
    key = "scvi_cells"
    directory = paths.output("06_train", pair, "J", unit) / "scvi"
    path = out_dir(pair, unit) / key / f"s{e}.parquet"
    sources = [*folds.inputs(pair), paths.output("04_panel", pair, "panel.csv", unit), directory / "checkpoints" / f"s{e}" / "model.pt",
               directory / "embeddings" / f"s{e}.h5ad", heads_file(pair, unit, key)]
    record = record_for(pair, unit, key, sources, seed=e)
    if manifest.is_done([path], record):
        return
    cost = manifest.begin(device=DEVICE)
    ids = list(samples["sample_id"])
    model = adapter.load(directory / "checkpoints" / f"s{e}", cells.X)
    module = model.module.to(DEVICE).eval()
    E_train, E_bulk = stage06_embedding(pair, unit, key, e)
    heads, _ = read_heads(pair, unit, key)
    sample_strata = dict(zip(ids, strata(pair, samples)))
    bulk = bulk_table(pair, unit)
    modality = attribution.modality_direction(E_train.loc[ids].to_numpy(), E_bulk.loc[bulk["sample_id"]].to_numpy(),
                                              strata(pair, samples), strata(pair, bulk))
    test = bags.test_bags(cells.cell_id, cells.sample_id, ids, T.FROZEN_NCELL, representations.TEST_TIME_BAGS, pair, unit)
    per_sample, sums, deltas, reproduction, present = [], [], [], [], []
    for s in ids:
        members = np.unique(np.concatenate(test[s]))
        counts = cells.X[cells.row_of(members)].toarray()
        weight = attribution.scvi_weights(test[s], members)
        A, delta, embedding = attribution.scvi_gradient_x_input(module, counts, weight, targets_of(heads, e, sample_strata[s], modality), DEVICE)
        expected = E_train.loc[s].to_numpy()
        error = float(np.abs(embedding - expected).max())
        assert error <= G.REPRODUCTION_TOLERANCE * max(1.0, float(np.abs(expected).max())), f"{pair} {unit} scVI s{e} {s}: embedding not reproduced"
        per_sample.append(A)
        sums.append({k: a.sum() for k, a in A.items()})
        deltas.append(delta)
        reproduction.append(error)
        present.append((counts > 0).any(axis=0))
    halves = attribution.donor_halves(samples, pair, unit)
    write_csv(out_dir(pair, unit) / key / f"completeness_s{e}.csv", completeness_rows(ids, sums, deltas, reproduction), record, cost, sources)
    write_attributions(path, key_columns(pair, unit, key), e, key, results_of(per_sample, ids, np.vstack(present), halves, cells.genes),
                       record, cost, sources)


# ---------------------------------------------------------------------------------------------- the foundation models


def pilot_path(model) -> Path:
    return paths.output(STAGE, G.PILOT["pair"], f"pilot/pilot_{model}.json")


def readable(pair, unit, model) -> pd.DataFrame:
    """Stage 07's readable-gene map of the unit's panel, plus `credited`: a readable gene whose token is its own.
    Genes stage 07 summed into one token stay in the input (collapsed() uses `target`) but are credited to no gene.
    """
    table = pd.read_csv(paths.output("07_embed", pair, f"{model}_panel_genes.csv", unit), keep_default_na=False)
    assert list(table["gene"]) == training_inputs.panel(pair, unit), "stage 07's gene map isn't the unit's panel"
    return table.assign(credited=table["readable"] & ~table["target"].duplicated(keep=False))


def token_genes(table) -> dict:
    """token -> panel index, over the credited genes only."""
    return {int(t): i for i, (t, c) in enumerate(zip(table["target"], table["credited"])) if c}


def collapsed(X, table):
    """Counts on the panel -> (inputs x tokens, the tokens), as stage 07 (fm.collapse)."""
    from sc2bulk import fm
    return fm.collapse(X, table["target"].to_numpy())


def store_counts(pair, unit, table, what) -> anndata.AnnData:
    """The unit's training cells or pseudobulk profiles (counts) on its panel genes, via sc2bulk.folds."""
    data = folds.training_cells(pair, *unit) if what == "cells" else folds.training_pseudobulk(pair, *unit)
    columns = data.var_names.get_indexer(table["gene"])
    assert (columns >= 0).all()
    return data[:, columns]


def fm_heads(pair, unit, key, ids, model, samples):
    """(averaged heads at encoder seed -1, modality direction or None) for a foundation-model configuration."""
    heads, _ = read_heads(pair, unit, key)
    if key not in MODALITY:
        return heads, None
    bulk = bulk_table(pair, unit)
    X_bulk = fm_embedding(pair, unit, model, "bulk", bulk["sample_id"]).to_numpy()
    return heads, attribution.modality_direction(stage08_bag_means(pair, unit, model, ids), X_bulk, strata(pair, samples), strata(pair, bulk))


def scgpt_profiles(model, X, gene_ids, seeds_) -> list:
    """Stage 07's tokens for each profile: (genes, binned values) from its own collator, seeded per profile."""
    return [model._collate_one(X[i].toarray().ravel().astype(np.float32), gene_ids, int(seeds_[i]))[:2] for i in range(X.shape[0])]


def scgpt_batch(collated, pad_id, pad_value):
    length = max(len(g) for g, _ in collated)
    genes = torch.full((len(collated), length), pad_id, dtype=torch.long)
    expr = torch.full((len(collated), length), float(pad_value))
    for i, (g, e) in enumerate(collated):
        genes[i, :len(g)], expr[i, :len(e)] = g, e
    return genes.to(DEVICE), expr.to(DEVICE)


def scgpt_sample(model, collated, targets, weight, autocast, token_gene: dict, n_genes) -> tuple:
    """One sample's per-gene attribution (inputs = its cells, or its profile), completeness delta and embeddings,
    in chunks of SCGPT_CHUNK inputs (exact: the target sums over inputs and scGPT attends within one)."""
    pad_id, pad_value = model.vocab[model.configs["pad_token"]], model.configs["pad_value"]
    encode = lambda g, v, mask: model.model._encode(g, v, mask)
    out, delta, embeddings = {name: np.zeros(n_genes) for name in targets}, {name: 0.0 for name in targets}, []
    for start in range(0, len(collated), G.SCGPT_CHUNK):
        genes, expr = scgpt_batch(collated[start:start + G.SCGPT_CHUNK], pad_id, pad_value)
        A, d, embedding = attribution.scgpt_gradient_x_input(encode, genes, expr, pad_id, targets,
                                                             weight[start:start + G.SCGPT_CHUNK], autocast)
        index = np.vectorize(lambda t: token_gene.get(int(t), -1))(genes.cpu().numpy())
        index[:, 0] = -1   # <cls>
        keep = index >= 0
        for name, a in A.items():
            np.add.at(out[name], index[keep], a[keep])
            delta[name] += d[name]
        embeddings.append(embedding)
    return out, delta, np.vstack(embeddings)


def scgpt_job(pair, unit, model):
    """scGPT, both configurations (per fold on human blood, aging)."""
    from sc2bulk import fm
    pilot = json.loads(pilot_path("scgpt").read_text())
    autocast = pilot["precision"] == "autocast"
    table = readable(pair, unit, "scgpt")
    samples = folds.training_samples(pair, *unit)
    ids = list(samples["sample_id"])
    panel = list(table["gene"])
    token_gene = token_genes(table)
    halves = attribution.donor_halves(samples, pair, unit)
    sample_strata = dict(zip(ids, strata(pair, samples)))
    for key in MODELS["scgpt"]:
        what = "cells" if key == "scgpt_cells" else "pseudobulk"
        path = out_dir(pair, unit) / key / "s-1.parquet"
        sources = [*folds.inputs(pair), paths.output("07_embed", pair, "scgpt_panel_genes.csv", unit),
                   paths.output("07_embed", pair, f"scgpt_panel_{what}_all.h5ad", unit), heads_file(pair, unit, key), pilot_path("scgpt")]
        record = record_for(pair, unit, key, sources, precision=pilot["precision"])
        if manifest.is_done([path], record):
            continue
        cost = manifest.begin(device=DEVICE)
        data = store_counts(pair, unit, table, what)
        heads, modality = fm_heads(pair, unit, key, ids, "scgpt", samples)
        inputs_of = {s: [s] for s in ids}
        if what == "cells":
            sample_of = data.obs["sample_id"].astype(str)
            inputs_of = {s: list(attribution.fm_cells(pair, s, data.obs_names[sample_of == s].astype(str))) for s in ids}
        stored_all = fm_embedding(pair, unit, "scgpt", what, [c for s in ids for c in inputs_of[s]])
        per_sample, sums, deltas, reproduction, present = [], [], [], [], []
        for s in ids:
            profile_ids = inputs_of[s]
            X, tokens = collapsed(data[profile_ids].X, table)
            collated = scgpt_profiles(model, X, tokens, fm.profile_seeds(pair, profile_ids, 0))
            weight = np.full(len(profile_ids), 1.0 / len(profile_ids))
            A, delta, embedding = scgpt_sample(model, collated, targets_of(heads, -1, sample_strata[s], modality), weight, autocast,
                                               token_gene, len(panel))
            stored = stored_all.loc[profile_ids].to_numpy()
            cosine = float(np.min((embedding * stored).sum(axis=1) / np.linalg.norm(stored, axis=1) / np.linalg.norm(embedding, axis=1)))
            assert not autocast or cosine >= G.PILOT["reproduction"], \
                f"{pair} {unit} {key} {s}: stage 07's embedding not reproduced (cosine {cosine:.6f})"   # stage 07 ran under autocast
            per_sample.append(A)
            sums.append({k: a.sum() for k, a in A.items()})
            deltas.append(delta)
            reproduction.append(cosine)
            counts = data[profile_ids].X
            present.append((np.asarray((counts > 0).sum(axis=0)).ravel() > 0) & table["credited"].to_numpy())
        write_csv(out_dir(pair, unit) / key / "completeness_s-1.csv", completeness_rows(ids, sums, deltas, reproduction), record, cost, sources)
        write_attributions(path, key_columns(pair, unit, key), -1, key, results_of(per_sample, ids, np.vstack(present), halves, panel),
                           record, cost, sources)


def geneformer_embedder(model, fp16: bool):
    """ids (a list of id arrays of one length, or padded) -> the <cls> output at layer n - 1 (stage 07's path),
    float64; fp16 autocast if the pilot allowed it."""
    pad = model.token["<pad>"]

    def embed_ids(id_lists):
        length = max(len(i) for i in id_lists)
        ids = torch.full((len(id_lists), length), pad, dtype=torch.long)
        for row, i in enumerate(id_lists):
            ids[row, :len(i)] = torch.as_tensor(np.asarray(i, dtype=np.int64))
        mask = (ids != pad).long()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and DEVICE == "cuda"):
            hidden = model.model.bert(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).hidden_states
        return hidden[model.layer][:, 0, :].double().cpu().numpy()
    return embed_ids


def medians_of(model, tokens) -> np.ndarray:
    token_to_id = {t: e for e, t in model.token.items()}
    return np.array([model.median[token_to_id[t]] for t in tokens])


def deletion_cells(step) -> int:
    """m for the selected step (the screen keeps the last non-screen step's: 50, or 25 under fp32)."""
    return step["cells"] if step["cells"] is not None else json.loads(pilot_path("geneformer").read_text())["screen_cells"]


def deletion_plan(pair, unit) -> tuple:
    """(the pilot's selected step, precision, this unit's deletion samples)."""
    pilot = json.loads(pilot_path("geneformer").read_text())
    step = pilot["step"]
    assert step["step"] != "triage", "the Geneformer deletion was triaged by the fallback order (frozen §48)"
    samples = folds.training_samples(pair, *unit)
    chosen = attribution.deletion_samples(pair, samples, step["per_donor"], unit[0] == "crossfit")
    assert set(chosen) <= set(samples["sample_id"]), "a deletion sample isn't one of the unit's training samples"
    return step, pilot["precision"], chosen


def screen_candidates(pair, unit, key, n) -> set:
    """The two-stage screen's candidates: the token columns of stage 1's top n genes."""
    heads, _ = read_heads(pair, unit, key)
    table = readable(pair, unit, "geneformer").set_index("gene")
    return set(table.loc[attribution.top(screen_scores(pair, unit, key, heads), n), "target"])


def deletion_input(model, embed_ids, row, tokens, medians, only=None) -> tuple:
    ids, variants, rank, truncated = attribution.geneformer_inputs(model._encode, row, tokens, medians)
    if only is not None:
        variants = [(j, v) for j, v in variants if j in only]
    e0, deltas = attribution.deletion_deltas(embed_ids, ids, variants, G.GF_BATCH)
    return ids, rank, truncated, e0, deltas


def geneformer_job(pair, unit, model, shard, shards, screen1=False, candidates=G.GF_SCREEN["candidates"]):
    """The deletion, one file per (unit, configuration, sample)."""
    assert os.environ.get("SC2BULK_GENEFORMER_GO") == "1", \
        "Geneformer deletion jobs run last, after the pilots; set SC2BULK_GENEFORMER_GO=1"
    step, precision, chosen = deletion_plan(pair, unit)
    assert step["screen"] or not screen1, "--screen1 runs only when the pilot selected the two-stage screen"
    m = deletion_cells(step)
    embed_ids, embed_fp32 = geneformer_embedder(model, precision == "fp16"), geneformer_embedder(model, False)
    table = readable(pair, unit, "geneformer")
    panel_index = token_genes(table)
    cells = store_counts(pair, unit, table, "cells")
    profiles = store_counts(pair, unit, table, "pseudobulk")
    for key in GENEFORMER:
        what = "cells" if key == "geneformer_cells" else "pseudobulk"
        only = None if screen1 or not step["screen"] else screen_candidates(pair, unit, key, candidates)
        folder = "screen1" if screen1 else ("delta" if not step["screen"] else f"delta_c{candidates}")
        stored_path = paths.output("07_embed", pair, f"geneformer_panel_{what}_all.h5ad", unit)
        for s in chosen[shard::shards]:
            path = gf_file(pair, unit, key, folder, s)
            sources = [*folds.inputs(pair), paths.output("07_embed", pair, "geneformer_panel_genes.csv", unit), stored_path, pilot_path("geneformer")]
            record = record_for(pair, unit, key, sources, sample=s, step=step, precision=precision, screen1=screen1, candidates=candidates)
            if manifest.is_done([path], record):
                continue
            cost = manifest.begin(device=DEVICE)
            if what == "cells":
                sample_cells = cells.obs_names[cells.obs["sample_id"].astype(str) == s].astype(str)
                main = attribution.fm_cells(pair, s, sample_cells)[:m]
                chosen_cells = attribution.screen_cells(pair, s, sample_cells, main, G.GF_SCREEN["cells"]) if screen1 else main
                X, tokens = collapsed(cells[list(chosen_cells)].X, table)
                profile_ids = list(chosen_cells)
            else:
                X, tokens = collapsed(profiles[[s]].X, table)
                profile_ids = [s]
            medians = medians_of(model, tokens)
            column_panel = np.array([panel_index.get(int(t), -1) for t in tokens])   # -1: a shared token, deleted but not written
            X = X.toarray()
            only_columns = None if only is None else {j for j, t in enumerate(tokens) if int(t) in only}
            results = [deletion_input(model, embed_ids, row, tokens, medians, only_columns) for row in X]
            exact = np.vstack([embed_fp32([r[0] for r in results[i:i + G.GF_BATCH]]) for i in range(0, len(results), G.GF_BATCH)])
            stored = fm_embedding(pair, unit, "geneformer", what, profile_ids).to_numpy().mean(axis=0)
            cosine = float(exact.mean(axis=0) @ stored / np.linalg.norm(exact.mean(axis=0)) / np.linalg.norm(stored))
            assert cosine >= G.PILOT["reproduction"], f"{pair} {unit} {key} {s}: stage 07's embedding not reproduced (cosine {cosine:.6f})"
            n_present, rank_sum = np.zeros(len(tokens)), np.zeros(len(tokens))
            for _, rank, _, _, _ in results:
                for j, r in rank.items():
                    n_present[j] += 1
                    rank_sum[j] += r
            half = len(results) // 2 if what == "cells" else len(results)
            d = exact.shape[1]
            delta_A = attribution.mean_deltas([r[4] for r in results[:half]], len(tokens), d)
            delta_B = attribution.mean_deltas([r[4] for r in results[half:]], len(tokens), d) if what == "cells" else np.zeros((len(tokens), 0))
            column = np.flatnonzero((n_present > 0) & (column_panel >= 0))
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest.temporary(path), "wb") as handle:
                np.savez(handle, column=column, gene_idx=column_panel[column].astype(np.int32), delta_A=delta_A[column].astype(np.float32),
                         delta_B=delta_B[column].astype(np.float32), profile_ids=np.array(profile_ids), n_tokens=np.array([len(r[0]) - 2 for r in results]),
                         n_truncated=np.array([r[2] for r in results]), emb_cos_vs_stage07=cosine, n_present=n_present[column],
                         rank_sum=rank_sum[column], n_inputs=len(results), n_half_a=half)
            manifest.finish(path, record, cost, inputs=sources)
    assert not manifest.LABELS_READ


def gf_delta_folder(pair, unit) -> str:
    """The delta files' folder: "delta", or under the screen the largest candidates' run (an expansion after a
    failed acceptance supersedes the first)."""
    step = json.loads(pilot_path("geneformer").read_text())["step"]
    if not step["screen"]:
        return "delta"
    runs = [p.name for p in (out_dir(pair, unit) / "geneformer" / "geneformer_cells").glob("delta_c*")]
    return max(runs, key=lambda name: int(name.removeprefix("delta_c")))


def full_delta(data) -> np.ndarray:
    """A delta file's mean over all its inputs: the halves' means weighted by their sizes (cells), or the profile's."""
    if data["delta_B"].shape[1] == 0:
        return data["delta_A"]
    n, a = int(data["n_inputs"]), int(data["n_half_a"])
    return (a * data["delta_A"] + (n - a) * data["delta_B"]) / n


def gf_file(pair, unit, key, folder, s, half=None) -> Path:
    """A Geneformer per-sample file; the sample id is URL-encoded in its name (mouse organs' ids contain "/", as
    all_available.shard_path)."""
    name = quote(s, safe="") + ("" if half is None else f"_{half}")
    return out_dir(pair, unit) / "geneformer" / key / folder / f"{name}.npz"


def gf_files(pair, unit, key, folder) -> tuple:
    """(the deletion plan's samples, sorted; their files in one folder, all asserted present). Files are read by the
    plan's ids, never by listing the folder, so a stale or temporary file is never read."""
    ids = deletion_plan(pair, unit)[2]
    files = [gf_file(pair, unit, key, folder, s) for s in ids]
    missing = [f.name for f in files if not f.exists()]
    assert not missing, f"{pair} {unit} {key} {folder}: no file for {missing} (run that step for every deletion sample)"
    return ids, files


def gf_attributions(pair, unit, key, heads, folder):
    """{sample: (n_panel x ...) per-gene D per head, ...} from the delta files: (ids, {head name: samples x panel
    D}, {head name: (half A D, half B D)}, presence (samples x panel))."""
    panel = training_inputs.panel(pair, unit)
    ids, files = gf_files(pair, unit, key, folder)
    samples = folds.training_samples(pair, *unit).set_index("sample_id")
    sample_strata = strata(pair, samples.loc[ids].reset_index())
    D = {name: np.zeros((len(ids), len(panel))) for name in heads}
    halves = {name: (np.zeros((len(ids), len(panel))), np.zeros((len(ids), len(panel)))) for name in heads}
    present = np.zeros((len(ids), len(panel)), dtype=bool)
    for i, (f, st) in enumerate(zip(files, sample_strata)):
        data = np.load(f)
        cells_ = data["delta_B"].shape[1] > 0
        full = full_delta(data)
        present[i, data["gene_idx"]] = True
        for name, head in heads.items():
            w = head[st][0]
            D[name][i, data["gene_idx"]] = full @ w
            if cells_:
                halves[name][0][i, data["gene_idx"]] = data["delta_A"] @ w
                halves[name][1][i, data["gene_idx"]] = data["delta_B"] @ w
    return ids, D, halves, present, panel


def geneformer_joint(pair, unit, model):
    """The checks' joint deletions: per donor half h, the top genes ranked on h's samples
    (ridge, averaged head) and GF_RANDOM_SETS matched random 50-gene sets, each deleted jointly on the other
    half's samples (every cell of the subsample, or the profile)."""
    assert os.environ.get("SC2BULK_GENEFORMER_GO") == "1", "Geneformer jobs run last, after the pilots; set SC2BULK_GENEFORMER_GO=1"
    step, precision, chosen = deletion_plan(pair, unit)
    embed_ids = geneformer_embedder(model, precision == "fp16")
    table = readable(pair, unit, "geneformer")
    panel_index = token_genes(table)
    samples = folds.training_samples(pair, *unit)
    halves = attribution.donor_halves(samples, pair, unit)
    cells = store_counts(pair, unit, table, "cells")
    profiles = store_counts(pair, unit, table, "pseudobulk")
    folder = gf_delta_folder(pair, unit)
    for key in GENEFORMER:
        what = "cells" if key == "geneformer_cells" else "pseudobulk"
        heads, _ = read_heads(pair, unit, key)
        ids, D, _, present, panel = gf_attributions(pair, unit, key, {"ridge": heads[(-1, "ridge")]}, folder)
        side = halves.loc[ids].to_numpy()
        stats = joint_stats(pair, unit, key, folder, ids, panel)
        for h, index in (("a", 0), ("b", 1)):
            score = pd.Series(np.abs(D["ridge"][side == h]).mean(axis=0), index=panel)
            ranked = attribution.top(score, max(G.GF_CURVE_K))
            evaluated = [s for s, x in zip(ids, side) if x != h]
            frac, mean_rank = stats(evaluated)
            top = [g for g in ranked[:G.TOP_GENES] if g in frac.index]   # present in the evaluated half's inputs
            grid, bins = attribution.matching_grid(frac, mean_rank, top)
            rng = np.random.default_rng(seeds.derived("deletion", pair, index, *unit))
            sets = {f"top_{k}": ranked[:k] for k in G.GF_CURVE_K}
            sets.update({f"random_{i}": genes_ for i, genes_ in
                         enumerate(attribution.matched_random_sets(bins, top, G.GF_RANDOM_SETS, rng))})
            for s in evaluated:
                path = gf_file(pair, unit, key, "joint", s, h)
                sources = [*folds.inputs(pair), heads_file(pair, unit, key), pilot_path("geneformer")]
                record = record_for(pair, unit, key, sources, sample=s, half=h, sets=manifest.settings_hash(sets), precision=precision)
                if manifest.is_done([path], record):
                    continue
                cost = manifest.begin(device=DEVICE)
                if what == "cells":
                    sample_cells = cells.obs_names[cells.obs["sample_id"].astype(str) == s].astype(str)
                    X, tokens = collapsed(cells[list(attribution.fm_cells(pair, s, sample_cells)[:deletion_cells(step)])].X, table)
                else:
                    X, tokens = collapsed(profiles[[s]].X, table)
                medians = medians_of(model, tokens)
                column_of = {panel[panel_index[int(t)]]: j for j, t in enumerate(tokens) if int(t) in panel_index}
                rows = list(X.toarray())
                encoded = [model._encode(r, tokens, medians)[0] for r in rows]
                e0 = [e for i in range(0, len(encoded), G.GF_BATCH) for e in embed_ids(encoded[i:i + G.GF_BATCH])]
                columns = {name: [column_of[g] for g in genes_ if g in column_of] for name, genes_ in sets.items()}
                out = attribution.joint_deltas(embed_ids, model._encode, rows, e0, tokens, medians, columns, G.GF_BATCH)
                names = sorted(out)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(manifest.temporary(path), "wb") as handle:
                    np.savez(handle, set_id=np.array(names), delta=np.vstack([out[n] for n in names]).astype(np.float32),
                             genes=np.array(["|".join(sets[n]) for n in names]), grid=grid, matched=len(top))
                manifest.finish(path, record, cost, inputs=sources)
    assert not manifest.LABELS_READ


def joint_stats(pair, unit, key, folder, ids, panel):
    """A function of a sample list -> (fraction of their inputs containing each gene, mean rank when present),
    over genes present at least once (the matching grid's axes; from the delta files)."""
    files = {s: np.load(gf_file(pair, unit, key, folder, s)) for s in ids}

    def stats(sample_ids):
        n_present, rank_sum, n_inputs = np.zeros(len(panel)), np.zeros(len(panel)), 0
        for s in sample_ids:
            data = files[s]
            n_present[data["gene_idx"]] += data["n_present"]
            rank_sum[data["gene_idx"]] += data["rank_sum"]
            n_inputs += int(data["n_inputs"])
        keep = n_present > 0
        names = np.asarray(panel)[keep]
        return pd.Series(n_present[keep] / n_inputs, index=names), pd.Series(rank_sum[keep] / n_present[keep], index=names)
    return stats


# ---------------------------------------------------------------------------------------------- pilots


def pilot_head(pair, unit, key, model, ids, samples):
    """Gate (b)'s head: stage 08's averaged ridge head if it exists, else readout.run on the single-cell
    training rows of stage 07's features (ridge, read-out seed 0). Single-cell labels only."""
    if paths.output("08_readout", pair, f"{key}_heads.parquet", unit).exists():
        return read_heads(pair, unit, key)[0][(-1, "ridge")]
    X = stage08_bag_means(pair, unit, model, ids) if key.endswith("_cells") else fm_embedding(pair, unit, model, "pseudobulk", ids).to_numpy()
    out = readout.run(X, samples.set_index("sample_id").loc[ids, "group"].to_numpy(), S08["readout_folds"](pair, unit, ids, 0), None)
    mean, scale = out["scalers"]["single_cell"]["all"]
    return {"all": (out["coef"] / scale, out["intercept"] - float((out["coef"] * mean / scale).sum()))}


def precision_gate(fp16: np.ndarray, fp32: np.ndarray) -> dict:
    """Gate (b) on (samples x genes) attributions: Spearman over the pooled values where either precision is
    nonzero (genes absent at both would add tied zeros and inflate rho), top-50 overlap of mean |.|."""
    nonzero = (fp16 != 0) | (fp32 != 0)
    rho = float(spearmanr(fp16[nonzero], fp32[nonzero])[0])
    genes = np.arange(fp16.shape[1]).astype(str)
    top16 = attribution.top(pd.Series(np.abs(fp16).mean(axis=0), index=genes), G.TOP_GENES)
    top32 = attribution.top(pd.Series(np.abs(fp32).mean(axis=0), index=genes), G.TOP_GENES)
    lap = attribution.overlap(top16, top32)
    return {"spearman": rho, "overlap": lap, "passed": rho >= G.PILOT["spearman"] and lap >= G.PILOT["overlap"]}


def pilot_geneformer(model):
    """The Geneformer pilot: gates (a)-(c) on mouse microglia, then the
    budget of every fallback step and the step the order selects."""
    pair, unit = G.PILOT["pair"], ("all", 0)
    table = readable(pair, unit, "geneformer")
    gene_of = token_genes(table)
    samples = folds.training_samples(pair, *unit)
    ids = attribution.deletion_samples(pair, samples, 2, False)[:G.PILOT["samples"]]
    cells, profiles = store_counts(pair, unit, table, "cells"), store_counts(pair, unit, table, "pseudobulk")
    gates, timing = {}, {}
    for key in GENEFORMER:
        what = "cells" if key == "geneformer_cells" else "pseudobulk"
        head = pilot_head(pair, unit, key, "geneformer", list(samples["sample_id"]), samples)["all"][0]
        D = {}
        for precision in ("fp32", "fp16"):
            embed_ids = geneformer_embedder(model, precision == "fp16")
            rows = []
            for s in ids:
                if what == "cells":
                    sample_cells = cells.obs_names[cells.obs["sample_id"].astype(str) == s].astype(str)
                    chosen = list(attribution.fm_cells(pair, s, sample_cells)[:G.PILOT["cells"]])
                    X, tokens = collapsed(cells[chosen].X, table)
                else:
                    chosen = [s]
                    X, tokens = collapsed(profiles[[s]].X, table)
                medians = medians_of(model, tokens)
                start = time.perf_counter()
                results = [deletion_input(model, embed_ids, r, tokens, medians) for r in X.toarray()]
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                timing.setdefault(precision, {"tokens": 0.0, "seconds": 0.0})   # the deletion only (no file reads)
                timing[precision]["tokens"] += attribution.deletion_tokens([len(r[0]) for r in results])
                timing[precision]["seconds"] += time.perf_counter() - start
                if precision == "fp32":   # gate (a): the unperturbed fp32 embedding vs stage 07's
                    exact = np.mean([r[3] for r in results], axis=0)
                    stored = fm_embedding(pair, unit, "geneformer", what, chosen).to_numpy().mean(axis=0)
                    cosine = float(exact @ stored / np.linalg.norm(exact) / np.linalg.norm(stored))
                    assert cosine >= G.PILOT["reproduction"], f"pilot gate (a): {key} {s} cosine {cosine:.6f}"
                delta = attribution.mean_deltas([r[4] for r in results], len(tokens), results[0][3].shape[0])
                row = np.zeros(len(table))
                kept = [j for j, t in enumerate(tokens) if int(t) in gene_of]
                row[[gene_of[int(tokens[j])] for j in kept]] = (delta @ head)[kept]
                rows.append(row)
            D[precision] = np.vstack(rows)
        gates[key] = precision_gate(D["fp16"], D["fp32"])
    precision = "fp16" if all(g["passed"] for g in gates.values()) else "fp32"
    tokens_per_second = timing[precision]["tokens"] / timing[precision]["seconds"]
    lengths = [item for p in PAIR_ORDER for item in pilot_lengths(model, p)]   # gate (c): no forward passes
    budget = {step["step"]: budget_days(lengths, step, tokens_per_second, screen_cell_count(precision))
              for step in G.GF_FALLBACK_ORDER if step["step"] != "triage"}
    chosen = attribution.fallback(budget, precision)
    medians = pd.DataFrame([{"pair": i["pair"], "cells": np.median(i["cells"]), "pseudobulk": i["pseudobulk"][0]} for i in lengths]).groupby("pair").median()
    result = {"gates": gates, "precision": precision, "tokens_per_second": tokens_per_second, "budget_days": budget, "step": chosen,
              "screen_cells": screen_cell_count(precision), "median_tokens": medians.to_dict(orient="index")}
    write_pilot("geneformer", result)


def screen_cell_count(precision) -> int:
    """The screen's main cell count: the last non-screen step's (50, or 25 under fp32)."""
    return 25 if precision == "fp32" else 50


def pilot_lengths(model, pair) -> list:
    """Gate (c), no forward passes: per planned (unit, sample) its inputs' token counts (the cells under m = 50,
    and the profile). Cross-fitted units list each per-donor variant (1 and 2) separately."""
    out, room = [], model.input_size - 2
    for unit in units(pair):
        table = readable(pair, unit, "geneformer")
        samples = folds.training_samples(pair, *unit)
        cells, profiles = store_counts(pair, unit, table, "cells"), store_counts(pair, unit, table, "pseudobulk")
        cross_fitted = unit[0] == "crossfit"
        for per_donor in (1, 2) if cross_fitted else (None,):
            for s in attribution.deletion_samples(pair, samples, per_donor or 2, cross_fitted):
                sample_cells = cells.obs_names[cells.obs["sample_id"].astype(str) == s].astype(str)
                X, _ = collapsed(cells[list(attribution.fm_cells(pair, s, sample_cells))].X, table)
                P, _ = collapsed(profiles[[s]].X, table)
                out.append({"pair": pair, "unit": folds.unit_name(*unit), "per_donor": per_donor, "sample": s,
                            "cells": [min(int(n), room) for n in np.asarray((X > 0).sum(axis=1)).ravel()],
                            "pseudobulk": [min(int((P > 0).sum()), room)]})
    return out


def budget_days(lengths: list, step: dict, tokens_per_second: float, screen_cells: int) -> float:
    """One fallback step's deletion budget in days on PILOT["gpus"] GPUs (cost model, with the checks'
    overhead; the screen: stage 1 on GF_SCREEN cells, stage 2 the candidates on the main cells)."""
    total = 0.0
    for item in lengths:
        if item["per_donor"] not in (None, step["per_donor"]):
            continue
        m = step["cells"] or screen_cells
        if step["screen"]:
            cell_tokens = attribution.deletion_tokens(item["cells"][:G.GF_SCREEN["cells"]]) + sum(
                min(G.GF_SCREEN["candidates"], n) * n * float(attribution.overhead(n)) for n in item["cells"][:m])
        else:
            cell_tokens = attribution.deletion_tokens(item["cells"][:m])
        total += cell_tokens * (1 + G.CHECK_OVERHEAD["cells"]) + attribution.deletion_tokens(item["pseudobulk"]) * (1 + G.CHECK_OVERHEAD["pseudobulk"])
    return total / tokens_per_second / 3600 / 24 / G.PILOT["gpus"]


def pilot_scgpt(model):
    """The scGPT pilot: reproduction of stage 07's embeddings (cosine), and autocast vs fp32
    gradient x input (gate (b)); the precision every scGPT job then uses."""
    from sc2bulk import fm
    pair, unit = G.PILOT["pair"], ("all", 0)
    table = readable(pair, unit, "scgpt")
    samples = folds.training_samples(pair, *unit)
    ids = attribution.deletion_samples(pair, samples, 2, False)[:G.PILOT["samples"]]
    token_gene = token_genes(table)
    gates = {}
    for key in MODELS["scgpt"]:
        what = "cells" if key == "scgpt_cells" else "pseudobulk"
        data = store_counts(pair, unit, table, what)
        head = pilot_head(pair, unit, key, "scgpt", list(samples["sample_id"]), samples)["all"][0]
        A = {}
        for autocast in (True, False):
            rows = []
            for s in ids:
                if what == "cells":
                    sample_cells = data.obs_names[data.obs["sample_id"].astype(str) == s].astype(str)
                    chosen = list(attribution.fm_cells(pair, s, sample_cells)[:G.PILOT["cells"]])
                else:
                    chosen = [s]
                X, tokens = collapsed(data[chosen].X, table)
                collated = scgpt_profiles(model, X, tokens, fm.profile_seeds(pair, chosen, 0))
                a, _, embedding = scgpt_sample(model, collated, {"ridge": head}, np.full(len(chosen), 1.0 / len(chosen)), autocast,
                                               token_gene, len(table))
                stored = fm_embedding(pair, unit, "scgpt", what, chosen).to_numpy()
                cosine = float(np.min((embedding * stored).sum(axis=1) / np.linalg.norm(stored, axis=1) / np.linalg.norm(embedding, axis=1)))
                assert not autocast or cosine >= G.PILOT["reproduction"], f"scGPT pilot: {key} {s} cosine {cosine:.6f}"
                rows.append(a["ridge"])
            A[autocast] = np.vstack(rows)
        gates[key] = precision_gate(A[True], A[False])
    write_pilot("scgpt", {"gates": gates, "precision": "autocast" if all(g["passed"] for g in gates.values()) else "fp32"})


def write_pilot(model, result):
    path = pilot_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = record_for(G.PILOT["pair"], ("all", 0), f"pilot_{model}", [*folds.inputs(G.PILOT["pair"])])
    cost = manifest.begin(device=DEVICE)
    manifest.temporary(path).write_text(json.dumps(result, indent=1, default=str))
    manifest.finish(path, record, cost)
    print(json.dumps(result, indent=1, default=str), flush=True)


# ---------------------------------------------------------------------------------------------- gather


def seed_tables(pair, unit, key) -> dict:
    """{encoder seed: attributions table} of one unit and configuration."""
    files = sorted((out_dir(pair, unit) / key).glob("s*.parquet"))
    return {int(f.stem[1:]): schemas.read("attributions", f) for f in files}


def expected_seeds(pair, key) -> list:
    """Stage 08's encoder seeds (its NOT_TRAINED excluded), or -1 for a fixed representation."""
    return S08["encoder_seeds"](pair, key) if key in TRAINED else [-1]


def geneformer_tables(pair, unit, key, folder) -> tuple:
    """Geneformer's per-unit attributions table from the delta files (heads applied here), with the
    cell-half and per-read-out-seed scores its gates need."""
    heads, _ = read_heads(pair, unit, key)
    per_seed, _ = read_heads(pair, unit, key, per_seed=True)
    samples = folds.training_samples(pair, *unit)
    bulk = bulk_table(pair, unit)
    named = {(t, r): heads[(-1, r)] for t, r in (("phenotype", "ridge"), ("phenotype", "lasso"))}
    if key in MODALITY:
        ids_all = list(samples["sample_id"])
        X_bulk = fm_embedding(pair, unit, "geneformer", "bulk", bulk["sample_id"]).to_numpy()
        named[("modality_direction", "none")] = attribution.modality_direction(stage08_bag_means(pair, unit, "geneformer", ids_all), X_bulk,
                                                                               strata(pair, samples), strata(pair, bulk))
    seeds_ = {r: h for (e, readout_name, r), h in per_seed.items() if readout_name == "ridge"}
    ids, D, halves, present, panel = gf_attributions(pair, unit, key, {**named, **{("seed", r): h for r, h in seeds_.items()}}, folder)
    side = attribution.donor_halves(samples, pair, unit).loc[ids].to_numpy()
    results = []
    for (t, r) in named:
        table = attribution.summarize(D[(t, r)], present, side, panel)
        if key == "geneformer_cells":
            table = table.assign(mean_abs_cells_a=np.abs(halves[(t, r)][0]).mean(axis=0), mean_abs_cells_b=np.abs(halves[(t, r)][1]).mean(axis=0))
        results.append((r, t, table))
    seed_scores = [pd.Series(np.abs(D[("seed", r)]).mean(axis=0), index=panel) for r in sorted(seeds_)]
    return results, seed_scores, (ids, D, panel)


def level_difference(pair, unit) -> pd.DataFrame:
    """Fig 6B's level difference for one unit (attribution.level_difference): scored bulk vs the training samples'
    pseudobulk, both log-CP10k over the shared genes on the panel; markers = 04b's primary list (mouse
    microglia has none: one type)."""
    samples = folds.training_samples(pair, *unit)
    profiles = training_inputs.pseudobulk_log(pair, unit)
    bulk = bulk_table(pair, unit)
    markers = set()
    if deconvolution.LEVELS[pair] is not None:
        markers = set(pd.read_csv(paths.output("04b_deconvolve", pair, "markers.csv", unit), keep_default_na=False)["gene"])
    return attribution.level_difference(training_inputs.bulk_profiles(pair, unit, bulk["sample_id"]).X.toarray(),
                                        np.vstack([profiles[s] for s in samples["sample_id"]]), strata(pair, bulk), strata(pair, samples),
                                        training_inputs.panel(pair, unit), markers)


def human_symbols(pair, unit, genes) -> dict:
    """Panel gene -> human symbol (mouse pairs: the one-to-one ortholog; unmapped genes keep their symbol)."""
    if PAIRS[pair]["species"] == "human":
        return {g: g for g in genes}
    mapped = pd.read_csv(paths.output("04_panel", pair, "orthologs.csv", unit), keep_default_na=False).set_index("gene")["human_symbol"]
    return {g: (mapped.get(g) or g) for g in genes}


def human_ensembl(pair, unit, genes) -> dict:
    """Panel gene -> its human Ensembl gene ids, ';'-joined, '' if none: human pairs, stage 02's
    single-cell var; mouse pairs, the one-to-one ortholog ('' when unmapped)."""
    if PAIRS[pair]["species"] == "human":
        ids = folds.single_cell_genes(pair)["ensembl"]
    else:
        ids = pd.read_csv(paths.output("04_panel", pair, "orthologs.csv", unit), keep_default_na=False).set_index("gene")["human_ensembl"]
    return {g: ids[g] for g in genes}


def housekeeping_set() -> set:
    """The HRT Atlas list's human Ensembl gene ids."""
    assert G.HOUSEKEEPING_SHA256 is not None, "pin the HRT Atlas file and its SHA-256 in config/genes.py (spec Q8)"
    assert hashlib.sha256(G.HOUSEKEEPING_FILE.read_bytes()).hexdigest() == G.HOUSEKEEPING_SHA256, "the housekeeping file differs from its pin"
    table = pd.read_csv(G.HOUSEKEEPING_FILE, sep="\t", comment="#", keep_default_na=False)
    return set(table["ensembl"])


def check_row(pair, unit, key, readout_name, target, check, value, threshold=None, passed=None, half=""):
    return {"pair": pair, "fold_set": unit[0], "fold": unit[1], "configuration": key, "readout": readout_name, "target": target,
            "check": check, "half": half, "value": value, "threshold": threshold, "passed": passed}


def geneformer_checks(pair, unit, key, folder, results, seed_scores, gathered) -> tuple:
    """One unit's ingredients of the checks (pooled over units by gate_rows), its deletion curve rows
    and the screen's acceptance row: (ingredients, curve, rows)."""
    ids, D, panel = gathered
    ridge = D[("phenotype", "ridge")]
    heads, _ = read_heads(pair, unit, key)
    samples = folds.training_samples(pair, *unit).set_index("sample_id")
    rows, curve = [], []
    cells = None
    if key == "geneformer_cells":
        table = next(t for r, tg, t in results if r == "ridge" and tg == "phenotype").set_index("gene")
        cells = (table["mean_abs_cells_a"], table["mean_abs_cells_b"])
    side = attribution.donor_halves(samples.reset_index(), pair, unit).loc[ids].to_numpy()
    sample_strata = dict(zip(samples.index, strata(pair, samples.reset_index())))
    J, joint, summed, matching = {"a": [], "b": []}, {10: [], 50: []}, {10: [], 50: []}, {}
    for h in ("a", "b"):
        score = pd.Series(np.abs(ridge[side == h]).mean(axis=0), index=panel)
        ranked = attribution.top(score, max(G.GF_CURVE_K))
        for i, s in enumerate(ids):
            if side[i] == h:
                continue
            data = np.load(gf_file(pair, unit, key, "joint", s, h))
            assert data["genes"][list(data["set_id"]).index("top_50")] == "|".join(ranked[:50]), \
                f"{pair} {unit} {key} {s}_{h}: the joint file's top 50 isn't this ranking's (rerun --joint)"
            grid = (int(data["grid"]), int(data["matched"]))
            assert matching.setdefault(h, grid) == grid, f"{pair} {unit} {key} {s}_{h}: the half's joint files differ in their matching grid (rerun --joint)"
            w = heads[(-1, "ridge")][sample_strata[s]][0]
            j = dict(zip(data["set_id"], data["delta"] @ w))
            J[h].append(j)
            for k in G.GF_CURVE_K:
                curve.append({"pair": pair, "fold_set": unit[0], "fold": unit[1], "configuration": key, "half": h, "sample_id": s,
                              "set": "top", "k": k, "delta_pred": j[f"top_{k}"]})
            curve += [{"pair": pair, "fold_set": unit[0], "fold": unit[1], "configuration": key, "half": h, "sample_id": s,
                       "set": f"random_{n}", "k": G.TOP_GENES, "delta_pred": j[f"random_{n}"]} for n in range(G.GF_RANDOM_SETS)]
            for k in (10, 50):
                joint[k].append(j[f"top_{k}"])
                summed[k].append(ridge[i, [panel.index(g) for g in ranked[:k]]].sum())
    if folder != "delta":   # the two-stage screen's acceptance: every final top-100 within stage 1's top margin
        stage1 = attribution.top(screen_scores(pair, unit, key, heads), G.GF_SCREEN["margin"])
        final = attribution.top(pd.Series(np.abs(ridge).mean(axis=0), index=panel), G.GF_SCREEN["top"])
        accepted = set(final) <= set(stage1)
        expanded = folder == f"delta_c{G.GF_SCREEN['expanded']}"
        # a failed acceptance expands the candidates, it doesn't exclude Geneformer; after the expansion
        # (the GF spec's last step) acceptance is reported, not gated
        assert accepted or expanded, f"{pair} {unit} {key}: screen acceptance failed; rerun the deletion with --candidates {G.GF_SCREEN['expanded']}"
        rows.append(check_row(pair, unit, key, "ridge", "phenotype", "screen_acceptance", float(accepted), folder, accepted or None))
    return {"cells": cells, "seeds": seed_scores, "J": J, "joint": joint, "summed": summed, "matching": matching}, curve, rows


def gate_rows(pair, unit, key, parts: list) -> list:
    """The stability, top vs matched random (both halves) and additivity checks with their thresholds,
    on one unit's ingredients or several units' pooled (human blood, aging: the folds', on the across-fold
    ranking's footing; inclusion is decided per (configuration, pair)): the
    stabilities compare across() of the per-fold scores, T and beta pool every fold's evaluated samples, random
    set n pooled across folds by index."""
    C = G.ATTRIBUTION_CHECKS
    row = lambda check, value, threshold=None, passed=None, half="": check_row(pair, unit, key, "ridge", "phenotype", check, value, threshold,
                                                                              passed, half)
    pooled = lambda pick: attribution.across({i: [pick(p)] for i, p in enumerate(parts)})
    seeds_ = attribution.mean_pairwise_overlap([pooled(lambda p: p["seeds"][r]) for r in range(len(parts[0]["seeds"]))], G.TOP_GENES)
    rows = []
    if parts[0]["cells"] is not None:
        value = attribution.overlap(attribution.top(pooled(lambda p: p["cells"][0]), G.TOP_GENES),
                                    attribution.top(pooled(lambda p: p["cells"][1]), G.TOP_GENES))
        rows += [row("stability_cells", value, C["stability"], value >= C["stability"]), row("stability_readout_seeds", seeds_)]
    else:
        rows.append(row("stability_readout_seeds", seeds_, C["stability"], seeds_ >= C["stability"]))
    for h in ("a", "b"):
        J = [j for p in parts for j in p["J"][h]]
        T_top = float(np.median([abs(j["top_50"]) for j in J]))
        T_random = [float(np.median([abs(j[f"random_{n}"]) for j in J])) for n in range(G.GF_RANDOM_SETS)]
        p95 = float(np.percentile(T_random, C["random_pct"]))
        rows.append(row("random_T_top", T_top, p95, T_top > p95, half=h))
        rows.append(row("random_match_grid", min(p["matching"][h][0] for p in parts), half=h))   # reported
        rows.append(row("random_set_size", min(p["matching"][h][1] for p in parts), half=h))
    joint = {k: np.array([v for p in parts for v in p["joint"][k]]) for k in (10, 50)}
    summed = {k: np.array([v for p in parts for v in p["summed"][k]]) for k in (10, 50)}
    low, high = C["additivity"]
    beta = attribution.additivity(joint[50], summed[50])
    rows += [row("additivity_beta_k50", beta, f"[{low}, {high}]", low <= beta <= high),
             row("additivity_beta_k10", attribution.additivity(joint[10], summed[10])),
             row("additivity_spearman", float(spearmanr(joint[50], summed[50])[0]))]
    return rows


def screen_scores(pair, unit, key, heads) -> pd.Series:
    """Stage 1's score per panel gene (mean |D| over the samples, ridge, averaged head)."""
    panel = training_inputs.panel(pair, unit)
    samples = folds.training_samples(pair, *unit).set_index("sample_id")
    ids, files = gf_files(pair, unit, key, "screen1")
    score = np.zeros(len(panel))
    for s, f in zip(ids, files):
        data = np.load(f)
        full = full_delta(data)
        score[data["gene_idx"]] += np.abs(full @ heads[(-1, "ridge")][strata(pair, samples.loc[[s]].reset_index())[0]][0]) / len(files)
    return pd.Series(score, index=panel)


def gather(pair, pending=False):
    """The rankings over seeds and folds, Geneformer's attributions, the checks, inclusion, the level difference
    and the modality summary (CPU). pending: Geneformer left out until its deletion has run."""
    pilot = json.loads(pilot_path("geneformer").read_text())
    triaged = pilot["step"]["step"] == "triage"
    assert not (triaged and pending), "the Geneformer deletion was triaged, not pending"
    dropped = "triaged" if triaged else "pending" if pending else None
    keys = [k for k in configurations.CONFIGURATION_ORDER if not (dropped and k in GENEFORMER)]
    hk_genes = housekeeping_set()
    record = record_for(pair, None, "gather", [*folds.inputs(pair)], geneformer=dropped)
    cost = manifest.begin()
    checks, curve, inclusion, summary = [], [], [], []
    levels = {}
    for unit in units(pair):
        levels[unit] = level_difference(pair, unit)
        write_csv(out_dir(pair, unit) / "level_difference.csv", levels[unit], record, cost)
    rankings, unit_scores, gf = {}, {}, {}
    for key in keys:
        per_fold, gf[key] = {}, {}
        for unit in units(pair):
            if key in GENEFORMER:
                folder = gf_delta_folder(pair, unit)
                results, seed_scores, gathered = geneformer_tables(pair, unit, key, folder)
                path = out_dir(pair, unit) / key / "s-1.parquet"
                write_attributions(path, key_columns(pair, unit, key), -1, key, results, record, cost, [])
                gf[key][unit] = (folder, results, seed_scores, gathered)
            tables = seed_tables(pair, unit, key)
            assert sorted(tables) == sorted(expected_seeds(pair, key)), f"{pair} {unit} {key}: encoder seeds {sorted(tables)}"
            for e, table in tables.items():
                for (r, t), part in table.groupby(["readout", "target"]):
                    per_fold.setdefault((r, t), {}).setdefault(unit, []).append(part.set_index("gene")["attribution"])
                    unit_scores.setdefault((key, r, t, unit), []).append((e, part.set_index("gene")))
            checks += seed_checks(pair, unit, key, tables)
        rankings[key], lengths = write_ranking(pair, key, per_fold, record, cost)
        checks += lengths
        if len(units(pair)) > 1:
            for (r, t), by_fold in per_fold.items():
                fold_scores = [pd.concat(v, axis=1).mean(axis=1) for v in by_fold.values()]
                checks.append(check_row(pair, (units(pair)[0][0], -1), key, r, t, "stability_folds",
                                        attribution.mean_pairwise_overlap(fold_scores, G.TOP_GENES)))
    unit0 = units(pair)[0]
    symbols, ids = {}, {}   # symbols for the RPL/RPS/MT- fraction only; housekeeping is matched on gene ids
    for unit in units(pair):
        panel = training_inputs.panel(pair, unit)
        symbols.update(human_symbols(pair, unit, panel))
        ids.update(human_ensembl(pair, unit, panel))
    hk_panel = {g for g, i in ids.items() if hk_genes & set(i.split(";"))}
    pseudobulk_top = rankings["pseudobulk"][("ridge", "phenotype")]
    for key in keys:
        attributable = attributable_genes(pair, key)
        hk = attribution.housekeeping(rankings[key][("ridge", "phenotype")], attributable, hk_panel, symbols, pseudobulk_top)
        hk["hk_no_human_id"] = sum(not ids[g] for g in attributable)   # no id or no ortholog: counted as not housekeeping
        fold_all = (unit0[0], -1 if len(units(pair)) > 1 else unit0[1])
        checks += [check_row(pair, fold_all, key, "ridge", "phenotype", name, value) for name, value in hk.items()]
        if key in GENEFORMER:
            dominated = hk["hk_p"] < G.ATTRIBUTION_CHECKS["hk_p"] and hk["hk_excess_vs_pseudobulk"] >= G.ATTRIBUTION_CHECKS["hk_excess"]
            gated = [check_row(pair, fold_all, key, "ridge", "phenotype", "hk_dominated", float(dominated), None, not dominated)]
            parts = {}
            for unit, (folder, results, seed_scores, gathered) in gf[key].items():
                parts[unit], c, rows = geneformer_checks(pair, unit, key, folder, results, seed_scores, gathered)
                curve += c
                gated += rows
            if len(parts) > 1:   # human blood, aging: per-fold values reported; the pooled (fold -1) rows gate
                checks += [r for unit, p in parts.items() for r in gate_rows(pair, unit, key, [p])]
            gated += gate_rows(pair, fold_all, key, list(parts.values()))
            checks += gated
            failed = [r["check"] for r in gated if r["passed"] is False]
            inclusion.append({"pair": pair, "configuration": key, "included": not failed, "failed_checks": ";".join(sorted(set(failed)))})
        if key in MODALITY:
            summary += modality_summary(pair, key, levels, unit_scores)
    for key in GENEFORMER:
        if dropped:
            inclusion.append({"pair": pair, "configuration": key, "included": False, "failed_checks": dropped})
    write_csv(out_dir(pair) / "checks.csv", pd.DataFrame(checks), record, cost)
    write_csv(out_dir(pair) / "inclusion.csv", pd.DataFrame(inclusion, columns=["pair", "configuration", "included", "failed_checks"]), record, cost)
    write_csv(out_dir(pair) / "curve.csv", pd.DataFrame(curve), record, cost)
    write_csv(out_dir(pair) / "modality_summary.csv", pd.DataFrame(summary), record, cost)
    assert not manifest.LABELS_READ


def attributable_genes(pair, key) -> list:
    """The genes a configuration can credit (union over the units): the FMs' credited panel genes, else the panel."""
    out = set()
    for unit in units(pair):
        if key.startswith(("geneformer", "scgpt")):
            table = readable(pair, unit, key.split("_")[0])
            out |= set(table.loc[table["credited"], "gene"])
        else:
            out |= set(training_inputs.panel(pair, unit))
    return sorted(out)


def seed_checks(pair, unit, key, tables) -> list:
    """S14A-style rows per (readout, target): the mean pairwise top-50 overlap of single-seed rankings, seeds 0-9
    vs 10-19, the donor halves' overlap; completeness (gradient methods)."""
    rows = []
    stacked = pd.concat([t.assign(encoder_seed=e) for e, t in tables.items()])
    for (r, t), part in stacked.groupby(["readout", "target"]):
        by_seed = {e: p.set_index("gene") for e, p in part.groupby("encoder_seed")}
        if len(by_seed) > 1:
            rows.append(check_row(pair, unit, key, r, t, "stability_seeds",
                                  attribution.mean_pairwise_overlap([p["attribution"] for p in by_seed.values()], G.TOP_GENES)))
        if any(e >= 10 for e in by_seed):
            first = pd.concat([p["attribution"] for e, p in by_seed.items() if e < 10], axis=1).mean(axis=1)
            second = pd.concat([p["attribution"] for e, p in by_seed.items() if e >= 10], axis=1).mean(axis=1)
            rows.append(check_row(pair, unit, key, r, t, "stability_split_half",
                                  attribution.overlap(attribution.top(first, G.TOP_GENES), attribution.top(second, G.TOP_GENES))))
        a = pd.concat([p["mean_abs_half_a"] for p in by_seed.values()], axis=1).mean(axis=1)
        b = pd.concat([p["mean_abs_half_b"] for p in by_seed.values()], axis=1).mean(axis=1)
        rows.append(check_row(pair, unit, key, r, t, "stability_donors", attribution.overlap(attribution.top(a, G.TOP_GENES), attribution.top(b, G.TOP_GENES))))
    if key not in GENEFORMER:
        completeness = pd.concat([pd.read_csv(out_dir(pair, unit) / key / f"completeness_s{e}.csv") for e in tables])
        for (r, t), part in completeness.groupby(["readout", "target"]):
            rows += [check_row(pair, unit, key, r, t, name, value)
                     for name, value in attribution.completeness(part["sum_attribution"], part["delta_prediction"]).items()]
    return rows


def write_ranking(pair, key, per_fold, record, cost) -> dict:
    """<configuration>_ranking.parquet: per (readout, target) the score over folds x seeds, rank and top-k flags.
    A top-k list keeps only genes with attribution > 0, so it can be shorter than k (a sparse averaged LASSO head
    would otherwise pad it with zero-score genes in symbol order); each list's length is a checks row. Returns
    ({(readout, target): the top-TOP_GENES genes}, the length rows)."""
    frames, tops, lengths = [], {}, []
    unit = units(pair)[0]
    fold = -1 if len(units(pair)) > 1 else unit[1]
    for (r, t), by_fold in per_fold.items():
        score = attribution.across(by_fold)
        rank = attribution.ranks(score)
        frame = pd.DataFrame({"gene": score.index, "attribution": score.to_numpy(), "rank": rank.to_numpy()})
        for k in G.TOP_K:
            frame[f"top{k}"] = (frame["rank"] <= k) & (frame["attribution"] > 0)
            lengths.append(check_row(pair, (unit[0], fold), key, r, t, f"top{k}_length", int(frame[f"top{k}"].sum()), k))
        frames.append(frame.assign(**key_columns(pair, (unit[0], fold), key), encoder_seed=-1, readout=r, readout_seed=-1, target=t,
                                   method=method(key)))
        tops[(r, t)] = list(frame[frame[f"top{G.TOP_GENES}"]].sort_values("rank")["gene"])
    path = out_dir(pair) / f"{key}_ranking.parquet"
    schemas.write("attributions", schemas.with_neutral_keys(pd.concat(frames, ignore_index=True)), manifest.temporary(path))
    manifest.finish(path, record, cost)
    return tops, lengths


def modality_summary(pair, key, levels, unit_scores) -> list:
    """Fig 6B's cross-model summary: per unit, the share of the configuration's top-50 modality
    genes (that unit's mean over seeds) that are large-difference genes, markers, or either, with the base
    rates over the panel; the cross-fitted pair adds the mean over folds (fold -1)."""
    rows = []
    for unit, level in levels.items():
        scored = unit_scores[(key, "none", "modality_direction", unit)]
        score = pd.concat([t["attribution"] for _, t in scored], axis=1).mean(axis=1)
        chosen = level.set_index("gene").loc[attribution.top(score, G.TOP_GENES)]
        either = level["large"] | level["marker"]
        rows.append({"pair": pair, "fold_set": unit[0], "fold": unit[1], "configuration": key, "share_large": chosen["large"].mean(),
                     "share_marker": chosen["marker"].mean(), "share_either": (chosen["large"] | chosen["marker"]).mean(),
                     "base_large": level["large"].mean(), "base_marker": level["marker"].mean(), "base_either": either.mean(),
                     "markers_available": deconvolution.LEVELS[pair] is not None})
    if len(rows) > 1:
        mean = pd.DataFrame(rows).drop(columns=["pair", "fold_set", "fold", "configuration", "markers_available"]).mean()
        rows.append({"pair": pair, "fold_set": rows[0]["fold_set"], "fold": -1, "configuration": key, **mean.to_dict(),
                     "markers_available": rows[0]["markers_available"]})
    return rows


# ---------------------------------------------------------------------------------------------- driver


def set_main(pair, model_name, unit_name, seed):
    for unit in units(pair):
        if unit_name and folds.unit_name(*unit) != unit_name:
            continue
        samples = folds.training_samples(pair, *unit)
        if model_name == "set":
            pseudobulk_job(pair, unit)
        keys = [k for k in MODELS[model_name] if k in TRAINED]
        if not keys:
            continue
        if model_name == "scvi":
            cells = training_inputs.unit_cells(pair, unit, counts=True)
            for e in [seed] if seed is not None else expected_seeds(pair, "scvi_cells"):
                scvi_job(pair, unit, e, cells, samples)
            continue
        data = (training_inputs.unit_cells(pair, unit), training_inputs.pseudobulk_log(pair, unit), samples)
        for key in keys:
            for e in [seed] if seed is not None else expected_seeds(pair, key):
                set_encoder_job(pair, unit, key, e, data)
                print(f"{pair} {folds.unit_name(*unit)} {key} s{e}: attributed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER, required=True)
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--set", help="one unit (e.g. crossfit_0)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gather", action="store_true")
    parser.add_argument("--geneformer-pending", action="store_true", help="--gather without Geneformer, whose deletion runs last")
    parser.add_argument("--pilot", action="store_true", help="the FM pilot (mouse microglia; --model scgpt or geneformer)")
    parser.add_argument("--joint", action="store_true", help="Geneformer: the checks' joint deletions")
    parser.add_argument("--screen1", action="store_true", help="Geneformer: the two-stage screen's stage 1")
    parser.add_argument("--candidates", type=int, default=G.GF_SCREEN["candidates"], help="the screen's candidates (expanded on a failed acceptance)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    determinism.configure()
    assert args.gather != bool(args.model), "give --model or --gather"
    assert args.gather or not args.geneformer_pending, "--geneformer-pending goes with --gather"
    if args.gather:
        gather(args.pair, args.geneformer_pending)
    elif args.model in ("scgpt", "geneformer"):
        model, _ = S07["load_model"](args.model)
        if args.pilot:
            (pilot_scgpt if args.model == "scgpt" else pilot_geneformer)(model)
            return
        for unit in units(args.pair):
            if args.set and folds.unit_name(*unit) != args.set:
                continue
            if args.model == "scgpt":
                scgpt_job(args.pair, unit, model)
            elif args.joint:
                geneformer_joint(args.pair, unit, model)
            else:
                geneformer_job(args.pair, unit, model, args.shard, args.shards, args.screen1, args.candidates)
    else:
        set_main(args.pair, args.model, args.set, args.seed)
    assert not manifest.LABELS_READ, "stage 10 read bulk labels"


if __name__ == "__main__":
    main()
