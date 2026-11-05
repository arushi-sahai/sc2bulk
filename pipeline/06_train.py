"""Stage 06: trainings at each unit's frozen configuration, and their control, curve and recipe rows.
Label-blind on the bulk side.

Rows, per unit, model and encoder seed e. Every row reuses row A seed e's streams (split, init, training and
validation bags), so it differs from row A only in its stated change; draws() lists each job's streams.
  A  the set encoders on cells at the frozen configuration (stage 05's selection; human blood, aging: each
     fold's own), seeds 0-9 (headline) and 10-19 (attribution; stage 03b's splits; not human blood, aging;
     training_inputs.encoder_seeds, the one rule stage 08 shares).
     CytoSet's native head also scores every embedded side (S9).
  B  "CytoSet, pseudobulk": the cells' frozen configuration on one-profile bags (each training sample's
     log-CP10k pseudobulk), Inputs fit on the training-side cells (U1(1)), per_sample on the population
     route, abundance weighting and the gene scaler off (no effect on one-profile inputs; stated), row A's
     total optimizer steps with no patience stop and its own best-validation checkpoint
  C  untrained CytoSet: the frozen configuration at encoder_init e, embedded without training (embeddings
     only: no TrainingRecord; exempt from did-it-train)
  D  label-permuted: the frozen configuration with the single-cell 0/1 labels of the unit's holdout groups
     permuted within each side of split j (training_inputs.permuted_groups; class counts per side kept), at
     encoder seed j, j = 0-19 on every pair (10-19 on stage 03b's splits). Every label consumer reads them:
     BCE, the bags' class balance, early stopping and did-it-train, the distance feature's class means, the
     age loss's ages (the source group's mean age). The three set encoders and cytoset_pseudobulk (row B's
     recipe at row D CytoSet's total steps for the same j). Exempt from did-it-train (expected to fail; its
     pass rate is the rule's false-positive check).
  E  learning curve (training.LEARNING_CURVE_PAIRS): split e kept to 25% / 50% of each side and label's
     holdout groups (training_inputs.learning_curve_subset; nested), seeds 0-4. The three set encoders,
     cytoset_pseudobulk at row E CytoSet's steps for the same fraction and seed, and scVI refit on the
     subset (scvi_init e; its thinning median from the subset). 100% is row A (and J), reused.
  F  Fig 4A sweep: the frozen configuration with only the bag size n changed (train = validation = test = n),
     n in FIG4A_BAG_SIZES but 500 (row A, reused), seeds 0-4; n = 1 takes per_sample's population route.
     "all" (training.FIG4A_ALL_PAIRS) trains on the 02b store with bags of ALL_AVAILABLE_CAP cells (each the
     whole sample), Inputs fit on those cells with the gene scaler's p their own pseudobulk, and
     embeds each sample's cells as one bag (host gather where large); not CytoSet's where its frozen configuration
     has the age-similarity loss. scVI: row J's model at seed e embeds the same test bags (no
     retraining; its bulk rows copied from row J). With the age loss above AGE_LOSS_DEVICE_MAX_NCELL cells, both
     residents take the host gather (on_device()).
  G  own recipes: each set encoder's base plus training.OWN_RECIPES (CytoSet ICS;
     Set Transformer's repository recipe; ScRAT's run.sh with its sample mixup: sc2bulk.mixup, built once from
     the training side, train_bags at MIXUP_OFFSET + e, cell types at the deconvolution level), seeds 0-9;
     test bags at the recipe's bag size (ICS's 1,024 and Set Transformer's 1,000 are whole samples; disclosed).
  H  one type (Fig 6C; pairs with a one-type store): the frozen configuration on the one-type cells (bags of
     min(500, available)), split e kept to the store's samples (>= 2 groups per label per side), Inputs and
     the gene scaler's p from the one-type data; bulk stays the full profile. Seeds 0-4. The three set
     encoders, cytoset_pseudobulk on one-type pseudobulk at row H CytoSet's steps, and scVI (row J's model)
     on one-type test bags. CytoSet's seed 0 writes each sample's effective bag size (S3).
  I  per-sample route: where the frozen configuration uses per_sample, stage 05's
     per_sample_route configuration, trained on row A's path (no native head), seeds 0-9.
  J  scVI at stage 05v's selection, fit on split e's training side (validation: its validation side); the
     model is saved (rows F and H embed other cell sets with it); seeds 0-9
  MSE  CytoSet's MSE variant (aging pairs, Fig 4C; outside the search): the CytoSet base with objective
     mse (one change from the base) at the search seeds, on stage 05's base streams; the target is the
     z-scored age, early stopping on class-balanced validation MSE, did-it-train on d = 1 - MSE / MSE_const
     (harness.did_it_train_mse). Reported only.
H3, the thinned-bulk sensitivity: rows A and J at seeds 0-9 also embed each scored
bulk sample as 10 bags of FROZEN_NCELL pseudo-cells (training_inputs.thinned_bulk; the same draws for every
model), averaged: bulk rows of variant thinned_pseudocells.
Units: every unit for rows A, B and J; split67 takes no other row. Row E runs on its pairs' primary units,
row H on the pairs with a one-type store, MSE on the aging pairs.
Seeds: encoder_split / encoder_init / train_bags e, validation bags VALIDATION_BAGS_PURPOSE e, scvi_init e,
scVI's bulk thinning scvi_init SCVI_THIN_OFFSET + e; the rows' own streams in draws().
Embeddings: every training sample of the unit (row E: the subset's; row H: the store's; the mean over the
shared 10 test bags of the row's bag size; for scVI the mean latent over the same bags, U7), the fold's
held-out donors' single-cell samples (human blood, aging crossfit units; rows A, B, C, H, I and J), and each
scored bulk sample as one profile (scVI: thinned to the median training-cell library size, D3, plus the
observed-size variant). Single-cell rows' scoring unit follows the pair's rule (donor_id where the scoring
unit is the donor, else the sample), as bulk's does.
Checkpoints (rows A, B, G, I, J) hold state, configuration, record and any PCA basis; the Inputs statistics
are recomputed deterministically from the same split. Rows C, D, E, F and H keep embeddings only.

Outputs, in <outputs>/06_train/<pair>/<unit>/<row>/<model>/[<point>/] (model: cytoset, set_transformer,
scrat, scvi or cytoset_pseudobulk; point: f25 / f50 for row E, n<size> / n_all for row F):
  checkpoints/s<e>.pt (A, B, G, I) or checkpoints/s<e>/model.pt (J: scvi-tools' model directory)
  embeddings/s<e>.h5ad    obs: side, sample_id, scoring_unit_id, variant; X float32 (written last)
  native_head_s<e>.csv    CytoSet row A: side, sample_id, logit (the mean over the test bags)
  record_s<e>.json        the TrainingRecord (every trained row) or scVI's fit record (J, E); a trained set
                          encoder's also carries bulk_logs() (the capped bulk fraction, the embedding norms), and
                          on cells on_device (and host_reason when on_device() forced the host gather)
  bag_sizes.csv           row H CytoSet (seed 0): side, sample_id, cells (the effective bag size)
  (MSE: embeddings/search_s<i>.h5ad and record_search_s<i>.json, i the search seed)

  python pipeline/06_train.py --pair PAIR [--set UNIT --row ROW --model MODEL --seed E [--point POINT]]
  (with --pair alone, every job of the pair in order: the set encoders but ScRAT in the main environment,
  or only ScRAT's with --scrat, or only scVI's with --scvi)
  python pipeline/06_train.py --pair PAIR --list   every job of every environment as a JSON line, with whether it
                                                   is done and the jobs it reads; runs nothing (the GPU scheduler)
"""
import argparse
import dataclasses
import json
import math
import os
import sys

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from sc2bulk import bags, determinism, folds, harness, manifest, mixup, paths, preprocess, protocol, seeds, training_inputs
from sc2bulk.config import representations, search_space, settings
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIRS

STAGE = "06_train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TUPLE_SETTINGS = ("step_lr",)   # settings whose JSON lists are tuples in config
PSEUDOBULK = "cytoset_pseudobulk"
CHECKPOINT_ROWS = ("A", "B", "G", "I")   # and J's scVI model
HELDOUT_ROWS = ("A", "B", "C", "H", "I", "J")
AGE_LOSS_DEVICE_MAX_NCELL = 600   # wave 1b: the largest bag size at which CytoSet's largest batch, 200, was measured
                                  # to fit with the age loss on the largest resident unit


def network(encoder, config, in_dim):
    if encoder == "cytoset":
        from sc2bulk.encoders import cytoset
        return cytoset.network(config, in_dim)
    if encoder == "set_transformer":
        from sc2bulk.encoders import set_transformer
        return set_transformer.network(config, in_dim)
    from sc2bulk.encoders import scrat
    return scrat.network(config, in_dim, os.environ["SC2BULK_SCRAT"])


def selection_path(pair, unit, encoder):
    return (paths.output("05v_scvi_search", pair, "selection.json", unit) if encoder == "scvi"
            else paths.output("05_search", pair, encoder, unit) / "selection.json")


def frozen(pair, unit, encoder, key="frozen") -> dict:
    """The unit's frozen configuration: stage 05's selection (05v's for scVI); key "per_sample_route" for row I."""
    config = json.loads(selection_path(pair, unit, encoder).read_text())[key]
    return {k: tuple(v) if k in TUPLE_SETTINGS and v is not None else v for k, v in config.items()}


def has_per_sample_route(pair, unit, encoder) -> bool:
    return "per_sample_route" in json.loads(selection_path(pair, unit, encoder).read_text())


def age_loss_all_skipped(pair, unit, row, model, point) -> bool:
    """Row F's "all" for a unit whose frozen CytoSet configuration has the age-similarity loss isn't trained: the whole batch must pass at once, above the 3,000-cell micro-batch cap. 08b and 11b re-derive it."""
    return row == "F" and model == "cytoset" and point == "n_all" and bool(frozen(pair, unit, "cytoset").get("age_similarity_loss"))


def on_device(config, train, val) -> tuple:
    """(whether both residents are on the device, the reason when forced to the host). With the age-similarity loss
    the whole batch passes at once, so above AGE_LOSS_DEVICE_MAX_NCELL cells it takes the host gather."""
    if config.get("age_similarity_loss") and config["ncell"] > AGE_LOSS_DEVICE_MAX_NCELL:
        return False, f"age-similarity loss: whole batch in one pass, above {AGE_LOSS_DEVICE_MAX_NCELL} cells"
    return harness.fits_on_device(train, val), None


def encoder_of(row, model):
    return "cytoset" if row in ("B", "C", "MSE") or model == PSEUDOBULK else model


def configuration(row, pair, unit, model, point) -> dict:
    """The configuration a job trains: the frozen one (row F with its bag size; row I the per-sample route), or
    the base plus the own recipe (G) or the MSE objective."""
    encoder = encoder_of(row, model)
    if row == "G":
        return {**search_space.base(encoder), **T.OWN_RECIPES[encoder], "encoder": encoder}
    if row == "MSE":
        return {**search_space.base("cytoset"), **search_space.OUTSIDE_SEARCH["objective_mse"], "encoder": encoder}
    config = {**frozen(pair, unit, encoder, "per_sample_route" if row == "I" else "frozen"), "encoder": encoder}
    assert config["ncell"] == T.FROZEN_NCELL, "bag size is out of the search (frozen §38 H2): stage 08 labels rows FROZEN_NCELL"
    if row == "F":
        config["ncell"] = representations.ALL_AVAILABLE_CAP if point == "n_all" else int(point[1:])
    return config


def draws(row, model, e) -> dict:
    """role -> (seed purpose, index): every stream a job draws, unit-keyed. Rows reuse row A seed
    e's streams; the MSE variant stage 05's base streams at search seed e (an intended pairing), row E's scVI
    row J's. The test bags (test_bags 0) are shared by every model (U7)."""
    if model == "scvi":
        out = {"split": ("encoder_split", e), "scvi_init": ("scvi_init", e), "scvi_thinning": ("scvi_init", T.SCVI_THIN_OFFSET + e)}
    elif row == "MSE":
        out = {"search_split": ("search_split", e), "search_init": ("search_init", e), "search_bags": ("search_sampling", e),
               "validation_bags": (T.VALIDATION_BAGS_PURPOSE, T.SEARCH_INDEX_OFFSET + e)}
    else:
        out = {"split": ("encoder_split", e), "init": ("encoder_init", e), "bags": ("train_bags", e),
               "validation_bags": (T.VALIDATION_BAGS_PURPOSE, e)}
    if row == "D":
        out["permuted_labels"] = ("permutation", T.PERMUTATION_OFFSET + e)
    if row == "E":
        out["subset"] = ("learning_curve", e)
    if row == "G" and model == "scrat":
        out["mixup"] = ("train_bags", T.MIXUP_OFFSET + e)
    if row in ("A", "J") and e < 10:
        out["thinned_bulk"] = ("spare_3", 0)
    return out


def stream(pair, unit, row, model, e):
    """harness.train's seeds: init, training bags, validation bags."""
    d = draws(row, model, e)
    roles = ("search_init", "search_bags", "validation_bags") if row == "MSE" else ("init", "bags", "validation_bags")
    init, bag, val = (seeds.derived(d[r][0], pair, d[r][1], *unit) for r in roles)
    return dict(init_seed=init, bag_seed=bag, val_seed=val)


def cell_subset(row):
    return "one_type" if row == "H" else "all"


def row_cells(row, pair, unit, point, counts=False, deconvolution_types=False) -> protocol.Cells:
    if point == "n_all":
        return training_inputs.unit_cells_all_available(pair, unit, counts=counts)
    return training_inputs.unit_cells(pair, unit, counts=counts, cell_subset=cell_subset(row), deconvolution_types=deconvolution_types)


def row_sides(row, pair, unit, e, point, cells) -> dict:
    """sample_id -> "train" / "validation": split e (the MSE variant: search split e); row E its subset; row H
    the samples in the one-type store, with >= 2 holdout groups per label on each side."""
    if row == "MSE":
        return training_inputs.split_sides(pair, unit, "search_split", e)
    if row == "E":
        return training_inputs.learning_curve_subset(pair, unit, e, int(point[1:]) / 100)
    sides = training_inputs.encoder_sides(pair, unit, e)
    if row == "H":
        sides = one_type_sides(sides, cells)
        short = one_type_shortfall(sides, cells)
        assert not short, f"{pair} {unit} split {e}: " + "; ".join(short)
    return sides


def one_type_sides(sides, cells) -> dict:
    """Split e's samples that are in the one-type store."""
    stored = set(cells.sample_id)
    return {s: side for s, side in sides.items() if s in stored}


def one_type_shortfall(sides, cells) -> list:
    """Row H's sides short of their holdout groups per label: >= 2 on each side. Smoke keeps few
    groups, so there the training side needs only 1; the validation side needs 2 in both (harness.did_it_train)."""
    need = {"train": 1 if settings.SMOKE else 2, "validation": 2}
    kept = cells.samples.loc[list(sides)].assign(side=pd.Series(sides))
    counts = {(side, label): kept.loc[(kept["side"] == side) & (kept["group"] == label), "holdout_group"].nunique()
              for side in need for label in (0, 1)}
    return [f"{n} one-type {side} groups of label {label}" for (side, label), n in counts.items() if n < need[side]]


def learning_curve_counts(pair, unit, e, fraction) -> dict:
    """label -> row E's validation holdout groups at the fraction: ceil(fraction x n) of the n of that label on
    split e's validation side, as training_inputs.learning_curve_subset keeps them (it asserts >= 2)."""
    sides = training_inputs.encoder_sides(pair, unit, e)
    samples = folds.training_samples(pair, *unit)
    groups = samples.assign(side=samples["sample_id"].map(sides)).groupby("holdout_group")[["group", "side"]].first()
    validation = groups[groups["side"] == "validation"]
    return {label: math.ceil(fraction * int((validation["group"] == label).sum())) for label in (0, 1)}


def learning_curve_shortfall(pair, unit, e, point) -> list:
    """Row E's labels short of 2 validation holdout groups at split e and point (learning_curve_counts)."""
    counts = learning_curve_counts(pair, unit, e, int(point[1:]) / 100)
    return [f"{n} validation groups of label {label}" for label, n in counts.items() if n < 2]


def smoke_marker(pair, unit, row):
    """Smoke only: written when the unit's row (H or E) is skipped (08b and 11b treat that row as absent there)."""
    return paths.output(STAGE, pair, row, unit) / "smoke_skipped.json"


def smoke_skip(pair, unit, row, short) -> bool:
    """Smoke only: whether any of the row's splits falls short (short: {split: shortfalls}). Then the unit's jobs of
    that row, every seed's, point's and model's, aren't listed; the shortfalls go to stderr and smoke_marker."""
    short = {k: s for k, s in short.items() if s}
    marker = smoke_marker(pair, unit, row)
    marker.unlink(missing_ok=True)
    if short:
        print(f"smoke: {pair} {folds.unit_name(*unit)} row {row} skipped: {short}", file=sys.stderr, flush=True)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(short))
    return bool(short)


def smoke_skips_one_type(pair, unit) -> bool:
    """Smoke only: whether any seed's row H split falls short (one_type_shortfall; smoke_skip). The real run lists
    every seed and row_sides asserts."""
    cells = row_cells("H", pair, unit, "")
    return smoke_skip(pair, unit, "H", {e: one_type_shortfall(one_type_sides(training_inputs.encoder_sides(pair, unit, e), cells), cells)
                                        for e in training_inputs.encoder_seeds(pair, "H")})


def smoke_skips_learning_curve(pair, unit) -> bool:
    """Smoke only: whether any seed's row E subset at any point falls short (learning_curve_shortfall; smoke_skip).
    The real run lists every seed and point and training_inputs.learning_curve_subset asserts."""
    return smoke_skip(pair, unit, "E", {f"s{e} {point}": learning_curve_shortfall(pair, unit, e, point)
                                        for point in points(pair, "E") for e in training_inputs.encoder_seeds(pair, "E")})


def heldout_cells(pair, unit, counts=False, cell_subset="all"):
    """The fold's held-out donors' stored cells (or one-type cells) on the unit's panel (human blood, aging
    crossfit only): log-CP10k (or counts), and their samples' log-CP10k pseudobulk (summed counts over the
    shared genes, then the panel, as stage 02)."""
    data = folds.scored_cells(pair, *unit, cell_subset=cell_subset)
    genes, chosen = training_inputs.shared(pair), training_inputs.panel(pair, unit)
    columns = pd.Index(genes).get_indexer(chosen)
    raw = data[:, genes].X
    X = (sp.csr_matrix(raw, dtype=np.float32) if counts else preprocess.log_normalize(raw))[:, columns].tocsr()
    sample_id = data.obs["sample_id"].astype(str).to_numpy()
    samples = folds.single_cell_samples(pair).set_index("sample_id").loc[sorted(set(sample_id))]
    pseudobulk = {s: preprocess.log_normalize(preprocess.summed(raw[sample_id == s])[None, :])[:, columns].toarray().ravel()
                  for s in samples.index}
    cells = protocol.Cells(X=X, genes=chosen, cell_id=data.obs_names.astype(str).to_numpy(), sample_id=sample_id,
                           cell_type=data.obs["cell_type"].astype(str).to_numpy(),
                           library_size=data.obs["total_counts"].to_numpy(dtype=float), samples=samples[["group", "holdout_group"]])
    return cells, pseudobulk


def write_embeddings(path, blocks, record, cost):
    """blocks: (side, sample ids, scoring unit ids, variant, X) -> one h5ad, written through manifest."""
    obs = pd.concat([pd.DataFrame({"side": side, "sample_id": ids, "scoring_unit_id": units, "variant": variant})
                     for side, ids, units, variant, _ in blocks], ignore_index=True)
    X = np.vstack([x for *_, x in blocks]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    anndata.AnnData(X, obs=obs.set_index(obs.index.astype(str))).write_h5ad(manifest.temporary(path))
    manifest.finish(path, record, cost)


def side_of(sides, s):
    return "single_cell_train" if sides[s] == "train" else "single_cell_validation"


def scoring_units(pair) -> dict:
    """sample -> its scoring unit: the donor where the pair scores donors, else the sample."""
    samples = folds.single_cell_samples(pair)
    by = "donor_id" if PAIRS[pair]["scoring_unit"] == "donor" else "sample_id"
    return dict(zip(samples["sample_id"].astype(str), samples[by].astype(str)))


def single_cell_blocks(samples, sides, scoring_unit, embedding):
    return [(side_of(sides, s), [s], [scoring_unit[s]], "primary", embedding[i:i + 1]) for i, s in enumerate(samples)]


def native_head(net, cells, inputs, sample_ids, n_cells, pair, unit, residents=None) -> np.ndarray:
    """CytoSet's own head on each sample's test bags, averaged over the bags (S9's native-head row)."""
    residents = residents or (harness.Resident(cells, inputs, DEVICE, harness.fits_on_device(cells)),)
    holder = {s: r for r in residents for s in np.unique(r.cells.sample_id)}
    test = bags.test_bags(cells.cell_id, cells.sample_id, list(sample_ids), n_cells, representations.TEST_TIME_BAGS, pair, unit)
    net.eval()
    with torch.no_grad():
        return np.array([float(harness.forward_bags(net, [(s, ids) for ids in test[s]], holder[s])[1].mean()) for s in sample_ids])


def cells_steps(pair, unit, row, e, point) -> int:
    """The CytoSet-cells job's total optimizer steps at seed e: a pseudobulk retraining's budget (row B: row A's; rows D, E, H: the same row's, fraction and seed)."""
    out = paths.output(STAGE, pair, "A" if row == "B" else row, unit) / "cytoset" / point
    return json.loads((out / f"record_s{e}.json").read_text())["steps"]


def profiles(ids, profile_of: dict, genes) -> protocol.Profiles:
    return protocol.Profiles(X=sp.csr_matrix(np.vstack([profile_of[s] for s in ids])), genes=genes, sample_id=np.array(ids),
                             library_size=np.ones(len(ids)))


def profile_cells(side: protocol.Cells, pseudobulk: dict) -> protocol.Cells:
    """A side's samples as one-profile 'cells' (their log-CP10k pseudobulk), for row B and cytoset_pseudobulk."""
    ids = sorted(side.samples.index)
    return protocol.Cells(X=sp.csr_matrix(np.vstack([pseudobulk[s] for s in ids])), genes=side.genes, cell_id=np.array(ids),
                          sample_id=np.array(ids), cell_type=np.array(["profile"] * len(ids)), library_size=np.ones(len(ids)),
                          samples=side.samples.loc[ids])


def log_cp10k(counts, totals) -> np.ndarray:
    """Pseudo-cells' panel counts as log-CP10k over the shared genes (their totals), as preprocess.log_normalize."""
    return np.log1p(counts / totals[..., None] * settings.TARGET_SUM)


def set_encoder_row(row, pair, unit, model, e, point, record, cost, out):
    config = configuration(row, pair, unit, model, point)
    one_profile = row == "B" or model == PSEUDOBULK
    cells = row_cells(row, pair, unit, point, deconvolution_types=bool(config.get("mixup")))
    sides = row_sides(row, pair, unit, e, point, cells)
    permuted = training_inputs.permuted_groups(pair, unit, e) if row == "D" else None
    if permuted is not None:   # every label consumer reads the samples' group
        cells = dataclasses.replace(cells, samples=cells.samples.assign(group=permuted["group"]))
    train, val = training_inputs.split_cells(cells, sides)
    scoring_unit = scoring_units(pair)
    pseudobulk = (training_inputs.pseudobulk_log_all_available(pair, unit) if point == "n_all"
                  else training_inputs.pseudobulk_log(pair, unit, cell_subset(row)))
    bulk_table = training_inputs.scored_bulk(pair, unit)
    bulk = training_inputs.bulk_profiles(pair, unit, bulk_table["sample_id"])
    build = lambda c: (lambda: network(c["encoder"], c, harness.input_width(c, len(cells.genes))))
    samples = sorted(set(train.samples.index) | set(val.samples.index))
    ages = (training_inputs.training_ages(pair, unit, train.samples.index, source=None if permuted is None else permuted["source"])
            if config.get("age_similarity_loss") else None)
    targets = training_inputs.training_ages(pair, unit, train.samples.index, also=val.samples.index) if row == "MSE" else None
    test_n = 0 if point == "n_all" else config["ncell"]   # "all": each sample's cells as one bag
    if one_profile:   # one-profile bags of the training samples' pseudobulk
        config = {**config, "ncell": 1, "abundance_weighting": False, "pseudobulk_gene_scaler": False,
                  "n_epochs": 10 ** 6, "patience": None, "max_steps": cells_steps(pair, unit, row, e, point)}
    config["micro_batch"] = harness.micro_batch_for(config["encoder"], config)
    inputs = harness.Inputs(config, train, pseudobulk)   # moments from the training-side cells (U1(1))
    trained, heads, placement = None, [], {}
    if one_profile:
        net, state, trained, _, _ = harness.train(build(config), config, profile_cells(train, pseudobulk), profile_cells(val, pseudobulk),
                                                  inputs, device=DEVICE, ages=ages, **stream(pair, unit, row, model, e))
        sample_embedding = harness.embed_profiles(net, profiles(samples, pseudobulk, cells.genes), inputs, DEVICE)
    else:   # one device copy of the training and validation cells, for training and embedding
        fits, host_reason = on_device(config, train, val)
        placement = {"on_device": fits, **({"host_reason": host_reason} if host_reason else {})}
        residents = harness.Resident(train, inputs, DEVICE, fits), harness.Resident(val, inputs, DEVICE, fits)
        if row == "C":   # untrained: the network as initialized at encoder_init e
            torch.manual_seed(stream(pair, unit, row, model, e)["init_seed"])
            net = build(config)().to(DEVICE).eval()
            state = None
        elif config.get("mixup"):   # ScRAT's recipe: trains on the synthetic samples only (inter_only)
            purpose, index = draws(row, model, e)["mixup"]
            mixed_seed = seeds.derived(purpose, pair, index, *unit)
            mixed = mixup.synthetic(train, inputs, training_inputs.sample_strata(pair, unit), np.random.default_rng([mixed_seed, 0]))
            net, state, trained, _, _ = harness.train(
                build(config), config, mixed, val, inputs, device=DEVICE, residents=(harness.Precomputed(mixed, inputs, DEVICE), residents[1]),
                train_bag_list=mixup.fixed_bags(mixed, config["ncell"], np.random.default_rng([mixed_seed, 1])),
                **stream(pair, unit, row, model, e))
            del mixed
        else:
            net, state, trained, _, _ = harness.train(build(config), config, train, val, inputs, device=DEVICE, ages=ages, targets=targets,
                                                      residents=residents, **stream(pair, unit, row, model, e))
        sample_embedding = harness.embed_samples(net, cells, inputs, samples, test_n, pair, unit, DEVICE, residents)
        if row == "A" and model == "cytoset":
            heads.append(pd.DataFrame({"side": [side_of(sides, s) for s in samples], "sample_id": samples,
                                       "logit": native_head(net, cells, inputs, samples, test_n, pair, unit, residents)}))
        del residents   # free the training device copies before the held-out cells take theirs
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    blocks = single_cell_blocks(samples, sides, scoring_unit, sample_embedding)
    if row == "H" and model == "cytoset" and e == 0:   # each sample's effective bag size
        n_cells = pd.Series(cells.sample_id).value_counts()
        write_csv(out / "bag_sizes.csv", pd.DataFrame({"side": [side_of(sides, s) for s in samples], "sample_id": samples,
                                                       "cells": [min(int(n_cells[s]), config["ncell"]) for s in samples]}), record, cost)
    if unit[0] == "crossfit" and row in HELDOUT_ROWS:   # the fold's held-out donors' single-cell samples (Fig 2B, Fig 6C)
        held, held_profiles = heldout_cells(pair, unit, cell_subset=cell_subset(row))
        ids = sorted(held.samples.index)
        if one_profile:
            held_embedding = harness.embed_profiles(net, profiles(ids, held_profiles, cells.genes), inputs, DEVICE)
        else:
            inputs.pseudobulk_log.update(held_profiles)   # the gene scaler's p for held-out samples (label-free)
            inputs.prepare(held)
            held_embedding = harness.embed_samples(net, held, inputs, ids, test_n, pair, unit, DEVICE)
            if row == "A" and model == "cytoset":
                heads.append(pd.DataFrame({"side": "single_cell_heldout", "sample_id": ids,
                                           "logit": native_head(net, held, inputs, ids, test_n, pair, unit)}))
        blocks.append(("single_cell_heldout", ids, [scoring_unit[s] for s in ids], "primary", held_embedding))
    bulk_ids, bulk_units = list(bulk_table["sample_id"]), list(bulk_table["scoring_unit_id"])
    bulk_embedding = harness.embed_profiles(net, bulk, inputs, DEVICE)
    blocks.append(("bulk", bulk_ids, bulk_units, "primary", bulk_embedding))
    if "thinned_bulk" in draws(row, model, e):   # H3: bags of thinned pseudo-cells, the same draws for every model
        thinned = [harness.embed_pseudocells(net, log_cp10k(c, t), t, inputs, DEVICE) for _, c, t in training_inputs.thinned_bulk(pair, unit, bulk_table)]
        blocks.append(("bulk", bulk_ids, bulk_units, "thinned_pseudocells", np.vstack(thinned)))
    if heads:
        with torch.no_grad():
            bulk_logits = net.head(torch.as_tensor(bulk_embedding, device=DEVICE)).cpu().numpy()
        heads.append(pd.DataFrame({"side": "bulk", "sample_id": bulk_table["sample_id"], "logit": bulk_logits}))
        write_csv(out / f"native_head_s{e}.csv", pd.concat(heads, ignore_index=True), record, cost)
    if trained is not None:
        note = {"recipe": encoder_of(row, model)} if row == "G" else {"objective": "mse"} if row == "MSE" else {}
        logged = bulk_logs(inputs, bulk, bulk_embedding, sample_embedding)
        write_json(out / f"record_{stem(row, e)}.json", {**dataclasses.asdict(trained), **note, **placement, **logged}, record, cost)
        if row in CHECKPOINT_ROWS:
            save(out, e, state, config, trained, inputs, note, record, cost)
    write_embeddings(out / "embeddings" / f"{stem(row, e)}.h5ad", blocks, record, cost)   # last: its manifest marks the job done


def bulk_logs(inputs, bulk: protocol.Profiles, bulk_embedding, sample_embedding) -> dict:
    """capped_bulk_fraction: the share of the scored bulk's standardized input values at the ±INPUT_CAP clamp
    (0 under "none", which standardizes nothing). The mean L2 norm of the bulk embeddings and of the
    unit's single-cell sample embeddings, and their ratio (for ScRAT, item 13's
    post-norm one-token norm shift: a one-profile bag's single token against a mean over many)."""
    z = inputs._standardize(sp.csr_matrix(bulk.X).toarray(), own=False)
    capped = 0.0 if inputs.level == "none" else float((np.abs(z) >= T.INPUT_CAP).mean())
    bulk_norm = float(np.linalg.norm(bulk_embedding, axis=1).mean())
    cell_norm = float(np.linalg.norm(sample_embedding, axis=1).mean())
    return {"capped_bulk_fraction": capped, "bulk_embedding_norm": bulk_norm, "single_cell_embedding_norm": cell_norm,
            "embedding_norm_ratio": bulk_norm / cell_norm}


def scvi_row(row, pair, unit, e, point, record, cost, out):
    """Row J (and E: refit on the subset): scVI fit on the split's training side. Rows F and H: row J's model at
    seed e embeds other cell sets (n-cell, all-available or one-type test bags; no retraining),
    and the bulk rows are copied from row J's embeddings so each file stands alone."""
    from sc2bulk.encoders import scvi as adapter
    cells = row_cells(row, pair, unit, point, counts=True)
    sides = row_sides(row, pair, unit, e, point, cells)
    scoring_unit = scoring_units(pair)
    streams = {k: seeds.derived(purpose, pair, index, *unit) for k, (purpose, index) in draws(row, "scvi", e).items()}
    row_j = paths.output(STAGE, pair, "J", unit) / "scvi"
    if row in ("J", "E"):
        train, val = training_inputs.split_cells(cells, sides)
        model, fit = adapter.fit(train.X, val.X, frozen(pair, unit, "scvi"), seed=streams["scvi_init"])
    else:
        model = adapter.load(row_j / "checkpoints" / f"s{e}", cells.X)
    n_cells = 0 if point == "n_all" else int(point[1:]) if row == "F" else T.FROZEN_NCELL
    samples = sorted(sides)
    blocks = single_cell_blocks(samples, sides, scoring_unit, bag_latent(adapter, model, cells, samples, pair, unit, n_cells))
    if unit[0] == "crossfit" and row in HELDOUT_ROWS:
        held, _ = heldout_cells(pair, unit, counts=True, cell_subset=cell_subset(row))
        ids = sorted(held.samples.index)
        blocks.append(("single_cell_heldout", ids, [scoring_unit[s] for s in ids], "primary",
                       bag_latent(adapter, model, held, ids, pair, unit, n_cells)))
    if row in ("F", "H"):
        j = anndata.read_h5ad(row_j / "embeddings" / f"s{e}.h5ad")
        j = j[j.obs["side"] == "bulk"]
        for variant, part in j.obs.groupby("variant", sort=False, observed=True):
            blocks.append(("bulk", list(part["sample_id"]), list(part["scoring_unit_id"]), variant, j[part.index].X))
        write_embeddings(out / "embeddings" / f"s{e}.h5ad", blocks, record, cost)
        return
    bulk_table = training_inputs.scored_bulk(pair, unit)
    counts = training_inputs.bulk_profiles(pair, unit, bulk_table["sample_id"], counts=True).X
    ids, units = list(bulk_table["sample_id"]), list(bulk_table["scoring_unit_id"])
    blocks.append(("bulk", ids, units, "primary", adapter.embed_profiles(model, counts, fit["median_library_size"], streams["scvi_thinning"])))
    blocks.append(("bulk", ids, units, "observed_library", adapter.embed_profiles(model, counts, 0.0, streams["scvi_thinning"], thinned=False)))
    if "thinned_bulk" in streams:   # H3: each bag's mean latent, averaged over the bags
        thinned = [adapter.latent(model, c.reshape(-1, c.shape[2])).reshape(*c.shape[:2], -1).mean(axis=(0, 1))
                   for _, c, _ in training_inputs.thinned_bulk(pair, unit, bulk_table)]
        blocks.append(("bulk", ids, units, "thinned_pseudocells", np.vstack(thinned)))
    if row == "J":
        saved, final = out / "checkpoints" / f".save_s{e}", out / "checkpoints" / f"s{e}" / "model.pt"
        model.save(str(saved), overwrite=True)   # scvi-tools writes model.pt into a directory
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(saved / "model.pt", manifest.temporary(final))
        saved.rmdir()
        manifest.finish(final, record, cost)
    write_json(out / f"record_s{e}.json", {**fit, "config": frozen(pair, unit, "scvi")}, record, cost)
    write_embeddings(out / "embeddings" / f"s{e}.h5ad", blocks, record, cost)   # last: its manifest marks the job done


def bag_latent(adapter, model, cells, samples, pair, unit, n_cells) -> np.ndarray:
    """Each sample's mean latent over the shared 10 test bags of n_cells cells (U7; 0: all its cells, once)."""
    test = bags.test_bags(cells.cell_id, cells.sample_id, samples, n_cells, representations.TEST_TIME_BAGS, pair, unit)
    z = adapter.latent(model, cells.X)
    row_of = pd.Series(np.arange(len(cells.cell_id)), index=cells.cell_id)
    return np.vstack([np.mean([z[row_of[b].to_numpy()].mean(axis=0) for b in test[s]], axis=0) for s in samples])


def stem(row, e):
    return f"search_s{e}" if row == "MSE" else f"s{e}"


def write_csv(path, table, record, cost):
    table.to_csv(manifest.temporary(path), index=False)
    manifest.finish(path, record, cost)


def write_json(path, content, record, cost):
    manifest.temporary(path).write_text(json.dumps(content, default=str))
    manifest.finish(path, record, cost)


def save(out, e, state, config, trained, inputs, note, record, cost):
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoints" / f"s{e}.pt"
    basis = {"pca_mean": inputs.pca_mean, "components": inputs.components} if inputs.pca else {}
    torch.save({"state": state, "config": config, "record": dataclasses.asdict(trained), **note, **basis}, manifest.temporary(checkpoint))
    manifest.finish(checkpoint, record, cost)


def points(pair, row) -> list:
    """A row's curve points: row E's fractions (f25, f50), row F's bag sizes (n<size>, n_all), else one ("")."""
    if row == "E":
        return [f"f{round(100 * f)}" for f in T.LEARNING_CURVE_FRACTIONS]
    if row == "F":
        return [f"n{n}" for n in search_space.FIG4A_BAG_SIZES if n not in (T.FROZEN_NCELL, "all")] + (
            ["n_all"] if pair in T.FIG4A_ALL_PAIRS else [])
    return [""]


def jobs(pair, environment) -> list:
    """Every (unit, row, model, seed, point) of a pair for one environment, in dependency order: row B after row
    A's CytoSet, each cytoset_pseudobulk after its row's CytoSet-cells jobs, rows F and H's scVI after row J. Smoke
    leaves out a unit's whole row H when any seed's one-type split falls short (smoke_skips_one_type), and its whole
    row E when any seed's subset at any point does (smoke_skips_learning_curve)."""
    out = []
    encoders = {"main": ("cytoset", "set_transformer"), "scrat": ("scrat",), "scvi": ()}[environment]
    seeds_of = lambda row: training_inputs.encoder_seeds(pair, row)
    for unit in folds.units(pair):
        rows = ["A", "B", "J"] if unit[0] == "split67" else ["A", "B", "C", "J", "D", "E", "F", "G", "H", "I", "MSE"]
        if pair not in T.LEARNING_CURVE_PAIRS:
            rows.remove("E")
        if "one_type" not in PAIRS[pair] and "H" in rows:
            rows.remove("H")
        if pair not in T.AGE_LOSS_PAIRS and "MSE" in rows:   # the aging pairs
            rows.remove("MSE")
        if settings.SMOKE and "H" in rows and smoke_skips_one_type(pair, unit):
            rows.remove("H")
        if settings.SMOKE and "E" in rows and smoke_skips_learning_curve(pair, unit):
            rows.remove("E")
        for row in rows:
            if row in ("B", "C", "MSE"):
                models = ("cytoset",) if environment == "main" else ()
            elif row == "J":
                models = ("scvi",) if environment == "scvi" else ()
            elif row == "I":
                models = tuple(m for m in encoders if has_per_sample_route(pair, unit, m))
            else:
                models = encoders + ((PSEUDOBULK,) if environment == "main" and row in ("D", "E", "H") else ()) \
                    + (("scvi",) if environment == "scvi" and row in ("E", "F", "H") else ())
            out += [(unit, row, model, e, point) for point in points(pair, row) for model in models for e in seeds_of(row)
                    if not age_loss_all_skipped(pair, unit, row, model, point)]
    return out


def run_record(pair, unit, row, model, e, point, code) -> dict:
    """The job's run record (manifest.is_done compares it); code: the stage code hash. Row B and cytoset_pseudobulk
    read their CytoSet-cells job's record, scVI rows F and H row J's model: build it only once those are done."""
    encoder = "scvi" if model == "scvi" else encoder_of(row, model)
    sources = [*folds.inputs(pair, one_type=row == "H"), settings.OUTPUTS / "03_splits" / pair / "splits.csv",
               settings.OUTPUTS / "03_splits" / pair / "donor_folds.csv", paths.output("04_panel", pair, "panel.csv", unit),
               paths.output("04_panel", pair, "shared_genes.csv"), settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad",
               settings.OUTPUTS / "02_preprocess" / pair / "bulk_samples.csv"]
    if row not in ("G", "MSE"):   # their configurations are bases and recipes, not stage 05's selection
        sources.append(selection_path(pair, unit, encoder))
    if row != "MSE" and e >= 10:
        sources.append(settings.OUTPUTS / "03b_attribution_splits" / pair / "splits.csv")
    if point == "n_all":
        index = paths.output("02b_all_available", pair, "index.csv")   # shard content hashes
        cells = pd.read_csv(index, dtype={"sample_id": str}).set_index("sample_id")["cells"]
        assert (cells.loc[folds.training_samples(pair, *unit)["sample_id"]] > settings.MAX_STORED_CELLS).any(), \
            f"{pair} {unit}: no training sample has more cells than the stored {settings.MAX_STORED_CELLS} (training.FIG4A_ALL_PAIRS)"
        sources.append(index)
    if model == "scvi" and row in ("F", "H"):   # row J's model and its bulk rows
        sources += [paths.output(STAGE, pair, "J", unit) / "scvi" / "checkpoints" / f"s{e}" / "model.pt",
                    paths.output(STAGE, pair, "J", unit) / "scvi" / "embeddings" / f"s{e}.h5ad"]
    record = {"stage": STAGE, "pair": pair, "unit": unit, "row": row, "model": model, "seed": e, "point": point, "smoke": settings.SMOKE,
              "code": code, "inputs": manifest.input_hashes(sources)}
    if row == "B" or model == PSEUDOBULK:   # the budget itself, not the cells job's record (whose timing fields change on every rerun)
        record["cells_steps"] = cells_steps(pair, unit, row, e, point)
    return record


def done_marker(pair, unit, row, model, e, point):
    """The embeddings file, written last, after every other output: its manifest marks the job done."""
    return paths.output(STAGE, pair, row, unit) / model / point / "embeddings" / f"{stem(row, e)}.h5ad"


def upstream(unit, row, model, e, point) -> list:
    """The jobs whose outputs a job reads (jobs() lists them first): row B row A's CytoSet at seed e, each
    cytoset_pseudobulk job its row's CytoSet-cells job, scVI rows F and H row J."""
    if row == "B":
        return [(unit, "A", "cytoset", e, "")]
    if model == PSEUDOBULK:
        return [(unit, row, "cytoset", e, point)]
    if model == "scvi" and row in ("F", "H"):
        return [(unit, "J", "scvi", e, "")]
    return []


def listing(pair):
    """--list: one JSON line per job of every environment, in jobs()' order, for the GPU scheduler:
    the job, its environment, its upstream jobs (after), and whether it is done.
    A job is done only when its upstream jobs are, so a record that reads an upstream's outputs is built only then."""
    code = manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair)
    done = {}
    for environment in ("main", "scrat", "scvi"):
        for job in jobs(pair, environment):
            after = upstream(*job)
            done[job] = all(done[a] for a in after) and manifest.is_done([done_marker(pair, *job)], run_record(pair, *job, code))
            unit, row, model, e, point = job
            print(json.dumps({"unit": folds.unit_name(*unit), "row": row, "model": model, "seed": e, "point": point,
                              "env": environment, "done": done[job],
                              "after": [[folds.unit_name(*u), r, m, s, p] for u, r, m, s, p in after]}), flush=True)


def run(pair, unit, row, model, e, point):
    out = paths.output(STAGE, pair, row, unit) / model / point
    record = run_record(pair, unit, row, model, e, point, manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair))
    if manifest.is_done([done_marker(pair, unit, row, model, e, point)], record):
        print(f"{pair} {folds.unit_name(*unit)} {row} {model} {point} s{e}: done", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    cost = manifest.begin(device=DEVICE)
    if model == "scvi":
        scvi_row(row, pair, unit, e, point, record, cost, out)
    else:
        set_encoder_row(row, pair, unit, model, e, point, record, cost, out)
    print(f"{pair} {folds.unit_name(*unit)} {row} {model} {point} s{e}: trained and embedded", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--set")
    parser.add_argument("--row", choices=("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "MSE"))
    parser.add_argument("--model", choices=("cytoset", "set_transformer", "scrat", "scvi", PSEUDOBULK))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--point", default="", help="row E: f25 / f50; row F: n<size> / n_all")
    parser.add_argument("--scrat", action="store_true", help="with --pair alone: ScRAT's jobs (its environment)")
    parser.add_argument("--scvi", action="store_true", help="with --pair alone: scVI's jobs (its environment)")
    parser.add_argument("--list", action="store_true", help="with --pair alone: every job as a JSON line, run nothing (the scheduler)")
    args = parser.parse_args()
    determinism.configure()
    one_job = (args.set, args.row, args.model, args.seed)
    assert all(a is None for a in one_job) or all(a is not None for a in one_job), "one job takes --set, --row, --model, --seed"
    assert not (args.set and (args.scrat or args.scvi or args.list)), "--scrat / --scvi / --list take --pair alone"
    if args.list:
        listing(args.pair)
        return
    if args.set is None:
        for job in jobs(args.pair, "scrat" if args.scrat else "scvi" if args.scvi else "main"):
            run(args.pair, *job)
        return
    unit = next(u for u in folds.units(args.pair) if folds.unit_name(*u) == args.set)
    job = (unit, args.row, args.model, args.seed, args.point)
    assert any(job in jobs(args.pair, env) for env in ("main", "scrat", "scvi")), f"{args.pair}: {job} isn't one of stage 06's jobs"
    run(args.pair, *job)


if __name__ == "__main__":
    main()
