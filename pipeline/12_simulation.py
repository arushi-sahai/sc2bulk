"""Stage 12: the planted-signal simulation, a positive control (Fig 5D, S9 Table). No bulk
data and no bulk label is read: the "simulated bulk" is each sample's simulated cells summed.

On human ileum, Crohn's (unit all_0; config/simulation.py), from its real stored cells:
- design (label-free but for stratifying the simulated label by the real single-cell 0/1 group): 50 planted panel genes
  G (and halves G+ / G-), a simulated 0/1 label per holdout group, and per scenario the strength (the weakest grid level
  at which a fixed oracle reaches AUROC 0.90 over all the unit's samples); half A / half B of each encoder split's
  validation side;
- trainings: per scenario (mean_shift, structure_only), set encoder (stage 05's frozen configuration for the unit) and
  encoder split e = 0-4, the harness retrains on the simulated cells of split e's training side with the simulated
  labels; half A early-stops, keeps the checkpoint and decides did-it-train; half B is used by no training. Row A's
  streams at seed e (encoder_init, train_bags, validation bags; an intended pairing); every split-e sample embedded with
  the shared 10 x 500-cell test bags, and half B's simulated bulk profiles as one-profile inputs;
- read-out: ridge (readout.run) on the simulated labels, fit on split e's training side with stage 03's encoder_split_e
  folds, r = 0-9; within single cell = half B's rows (training-side scaler), simulated bulk = half B's summed profiles
  (their own scaler). Rows: pseudobulk, mean_log_cells (its bulk input is the summed profile as a one-cell set) and the
  set encoders kept after triage (config/simulation.py). AUROC per (e, r) on half B (scoring-unit means). The success
  criterion, its readings and the by-construction values (simulation.criterion) go to criterion.json.
Simulated counts are never stored: every process regenerates them deterministically (seeds: `simulation`, unit-keyed;
config/simulation.py). In the structure-only scenario every sample's pseudobulk is asserted bit-identical to stage 02's.

Outputs, in <outputs>/12_simulation/crohns/all_0/:
  planted_genes.csv, simulated_labels.csv, calibration.csv, half_split.csv     the design (main environment's process)
  <scenario>/<encoder>/record_s<e>.json       the TrainingRecord
  <scenario>/<encoder>/embeddings/s<e>.h5ad   obs: side (train / half_a / half_b / simulated_bulk), sample_id; X float32
  simulation.parquet                          simulation kind: per row key, encoder seed e, read-out seed r and scenario
  criterion.json                              Delta per set encoder and mean_log_cells, pass flags, readings, routes

  python pipeline/12_simulation.py --pair PAIR [--scrat | --readout] [--scenario S --encoder M --seed E]
  (other pairs: nothing to do; the main environment trains CytoSet and Set Transformer and writes the design, --scrat
  trains ScRAT in its environment, --readout reads out and writes the criterion after both)
"""
import argparse
import dataclasses
import json
import runpy

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk import codehash, determinism, folds, harness, manifest, paths, preprocess, properties, protocol, readout, schemas
from sc2bulk import seeds, simulation, statistics, training_inputs
from sc2bulk.config import settings, smoke
from sc2bulk.config import simulation as SIM
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIR_ORDER

STAGE = "12_simulation"
TRAIN = runpy.run_path(str(settings.REPO / "pipeline" / "06_train.py"), run_name="stage06")   # its helpers, not its main
DEVICE = TRAIN["DEVICE"]
N_READOUT = smoke.N_READOUT_SEEDS if settings.SMOKE else settings.N_READOUT_SEEDS
ENVIRONMENT_ENCODERS = {"main": ("cytoset", "set_transformer"), "scrat": ("scrat",)}


@dataclasses.dataclass
class Source:
    """The unit's real stored cells: counts on the shared genes (CSR), the panel, per-cell ids, samples, types and total
    counts over all genes, and the samples' 0/1 group and holdout group (index sample_id)."""
    counts: object
    genes: list
    panel: list
    cell_id: np.ndarray
    sample_id: np.ndarray
    cell_type: np.ndarray
    total_counts: np.ndarray
    samples: pd.DataFrame


def load(pair, unit) -> Source:
    data = folds.training_cells(pair, *unit)
    genes = training_inputs.shared(pair)
    return Source(counts=sp.csr_matrix(data[:, genes].X), genes=genes, panel=training_inputs.panel(pair, unit),
                  cell_id=data.obs_names.astype(str).to_numpy(), sample_id=data.obs["sample_id"].astype(str).to_numpy(),
                  cell_type=data.obs["cell_type"].astype(str).to_numpy(), total_counts=data.obs["total_counts"].to_numpy(dtype=float),
                  samples=folds.training_samples(pair, *unit).set_index("sample_id")[["group", "holdout_group"]])


def design(pair, unit, source: Source) -> dict:
    """Everything fixed before any training: the planted genes, the simulated labels, each sample's G block (cells x |G|
    counts) and library size outside G, the calibration and the halves of each split's validation side."""
    columns = pd.Index(source.genes)
    panel_counts = source.counts[:, columns.get_indexer(source.panel)]
    detected = pd.Series(np.asarray((panel_counts > 0).sum(axis=0)).ravel() / panel_counts.shape[0], index=source.panel)
    planted = simulation.planted_genes(source.panel, detected, seeds.derived("simulation", pair, SIM.GENES_INDEX, *unit))
    groups = source.samples.groupby("holdout_group")["group"].first()
    stratum = groups.astype(str)
    if settings.SMOKE:   # too few validation groups to halve: also stratified by split 0's side (config/simulation.py)
        side = pd.Series(training_inputs.encoder_sides(pair, unit, 0)).groupby(source.samples["holdout_group"]).first()
        stratum = stratum + "/" + side.loc[stratum.index]
    sim_group = simulation.simulated_labels(stratum, seeds.derived("simulation", pair, SIM.LABELS_INDEX, *unit))
    label = source.samples["holdout_group"].map(sim_group)
    G = source.counts[:, columns.get_indexer(planted.index)].toarray().astype(np.int64)
    d = {"planted": planted, "plus": (planted == "plus").to_numpy(), "sim_group": sim_group, "label": label, "G": G,
         "weight": source.total_counts - G.sum(axis=1), "shared_totals": np.asarray(source.counts.sum(axis=1), dtype=np.float64).ravel(),
         "rows": {s: np.flatnonzero(source.sample_id == s) for s in sorted(source.samples.index)},
         "pseudobulk": training_inputs.pseudobulk_log(pair, unit)}
    d["calibration"], d["level"] = calibration(pair, unit, d)
    d["halves"] = {e: halves(pair, unit, e, source, sim_group) for e in SIM.SPLITS}
    return d


def halves(pair, unit, e, source, sim_group) -> tuple:
    """(half A sample ids, half B sample ids) of split e's validation side (the smoke run: both the whole side)."""
    sides = training_inputs.encoder_sides(pair, unit, e)
    validation = sorted(s for s, side in sides.items() if side == "validation")
    if settings.SMOKE:
        return validation, validation
    groups = sorted(set(source.samples.loc[validation, "holdout_group"]))
    half = simulation.half_split(sim_group.loc[groups], seeds.derived("simulation", pair, SIM.HALF_OFFSET + e, *unit), SIM.MIN_HALF_GROUPS)
    of = source.samples.loc[validation, "holdout_group"].map(half)
    return sorted(of.index[of == "A"]), sorted(of.index[of == "B"])


def simulated_block(pair, unit, scenario, level, d) -> np.ndarray:
    """Every cell's simulated G counts at a grid level (cells x |G|), sample by sample, from the scenario's draws."""
    base = seeds.derived("simulation", pair, SIM.DRAWS_INDEX[scenario], *unit)
    out = d["G"].copy()
    for s, rows in d["rows"].items():
        block, label, key = d["G"][rows], int(d["label"][s]), preprocess.sample_seed(pair, s) % 2**32
        if scenario == "mean_shift":
            u = simulation.uniforms(block, np.random.default_rng([base, key, SIM.ORDER_ROLE]))
            out[rows] = simulation.mean_shift(block, ~d["plus"] if label == 1 else d["plus"], SIM.RHO_GRID[level], u)
        else:
            order = np.random.default_rng([base, key, SIM.ORDER_ROLE]).permutation(len(rows))
            subset = simulation.subset_of(order, SIM.Q_GRID[level]) if label == 1 else np.arange(len(rows))
            role = SIM.LABEL1_ROLE + level if label == 1 else SIM.LABEL0_ROLE
            out[rows] = simulation.structure_only(block, d["weight"][rows], subset, np.random.default_rng([base, key, role]))
    return out


def oracle(scenario, G_new, d) -> np.ndarray:
    """The scenario's oracle score per sample (d["rows"] order): no fitting, knows G and the direction."""
    cell_totals = d["shared_totals"] - d["G"].sum(axis=1) + G_new.sum(axis=1)
    if scenario == "mean_shift":
        summed = np.vstack([G_new[rows].sum(axis=0) for rows in d["rows"].values()])
        return simulation.oracle_mean(summed, np.array([cell_totals[rows].sum() for rows in d["rows"].values()]), d["plus"])
    return np.array([simulation.oracle_structure(G_new[rows], cell_totals[rows]) for rows in d["rows"].values()])


def calibration(pair, unit, d) -> tuple:
    """(table: scenario, level, oracle_auroc, chosen; {scenario: chosen level index}): every grid level's oracle AUROC
    over all the unit's samples, the weakest reaching ORACLE_TARGET chosen (else the strongest). Reads no model."""
    labels = np.array([d["label"][s] for s in d["rows"]])
    rows, chosen = [], {}
    for scenario in SIM.SCENARIOS:
        aurocs = [statistics.auroc(oracle(scenario, simulated_block(pair, unit, scenario, i, d), d), labels)
                  for i in range(len(SIM.GRIDS[scenario]))]
        chosen[scenario], reached = simulation.calibrate(aurocs)
        rows += [{"scenario": scenario, "level": level, "oracle_auroc": a, "chosen": i == chosen[scenario], "target_reached": reached}
                 for i, (level, a) in enumerate(zip(SIM.GRIDS[scenario], aurocs))]
    return pd.DataFrame(rows), chosen


def simulated(pair, unit, scenario, source, d) -> tuple:
    """The scenario at its chosen level: (Cells on the panel, log-CP10k over the shared genes with the recomputed
    totals, simulated labels as the samples' group; sample -> its simulated log-CP10k pseudobulk = its simulated bulk
    profile)."""
    G_new = simulated_block(pair, unit, scenario, d["level"][scenario], d)
    columns = pd.Index(source.genes)
    panel_of_g = pd.Index(source.panel).get_indexer(d["planted"].index)
    keep = np.ones(len(source.panel))
    keep[panel_of_g] = 0.0
    place = sp.csr_matrix((np.ones(len(panel_of_g)), (np.arange(len(panel_of_g)), panel_of_g)), shape=(len(panel_of_g), len(source.panel)))
    counts = sp.csr_matrix(source.counts[:, columns.get_indexer(source.panel)] @ sp.diags(keep)) + sp.csr_matrix(G_new) @ place
    counts.eliminate_zeros()
    X = simulation.log_normalize_rows(counts, d["shared_totals"] - d["G"].sum(axis=1) + G_new.sum(axis=1))
    ids = list(d["rows"])
    summed = np.vstack([np.asarray(source.counts[rows].sum(axis=0)).ravel() for rows in d["rows"].values()]).astype(np.float64)
    summed[:, columns.get_indexer(d["planted"].index)] = np.vstack([G_new[rows].sum(axis=0) for rows in d["rows"].values()])
    profile = preprocess.log_normalize(summed)[:, columns.get_indexer(source.panel)].toarray()
    pseudobulk = dict(zip(ids, profile))
    if scenario == "structure_only":   # per-gene sums preserved: stage 02's pseudobulk, bit for bit
        assert all(np.array_equal(pseudobulk[s], d["pseudobulk"][s]) for s in ids), "structure_only changed a sample's pseudobulk"
    cells = protocol.Cells(X=X, genes=source.panel, cell_id=source.cell_id, sample_id=source.sample_id, cell_type=source.cell_type,
                           library_size=source.total_counts - d["G"].sum(axis=1) + G_new.sum(axis=1),
                           samples=source.samples.assign(group=d["label"]))
    return cells, pseudobulk


def train_job(pair, unit, scenario, encoder, e, cells, pseudobulk, d, record, cost, out):
    """One training on split e's simulated training side (half A early-stops), then the embeddings."""
    config = {**TRAIN["frozen"](pair, unit, encoder), "encoder": encoder}
    assert config["ncell"] == T.FROZEN_NCELL and not config.get("age_similarity_loss") and not config.get("mixup")
    config["micro_batch"] = harness.micro_batch_for(encoder, config)
    sides = training_inputs.encoder_sides(pair, unit, e)
    train_ids = sorted(s for s, side in sides.items() if side == "train")
    a_ids, b_ids = d["halves"][e]
    train, half_a, half_b = (training_inputs.subset(cells, ids) for ids in (train_ids, a_ids, b_ids))
    inputs = harness.Inputs(config, train, pseudobulk)
    assert inputs.pseudobulk_log is pseudobulk   # moments and gene scaler from the simulated training side (tie-break Q16)
    fits = harness.fits_on_device(train, half_a, half_b)
    residents = tuple(harness.Resident(c, inputs, DEVICE, fits) for c in (train, half_a, half_b))
    build = lambda: TRAIN["network"](encoder, config, harness.input_width(config, len(cells.genes)))
    net, _, trained, val_bags, _ = harness.train(build, config, train, half_a, inputs, device=DEVICE, residents=residents[:2],
                                                 **TRAIN["stream"](pair, unit, "A", encoder, e))
    if not settings.SMOKE:   # half B reaches no training, early stopping or checkpoint
        assert not set(b_ids) & (set(train.samples.index) | set(half_a.samples.index) | {s for s, _ in val_bags})
    ids = sorted(set(train_ids) | set(a_ids) | set(b_ids))
    embedding = pd.DataFrame(harness.embed_samples(net, cells, inputs, ids, T.FROZEN_NCELL, pair, unit, DEVICE, residents), index=ids)
    bulk = harness.embed_profiles(net, TRAIN["profiles"](b_ids, pseudobulk, cells.genes), inputs, DEVICE)
    blocks = [(side, part, part, "primary", embedding.loc[part].to_numpy()) for side, part in (("train", train_ids), ("half_a", a_ids), ("half_b", b_ids))]
    blocks.append(("simulated_bulk", b_ids, b_ids, "primary", bulk))
    TRAIN["write_json"](out / f"record_s{e}.json", dataclasses.asdict(trained), record, cost)
    TRAIN["write_embeddings"](out / "embeddings" / f"s{e}.h5ad", blocks, record, cost)   # last: its manifest marks the job done


def readout_folds(pair, unit, ids, r, context) -> np.ndarray:
    """Stage 03's read-out folds of the given samples (as stage 08 reads them)."""
    table = pd.read_csv(settings.OUTPUTS / "03_splits" / pair / "readout_folds.csv", dtype={"sample_id": str})
    rows = table[(table["fold_set"] == unit[0]) & (table["fold"] == unit[1]) & (table["context"] == context)
                 & (table["readout_index"] == r)].set_index("sample_id")
    assert set(rows.index) == set(ids), f"{pair} {unit}: read-out folds don't cover the samples"
    return rows.loc[list(ids), "readout_fold"].to_numpy()


def row_keys(pair, unit, key) -> dict:
    if key == "mean_log_cells":
        return {"pair": pair, "fold_set": unit[0], "fold": unit[1], "row_kind": "reference", "key": key, "model": "none",
                "input": "cells", "n_cells": 0, "pooling": "native"}
    model = "pseudobulk" if key == "pseudobulk" else key.removesuffix("_cells")
    return {"pair": pair, "fold_set": unit[0], "fold": unit[1], "row_kind": "configuration", "key": key, "model": model,
            "input": "pseudobulk" if key == "pseudobulk" else "cells", "n_cells": 0 if key == "pseudobulk" else T.FROZEN_NCELL,
            "pooling": "native"}


def encoder_rows(path, train_ids, b_ids) -> tuple:
    """A training's embeddings: (training side, half B, half B's simulated bulk) rows."""
    data = anndata.read_h5ad(path)
    X = pd.DataFrame(np.asarray(data.X, dtype=np.float64))
    pick = lambda side, ids: X[(data.obs["side"] == side).to_numpy()].set_axis(data.obs.loc[data.obs["side"] == side, "sample_id"]).loc[ids].to_numpy()
    return pick("train", train_ids), pick("half_b", b_ids), pick("simulated_bulk", b_ids)


def readout_rows(pair, unit, source, d) -> pd.DataFrame:
    """Every row's within-single-cell and simulated-bulk AUROC per (e, r, scenario) on half B."""
    scoring_unit = TRAIN["scoring_units"](pair)
    rows = []
    for scenario in SIM.SCENARIOS:
        cells, pseudobulk = simulated(pair, unit, scenario, source, d)
        ids = list(d["rows"])
        mean_log = dict(zip(ids, properties.mean_log_cells(cells.X, cells.sample_id, ids)))
        del cells
        strength = SIM.GRIDS[scenario][d["level"][scenario]]
        for e in SIM.SPLITS:
            sides = training_inputs.encoder_sides(pair, unit, e)
            train_ids = sorted(s for s, side in sides.items() if side == "train")
            b_ids = d["halves"][e][1]
            stack = lambda of, part: np.vstack([of[s] for s in part])
            features = {"pseudobulk": (stack(pseudobulk, train_ids), stack(pseudobulk, b_ids), stack(pseudobulk, b_ids), None),
                        "mean_log_cells": (stack(mean_log, train_ids), stack(mean_log, b_ids), stack(pseudobulk, b_ids), None)}
            for encoder in SIM.ENCODERS:
                out = output_dir(pair, unit, scenario, encoder)
                trained = json.loads((out / f"record_s{e}.json").read_text())["did_train"]
                features[f"{encoder}_cells"] = (*encoder_rows(out / "embeddings" / f"s{e}.h5ad", train_ids, b_ids), trained)
            y = d["label"].loc[train_ids].to_numpy()
            label, units = d["label"].loc[b_ids].to_numpy(), [scoring_unit[s] for s in b_ids]
            cv = [readout_folds(pair, unit, train_ids, r, f"encoder_split_{e}") for r in range(N_READOUT)]
            for key, (X_train, X_scored, X_bulk, trained) in features.items():
                for r in range(N_READOUT):
                    fit = readout.run(X_train, y, cv[r], X_bulk, X_scored=X_scored)
                    rows.append({**row_keys(pair, unit, key), "encoder_seed": e, "readout": "ridge", "readout_seed": r, "scenario": scenario,
                                 "within_sc_auroc": properties.unit_auroc(fit["scored_score"], label, units),
                                 "simulated_bulk_auroc": properties.unit_auroc(fit["score"], label, units),
                                 "strength": strength, "n_scored": len(b_ids), "did_train": trained})
    return pd.DataFrame(rows)


def output_dir(pair, unit, scenario, encoder):
    return paths.output(STAGE, pair, scenario, unit) / encoder


def design_inputs(pair, unit) -> list:
    return [*folds.inputs(pair), settings.OUTPUTS / "03_splits" / pair / "splits.csv",
            paths.output("04_panel", pair, "panel.csv", unit), paths.output("04_panel", pair, "shared_genes.csv")]


def base_record(pair, unit, sources) -> dict:
    return {"stage": STAGE, "pair": pair, "unit": unit, "smoke": settings.SMOKE,
            "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair, codehash.extra(STAGE, pair)),
            "inputs": manifest.input_hashes(sources)}


def write_design(pair, unit, source, d):
    sources = design_inputs(pair, unit)
    record = {**base_record(pair, unit, sources), "part": "design"}
    halves_table = pd.DataFrame([{"encoder_seed": e, "sample_id": s, "half": half} for e, (a, b) in d["halves"].items()
                                 for half, part in (("A", a), ("B", b)) for s in part])
    tables = {"planted_genes.csv": d["planted"].rename_axis("gene").rename("half").reset_index(),
              "simulated_labels.csv": source.samples[["holdout_group"]].assign(sim_group=d["label"]).rename_axis("sample_id").reset_index(),
              "calibration.csv": d["calibration"], "half_split.csv": halves_table}
    targets = [paths.output(STAGE, pair, name, unit) for name in tables]
    if manifest.is_done(targets, record):
        return
    cost = manifest.begin()
    for path, table in zip(targets, tables.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(manifest.temporary(path), index=False)
        manifest.finish(path, record, cost, inputs=sources)


def train_all(pair, unit, source, d, environment, only):
    """Every training job of an environment (or the one given), scenario by scenario, skipping finished ones."""
    for scenario in SIM.SCENARIOS:
        todo = []
        for encoder in [m for m in ENVIRONMENT_ENCODERS[environment] if m in SIM.ENCODERS]:
            for e in SIM.SPLITS:
                if only is not None and only != (scenario, encoder, e):
                    continue
                sources = [*design_inputs(pair, unit), TRAIN["selection_path"](pair, unit, encoder)]
                record = {**base_record(pair, unit, sources), "scenario": scenario, "encoder": encoder, "seed": e}
                out = output_dir(pair, unit, scenario, encoder)
                if manifest.is_done([out / "embeddings" / f"s{e}.h5ad"], record):
                    print(f"{scenario} {encoder} s{e}: done", flush=True)
                    continue
                todo.append((encoder, e, record, out))
        if not todo:
            continue
        cells, pseudobulk = simulated(pair, unit, scenario, source, d)
        for encoder, e, record, out in todo:
            out.mkdir(parents=True, exist_ok=True)
            train_job(pair, unit, scenario, encoder, e, cells, pseudobulk, d, record, manifest.begin(device=DEVICE), out)
            print(f"{scenario} {encoder} s{e}: trained and embedded", flush=True)
        del cells


def readout_all(pair, unit, source, d):
    """The read-outs, simulation.parquet and criterion.json, after every training."""
    trainings = [output_dir(pair, unit, s, m) / name for s in SIM.SCENARIOS for m in SIM.ENCODERS for e in SIM.SPLITS
                 for name in (f"record_s{e}.json", f"embeddings/s{e}.h5ad")]
    sources = [*design_inputs(pair, unit), settings.OUTPUTS / "03_splits" / pair / "readout_folds.csv",
               *(TRAIN["selection_path"](pair, unit, m) for m in SIM.ENCODERS), *trainings]
    record = {**base_record(pair, unit, sources), "part": "readout"}
    table_path, criterion_path = paths.output(STAGE, pair, "simulation.parquet", unit), paths.output(STAGE, pair, "criterion.json", unit)
    if manifest.is_done([table_path, criterion_path], record):
        print(f"{pair}: read-outs done", flush=True)
        return
    cost = manifest.begin()
    table = readout_rows(pair, unit, source, d)
    schemas.write("simulation", schemas.with_neutral_keys(table), manifest.temporary(table_path))
    manifest.finish(table_path, record, cost, inputs=sources)
    routes = {m: TRAIN["frozen"](pair, unit, m)["cell_normalization"] for m in SIM.ENCODERS}
    passed = table[table["key"].isin([f"{m}_cells" for m in SIM.ENCODERS])].drop_duplicates(["key", "scenario", "encoder_seed"])
    result = {**simulation.criterion(table, routes),
              "strength": {s: SIM.GRIDS[s][d["level"][s]] for s in SIM.SCENARIOS},
              "oracle_target_reached": d["calibration"].groupby("scenario")["target_reached"].first().to_dict(),
              "did_train_passed": {f"{k} {s}": f"{int(g['did_train'].sum())}/{len(g)}" for (k, s), g in passed.groupby(["key", "scenario"])}}
    manifest.temporary(criterion_path).write_text(json.dumps(result, indent=1, default=bool))
    manifest.finish(criterion_path, record, cost, inputs=sources)
    print(f"{pair}: positive control {'passed' if result['control_passes'] else 'failed'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER, required=True)
    parser.add_argument("--scrat", action="store_true", help="ScRAT's trainings (its environment)")
    parser.add_argument("--readout", action="store_true", help="the read-outs and the criterion, after every training")
    parser.add_argument("--scenario", choices=SIM.SCENARIOS)
    parser.add_argument("--encoder", choices=SIM.ENCODERS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.pair != SIM.PAIR:
        print(f"{args.pair}: the simulation runs on {SIM.PAIR} only", flush=True)
        return
    determinism.configure()
    one_job = (args.scenario, args.encoder, args.seed)
    assert all(a is None for a in one_job) or all(a is not None for a in one_job), "one job takes --scenario, --encoder, --seed"
    pair, unit = SIM.PAIR, SIM.UNIT
    source = load(pair, unit)
    d = design(pair, unit, source)
    if args.readout:
        readout_all(pair, unit, source, d)
        return
    environment = "scrat" if args.scrat else "main"
    if environment == "main":
        write_design(pair, unit, source, d)
    train_all(pair, unit, source, d, environment, None if args.scenario is None else one_job)


if __name__ == "__main__":
    main()
