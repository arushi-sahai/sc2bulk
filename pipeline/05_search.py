"""Stage 05: the configuration search. Label-blind: selection reads only single-cell validation BCE; every candidate's bulk
read-out is stage 08's.

Phases (one job each; the scheduler runs the train jobs longest first, one per GPU):
  train    one (unit, encoder, search seed i): the base and every candidate, grouped by input transform
           so each group builds its inputs and device copies once. Per training: the
           checkpoint (state, configuration, TrainingRecord, PCA basis), the embeddings (training samples:
           10 test bags of the frozen size, averaged; scored bulk: one profile each), a candidates row.
  select   one (unit, encoder): eligibility, group winners, merge conflicts, the merged
           configuration trained at the k seeds when >= 2 groups win (then adopted or not), the frozen
           configuration; the per-sample sensitivity row when the frozen one uses per_sample.
  modal    human blood, aging: the configuration selected in the most crossfit folds (ties: the base,
           else fold 0's).

Seeds: search seed i uses stage 03's search_split i, search_init i and search_sampling i; the validation
bags use VALIDATION_BAGS_PURPOSE at SEARCH_INDEX_OFFSET + i (no stream shared with stage 06). With one
search seed (the smoke run) adoption is skipped and the frozen configuration is the base.

Outputs, in <outputs>/05_search/<pair>/<unit>/<encoder>/:
  candidates_s<i>.parquet          candidates kind: one row per configuration trained at seed i
  checkpoints/<name>_s<i>.pt       state dict, configuration, TrainingRecord, PCA basis (if any)
  embeddings/<name>_s<i>.h5ad      obs: side, sample_id, scoring_unit_id; X float32 (label-free)
  selection.json                   eligibility, winners, conflicts, the merge, the frozen configuration
  (<outputs>/05_search/soundlife/modal_<encoder>.json for the modal fold selection)

  python pipeline/05_search.py --phase train --pair PAIR --set UNIT --encoder ENC --seed I
  python pipeline/05_search.py --phase select --pair PAIR --set UNIT --encoder ENC
  python pipeline/05_search.py --phase modal --pair soundlife --encoder ENC
  python pipeline/05_search.py --pair PAIR [--scrat]   every job of the pair in order (train, select, modal):
                                                        CytoSet and Set Transformer, or ScRAT (its environment)
"""
import argparse
import dataclasses
import json
import math
import os

import anndata
import numpy as np
import pandas as pd
import torch

from sc2bulk import determinism, folds, harness, manifest, paths, schemas, search, seeds, training_inputs
from sc2bulk.config import search_space, settings, smoke
from sc2bulk.config import training as T

STAGE = "05_search"
K = smoke.N_SEARCH_SEEDS if settings.SMOKE else settings.N_SEARCH_SEEDS
INPUT_KEYS = ("cell_normalization", "pseudobulk_gene_scaler", "population_distance_feature", "pca")


def network(encoder, config, in_dim):
    if encoder == "cytoset":
        from sc2bulk.encoders import cytoset
        return cytoset.network(config, in_dim)
    if encoder == "set_transformer":
        from sc2bulk.encoders import set_transformer
        return set_transformer.network(config, in_dim)
    from sc2bulk.encoders import scrat
    return scrat.network(config, in_dim, os.environ["SC2BULK_SCRAT"])


def trials(encoder, pair) -> dict:
    """name -> (group, configuration): the base and each candidate (one change from the base; ScRAT's
    norm_per_sample: two)."""
    base = search_space.base(encoder)
    out = {"base": (None, base)}
    for group, candidates in search_space.candidates(encoder, pair).items():
        out.update({name: (group, {**base, **change}) for name, change in candidates.items()})
    return out


def finished(config) -> dict:
    """A configuration as trained: the micro-batch from the pinned cap."""
    return {**config, "micro_batch": harness.micro_batch_for(config["encoder"], config)}


def unit_dir(pair, unit, encoder):
    return paths.output(STAGE, pair, encoder, unit)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_seed(pair, unit, encoder, i, configs: dict, record: dict, cost: dict):
    """Train every configuration at search seed i, grouped by input transform; its checkpoint and
    embeddings go through manifest with the seed's record."""
    device = DEVICE
    cells = training_inputs.unit_cells(pair, unit)
    sides = training_inputs.split_sides(pair, unit, "search_split", i)
    train, val = training_inputs.split_cells(cells, sides)
    pseudobulk = training_inputs.pseudobulk_log(pair, unit)
    bulk_table = training_inputs.scored_bulk(pair, unit)
    bulk = training_inputs.bulk_profiles(pair, unit, bulk_table["sample_id"])
    stream = dict(init_seed=seeds.derived("search_init", pair, i, *unit), bag_seed=seeds.derived("search_sampling", pair, i, *unit),
                  val_seed=seeds.derived(T.VALIDATION_BAGS_PURPOSE, pair, T.SEARCH_INDEX_OFFSET + i, *unit))
    out = unit_dir(pair, unit, encoder)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "embeddings").mkdir(parents=True, exist_ok=True)
    rows = []
    by_inputs = {}
    for name, (group, config) in configs.items():
        by_inputs.setdefault(tuple(config.get(k) for k in INPUT_KEYS), []).append((name, group, config))
    for members in by_inputs.values():
        inputs = harness.Inputs(members[0][2], train, pseudobulk)
        fits = harness.fits_on_device(train, val)
        residents = harness.Resident(train, inputs, device, fits), harness.Resident(val, inputs, device, fits)
        for name, group, config in members:
            config = finished({**config, "encoder": encoder})
            ages = training_inputs.training_ages(pair, unit, train.samples.index) if config.get("age_similarity_loss") else None
            net, state, trained, _, _ = harness.train(lambda: network(encoder, config, harness.input_width(config, len(cells.genes))), config, train, val, inputs,
                                                     device=device, ages=ages, residents=residents, **stream)
            samples = sorted(set(train.samples.index) | set(val.samples.index))
            emb = harness.embed_samples(net, cells, inputs, samples, config["ncell"], pair, unit, device, residents)
            emb_bulk = harness.embed_profiles(net, bulk, inputs, device)
            obs = pd.DataFrame({"side": ["single_cell_train" if sides[s] == "train" else "single_cell_validation" for s in samples]
                                + ["bulk"] * len(bulk_table),
                                "sample_id": samples + list(bulk_table["sample_id"]),
                                "scoring_unit_id": samples + list(bulk_table["scoring_unit_id"])})
            stem = f"{name}_s{i}"
            embedding, checkpoint = out / "embeddings" / f"{stem}.h5ad", out / "checkpoints" / f"{stem}.pt"
            anndata.AnnData(np.vstack([emb, emb_bulk]).astype(np.float32), obs=obs.set_index(obs.index.astype(str))).write_h5ad(
                manifest.temporary(embedding))
            basis = {"pca_mean": inputs.pca_mean, "components": inputs.components} if inputs.pca else {}
            torch.save({"state": state, "config": config, "record": dataclasses.asdict(trained), **basis},
                       manifest.temporary(checkpoint))   # search checkpoints aren't re-embedded: stage 08 reads the embeddings
            for artifact in (embedding, checkpoint):
                manifest.finish(artifact, {**record, "trial": name}, cost)
            rows.append({"key": name, "change_group": group or "base", "change_value": json.dumps(
                {k: v for k, v in config.items() if search_space.base(encoder).get(k) != v and k not in ("encoder", "micro_batch")},
                default=str), "validation_bce": trained.validation_curve[trained.stop_epoch - 1], "did_train": trained.did_train,
                "did_train_reason": trained.did_train_reason, "d": trained.margin_d, "d_star": trained.margin_d_star,
                "t_star": trained.margin_temperature, "p": trained.margin_p, "testable": trained.testable,
                "epochs": trained.epochs, "kept_epoch": trained.stop_epoch, "seconds": trained.seconds,
                "validation_curve": json.dumps(trained.validation_curve),
                "peak_gpu_bytes": trained.peak_gpu_bytes})
        del residents
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows


def write_candidates(pair, unit, encoder, i, rows, record, cost, tag=""):
    table = pd.DataFrame(rows).assign(pair=pair, fold_set=unit[0], fold=unit[1], row_kind="candidate", model=encoder,
                                      input="cells", encoder_seed=i, n_cells=T.FROZEN_NCELL, pooling="native",
                                      selected=False, merged=bool(tag))
    path = unit_dir(pair, unit, encoder) / f"candidates{tag}_s{i}.parquet"
    schemas.write("candidates", schemas.with_neutral_keys(table), manifest.temporary(path))
    manifest.finish(path, record, cost)
    return path


def per_seed(table, key) -> pd.DataFrame:
    """A configuration's rows, one per search seed 0..K-1 in order (asserted: the pairing is by seed)."""
    rows = table[table["key"] == key].sort_values("encoder_seed")
    assert list(rows["encoder_seed"]) == list(range(K)), f"{key}: seeds {list(rows['encoder_seed'])}"
    return rows


def select(pair, unit, encoder, record, cost):
    """Eligibility, winners and the frozen configuration, with each
    rule's numbers (S4 Table)."""
    out = unit_dir(pair, unit, encoder)
    table = pd.concat([schemas.read("candidates", out / f"candidates_s{i}.parquet") for i in range(K)])
    testable = (pair, *unit) not in T.UNTESTABLE_UNITS and not settings.SMOKE   # smoke splits have < MIN_TESTABLE labellings
    assert (table["testable"] == testable).all(), "testability differs between config and the harness"
    all_trials = trials(encoder, pair)
    rows = {k: per_seed(table, k) for k in all_trials}
    eligible = {k: search.eligible(r["d_star"], r["p"], testable) for k, r in rows.items()}
    bce = {k: r["validation_bce"].to_numpy() for k, r in rows.items()}
    base = all_trials["base"][1]
    changes = {name: {k: v for k, v in config.items() if base.get(k) != v} for name, (_, config) in all_trials.items()}
    floor = T.MARGIN_FLOOR * math.log(2)
    decision = {"search_seeds": K, "candidates": {
        k: {"eligible": bool(eligible[k]), "mean_bce": float(bce[k].mean()), "seeds_above_floor": int((r["d_star"] > floor).sum()),
            "seeds_p_le_alpha": int((r["p"] <= T.NULL_ALPHA).sum()) if testable else None,
            **({"paired_improvement": dict(zip(("mean", "sd"), search.improvement(bce["base"], bce[k])))} if K >= 2 and k != "base" else {})}
        for k, r in rows.items()}}
    decision["identical_to_base"] = sorted(   # the run-level no-op check: byte-identical curves on every seed
        k for k, r in rows.items() if k != "base" and list(r["validation_curve"]) == list(rows["base"]["validation_curve"]))
    if K < 2:   # the smoke run: training only, no adoption (search.improvement needs 2 seeds)
        decision.update(frozen=base, how="base (one search seed)")
    else:
        groups = {}
        for name, (group, _) in all_trials.items():
            if group:
                groups.setdefault(group, {})[name] = bce[name]
        decision["frozen"], decision["how"], decision["winners"] = choose(pair, unit, encoder, groups, eligible, bce, changes, base,
                                                                          testable, (), "merged", record, cost)
        if decision["frozen"].get("cell_normalization") == "per_sample":
            decision["per_sample_route"], how, decision["per_sample_route_winners"] = choose(
                pair, unit, encoder, groups, eligible, bce, changes, base, testable, ("norm_per_sample",), "merged_no_per_sample",
                record, cost)
            decision["per_sample_route_how"] = how
    path = out / "selection.json"
    manifest.temporary(path).write_text(json.dumps(decision, indent=1, default=str))
    manifest.finish(path, record, cost)
    return decision


def choose(pair, unit, encoder, groups, eligible, bce, changes, base, testable, exclude, name, record, cost):
    """Winners -> conflicts -> the merged configuration (trained at the k seeds if >= 2 winners)."""
    winners = search.winners(bce["base"], groups, {k for k, ok in eligible.items() if ok}, exclude=exclude)
    gains = {w: search.improvement(bce["base"], bce[w])[0] for w in winners.values()}
    kept = search.resolve(winners, changes, gains)
    merged_result = None
    if len(kept) >= 2:
        merged = search.merged_config(base, kept, changes)
        rows = []
        for i in range(K):
            seed_record = {**record, "phase": "merge", "seed": i, "merged": merged}
            rows.append(train_seed(pair, unit, encoder, i, {name: ("merged", merged)}, seed_record, cost)[0])
            write_candidates(pair, unit, encoder, i, rows[-1:], seed_record, cost, tag=f"_{name}")
        merged_result = (bool(search.eligible([r["d_star"] for r in rows], [r["p"] for r in rows], testable)),
                         bool(search.beats(bce["base"], np.array([r["validation_bce"] for r in rows]))))
    frozen, how = search.frozen(base, kept, changes, gains, merged_result)
    return frozen, how, {"winners": winners, "kept": kept, "gains": gains, "merged_result": merged_result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "select", "modal"))
    parser.add_argument("--pair", required=True)
    parser.add_argument("--set", help="the unit (e.g. crossfit_0)")
    parser.add_argument("--encoder", choices=("cytoset", "set_transformer", "scrat"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scrat", action="store_true", help="with --pair alone: ScRAT's jobs (its environment)")
    args = parser.parse_args()
    determinism.configure()
    if args.phase is None:   # every job of the pair, in order (run_all; the GPU scheduler runs them in parallel)
        assert args.set is None and args.encoder is None and args.seed is None, "--set/--encoder/--seed need --phase"
        for encoder in (("scrat",) if args.scrat else ("cytoset", "set_transformer")):
            units = folds.units(args.pair)
            for unit in units:
                for i in range(K):
                    job("train", args.pair, unit, encoder, i)
                job("select", args.pair, unit, encoder, None)
            if any(u[0] == "crossfit" for u in units):
                job("modal", args.pair, None, encoder, None)
        return
    unit = None if args.phase == "modal" else next(u for u in folds.units(args.pair) if folds.unit_name(*u) == args.set)
    job(args.phase, args.pair, unit, args.encoder, args.seed)


def job(phase, pair, unit, encoder, seed):
    if phase == "modal":
        units = [u for u in folds.units(pair) if u[0] == "crossfit"]
        sources = [unit_dir(pair, u, encoder) / "selection.json" for u in units]
        record = {"stage": STAGE, "pair": pair, "encoder": encoder, "phase": "modal",
                  "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair),
                  "inputs": manifest.input_hashes(sources)}
        path = paths.output(STAGE, pair, f"modal_{encoder}.json")
        if manifest.is_done([path], record):
            return
        cost = manifest.begin()
        selections = [json.loads(p.read_text())["frozen"] for p in sources]
        manifest.temporary(path).write_text(json.dumps(search.modal(selections, search_space.base(encoder)), indent=1, default=str))
        manifest.finish(path, record, cost, inputs=sources)
        return
    sources = [*folds.inputs(pair), settings.OUTPUTS / "03_splits" / pair / "splits.csv", paths.output("04_panel", pair, "panel.csv", unit),
               paths.output("04_panel", pair, "shared_genes.csv"), settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad",
               settings.OUTPUTS / "02_preprocess" / pair / "bulk_samples.csv", settings.OUTPUTS / "03_splits" / pair / "donor_folds.csv"]
    record = {"stage": STAGE, "pair": pair, "unit": unit, "encoder": encoder, "smoke": settings.SMOKE, "search_seeds": K,
              "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair),
              "inputs": manifest.input_hashes(sources)}
    if phase == "train":
        assert seed in range(K), f"search seeds are 0..{K - 1}"
        seed_record = {**record, "phase": "train", "seed": seed}
        path = unit_dir(pair, unit, encoder) / f"candidates_s{seed}.parquet"
        if manifest.is_done([path], seed_record):
            print(f"{pair} {folds.unit_name(*unit)} {encoder} s{seed}: done", flush=True)
            return
        cost = manifest.begin(device=DEVICE)
        rows = train_seed(pair, unit, encoder, seed, trials(encoder, pair), seed_record, cost)
        write_candidates(pair, unit, encoder, seed, rows, seed_record, cost)
        return
    candidates = [unit_dir(pair, unit, encoder) / f"candidates_s{i}.parquet" for i in range(K)]
    select_record = {**record, "phase": "select", "candidates": manifest.input_hashes(candidates)}
    if manifest.is_done([unit_dir(pair, unit, encoder) / "selection.json"], select_record):
        print(f"{pair} {folds.unit_name(*unit)} {encoder}: selection done", flush=True)
        return
    cost = manifest.begin(device=DEVICE)
    print(json.dumps(select(pair, unit, encoder, select_record, cost), indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
