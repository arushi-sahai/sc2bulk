"""Stage 05t: the timing test, v2.

Measurements for the budget decision (the scenario table) and the pre-run checks; nothing downstream
reads them, so they are written outside manifest, one JSON row per job (failed and killed jobs leave a
"started" row), each with the git state, environment, torch/CUDA, GPU and threads. Label-free.

Blocks (one job per invocation; the scheduler runs one job per GPU, longest first):
  base        a set encoder's base at FROZEN_NCELL, a search seed (A)
  candidate   one search candidate, capped at the burn-in + 2 epochs (B)
  sweep       Fig 4A bag size 1,000 or "all" (the 02b store, host gather), time-boxed by the scheduler (D)
  scvi        one scVI grid point (E)
  memory      a few optimizer steps per micro-batch size at the widest input (per_sample + distance
              feature, CytoSet) and the largest architecture: peak reserved memory per micro-batch (F)
  equality    device-resident vs host gather on real data, 200 bags per setting (pre-run check)
The own-recipe, label-permuted and MSE rows (blocks C, D) join with stage 06's code.

Per-epoch progress goes to <name>.epochs.jsonl as it happens, so a time-boxed job is projectable.

  python pipeline/05t_timing.py --block BLOCK --model MODEL --pair PAIR [--seed I] [--candidate NAME] [--ncell N|all]
"""
import argparse
import json
import os
import resource
import time

import numpy as np
import pandas as pd
import torch

from sc2bulk import determinism, folds, harness, manifest, paths, seeds, training_inputs
from sc2bulk.config import search_space, settings
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIR_ORDER

STAGE = "05t_timing"
# the memory-worst configurations per set encoder (block F): every width/depth candidate at its maximum,
# merged; Set Transformer also with SAB (full attention over the bag); CytoSet also with the age loss at
# micro-batch = batch (its accumulation is per batch). The cap pinned is the smallest that fits any of them.
CYTOSET_WIDEST = {"h_dim": 512, "nblock": 3, "pool": "attention", "out_pool": "attention",
                  "cell_normalization": "per_sample", "population_distance_feature": True}
WIDEST = {"cytoset": [CYTOSET_WIDEST, {**CYTOSET_WIDEST, "age_similarity_loss": True}],   # the age loss can't be split
          "set_transformer": [{"dim_hidden": 512, "num_heads": 8, "num_inds": 64, "encoder_layers": 3, "decoder_sab": True,
                               "cell_normalization": "per_sample"},
                              {"encoder_block": "sab", "dim_hidden": 512, "num_heads": 8, "encoder_layers": 3,
                               "decoder_sab": True, "cell_normalization": "per_sample"}],
          "scrat": [{"model_dim": 512, "num_heads": 16, "num_layers": 4, "cell_normalization": "per_sample", "pca": False}]}
MICRO_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def network(model, config, in_dim):
    if model == "cytoset":
        from sc2bulk.encoders import cytoset
        return cytoset.network(config, in_dim)
    if model == "set_transformer":
        from sc2bulk.encoders import set_transformer
        return set_transformer.network(config, in_dim)
    from sc2bulk.encoders import scrat
    return scrat.network(config, in_dim, os.environ["SC2BULK_SCRAT"])


def stream_seeds(pair, unit, i):
    return dict(init_seed=seeds.derived("search_init", pair, i, *unit), bag_seed=seeds.derived("search_sampling", pair, i, *unit),
                val_seed=seeds.derived(T.VALIDATION_BAGS_PURPOSE, pair, T.SEARCH_INDEX_OFFSET + i, *unit))


def unit_inputs(pair, unit, config, i):
    cells = (training_inputs.unit_cells_all_available(pair, unit) if config["ncell"] > settings.MAX_STORED_CELLS
             else training_inputs.unit_cells(pair, unit))
    train, val = training_inputs.split_cells(cells, training_inputs.split_sides(pair, unit, "search_split", i))
    return cells, train, val, harness.Inputs(config, train, training_inputs.pseudobulk_log(pair, unit))


def train_job(args, pair, unit, config, log):
    cells, train, val, inputs = unit_inputs(pair, unit, config, args.seed)
    device = DEVICE
    ages = training_inputs.training_ages(pair, unit, train.samples.index) if config.get("age_similarity_loss") else None
    epochs = open(log, "w")
    record_epoch = lambda e, s, score: (epochs.write(json.dumps({"epoch": e, "seconds": round(s, 2), "bce": score}) + "\n"), epochs.flush())
    _, state, record, _, _ = harness.train(lambda: network(args.model, config, harness.input_width(config, len(cells.genes))), config, train, val, inputs,
                                           device=device, ages=ages, on_epoch=record_epoch, **stream_seeds(pair, unit, args.seed))
    bytes_ = sum(t.numel() * t.element_size() for t in state.values())
    return {"training_seconds": record.seconds, "epochs": record.epochs, "kept_epoch": record.stop_epoch, "steps": record.steps,
            "seconds_per_epoch_with_setup": round(record.seconds / record.epochs, 2),   # per-epoch: from the JSONL
            "cap_hit": record.epochs == config["n_epochs"] if args.block == "base" else None,
            "did_train": record.did_train, "d": record.margin_d, "d_star": record.margin_d_star, "p": record.margin_p,
            "testable": record.testable, "peak_gpu_allocated": gpu_peak("allocated"),
            "peak_gpu_reserved": gpu_peak("reserved"), "resident": harness.fits_on_device(train, val),
            "checkpoint_bytes": bytes_, "micro_batch": config["micro_batch"]}


def memory_job(args, pair, unit, base):
    """Peak reserved GPU memory of a few optimizer steps per micro-batch size, for one memory-worst
    configuration (--variant; one per process, so a failed size leaves nothing behind for the next), with
    both sides resident as in training (the largest size that fits wins). GPU only."""
    assert DEVICE == "cuda", "the memory block measures the GPU"
    out = {}
    for variant, worst in [(args.variant, WIDEST[args.model][args.variant])]:
        config = {**base, **worst}
        cells, train, val, inputs = unit_inputs(pair, unit, config, 0)
        fits = harness.fits_on_device(train, val)
        residents = harness.Resident(train, inputs, "cuda", fits), harness.Resident(val, inputs, "cuda", fits)
        sizes = train.samples.assign(n=pd.Series(train.sample_id).value_counts()).sort_values("n", ascending=False)
        rng = np.random.default_rng(0)
        peaks = {}
        for micro in sorted(set(MICRO_SIZES) | ({config["batch_size"]} if config.get("age_similarity_loss") else set())):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(0)
            net = network(args.model, config, harness.input_width(config, len(cells.genes))).cuda().train()
            opt = torch.optim.Adam(net.parameters(), lr=1e-4)
            biggest = list(sizes.index[:micro])   # the samples with the most cells: full-size bags
            part = harness.bags.fixed_bags(train.cell_id, train.sample_id, biggest, config["ncell"], 1, rng)
            part = (part * (micro // len(part) + 1))[:micro]
            try:
                for _ in range(3):
                    opt.zero_grad()
                    _, logits = harness.forward_bags(net, part, residents[0])
                    logits.sum().backward()
                    opt.step()
                peaks[micro] = {"reserved": torch.cuda.max_memory_reserved(), "smallest_bag": min(len(ids) for _, ids in part),
                                "peak_host_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}   # the process peak so far
            except torch.cuda.OutOfMemoryError:   # the measurement itself: the size doesn't fit
                peaks[micro] = "oom"
                break
            del net, opt
        out[f"variant_{variant}"] = {"config": worst, "resident": fits, "peak_by_micro_batch": peaks}
        del residents
    return {"memory": out}


EQUALITY_SETTINGS = ({"cell_normalization": "none"}, {"cell_normalization": "population"}, {"cell_normalization": "per_sample"},
                     {"cell_normalization": "population", "pseudobulk_gene_scaler": True},
                     {"cell_normalization": "per_sample", "population_distance_feature": True},
                     {"cell_normalization": "none", "pca": True},          # ScRAT's base
                     {"cell_normalization": "population", "pca": True})   # ScRAT's population candidate


EQUALITY_CHUNK = 8


def gathered_vs_host(resident, inputs, side, part):
    worst = 0.0
    for size, (idx, positions) in resident.indices(part).items():
        for start in range(0, len(positions), EQUALITY_CHUNK):   # a few bags at a time: the whole gather doesn't fit
            x, _ = resident.gather(idx[start:start + EQUALITY_CHUNK])
            for row, position in enumerate(positions[start:start + EQUALITY_CHUNK]):
                s, ids = part[position]
                host = inputs.cells(side.X[side.row_of(ids)].toarray(), s)
                worst = max(worst, float(np.abs(x[row].cpu().numpy() - host).max()))
            del x
    return worst


def equality_job(args, pair, unit):
    """Device-resident gather vs the host transform (Inputs.cells) on real bags, training and validation
    sides (the cached validation bags), 500 and 1,000 cells and the up-to-3,000 store where it fits: max |diff|."""
    out = {}
    for store in ("stored", "all"):
        ncell = representations_cap() if store == "all" else 1000
        config0 = {**search_space.base("cytoset"), "ncell": ncell}
        cells = training_inputs.unit_cells_all_available(pair, unit) if store == "all" else training_inputs.unit_cells(pair, unit)
        train, val = training_inputs.split_cells(cells, training_inputs.split_sides(pair, unit, "search_split", 0))
        if store == "all" and not harness.fits_on_device(train, val):
            out["all: host gather (not resident)"] = None
            continue
        pseudobulk = training_inputs.pseudobulk_log(pair, unit)
        for extras in EQUALITY_SETTINGS:
            inputs = harness.Inputs({**config0, **extras}, train, pseudobulk)
            for name, side in (("train", train), ("validation", val)):
                resident = harness.Resident(side, inputs, DEVICE, True)
                for n in ((500, 1000) if store == "stored" else (ncell,)):
                    rng = np.random.default_rng(1)
                    part = harness.bags.fixed_bags(side.cell_id, side.sample_id, side.samples.index[:20], n, 10, rng)
                    part += [(s, ids[:1]) for s, ids in part[:5]]   # 1-cell bags (per_sample's population fallback)
                    out[f"{store} {name} {n} {json.dumps(extras, sort_keys=True)}"] = gathered_vs_host(resident, inputs, side, part)
                del resident
    values = [v for v in out.values() if v is not None]
    return {"max_abs_diff": out, "passed": max(values) <= 1e-4}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=("base", "candidate", "sweep", "scvi", "memory", "equality"), required=True)
    parser.add_argument("--model", choices=("cytoset", "set_transformer", "scrat", "scvi"), required=True)
    parser.add_argument("--pair", choices=PAIR_ORDER, required=True)
    parser.add_argument("--unit", default=None, help="a unit name (default: the pair's first unit)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate", help="a search candidate name (block candidate / scvi)")
    parser.add_argument("--ncell", help="bag size (sweep and memory blocks): an integer or 'all'")
    parser.add_argument("--variant", type=int, default=0, help="the memory block's WIDEST variant")
    args = parser.parse_args()
    determinism.configure()
    assert not manifest.git_state()["dirty"], "the timing test runs on a clean, committed tree (spec v2)"
    pair = args.pair
    unit = next(u for u in folds.units(pair) if args.unit in (None, folds.unit_name(*u)))
    ncell = (representations_cap() if args.ncell == "all" else int(args.ncell)) if args.ncell else T.FROZEN_NCELL
    config = search_space.base(args.model) if args.model != "scvi" else dict(search_space.SCVI_BASE)
    if args.candidate:
        groups = search_space.SCVI_GROUPS if args.model == "scvi" else search_space.GROUPS_BY_ENCODER[args.model]
        config.update(next(g[args.candidate] for g in groups.values() if args.candidate in g))
    if args.model != "scvi":
        config["ncell"] = ncell
        if args.block not in ("memory", "equality"):   # those blocks measure the caps; they don't train
            config["micro_batch"] = harness.micro_batch_for(args.model, config)
        if args.block == "candidate":
            config["n_epochs"] = T.BURN_IN_EPOCHS + 2
    name = f"{args.block}_{args.model}_{folds.unit_name(*unit)}_s{args.seed}" + (f"_{args.candidate}" if args.candidate else "") \
        + (f"_n{args.ncell}" if args.ncell else "") + (f"_v{args.variant}" if args.block == "memory" else "")
    path = paths.output(STAGE, pair, f"{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"block": args.block, "model": args.model, "pair": pair, "unit": folds.unit_name(*unit), "seed": args.seed,
           "candidate": args.candidate, "ncell": args.ncell or T.FROZEN_NCELL, "git": manifest.git_state(),
           "environment": manifest.environment()["name"], "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, "threads": settings.N_THREADS,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "packing": os.environ.get("SC2BULK_JOBS_PER_GPU"),
           "status": "started",
           "load_average_start": os.getloadavg()}
    path.write_text(json.dumps(row, indent=1, default=str))
    start = time.perf_counter()
    if args.block == "equality":
        row.update(equality_job(args, pair, unit))
    elif args.block == "memory":
        row.update(memory_job(args, pair, unit, config))
    elif args.block == "scvi":
        from sc2bulk.encoders import scvi as adapter
        cells = training_inputs.unit_cells(pair, unit, counts=True)
        train, val = training_inputs.split_cells(cells, training_inputs.split_sides(pair, unit, "search_split", args.seed))
        _, record = adapter.fit(train.X, val.X, config, seed=seeds.derived("scvi_init", pair, T.SEARCH_INDEX_OFFSET + args.seed, *unit))
        row.update({"epochs": record["epochs"], "validation_elbo": record["validation_elbo"], "peak_gpu_reserved": gpu_peak("reserved")})
    else:
        row.update(train_job(args, pair, unit, config, path.with_suffix(".epochs.jsonl")))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    row.update({"status": "done", "total_seconds": round(time.perf_counter() - start, 1), "peak_host_rss_kb": usage.ru_maxrss,
                "cpu_seconds": round(usage.ru_utime + usage.ru_stime, 1), "load_average_end": os.getloadavg()})
    path.write_text(json.dumps(row, indent=1, default=str))
    print(json.dumps(row, default=str), flush=True)
    assert row.get("passed", True), "the device-resident gather differs from the host transform: stop the timing test"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # cpu: wiring checks only


def gpu_peak(kind):
    if DEVICE != "cuda":
        return None
    return torch.cuda.max_memory_allocated() if kind == "allocated" else torch.cuda.max_memory_reserved()


def representations_cap():
    from sc2bulk.config import representations
    return representations.ALL_AVAILABLE_CAP


if __name__ == "__main__":
    main()
