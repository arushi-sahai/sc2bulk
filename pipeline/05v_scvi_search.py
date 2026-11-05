"""Stage 05v: scVI's search, outside the set-encoder
budget. One change from the scvi-tools defaults (search_space.SCVI_BASE / SCVI_GROUPS), trained on the
split's training-side cells (counts on the unit's panel) at each search seed, selected on the returned
model's validation ELBO (lower is better: scvi logs the negative ELBO, with the full KL, no warm-up weight;
checked in scvi-tools 1.3.3's training plan) with the set encoders' rules: the mean paired improvement
over the base must exceed its SD, winners merged (trained at the k seeds) or the single best, else the
base. No did-it-train screen (unsupervised); the ELBO is asserted finite. Runs in the `scvi` environment.
Models and embeddings aren't kept: stage 06 refits scVI at the frozen configuration.

Outputs, in <outputs>/05v_scvi_search/<pair>/<unit>/:
  elbo_s<i>.csv         candidate, change, validation ELBO, epochs
  elbo_merged_s<i>.csv  the merged configuration's fit, when >= 2 groups win
  selection.json    winners, the merge, the frozen configuration

  python pipeline/05v_scvi_search.py --phase train --pair PAIR --set UNIT --seed I
  python pipeline/05v_scvi_search.py --phase select --pair PAIR --set UNIT
  python pipeline/05v_scvi_search.py --pair PAIR      every unit's train jobs, then its selection
"""
import argparse
import json

import numpy as np
import pandas as pd

from sc2bulk import determinism, folds, manifest, paths, search, seeds, training_inputs
from sc2bulk.config import search_space, settings, smoke
from sc2bulk.config import training as T
from sc2bulk.encoders import scvi as adapter

STAGE = "05v_scvi_search"
K = smoke.N_SEARCH_SEEDS if settings.SMOKE else settings.N_SEARCH_SEEDS


def trials() -> dict:
    out = {"base": (None, dict(search_space.SCVI_BASE))}
    for group, candidates in search_space.SCVI_GROUPS.items():
        out.update({name: (group, {**search_space.SCVI_BASE, **change}) for name, change in candidates.items()})
    return out


def fit_seed(pair, unit, i, configs: dict) -> list:
    cells = training_inputs.unit_cells(pair, unit, counts=True)
    train, val = training_inputs.split_cells(cells, training_inputs.split_sides(pair, unit, "search_split", i))
    seed = seeds.derived("scvi_init", pair, T.SEARCH_INDEX_OFFSET + i, *unit)   # no stream shared with stage 06
    rows = []
    for name, (group, config) in configs.items():
        _, record = adapter.fit(train.X, val.X, config, seed=seed)
        assert np.isfinite(record["validation_elbo"]), f"{pair} {unit} {name} s{i}: non-finite ELBO"
        rows.append({"candidate": name, "group": group or "base", "config": json.dumps(config),
                     "validation_elbo": record["validation_elbo"], "epochs": record["epochs"]})
    return rows


def per_seed(table, key) -> np.ndarray:
    """A candidate's ELBO at seeds 0..K-1 in order (asserted: the pairing is by seed)."""
    rows = table[table["candidate"] == key].sort_values("seed")
    assert list(rows["seed"]) == list(range(K)), f"{key}: seeds {list(rows['seed'])}"
    return rows["validation_elbo"].to_numpy()


def select(pair, unit, out, record, cost) -> dict:
    table = pd.concat([pd.read_csv(out / f"elbo_s{i}.csv") for i in range(K)], keys=range(K), names=["seed"]).reset_index(level=0)
    all_trials = trials()
    elbo = {k: per_seed(table, k) for k in all_trials}
    base = all_trials["base"][1]
    if K < 2:
        return {"frozen": base, "how": "base (one search seed)"}
    groups = {}
    for name, (group, _) in all_trials.items():
        if group:
            groups.setdefault(group, {})[name] = elbo[name]
    changes = {name: {k: v for k, v in config.items() if base.get(k) != v} for name, (_, config) in all_trials.items()}
    winners = search.winners(elbo["base"], groups, set(elbo))   # every candidate eligible (no did-it-train screen)
    gains = {name: search.improvement(elbo["base"], elbo[name])[0] for name in winners.values()}
    merged_result = None
    if len(winners) >= 2:
        merged = search.merged_config(base, winners, changes)
        merged_elbo = []
        for i in range(K):
            rows = fit_seed(pair, unit, i, {"merged": (None, merged)})
            path = out / f"elbo_merged_s{i}.csv"
            pd.DataFrame(rows).to_csv(manifest.temporary(path), index=False)
            manifest.finish(path, {**record, "seed": i, "merged": merged}, cost)
            merged_elbo.append(rows[0]["validation_elbo"])
        merged_result = (True, bool(search.beats(elbo["base"], np.array(merged_elbo))))
    frozen, how = search.frozen(base, winners, changes, gains, merged_result)
    return {"frozen": frozen, "how": how, "winners": winners, "gains": gains, "merged_result": merged_result,
            "mean_elbo": {k: float(v.mean()) for k, v in elbo.items()}}


def unit_dir(pair, unit):
    return paths.output(STAGE, pair, "elbo", unit).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "select"))
    parser.add_argument("--pair", required=True)
    parser.add_argument("--set")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    determinism.configure()
    if args.phase is None:   # every job of the pair, in order (run_all)
        assert args.set is None and args.seed is None, "--set/--seed need --phase"
        for unit in folds.units(args.pair):
            for i in range(K):
                job("train", args.pair, unit, i)
            job("select", args.pair, unit, None)
        return
    job(args.phase, args.pair, next(u for u in folds.units(args.pair) if folds.unit_name(*u) == args.set), args.seed)


def job(phase, pair, unit, seed):
    out = unit_dir(pair, unit)
    out.mkdir(parents=True, exist_ok=True)
    record = {"stage": STAGE, "pair": pair, "unit": unit, "smoke": settings.SMOKE, "search_seeds": K,
              "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair),
              "inputs": manifest.input_hashes([*folds.inputs(pair), settings.OUTPUTS / "03_splits" / pair / "splits.csv",
                                               paths.output("04_panel", pair, "panel.csv", unit),
                                               paths.output("04_panel", pair, "shared_genes.csv"),
                                               settings.OUTPUTS / "03_splits" / pair / "donor_folds.csv"])}
    if phase == "train":
        path = out / f"elbo_s{seed}.csv"
        if manifest.is_done([path], {**record, "seed": seed}):
            return
        cost = manifest.begin(device="cuda")
        pd.DataFrame(fit_seed(pair, unit, seed, trials())).to_csv(manifest.temporary(path), index=False)
        manifest.finish(path, {**record, "seed": seed}, cost)
        return
    sources = [out / f"elbo_s{i}.csv" for i in range(K)]
    select_record = {**record, "phase": "select", "tables": manifest.input_hashes(sources)}
    if manifest.is_done([out / "selection.json"], select_record):
        return
    cost = manifest.begin(device="cuda")
    decision = select(pair, unit, out, select_record, cost)
    path = out / "selection.json"
    manifest.temporary(path).write_text(json.dumps(decision, indent=1, default=str))
    manifest.finish(path, select_record, cost, inputs=sources)
    print(json.dumps(decision, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
