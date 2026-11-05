"""Stage 08a: the ridge alpha-edge rule, run after stage 08's primary pass
over every pair and before any bulk metric. It reads alphas only (stage 08's heads): no label of any kind.

Counted fits: one per (pair, unit, configuration, encoder seed, read-out seed) of the primary transfer
ridge read-out, on the pair's primary fold set, encoder seeds 0-9 (-1 for the fixed representations;
seeds that did not train count). Not counted: seeds 10-19, split67, Fig 2B, the
out-of-fold row, 08b, LASSO (its ties go to the larger alpha). Per configuration, pooled over pairs,
each edge separately: the fraction of its fits whose alpha is that edge of the grid in use. An edge is
extended if the fraction exceeds ALPHA_EDGE_FRACTION for any configuration; the extension applies to every
configuration and every ridge read-out (one RIDGE_ALPHAS).

The mode is set by the committed config.readout.RIDGE_ALPHA_EXTENSION:
- empty: decide. Writes <outputs>/08_readout/alpha_edges_decision.json once (a later run asserts the same
  extension decision, never overwrites it, and writes its own counts to alpha_edges_after.json) and, if an edge must be extended, exits nonzero naming the value to
  commit; stage 08 then reruns everything (config/readout.py is in its code hash).
- set: report. Asserts it equals the recorded decision, then writes the fractions at the new edges to
  alpha_edges_after.json. It never decides again, so the grid is extended at most once.
The smoke run only prints the counts: it writes and checks no record (its alphas change whenever upstream
code does, and it has no committed decision to check).

  python pipeline/08a_alpha_edges.py [--pair PAIR]   (the rule pools every pair; --pair, which run_all
                                                     passes, changes nothing)
08a pools every pair: run it through a full run_all or `run_all --only 08a_alpha_edges`; `run_all --pair X`
stops here unless every pair's stage 08 output is current.
"""
import argparse
import json
import sys

import pandas as pd

from sc2bulk import folds, manifest, paths
from sc2bulk.config import configurations, settings
from sc2bulk.config import readout as R
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

STAGE = "08_readout"
DECISION = settings.OUTPUTS / STAGE / "alpha_edges_decision.json"
AFTER = settings.OUTPUTS / STAGE / "alpha_edges_after.json"


def counted_alphas(pair: str, key: str) -> list:
    """The chosen alphas of a configuration's counted fits in one pair."""
    out = []
    for unit in folds.units(pair):
        if unit[0] != next(iter(PAIRS[pair]["fold_sets"])):
            continue
        path = paths.output(STAGE, pair, f"{key}_heads.parquet", unit)
        assert manifest.read(path)["settings"]["code"] == manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair), \
            f"{path}: not made by the current stage 08 (and grid)"
        heads = pd.read_parquet(path, columns=["encoder_seed", "readout", "readout_seed", "variant", "alpha"])
        heads = heads[(heads["readout"] == "ridge") & (heads["variant"] == "primary") & (heads["encoder_seed"] < 10)]
        out += list(heads.drop_duplicates(["encoder_seed", "readout_seed"])["alpha"])
    return out


def edges() -> dict:
    """Per configuration, pooled over pairs: the counted fits and the number and fraction at each edge."""
    low, high = min(R.RIDGE_ALPHAS), max(R.RIDGE_ALPHAS)
    out = {}
    for key in configurations.CONFIGURATION_ORDER:
        alphas = [a for pair in PAIR_ORDER for a in counted_alphas(pair, key)]
        n_low, n_high = sum(a == low for a in alphas), sum(a == high for a in alphas)
        out[key] = {"fits": len(alphas), "lower": n_low, "upper": n_high,
                    "fraction_lower": n_low / len(alphas), "fraction_upper": n_high / len(alphas)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER)
    parser.parse_args()
    counts = edges()
    result = {"grid": {"lower": min(R.RIDGE_ALPHAS), "upper": max(R.RIDGE_ALPHAS), "points": len(R.RIDGE_ALPHAS),
                       "extension": list(R.RIDGE_ALPHA_EXTENSION)}, "configurations": counts}
    if settings.SMOKE:
        print(json.dumps(result), flush=True)
        return
    if R.RIDGE_ALPHA_EXTENSION:
        decision = json.loads(DECISION.read_text())
        assert decision["extension"] == list(R.RIDGE_ALPHA_EXTENSION), \
            f"RIDGE_ALPHA_EXTENSION {R.RIDGE_ALPHA_EXTENSION} differs from the recorded decision {decision['extension']}"
        AFTER.write_text(json.dumps({**result, "git": manifest.git_state()}, indent=1))
        print(json.dumps(result), flush=True)
        return
    extension = [edge for edge in ("lower", "upper") if any(c[f"fraction_{edge}"] > R.ALPHA_EDGE_FRACTION for c in counts.values())]
    result["extension"] = extension
    if DECISION.exists():   # decided once: a rerun must reach the same decision (counts may drift: AFTER records them)
        recorded = json.loads(DECISION.read_text())
        assert recorded["extension"] == extension, f"{DECISION}: recorded {recorded['extension']}, this run decides {extension}"
        AFTER.write_text(json.dumps({**result, "git": manifest.git_state()}, indent=1))
    else:
        DECISION.parent.mkdir(parents=True, exist_ok=True)
        DECISION.write_text(json.dumps({**result, "git": manifest.git_state()}, indent=1))
    print(json.dumps(result), flush=True)
    if extension:
        sys.exit(f"alpha edges: set RIDGE_ALPHA_EXTENSION = {tuple(extension)} in config/readout.py and rerun stage 08")


if __name__ == "__main__":
    main()
