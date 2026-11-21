"""CytoSet under the out-of-fold read-out against pseudobulk under the primary read-out (the practitioner's
comparison), for cytoset_cells and cytoset_pseudobulk. Stage 11b's functions do every step, loaded unedited
(runpy, not its main): the members as 11b's out_of_fold builds them (each unit's passing splits, from
out_of_fold_guard.csv), the comparator as against_primary's (stage 08's primary pseudobulk), 11b's bootstrap on a new
stream (ROW_STREAMS["practitioner"] = 35, set here at run time; config/statistics.py is not edited), rows() with Holm
within these two rows (per pair and pooled), and 11b's equal-weight pooled() over stage 11's non-ceiling pairs (its
claims' excluded_at_ceiling, as 11b's assert_stage11_equal reads it); a pair with a unit of fewer than 2 passing
splits is not estimable and is not pooled, as in 11b. Real labels only (labels.require_access).

Outputs, in $SC2BULK_OUTPUTS/11c_practitioner/:
  labelled_practitioner_comparison.csv   the full rows (secondary_statistics columns)
  practitioner_comparison_summary.csv    aggregate only: pair, configuration, delta, CI, u, p, Holm p, n

A check printed beside the table (not written): per pair, the point identity
  delta(CytoSet OOF - pseudobulk primary) = delta(CytoSet OOF - pseudobulk OOF) + delta(pseudobulk OOF - pseudobulk primary),
with the middle term recomputed and compared with 11b's written out_of_fold row.

  python scripts/practitioner_comparison.py
"""
import json
import time

import numpy as np
import pandas as pd

from sc2bulk import determinism, labels, manifest, paths, schemas
from sc2bulk.config import settings
from sc2bulk.config import statistics as S
from sc2bulk.config.pairs import PAIR_ORDER

D = __import__("runpy").run_path(str(settings.REPO / "pipeline" / "11b_secondary.py"), run_name="practitioner")
ROW = "practitioner"
CONFIGS = ("cytoset_cells", "cytoset_pseudobulk")
OUT = settings.LABELLED_OUTPUTS / "11c_practitioner"
SUMMARY = ["pair", "key", "delta", "ci_low", "ci_high", "u", "p_better", "p_worse", "p_better_holm", "p_worse_holm",
           "n0", "n1", "n_seeds", "n_pairs"]


def oof_members(pair, kept):
    return {k: D["member"](D["stage08"](pair, f"{k}_out_of_fold_predictions.parquet"), D["config_keys"](k, variant="out_of_fold_readout"),
                           lambda unit: kept[unit]) for k in CONFIGS}


def practitioner(pair, notes):
    """(rows, (results, members) or None, seeds per unit): CytoSet out-of-fold against primary pseudobulk."""
    units = D["primary_units"](pair)
    kept = {u: D["guard_passes"](pair, u) for u in units}
    members = {**oof_members(pair, kept), D["REFERENCE"]: D["primary"](pair, D["REFERENCE"])}
    seeds_text = "/".join(str(len(kept[u])) for u in units)
    if any(len(kept[u]) < D["MIN_SEEDS"] for u in units):
        notes["not_estimable"].append(f"{ROW} {pair}: a unit with fewer than {D['MIN_SEEDS']} passing splits")
        return D["rows"]({}, members, ROW, pair, units[0][0], missing=CONFIGS), None, seeds_text
    results, n, missing = D["bootstrap"](pair, units, members, ROW)
    assert not missing
    return D["rows"](results, members, ROW, pair, units[0][0], family=True, n=n), (results, members), seeds_text


def identity_check(pair, results):
    """Point deltas only: CytoSet OOF vs pseudobulk OOF (11b's out_of_fold row) and pseudobulk OOF vs primary."""
    units = D["primary_units"](pair)
    kept = {u: D["guard_passes"](pair, u) for u in units}
    pb_oof = D["member"](D["stage08"](pair, f"{D['REFERENCE']}_out_of_fold_predictions.parquet"),
                         D["config_keys"](D["REFERENCE"], variant="out_of_fold_readout"), lambda unit: kept[unit])
    oof, _, _ = D["bootstrap"](pair, units, {**oof_members(pair, kept), D["REFERENCE"]: pb_oof}, ROW)
    shift, _, _ = D["bootstrap"](pair, units, {"pseudobulk_out_of_fold": pb_oof, D["REFERENCE"]: D["primary"](pair, D["REFERENCE"])}, ROW)
    s = shift["pseudobulk_out_of_fold"][0]
    return [{"pair": pair, "key": k, "practitioner": results[k][0], "oof_row": oof[k][0], "pb_oof_minus_pb_primary": s,
             "residual": results[k][0] - (oof[k][0] + s)} for k in CONFIGS]


def main():
    start = time.time()
    assert settings.BLINDED is None, "the practitioner comparison is a real-label run"
    assert ROW not in S.ROW_STREAMS and 35 not in S.ROW_STREAMS.values(), "stream 35 is taken"
    S.ROW_STREAMS[ROW] = 35
    determinism.configure()
    labels.require_access()
    cost = manifest.begin()
    claims = json.loads(paths.output("11_statistics", "pooled", f"{settings.LABELLED}_claims.json").read_text())
    kept = [p for p in PAIR_ORDER if p not in claims["excluded_at_ceiling"]]
    notes = {"not_estimable": [], "not_pooled": []}
    frames, per_pair, seeds_text, checks = [], {}, {}, []
    for pair in PAIR_ORDER:
        D["CACHE"].clear()
        frame, per_pair[pair], seeds_text[pair] = practitioner(pair, notes)
        frames.append(frame)
        if per_pair[pair] is not None:
            checks += identity_check(pair, per_pair[pair][0])
    made = D["pooled"](per_pair, kept, ROW, True, notes)
    pooled_pairs = [p for p in kept if per_pair.get(p) is not None]
    if made:
        frames += made[0]
    table = pd.concat(frames, ignore_index=True)

    OUT.mkdir(parents=True, exist_ok=True)
    full = OUT / f"{settings.LABELLED}_practitioner_comparison.csv"
    table.to_csv(manifest.temporary(full), index=False)
    record = {"script": "scripts/practitioner_comparison.py", "stream": 35, "resamples": D["B"], "pooled_pairs": pooled_pairs,
              "notes": notes, "code": manifest.stage_code_hash(settings.REPO / "pipeline" / "11b_secondary.py")}
    manifest.finish(full, record, cost, inputs=sorted(D["READ"]))

    summary = table.assign(n_seeds=table["pair"].map(seeds_text).fillna("pairs: " + ",".join(pooled_pairs)),
                           n_pairs=np.where(table["pair"] == "pooled", len(pooled_pairs), 1))[SUMMARY]
    # aggregate only (no sample IDs, per-sample values or labels), so written outside manifest.finish's labelled_ rule
    summary.to_csv(OUT / "practitioner_comparison_summary.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nstage 11 ceiling pairs: {claims['excluded_at_ceiling']}; pooled pairs: {pooled_pairs}")
    print(f"notes: {json.dumps(notes)}")

    check = pd.DataFrame(checks)
    written = schemas.read("secondary_statistics", paths.output("11b_secondary", "pooled", f"{settings.LABELLED}_secondary_statistics.parquet"))
    written = written[(written["row"] == "out_of_fold") & written["key"].isin(CONFIGS) & (written["pair"] != "pooled")]
    check = check.merge(written[["pair", "key", "delta"]].rename(columns={"delta": "oof_row_written"}), on=["pair", "key"], how="left")
    print("\nidentity check (points): practitioner = oof_row + pb_oof_minus_pb_primary; oof_row recomputed vs 11b's written row")
    print(check.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    assert np.allclose(check["residual"], 0, atol=1e-12), "the identity fails"
    assert np.allclose(check["oof_row"], check["oof_row_written"], atol=1e-12), "the recomputed out-of-fold delta differs from 11b's"
    print(f"\nruntime {time.time() - start:.0f} s; peak RSS {manifest.cost(cost)['peak_rss_kb'] / 1e6:.2f} GB")


if __name__ == "__main__":
    main()
