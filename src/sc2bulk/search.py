"""Stage 05's selection rules, as pure
functions of the search seeds' results, so every branch is testable without training.

A trial is one candidate's results over the k search seeds, in seed order: its class-balanced
validation BCE at the kept checkpoint and its did-it-train fields (TrainingRecord.margin_d_star, margin_p).
Seeds are paired across candidates (the same splits, bags and inits), so differences are taken seed
by seed.
"""
import json
import math

import numpy as np

from sc2bulk.config import training as T


def eligible(d_star, p, testable: bool) -> bool:
    """D* > MARGIN_FLOOR x ln 2 on at least ceil(k/2) seeds (the scale-invariant collapse
    screen); on a testable unit, also at least ceil(k/2) seeds with p <= NULL_ALPHA (the evidence screen)."""
    d_star, p = np.asarray(d_star, dtype=float), np.asarray(p, dtype=float)
    majority = math.ceil(len(d_star) / 2)
    floor = (d_star > T.MARGIN_FLOOR * math.log(2)).sum() >= majority
    return bool(floor and (not testable or (p <= T.NULL_ALPHA).sum() >= majority))


def improvement(base_bce, bce) -> tuple:
    """(mean, SD with ddof 1) of the seed-paired improvement base - candidate. Needs k >= 2 seeds (the
    smoke run's single search seed exercises training only; its frozen configuration is the base)."""
    diff = np.asarray(base_bce, dtype=float) - np.asarray(bce, dtype=float)
    assert len(diff) >= 2, "the adoption margin needs at least 2 search seeds"
    return float(diff.mean()), float(diff.std(ddof=1))


def beats(base_bce, bce) -> bool:
    """The adoption margin: the mean paired improvement exceeds its SD."""
    mean, sd = improvement(base_bce, bce)
    return mean > sd


def winners(base_bce, groups: dict, eligible_names: set, exclude=()) -> dict:
    """group -> winner over every group that has one ({group: {name: per-seed BCE}}); candidates in
    `exclude` are left out (the per-sample sensitivity row reruns the rule without
    norm_per_sample)."""
    out = {}
    for group, candidates in groups.items():
        name = group_winner(base_bce, candidates, set(eligible_names) - set(exclude))
        if name is not None:
            out[group] = name
    return out


def group_winner(base_bce, candidates: dict, eligible_names: set):
    """Among a group's eligible candidates ({name: per-seed BCE}), the one with the lowest mean BCE,
    if it beats the base; else None."""
    pool = {name: bce for name, bce in candidates.items() if name in eligible_names}
    if not pool:
        return None
    best = min(sorted(pool), key=lambda name: np.mean(pool[name]))
    return best if beats(base_bce, pool[best]) else None


# Pairs of groups whose winners can't be merged. Abundance weighting replaces
# the output pooling: the one with the larger mean paired improvement is kept. The gene scaler has no
# effect under per_sample: the scaler is dropped. (ScRAT's per_sample candidate sets pca=False itself,
# so it merges with the panel input)
CONFLICTS = (("abundance_weighting", "out_pool"), ("pseudobulk_gene_scaler", "cell_normalization"))


def resolve(winners: dict, changes: dict, gains: dict) -> dict:
    """Drop the weaker winner of each conflicting pair. winners: group -> candidate name; changes:
    candidate -> its setting change; gains: candidate -> mean paired improvement. The scaler conflict
    applies only when the normalization winner is per_sample."""
    kept = dict(winners)
    for a, b in CONFLICTS:
        if a in kept and b in kept:
            if b == "cell_normalization" and changes[kept[b]]["cell_normalization"] != "per_sample":
                continue
            loser = a if a == "pseudobulk_gene_scaler" or gains[kept[a]] < gains[kept[b]] else b
            del kept[loser]
    return kept


def merged_config(base: dict, kept: dict, changes: dict) -> dict:
    """The base with every kept winner's change applied."""
    config = dict(base)
    for name in kept.values():
        config.update(changes[name])
    return config


def frozen(base: dict, kept: dict, changes: dict, gains: dict, merged_result):
    """The unit's frozen configuration and how it was reached. With no winner: the base. With one:
    that winner (the merged training is skipped; it would be identical). With several:
    merged_result = (eligible, beats the base) of the merged configuration's training; if both hold,
    the merge; else the single best winner (largest mean paired improvement)."""
    if not kept:
        return dict(base), "base"
    if len(kept) == 1:
        return merged_config(base, kept, changes), "single_winner"
    assert merged_result is not None, "several winners: the merged configuration must be trained"
    if all(merged_result):
        return merged_config(base, kept, changes), "merged"
    best = max(sorted(kept.values()), key=lambda name: gains[name])
    return merged_config(base, {"best": best}, changes), "best_single"


def modal(selections: list, base: dict) -> dict:
    """Human blood, aging: the configuration selected in the most folds (whole configurations
    compared); ties go to the base if it is among the tied, else to fold 0's selection."""
    canonical = lambda c: json.dumps(c, sort_keys=True, default=str)   # also equates a stored list with a tuple
    keys = [canonical(json.loads(canonical(s))) for s in selections]
    counts = {k: keys.count(k) for k in keys}
    top = max(counts.values())
    tied = [k for k in counts if counts[k] == top]
    if len(tied) == 1:
        return selections[keys.index(tied[0])]
    if canonical(json.loads(canonical(base))) in tied:
        return dict(base)
    assert keys[0] in tied, "fold 0's selection must be among the tied"
    return selections[0]
