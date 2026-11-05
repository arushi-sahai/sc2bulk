"""The planted-signal simulation's pure functions (stage 12_simulation, Fig 5D).

Real cells, a simulated 0/1 label, and a phenotype planted in 50 panel genes G, applied to every sample's stored counts
of G (cells x |G| blocks; nothing else changes):
- mean shift: label-1 cells' counts of G- are binomially thinned (keep probability rho), label-0 cells' of G+;
- structure only: each sample's total count of each gene of G is reallocated among its cells by a multinomial draw,
  over all its cells (label 0) or over one random subset of a fraction q of them (label 1), with probabilities
  proportional to each cell's library size excluding G; every sample's per-gene sums are preserved exactly.
The strength is the weakest grid level at which a fixed oracle (no fitting) reaches the target AUROC. Nothing here reads
a bulk label, and the calibration reads no model output.
"""
import math

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk.config import settings
from sc2bulk.config import simulation as SIM


def planted_genes(panel, detected_fraction, seed: int) -> pd.Series:
    """gene -> "plus" / "minus": N_PLANTED genes drawn uniformly (sorted candidates) from the panel genes detected in
    >= DETECTION_MIN_FRACTION of the training cells and not mitochondrial, then split at random into two halves.
    detected_fraction: gene -> fraction of cells with a nonzero count. Label-free."""
    detected_fraction = pd.Series(detected_fraction)
    candidates = sorted(g for g in panel if detected_fraction[g] >= SIM.DETECTION_MIN_FRACTION
                        and not g.startswith(SIM.MITO_PREFIXES))
    assert len(candidates) >= SIM.N_PLANTED, f"{len(candidates)} candidate genes for {SIM.N_PLANTED} planted"
    rng = np.random.default_rng(seed)
    genes = list(rng.choice(candidates, SIM.N_PLANTED, replace=False))
    order = rng.permutation(SIM.N_PLANTED)
    half = SIM.N_PLANTED // 2
    return pd.Series({genes[i]: "plus" if k < half else "minus" for k, i in enumerate(order)}).sort_index()


def simulated_labels(stratum: pd.Series, seed: int) -> pd.Series:
    """holdout_group -> simulated 0/1 label: within each stratum (the real 0/1 group), the holdout groups (sorted, then
    shuffled) are split in half, the extra group of an odd stratum going to the label a seeded coin picks.
    stratum: holdout_group -> its stratum key."""
    rng = np.random.default_rng(seed)
    out = {}
    for key in sorted(set(stratum)):
        members = rng.permutation(sorted(stratum.index[stratum == key]))
        n0 = len(members) // 2 + (int(rng.integers(2)) if len(members) % 2 else 0)
        out.update({g: int(i >= n0) for i, g in enumerate(members)})
    return pd.Series(out).sort_index()


def half_split(sim_label: pd.Series, seed: int, min_groups: int) -> pd.Series:
    """holdout_group -> "A" / "B" for one validation side's groups: within each simulated label, the groups (sorted, then
    shuffled) are split, the first ceil(n / 2) to half A (early stopping, checkpoint, did-it-train), the rest to half B
    (scored). Asserts >= min_groups groups per label in each half."""
    rng = np.random.default_rng(seed)
    out = {}
    for label in (0, 1):
        members = rng.permutation(sorted(sim_label.index[sim_label == label]))
        n_a = math.ceil(len(members) / 2)
        assert min(n_a, len(members) - n_a) >= min_groups, f"label {label}: {len(members)} groups can't fill two halves"
        out.update({g: "A" if i < n_a else "B" for i, g in enumerate(members)})
    return pd.Series(out).sort_index()


def uniforms(block: np.ndarray, rng) -> np.ndarray:
    """One uniform per count unit of a sample's G block (row-major entries), drawn once and shared by every level."""
    return rng.random(int(block.sum()))


def mean_shift(block: np.ndarray, thinned: np.ndarray, rho: float, u: np.ndarray) -> np.ndarray:
    """A sample's G block with the columns `thinned` binomially thinned: each count unit is kept if its uniform is < rho
    (u from uniforms(); the same u at every level, so stronger levels keep subsets of weaker ones)."""
    flat = block.ravel()
    owner = np.repeat(np.arange(flat.size), flat)
    kept = np.bincount(owner, weights=u < rho, minlength=flat.size).astype(block.dtype).reshape(block.shape)
    return np.where(thinned[None, :], kept, block)


def structure_only(block: np.ndarray, weight: np.ndarray, subset: np.ndarray, rng) -> np.ndarray:
    """A sample's G block with each gene's total count reallocated by a multinomial draw over the cells `subset` (row
    indices; all cells for label 0), with probabilities proportional to `weight` (each cell's library size excluding G) within the subset. Per-gene sums are preserved exactly."""
    assert (weight[subset] > 0).all(), "a cell with no counts outside G"
    p = weight[subset] / weight[subset].sum()
    out = np.zeros_like(block)
    out[subset] = rng.multinomial(block.sum(axis=0), p).T
    return out


def subset_of(order: np.ndarray, q: float) -> np.ndarray:
    """Label 1's cells at level q: the first ceil(q n) cells of the sample's one seeded ordering (so subsets nest)."""
    return np.sort(order[:math.ceil(q * len(order))])


def log_cp10k(counts: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """log1p(counts / totals x TARGET_SUM), per row: preprocess.log_normalize's arithmetic (float32 counts and scale) for
    rows whose shared-gene totals are given."""
    scale = (settings.TARGET_SUM / np.asarray(totals, dtype=np.float64)).astype(np.float32)
    return np.log1p(np.asarray(counts, dtype=np.float32) * scale[:, None])


def log_normalize_rows(counts, totals):
    """preprocess.log_normalize for rows whose shared-gene totals are given (counts: cells x panel genes, CSR): the same
    float32 arithmetic, so unchanged cells come out bit-identical to the unsimulated data. Returns float32 CSR."""
    X = sp.csr_matrix(counts, dtype=np.float32, copy=True)
    X.data *= np.repeat(settings.TARGET_SUM / np.asarray(totals, dtype=np.float64), np.diff(X.indptr)).astype(np.float32)
    np.log1p(X.data, out=X.data)
    return X


def oracle_mean(pseudobulk_g: np.ndarray, sample_totals: np.ndarray, plus: np.ndarray) -> np.ndarray:
    """The mean scenario's oracle per sample: mean log-CP10k of the pseudobulk over G+ minus over G- (label 1 thins G-).
    pseudobulk_g: samples x |G| summed counts; sample_totals: the samples' summed shared-gene counts."""
    x = log_cp10k(pseudobulk_g, sample_totals)
    return x[:, plus].mean(axis=1) - x[:, ~plus].mean(axis=1)


def oracle_structure(block: np.ndarray, totals: np.ndarray) -> float:
    """The structure scenario's oracle for one sample: minus the mean over its cells of the mean over G of
    log1p(CP10k) (concentration lowers the mean of logs; the log of the sum is unchanged). totals: the cells' simulated
    shared-gene totals."""
    return -float(log_cp10k(block, totals).mean())


def calibrate(aurocs) -> tuple:
    """(index, reached): the first grid level (weakest first) whose oracle AUROC is >= ORACLE_TARGET, else the last."""
    reached = [i for i, a in enumerate(aurocs) if a >= SIM.ORACLE_TARGET]
    return (reached[0], True) if reached else (len(aurocs) - 1, False)


READINGS = {   # (encoder passes, mean_log_cells passes) -> the pre-stated reading
    (True, True): "detects non-sum signal at this strength, as a fixed mean of logs also does; no capacity beyond a fixed aggregator shown",
    (True, False): "detects structure-only signal that the fixed aggregator, with the same read-out, does not detect at this n",
    (False, True): "failed to learn a signal that a fixed aggregator recovers: an encoder-specific failure",
    (False, False): "inconclusive about set encoders specifically: the planted structure signal was not learnable by a fixed "
                    "aggregator with a ridge read-out at this sample size; the positive control is still reported as failed",
}
PER_SAMPLE_MEAN_READING = ("mean contrast removed by construction (per-sample normalization); value reported, not read as "
                           "mean-signal detection; no format reading (its one-profile bulk takes the population route)")
BY_CONSTRUCTION = {   # the table: values drawn as references, never read as results
    "structure_only: pseudobulk within_sc_auroc": "chance by construction: pseudobulk's features are the unsimulated data's "
                                                  "(per-gene sums preserved, asserted) and the simulated labels are random",
    "structure_only: simulated_bulk_auroc, every row": "chance by construction: every summed profile is the unsimulated data's "
                                                       "(asserted); a consequence of summation, not a finding",
    "mean_shift: pseudobulk simulated_bulk_auroc vs within_sc_auroc": "nearly equal by construction: the same profiles, "
                                                                      "scaled on a different side (a sanity check)",
}
FORMAT_READING = ("format reading made: simulated-bulk vs within-single-cell AUROC (descriptive; a drop means the bag -> "
                  "one-profile change alone loses a learned mean signal)")


def criterion(table: pd.DataFrame, routes: dict) -> dict:
    """The pre-declared success criterion and its readings. table: one row per (key,
    encoder_seed, readout_seed, scenario) with within_sc_auroc; every seed counts, did-it-train failures included.
    Delta(m) = mean over (e, r) of [AUROC_within(m) - AUROC_within(pseudobulk)] in the structure-only scenario, for each
    set encoder and for mean_log_cells; each passes if Delta >= SUCCESS_MARGIN (1e-12 tolerance); the control passes
    if every set encoder kept after triage does. routes: kept encoder -> its frozen cell_normalization (stage 05's
    selection)."""
    part = table[table["scenario"] == SIM.CRITERION_SCENARIO].set_index(["encoder_seed", "readout_seed", "key"])["within_sc_auroc"].unstack("key")
    assert part.notna().all().all(), "every (e, r) needs every row"
    keys = {m: f"{m}_cells" for m in routes} | {"mean_log_cells": "mean_log_cells"}
    delta = {m: float((part[k] - part["pseudobulk"]).mean()) for m, k in keys.items()}
    passes = {m: d >= SIM.SUCCESS_MARGIN - SIM.MARGIN_TOLERANCE for m, d in delta.items()}
    encoders = list(routes)
    return {"scenario": SIM.CRITERION_SCENARIO, "margin": SIM.SUCCESS_MARGIN, "splits_x_readout_seeds": len(part),
            "delta": delta, "passes": passes, "set_encoders": encoders,
            "cut": [m for m in SIM.SET_ENCODERS if m not in SIM.ENCODERS],
            "control_passes": all(passes[m] for m in encoders), "failing": [m for m in encoders if not passes[m]],
            "readings_note": "the readings are interpretation only; they never change the pass/fail (frozen §47)",
            "by_construction": BY_CONSTRUCTION,
            "structure_readings": {m: READINGS[(passes[m], passes["mean_log_cells"])] for m in encoders},
            "normalization": dict(routes),
            "mean_shift_readings": {m: PER_SAMPLE_MEAN_READING if routes[m] == "per_sample" else FORMAT_READING for m in encoders},
            "seeds": "all encoder seeds, did-it-train failures included"}
