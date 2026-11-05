"""A unit's training inputs for the set encoders and scVI: one split's
training and validation samples' cells as protocol.Cells on the unit's panel (log-CP10k over the
shared genes, or counts for scVI), and the samples' log-CP10k pseudobulk (the gene scaler's input).
Also stage 06's row inputs that stage 08 shares: row D's permuted labels, row E's learning-curve
subsets and H3's thinned bulk. Reads only through sc2bulk.folds (the unit's training samples) and
stage 03's splits; no labels beyond the single-cell 0/1 group and age."""
import math

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sc2bulk import all_available, folds, paths, preprocess, protocol, seeds
from sc2bulk.config import deconvolution, representations, settings
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIRS


def panel(pair: str, unit: tuple) -> list:
    return list(pd.read_csv(paths.output("04_panel", pair, "panel.csv", unit), keep_default_na=False)["gene"])


def shared(pair: str) -> list:
    return list(pd.read_csv(paths.output("04_panel", pair, "shared_genes.csv"), keep_default_na=False)["gene"])


def split_sides(pair: str, unit: tuple, purpose: str, index: int) -> dict:
    """sample_id -> "train" / "validation" for one of stage 03's splits of the unit."""
    table = pd.read_csv(settings.OUTPUTS / "03_splits" / pair / "splits.csv", dtype={"sample_id": str})
    rows = table[(table["fold_set"] == unit[0]) & (table["fold"] == unit[1]) & (table["purpose"] == purpose) & (table["index"] == index)]
    return dict(zip(rows["sample_id"], rows["side"]))


def encoder_seeds(pair: str, row: str) -> list:
    """A stage 06 row's encoder seeds (one rule for stages 06 and 08): 0-9 (the smoke run's count under
    SC2BULK_SMOKE), plus the attribution seeds 10-19 for every trained configuration with an encoder level
    (rows A, B and J: the set encoders, CytoSet on pseudobulk, scVI) outside human blood, aging and the smoke
    run (20 seeds, 10 for human blood, aging). Row D: permutation j at
    encoder seed j, j < N_PERMUTED (every pair; 10-19 read stage 03b's splits); rows E, F, H: 0-4; the MSE variant: the search seeds (stage 05's base streams)."""
    from sc2bulk.config import smoke
    n = smoke.N_ENCODER_SEEDS if settings.SMOKE else settings.N_ENCODER_SEEDS
    if row == "D":
        return list(range(n if settings.SMOKE else T.N_PERMUTED))
    if row in ("E", "F", "H"):
        return list(range(min(n, T.CURVE_SEEDS)))
    if row == "MSE":
        return list(range(smoke.N_SEARCH_SEEDS if settings.SMOKE else settings.N_SEARCH_SEEDS))
    attributed = row in ("A", "B", "J") and not settings.SMOKE and pair != "soundlife"
    return list(range(n)) + (list(range(10, 20)) if attributed else [])


def encoder_sides(pair: str, unit: tuple, seed: int) -> dict:
    """Encoder seed e's split: stage 03's encoder_split for 0-9, stage 03b's for the attribution seeds
    10-19."""
    if seed < 10:
        return split_sides(pair, unit, "encoder_split", seed)
    table = pd.read_csv(settings.OUTPUTS / "03b_attribution_splits" / pair / "splits.csv", dtype={"sample_id": str})
    rows = table[(table["fold_set"] == unit[0]) & (table["fold"] == unit[1]) & (table["index"] == seed)]
    assert len(rows), f"{pair} {unit}: no attribution split {seed}"
    return dict(zip(rows["sample_id"], rows["side"]))


def unit_cells(pair: str, unit: tuple, counts: bool = False, cell_subset: str = "all",
               deconvolution_types: bool = False) -> protocol.Cells:
    """The unit's training samples' stored cells on its panel: log-CP10k over the shared genes (or
    raw counts, for scVI), CSR. deconvolution_types: cell types from config/deconvolution.LEVELS (its
    column and map; one type where the pair has none), ScRAT's mixup's (stage 06 row G)."""
    data = folds.training_cells(pair, *unit, cell_subset=cell_subset)
    genes, chosen = shared(pair), panel(pair, unit)
    shared_columns, panel_columns = data.var_names.get_indexer(genes), data.var_names.get_indexer(chosen)
    assert (shared_columns >= 0).all() and set(chosen) <= set(genes), f"{pair} {unit}: genes missing from the store"
    if counts:
        X = sp.csr_matrix(data.X[:, panel_columns], dtype=np.float32)
    else:
        X = log_normalized_panel(data.X, shared_columns, panel_columns)
    samples = folds.training_samples(pair, *unit).set_index("sample_id")[["group", "holdout_group"]]
    cell_type = level_types(pair, data.obs) if deconvolution_types else data.obs["cell_type"].astype(str).to_numpy()
    return protocol.Cells(X=X, genes=chosen, cell_id=data.obs_names.astype(str).to_numpy(),
                          sample_id=data.obs["sample_id"].astype(str).to_numpy(), cell_type=cell_type,
                          library_size=data.obs["total_counts"].to_numpy(dtype=float), samples=samples)


def log_normalized_panel(X, shared_columns, panel_columns, chunk=20_000) -> sp.csr_matrix:
    """preprocess.log_normalize(X[:, shared_columns])[:, panel_columns], value for value, without holding the
    shared-gene columns in full (the timing test's diagnosis): each
    cell's shared-gene total is summed in row chunks with log_normalize's arithmetic (float32 counts, a float32
    sum per row, then float64), and only the panel columns are scaled and log1p'd. X: cells x stored genes, CSR."""
    totals = []
    for i in range(0, X.shape[0], chunk):   # one row chunk of the shared-gene counts at a time
        counts = sp.csr_matrix(X[i:i + chunk][:, shared_columns], dtype=np.float32)
        totals.append(np.asarray(counts.sum(axis=1, dtype=np.float64)).ravel())
    totals = np.concatenate(totals)
    assert (totals > 0).all(), "a profile has no counts on the shared genes"
    out = sp.csr_matrix(X[:, panel_columns], dtype=np.float32)
    out.data *= np.repeat((settings.TARGET_SUM / totals).astype(np.float32), np.diff(out.indptr))
    np.log1p(out.data, out=out.data)
    return out


def level_types(pair: str, obs: pd.DataFrame) -> np.ndarray:
    """Each cell's type at the pair's deconvolution level (config/deconvolution.LEVELS); asserted that every
    cell maps. One type for a pair without a level (mouse microglia)."""
    level = deconvolution.LEVELS[pair]
    if level is None:
        return np.full(len(obs), "one type")
    types = obs[level["column"]].astype(str)
    if level["map"] is not None:
        types = types.map(level["map"])
    assert types.notna().all(), f"{pair}: cells whose type isn't in the deconvolution level's map"
    return types.to_numpy()


def subset(cells: protocol.Cells, sample_ids) -> protocol.Cells:
    keep = np.isin(cells.sample_id, list(sample_ids))
    return protocol.Cells(X=cells.X[keep], genes=cells.genes, cell_id=cells.cell_id[keep], sample_id=cells.sample_id[keep],
                          cell_type=cells.cell_type[keep], library_size=cells.library_size[keep],
                          samples=cells.samples.loc[sorted(set(sample_ids))])


def split_cells(cells: protocol.Cells, sides: dict) -> tuple:
    """(training-side Cells, validation Cells) of one split."""
    return (subset(cells, [s for s, side in sides.items() if side == "train"]),
            subset(cells, [s for s, side in sides.items() if side == "validation"]))


def pseudobulk_log(pair: str, unit: tuple, cell_subset: str = "all") -> dict:
    """sample -> its log-CP10k pseudobulk on the unit's panel (normalized over the shared genes); with
    cell_subset "one_type", of its one-type cells (Fig 6C)."""
    data = folds.training_pseudobulk(pair, *unit, cell_subset=cell_subset)
    genes, chosen = shared(pair), panel(pair, unit)
    X = preprocess.log_normalize(data[:, genes].X)[:, pd.Index(genes).get_indexer(chosen)].toarray()
    return dict(zip(data.obs_names.astype(str), X))


def unit_cells_all_available(pair: str, unit: tuple, counts: bool = False) -> protocol.Cells:
    """The unit's training samples' cells from the all-available store (up to 3,000 per sample;
    stage 02b), one shard at a time, on the panel as log-CP10k over the shared genes (or raw counts)."""
    genes, chosen = shared(pair), panel(pair, unit)
    X, ids, sample, library = [], [], [], []
    for sample_id, path in sorted(all_available.training_shards(pair, *unit).items()):
        shard = all_available.load(path)
        x = sp.csr_matrix(shard[:, genes].X, dtype=np.float32) if counts else preprocess.log_normalize(shard[:, genes].X)
        X.append(x[:, pd.Index(genes).get_indexer(chosen)].tocsr())
        ids.extend(shard.obs_names.astype(str))
        sample.extend([sample_id] * shard.n_obs)
        library.append(np.asarray(shard.X.sum(axis=1)).ravel())   # stage 02's gene axis, as cell_qc's total_counts
    library = np.concatenate(library)
    samples = folds.training_samples(pair, *unit).set_index("sample_id")[["group", "holdout_group"]]
    return protocol.Cells(X=sp.vstack(X).tocsr(), genes=chosen, cell_id=np.array(ids), sample_id=np.array(sample),
                          cell_type=np.array(["unannotated"] * len(ids)), library_size=library.astype(float), samples=samples)


def pseudobulk_log_all_available(pair: str, unit: tuple) -> dict:
    """sample -> the log-CP10k pseudobulk of its all-available cells (their summed shard counts over the
    shared genes, then the panel): the gene scaler's p at row F's "all"."""
    genes, chosen = shared(pair), panel(pair, unit)
    out = {}
    for sample_id, path in sorted(all_available.training_shards(pair, *unit).items()):
        summed = preprocess.summed(all_available.load(path)[:, genes].X)[None, :]
        out[sample_id] = preprocess.log_normalize(summed)[:, pd.Index(genes).get_indexer(chosen)].toarray().ravel()
    return out


def scored_bulk(pair: str, unit: tuple) -> pd.DataFrame:
    """The unit's scored bulk samples (kept; label-free bulk_samples.csv rows) with their scoring unit."""
    bulk = pd.read_csv(settings.OUTPUTS / "02_preprocess" / pair / "bulk_samples.csv", dtype={"sample_id": str, "donor_id": str})
    table = folds.scored_bulk(pair, *unit, bulk[bulk["drop_reason"].isna()]).reset_index(drop=True)
    return table.assign(scoring_unit_id=table["donor_id" if PAIRS[pair]["scoring_unit"] == "donor" else "sample_id"])


def bulk_profiles(pair: str, unit: tuple, sample_ids, counts: bool = False) -> protocol.Profiles:
    """Bulk samples as one-profile inputs on the unit's panel: log-CP10k over the shared genes (the
    harness standardizes them with the training cells' moments), or raw counts on the panel (scVI)."""
    genes, chosen = shared(pair), panel(pair, unit)
    data = anndata.read_h5ad(settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad")[list(sample_ids)]
    X = data[:, genes].X
    X = (sp.csr_matrix(X) if counts else preprocess.log_normalize(X))[:, pd.Index(genes).get_indexer(chosen)].tocsr()
    return protocol.Profiles(X=X, genes=chosen, sample_id=np.asarray(sample_ids, dtype=str),
                             library_size=np.asarray(data.X.sum(axis=1)).ravel().astype(float))


def training_ages(pair: str, unit: tuple, sample_ids, also=(), source: pd.Series = None) -> dict:
    """sample -> its single-cell age (label_value) z-scored over the given (training-side) samples: the
    age-similarity loss's input (aging pairs only), and the MSE variant's target. also:
    further samples z-scored with the same moments (the MSE variant's validation side). source (row D:
    sample -> the holdout group whose label it took): each sample takes that group's mean age (sc2bulk.labels' blinded bulk permutation takes the earliest instead, stated in Methods)."""
    samples = folds.training_samples(pair, *unit).set_index("sample_id")
    age = samples["label_value"].astype(float)
    if source is not None:
        age = age.groupby(samples["holdout_group"]).mean().loc[source.loc[samples.index]].set_axis(samples.index)
    z = (age - age.loc[list(sample_ids)].mean()) / age.loc[list(sample_ids)].std(ddof=0)
    return z.loc[list(sample_ids) + list(also)].to_dict()


def permuted_groups(pair: str, unit: tuple, j: int) -> pd.DataFrame:
    """Row D's labels: the unit's holdout groups' 0/1 labels permuted within
    each side of encoder split j, so class counts per side are preserved (a group's samples move together;
    mouse organs: no organ stratum, a mouse's organs move together), drawn from `permutation` at
    PERMUTATION_OFFSET + j (unit-keyed). Single-cell labels only; stage 08 reads the same. Returns the unit's
    training samples (index sample_id): group (permuted) and source (the holdout group whose label it took)."""
    samples = folds.training_samples(pair, *unit).set_index("sample_id")
    samples = samples.assign(side=pd.Series(encoder_sides(pair, unit, j)).loc[samples.index])
    assert (samples.groupby("holdout_group")[["group", "side"]].nunique() == 1).all().all(), "a holdout group has two labels or sides"
    groups = samples.groupby("holdout_group")[["group", "side"]].first()
    rng = np.random.default_rng(seeds.derived("permutation", pair, T.PERMUTATION_OFFSET + j, *unit))
    source = {}
    for side in ("train", "validation"):
        members = groups.index[groups["side"] == side]
        source.update(zip(members, members[rng.permutation(len(members))]))
    taken = samples["holdout_group"].map(source)
    return pd.DataFrame({"group": groups.loc[taken, "group"].to_numpy(), "source": taken.to_numpy()}, index=samples.index)


def learning_curve_subset(pair: str, unit: tuple, e: int, fraction: float) -> dict:
    """Row E's split: encoder split e's sides, keeping within each side and 0/1 label
    the first ceil(fraction x n) of its n holdout groups in a random order from `learning_curve` e
    (unit-keyed; the order doesn't depend on the fraction, so 25% within 50% within 100%). Asserts >= 2
    validation groups per label. sample_id -> "train" / "validation" for the kept samples; stage 08 reads the same."""
    sides = encoder_sides(pair, unit, e)
    samples = folds.training_samples(pair, *unit)
    groups = samples.assign(side=samples["sample_id"].map(sides)).groupby("holdout_group")[["group", "side"]].first()
    rng = np.random.default_rng(seeds.derived("learning_curve", pair, e, *unit))
    kept = set()
    for side in ("train", "validation"):
        for label in (0, 1):
            members = groups.index[(groups["side"] == side) & (groups["group"] == label)]
            n = math.ceil(fraction * len(members))
            assert side == "train" or n >= 2, f"{pair} {unit} split {e}: {n} validation groups of label {label} at {fraction}"
            kept |= set(members[rng.permutation(len(members))][:n])
    holdout = dict(zip(samples["sample_id"], samples["holdout_group"]))
    return {s: side for s, side in sides.items() if holdout[s] in kept}


def sample_strata(pair: str, unit: tuple) -> pd.Series:
    """sample_id -> its stratum for the unit's training samples: the organ on mouse organs
    (training.ORGAN_STRATUM), else "all"."""
    samples = folds.training_samples(pair, *unit).set_index("sample_id")
    return samples[T.ORGAN_STRATUM[pair]].astype(str) if pair in T.ORGAN_STRATUM else pd.Series("all", index=samples.index)


def thinned_bulk(pair: str, unit: tuple, bulk_table: pd.DataFrame):
    """H3, the thinned-bulk sensitivity: each scored bulk sample
    as TEST_TIME_BAGS bags of FROZEN_NCELL pseudo-cells. Each pseudo-cell is a binomial thinning of the bulk
    sample's shared-gene counts to a library size drawn with replacement from the unit's training samples'
    stored cells' shared-gene totals (mouse organs: the bulk sample's organ's cells). Exact in distribution:
    per-gene binomials on the panel, one binomial for the rest of the shared genes. Seeded by spare_3 index 0
    (unit-keyed) and the bulk sample, so every model sees the same draws. Yields (sample_id, panel counts
    (bags, cells, genes), shared-gene totals (bags, cells)), one bulk sample at a time."""
    genes, chosen = shared(pair), panel(pair, unit)
    cells = folds.training_cells(pair, *unit)
    library = np.asarray(cells[:, genes].X.sum(axis=1)).ravel()
    cell_stratum = sample_strata(pair, unit).loc[cells.obs["sample_id"].astype(str)].to_numpy()
    del cells   # the full-gene cells aren't kept while the generator yields
    bulk_stratum = (bulk_table[T.ORGAN_STRATUM[pair]].astype(str) if pair in T.ORGAN_STRATUM
                    else pd.Series("all", index=bulk_table.index)).to_numpy()
    ids = list(bulk_table["sample_id"])
    data = sp.csr_matrix(anndata.read_h5ad(settings.OUTPUTS / "02_preprocess" / pair / "bulk.h5ad")[ids][:, genes].X)
    columns = pd.Index(genes).get_indexer(chosen)
    base = seeds.derived("spare_3", pair, 0, *unit)
    shape = (representations.TEST_TIME_BAGS, T.FROZEN_NCELL)
    for i, s in enumerate(ids):
        counts = data[i].toarray().ravel().astype(np.int64)
        rng = np.random.default_rng([base, preprocess.sample_seed(pair, s) % 2**32])
        p = np.minimum(1.0, rng.choice(library[cell_stratum == bulk_stratum[i]], shape) / counts.sum())
        on_panel = rng.binomial(counts[columns][None, None, :], p[..., None])
        yield s, on_panel, on_panel.sum(axis=2) + rng.binomial(counts.sum() - counts[columns].sum(), p)
