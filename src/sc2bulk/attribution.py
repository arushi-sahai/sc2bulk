"""Stage 10's computations: gene attribution of the read-out's phenotype prediction and of the modality
direction. Label-free: heads come from stage 08's heads files (single-cell side), nothing reads a bulk label.

One attributed quantity for every method: the read-out-averaged head's prediction for a single-cell training
sample, yhat(s) = b + w . f(s), w = mean_r coef_r / scale_r, in raw feature space per stratum
(affine_head). A_{s,g} is gradient x input with a zero baseline (pseudobulk: coefficient x value, the same
thing for a linear head; Geneformer: in-silico deletion, D = w . delta_e). Positive = gene g pushes the
prediction toward 1.

TorchInputs is a torch copy of harness.Inputs' transform (harness.py is hashed into stages 05 and 06 and is not
edited): it treats as constants everything fixed per split or sample (population moments, gene scaler, the
distance feature's class means and moments, the PCA basis, depth weights) and differentiates what is computed
from the bag itself (per_sample's bag mean and SD, the distances); capped entries get gradient 0.
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom, spearmanr

from sc2bulk import folds, preprocess, seeds
from sc2bulk.config import genes as G
from sc2bulk.config import training as T

# ---------------------------------------------------------------------------------------------- heads


def affine_head(rows: pd.DataFrame) -> dict:
    """{stratum: (w, b)} in raw feature space from stage 08 head rows (single-cell side) of one encoder seed and
    read-out: w = mean_r coef_r / scale_r, b = mean_r (intercept_r - coef_r . mean_r / scale_r), so w . f + b
    is the mean prediction over the read-out seeds given (one seed's rows give that seed's head)."""
    head = {}
    for stratum, part in rows.groupby("stratum", sort=True):
        w, b = [], []
        for _, one in part.groupby("readout_seed", sort=True):
            coef, mean, scale = (one[c].to_numpy(dtype=np.float64) for c in ("coef", "scaler_mean", "scaler_scale"))
            w.append(coef / scale)
            b.append(float(one["intercept"].iloc[0]) - float((coef * mean / scale).sum()))
        head[stratum] = (np.mean(w, axis=0), float(np.mean(b)))
    return head


def heads(table: pd.DataFrame, per_seed: bool = False) -> tuple:
    """({(encoder_seed, readout): head} (per_seed: keyed (encoder_seed, readout, readout_seed)), the features in
    order) from a stage 08 heads table; asserts every fit lists the same features in the same order."""
    table = table[table["side"] == "single_cell"]
    fits = list(table.groupby(["encoder_seed", "readout", "readout_seed", "stratum"], sort=True))
    dims = list(fits[0][1]["gene_or_dim"])
    assert all(list(part["gene_or_dim"]) == dims for _, part in fits), "stage 08's fits list different features"
    by = ["encoder_seed", "readout", "readout_seed"] if per_seed else ["encoder_seed", "readout"]
    out = {}
    for key, part in table.groupby(by, sort=True):
        out[(int(key[0]), key[1]) + ((int(key[2]),) if per_seed else ())] = affine_head(part)
    return out, dims


def direction(head: dict, strata) -> np.ndarray:
    """Each sample's head vector w (samples x features), by its stratum."""
    return np.vstack([head[s][0] for s in strata])


def predict(head: dict, X: np.ndarray, strata) -> np.ndarray:
    return np.array([head[s][0] @ x + head[s][1] for x, s in zip(X, strata)])


def modality_direction(X_train: np.ndarray, X_bulk: np.ndarray, strata_train, strata_bulk) -> dict:
    """{stratum: (u, 0)} (head-shaped): u = mean of the scored bulk's embeddings - mean of the single-cell
    training samples', unit length, in the raw embedding space (mouse organs per organ, Q9)."""
    strata_train, strata_bulk = np.asarray(strata_train), np.asarray(strata_bulk)
    out = {}
    for s in sorted(set(strata_train)):
        u = X_bulk[strata_bulk == s].mean(axis=0) - X_train[strata_train == s].mean(axis=0)
        out[s] = (u / np.linalg.norm(u), 0.0)
    return out


# ---------------------------------------------------------------------------------------------- samples and draws


def donor_halves(samples: pd.DataFrame, pair: str, unit: tuple) -> pd.Series:
    """sample_id -> "a" / "b": a 2-fold split of the unit's holdout groups, stratified by the 0/1 group
    (folds.deal; `attribution` index 2, unit-keyed; mouse organs: holdout group = mouse)."""
    group = samples.groupby("holdout_group")["group"].first()
    fold_of = folds.deal(group, 2, seeds.derived("attribution", pair, 2, *unit))
    return pd.Series(samples["holdout_group"].map(fold_of).map({0: "a", 1: "b"}).to_numpy(), index=samples["sample_id"].to_numpy())


def fm_cells(pair: str, sample_id: str, cell_ids, m: int = G.FM_CELLS) -> np.ndarray:
    """The sample's foundation-model cell subsample: the first m of a seeded permutation of its stored cells in
    sorted-id order (`attribution` index 0, pair-level, and the sample: the same cells in every fold, for scGPT
    and Geneformer). Halves: the first and second m/2."""
    rng = np.random.default_rng([seeds.derived("attribution", pair, 0), preprocess.sample_seed(pair, sample_id) % 2**32])
    return rng.permutation(np.sort(np.asarray(cell_ids)))[:m]


def screen_cells(pair: str, sample_id: str, cell_ids, main: np.ndarray, n: int) -> np.ndarray:
    """The two-stage screen's stage-1 cells (fallback only): n cells disjoint from the main subsample
    (`deletion` index 10 and the sample)."""
    rng = np.random.default_rng([seeds.derived("deletion", pair, 10), preprocess.sample_seed(pair, sample_id) % 2**32])
    rest = np.setdiff1d(np.sort(np.asarray(cell_ids)), main)
    return rng.permutation(rest)[:n]


def deletion_samples(pair: str, samples: pd.DataFrame, per_donor: int, cross_fitted: bool) -> list:
    """Geneformer's explained samples among a unit's training samples. Cross-fitted
    pair: per_donor of each donor's samples, the first of a donor-keyed permutation (`attribution` index 1,
    pair-level, so a donor's samples are the same in each of its training folds and 1 is nested in 2). Other
    pairs: all, capped at DELETION_CAP by a round robin over holdout groups within each 0/1 group (group shares
    proportional to their sample counts)."""
    base = seeds.derived("attribution", pair, 1)
    if cross_fitted:
        out = []
        for donor, part in samples.groupby("donor_id", sort=True):
            rng = np.random.default_rng([base, preprocess.sample_seed(pair, donor) % 2**32])
            out += list(rng.permutation(np.sort(part["sample_id"].to_numpy()))[:per_donor])
        return sorted(out)
    if len(samples) <= G.DELETION_CAP:
        return sorted(samples["sample_id"])
    rng = np.random.default_rng(base)
    sizes = samples["group"].value_counts().sort_index()
    quota = {0: round(G.DELETION_CAP * sizes[0] / len(samples))}
    quota[1] = G.DELETION_CAP - quota[0]
    out = []
    for g in (0, 1):
        part = samples[samples["group"] == g]
        holdout = rng.permutation(np.sort(part["holdout_group"].unique()))
        queues = [list(rng.permutation(np.sort(part.loc[part["holdout_group"] == h, "sample_id"].to_numpy()))) for h in holdout]
        order = [q[i] for i in range(max(map(len, queues))) for q in queues if i < len(q)]
        out += order[:quota[g]]
    return sorted(out)


def scvi_weights(bag_list: list, cell_ids) -> np.ndarray:
    """Each cell's weight in a sample's embedding (the mean over its test bags of each bag's mean latent):
    (1/n_bags) sum_b k_b / |bag_b|, k_b = the cell's count in bag b (06_train.bag_latent)."""
    index = pd.Index(cell_ids)
    w = np.zeros(len(index))
    for b in bag_list:
        np.add.at(w, index.get_indexer(b), 1.0 / len(b))
    return w / len(bag_list)


# ---------------------------------------------------------------------------------------------- the set encoders' inputs


class TorchInputs:
    """harness.Inputs' transform in torch, from a fitted Inputs (see the module docstring). Inputs are float64
    tensors; the network gets float32, as stage 06's (float64 transform, float32 storage)."""

    def __init__(self, inputs, device="cpu"):
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), device=device)
        self.level, self.scale, self.distance, self.pca = inputs.level, inputs.scale, inputs.distance, inputs.pca
        self.mean, self.sd = t(inputs.mean), t(inputs.sd)
        self.scalers = {s: t(v) for s, v in inputs.scalers.items()}
        if self.distance:
            self.group_means, self.d_mean, self.d_sd = t(inputs.group_means), t(inputs.d_mean), t(inputs.d_sd)
        if self.pca:
            self.pca_mean, self.components = t(inputs.pca_mean), t(inputs.components)
        self.device = device

    def _distances(self, x):
        return torch.sqrt(((x[:, :, None, :] - self.group_means) ** 2).sum(dim=3))

    def _finish(self, z, x):
        if self.distance:
            z = torch.cat([z, (self._distances(x) - self.d_mean) / self.d_sd], dim=2)
        return ((z - self.pca_mean) @ self.components if self.pca else z).float()

    def cells(self, x: torch.Tensor, sample: str) -> torch.Tensor:
        """Bags of one sample's cells (bags x cells x genes, log-CP10k) -> the network input (Inputs.cells)."""
        x = x * self.scalers[sample] if self.scale else x
        if self.level == "none":
            z = x
        elif self.level == "per_sample" and x.shape[1] >= 2:   # the bag on itself
            sd = x.std(dim=1, unbiased=False, keepdim=True)
            z = ((x - x.mean(dim=1, keepdim=True)) / torch.where(sd > 0, sd, torch.ones_like(sd))).clamp(-T.INPUT_CAP, T.INPUT_CAP)
        else:
            z = ((x - self.mean) / self.sd).clamp(-T.INPUT_CAP, T.INPUT_CAP)
        return self._finish(z, x)

    def profile(self, x: torch.Tensor) -> torch.Tensor:
        """One-profile inputs (profiles x genes) as 1-cell bags: no scaler, population moments (Inputs.profile)."""
        x = x[:, None, :]
        z = x if self.level == "none" else ((x - self.mean) / self.sd).clamp(-T.INPUT_CAP, T.INPUT_CAP)
        return self._finish(z, x)


def depth_weights(library_size: np.ndarray, device="cpu") -> torch.Tensor:
    """A bag's depth weights (bags x cells): log1p(library size), normalized within each bag (harness.Resident)."""
    w = torch.as_tensor(np.log1p(library_size), dtype=torch.float32, device=device)
    return w / w.sum(dim=1, keepdim=True)


def gradient_x_input(embed, x: np.ndarray, targets: dict, device="cpu") -> tuple:
    """Gradient x input of each target's v . mean_b embed(x)_b over a sample's bags (x: bags x cells x genes,
    float64; embed: tensor -> (bags, d) embeddings). Returns ({name: per-gene attribution (sum over cells,
    mean over bags)}, {name: v . (mean_b e(x)_b - mean_b e(0)_b)}, the mean embedding): the completeness check's
    two sides are sum_g A_g and the second."""
    xt = torch.tensor(x, dtype=torch.float64, device=device, requires_grad=True)
    e = embed(xt)
    with torch.no_grad():
        zero = embed(torch.zeros_like(xt)).mean(dim=0)
    attribution, delta = {}, {}
    for name, v in targets.items():
        v = torch.as_tensor(v, dtype=torch.float32, device=device)
        grad, = torch.autograd.grad((e @ v).mean(), xt, retain_graph=True)
        attribution[name] = (grad * xt).detach().sum(dim=(0, 1)).cpu().numpy()
        delta[name] = float(((e.detach().mean(dim=0) - zero) @ v).cpu())
    return attribution, delta, e.detach().mean(dim=0).cpu().numpy()


def scvi_gradient_x_input(module, counts: np.ndarray, weight: np.ndarray, targets: dict, device="cpu") -> tuple:
    """scVI: gradient x input of each target's v . sum_c weight_c z_c with respect to
    the encoder's own input log1p(counts) (z_c: the encoder's latent mean; no sampling). Returns as
    gradient_x_input (the embedding is sum_c weight_c z_c)."""
    assert module.log_variational and not module.encode_covariates, "scVI's encoder input isn't log1p(counts) alone"
    y = torch.log1p(torch.tensor(counts, dtype=torch.float32, device=device)).requires_grad_()
    w = torch.as_tensor(weight, dtype=torch.float32, device=device)
    z = module.z_encoder(y)[0].loc
    with torch.no_grad():
        zero = module.z_encoder(torch.zeros_like(y))[0].loc
    attribution, delta = {}, {}
    for name, v in targets.items():
        v = torch.as_tensor(v, dtype=torch.float32, device=device)
        grad, = torch.autograd.grad(((z @ v) * w).sum(), y, retain_graph=True)
        attribution[name] = (grad * y).detach().sum(dim=0).cpu().numpy()
        delta[name] = float((((z.detach() - zero) @ v) * w).sum().cpu())
    return attribution, delta, (w[:, None] * z.detach()).sum(dim=0).cpu().numpy()


def scgpt_gradient_x_input(encode, genes: torch.Tensor, expr: torch.Tensor, pad_id: int, targets: dict, weight: np.ndarray,
                           autocast: bool) -> tuple:
    """scGPT: gradient x input with respect to the continuous value input (the bin values; bin 0 =
    unexpressed, the zero baseline) of each target's sum_c weight_c v . e_c, e_c = the <cls> output
    L2-normalized. encode(genes, expr, mask) -> (n, length, d) (the model's _encode). genes/expr: (profiles x
    tokens), position 0 = <cls>. Returns ({name: (profiles x tokens) attribution, 0 at <cls> and padding},
    {name: completeness delta}, the profiles' embeddings)."""
    mask = genes.eq(pad_id)
    expr = expr.clone().requires_grad_()
    zero_expr = torch.where(torch.arange(expr.shape[1], device=expr.device)[None, :] == 0, expr.detach(), torch.zeros_like(expr))

    def embedding(values):
        with torch.autocast("cuda", enabled=autocast and genes.is_cuda):
            out = encode(genes, values, mask)[:, 0, :].float()
        return out / out.norm(dim=1, keepdim=True)

    e = embedding(expr)
    with torch.no_grad():
        zero = embedding(zero_expr)
    w = torch.as_tensor(weight, dtype=torch.float32, device=expr.device)
    keep = (~mask).float()
    keep[:, 0] = 0.0
    attribution, delta = {}, {}
    for name, v in targets.items():
        v = torch.as_tensor(v, dtype=torch.float32, device=expr.device)
        grad, = torch.autograd.grad(((e @ v) * w).sum(), expr, retain_graph=True)
        attribution[name] = (grad * expr * keep).detach().cpu().numpy()
        delta[name] = float((((e.detach() - zero) @ v) * w).sum().cpu())
    return attribution, delta, e.detach().cpu().numpy()


# ---------------------------------------------------------------------------------------------- Geneformer deletion


def geneformer_inputs(encode_one, row: np.ndarray, tokens, medians) -> tuple:
    """One profile's deletion inputs: the unperturbed token ids, and for every gene in
    its input (expressed and within the input size) the ids of the profile with that gene's count set to 0,
    re-encoded by stage 07's own encoder (encode_one = fm.Geneformer._encode: rank, truncate, backfill).
    Returns (ids, [(column, ids without it)], {column: 1-based rank}, genes truncated)."""
    ids, truncated = encode_one(row, tokens, medians)
    column = {int(t): j for j, t in enumerate(tokens)}
    present = [column[int(t)] for t in ids[1:-1]]
    variants = []
    for j in present:
        without = row.copy()
        without[j] = 0
        variants.append((j, encode_one(without, tokens, medians)[0]))
    return ids, variants, {j: r for r, j in enumerate(present, start=1)}, truncated


def deletion_deltas(embed_ids, ids, variants, batch: int) -> tuple:
    """e(ids) and {column: e(ids) - e(ids without the gene)} (one input; variants share a length, so batches
    need no padding). embed_ids(list of id arrays of one length) -> (n, d) float64."""
    e0 = embed_ids([ids])[0]
    out = {}
    for start in range(0, len(variants), batch):
        part = variants[start:start + batch]
        lengths = {len(v) for _, v in part}
        assert len(lengths) == 1, "deletion variants of one input differ in length"
        e = embed_ids([v for _, v in part])
        out.update({j: e0 - row for (j, _), row in zip(part, e)})
    return e0, out


def mean_deltas(per_input: list, n_genes: int, d: int) -> np.ndarray:
    """The mean over inputs (cells) of their per-gene embedding changes, inputs where a gene is absent counting
    0 (never average over expressing cells only): (n_genes x d)."""
    out = np.zeros((n_genes, d))
    for deltas in per_input:
        for j, v in deltas.items():
            out[j] += v
    return out / len(per_input)


def joint_deltas(embed_ids, encode_one, rows: list, e0: list, tokens, medians, sets: dict, batch: int) -> dict:
    """{set name: the mean over the inputs (rows, with their unperturbed embeddings e0) of e(x) - e(x with every
    gene of the set deleted)} (sets: name -> token columns). Variants are batched by length."""
    out = {name: 0.0 for name in sets}
    for row, base in zip(rows, e0):
        variants = []
        for name, columns in sets.items():
            without = row.copy()
            without[list(columns)] = 0
            variants.append((name, encode_one(without, tokens, medians)[0]))
        by_length = {}
        for name, ids in variants:
            by_length.setdefault(len(ids), []).append((name, ids))
        for part in by_length.values():
            for start in range(0, len(part), batch):
                chunk = part[start:start + batch]
                for (name, _), e in zip(chunk, embed_ids([ids for _, ids in chunk])):
                    out[name] = out[name] + (base - e) / len(rows)
    return out


def overhead(length) -> np.ndarray:
    points, factors = zip(*G.ATTENTION_OVERHEAD)
    return np.interp(length, points, factors)


def deletion_tokens(lengths) -> float:
    """Tokens processed to delete every gene of inputs of these lengths (L passes of length about L), with the
    attention overhead."""
    L = np.asarray(lengths, dtype=float)
    return float((L * L * overhead(L)).sum())


def fallback(budget_days: dict, precision: str) -> dict:
    """The first step of GF_FALLBACK_ORDER whose budget (days, from budget_days[step]) is within the pilot's
    limit, fp32_only steps only under fp32; else the triage step."""
    for step in G.GF_FALLBACK_ORDER:
        if step["step"] == "triage":
            return step
        if step["fp32_only"] and precision != "fp32":
            continue
        if budget_days[step["step"]] <= G.PILOT["budget_days"]:
            return step
    raise AssertionError("GF_FALLBACK_ORDER must end with triage")


# ---------------------------------------------------------------------------------------------- aggregates and rankings


def summarize(A: np.ndarray, present: np.ndarray, halves: np.ndarray, genes) -> pd.DataFrame:
    """Per gene over the explained samples (rows of A, signed attributions): attribution (= mean |A|),
    mean_signed, mean_abs_half_a/b (donor halves), n_present."""
    halves = np.asarray(halves)
    return pd.DataFrame({"gene": list(genes), "attribution": np.abs(A).mean(axis=0), "mean_signed": A.mean(axis=0),
                         "mean_abs_half_a": np.abs(A[halves == "a"]).mean(axis=0), "mean_abs_half_b": np.abs(A[halves == "b"]).mean(axis=0),
                         "n_present": present.sum(axis=0).astype(int)})


def ranks(score: pd.Series) -> pd.Series:
    """1-based ranks of genes (the index) by descending score, ties by symbol."""
    order = sorted(score.index, key=lambda g: (-score[g], g))
    return pd.Series(np.arange(1, len(order) + 1), index=order).loc[score.index]


def top(score: pd.Series, k: int) -> list:
    return sorted(score.index, key=lambda g: (-score[g], g))[:k]


def overlap(a, b) -> float:
    return len(set(a) & set(b)) / len(a)


def mean_pairwise_overlap(scores: list, k: int) -> float:
    tops = [top(s, k) for s in scores]
    return float(np.mean([overlap(tops[i], tops[j]) for i in range(len(tops)) for j in range(i + 1, len(tops))]))


def across(per_fold: dict) -> pd.Series:
    """The ranking score over folds x seeds: per_fold {fold: [per-seed score Series]}; within a
    fold the mean over its seeds, then the mean over folds (each fold weighted equally), on the union of the
    folds' genes with a gene absent from a fold's panel counting 0 there."""
    genes = sorted(set().union(*(s.index for scores in per_fold.values() for s in scores)))
    return pd.concat([pd.concat([s.reindex(genes, fill_value=0.0) for s in scores], axis=1).mean(axis=1)
                      for scores in per_fold.values()], axis=1).mean(axis=1)


def completeness(sum_attribution, delta_prediction) -> dict:
    """Spearman across samples and the median ratio of sum_g A_g to yhat(x) - yhat(0) (reported, not gated)."""
    a, d = np.asarray(sum_attribution), np.asarray(delta_prediction)
    return {"completeness_spearman": float(spearmanr(a, d)[0]), "completeness_median_ratio": float(np.median(a / d))}


# ---------------------------------------------------------------------------------------------- checks


def housekeeping(top_genes, attributable, housekeeping_genes: set, symbols: dict, pseudobulk_top=None) -> dict:
    """The top-k housekeeping fraction, its base rate among the attributable genes, the one-sided
    hypergeometric p, the excess over pseudobulk's top-k fraction, the RPL/RPS/MT- fraction. All genes are panel
    genes; housekeeping_genes: the panel genes matched to the list on human Ensembl gene ids;
    symbols: panel gene -> human symbol, used only for the RPL/RPS/MT- fraction."""
    top_genes, attributable = list(top_genes), list(attributable)
    k, n_hk = len(top_genes), sum(g in housekeeping_genes for g in attributable)
    hits = sum(g in housekeeping_genes for g in top_genes)
    out = {"hk_fraction": hits / k, "hk_base_rate": n_hk / len(attributable),
           "hk_p": float(hypergeom.sf(hits - 1, len(attributable), n_hk, k)),
           "rp_mt_fraction": sum(symbols[g].startswith(("RPL", "RPS", "MT-")) for g in top_genes) / k}
    if pseudobulk_top is not None:
        out["hk_excess_vs_pseudobulk"] = out["hk_fraction"] - sum(g in housekeeping_genes for g in pseudobulk_top) / len(pseudobulk_top)
    return out


def matching_bins(frac: pd.Series, mean_rank: pd.Series, n: int) -> pd.Series:
    """Each gene's cell of the n x n grid of (fraction of cells containing it) x (mean rank when present), quantile
    edges over the genes given, ties broken in panel order."""
    a = pd.qcut(frac.rank(method="first"), n, labels=False)
    b = pd.qcut(mean_rank.rank(method="first"), n, labels=False)
    return a * n + b


def matching_grid(frac: pd.Series, mean_rank: pd.Series, top_genes) -> tuple:
    """(n, bins): the finest grid of GF_MATCH_BINS in which every cell holding top genes holds >= GF_MATCH_POOL x as
    many other genes. top_genes must all be in frac's index. No grid down to 2 x 2 stops the stage
    (not expected above ~600 genes; 1 x 1, no matching, was left out)."""
    top_genes = list(top_genes)
    for n in G.GF_MATCH_BINS:
        bins = matching_bins(frac, mean_rank, n)
        need = bins.loc[top_genes].value_counts()
        pool = bins.drop(top_genes).value_counts().reindex(need.index, fill_value=0)
        if (pool >= G.GF_MATCH_POOL * need).all():
            return n, bins
    raise AssertionError(f"no matching grid in {G.GF_MATCH_BINS} holds {G.GF_MATCH_POOL}x the top genes in every cell")


def matched_random_sets(bins: pd.Series, top_genes, n_sets: int, rng) -> list:
    """n_sets random gene sets drawn to the top genes' bin counts from the genes outside them."""
    pool = bins.drop(list(top_genes))
    need = bins.loc[list(top_genes)].value_counts().sort_index()
    out = []
    for _ in range(n_sets):
        chosen = []
        for b, n in need.items():
            candidates = np.sort(pool.index[pool == b].to_numpy())
            assert len(candidates) >= n, f"matching bin {b}: {len(candidates)} genes for {n}"
            chosen += list(rng.choice(candidates, n, replace=False))
        out.append(chosen)
    return out


def level_difference(X_bulk: np.ndarray, X_sc: np.ndarray, bulk_strata, sc_strata, genes, markers: set) -> pd.DataFrame:
    """Fig 6B's level difference, label-free: per gene, delta = mean scored-bulk
    log-CP10k - mean training-pseudobulk log-CP10k and smd = delta / pooled SD (ddof 1; 0 where the SD is 0),
    within each stratum then averaged (mouse organs); large = the top LARGE_DIFFERENCE_FRACTION of genes by
    |smd| (ties by symbol); marker = in the given marker list."""
    bulk_strata, sc_strata = np.asarray(bulk_strata), np.asarray(sc_strata)
    deltas, smds = [], []
    for s in sorted(set(sc_strata)):
        a, b = X_bulk[bulk_strata == s], X_sc[sc_strata == s]
        delta = a.mean(axis=0) - b.mean(axis=0)
        pooled = np.sqrt(((len(a) - 1) * a.var(axis=0, ddof=1) + (len(b) - 1) * b.var(axis=0, ddof=1)) / (len(a) + len(b) - 2))
        deltas.append(delta)
        smds.append(np.divide(delta, pooled, out=np.zeros_like(delta), where=pooled > 0))
    table = pd.DataFrame({"gene": list(genes), "delta": np.mean(deltas, axis=0), "smd": np.mean(smds, axis=0)})
    large = top(table.set_index("gene")["smd"].abs(), int(round(G.LARGE_DIFFERENCE_FRACTION * len(table))))
    return table.assign(large=table["gene"].isin(large), marker=table["gene"].isin(markers))


def additivity(joint: np.ndarray, summed: np.ndarray) -> float:
    """The slope through the origin of the joint deletion effect on the sum of single deletions."""
    return float((joint * summed).sum() / (summed ** 2).sum())
