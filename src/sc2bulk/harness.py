"""The training harness: the one training path every set encoder uses. Encoders differ only in the network (sc2bulk.encoders);
inputs, normalization, bags, the objective, early stopping, checkpoints, "did it train?" and the
embedding rule are shared here.

A network exposes embed(x, weights) -> (bags, d) and head(embedding) -> (bags,) logits, where x is a
dense (bags, cells, features) tensor of one bag size and weights the cells' depth weights.
Every statistic is fit on the split's training side (Inputs); bags come from sc2bulk.bags and depend
only on the pair, unit and seeds, never on the encoder.
"""
import copy
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from sc2bulk import bags, protocol
from sc2bulk.config import representations
from sc2bulk.config import training as T

LN2 = float(np.log(2.0))


def gene_scaler(cells_log: np.ndarray, pseudobulk_log: np.ndarray) -> np.ndarray:
    """Per gene s = p / q (p: the sample's log-CP10k pseudobulk; q: the mean of its cells' log-CP10k);
    1 where q < GENE_SCALER_MIN_Q; clamped to [0, GENE_SCALER_MAX]."""
    q = cells_log.mean(axis=0)
    s = np.divide(pseudobulk_log, q, out=np.ones_like(q), where=q >= T.GENE_SCALER_MIN_Q)
    return np.clip(s, 0.0, T.GENE_SCALER_MAX)


class Inputs:
    """The input transform of one training, fit on its training side: the optional pseudobulk gene
    scaler, normalization (none / population / per_sample; ±INPUT_CAP for the standardizing levels),
    and the optional population-distance feature (appended after normalization, standardized with
    the training cells' distance moments). Statistics are accumulated one sample at a time, so no
    dense all-cell matrix is built. pseudobulk_log: sample -> its log-CP10k pseudobulk on the panel
    (the gene scaler's p), for every sample the training will see."""

    def __init__(self, config: dict, train: protocol.Cells, pseudobulk_log: dict):
        self.level = config["cell_normalization"]
        self.scale = bool(config.get("pseudobulk_gene_scaler")) and self.level != "per_sample"
        self.distance = bool(config.get("population_distance_feature"))
        self.pseudobulk_log = pseudobulk_log
        self.scalers = {}
        self.prepare(train)
        names = np.unique(train.sample_id)
        total, total_sq, n = 0.0, 0.0, 0
        class_sum, class_n = {0: 0.0, 1: 0}, {0: 0, 1: 0}
        for s in names:   # one sample's dense block at a time, float64 sums
            x = self._scaled(train, s).astype(np.float64)
            total, total_sq, n = total + x.sum(axis=0), total_sq + (x ** 2).sum(axis=0), n + len(x)
            g = int(train.samples.loc[s, "group"])
            class_sum[g], class_n[g] = class_sum[g] + x.sum(axis=0), class_n[g] + len(x)
        mean = total / n
        sd = np.sqrt(np.maximum(total_sq / n - mean ** 2, 0.0))
        self.mean, self.sd = mean, np.where(sd > 0, sd, 1.0)
        if self.distance:
            self.group_means = np.vstack([class_sum[c] / class_n[c] for c in (0, 1)])
            d_sum, d_sq = 0.0, 0.0
            for s in names:
                d = self._distances(self._scaled(train, s).astype(np.float64))
                d_sum, d_sq = d_sum + d.sum(axis=0), d_sq + (d ** 2).sum(axis=0)
            d_mean = d_sum / n
            d_sd = np.sqrt(np.maximum(d_sq / n - d_mean ** 2, 0.0))
            self.d_mean, self.d_sd = d_mean, np.where(d_sd > 0, d_sd, 1.0)
        self.pca = bool(config.get("pca"))
        if self.pca:   # ScRAT's PCA input
            assert self.level != "per_sample" and not self.distance and not self.scale, \
                "ScRAT's PCA input is fit on the none/population route only, without CytoSet's additions"
            first, second = 0.0, 0.0
            for s in names:   # the exact float64 covariance of the transformed training cells
                z = self._standardize(self._scaled(train, s).astype(np.float64), own=False)
                first, second = first + z.sum(axis=0), second + z.T @ z
            self.pca_mean = first / n
            values, vectors = np.linalg.eigh(second / n - np.outer(self.pca_mean, self.pca_mean))
            top = vectors[:, ::-1][:, :T.SCRAT_PCA_COMPONENTS]
            sign = np.sign(top[np.abs(top).argmax(axis=0), np.arange(top.shape[1])])
            self.components = top * sign   # each component's largest-|loading| entry positive
            assert self.components.shape[1] == T.SCRAT_PCA_COMPONENTS, "fewer genes than PCA components"

    def project(self, x: np.ndarray) -> np.ndarray:
        """Transformed inputs onto the training cells' principal components (when pca)."""
        return (x - self.pca_mean) @ self.components if self.pca else x

    def prepare(self, cells: protocol.Cells):
        """The gene scaler of every sample in `cells`, from its own cells and pseudobulk (label-free)."""
        if self.scale:
            for s in np.unique(cells.sample_id):
                self.scalers[s] = gene_scaler(cells.X[cells.sample_id == s].toarray(), self.pseudobulk_log[s])

    def _scaled(self, cells, sample) -> np.ndarray:
        x = cells.X[cells.sample_id == sample].toarray()
        return x * self.scalers[sample] if self.scale else x

    def _distances(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(((x[:, None, :] - self.group_means[None]) ** 2).sum(axis=2))

    def _standardize(self, x: np.ndarray, own: bool) -> np.ndarray:
        if self.level == "none":
            return x
        if own:   # per_sample, k >= 2: the bag on itself, moments in float64 (float32 sums along the cell axis drift with k)
            x = x.astype(np.float64)
            m, s = x.mean(axis=0), x.std(axis=0)
            x = (x - m) / np.where(s > 0, s, 1.0)
        else:
            x = (x - self.mean) / self.sd
        return np.clip(x, -T.INPUT_CAP, T.INPUT_CAP)

    def cells(self, x: np.ndarray, sample: str) -> np.ndarray:
        """A bag of a sample's cells (cells x genes, log-CP10k); the sample must have been prepared."""
        return self.transform(x * self.scalers[sample] if self.scale else x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """A bag of cells after the gene scaler (or none: H3's bulk pseudo-cells, as bulk takes none)."""
        out = self._standardize(x, own=self.level == "per_sample" and len(x) >= 2)
        if self.distance:
            out = np.hstack([out, (self._distances(x) - self.d_mean) / self.d_sd])
        return self.project(out)

    def profile(self, x: np.ndarray) -> np.ndarray:
        """A one-profile input (bulk, pseudobulk) as a 1-cell bag: no scaler, population moments."""
        x = x[None, :]
        out = self._standardize(x, own=False)
        if self.distance:
            out = np.hstack([out, (self._distances(x) - self.d_mean) / self.d_sd])
        return self.project(out)


def input_width(config: dict, genes: int) -> int:
    """The network's input width: the panel's genes, plus the population-distance feature's 2 columns
    (ScRAT's PCA width is set in its adapter)."""
    return genes + (2 if config.get("population_distance_feature") else 0)


def fits_on_device(*cells: protocol.Cells) -> bool:
    """Whether these Cells' dense inputs together stay within RESIDENT_MAX_BYTES (one training's
    training and validation sides are decided together; above it, both use the host gather)."""
    return sum(c.X.shape[0] * (c.X.shape[1] + 2) * 4 for c in cells) <= T.RESIDENT_MAX_BYTES


class Resident:
    """One Cells' transformed cells held on the device, so bags are gathered by row index there
    (a pure speedup). The per-cell
    transforms (gene scaler, none/population standardization, the population-distance feature) are
    applied once, one sample block at a time in float64, and stored as float32; per_sample
    standardization, which depends on the bag, is applied to each gathered bag on the device. Cells'
    depth weights are log1p(library size), normalized within each bag."""

    def __init__(self, cells: protocol.Cells, inputs: Inputs, device, on_device: bool = True):
        inputs.prepare(cells)
        self.cells, self.inputs, self.device = cells, inputs, device
        self.per_sample = inputs.level == "per_sample"
        genes = cells.X.shape[1]
        self.genes = genes
        self.on_device = on_device
        self.logw = torch.as_tensor(np.log1p(cells.library_size), dtype=torch.float32, device=device)
        if not self.on_device:   # too large (the 3,000-cell store): transform each gathered bag on the host
            return
        width = inputs.components.shape[1] if inputs.pca else genes + (2 if inputs.distance else 0)
        self.X = torch.empty((cells.X.shape[0], width), dtype=torch.float32, device=device)
        for s in np.unique(cells.sample_id):
            rows = np.flatnonzero(cells.sample_id == s)
            x = cells.X[rows].toarray().astype(np.float64)
            if inputs.scale:
                x = x * inputs.scalers[s]
            block = x if self.per_sample or inputs.level == "none" else inputs._standardize(x, own=False)
            if inputs.distance:
                block = np.hstack([block, (inputs._distances(x) - inputs.d_mean) / inputs.d_sd])
            self.X[torch.as_tensor(rows, device=device)] = torch.as_tensor(inputs.project(block), dtype=torch.float32, device=device)
        self.mean = torch.as_tensor(inputs.mean, dtype=torch.float32, device=device)
        self.sd = torch.as_tensor(inputs.sd, dtype=torch.float32, device=device)

    def indices(self, bag_list) -> dict:
        """Bags grouped by size: {size: (row-index tensor (bags, size), positions)}."""
        groups = {}
        for position, (_, ids) in enumerate(bag_list):
            groups.setdefault(len(ids), []).append((self.cells.row_of(ids), position))
        return {n: (torch.as_tensor(np.stack([g[0] for g in part]), device=self.device), np.array([g[1] for g in part]))
                for n, part in groups.items()}

    def gather(self, idx: torch.Tensor) -> tuple:
        """(x (bags, size, features), weights (bags, size)) for a row-index tensor of one bag size."""
        w = self.logw[idx]
        if not self.on_device:
            rows = idx.cpu().numpy()
            sample = self.cells.sample_id[rows[:, 0]]
            x = np.stack([self.inputs.cells(self.cells.X[r].toarray(), s) for r, s in zip(rows, sample)])
            return torch.as_tensor(x, dtype=torch.float32, device=self.device), w / w.sum(dim=1, keepdim=True)
        x = self.X[idx]
        if self.per_sample:
            g = x[:, :, :self.genes]
            if idx.shape[1] >= 2:   # the bag on itself
                sd = g.std(dim=1, unbiased=False, keepdim=True)
                g = (g - g.mean(dim=1, keepdim=True)) / torch.where(sd > 0, sd, torch.ones_like(sd))
            else:                   # a 1-cell bag: the training-population moments
                g = (g - self.mean) / self.sd
            x = torch.cat([g.clamp(-T.INPUT_CAP, T.INPUT_CAP), x[:, :, self.genes:]], dim=2)
        return x, w / w.sum(dim=1, keepdim=True)


class Precomputed(Resident):
    """Cells whose inputs are already transformed (ScRAT's mixup samples, in PCA space; stage 06 row G):
    cells.X (cells x features, dense) held on the device as is."""

    def __init__(self, cells: protocol.Cells, inputs: Inputs, device):
        self.cells, self.inputs, self.device = cells, inputs, device
        self.per_sample, self.on_device = False, True
        self.logw = torch.as_tensor(np.log1p(cells.library_size), dtype=torch.float32, device=device)
        self.X = torch.as_tensor(cells.X, dtype=torch.float32, device=device)


def forward(net, groups: dict, n_bags: int, resident: Resident, embed_only=False):
    """Embeddings (and logits) of bags of mixed sizes, from Resident.indices(): one forward pass per
    size group, in the original bag order (exact; no padding)."""
    out_e, out_l = [None] * n_bags, [None] * n_bags
    for idx, positions in groups.values():
        x, w = resident.gather(idx)
        e = net.embed(x, w)
        logits = None if embed_only else net.head(e)
        for k, p in enumerate(positions):
            out_e[p] = e[k]
            if logits is not None:
                out_l[p] = logits[k]
    return torch.stack(out_e), (None if embed_only else torch.stack(out_l))


def forward_bags(net, bag_list, resident: Resident, embed_only=False):
    return forward(net, resident.indices(bag_list), len(bag_list), resident, embed_only)


def age_loss(e: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
    """The age-similarity penalty: Gaussian-weighted squared embedding distances of
    bags with similar ages (z-scored on the training samples), diagonal excluded."""
    diff = (ages[:, None] - ages[None, :]).abs()
    k = torch.exp(-0.5 * (diff / T.AGE_LOSS_SIGMA) ** 2) * (1.0 - torch.eye(len(ages), device=e.device))
    return (k * torch.cdist(e, e).pow(2)).sum() / k.sum().clamp_min(1e-8)


def validation_logits(net, chunks, resident: Resident) -> np.ndarray:
    """Per-bag validation logits (eval mode) from the fixed validation bags' cached index chunks
    (each one micro-batch: [(groups, n_bags)])."""
    net.eval()
    with torch.no_grad():
        out = [forward(net, groups, n, resident)[1].cpu().numpy() for groups, n in chunks]
    net.train()
    return np.concatenate(out).astype(np.float64)


def bce(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-bag BCE of logits against 0/1 labels (numerically stable)."""
    return np.logaddexp(0.0, logits) - y * logits


def balanced(per_bag: np.ndarray, val_bags, samples) -> float:
    """Class-balanced validation BCE: the mean of the two classes' per-sample means."""
    per_sample = {}
    for loss, (s, _) in zip(per_bag, val_bags):
        per_sample.setdefault(s, []).append(loss)
    means = {s: np.mean(v) for s, v in per_sample.items()}
    return float(np.mean([np.mean([m for s, m in means.items() if samples.loc[s, "group"] == g]) for g in (0, 1)]))


def did_it_train(logits: np.ndarray, val_bags, samples, seed: int) -> tuple:
    """One seed's check: (passed, d, d*, T*, p, testable, reason). d = the class-balanced
    mean over validation holdout groups of ln 2 - the group's mean BCE; d* = d's maximum over the
    temperatures training.TEMPERATURES (logits / T; T* the maximizing one; d* >= d), the collapse
    screen's statistic, which doesn't depend on the logits' scale. The null keeps the raw logits and
    permutes the 0/1 labels across holdout groups (class counts preserved). With at most
    NULL_PERMUTATIONS labellings, all are enumerated (the observed one included) and
    p = #{null >= d} / N; otherwise NULL_PERMUTATIONS seeded draws and p = (1 + #{null >= d}) / (1 + N)
    (Phipson & Smyth 2010). The split is testable with at least MIN_TESTABLE labellings. The seed
    passes if d* > MARGIN_FLOOR x ln 2 (the collapse screen) and, on a testable split, p <= NULL_ALPHA.
    Search eligibility and the configuration label combine seeds (stage 05/06)."""
    from itertools import combinations
    from math import comb
    group_of = np.array([samples.loc[s, "holdout_group"] for s, _ in val_bags])
    names = np.array(sorted(set(group_of)))
    label = samples.groupby("holdout_group")["group"].first().loc[names].to_numpy()
    n1 = int(label.sum())
    assert min(n1, len(names) - n1) >= 2, "did it train? needs >= 2 validation holdout groups per class"
    member = np.searchsorted(names, group_of)

    def group_bce(z):   # each group's mean BCE if labelled 0 and if labelled 1
        return (np.bincount(member, weights=bce(z, np.zeros_like(z))) / np.bincount(member),
                np.bincount(member, weights=bce(z, np.ones_like(z))) / np.bincount(member))

    def statistic(lab, as0, as1):
        d = LN2 - np.where(lab == 1, as1, as0)
        return float(np.mean([d[lab == c].mean() for c in (0, 1)]))

    as0, as1 = group_bce(logits)
    observed = statistic(label, as0, as1)
    scaled = [statistic(label, *group_bce(logits / t)) for t in T.TEMPERATURES]
    best = int(np.argmax(scaled))
    d_star, t_star = scaled[best], T.TEMPERATURES[best]
    labellings = comb(len(names), n1)
    floor = d_star > T.MARGIN_FLOOR * LN2
    screen = f"d {observed:.4f}, d* {d_star:.4f} (T {t_star:.3g})"
    if labellings < T.MIN_TESTABLE:
        return floor, observed, d_star, t_star, np.nan, False, (f"{screen}: not testable ({labellings} labellings); collapse screen "
                                                                + ("passed" if floor else "failed"))
    if labellings <= T.NULL_PERMUTATIONS:
        null = np.array([statistic(np.isin(np.arange(len(names)), ones).astype(int), as0, as1)
                         for ones in combinations(range(len(names)), n1)])
        p = float((null >= observed).mean())
    else:
        rng = np.random.default_rng([seed, 1])   # its own stream (the validation bags use [seed])
        null = np.array([statistic(rng.permutation(label), as0, as1) for _ in range(T.NULL_PERMUTATIONS)])
        p = float((1 + (null >= observed).sum()) / (1 + len(null)))
    passed = p <= T.NULL_ALPHA and floor
    reason = "" if passed else (": collapsed (d* at or below the floor)" if not floor else ": not better than chance")
    return passed, observed, d_star, t_star, p, True, (f"{screen}, permutation p {p:.4f} "
                                                       f"({min(labellings, T.NULL_PERMUTATIONS)} labellings){reason}")


def did_it_train_mse(predictions: np.ndarray, val_bags, samples, targets: dict, seed: int) -> tuple:
    """The MSE variant's check, shaped as did_it_train's: d = 1 - MSE / MSE_const, both
    class-balanced means over validation holdout groups, each group's target its samples' mean z-scored age
    and MSE_const the training-mean predictor's (0 in z units, so the mean squared target). The null permutes
    the groups' (target, 0/1 label) rows jointly, predictions fixed, NULL_PERMUTATIONS seeded draws,
    p = (1 + #{null >= d}) / (1 + N). Testable as did_it_train. Passes if d > MARGIN_FLOOR and, on a testable
    split, p <= NULL_ALPHA. No d* (a temperature has no meaning for MSE): d* and T* are None."""
    from math import comb
    group_of = np.array([samples.loc[s, "holdout_group"] for s, _ in val_bags])
    names = np.array(sorted(set(group_of)))
    member = np.searchsorted(names, group_of)
    label = samples.groupby("holdout_group")["group"].first().loc[names].to_numpy()
    target = pd.Series(targets).loc[samples.index].groupby(samples["holdout_group"]).mean().loc[names].to_numpy()
    n1 = int(label.sum())
    assert min(n1, len(names) - n1) >= 2, "did it train? needs >= 2 validation holdout groups per class"
    count = np.bincount(member)
    first, second = np.bincount(member, weights=predictions) / count, np.bincount(member, weights=predictions ** 2) / count

    def statistic(lab, tgt):
        balanced_mean = lambda v: np.mean([v[lab == c].mean() for c in (0, 1)])
        return float(1.0 - balanced_mean(second - 2 * tgt * first + tgt ** 2) / balanced_mean(tgt ** 2))

    observed = statistic(label, target)
    floor = observed > T.MARGIN_FLOOR
    if comb(len(names), n1) < T.MIN_TESTABLE:   # as did_it_train: the floor alone, margin_p NaN (protocol.TrainingRecord)
        return floor, observed, None, None, np.nan, False, f"MSE d {observed:.4f}: not testable; floor " + ("passed" if floor else "failed")
    rng = np.random.default_rng([seed, 1])
    null = np.array([statistic(label[order], target[order]) for order in (rng.permutation(len(names)) for _ in range(T.NULL_PERMUTATIONS))])
    p = float((1 + (null >= observed).sum()) / (1 + len(null)))
    return floor and p <= T.NULL_ALPHA, observed, None, None, p, True, f"MSE d {observed:.4f}, permutation p {p:.4f}"


def micro_batch(batch: int, cap: int) -> int:
    """The micro-batch for a configured batch: its largest divisor at or below the memory cap pinned per
    (encoder, bag size), so gradients accumulate to exactly the batch."""
    return max(d for d in range(1, min(batch, cap) + 1) if batch % d == 0)


def micro_batch_for(encoder: str, config: dict) -> int:
    """A configuration's micro-batch: the whole batch with the age-similarity loss, else from the pinned
    cap for its encoder at the smallest pinned bag size at or above its own (memory grows with the bag
    size, so a smaller bag fits under a larger bag's cap; training.MICRO_BATCH)."""
    if config.get("age_similarity_loss"):
        return config["batch_size"]
    pinned = [n for n in T.MICRO_BATCH[encoder] if n >= config["ncell"]]
    assert pinned, f"{encoder}: no micro-batch cap pinned for bags of {config['ncell']} cells or more (timing test block F)"
    return micro_batch(config["batch_size"], T.MICRO_BATCH[encoder][min(pinned)])


def train(build, config: dict, train_cells: protocol.Cells, val_cells: protocol.Cells, inputs: Inputs,
          init_seed: int, bag_seed: int, val_seed: int, device="cpu", ages: dict = None, on_epoch=None,
          residents: tuple = None, targets: dict = None, train_bag_list: list = None) -> tuple:
    """Train the network that build() returns (built after seeding torch with init_seed, so its
    initial weights are seeded); returns (the network, the kept state_dict, TrainingRecord, validation
    bags, per-bag validation logits at the kept checkpoint). ages (sample -> z-scored age) switches on the
    age-similarity loss. config["max_steps"] (optional) stops training after exactly that many optimizer
    steps; config["patience"] None trains without a patience stop. on_epoch(epoch, seconds since the start,
    validation BCE), when given, is called
    after every epoch (the timing test's progress log); it can't change the training. residents: the
    (training, validation) Residents built once for every candidate that shares these inputs (stage 05), else built here. targets (sample -> z-scored age, both sides) go with
    config["objective"] "mse" (the MSE variant: MSE loss, class-balanced validation MSE, did_it_train_mse).
    train_bag_list: a fixed list of training bags (ScRAT's mixup: drawn once, reshuffled every epoch, its last
    partial batch kept), labelled by the samples' group, which may be soft."""
    start_time = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(init_seed)
    net = build().to(device).train()
    if residents is None:
        fits = fits_on_device(train_cells, val_cells)
        residents = Resident(train_cells, inputs, device, fits), Resident(val_cells, inputs, device, fits)
    train_res, val_res = residents
    assert train_res.inputs is inputs and val_res.inputs is inputs, "the residents were built with other inputs"
    opt = torch.optim.Adam(net.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]),
                           weight_decay=config["wts_decay"])
    # an encoder's own learning-rate schedule, where its authors' default has one (ScRAT: StepLR(50, 0.5))
    schedule = torch.optim.lr_scheduler.StepLR(opt, *config["step_lr"]) if config.get("step_lr") else None
    groups = train_cells.samples["group"].to_dict()
    mse = config.get("objective") == "mse"
    assert mse == (targets is not None), "the MSE objective needs the z-scored age targets (and only it)"
    y_train, y_val = (targets, targets) if mse else (groups, val_cells.samples["group"].to_dict())
    val_bags = bags.fixed_bags(val_cells.cell_id, val_cells.sample_id, val_cells.samples.index, config["ncell"],
                               T.VALIDATION_BAGS_PER_SAMPLE, np.random.default_rng(val_seed))
    assert bool(config.get("age_similarity_loss")) == (ages is not None), "age_similarity_loss needs the training ages (and only then)"
    batch = config["batch_size"]
    micro = min(config.get("micro_batch", batch), batch)
    assert batch % micro == 0 and (ages is None or micro == batch), "micro-batches must divide the batch (none with the age loss)"
    val_chunks = [(val_res.indices(val_bags[c:c + micro]), len(val_bags[c:c + micro])) for c in range(0, len(val_bags), micro)]
    best, best_state, best_logits, kept_epoch, kept_step, wait, steps, curve = np.inf, None, None, None, None, 0, 0, []
    for epoch in range(config["n_epochs"]):
        if steps == config.get("max_steps"):   # the budget was reached in the last epoch (which was then scored)
            break
        rng = np.random.default_rng([bag_seed, epoch])
        if train_bag_list is None:
            epoch_bags = bags.train_bags(train_cells.cell_id, train_cells.sample_id, groups, config["ncell"],
                                         config["nsubset"], rng)
            end = len(epoch_bags) - batch + 1   # drop_last, as the original
        else:   # ScRAT's mixup: its fixed bags reshuffled, the last partial batch kept (as its DataLoader)
            epoch_bags = [train_bag_list[k] for k in rng.permutation(len(train_bag_list))]
            end = len(epoch_bags)
        for b in range(0, end, batch):
            chunk = epoch_bags[b:b + batch]
            opt.zero_grad()
            for m in range(0, len(chunk), micro):
                part = chunk[m:m + micro]
                e, logits = forward_bags(net, part, train_res)
                y = torch.tensor([float(y_train[s]) for s, _ in part], device=device)
                loss = (F.mse_loss(logits, y, reduction="sum") if mse
                        else F.binary_cross_entropy_with_logits(logits, y, reduction="sum")) / len(chunk)
                if ages is not None:
                    loss = loss + T.AGE_LOSS_WEIGHT * age_loss(e, torch.tensor([ages[s] for s, _ in part], device=device))
                loss.backward()
            opt.step()
            steps += 1
            if steps == config.get("max_steps"):   # a fixed step budget (row B: row A's total steps)
                break
        if schedule is not None:
            schedule.step()
        logits = validation_logits(net, val_chunks, val_res)
        y = np.array([float(y_val[s]) for s, _ in val_bags])
        per_bag = (logits - y) ** 2 if mse else bce(logits, y)
        score = balanced(per_bag, val_bags, val_cells.samples)
        curve.append(score)
        if on_epoch is not None:
            on_epoch(epoch + 1, time.perf_counter() - start_time, score)
        if epoch + 1 < T.BURN_IN_EPOCHS or steps < T.MIN_STEPS:   # the first keepable epoch is BURN_IN_EPOCHS
            continue
        if score < best:
            best, best_state, best_logits, kept_epoch, kept_step, wait = score, copy.deepcopy(net.state_dict()), logits, epoch + 1, steps, 0
        else:
            wait += 1
            if config["patience"] is not None and wait >= config["patience"]:   # None: no patience stop
                break
    assert best_state is not None, "no checkpoint was kept (n_epochs below the burn-in?)"
    passed, d, d_star, t_star, p, testable, reason = (did_it_train_mse(best_logits, val_bags, val_cells.samples, targets, val_seed)
                                                      if mse else did_it_train(best_logits, val_bags, val_cells.samples, val_seed))
    record = protocol.TrainingRecord(
        epochs=len(curve), stop_epoch=kept_epoch, validation_curve=curve, did_train=passed, did_train_reason=reason,
        seconds=round(time.perf_counter() - start_time, 2), steps=steps, kept_step=kept_step, margin_d=d, margin_d_star=d_star,
        margin_temperature=t_star, margin_p=p, testable=testable,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None)
    net.load_state_dict(best_state)
    return net, best_state, record, val_bags, best_logits


def embed_samples(net, cells: protocol.Cells, inputs: Inputs, sample_ids, n_cells: int, pair: str, unit: tuple,
                  device="cpu", residents: tuple = None) -> np.ndarray:
    """Each sample's embedding: the mean over its 10 test bags of the training bag size
    (sc2bulk.bags.test_bags, the same cells for every model). residents: Residents that
    together hold every sample's cells (a training's own, so no third device copy is built; search spec
    v2 item 5), else one is built for `cells`."""
    residents = residents or (Resident(cells, inputs, device, fits_on_device(cells)),)
    holder = {s: r for r in residents for s in np.unique(r.cells.sample_id)}
    test = bags.test_bags(cells.cell_id, cells.sample_id, list(sample_ids), n_cells, representations.TEST_TIME_BAGS, pair, unit)
    net.eval()
    with torch.no_grad():
        out = []
        for s in sample_ids:
            e, _ = forward_bags(net, [(s, ids) for ids in test[s]], holder[s], embed_only=True)
            out.append(e.mean(dim=0).cpu().numpy())
    return np.vstack(out)


def embed_profiles(net, profiles: protocol.Profiles, inputs: Inputs, device="cpu") -> np.ndarray:
    """One-profile inputs (bulk, pseudobulk) as 1-cell bags (weights 1)."""
    X = sp.csr_matrix(profiles.X)
    net.eval()   # one-profile inputs carry no depth weights: a 1-cell bag's weight is 1
    with torch.no_grad():
        out = []
        for i in range(X.shape[0]):
            x = torch.tensor(inputs.profile(X[i].toarray().ravel())[None], dtype=torch.float32, device=device)
            out.append(net.embed(x, torch.ones((1, 1), device=device)).cpu().numpy())
    return np.vstack(out)


def embed_pseudocells(net, bags_log: np.ndarray, totals: np.ndarray, inputs: Inputs, device="cpu") -> np.ndarray:
    """H3: one bulk sample's bags of pseudo-cells (bags x cells x genes, log-CP10k on the panel), each
    embedded as a bag of cells (no gene scaler, as bulk takes none; depth weights log1p of each pseudo-cell's
    shared-gene total, normalized within the bag), then averaged over the bags."""
    x = torch.as_tensor(np.stack([inputs.transform(b) for b in bags_log]), dtype=torch.float32, device=device)
    w = torch.as_tensor(np.log1p(totals), dtype=torch.float32, device=device)
    net.eval()
    with torch.no_grad():
        return net.embed(x, w / w.sum(dim=1, keepdim=True)).mean(dim=0).cpu().numpy()
