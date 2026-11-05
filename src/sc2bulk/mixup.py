"""ScRAT's sample mixup, for its own-recipe row (stage 06 row G), following utils.mixups() at training.SCRAT_COMMIT with run.sh's settings (training.SCRAT_MIXUP):
augment_num synthetic samples built once from the training side, in the input space the network sees (PCA).
Each pairs a case with a control (same_pheno -1: the two phenotypes in random order), with one
lam ~ Beta(alpha, alpha). Per cell type present in either sample (in sorted order), max(int(min_size x
(lam f1 + (1 - lam) f2)), 1) cells (f: the type's fraction of the sample's cells), each lam x1 + (1 - lam) x2
with x1, x2 drawn with replacement from that type's cells; a type absent from one sample takes the zero
vector (the training mean in PCA space) as its partner, and that block gets Gaussian noise of variance
noise_var. Soft label lam y1 + (1 - lam) y2. inter_only: training uses the synthetic samples only. Mouse
organs: the organ is drawn first, in proportion to its training samples, and both samples come from it.
Then bags_per_sample bags of the bag size per synthetic sample, drawn once (its sampling())."""
import numpy as np
import pandas as pd

from sc2bulk import bags, protocol
from sc2bulk.config import training as T


def synthetic(train: protocol.Cells, inputs, strata: pd.Series, rng) -> protocol.Cells:
    """The synthetic samples as Cells: X their transformed cells (dense, float32), group the soft label.
    train: the training-side Cells (cell_type at the deconvolution level); inputs: the fitted Inputs (PCA);
    strata: sample -> its stratum (training_inputs.sample_strata)."""
    S = T.SCRAT_MIXUP
    X = np.zeros((len(train.cell_id), inputs.components.shape[1]), dtype=np.float32)
    rows = {s: np.flatnonzero(train.sample_id == s) for s in train.samples.index}
    for s, r in rows.items():
        X[r] = inputs.cells(train.X[r].toarray(), s)
    groups = train.samples["group"]
    sizes = strata.loc[groups.index].value_counts().sort_index()
    blocks, labels = [], {}
    for k in range(S["augment_num"]):
        lam = rng.beta(S["alpha"], S["alpha"])
        stratum = rng.choice(sizes.index, p=(sizes / sizes.sum()).to_numpy())
        members = groups[strata.loc[groups.index] == stratum]
        y1, y2 = rng.choice(2, 2, replace=False)
        s1, s2 = (rng.choice(sorted(members.index[members == y])) for y in (y1, y2))
        mixed = []
        for t in sorted(set(train.cell_type[rows[s1]]) | set(train.cell_type[rows[s2]])):
            r1, r2 = (rows[s][train.cell_type[rows[s]] == t] for s in (s1, s2))
            n = max(int(S["min_size"] * (lam * len(r1) / len(rows[s1]) + (1 - lam) * len(r2) / len(rows[s2]))), 1)
            x1 = X[rng.choice(r1, n)] if len(r1) else np.zeros((n, X.shape[1]), dtype=np.float32)
            x2 = X[rng.choice(r2, n)] if len(r2) else np.zeros((n, X.shape[1]), dtype=np.float32)
            x = lam * x1 + (1 - lam) * x2
            if not (len(r1) and len(r2)):
                x = x + rng.normal(0.0, S["noise_var"] ** 0.5, x.shape)
            mixed.append(x.astype(np.float32))
        blocks.append(np.vstack(mixed))
        labels[f"mix{k}"] = lam * y1 + (1 - lam) * y2
    sample_id = np.concatenate([[f"mix{k}"] * len(b) for k, b in enumerate(blocks)])
    cell_id = np.concatenate([[f"mix{k}_{i}" for i in range(len(b))] for k, b in enumerate(blocks)])
    samples = pd.DataFrame({"group": labels, "holdout_group": {s: s for s in labels}})
    return protocol.Cells(X=np.vstack(blocks), genes=[f"pc{i}" for i in range(X.shape[1])], cell_id=cell_id,
                          sample_id=sample_id, cell_type=np.full(len(cell_id), "mixed"), library_size=np.ones(len(cell_id)),
                          samples=samples)


def fixed_bags(mixed: protocol.Cells, n_cells: int, rng) -> list:
    """bags_per_sample bags of min(n_cells, available) cells per synthetic sample, drawn once (no class balancing)."""
    return bags.fixed_bags(mixed.cell_id, mixed.sample_id, mixed.samples.index, n_cells, T.SCRAT_MIXUP["bags_per_sample"], rng)
