"""The one bag drawer. Every model's bags come from here, as
cell ids, so all models see the same cells whatever their row order: training bags
(class-balanced random bags of a sample's cells) and test-time bags (n_bags seeded draws per
sample, the same for every model)."""
import numpy as np

from sc2bulk import preprocess, seeds


def _by_sample(cell_ids: np.ndarray, cell_sample: np.ndarray) -> dict:
    order = np.argsort(cell_sample, kind="stable")
    samples, starts = np.unique(cell_sample[order], return_index=True)
    return {s: cell_ids[part] for s, part in zip(samples, np.split(order, starts[1:]))}


def test_bags(cell_ids, cell_sample, sample_ids, n_cells: int, n_bags: int, pair: str, unit: tuple) -> dict:
    """sample_id -> list of n_bags arrays of cell ids (each min(n_cells, available) cells, drawn
    without replacement from the sample's cells in sorted-id order); n_cells = 0 means all of
    the sample's cells, once. Keyed by sample id, so every caller gets the same bags."""
    by_sample = _by_sample(np.asarray(cell_ids), np.asarray(cell_sample))
    unit_seed = seeds.derived("test_bags", pair, 0, *unit)
    bags = {}
    for sample in sample_ids:
        cells = np.sort(by_sample[sample])
        if n_cells == 0:
            bags[sample] = [cells]
            continue
        rng = np.random.default_rng([unit_seed, preprocess.sample_seed(pair, sample) % 2**32, n_cells])
        bags[sample] = [np.sort(rng.choice(cells, min(n_cells, len(cells)), replace=False)) for _ in range(n_bags)]
    return bags


def train_bags(cell_ids, cell_sample, groups: dict, n_cells: int, bags_per_class: int, rng) -> list:
    """One epoch of class-balanced training bags (CytoSet's original procedure): each 0/1 group gets
    bags_per_class bags spread evenly over its samples (bags_per_class // samples per sample), each
    of min(n_cells, available) random cells without replacement. Samples in sorted order; returns
    (sample_id, cell-id array) pairs, shuffled with rng."""
    by_sample = {s: np.sort(c) for s, c in _by_sample(np.asarray(cell_ids), np.asarray(cell_sample)).items()}
    out = []
    for group in (0, 1):
        members = sorted(s for s, g in groups.items() if g == group)
        per_sample = bags_per_class // len(members)
        assert per_sample >= 1, f"group {group}: more samples than bags per class"
        for sample in members:
            cells = by_sample[sample]
            out.extend((sample, np.sort(rng.choice(cells, min(n_cells, len(cells)), replace=False)))
                       for _ in range(per_sample))
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def fixed_bags(cell_ids, cell_sample, sample_ids, n_cells: int, per_sample: int, rng) -> list:
    """A fixed set of bags (validation): per_sample bags of min(n_cells, available) random cells for
    each given sample, in sorted sample order. Returns (sample_id, cell-id array) pairs."""
    by_sample = {s: np.sort(c) for s, c in _by_sample(np.asarray(cell_ids), np.asarray(cell_sample)).items()}
    return [(s, np.sort(rng.choice(by_sample[s], min(n_cells, len(by_sample[s])), replace=False)))
            for s in sorted(sample_ids) for _ in range(per_sample)]
