"""Seeds for every random choice, one per (purpose, pair, fold set, fold, index)."""
import numpy as np

from sc2bulk.config import settings


def derived(purpose: str, pair: str, index: int, fold_set: str = "all", fold: int = 0) -> int:
    """A seed (uint32) for one random choice: distinct for every purpose, pair, fold and index,
    and stable across runs, machines and changes to the pair order."""
    entropy = [settings.SEED_BASE, settings.SEED_PURPOSES[purpose], settings.SEED_PAIRS[pair],
               settings.FOLD_SET_IDS[fold_set], fold, index]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])
