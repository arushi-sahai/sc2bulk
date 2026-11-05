"""Train/test splits.

SoundLife's single-cell and bulk samples come from the same donors, so every donor is
either training or test, never both: 30% of donors in each age group go to test, drawn
with a fixed seed. The split is pre-declared: its test donors are written in config
(pairs.py), and this function reproduces them.
"""
import numpy as np
import pandas as pd


def donor_split(donors: pd.DataFrame, test_fraction: float, seed: int) -> set:
    """The test donors, from a table with one row per donor (donor_id, age_group): within
    each age group (in sorted order), donors in sorted order are permuted with the seed, and
    the first round(test_fraction * n) are test."""
    donors = donors.drop_duplicates("donor_id").sort_values("donor_id")
    rng = np.random.RandomState(seed)
    test = set()
    for _, group in donors.groupby("age_group"):
        order = group["donor_id"].to_numpy()[rng.permutation(len(group))]
        test |= set(order[:round(test_fraction * len(order))])
    return test
