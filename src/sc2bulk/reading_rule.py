"""The out-of-fold row's reading rule (Dec 7), for stage 11b.

The primary analysis always stands. Each row is summarized on its own (statistics.summarize with
family=True: Holm within that row's own better and worse families; the out-of-fold family is the
configurations estimable in that row; the pooled row's ceiling set is the primary analysis's). Per
pair, Results says so for a configuration whose out-of-fold row differs from the primary in any of:
  (a) sign: the deltas have strictly opposite signs (an exact 0 is no change);
  (b) the Holm-adjusted better call at ALPHA differs, or the worse call does;
  (c) the margin: u < X on one row and u >= X on the other.
A configuration not estimable in one of the rows is reported as such and never counts as a change. The
same comparison of the pooled rows is reported in the Discussion and triggers nothing.
"""
import pandas as pd

from sc2bulk.config import statistics as S


def changes(primary: pd.DataFrame, out_of_fold: pd.DataFrame) -> pd.DataFrame:
    """Per configuration (key) of two summarize() tables of one pair (or of the pooled rows): estimable
    (in both rows), whether the sign, the better call, the worse call and the margin conclusion changed,
    and flagged (any of them)."""
    a, b = primary.set_index("key"), out_of_fold.set_index("key")
    rows = []
    for key in dict.fromkeys([*a.index, *b.index]):
        row = {"key": key, "estimable": key in a.index and key in b.index, "sign": False, "better": False, "worse": False, "margin": False}
        if row["estimable"]:
            p, o = a.loc[key], b.loc[key]
            assert p[["p_better_holm", "p_worse_holm"]].notna().all() and o[["p_better_holm", "p_worse_holm"]].notna().all(), \
                f"{key}: the reading rule compares Holm-adjusted calls"
            row["sign"] = bool(p["delta"] * o["delta"] < 0)
            row["better"] = bool((p["p_better_holm"] <= S.ALPHA) != (o["p_better_holm"] <= S.ALPHA))
            row["worse"] = bool((p["p_worse_holm"] <= S.ALPHA) != (o["p_worse_holm"] <= S.ALPHA))
            row["margin"] = bool((p["u"] < S.MARGIN) != (o["u"] < S.MARGIN))
        row["flagged"] = row["sign"] or row["better"] or row["worse"] or row["margin"]
        rows.append(row)
    return pd.DataFrame(rows)
