"""Stage 10c (sc2bulk.concordance): shifts with
planted signs, per scoring unit and within organ; a zero shift is not shared; the reference marks on a toy; the
matched reference is deterministic with ties by symbol; the label gate; label-bearing outputs carry the
labelled_ prefix; the smoke-only skip of thin strata. Synthetic labels only."""
import numpy as np
import pandas as pd
import pytest

from sc2bulk import concordance, labels, paths, schemas
from sc2bulk.config import settings


def table(groups, units=None, strata=None):
    n = len(groups)
    return pd.DataFrame({"group": groups, "scoring_unit_id": units if units is not None else [f"u{i}" for i in range(n)],
                         "stratum": strata if strata is not None else ["all"] * n})


def test_planted_signs():
    rng = np.random.default_rng(0)
    group = np.array([0] * 6 + [1] * 6)
    X = rng.normal(size=(12, 3))
    X[group == 1, 0] += 5.0     # up
    X[group == 1, 1] -= 5.0     # down
    out = concordance.side_shift(X, table(group))
    assert out[0] > 2 and out[1] < -2
    a, b = X[group == 1, 0], X[group == 0, 0]
    pooled = np.sqrt((5 * a.var(ddof=1) + 5 * b.var(ddof=1)) / 10)
    assert np.isclose(out[0], (a.mean() - b.mean()) / pooled)


def test_scoring_units_are_averaged_first_and_organs_weighted_equally():
    X = np.array([[0.0], [2.0], [1.0], [10.0], [12.0], [13.0], [14.0]])
    group = [0, 0, 0, 1, 1, 1, 1]
    units = ["d0", "d0", "d1", "d2", "d3", "d3", "d3"]   # d0's two samples count once
    by_unit = concordance.side_shift(X, table(group, units))
    assert np.isclose(by_unit[0], concordance.shift(np.array([[1.0], [1.0], [10.0], [13.0]]), np.array([0, 0, 1, 1]))[0])
    Y = np.vstack([X, X * 2])
    strata = ["kidney"] * 7 + ["lung"] * 7
    within = concordance.side_shift(Y, table(group * 2, [f"{s}_{u}" for s, u in zip(strata, units * 2)], strata))
    expected = np.mean([concordance.side_shift(X, table(group, units))[0], concordance.side_shift(X * 2, table(group, units))[0]])
    assert np.isclose(within[0], expected)


def test_zero_shift_is_not_shared_and_base_all():
    sc = pd.Series({"a": 1.0, "b": -1.0, "c": 0.0, "d": 2.0})
    bulk = pd.Series({"a": 0.5, "b": 1.0, "c": 0.0, "d": 3.0})
    shared = concordance.is_shared(sc, bulk)
    assert shared.to_dict() == {"a": True, "b": False, "c": False, "d": True}
    assert concordance.shares(["a", "b"], sc, bulk)["base_all"] == 0.5
    assert concordance.shares(["a", "b"], sc, bulk)["shared_share"] == 0.5


def test_matched_reference_is_deterministic_with_ties_by_symbol():
    sc = pd.Series({"top": 1.0, "x1": 0.9, "x2": -1.1, "b": 3.0, "a": 3.0, "far": 10.0})
    shared = pd.Series({"top": True, "x1": True, "x2": False, "b": False, "a": True, "far": True})
    assert concordance.matched_base(["top"], sc, shared, n=2) == 0.5          # x1 and x2 (both 0.1 away)
    assert concordance.matched_base(["top"], sc, shared, n=3) == pytest.approx(2 / 3)   # then a before b (2.0 away, tie by symbol)
    two = concordance.matched_base(["top", "far"], sc, shared, n=1)             # with multiplicity: x1 for top, ...
    assert two == concordance.matched_base(["top", "far"], sc, shared, n=1)
    assert two == np.mean([shared["x1"], shared["a"]])                          # ... a for far (|3| nearest |10|, tie with b)


def test_require_access_refuses_without_the_analysis_fixed_tag_or_blinding(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "BLINDED", None)
    monkeypatch.setattr(labels, "ANALYSIS_FIXED", tmp_path / "analysis_fixed.json")
    with pytest.raises(AssertionError):
        labels.require_access()
    monkeypatch.setattr(settings, "BLINDED", 3)
    labels.require_access()


def test_outputs_are_label_bearing_and_marked():
    assert schemas.KINDS["gene_shifts"][2] and schemas.KINDS["concordance"][2]
    path = paths.output("10c_concordance", "lupus", f"{settings.LABELLED}_concordance.parquet")
    assert path.is_relative_to(settings.LABELLED_OUTPUTS) and path.name.startswith("labelled_")
    text = (settings.REPO / "pipeline" / "10c_concordance.py").read_text()
    assert 'f"{settings.LABELLED}_{name}.parquet"' in text and "labels.require_access()" in text


def test_side_shift_skips_thin_strata_only_when_asked():
    """The smoke run skips a stratum with < 2 scoring units in a group; the real run keeps the assert."""
    table = pd.DataFrame({"group": [0, 0, 1, 1, 0, 1, 1], "scoring_unit_id": ["a", "b", "c", "d", "e", "f", "g"],
                          "stratum": ["s1", "s1", "s1", "s1", "s2", "s2", "s2"]})   # s2: one in group 0, two in group 1
    X = np.arange(14, dtype=float).reshape(7, 2)
    full = concordance.side_shift(X[:4], table.iloc[:4])
    assert np.allclose(concordance.side_shift(X, table, skip_short=True), full)   # s2 (one unit per group) skipped
    with pytest.raises(AssertionError, match=">= 2 scoring units"):
        concordance.side_shift(X, table)
    with pytest.raises(AssertionError, match="no stratum"):
        concordance.side_shift(X[4:], table.iloc[4:], skip_short=True)
    text = (settings.REPO / "pipeline" / "10c_concordance.py").read_text()
    assert text.count("skip_short=settings.SMOKE") == 2                       # both sides, the smoke run only
