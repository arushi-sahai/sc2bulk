"""Titles and conditional text: every reachable Fig 2, Fig 4 and Fig 6 title at TITLE_MAX_WORDS with the longest labels and pair
names, before any run; Fig 2 on claims from the real stage 11 (K >= 2, 1, 0) and on hand-built claims with stage 11's
own keys; the 07d and age-loss sentences."""
import itertools
import sys

import pandas as pd
import pytest

from sc2bulk.config import configurations, settings
from sc2bulk.config import figures as F
from sc2bulk.config import properties as P
from sc2bulk.config.pairs import PAIR_ORDER

import test_statistics_driver as driver

sys.path.insert(0, str(settings.REPO / "figures"))
import titles  # noqa: E402

ALTERNATIVES = [k for k in configurations.CONFIGURATION_ORDER if k != "pseudobulk"]
LONGEST = sorted(ALTERNATIVES, key=lambda k: -len(configurations.CONFIGURATIONS[k]["label"].split()))


def pooled(keys) -> pd.DataFrame:
    return pd.DataFrame({"key": keys, "delta": 0.123, "ci_low": 0.045, "ci_high": 0.201}).set_index("key")


def claims(better=(), inconclusive=(), per_pair=(), excluded=("hammond",), k0=False) -> dict:
    """Hand-built claims with exactly the keys stage 11's claims() and main() write."""
    if k0:
        return {"no_configuration_outperformed": None, "within_margin_all": None, "per_pair_better": [list(x) for x in per_pair], "wording": "",
                "configurations": ALTERNATIVES, "cut": [], "resamples": 2000, "excluded_at_ceiling": list(PAIR_ORDER), "lupus_crohns_formed": False,
                "lupus_crohns_note": ""}
    return {"no_configuration_outperformed": not better, "within_margin_all": not inconclusive, "inconclusive_at_margin": list(inconclusive),
            "pooled_u": {}, "pooled_better": list(better), "pooled_worse": [], "per_pair_better": [list(x) for x in per_pair], "wording": "",
            "configurations": ALTERNATIVES, "cut": [], "resamples": 2000, "excluded_at_ceiling": list(excluded), "lupus_crohns_formed": True,
            "lupus_crohns_note": ""}


def test_hand_built_claims_have_stage11s_keys(tmp_path, monkeypatch):
    table, real = driver.run(tmp_path, monkeypatch, at_ceiling={"hammond"})
    assert set(claims()) == set(real)
    table, real = driver.run(tmp_path / "none", monkeypatch, at_ceiling=set(driver.PAIRS))
    assert set(claims(k0=True)) == set(real)


def test_fig2_on_the_real_stage11(tmp_path, monkeypatch):
    """K >= 2, K = 1 and K = 0 from stage 11's own claims (the driver fixture: four pairs)."""
    monkeypatch.setattr(titles, "PAIR_ORDER", driver.PAIRS)
    for case, ceiling in (("k2", {"hammond"}), ("k1", {"hammond", "tms", "soundlife"}), ("k0", set(driver.PAIRS))):
        table, real = driver.run(tmp_path / case, monkeypatch, at_ceiling=ceiling)
        rows = table[(table["pool_set"] == "nonceiling_equal") & (table["readout"] == "ridge")].set_index("key")
        made = titles.fig2(real, rows)
        if case == "k0":
            assert made["title"] == F.FIG2_TITLES["not_evaluable"]
            continue
        assert made["title"].startswith(("Under the main readout, no configuration outperformed pseudobulk, ", "No configuration outperformed pseudobulk, ")) or real["pooled_better"]
        assert ("single pooled pair: human blood, lupus" in made["sentences"]) == (case == "k1")


@pytest.mark.parametrize("single", [False, True])
def test_every_fig2_variant_fits(single):
    excluded = PAIR_ORDER[:-1] if single else ("hammond",)
    forms = F.FIG2_TITLES["single" if single else "pooled"]
    for n in range(1, len(ALTERNATIVES) + 1):   # 1, 2, 3+ pooled "better", the longest labels first
        made = titles.fig2(claims(better=LONGEST[:n], excluded=excluded), pooled(LONGEST[:n]))
        assert titles.words(made["title"]) <= F.TITLE_MAX_WORDS
        assert len(made["sentences"]) >= n
    assert titles.fig2(claims(better=LONGEST[:3]), pooled(LONGEST[:3]))["title"].startswith("3 configurations")
    made = titles.fig2(claims(excluded=excluded), pooled([]))
    assert made["title"] == forms["within_margin"].format(margin="0.05") and titles.words(made["title"]) <= F.TITLE_MAX_WORDS
    assert made["sentences"] == ([F.FIG2_SENTENCES["single_pair"].format(pair="human motor cortex, ALS")] if single else [])
    for n in range(1, len(ALTERNATIVES) + 1):
        made = titles.fig2(claims(inconclusive=LONGEST[:n], per_pair=[("lupus", "scvi_cells")], excluded=excluded), pooled([]))
        assert made["title"] == forms["inconclusive"].format(k=n, n=len(ALTERNATIVES), verb="is" if n == 1 else "are", margin="0.05")
        assert titles.words(made["title"]) <= F.TITLE_MAX_WORDS and made["sentences"][-1].endswith("(per pair; not adjusted across pairs)")
    assert ("pooled" in titles.fig2(claims(better=LONGEST[:1], excluded=excluded), pooled(LONGEST[:1]))["title"]) != single
    made = titles.fig2(claims(k0=True, per_pair=[("tms", "cytoset_cells")]), pooled([]))
    assert made["title"] == F.FIG2_TITLES["not_evaluable"] and "mouse organs, aging" in made["sentences"][0]


def test_every_fig4_variant_fits():
    longest = sorted(PAIR_ORDER, key=lambda p: -len(titles.display(p).split()))
    for n in range(0, len(PAIR_ORDER) + 1):
        for chosen in itertools.combinations(longest, n):
            tests = pd.DataFrame({"pair": PAIR_ORDER, "p_better": [0.01 if p in chosen else 0.4 for p in PAIR_ORDER]})
            made = titles.fig4(tests, {p: i % 2 == 0 for i, p in enumerate(chosen)})
            assert titles.words(made["title"]) <= F.TITLE_MAX_WORDS
            assert (made["title"] == F.FIG4_TITLES["none"]) == (n == 0) == (made["sentences"] == [F.FIG4_TEST["none"]])
            assert len(made["sentences"]) == (1 if n < 2 else 2)   # one test sentence per status group of rejecting pairs
    tests = pd.DataFrame({"pair": PAIR_ORDER, "p_better": [0.01, float("nan"), 0.4, 0.4, 0.4, 0.4]})
    made = titles.fig4(tests, {"hammond": False})
    assert made["title"] == F.FIG4_TITLES["pairs"].format(pairs="mouse microglia, aging")
    assert made["sentences"] == [F.FIG4_TEST["pairs"].format(pairs="mouse microglia, aging", which="an unselected"),
                                 F.FIG4_NOT_FORMED.format(pairs="mouse organs, aging")]


def test_fig6_and_fixed_titles():
    for licensed in (True, False):
        made = titles.fig6({"licensed": licensed, "title": P.LICENSED_TITLE if licensed else P.DESCRIPTIVE_TITLE, "named": ["cytoset_cells"]})
        assert made["title"] == F.FIG6_TITLE and titles.words(made["title"]) <= F.TITLE_MAX_WORDS and "CytoSet, cells" in made["sentences"][0]
    with pytest.raises(AssertionError):
        titles.fig6({"licensed": True, "title": P.DESCRIPTIVE_TITLE, "named": []})
    for item, title in F.TITLES.items():
        assert titles.words(title) <= F.TITLE_MAX_WORDS, item
    with pytest.raises(AssertionError):
        titles.checked("one " * (F.TITLE_MAX_WORDS + 1))


def test_fig5_sentences():
    criterion = {"set_encoders": ["cytoset", "scrat"], "margin": 0.1, "delta": {"cytoset": 0.2, "scrat": 0.05}, "passes": {"cytoset": True, "scrat": False},
                 "control_passes": False, "structure_readings": {"cytoset": "x", "scrat": "y"}, "normalization": {"cytoset": "none", "scrat": "population"}}
    made = titles.fig5(criterion)
    assert len(made["sentences"]) == 1 and "failed" in made["sentences"][0]   # the readings stay in the Methods
    made = titles.fig5({**criterion, "normalization": {"cytoset": "per_sample", "scrat": "none"}})
    assert made["sentences"][1] == F.FIGURE_TEXT["fig5d_per_sample"]


def test_fig4a_sentences(monkeypatch):
    assert titles.fig4a_sentences([]) == []
    monkeypatch.setattr(F, "FIG4A_FM_ALL_AVAILABLE", ("scgpt",))
    assert titles.fig4a_sentences([]) == [F.FIGURE_TEXT["fig4a_scgpt_only"]]
    monkeypatch.setattr(F, "FIG4A_FM_ALL_AVAILABLE", ())
    assert titles.fig4a_sentences(["soundlife"]) == [F.FIGURE_TEXT["fig4a_fm_cut"], F.FIGURE_TEXT["fig4a_not_estimable"]]


def test_selected_carrier(monkeypatch):
    """Fig 4's "the selected / an unselected candidate": row A's point and a search candidate equal to the frozen
    configuration are the selected one; another bag size never is."""
    from sc2bulk.config import search_space
    import common
    base = search_space.base("cytoset")
    candidates = pd.DataFrame({"key": ["base", "out_max"], "change_value": ["{}", '{"out_pool": "max"}']})
    monkeypatch.setattr(common, "names", lambda *a, **k: ["cytoset/candidates_s0.parquet"])
    monkeypatch.setattr(common, "read_kind", lambda *a, **k: candidates)
    monkeypatch.setattr(common, "read_json", lambda *a, **k: {"frozen": dict(base)})
    assert titles.selected_carrier("lupus", "cytoset/base") and not titles.selected_carrier("lupus", "cytoset/out_max")
    monkeypatch.setattr(common, "read_json", lambda *a, **k: {"frozen": {**base, "out_pool": "max"}})
    assert titles.selected_carrier("lupus", "cytoset/out_max") and not titles.selected_carrier("lupus", "cytoset/base")
    assert titles.selected_carrier("lupus", "cytoset_cells@500") and not titles.selected_carrier("lupus", "cytoset_cells@200all")
