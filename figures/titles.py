"""Titles and conditional text:
every title and conditional sentence, chosen mechanically from stage outputs and config, written to
<labelled outputs>/figures/figure_text.json ({item: {"title": ..., "sentences": [...]}}; figure_text_partial.json,
marked partial, for a pair-subset run). Titles are never drawn.

Fig 2 follows stage 11's claims in claims()'s own precedence, K = 0 first (claims() then has no pooled keys); Fig 4 the
per-pair best-candidate test (stage 11b's row 18, unadjusted across pairs); Fig 6 its Results heading
(config/figures.FIG6_TITLE; licence.json, still checked); Fig 4A's sentences config/figures.FIG4A_FM_ALL_AVAILABLE and the age-similarity n.e.;
Fig 5 criterion.json's pass flag and readings. Every title is checked against TITLE_MAX_WORDS (tests/test_titles.py
checks every reachable variant before any run)."""
import json

import numpy as np
import pandas as pd

import common
import fig4 as figure4
from sc2bulk.config import configurations
from sc2bulk.config import figures as F
from sc2bulk.config import properties as P
from sc2bulk.config import simulation as SIM
from sc2bulk.config import search_space
from sc2bulk.config import statistics as S11
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS


def words(text) -> int:
    """Whitespace-separated tokens ("0.05" and "0.04–0.12" count as one)."""
    return len(text.split())


def checked(title) -> str:
    assert words(title) <= F.TITLE_MAX_WORDS, f"{words(title)} words: {title}"
    return title


def number(x) -> str:
    return f"{x:.{F.DECIMALS}f}"


def margin() -> str:
    return f"{S11.MARGIN:g}"


def display(pair) -> str:
    """A pair's display name in running text (lower-case first letter)."""
    name = PAIRS[pair]["name"]
    return name[0].lower() + name[1:]


def fig2(claims, pooled) -> dict:
    """claims: stage 11's claims.json; pooled: its nonceiling_equal ridge rows indexed by key."""
    label = common.label
    sentences = []
    if claims["no_configuration_outperformed"] is None:   # K = 0
        title = F.FIG2_TITLES["not_evaluable"]
    else:
        kept = [p for p in PAIR_ORDER if p not in claims["excluded_at_ceiling"]]
        forms = F.FIG2_TITLES["pooled" if len(kept) > 1 else "single"]
        better = claims["pooled_better"]
        if better:
            form = "better_one" if len(better) == 1 else "better_two" if len(better) == 2 else "better_many"
            title = forms[form].format(a=label(better[0]), b=label(better[-1]), n=len(better))
            sentences += [F.FIG2_SENTENCES["better"].format(configuration=label(k), delta=number(pooled.loc[k, "delta"]),
                                                             low=number(pooled.loc[k, "ci_low"]), high=number(pooled.loc[k, "ci_high"])) for k in better]
        elif claims["within_margin_all"]:
            title = forms["within_margin"].format(margin=margin())
        else:
            k, n = len(claims["inconclusive_at_margin"]), len(claims["configurations"])
            title = forms["inconclusive"].format(k=k, n=n, verb=F.FIG2_VERB.get(k, "are"), margin=margin())
            sentences.append(F.FIG2_SENTENCES["inconclusive"].format(margin=margin(),
                                                                      configurations=", ".join(label(k) for k in claims["inconclusive_at_margin"])))
        if len(kept) == 1:
            sentences.append(F.FIG2_SENTENCES["single_pair"].format(pair=display(kept[0])))
    sentences += [F.FIG2_SENTENCES["per_pair_better"].format(configuration=label(k), pair=display(p)) for p, k in claims["per_pair_better"]]
    return {"title": checked(title), "sentences": sentences}


def fig4(tests, selected) -> dict:
    """tests: stage 11b's best_candidate test rows (key best_candidate), one per pair; selected: pair -> whether the
    candidate carrying T is the selected configuration (selected_carrier)."""
    tests = tests.set_index("pair")
    formed = [p for p in PAIR_ORDER if p in tests.index and np.isfinite(tests.loc[p, "p_better"])]
    significant = [p for p in formed if tests.loc[p, "p_better"] <= S11.ALPHA]
    if not significant:
        title = F.FIG4_TITLES["none"]
    else:
        title = F.FIG4_TITLES["pairs"].format(pairs=" and ".join(display(p) for p in significant))
        if words(title) > F.TITLE_MAX_WORDS:
            title = F.FIG4_TITLES["count"].format(n=len(significant), total=len(PAIR_ORDER))
    sentences = [F.FIG4_TEST["none"]] if not significant else [
        F.FIG4_TEST["pairs"].format(pairs=" and ".join(display(p) for p in significant if selected[p] == status), which=F.FIG4_TEST[name])
        for status, name in ((True, "selected"), (False, "unselected")) if any(selected[p] == status for p in significant)]
    missing = [p for p in PAIR_ORDER if p not in formed]
    sentences += [F.FIG4_NOT_FORMED.format(pairs=" and ".join(display(p) for p in missing))] if missing else []
    return {"title": checked(title), "sentences": sentences}


def selected_carrier(pair, carrier) -> bool:
    """Whether the candidate carrying the best-candidate T (11b's notes) is the selected configuration: row A's point
    ("<encoder>_cells@500"), or a search candidate whose change from the base equals the frozen configuration's in every
    unit (stage 05's candidates and selection.json); another bag size is never the selected one."""
    if "@" in carrier:
        return carrier.split("@")[1] == str(T.FROZEN_NCELL)
    encoder, name = carrier.split("/")
    base = json.loads(json.dumps(search_space.base(encoder)))   # tuples as lists, as in selection.json
    for unit in common.primary_units(pair):
        frozen = common.read_json("05_search", pair, f"{encoder}/selection.json", unit)["frozen"]
        change = {k: v for k, v in frozen.items() if base.get(k) != v and k not in ("encoder", "micro_batch")}
        rows = pd.concat([common.read_kind("candidates", "05_search", pair, f, unit) for f in common.names("05_search", pair, f"{encoder}/candidates*_s0.parquet", unit)])
        if json.loads(rows.loc[rows["key"] == name, "change_value"].iloc[0]) != json.loads(json.dumps(change, default=str)):
            return False
    return True


def fig4a_sentences(lost_pairs) -> list:
    """The 07d sentence (FIG4A_FM_ALL_AVAILABLE) and the age-similarity n.e. sentence (pairs with an age-loss unit)."""
    out = []
    if not F.FIG4A_FM_ALL_AVAILABLE:
        out.append(F.FIGURE_TEXT["fig4a_fm_cut"])
    elif F.FIG4A_FM_ALL_AVAILABLE == ("scgpt",):
        out.append(F.FIGURE_TEXT["fig4a_scgpt_only"])
    else:
        assert F.FIG4A_FM_ALL_AVAILABLE == ("geneformer", "scgpt"), F.FIG4A_FM_ALL_AVAILABLE
    if lost_pairs:
        out.append(F.FIGURE_TEXT["fig4a_not_estimable"])
    return out


def fig6(licence) -> dict:
    """The title is the Results heading (config/figures.FIG6_TITLE); licence.json is still checked and names the rows."""
    assert licence["title"] == (P.LICENSED_TITLE if licence["licensed"] else P.DESCRIPTIVE_TITLE), licence["title"]
    named = ", ".join(common.label(k) for k in licence["named"]) or "none"
    return {"title": checked(F.FIG6_TITLE), "sentences": [F.FIG6_NAMED.format(n=P.LICENCE_MIN_PAIRS, names=named)]}


def fig5(criterion) -> dict:
    """FIG5_CRITERION (the per-encoder readings stay in the Methods), and the per-sample sentence iff a set encoder's
    selected configuration uses per-sample normalization."""
    per_encoder = "; ".join(f"{configurations.MODELS[m]['label']} Δ {number(criterion['delta'][m])} ({'passes' if criterion['passes'][m] else 'fails'})"
                            for m in criterion["set_encoders"])
    sentence = F.FIG5_CRITERION.format(margin=f"{criterion['margin']:g}", result="passed" if criterion["control_passes"] else "failed",
                                       per_encoder=per_encoder)
    per_sample = [F.FIGURE_TEXT["fig5d_per_sample"]] if "per_sample" in criterion["normalization"].values() else []
    return {"title": checked(F.TITLES["Fig 5"]), "sentences": [sentence, *per_sample]}


def main(pairs, panels):
    """Every item's title and sentences; the conditional ones only when their figure is fully drawn (Fig 2 and Fig 4 need
    stage 11/11b tables, which a blinded run doesn't write)."""
    out = {item: {"title": checked(title), "sentences": []} for item, title in F.TITLES.items()}
    whole = {item for item in F.PANELS if set(panels.get(item, ())) == set(F.PANELS[item])}
    if "Fig 2" in whole:
        pooled = common.pick(common.statistics(), neutral=False, pool_set="nonceiling_equal", readout="ridge", statistic="auroc_difference")
        out["Fig 2"] = fig2(common.claims(), pooled.set_index("key"))
    if "Fig 4" in whole:
        tests = common.pick(common.secondary(), neutral=False, row="best_candidate", key="best_candidate")
        tests = tests[tests["pair"] != "pooled"]
        carriers = common.read_json("11b_secondary", "pooled", f"{common.LABELLED}secondary_notes.json")["best_candidate"]
        significant = tests.loc[tests["p_better"] <= S11.ALPHA, "pair"]
        out["Fig 4"] = fig4(tests, {p: selected_carrier(p, carriers[p]["carrier"]) for p in significant})
        out["Fig 4"]["sentences"] += fig4a_sentences([p for p in pairs if figure4.age_loss_units(p)])
    if "Fig 6" in whole:
        out["Fig 6"] = fig6(common.read_json("12a_properties", "", "licence.json"))
    if "D" in panels.get("Fig 5", ()) and SIM.PAIR in pairs:
        out["Fig 5"] = fig5(common.read_json("12_simulation", SIM.PAIR, "criterion.json", SIM.UNIT))
    partial = list(pairs) != list(PAIR_ORDER)
    name = "figure_text_partial.json" if partial else "figure_text.json"
    record = {"partial": partial, "pairs": list(pairs), "items": out}
    common.write_bytes(common.root() / name, lambda p: p.write_text(json.dumps(record, indent=1, ensure_ascii=False)))
    return out
