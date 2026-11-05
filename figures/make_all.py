"""Every figure, figure_text.json and (real runs) Additional files 3–6, from finished stage outputs. One process, so each stage output is read once.

Modes, by settings.BLINDED:
  real (unset; after the analysis-fixed tag): every panel; every input asserted unblinded; data files built;
  blinded: only config/figures.BLINDED_PANELS (a blinded run writes no stage 11/11b table), every page
           stamped "BLINDED", no data files; nothing under blinded_<seed>/ leaves the server except the PNG previews;
  pytest: every panel on a synthetic outputs tree (tests/test_figures_end_to_end.py), stamped "SYNTHETIC".

  python figures/make_all.py [--pairs PAIR ...] [--only ITEM ...]
--pairs (default every pair) is a fixture and smoke-run convenience: pairs as columns; the pair-fixed panels of absent
pairs are n/a. --only draws the named items (e.g. the rehearsal's label-free set) and builds no data files."""
import argparse
import time

import common
import data_files
import fig1
import fig2
import fig3
import fig4
import fig5
import fig6
import s01
import s02_s07_qc
import s08
import s09
import s10
import s11
import s12
import s13
import s14
import s16
import s17
import style
import titles
from sc2bulk.config import figures as F
from sc2bulk.config import settings
from sc2bulk.config.pairs import PAIR_ORDER

SCRIPTS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "s01": s01,
           "s02_s07_qc": s02_s07_qc, "s08": s08, "s09": s09, "s10": s10, "s11": s11, "s12": s12, "s13": s13, "s14": s14,
           "s16": s16, "s17": s17}


def drawn_panels(blinded) -> dict:
    return {item: (F.BLINDED_PANELS.get(item, ()) if blinded else tuple(panels)) for item, panels in F.PANELS.items()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", choices=PAIR_ORDER, default=list(PAIR_ORDER))
    parser.add_argument("--only", nargs="+", choices=[i for i, spec in F.ITEMS.items() if spec["script"] in SCRIPTS])
    args = parser.parse_args(argv)
    pairs = [p for p in PAIR_ORDER if p in args.pairs]
    blinded = settings.BLINDED is not None
    style.apply()
    if blinded:
        common.WATERMARK = style.BLINDED_WATERMARK
    panels = drawn_panels(blinded)
    if args.only:
        panels = {item: (p if item in args.only else ()) for item, p in panels.items()}
    for item, spec in F.ITEMS.items():
        assert spec["script"] in (*SCRIPTS, "data_files", None), f"{item}: no script {spec['script']}"
    for name, module in SCRIPTS.items():
        items = [i for i, spec in F.ITEMS.items() if spec["script"] == name and (spec.get("pair") is None or spec["pair"] in pairs)]
        if not any(panels[i] for i in items):
            print(f"{name}: nothing to draw in this mode", flush=True)
            continue
        start = time.perf_counter()
        module.main(pairs, panels)
        print(f"{name}: {time.perf_counter() - start:.1f} s", flush=True)
    expected = {(item, p) for item, ps in panels.items() for p in ps if F.ITEMS[item].get("pair") in (None, *pairs)}
    assert expected <= common.WRITTEN, f"panels without a CSV: {sorted(expected - common.WRITTEN)}"
    if args.only:   # a partial run leaves figure_text.json and the data files as they are
        return
    titles.main(pairs, panels)
    if not blinded:
        start = time.perf_counter()
        data_files.main(pairs)
        print(f"data_files: {time.perf_counter() - start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
