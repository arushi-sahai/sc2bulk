"""The timing test's job lists, one wave at a time: smoke (one job per block on Pineda (ALS), the smallest pair, incl. the ScRAT environment's equality
check; the 02b host-gather sweep opens wave 2, since it needs block F's caps), wave1 (the equality gate and the memory sweep,
before the micro-batch caps are pinned), wave2 (bases, candidates, the Fig 4A sweep, scVI). Each line:
<environment> <time box, seconds or 0> <05t_timing.py arguments>; longest first.

  python scripts/timing_jobs.py WAVE > jobs.txt
"""
import sys

from sc2bulk import folds
from sc2bulk.config import search_space
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

ENV = {"cytoset": "sc2bulk", "set_transformer": "sc2bulk", "scrat": "scrat", "scvi": "scvi"}
ENCODERS = ("cytoset", "set_transformer", "scrat")
SLOW = ("soundlife", "crossfit_0")
SMALL = ("hammond", "all_0")
SIX_HOURS = 6 * 3600


def primary(pair):
    first = next(iter(PAIRS[pair]["fold_sets"]))
    return folds.unit_name(*next(u for u in folds.units(pair) if u[0] == first))


def line(model, box, block, pair, unit, extra=""):
    return f"{ENV[model]} {box} --block {block} --model {model} --pair {pair} --unit {unit}{extra}"


def wave(name, pairs):
    if name == "smoke":
        return ([line(m, 3600, "candidate", "pineda", "all_0", f" --candidate {next(iter(search_space.GROUPS_BY_ENCODER[m]['lr']))}")
                 for m in ENCODERS] + [line("cytoset", 3600, "memory", "pineda", "all_0"), line("cytoset", 3600, "equality", "pineda", "all_0"),
                                       line("scrat", 3600, "equality", "pineda", "all_0"), line("scvi", 3600, "scvi", "pineda", "all_0")])
    if name == "wave1":
        jobs = [line(m, 0, "equality", *unit) for unit in (SLOW, SMALL) for m in ("cytoset", "scrat")]
        from importlib.util import module_from_spec, spec_from_file_location
        from sc2bulk.config import settings
        spec = spec_from_file_location("timing", settings.REPO / "pipeline" / "05t_timing.py")
        timing = module_from_spec(spec)
        spec.loader.exec_module(timing)
        return jobs + [line(m, SIX_HOURS, "memory", *SLOW, f" --ncell {n} --variant {v}") for n in ("all", "1000", "500")
                       for m in ENCODERS for v in range(len(timing.WIDEST[m]))]
    jobs = [line("cytoset", 3600, "sweep", "pineda", "all_0", " --ncell all")]   # the 02b host-gather path first (needs block F's caps)
    jobs += [line(m, SIX_HOURS, "sweep", pair, unit, f" --ncell {n}") for n in ("all", "1000") for pair, unit in (SLOW, SMALL)
             for m in ENCODERS]
    jobs += [line(m, 0, "base", pair, primary(pair), f" --seed {s}") for pair in pairs for m in ENCODERS for s in range(5)]
    jobs += [line(m, 0, "candidate", pair, unit, f" --candidate {c}") for pair, unit in (SLOW, SMALL) for m in ENCODERS
             for group in search_space.GROUPS_BY_ENCODER[m].values() for c in group]
    jobs += [line("scvi", 0, "scvi", pair, primary(pair), f" --candidate {c}") for pair in pairs
             for group in search_space.SCVI_GROUPS.values() for c in group]
    jobs += [line("scvi", 0, "scvi", pair, primary(pair)) for pair in pairs]
    order = {"soundlife": 0, "crohns": 1}   # the largest units first
    return jobs[:1] + sorted(jobs[1:], key=lambda j: (" sweep " not in j, order.get(j.split("--pair ")[1].split()[0], 2)))


if __name__ == "__main__":
    pairs = sys.argv[2].split(",") if len(sys.argv) > 2 else list(PAIR_ORDER)
    print("\n".join(wave(sys.argv[1], pairs)))
