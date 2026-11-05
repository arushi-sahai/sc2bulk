"""The stage 05/06 GPU scheduler on synthetic queues: a fake server (launcher,
clock, memory, GPUs, scopes), no GPU or systemd. Stage 06's --list itself is tested in test_train06.py."""
import importlib.util
import json
import os
import random
import re
import runpy
import sys
from pathlib import Path

import pytest

from sc2bulk import codehash, folds
from sc2bulk.config import settings
from sc2bulk.config.pairs import PAIR_ORDER

SCRIPT = settings.REPO / "scripts" / "schedule.py"
RUN_ALL_STAGES = runpy.run_path(str(settings.REPO / "pipeline" / "run_all.py"), run_name="run_all")["STAGES"]
SWITCH = re.compile(r"\bBLINDED\b|\bLABELLED_OUTPUTS\b|SC2BULK_PERMUTE_BULK_LABELS")   # as test_permutation
UNIT_NAME = re.compile(r"^[A-Za-z0-9:_.\\-]+$")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


os.environ.setdefault("SC2BULK_SERVER_ROOT", "/server")   # read when schedule.py loads
G = load("schedule", SCRIPT)


class Process:
    def __init__(self, system, end, status):
        self.system, self.end, self.status = system, end, status

    def poll(self):
        return self.status if self.system.clock >= self.end else None


class Server:
    """The fake server: each launch runs for durations[id] seconds (default 60) and exits with the next of
    statuses[id] (default 0); memory, GPU free memory, disk and active scopes are set by the test."""

    def __init__(self, durations=None, statuses=None, mem=1000.0, disk=500.0):
        self.clock, self.durations, self.statuses = 0.0, durations or {}, {k: list(v) for k, v in (statuses or {}).items()}
        self.mem, self.disk, self.current, self.foreign, self.scopes, self.lists = mem, disk, {}, {}, [], {}
        self.launched = []   # (id, args, env, cwd, time)

    def now(self):
        return self.clock

    def sleep(self, seconds):
        self.clock += seconds

    def launch(self, args, env, cwd, out, err):
        if "--list" in args:
            job_id = f"list-{args[args.index('--pair') + 1]}"
            out.write(self.lists[job_id[5:]])
        else:
            job_id = next(a for a in args if a.startswith("--unit=sc2b-"))[len("--unit=sc2b-"):]   # a retry's ends in -retry
            out.write("maxrss_kb=1048576 elapsed_s=1\n")
        self.launched.append((job_id, args, env, cwd, self.clock))
        key = job_id.removesuffix("-retry")
        status = self.statuses[key].pop(0) if self.statuses.get(key) else 0
        return Process(self, self.clock + self.durations.get(key, 60), status)

    def run(self, args, env=None, cwd=None):
        if args[:2] == ["git", "status"]:
            return ""
        if args[:2] == ["git", "rev-parse"]:
            return "abc123\n"
        assert args[1] == "-c", args
        return f"{cwd}/src/sc2bulk\n{cwd}/third_party/cytoset\n{cwd}/third_party/set_transformer\n"

    def mem_available(self):
        return self.mem

    def memory_current(self, unit):
        return self.current.get(unit, 0.0)

    def gpu_free(self, gpu):
        return 24.0 - self.foreign.get(gpu, 0.0)

    def disk_free(self):
        return self.disk

    def active_scopes(self):
        return list(self.scopes)


def environment(monkeypatch):
    for key in G.PASSED:
        monkeypatch.setenv(key, f"/{key.lower()}")


def write_limits(logs, **changes):
    limits = {**G.DEFAULT_LIMITS, "gpus": [0, 1, 2, 3], "budget_gib": 84,
              "clones": {"05": str(logs / "launch05"), "06": str(logs / "launch06")}, **changes}
    (logs / "limits.json").write_text(json.dumps(limits))


def scheduler(tmp_path, monkeypatch, jobs=None, server=None, **limits):
    """A scheduler on the fake server; with jobs, only those (every pair's stage 06 counted as listed)."""
    environment(monkeypatch)
    G.prepare(tmp_path)
    write_limits(tmp_path, **limits)
    s = G.Scheduler(tmp_path, server or Server())
    if jobs is not None:
        s.jobs = {j.id: j for j in jobs}
        s.listed = {pair: "listed" for pair in PAIR_ORDER}
        s.selects = {pair: [] for pair in PAIR_ORDER}
    return s


def job(job_id, host="S", gpu="small", tier=2, minutes=10.0, after=(), pair="hammond"):
    return G.Job(job_id, "05_search.py", ["--x"], "sc2bulk", "05", pair, host, gpu, tier, minutes, list(after))


def run_until(s, seconds, step=20, each=None):
    end = s.system.clock + seconds
    while s.system.clock < end:
        s.tick()
        if each:
            each(s)
        s.system.clock += step


def gpus_of(s):
    on = {}
    for j in s.running():
        on.setdefault(j.gpu_index, []).append(j)
    return on


def line(unit, row, model, seed, point="", env="main", done=False, after=()):
    return json.dumps({"unit": unit, "row": row, "model": model, "seed": seed, "point": point, "env": env, "done": done,
                       "after": [list(a) for a in after]})


# ---- 1. stage 05 and 05v enumeration ----

def test_stage05_enumeration_matches_the_drivers(monkeypatch):
    calls = []
    search = load("stage05", settings.REPO / "pipeline" / "05_search.py")
    scvi = load("stage05v", settings.REPO / "pipeline" / "05v_scvi_search.py")
    monkeypatch.setattr(search, "job", lambda phase, pair, unit, encoder, seed: calls.append(
        ("05_search.py", phase, pair, unit and folds.unit_name(*unit), encoder, seed)))
    monkeypatch.setattr(scvi, "job", lambda phase, pair, unit, seed: calls.append(
        ("05v_scvi_search.py", phase, pair, folds.unit_name(*unit), None, seed)))
    monkeypatch.setattr(search.determinism, "configure", lambda: None)
    for pair in PAIR_ORDER:
        for module, extra in ((search, []), (search, ["--scrat"]), (scvi, [])):
            monkeypatch.setattr(sys, "argv", ["x", "--pair", pair, *extra])
            module.main()
    ours = []
    for pair in PAIR_ORDER:
        for j in G.stage05_jobs(pair):
            a = dict(zip(j.args[::2], j.args[1::2]))
            ours.append((j.script, a["--phase"], pair, a.get("--set"), a.get("--encoder"), int(a["--seed"]) if "--seed" in a else None))
    assert sorted(ours, key=str) == sorted(calls, key=str) and len(ours) == len(set(ours))
    assert sum(c[1] == "train" and c[0] == "05_search.py" for c in calls) == 10 * 3 * G.K


# ---- 3. readiness ----

def test_select_and_modal_wait_for_their_upstream(tmp_path, monkeypatch):
    s = scheduler(tmp_path, monkeypatch)
    trains = [s.jobs[f"05-train-soundlife-crossfit_0-cytoset-s{i}"] for i in range(G.K)]
    select = s.jobs["05-select-soundlife-crossfit_0-cytoset"]
    assert set(select.after) == {t.id for t in trains}
    for t in trains[:-1]:
        t.status = "done"
    assert not s.ready(select)
    trains[-1].status = "done"
    assert s.ready(select)
    modal = s.jobs["05-modal-soundlife-cytoset"]
    assert sorted(modal.after) == sorted(f"05-select-soundlife-crossfit_{f}-cytoset" for f in range(4))
    assert "05-modal-hammond-cytoset" not in s.jobs   # no crossfit units
    assert set(s.jobs["05v-select-lupus-all_0"].after) == {f"05v-train-lupus-all_0-s{i}" for i in range(G.K)}


def test_stage06_listed_after_the_pairs_selects_and_clone(tmp_path, monkeypatch):
    server = Server()
    server.lists["pineda"] = line("all_0", "A", "cytoset", 0) + "\n"
    s = scheduler(tmp_path, monkeypatch, server=server, gpus=[])
    selects = s.selects["pineda"]
    assert len(selects) == 3 + 1
    for sid in selects[:-1]:
        s.jobs[sid].status = "done"
    s.tick()
    assert s.listed["pineda"] is None and not server.launched
    s.jobs[selects[-1]].status = "done"
    write_limits(tmp_path, gpus=[], clones={"05": str(tmp_path / "launch05"), "06": None})
    s.tick()
    assert s.listed["pineda"] is None and not server.launched
    write_limits(tmp_path, gpus=[])
    s.tick()
    assert s.listed["pineda"] == "running"
    (job_id, args, env, cwd, _), = server.launched
    assert job_id == "list-pineda" and args[-3:] == ["--pair", "pineda", "--list"] and cwd == str(tmp_path / "launch06")
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["PYTHONPATH"] == f"{tmp_path / 'launch06'}/src:{tmp_path / 'launch06'}/third_party"
    assert s.used(s.limits()) == s.limits()["caps_gib"]["S"]   # the --list counts against the budget
    server.clock += 60
    s.tick()
    assert s.listed["pineda"] == "listed" and "06-pineda-all_0-A-cytoset-s0" in s.jobs


def test_stage06_rows_wait_for_their_upstream(tmp_path, monkeypatch):
    s = scheduler(tmp_path, monkeypatch, jobs=[])
    a, j = ("all_0", "A", "cytoset", 3, ""), ("all_0", "J", "scvi", 3, "")
    s.add_list("lupus", "\n".join([line(*a), line("all_0", "B", "cytoset", 3, after=[a]),
                                   line("all_0", "D", "cytoset", 3), line("all_0", "D", "cytoset_pseudobulk", 3, after=[("all_0", "D", "cytoset", 3, "")]),
                                   line(*j, env="scvi"), line("all_0", "F", "scvi", 3, "n100", env="scvi", after=[j]),
                                   line("all_0", "H", "scvi", 3, env="scvi", after=[j])]))
    b, pb, f, h = (s.jobs[k] for k in ("06-lupus-all_0-B-cytoset-s3", "06-lupus-all_0-D-cytoset_pseudobulk-s3",
                                       "06-lupus-all_0-F-scvi-n100-s3", "06-lupus-all_0-H-scvi-s3"))
    assert not any(s.ready(x) for x in (b, pb, f, h))
    for upstream in ("06-lupus-all_0-A-cytoset-s3", "06-lupus-all_0-D-cytoset-s3", "06-lupus-all_0-J-scvi-s3"):
        s.jobs[upstream].status = "done"
    assert all(s.ready(x) for x in (b, pb, f, h))


# ---- 4. admission ----

def test_running_caps_never_exceed_the_budget(tmp_path, monkeypatch):
    rng = random.Random(0)
    jobs = [job(f"j{i}", host=rng.choice("SLVH"), gpu=rng.choice(["small", "mid", "big", "light"]), tier=rng.randint(0, 4),
                minutes=rng.uniform(1, 100), pair=rng.choice(PAIR_ORDER)) for i in range(120)]
    server = Server(durations={j.id: rng.randint(20, 600) for j in jobs})
    s = scheduler(tmp_path, monkeypatch, jobs, server, pack=True, max_host=2)

    def check(s):
        started = [j for j in s.running() if j.started == s.system.clock]
        assert not started or sum(j.cap for j in s.running()) <= s.limits()["budget_gib"]
        if rng.random() < 0.1:
            write_limits(tmp_path, pack=True, max_host=2, budget_gib=rng.choice([44, 64, 84]))
    run_until(s, 40000, each=check)
    assert all(j.status == "done" for j in s.jobs.values())


def test_no_starvation_wave2_pattern(tmp_path, monkeypatch):
    """Four GPUs and a budget fitting two L jobs; L jobs interleaved in priority with S jobs, and more S jobs arriving
    all along. Each L job starts within the remaining run time of the jobs running when it first reserved."""
    big = [job(f"L{i}", host="L", gpu="big", minutes=100 - 10 * i, pair="soundlife") for i in range(6)]
    small = [job(f"S{i}", host="S", gpu="small", minutes=95 - 10 * i) for i in range(6)]
    durations = {**{j.id: 1800 for j in big}, **{f"S{i}": 300 for i in range(200)}}
    s = scheduler(tmp_path, monkeypatch, big + small, Server(durations=durations), budget_gib=76)
    arrived, reserved_at = [6], {}

    def stream(s):
        if s.reservation and s.reservation[0] not in reserved_at:
            reserved_at[s.reservation[0]] = (s.system.clock, max(j.started + durations[j.id] for j in s.running()))
        if s.system.clock % 100 == 0:
            s.jobs[f"S{arrived[0]}"] = job(f"S{arrived[0]}", minutes=1)
            arrived[0] += 1
    run_until(s, 4 * 3600, each=stream)
    assert all(j.status == "done" for j in big)
    assert {k for k in reserved_at if k.startswith("L")}, "the pattern never blocked an L job"
    for job_id, (_, deadline) in reserved_at.items():
        assert s.jobs[job_id].started <= deadline + 20, job_id


def test_max_host_and_a_skipped_host_job_reserves_nothing(tmp_path, monkeypatch):
    jobs = [job(f"H{i}", host="H", gpu="light", tier=1, minutes=500) for i in range(3)] + [job("S0", tier=4)]
    s = scheduler(tmp_path, monkeypatch, jobs, max_host=2, budget_gib=120)
    s.tick()
    assert {j.id for j in s.running()} == {"H0", "H1", "S0"} and s.reservation is None
    write_limits(tmp_path, max_host=0, budget_gib=120)
    s.system.clock += 120
    s.jobs["S1"] = job("S1", tier=4)
    s.tick()
    assert {j.id for j in s.running()} == {"S1"} and s.reservation is None


def test_max_host_is_one_while_headline_rows_remain(tmp_path, monkeypatch):
    jobs = [job(f"H{i}", host="H", gpu="light", tier=1, minutes=500) for i in range(2)] + [job("x", tier=0), job("A", tier=3, after=["x"])]
    s = scheduler(tmp_path, monkeypatch, jobs, Server(durations={"x": 10 ** 6, "H0": 10 ** 6, "H1": 10 ** 6}), max_host=2, budget_gib=120)
    s.tick()
    assert sum(j.host == "H" for j in s.running()) == 1
    s.jobs["A"].status = "done"
    s.tick()
    assert sum(j.host == "H" for j in s.running()) == 2


def test_reserved_gpu_never_takes_a_lower_job(tmp_path, monkeypatch):
    jobs = [job("big0", gpu="big", host="L", minutes=100), job("big1", gpu="big", host="L", minutes=90), job("small", minutes=1)]
    s = scheduler(tmp_path, monkeypatch, jobs, Server(durations={"big0": 10 ** 6}), gpus=[0, 1], budget_gib=500)
    s.system.foreign = {1: 10}   # 14 GiB free on GPU 1: too little for big1, which reserves it; enough for small
    s.tick()
    assert s.reservation[0] == "big1" and s.reservation[2] == 1 and s.jobs["small"].status == "queued"
    del s.jobs["big1"]
    s.tick()
    assert s.jobs["small"].gpu_index == 1 and s.reservation is None


def test_no_start_without_memory_or_gpu_room(tmp_path, monkeypatch):
    s = scheduler(tmp_path, monkeypatch, [job("a", host="L", gpu="big", pair="crohns")], Server(mem=36.0))
    s.tick()
    assert s.jobs["a"].status == "queued"   # 32 + 5 > 36
    s.system.mem, s.system.foreign = 100.0, {g: 3 for g in range(4)}
    s.tick()
    assert s.jobs["a"].status == "queued"   # 21 GiB free < 22
    s.system.foreign = {0: 3, 1: 3, 2: 0, 3: 3}
    s.tick()
    assert s.jobs["a"].gpu_index == 2       # the next GPU with room, not a block


def test_headroom_counts_unrealized_caps(tmp_path, monkeypatch):
    server = Server(durations={"first": 10 ** 6}, mem=60.0)
    s = scheduler(tmp_path, monkeypatch, [job("first", host="L", gpu="big", pair="crohns", minutes=50)], server)
    s.tick()
    server.current["sc2b-first"] = 5.0
    s.jobs["second"] = job("second", host="L", gpu="big", pair="crohns", minutes=40)
    s.tick()
    assert s.jobs["second"].status == "queued"   # MemAvailable 60 alone would allow 32 + 5, but "first" may still grow by 27
    server.current["sc2b-first"] = 32.0
    s.tick()
    assert s.jobs["second"].status == "running"


def test_reservation_holds_headroom(tmp_path, monkeypatch):
    """The reserved job's cap is held out of the headroom too, so a lower job can't take the memory it waits for."""
    server = Server(mem=30.0)
    s = scheduler(tmp_path, monkeypatch, [job("big", host="L", gpu="big", pair="crohns", minutes=50), job("small", minutes=1)], server,
                  budget_gib=500)
    s.tick()
    assert s.reservation[0] == "big" and s.jobs["small"].status == "queued"   # 30 - 32 < 12 + 5
    server.mem = 100.0
    s.tick()
    assert s.jobs["big"].status == "running" and s.jobs["small"].status == "running"


def test_per_pair_large_caps(tmp_path, monkeypatch):
    s = scheduler(tmp_path, monkeypatch, [job("c", host="L", gpu="big", pair="crohns"), job("l", host="L", gpu="big", pair="soundlife"),
                                         job("h", host="L", gpu="big", pair="hammond")],
                  caps_gib={**G.DEFAULT_LIMITS["caps_gib"], "L": {"soundlife": 20, "crohns": 24}})
    limits = s.limits()
    assert [s.cap(s.jobs[k], limits) for k in "clh"] == [24, 20, 24]


def test_job_with_cap_above_budget_is_skipped(tmp_path, monkeypatch):
    big = job("big", host="H", gpu="light", tier=0, minutes=100)
    big.retry = True   # 57 GiB
    s = scheduler(tmp_path, monkeypatch, [big, job("low", tier=4)], budget_gib=44)
    s.tick()
    assert big.status == "queued" and s.jobs["low"].status == "running" and s.reservation is None
    assert "cap above budget" in (tmp_path / "schedule.log").read_text()


# ---- 5. order ----

def test_order_tiers_longest_first_pineda_and_headline(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "age_loss", lambda pair, unit: False)
    lines = {pair: "\n".join(line(unit, row, "cytoset", seed) for row, seed in (("A", 0), ("D", 0), ("A", 12)))
             for pair, unit in (("lupus", "all_0"), ("pineda", "all_0"))}
    s = scheduler(tmp_path, monkeypatch, jobs=[], gpus=[0])
    for pair, text in lines.items():
        s.add_list(pair, text)
    tiers = {j.id: j.tier for j in s.jobs.values()}
    assert tiers == {"06-pineda-all_0-A-cytoset-s0": 1, "06-pineda-all_0-D-cytoset-s0": 1, "06-pineda-all_0-A-cytoset-s12": 1,
                     "06-lupus-all_0-A-cytoset-s0": 3, "06-lupus-all_0-D-cytoset-s0": 4, "06-lupus-all_0-A-cytoset-s12": 3}
    order = []
    run_until(s, 3600, each=lambda s: order.extend(j.id for j in s.running() if j.id not in order))
    assert order[:3] == sorted(order[:3]) and all(o.startswith("06-pineda") for o in order[:3])
    assert order.index("06-lupus-all_0-A-cytoset-s0") < order.index("06-lupus-all_0-D-cytoset-s0")
    s05 = scheduler(tmp_path / "b", monkeypatch)
    queue = sorted((j for j in s05.jobs.values() if s05.ready(j)), key=lambda j: (j.tier, -j.minutes, PAIR_ORDER.index(j.pair), j.id))
    pineda = [i for i, j in enumerate(queue) if j.pair == "pineda"]
    assert pineda == list(range(len(pineda))) and {j.tier for j in queue[:len(pineda)]} == {1}   # Pineda's 05 and 05v trains first
    rest = queue[len(pineda):]
    assert [j.minutes for j in rest] == sorted((j.minutes for j in rest), reverse=True) and rest[0].pair == "soundlife"


def test_classes_of_stage06_jobs(monkeypatch):
    monkeypatch.setattr(G, "age_loss", lambda pair, unit: unit == "crossfit_1")
    of = lambda pair, unit, row, model, point="": (lambda j: (j.host, j.gpu, j.tier))(G.stage06_job(pair, json.loads(line(unit, row, model, 0, point))))
    assert of("soundlife", "crossfit_0", "F", "scrat", "n_all") == ("H", "light", 6)
    assert of("soundlife", "crossfit_0", "F", "scvi", "n_all") == ("H", "light", 6)
    assert of("soundlife", "crossfit_1", "F", "cytoset", "n1000") == ("H", "light", 4)
    assert of("soundlife", "crossfit_0", "F", "cytoset", "n1000") == ("L", "big", 4)
    assert of("soundlife", "crossfit_0", "J", "scvi") == ("V", "light", 3)
    assert of("crohns", "all_0", "F", "cytoset", "n_all") == ("L", "big", 4)
    assert of("tms", "all_0", "F", "set_transformer", "n_all") == ("L", "big", 4)
    assert of("tms", "all_0", "A", "cytoset") == ("S", "mid", 3)
    assert of("tms", "all_0", "B", "cytoset") == ("S", "mid", 3)
    assert of("lupus", "all_0", "A", "scrat") == ("S", "small", 3)
    assert of("lupus", "all_0", "G", "scrat") == ("S", "big", 5)
    assert of("hammond", "all_0", "F", "cytoset", "n1000") == ("S", "big", 4)
    assert of("hammond", "all_0", "F", "cytoset", "n600") == ("S", "small", 4)
    assert of("pineda", "all_0", "D", "cytoset_pseudobulk") == ("S", "small", 1)


# ---- 6. placement ----

def placement(tmp_path, monkeypatch, jobs, **limits):
    s = scheduler(tmp_path, monkeypatch, jobs, Server(durations={j.id: 10 ** 6 for j in jobs}), gpus=[0, 1], budget_gib=500, **limits)
    s.tick()
    return {g: sorted(j.gpu for j in on) for g, on in gpus_of(s).items()}


def test_placement_pack_and_colocate(tmp_path, monkeypatch):
    smalls = [job(f"s{i}", minutes=10 - i) for i in range(4)]
    assert placement(tmp_path / "a", monkeypatch, smalls) == {0: ["small"], 1: ["small"]}           # pack off: one per GPU
    assert placement(tmp_path / "b", monkeypatch, smalls, pack=True) == {0: ["small", "small"], 1: ["small", "small"]}
    mids = [job("m0", gpu="mid", minutes=9), job("m1", gpu="mid", minutes=8), job("s0", minutes=7), job("s1", minutes=6)]
    assert placement(tmp_path / "c", monkeypatch, mids, pack=True) == {0: ["mid"], 1: ["mid"]}       # 16 + 11 + 1 > 23
    light = lambda: [job("b", gpu="big", minutes=9), job("s", minutes=8), job("l", gpu="light", host="V", minutes=7)]
    assert placement(tmp_path / "d", monkeypatch, light()) == {0: ["big"], 1: ["light", "small"]}   # beside the small job
    assert placement(tmp_path / "e", monkeypatch, light(), colocate=False) == {0: ["big"], 1: ["small"]}
    light_mid = [job("m", gpu="mid", minutes=9), job("m2", gpu="mid", minutes=8), job("l", gpu="light", host="V", minutes=7)]
    assert placement(tmp_path / "f", monkeypatch, light_mid, pack=True) == {0: ["mid"], 1: ["mid"]}  # 7 + 16 + 1 > 23
    lights = [job(f"l{i}", gpu="light", host="V", minutes=9 - i) for i in range(3)]
    assert placement(tmp_path / "g", monkeypatch, lights, pack=True) == {0: ["light"], 1: ["light"]}


# ---- 7. commands and environment ----

def test_command_and_environment(tmp_path, monkeypatch):
    s = scheduler(tmp_path, monkeypatch, [job("05-train-tms-all_0-cytoset-s3", host="S", gpu="mid")])
    monkeypatch.setenv("SC2BULK_FULL_RUN", "1")
    monkeypatch.setenv("SC2BULK_PERMUTE_BULK_LABELS", "7")
    s.tick()
    (job_id, args, env, cwd, _), = s.system.launched
    clone = str(tmp_path / "launch05")
    assert args == ["systemd-run", "--user", "--scope", "-q", "--collect", "--unit=sc2b-05-train-tms-all_0-cytoset-s3", "-p", "MemoryMax=12G",
                    "-p", "MemorySwapMax=0", "/usr/bin/time", "-f", "maxrss_kb=%M elapsed_s=%e",
                    f"{G.ROOT}/miniconda3/envs/sc2bulk/bin/python", f"{clone}/pipeline/05_search.py", "--x"]
    assert cwd == clone
    expected = {"PATH", *G.PASSED, "TMPDIR", "XDG_CACHE_HOME", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", *G.THREAD_VARIABLES}
    assert set(env) == expected | ({"SC2BULK_SMOKE"} if settings.SMOKE else set())
    assert env["PYTHONPATH"] == f"{clone}/src:{clone}/third_party" and env["CUDA_VISIBLE_DEVICES"] == "0" and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert all(env[k] == str(settings.N_THREADS) for k in G.THREAD_VARIABLES)


def test_smoke_switch_and_outputs_pass_through(monkeypatch):
    environment(monkeypatch)
    monkeypatch.setenv("SC2BULK_OUTPUTS", "/out")
    monkeypatch.setattr(settings, "OUTPUTS", Path("/out/smoke"))
    for smoke in (True, False):
        monkeypatch.setattr(settings, "SMOKE", smoke)
        env = G.job_env("/clone", 2)
        assert env["SC2BULK_OUTPUTS"] == "/out" and ("SC2BULK_SMOKE" in env) is smoke and env.get("SC2BULK_SMOKE", "1") == "1"


def test_unit_names_unique_and_valid():
    ids = [j.id for pair in PAIR_ORDER for j in G.stage05_jobs(pair)]
    ids += [G.stage06_id(pair, "crossfit_0", row, model, seed, point) for pair in PAIR_ORDER for row in "ABCDEFGHIJ"
            for model in ("cytoset", "set_transformer", "scrat", "scvi", "cytoset_pseudobulk") for seed in (0, 19)
            for point in ("", "f25", "n1", "n_all")] + [G.stage06_id("tms", "all_0", "MSE", "cytoset", 4, "")]
    assert len(ids) == len(set(ids)) and all(UNIT_NAME.match(f"sc2b-{i}") and len(i) < 200 for i in ids)


# ---- 8. static checks ----

def test_outside_every_stage_and_label_blind():
    text = SCRIPT.read_text()
    assert not SWITCH.search(text) and "bulk_labels" not in text and "labelled_" not in text
    for stage, script, *_ in RUN_ALL_STAGES:
        path = settings.REPO / "pipeline" / script
        if path.exists():
            for pair in PAIR_ORDER:
                assert SCRIPT.resolve() not in codehash.files(path, pair, codehash.extra(stage, pair)), (stage, pair)
    assert set(G.SCRIPTS) == {"05_search.py", "05v_scvi_search.py", "06_train.py"}
    assert {j.script for pair in PAIR_ORDER for j in G.stage05_jobs(pair)} <= set(G.SCRIPTS)
    assert re.findall(r"pipeline/[\w{}.]+", text) == ["pipeline/{job.script}", "pipeline/06_train.py", "pipeline/06_train.py"]


# ---- 9. operations ----

def test_stop_drains_then_exits(tmp_path, monkeypatch):
    s = scheduler(tmp_path, monkeypatch, [job("a", minutes=9), job("b", minutes=8)], Server(durations={"a": 100}), gpus=[0])
    s.tick()
    (tmp_path / "STOP").touch()
    run_until(s, 200)
    assert s.jobs["a"].status == "done" and s.jobs["b"].status == "queued" and s.finished()
    assert len(s.system.launched) == 1


def test_limits_change_applies_next_tick(tmp_path, monkeypatch):
    s = scheduler(tmp_path, monkeypatch, [job(f"j{i}", minutes=9 - i) for i in range(3)], Server(durations={f"j{i}": 10 ** 6 for i in range(3)}),
                  gpus=[0])
    s.tick()
    assert len(s.running()) == 1
    write_limits(tmp_path, gpus=[0, 3])
    s.tick()
    assert sorted(j.gpu_index for j in s.running()) == [0, 3]
    (tmp_path / "limits.json").write_text("{")
    with pytest.raises(json.JSONDecodeError):
        s.tick()


def test_active_scopes_refuse_or_wait(tmp_path, monkeypatch):
    s = scheduler(tmp_path, monkeypatch, [])
    monkeypatch.setenv("SC2BULK_OUTPUTS", str(tmp_path))
    monkeypatch.setattr(s, "limits", lambda: {"clones": {"05": str(settings.REPO)}})
    s.system.scopes = ["sc2b-x.scope"]
    with pytest.raises(SystemExit):
        G.preflight(s, wait=False)

    def sleep(seconds):
        s.system.clock += seconds
        s.system.scopes = []
    monkeypatch.setattr(s.system, "sleep", sleep)
    s.system.scopes = ["sc2b-x.scope"]
    G.preflight(s, wait=True)
    assert s.system.clock == G.TICK


def test_dry_run_launches_nothing(tmp_path, monkeypatch):
    s = scheduler(tmp_path, monkeypatch)
    G.dry_run(s, {**G.DEFAULT_LIMITS, "clones": {"05": str(tmp_path / "none"), "06": None}})
    assert not s.system.launched


# ---- 10. retries and failures ----

def test_retry_solo_then_failed_and_blocked(tmp_path, monkeypatch):
    jobs = [job("05-select-x", host="L", gpu="big", tier=0, pair="lupus"), job("child", after=["05-select-x"]),
            job("grandchild", after=["child"]), job("other", minutes=1)]
    server = Server(statuses={"05-select-x": [137, 1]}, durations={"other": 10 ** 6})
    s = scheduler(tmp_path, monkeypatch, jobs, server, gpus=[0, 1], pack=True)
    s.selects["lupus"], s.listed["lupus"] = ["05-select-x"], None
    write_limits(tmp_path, gpus=[0, 1], pack=True, clones={"05": str(tmp_path / "launch05"), "06": None})
    s.tick()
    s.system.clock += 60
    s.tick()
    first, second = [l for l in server.launched if l[0].startswith("05-select-x")]
    assert second[0] == "05-select-x-retry" and "--collect" in second[1]   # its own scope: the failed one may still be loaded
    assert second[1] == [a.replace("MemoryMax=32G", "MemoryMax=48G").replace("sc2b-05-select-x", "sc2b-05-select-x-retry") for a in first[1]]
    assert s.headroom() == 1000 - 48 - 12 and server.current.setdefault("sc2b-05-select-x-retry", 48.0) and s.headroom() == 1000 - 12
    assert s.jobs["05-select-x"].retry and s.jobs["05-select-x"].gpu_index != s.jobs["other"].gpu_index
    s.system.clock += 60
    s.tick()
    assert s.jobs["05-select-x"].status == "failed"
    assert s.jobs["child"].status == "blocked" and s.jobs["grandchild"].status == "blocked" and s.listed["lupus"] == "blocked"
    assert "FAILED 05-select-x" in (tmp_path / "schedule.log").read_text()


def test_solo_retry_blocks_the_other_place(tmp_path, monkeypatch):
    jobs = [job("r", minutes=9), job("s1", minutes=2), job("s2", minutes=1)]
    jobs[0].retry = True
    assert placement(tmp_path, monkeypatch, jobs, pack=True) == {0: ["small"], 1: ["small", "small"]}


def test_no_retry_under_stop(tmp_path, monkeypatch):
    s = scheduler(tmp_path, monkeypatch, [job("a")], Server(statuses={"a": [143]}))
    s.tick()
    (tmp_path / "STOP").touch()
    s.system.clock += 60
    s.tick()
    assert s.jobs["a"].status == "stopped" and not s.jobs["a"].retry and len(s.system.launched) == 1 and s.finished()


# ---- 13. exit ----

def test_idles_until_stage06_then_done(tmp_path, monkeypatch):
    s = scheduler(tmp_path, monkeypatch, [job("a")])
    s.listed = {pair: None for pair in PAIR_ORDER}
    write_limits(tmp_path, clones={"05": str(tmp_path / "launch05"), "06": None})
    run_until(s, 300)
    assert s.jobs["a"].status == "done" and not s.finished()
    s.listed = {pair: "listed" if pair != "lupus" else "blocked" for pair in PAIR_ORDER}
    assert s.finished()
    s.system.sleep = lambda seconds: setattr(s.system, "clock", s.system.clock + seconds)
    s.loop()
    assert (tmp_path / "schedule.log").read_text().rstrip().endswith("DONE")
    assert json.loads((tmp_path / "state.json").read_text())["counts"]


def test_failed_list_exits_nonzero_and_a_restart_lists_again(tmp_path, monkeypatch):
    server = Server(statuses={"list-lupus": [1]})
    server.lists = {pair: "" for pair in PAIR_ORDER}
    s = scheduler(tmp_path, monkeypatch, jobs=[], server=server, gpus=[])
    s.listed = {pair: "listed" if pair != "lupus" else None for pair in PAIR_ORDER}
    s.jobs = {"05-select-x": job("05-select-x", pair="lupus")}
    s.jobs["05-select-x"].status = "done"
    s.selects["lupus"] = ["05-select-x"]
    s.system.sleep = lambda seconds: setattr(s.system, "clock", s.system.clock + seconds)
    with pytest.raises(SystemExit):
        s.loop()
    log = (tmp_path / "schedule.log").read_text()
    assert "stage 06 of lupus never listed" in log and "DONE" not in log and s.listed["lupus"] == "failed"
    restart = scheduler(tmp_path, monkeypatch, jobs=[], server=server, gpus=[])   # the same --list now exits 0
    restart.listed = {pair: "listed" if pair != "lupus" else None for pair in PAIR_ORDER}
    restart.jobs = {"05-select-x": job("05-select-x", pair="lupus")}
    restart.jobs["05-select-x"].status = "done"
    restart.selects["lupus"] = ["05-select-x"]
    restart.system.sleep = s.system.sleep
    restart.loop()
    assert restart.listed["lupus"] == "listed" and (tmp_path / "schedule.log").read_text().rstrip().endswith("DONE")
