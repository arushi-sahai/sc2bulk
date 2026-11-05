"""The stage 05/05v/06 GPU scheduler. One long-lived
process owns every GPU slot and the whole queue, and launches each job as one call of its stage's single-job
interface under its own systemd scope (MemoryMax = the job's host-class cap, no swap). It computes nothing a
stage records, reads no labels, and writes only its own logs. It keeps no state that matters: a restart
relaunches every stage 05 job (a finished one exits after its is_done check) and lists stage 06 again
(06_train.py --list), so whether a job is done is always the stage's own decision. No stage imports scripts/,
so this file is in no stage's code hash.

Admission: the running jobs' caps never sum above the budget; a start also needs headroom (MemAvailable
less the running jobs' unrealized caps), a GPU with room (The placement; nvidia-smi's free memory) and free
disk. The first blocked job in priority order reserves its cap and a GPU, so no lower job can starve it. A
failed job is retried once, solo, at 1.5x its cap; a second failure marks it FAILED and its dependents BLOCKED.
Everything that changes resources is in <logs>/limits.json, re-read every tick.

  python scripts/schedule.py --logs DIR [--wait]          run (on the server: setsid nohup, PYTHONPATH=<launch05>/src:<launch05>/third_party)
  python scripts/schedule.py --dry-run [--logs DIR]       start nothing; the plan (06_train --list where it can)
  python scripts/schedule.py --status --logs DIR          the running jobs and counts (state.json)
  python scripts/schedule.py --packing-check GPU_A GPU_B --logs DIR
                                                          three Hammond CytoSet train jobs, two packed on GPU_A
                                                          and one alone on GPU_B: the packed throughput gain
  touch DIR/STOP                                          drain: no new starts or retries, exit when idle
"""
import argparse
import dataclasses
import json
import os
import re
import runpy
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

import sc2bulk
from sc2bulk import folds, paths
from sc2bulk.config import settings, smoke
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIR_ORDER

ROOT = os.environ["SC2BULK_SERVER_ROOT"]   # the server working directory: miniconda3/, tmp/, .cache/, launch05/
MINICONDA = f"{ROOT}/miniconda3"
RUN_ALL = runpy.run_path(str(settings.REPO / "pipeline" / "run_all.py"), run_name="run_all")
ENVIRONMENTS = RUN_ALL["ENVIRONMENTS"]
SCRIPTS = ("05_search.py", "05v_scvi_search.py", "06_train.py")
K = smoke.N_SEARCH_SEEDS if settings.SMOKE else settings.N_SEARCH_SEEDS   # as 05_search and 05v_scvi_search
ENCODERS = ("cytoset", "set_transformer", "scrat")
PSEUDOBULK = "cytoset_pseudobulk"
BIG_PAIRS = ("soundlife", "crohns")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
PASSED = ("HOME", "USER", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "SC2BULK_OUTPUTS", "SC2BULK_SCRAT")
TICK = 20                 # seconds between ticks (and right after a job exits)
MIN_DISK_GIB = 20         # no starts below this much free disk under the outputs tree
HEADROOM_GIB = 5          # MemAvailable less the running jobs' unrealized caps must exceed a new cap by this
GPU_SUM_GIB = 23          # two jobs share a 24 GiB card only within this sum
CONTEXT_GIB = 0.5         # each process's CUDA context and cuBLAS workspace, outside PyTorch's reserved memory
GPU_GIB = {"small": 11, "mid": 16, "big": 22, "light": 7}
RETRY_FACTOR = 1.5
HOUR = 3600
DEFAULT_LIMITS = {"gpus": [0, 2, 3], "budget_gib": 64, "caps_gib": {"S": 12, "L": {"soundlife": 32, "crohns": 32}, "V": 28, "H": 38},
                  "max_host": 1, "pack": False, "colocate": True, "clones": {"05": f"{ROOT}/launch05", "06": None}}

# Estimated minutes, for longest-first order only: wave 2's block A medians per training
BASE_MINUTES = {"hammond": {"cytoset": 0.43, "set_transformer": 0.66, "scrat": 1.03},
                "tms": {"cytoset": 0.83, "set_transformer": 1.16, "scrat": 1.20},
                "soundlife": {"cytoset": 3.70, "set_transformer": 2.72, "scrat": 2.62},
                "lupus": {"cytoset": 1.06, "set_transformer": 0.79, "scrat": 1.23},
                "crohns": {"cytoset": 2.46, "set_transformer": 2.67, "scrat": 3.14},
                "pineda": {"cytoset": 0.62, "set_transformer": 0.75, "scrat": 1.25}}
CANDIDATE_RATIO = {"cytoset": 1.09, "set_transformer": 1.02, "scrat": 1.09}   # block B
TRAININGS_PER_SEED = 28
ALL_MINUTES = {"cytoset": 250, "set_transformer": 80, "scrat": 85}            # Sound Life "all" (projected)
ALL_RATIO = 10                                                                 # other pairs' "all" (Hammond's 9-15x)
RECIPE_MINUTES = {"cytoset": 90, "set_transformer": 20, "scrat": 20}          # row G (The recipe estimates)
SCVI_FITS = 4                                                                  # 05v: the base and three candidates


@dataclasses.dataclass
class Job:
    id: str
    script: str
    args: list
    env: str
    clone: str              # "05" or "06": the limits file's clone it runs from
    pair: str
    host: str               # host class: S, L, V or H
    gpu: str | None         # GPU class: small, mid, big or light; None for a CPU job
    tier: int
    minutes: float
    after: list
    status: str = "queued"  # queued, running, done, failed, blocked, stopped
    retry: bool = False     # the one solo retry
    process: object = None
    gpu_index: int | None = None
    cap: float = 0.0
    started: float = 0.0


def encoder_of(row, model):
    return "cytoset" if row in ("B", "C", "MSE") or model == PSEUDOBULK else model


def gpu_class(pair, encoder, host, row="", point=""):
    """The GPU class: light for the host gather and scVI; big for Sound Life, Crohn's, row F at >= 1,000 cells,
    "all" and row G; small for Hammond and Pineda, and lupus' and TMS's Set Transformer and ScRAT; else mid."""
    if host == "H" or encoder == "scvi":
        return "light"
    if pair in BIG_PAIRS or row == "G" or point == "n_all" or (row == "F" and int(point[1:]) >= 1000):
        return "big"
    return "mid" if pair in ("lupus", "tms") and encoder == "cytoset" else "small"


def stage05_jobs(pair) -> list:
    """The jobs 05_search.main() (with and without --scrat) and 05v_scvi_search.main() run for the pair, with
    their upstream jobs: a select after its k trains, modal after the crossfit selects."""
    out, units = [], folds.units(pair)
    host = "L" if pair in BIG_PAIRS else "S"
    train_tier = 1 if pair == "pineda" else 2   # the blinded rehearsal needs Pineda's stage 06 first
    for encoder in ENCODERS:
        env = ENVIRONMENTS["scrat" if encoder == "scrat" else "main"]
        base = BASE_MINUTES[pair][encoder]
        selects = []
        for unit in units:
            name = folds.unit_name(*unit)
            common = ["--pair", pair, "--set", name, "--encoder", encoder]
            trains = [Job(f"05-train-{pair}-{name}-{encoder}-s{i}", "05_search.py", ["--phase", "train", *common, "--seed", str(i)],
                          env, "05", pair, host, gpu_class(pair, encoder, host), train_tier,
                          TRAININGS_PER_SEED * CANDIDATE_RATIO[encoder] * base, []) for i in range(K)]
            select = Job(f"05-select-{pair}-{name}-{encoder}", "05_search.py", ["--phase", "select", *common], env, "05", pair, host,
                         gpu_class(pair, encoder, host), 0, K * base, [t.id for t in trains])
            out += [*trains, select]
            selects.append(select)
        crossfit = [s.id for s, unit in zip(selects, units) if unit[0] == "crossfit"]
        if crossfit:
            out.append(Job(f"05-modal-{pair}-{encoder}", "05_search.py", ["--phase", "modal", "--pair", pair, "--encoder", encoder],
                           env, "05", pair, "S", None, 0, 1, crossfit))
    host = "V" if pair in BIG_PAIRS else "S"
    scvi_minutes = 15 if pair in BIG_PAIRS else 3
    for unit in units:
        name = folds.unit_name(*unit)
        trains = [Job(f"05v-train-{pair}-{name}-s{i}", "05v_scvi_search.py", ["--phase", "train", "--pair", pair, "--set", name, "--seed", str(i)],
                      ENVIRONMENTS["scvi"], "05", pair, host, "light", train_tier, SCVI_FITS * scvi_minutes, []) for i in range(K)]
        out += [*trains, Job(f"05v-select-{pair}-{name}", "05v_scvi_search.py", ["--phase", "select", "--pair", pair, "--set", name],
                             ENVIRONMENTS["scvi"], "05", pair, host, "light", 0, K * scvi_minutes, [t.id for t in trains])]
    return out


def stage06_id(pair, unit, row, model, seed, point):
    return f"06-{pair}-{unit}-{row}-{model}" + (f"-{point}" if point else "") + f"-s{seed}"


def age_loss(pair, unit_name) -> bool:
    """Whether the unit's frozen CytoSet configuration has the age-similarity loss (stage 05's selection)."""
    unit = next(u for u in folds.units(pair) if folds.unit_name(*u) == unit_name)
    return bool(json.loads((paths.output("05_search", pair, "cytoset", unit) / "selection.json").read_text())["frozen"].get("age_similarity_loss"))


def stage06_job(pair, line: dict) -> Job:
    """One 06_train.py --list line as a job: its classes, tier and estimate."""
    unit, row, model, seed, point = line["unit"], line["row"], line["model"], line["seed"], line["point"]
    encoder = "scvi" if model == "scvi" else encoder_of(row, model)
    host_gather = pair == "soundlife" and (point == "n_all" or (row, model, point) == ("F", "cytoset", "n1000") and age_loss(pair, unit))
    if host_gather:   # 02b's store, or the age loss above 600 cells (06_train.on_device)
        host = "H"
    elif pair in BIG_PAIRS:
        host = "V" if model == "scvi" else "L"
    else:
        host = "L" if point == "n_all" else "S"
    if pair == "soundlife" and point == "n_all":   # 2026-10-22: human blood, aging's 3,000-cell point runs last
        tier = 6
    elif pair == "pineda":
        tier = 1
    elif row in ("A", "B", "C", "J"):   # the headline rows (C: 08b's untrained control); every seed stage 08 scores
        tier = 3
    elif row in ("D", "F", "H", "I"):   # main-figure rows: label-permuted (Fig 2), bag size (Fig 4a), one type (Fig 6c), per-sample route
        tier = 4
    else:                               # E (learning curves), G (own recipes), MSE: supplementary
        tier = 5
    if model == "scvi":
        minutes = 15 if pair in BIG_PAIRS else 3
    elif row == "G":
        minutes = RECIPE_MINUTES[encoder]
    elif point == "n_all":
        minutes = ALL_MINUTES[encoder] if pair == "soundlife" else ALL_RATIO * BASE_MINUTES[pair][encoder]
    elif row == "F":
        minutes = BASE_MINUTES[pair][encoder] * max(int(point[1:]), 100) / T.FROZEN_NCELL
    else:
        minutes = BASE_MINUTES[pair][encoder]
    args = ["--pair", pair, "--set", unit, "--row", row, "--model", model, "--seed", str(seed)] + (["--point", point] if point else [])
    job = Job(stage06_id(pair, unit, row, model, seed, point), "06_train.py", args, ENVIRONMENTS[line["env"]], "06", pair, host,
              gpu_class(pair, encoder, host, row, point), tier, minutes, [stage06_id(pair, *a) for a in line["after"]])
    if line["done"]:
        job.status = "done"
    return job


def python(env):
    return f"{MINICONDA}/envs/{env}/bin/python"


def job_env(clone, gpu) -> dict:
    """Exactly the allowlisted variables: nothing else in the scheduler's environment reaches a job."""
    env = {"PATH": f"{MINICONDA}/bin:/usr/bin:/bin", **{k: os.environ[k] for k in PASSED},
           "TMPDIR": f"{ROOT}/tmp", "XDG_CACHE_HOME": f"{ROOT}/.cache", "PYTHONPATH": f"{clone}/src:{clone}/third_party",
           "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           **{name: str(settings.N_THREADS) for name in THREAD_VARIABLES}}
    if settings.SMOKE:
        env["SC2BULK_SMOKE"] = "1"
    return env


def unit(job) -> str:
    """The job's systemd scope. The retry has its own name: the failed scope may still be loaded when the retry
    launches in the same tick (systemd's cgroup-empty notification is asynchronous)."""
    return f"sc2b-{job.id}" + ("-retry" if job.retry else "")


def command(job, cap, clone) -> list:
    return ["systemd-run", "--user", "--scope", "-q", "--collect", f"--unit={unit(job)}", "-p", f"MemoryMax={cap:g}G",
            "-p", "MemorySwapMax=0", "/usr/bin/time", "-f", "maxrss_kb=%M elapsed_s=%e", python(job.env),
            f"{clone}/pipeline/{job.script}", *job.args]


def read_limits(path) -> dict:
    """The live limits file; a malformed one stops the scheduler (fix it and restart)."""
    limits = json.loads(Path(path).read_text())
    assert set(limits) == set(DEFAULT_LIMITS), f"{path}: keys {sorted(limits)}, expected {sorted(DEFAULT_LIMITS)}"
    assert set(limits["caps_gib"]) == {"S", "L", "V", "H"} and set(limits["caps_gib"]["L"]) == set(BIG_PAIRS), f"{path}: caps_gib"
    assert set(limits["clones"]) == {"05", "06"}, f"{path}: clones"
    return limits


class System:
    """The server: launches, memory, GPUs, disk, scopes and the clock. The tests swap in a fake."""

    def now(self):
        return time.time()

    def sleep(self, seconds):
        time.sleep(seconds)

    def launch(self, args, env, cwd, out, err):
        return subprocess.Popen(args, env=env, cwd=cwd, stdout=out, stderr=err, stdin=subprocess.DEVNULL)

    def run(self, args, env=None, cwd=None) -> str:
        return subprocess.run(args, env=env, cwd=cwd, capture_output=True, text=True, check=True).stdout

    def mem_available(self) -> float:
        line = next(l for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))
        return int(line.split()[1]) / 2 ** 20

    def memory_current(self, unit) -> float:
        """The scope's memory in GiB; 0 when systemd has none (the scope just ended): its whole cap counts unrealized."""
        value = self.run(["systemctl", "--user", "show", "-p", "MemoryCurrent", "--value", f"{unit}.scope"]).strip()
        return int(value) / 2 ** 30 if value.isdigit() else 0.0

    def gpu_free(self, gpu) -> float:
        return int(self.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)])) / 1024

    def disk_free(self) -> float:
        return shutil.disk_usage(settings.OUTPUTS).free / 2 ** 30

    def active_scopes(self) -> list:
        text = self.run(["systemctl", "--user", "list-units", "sc2b-*", "--state=active", "--plain", "--no-legend"])
        return [line.split()[0] for line in text.splitlines() if line.strip()]


class Scheduler:
    def __init__(self, logs, system):
        self.logs, self.system = Path(logs), system
        self.jobs = {job.id: job for pair in PAIR_ORDER for job in stage05_jobs(pair)}
        self.selects = {pair: [j.id for j in self.jobs.values() if j.pair == pair and j.id.startswith(("05-select-", "05v-select-"))]
                        for pair in PAIR_ORDER}
        self.listed = {pair: None for pair in PAIR_ORDER}   # None, running, listed, failed (the --list exited non-zero), blocked
        self.listers = {}          # pair -> its running --list process
        self.checked = set()       # clones and (clone, env) pairs checked before their first launch
        self.said = {}             # hourly messages -> when last logged
        self.reservation = None
        self.maxrss = {}           # host class -> the largest maxrss seen (GiB)

    # ---- logs ----
    def say(self, message):
        line = f"{time.strftime('%m-%d %H:%M', time.localtime(self.system.now()))} {message}"
        with open(self.logs / "schedule.log", "a") as f:
            f.write(line + "\n")

    def hourly(self, message):
        if self.system.now() - self.said.get(message, -HOUR) >= HOUR:
            self.said[message] = self.system.now()
            self.say(message)

    def limits(self) -> dict:
        return read_limits(self.logs / "limits.json")

    # ---- state ----
    def running(self) -> list:
        return [j for j in self.jobs.values() if j.status == "running"]

    def ready(self, job) -> bool:
        return job.status == "queued" and all(self.jobs[a].status == "done" for a in job.after)

    def cap(self, job, limits) -> float:
        cap = limits["caps_gib"][job.host]
        if job.host == "L":   # per pair; a small pair's "all" takes the larger measured cap
            cap = cap[job.pair] if job.pair in cap else max(cap.values())
        return cap * RETRY_FACTOR if job.retry else cap

    def max_host(self, limits) -> int:
        """At most one host-gather job while any headline-row job (tier 3) is queued, running or not yet listed;
        afterwards the file's value (which also overrides downward, e.g. 0 to pause them)."""
        headline = any(j.tier == 3 and j.status in ("queued", "running") for j in self.jobs.values()) or \
            any(state not in ("listed", "failed", "blocked") for state in self.listed.values())
        return min(limits["max_host"], 1) if headline else limits["max_host"]

    # ---- one tick ----
    def tick(self):
        limits = self.limits()
        stop = (self.logs / "STOP").exists()
        self.reap(stop)
        self.reap_lists()
        if stop:
            self.hourly("no starts: STOP")
        else:
            self.start_lists(limits)
            self.admit(limits)
        self.write_state(limits)

    def reap(self, stop):
        for job in self.running():
            status = job.process.poll()
            if status is None:
                continue
            found = re.findall(r"maxrss_kb=(\d+)", (self.logs / "jobs" / f"{job.id}.log").read_text())
            maxrss = int(found[-1]) / 2 ** 20 if found else None
            if maxrss is not None:
                self.maxrss[job.host] = max(self.maxrss.get(job.host, 0), maxrss)
            minutes = (self.system.now() - job.started) / 60
            self.say(f"end gpu{job.gpu_index} status={status} {int(minutes // 60)}h{int(minutes % 60):02d}m "
                     f"maxrss {'-' if maxrss is None else f'{maxrss:.1f}G'} cap {job.cap:g}G {job.id}")
            job.process, job.gpu_index = None, None
            if status == 0:
                job.status = "done"
            elif stop:   # "stop now": the scopes were stopped on purpose; nothing is retried
                job.status = "stopped"
                self.say(f"stopped {job.id}")
            elif not job.retry:
                job.status, job.retry = "queued", True
                self.say(f"retry solo at {job.cap * RETRY_FACTOR:g}G {job.id}")
            else:
                job.status = "failed"
                self.say(f"FAILED {job.id}")
                self.block(job.id)

    def block(self, failed):
        """Every job downstream of a failed one never starts; a failed select also keeps its pair's stage 06 unlisted."""
        for job in self.jobs.values():
            if failed in job.after and job.status == "queued":
                job.status = "blocked"
                self.say(f"BLOCKED {job.id} (after {failed})")
                self.block(job.id)
        for pair, ids in self.selects.items():
            if failed in ids and self.listed[pair] is None:
                self.listed[pair] = "blocked"
                self.say(f"BLOCKED stage 06 of {pair} (after {failed})")

    # ---- stage 06's job list ----
    def start_lists(self, limits):
        """A pair's stage 06 is listed once every 05 and 05v select of the pair exited 0 and its clone is set: a
        background --list process (no scope, no GPU; class S counted against the budget)."""
        clone = limits["clones"]["06"]
        for pair in PAIR_ORDER:
            if self.listed[pair] is not None or clone is None or not all(self.jobs[s].status == "done" for s in self.selects[pair]):
                continue
            if self.used(limits) + limits["caps_gib"]["S"] > limits["budget_gib"]:
                return
            self.check(clone, ENVIRONMENTS["main"])
            with open(self.logs / "lists" / f"{pair}.jsonl", "w") as out, open(self.logs / "jobs" / f"06-list-{pair}.log", "a") as err:
                self.listers[pair] = self.system.launch([python(ENVIRONMENTS["main"]), f"{clone}/pipeline/06_train.py", "--pair", pair, "--list"],
                                                        job_env(clone, None), clone, out, err)
            self.listed[pair] = "running"
            self.say(f"list stage 06 of {pair}")

    def reap_lists(self):
        for pair, process in list(self.listers.items()):
            status = process.poll()
            if status is None:
                continue
            del self.listers[pair]
            if status != 0:   # e.g. a missing prerequisite (02b's index, 03b's splits): this pair only
                self.listed[pair] = "failed"
                self.say(f"FAILED list of {pair} status={status} (jobs/06-list-{pair}.log)")
                continue
            added = self.add_list(pair, (self.logs / "lists" / f"{pair}.jsonl").read_text())
            self.say(f"listed {pair}: {len(added)} stage 06 jobs, {sum(j.status == 'done' for j in added)} done")

    def add_list(self, pair, text) -> list:
        added = [stage06_job(pair, json.loads(line)) for line in text.splitlines()]
        assert all(job.id not in self.jobs for job in added), f"{pair}: stage 06 listed twice"
        self.jobs.update({job.id: job for job in added})
        self.listed[pair] = "listed"
        return added

    # ---- admission ----
    def used(self, limits) -> float:
        return sum(j.cap for j in self.running()) + len(self.listers) * limits["caps_gib"]["S"]

    def headroom(self) -> float:
        """MemAvailable less what the running jobs may still grow by (their caps less MemoryCurrent)."""
        return self.system.mem_available() - sum(max(0.0, j.cap - self.system.memory_current(unit(j))) for j in self.running())

    def can_share(self, job, other, limits) -> bool:
        """two jobs per GPU only within GPU_SUM_GIB (each with its CUDA context); a light job beside a small one
        when colocate is on; two small or mid jobs when pack is on; a solo retry beside nothing."""
        if job.retry or other.retry or GPU_GIB[job.gpu] + GPU_GIB[other.gpu] + 2 * CONTEXT_GIB > GPU_SUM_GIB:
            return False
        if "light" in (job.gpu, other.gpu):
            return limits["colocate"] and {job.gpu, other.gpu} == {"light", "small"}
        return limits["pack"] and {job.gpu, other.gpu} <= {"small", "mid"}

    def place(self, job, limits, reserved):
        """The first GPU in placement order (light jobs: beside a job first; others: an empty GPU first) whose
        nvidia-smi free memory covers the job; never the reserved GPU. None if none has room."""
        placed = {g: [j for j in self.running() if j.gpu_index == g] for g in limits["gpus"] if g != reserved}
        empty = [g for g, on in placed.items() if not on]
        beside = [g for g, on in placed.items() if len(on) == 1 and self.can_share(job, on[0], limits)]
        order = beside + empty if job.gpu == "light" else empty + beside
        return next((g for g in order if self.system.gpu_free(g) >= GPU_GIB[job.gpu]), None)

    def reserve_gpu(self, limits):
        """The GPU whose placed jobs have the least estimated time left (an empty one first)."""
        def left(g):
            return max([j.minutes * 60 - (self.system.now() - j.started) for j in self.running() if j.gpu_index == g], default=0)
        return min(limits["gpus"], key=lambda g: (left(g), limits["gpus"].index(g)), default=None)

    def admit(self, limits):
        disk = self.system.disk_free()
        if disk < MIN_DISK_GIB:
            self.hourly(f"no starts: {disk:.0f} GiB free on the outputs disk")
            return
        queue = sorted((j for j in self.jobs.values() if self.ready(j)), key=lambda j: (j.tier, -j.minutes, PAIR_ORDER.index(j.pair), j.id))
        if not queue:
            self.reservation = None
            return
        budget, free, headroom = limits["budget_gib"], limits["budget_gib"] - self.used(limits), self.headroom()
        max_host = self.max_host(limits)
        reservation = None
        for job in queue:
            cap = self.cap(job, limits)
            if job.host == "H" and sum(j.host == "H" for j in self.running()) >= max_host:
                continue
            if cap > budget:   # it could never start: reserving would stop the queue
                self.hourly(f"cap above budget ({cap:g} > {budget:g}G): {job.id}")
                continue
            held, held_gpu = reservation[1:] if reservation else (0, None)
            gpu = self.place(job, limits, held_gpu) if job.gpu else None
            if cap <= free - held and headroom - held >= cap + HEADROOM_GIB and (gpu is not None or job.gpu is None):
                self.start(job, gpu, cap, limits)
                free, headroom = free - cap, headroom - cap
            elif reservation is None:
                reservation = (job.id, cap, self.reserve_gpu(limits) if job.gpu else None)
        if reservation != self.reservation and reservation is not None:
            self.say(f"reserve {reservation[1]:g}G gpu{reservation[2]} for {reservation[0]}")
        self.reservation = reservation

    def check(self, clone, env):
        """Before the first launch from a clone: a clean tree (its commit logged); for each environment, that
        sc2bulk and the vendored cytoset and set_transformer import from the clone (an editable install, possibly stale or
        missing a package, could win over PYTHONPATH)."""
        if clone not in self.checked:
            assert not self.system.run(["git", "status", "--porcelain"], cwd=clone).strip(), f"{clone}: the tree isn't clean"
            self.say(f"clone {clone} at {self.system.run(['git', 'rev-parse', 'HEAD'], cwd=clone).strip()}")
            self.checked.add(clone)
        if (clone, env) not in self.checked:
            found = self.system.run([python(env), "-c", "import sc2bulk, cytoset, set_transformer\n"
                                     "for m in (sc2bulk, cytoset, set_transformer): print(m.__path__[0])"], env=job_env(clone, None), cwd=clone)
            for path in found.split():
                assert Path(path).is_relative_to(clone), f"{env}: {path} is outside {clone}"
            self.checked.add((clone, env))

    def start(self, job, gpu, cap, limits):
        clone = limits["clones"][job.clone]
        self.check(clone, job.env)
        with open(self.logs / "jobs" / f"{job.id}.log", "a") as log:
            job.process = self.system.launch(command(job, cap, clone), job_env(clone, gpu), clone, log, log)
        job.status, job.gpu_index, job.cap, job.started = "running", gpu, cap, self.system.now()
        self.say(f"start {'cpu' if gpu is None else f'gpu{gpu}'} cap {cap:g}G tier {job.tier}{' solo' if job.retry else ''} {job.id}")

    # ---- the loop ----
    def finished(self) -> bool:
        """Under STOP: once nothing runs. Otherwise once nothing runs or waits but FAILED and BLOCKED jobs and every
        pair's stage 06 is listed, blocked or its --list failed (while clones.06 is null it idles after stage 05)."""
        if self.running() or self.listers:
            return False
        if (self.logs / "STOP").exists():
            return True
        return all(j.status in ("done", "failed", "blocked") for j in self.jobs.values()) and \
            all(state in ("listed", "failed", "blocked") for state in self.listed.values())

    def loop(self):
        """Tick until finished: DONE, or a non-zero exit naming each pair whose stage 06 --list failed."""
        while True:
            self.tick()
            if self.finished():
                unlisted = [pair for pair, state in self.listed.items() if state == "failed"]
                if unlisted and not (self.logs / "STOP").exists():   # a restart after fixing the cause lists them again
                    message = "; ".join(f"stage 06 of {pair} never listed (jobs/06-list-{pair}.log)" for pair in unlisted)
                    self.say(message)
                    sys.exit(message)
                self.say("DONE")
                return
            for _ in range(TICK):
                self.system.sleep(1)
                if any(j.process.poll() is not None for j in self.running()):
                    break

    def write_state(self, limits):
        counts = {}
        for job in self.jobs.values():
            counts.setdefault(f"tier {job.tier}", {}).setdefault(job.status, 0)
            counts[f"tier {job.tier}"][job.status] += 1
        state = {"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.system.now())),
                 "running": [{"id": j.id, "gpu": j.gpu_index, "cap_gib": j.cap, "started": time.strftime("%m-%d %H:%M", time.localtime(j.started))}
                             for j in self.running()],
                 "counts": counts, "stage06": self.listed,
                 "maxrss_gib": {host: {"max": rss, "cap": limits["caps_gib"][host]} for host, rss in self.maxrss.items()},
                 "budget_used_gib": self.used(limits), "disk_free_gib": round(self.system.disk_free(), 1),
                 "reservation": self.reservation}
        (self.logs / ".state.json").write_text(json.dumps(state, indent=1))
        os.replace(self.logs / ".state.json", self.logs / "state.json")


def prepare(logs):
    for directory in (logs / "jobs", logs / "lists"):
        directory.mkdir(parents=True, exist_ok=True)
    if not (logs / "limits.json").exists():
        (logs / "limits.json").write_text(json.dumps(DEFAULT_LIMITS, indent=1))


def preflight(scheduler, wait):
    """Before any start: the outputs root and sc2bulk's clone, the stages' label-blindness, and no sc2b-* scope
    left from an earlier run (so no job runs twice at once)."""
    assert os.environ["SC2BULK_OUTPUTS"], "SC2BULK_OUTPUTS is required"
    clone = scheduler.limits()["clones"]["05"]
    assert Path(sc2bulk.__file__).resolve().is_relative_to(Path(clone).resolve()), f"sc2bulk imports from {sc2bulk.__file__}, not {clone}"
    assert not any(labels for _, script, _, _, labels, _ in RUN_ALL["STAGES"] if script in SCRIPTS), "a scheduled stage reads bulk labels"
    active = scheduler.system.active_scopes()
    while active and wait:
        scheduler.system.sleep(TICK)
        active = scheduler.system.active_scopes()
    if active:
        sys.exit(f"active scopes from an earlier run: {' '.join(active)} (wait for them, or --wait)")


def dry_run(scheduler, limits):
    """Start nothing; run 06_train --list (read-only) for the pairs whose selections all exist."""
    for key, clone in limits["clones"].items():
        commit = scheduler.system.run(["git", "rev-parse", "HEAD"], cwd=clone).strip() if clone and Path(clone).exists() else "-"
        print(f"clone {key}: {clone} {commit}")
    print("job environment:", ", ".join(["PATH", *PASSED, "TMPDIR", "XDG_CACHE_HOME", "PYTHONPATH", "CUDA_VISIBLE_DEVICES",
                                         "CUDA_DEVICE_ORDER", *THREAD_VARIABLES] + (["SC2BULK_SMOKE"] if settings.SMOKE else [])))
    print("limits:", json.dumps(limits))
    for pair in PAIR_ORDER:
        missing = [s for s in scheduler.selects[pair] if not selection_file(scheduler.jobs[s]).exists()]
        if missing or limits["clones"]["06"] is None:
            print(f"stage 06 {pair}: not listed ({len(missing)} selections missing{'; clones.06 unset' if limits['clones']['06'] is None else ''})")
            continue
        clone = limits["clones"]["06"]
        scheduler.add_list(pair, scheduler.system.run([python(ENVIRONMENTS["main"]), f"{clone}/pipeline/06_train.py", "--pair", pair, "--list"],
                                                      env=job_env(clone, None), cwd=clone))
        print(f"stage 06 {pair}: listed")
    counts = {}
    for job in scheduler.jobs.values():
        status = job.status if job.status == "done" else "ready" if scheduler.ready(job) else "waiting"
        key = (job.tier, job.script.removesuffix(".py"), job.host, job.gpu or "cpu", status)
        counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts, key=str):
        print("tier {} {} host {} gpu {} {}: {}".format(*key, counts[key]))
    pending = [j for j in scheduler.jobs.values() if j.status != "done"]
    print(f"estimated GPU-hours: {sum(j.minutes for j in pending if j.gpu) / 60:.0f}")
    order = sorted(pending, key=lambda j: (not scheduler.ready(j), j.tier, -j.minutes, PAIR_ORDER.index(j.pair), j.id))
    for job in order[:30]:
        print(f"  tier {job.tier} {job.host} {scheduler.cap(job, limits):g}G {job.gpu or 'cpu'} {job.minutes:.0f} min {job.id}")


def selection_file(select: Job):
    """A 05 or 05v select's selection.json (the dry run's "can stage 06 be listed")."""
    arg = dict(zip(select.args[::2], select.args[1::2]))
    unit = next(u for u in folds.units(select.pair) if folds.unit_name(*u) == arg["--set"])
    if select.script == "05v_scvi_search.py":
        return paths.output("05v_scvi_search", select.pair, "selection.json", unit)
    return paths.output("05_search", select.pair, arg["--encoder"], unit) / "selection.json"


def packing_check(scheduler, gpu_a, gpu_b):
    """Hammond CytoSet train seeds 0 and 1 together on GPU_A, seed 2 alone on GPU_B (real stage 05 work, kept).
    The gain is 2 x the median, over the configurations and both packed seeds, of alone / packed seconds per epoch."""
    limits = scheduler.limits()
    unit = folds.units("hammond")[0]
    jobs = [scheduler.jobs[f"05-train-hammond-{folds.unit_name(*unit)}-cytoset-s{i}"] for i in range(3)]
    for job, gpu in zip(jobs, (gpu_a, gpu_a, gpu_b)):
        scheduler.start(job, gpu, scheduler.cap(job, limits), limits)
    while any(j.process.poll() is None for j in jobs):
        scheduler.system.sleep(TICK)
    assert all(j.process.poll() == 0 for j in jobs), "a packing-check job failed (jobs/<id>.log)"
    per_epoch = [pd.read_parquet(paths.output("05_search", "hammond", "cytoset", unit) / f"candidates_s{i}.parquet").set_index("key")
                 for i in range(3)]
    per_epoch = [t["seconds"] / t["epochs"] for t in per_epoch]
    ratios = [*(per_epoch[2] / per_epoch[0]), *(per_epoch[2] / per_epoch[1])]
    gain = 2 * statistics.median(ratios)
    scheduler.say(f"packing check: gain {gain:.2f} over {len(ratios)} matched configurations -> pack {str(gain >= 1.3).lower()}")
    print(f"packed throughput gain {gain:.2f} (2 x median alone/packed s per epoch, {len(ratios)} pairs); rule: pack = gain >= 1.3")


def status(logs):
    state = json.loads((logs / "state.json").read_text())
    print(state["time"], f"budget used {state['budget_used_gib']:g}G, disk free {state['disk_free_gib']:g}G, reservation {state['reservation']}")
    for job in state["running"]:
        print(f"  gpu{job['gpu']} {job['cap_gib']:g}G since {job['started']} {job['id']}")
    for tier, counts in sorted(state["counts"].items()):
        print(f"  {tier}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("  stage 06:", ", ".join(f"{p} {s}" for p, s in state["stage06"].items()))
    for host, seen in state["maxrss_gib"].items():
        print(f"  class {host}: max rss {seen['max']:.1f}G, cap {seen['cap']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--wait", action="store_true", help="wait for earlier sc2b-* scopes to end instead of refusing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--packing-check", nargs=2, type=int, metavar=("GPU_A", "GPU_B"))
    args = parser.parse_args()
    if args.dry_run:
        limits = read_limits(args.logs / "limits.json") if args.logs and (args.logs / "limits.json").exists() else DEFAULT_LIMITS
        dry_run(Scheduler(args.logs or Path("."), System()), limits)
        return
    assert args.logs, "--logs DIR (outside any clone and the outputs tree)"
    if args.status:
        status(args.logs)
        return
    prepare(args.logs)
    scheduler = Scheduler(args.logs, System())
    preflight(scheduler, args.wait)
    if args.packing_check:
        packing_check(scheduler, *args.packing_check)
        return
    scheduler.say(f"scheduler started (pid {os.getpid()}, outputs {settings.OUTPUTS}, smoke {settings.SMOKE})")
    scheduler.loop()


if __name__ == "__main__":
    main()
