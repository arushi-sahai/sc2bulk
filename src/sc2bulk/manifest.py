"""Run records and resumable work units.

Every output file has a manifest beside it (<output>.manifest.json) recording what made it: a
hash of the settings that determine it (which include the stage's scoped code hash,
sc2bulk.codehash, and its inputs' content hashes), the output's own content hash and size, the
seed, the git commit, the machine, the environment fingerprint, whether the bulk labels were
blinded, and the unit's cost (observations, outside the settings hash).

A work unit is started with begin() and skipped when every output exists with its recorded size
and its manifest's settings hash matches; otherwise it is recomputed. finish() removes the old
manifest before replacing the output and writes the new one by rename, so a crash never pairs a
new output with an old record. Because inputs are
recorded by content, an upstream rerun that writes identical bytes invalidates nothing
downstream. Outputs are written to a temporary file and renamed,
so a crash never leaves a half-written file.
"""
import functools
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from sc2bulk import codehash
from sc2bulk.config import settings as config


def git_state() -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.REPO, capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=config.REPO, capture_output=True,
                           text=True, check=True).stdout.strip() != ""
    return {"commit": commit, "dirty": dirty}


def stage_code_hash(stage_script, pair: str | None = None, extra=()) -> str:
    """The stage's scoped code hash (sc2bulk.codehash): its script, the modules it imports,
    their package data, and the configuration of the pair it runs."""
    return codehash.stage_hash(stage_script, pair, extra)


def settings_hash(settings: dict) -> str:
    """Hash of everything that determines an output (JSON-serializable)."""
    text = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def content_hash(path) -> str:
    """SHA-256 of a file's bytes (every output is written deterministically)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@functools.cache
def environment() -> dict:
    """The running environment: its name (the conda environment's directory), the Python version
    and a hash of pip freeze + conda list --explicit (conda packages such as CUDA and MKL aren't
    in pip freeze). The project's own line is left out of the freeze (it carries the commit), so
    the fingerprint changes only with the installed packages. conda is False if no conda is on
    PATH; full-run mode requires it."""
    pip = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    pip = "\n".join(line for line in pip.splitlines() if "sc2bulk" not in line)
    conda = (subprocess.run(["conda", "list", "--explicit", "--prefix", sys.prefix], capture_output=True, text=True,
                            check=True).stdout if shutil.which("conda") else None)
    fingerprint = hashlib.sha256((pip + (conda or "")).encode()).hexdigest()[:16]
    return {"name": Path(sys.prefix).name, "python": platform.python_version(), "fingerprint": fingerprint,
            "conda": conda is not None}


def manifest_path(output):
    return output.with_name(output.name + ".manifest.json")


def read(output) -> dict:
    """The manifest record of an output."""
    return json.loads(manifest_path(output).read_text())


def input_hashes(paths) -> dict:
    """{outputs-relative name: content hash} of upstream outputs, from their manifests: how a
    stage records its inputs."""
    return {str(Path(path).relative_to(config.OUTPUTS)): read(path)["content_hash"] for path in paths}


def is_done(outputs, settings: dict) -> bool:
    """True if every output exists, with the size its manifest recorded, made with these settings."""
    for output in outputs:
        path = manifest_path(output)
        if not (output.exists() and path.exists()):
            return False
        record = json.loads(path.read_text())
        if record["settings_hash"] != settings_hash(settings) or record.get("size") != output.stat().st_size:
            return False
    return True


def temporary(output):
    """The path to write to before finish() renames it into place. It keeps the output's
    extension, so libraries that infer the format from it still work."""
    return output.with_name(".tmp." + output.name)


def check_full_run():
    """Full-run mode (SC2BULK_FULL_RUN=1): a clean tree at a tag, never the smoke run, and conda
    visible for the environment fingerprint. Returns the tag (None outside full-run mode)."""
    if not config.FULL_RUN:
        return None
    assert not config.SMOKE, "the full run can't be a smoke run"
    assert not git_state()["dirty"], "full-run mode needs a clean git tree"
    tag = subprocess.run(["git", "describe", "--exact-match", "--tags"], cwd=config.REPO, capture_output=True,
                         text=True, check=True).stdout.strip()
    assert environment()["conda"], "full-run mode needs conda on PATH (environment fingerprint)"
    return tag


def check_environments(root):
    """Every output under root made in one environment (by name) was made with the same packages:
    the full run's preflight (pipeline/run_all.py). The smoke and blinded runs' subdirectories are
    left out."""
    fingerprints = {}
    for path in Path(root).rglob("*.manifest.json"):
        top = path.relative_to(root).parts[0]
        if top == "smoke" or top.startswith("blinded_"):
            continue
        env = json.loads(path.read_text())["environment"]
        fingerprints.setdefault(env["name"], set()).add(env["fingerprint"])
    changed = {name: sorted(values) for name, values in fingerprints.items() if len(values) > 1}
    assert not changed, f"outputs made with different packages in one environment: {changed}"


def assert_unblinded(paths):
    """For figures and the real statistics: no input was made from permuted bulk labels."""
    for path in paths:
        assert read(path)["blinded"] is None, f"{path} was made from permuted bulk labels"


# Set by sc2bulk.labels when it reads bulk labels: from then on the process may write only
# label-bearing outputs (settings.LABELLED prefix, under settings.LABELLED_OUTPUTS), so permuted labels never
# reach a file with a real-looking name.
LABELS_READ = False


def begin(device: str = "cpu") -> dict:
    """Start a work unit: reset the process's peak memory (Linux: writing 5 to clear_refs
    resets the peak RSS) and, on a GPU unit, the device's peak statistics. Within a process
    that ran earlier units, the peak still counts their leftover heap, so it is an upper
    bound; the unit's last-written output carries its full totals."""
    Path("/proc/self/clear_refs").write_text("5")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    return {"start": time.perf_counter(), "device": device, "tag": check_full_run()}


def cost(unit: dict) -> dict:
    """The unit's cost so far. CUDA is touched only on a GPU unit, so a CPU stage never
    opens a context on the shared GPU."""
    gpu = unit["device"].startswith("cuda")
    return {"seconds": round(time.perf_counter() - unit["start"], 2),
            "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "device": unit["device"],
            "gpu": torch.cuda.get_device_name(unit["device"]) if gpu else None,
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(unit["device"]) if gpu else None,
            "gpu_count": torch.cuda.device_count() if gpu else None}


def finish(output, settings: dict, unit: dict, seed=None, inputs=()):
    """Move the finished temporary file into place and write its manifest."""
    if LABELS_READ:
        assert output.parent.is_relative_to(config.LABELLED_OUTPUTS) and output.name.startswith(config.LABELLED + "_"), \
            f"{output}: a process that read bulk labels writes only {config.LABELLED}_ outputs under {config.LABELLED_OUTPUTS}"
    manifest_path(output).unlink(missing_ok=True)
    os.replace(temporary(output), output)
    record = {
        "settings_hash": settings_hash(settings),
        "settings": settings,
        "content_hash": content_hash(output),
        "size": output.stat().st_size,
        "seed": seed,
        "inputs": [str(p) for p in inputs],
        "git": git_state(),
        "tag": unit["tag"],
        "host": platform.node(),
        "environment": environment(),
        "n_threads": config.N_THREADS,
        "bookkeeping": codehash.bookkeeping_hash(),
        "blinded": config.BLINDED,
        "cost": cost(unit),
        "torch": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    written = temporary(manifest_path(output))
    written.write_text(json.dumps(record, indent=1, sort_keys=True, default=str))
    os.replace(written, manifest_path(output))
