"""Run the pipeline: every stage in order, one process per stage and pair (so each work
unit's recorded peak memory is its own) in the stage's environment, stopping at the first
failure. Every stage takes --pair. Each stage skips the work units it has already finished
(sc2bulk.manifest), so rerunning after a crash resumes where it stopped.

  python pipeline/run_all.py                    every stage
  python pipeline/run_all.py --from 05_search   from one stage on
  python pipeline/run_all.py --only 08_readout  one stage
  python pipeline/run_all.py --pair tms         one pair (passed to every stage)
  python pipeline/run_all.py --smoke            the smoke run (config/smoke.py)
  python pipeline/run_all.py --until-readout    only the stages before 08_readout: the label-blind
                                                compute allowed before the analysis choices are fixed
                                                (05 search, 06 training, 07 embeddings;
                                                08 on bulk and everything after waits for the analysis-fixed tag)
  python pipeline/run_all.py --dry-run          run nothing; per stage and pair, count the outputs
                                                whose recorded code hash differs from the current
                                                tree's (they would rerun)

Stages that read bulk labels also refuse to run on their own until the analysis-fixed tag (or a blinded
run; sc2bulk.labels.require_access). In full-run mode (SC2BULK_FULL_RUN=1), every output made
in one environment must have been made with the same packages, checked before each stage and at
the end. Figures are drawn afterwards by the scripts in figures/.
"""
import json
import argparse
import os
import subprocess
import sys
from pathlib import Path

from sc2bulk import codehash, manifest
from sc2bulk.config import settings
from sc2bulk.config.pairs import PAIR_ORDER

PIPELINE = Path(__file__).resolve().parent

# The conda environment each stage runs in
ENVIRONMENTS = {
    "main": os.environ.get("SC2BULK_ENV", "sc2bulk"),
    "geneformer": os.environ.get("SC2BULK_ENV_GENEFORMER", "fm-geneformer"),
    "scgpt": os.environ.get("SC2BULK_ENV_SCGPT", "fm-scgpt"),
    "scvi": os.environ.get("SC2BULK_ENV_SCVI", "scvi"),      # scvi-tools needs numpy 2: its own environment
    "scrat": os.environ.get("SC2BULK_ENV_SCRAT", "scrat"),   # ScRAT needs pytorch-lightning: its own environment
    "uce": os.environ.get("SC2BULK_ENV_UCE", "uce"),
}

# (stage, script, environment, extra arguments, reads bulk labels, uses the GPU). A stage that
# runs once per model environment is listed once per model. Scripts not written yet are skipped
# with a message (the table lists the whole planned pipeline).
STAGES = [
    ("01_download", "01_download.py", "main", [], False, False),
    ("02_preprocess", "02_preprocess.py", "main", [], False, False),
    ("02b_all_available", "02b_all_available.py", "main", [], False, False),
    ("03_splits", "03_splits.py", "main", [], False, False),
    ("03b_attribution_splits", "03b_attribution_splits.py", "main", [], False, False),
    ("04_panel", "04_panel.py", "main", [], False, False),
    ("04b_deconvolve", "04b_deconvolve.py", "main", [], False, False),
    ("04c_qc_embed", "04c_qc_embed.py", "main", [], False, False),
    ("04d_characteristics", "04d_characteristics.py", "main", [], False, False),
    ("05_search", "05_search.py", "main", [], False, True),
    ("05_search", "05_search.py", "scrat", ["--scrat"], False, True),
    ("05v_scvi_search", "05v_scvi_search.py", "scvi", [], False, True),
    ("06_train", "06_train.py", "main", [], False, True),
    ("06_train", "06_train.py", "scrat", ["--scrat"], False, True),
    ("06_train", "06_train.py", "scvi", ["--scvi"], False, True),
    ("07_embed", "07_embed.py", "geneformer", ["--model", "geneformer"], False, True),
    ("07_embed", "07_embed.py", "scgpt", ["--model", "scgpt"], False, True),
    ("07c_uce", "07c_uce.py", "uce", [], False, True),
    ("07d_embed_all_available", "07d_embed_all_available.py", "geneformer", ["--model", "geneformer"], False, True),
    ("07d_embed_all_available", "07d_embed_all_available.py", "scgpt", ["--model", "scgpt"], False, True),
    ("08_readout", "08_readout.py", "main", [], False, False),
    ("08a_alpha_edges", "08a_alpha_edges.py", "main", [], False, False),
    ("08b_secondary", "08b_secondary.py", "main", [], False, False),
    ("09_alignment", "09_alignment.py", "main", [], False, False),
    ("09c_geometry", "09c_geometry.py", "main", [], False, False),
    # stage 10: the FM pilots (--pilot, mouse microglia) run by hand first; the Geneformer rows need SC2BULK_GENEFORMER_GO=1
    ("10_attribution", "10_attribution.py", "main", ["--model", "set"], False, True),
    ("10_attribution", "10_attribution.py", "scrat", ["--model", "scrat"], False, True),
    ("10_attribution", "10_attribution.py", "scvi", ["--model", "scvi"], False, True),
    ("10_attribution", "10_attribution.py", "scgpt", ["--model", "scgpt"], False, True),
    ("10_attribution", "10_attribution.py", "geneformer", ["--model", "geneformer"], False, True),
    ("10_attribution", "10_attribution.py", "geneformer", ["--model", "geneformer", "--joint"], False, True),
    ("10_attribution", "10_attribution.py", "main", ["--gather"], False, False),
    ("10c_concordance", "10c_concordance.py", "main", [], True, False),
    ("11_statistics", "11_statistics.py", "main", [], True, False),
    ("11b_secondary", "11b_secondary.py", "main", [], True, False),
    ("12_simulation", "12_simulation.py", "main", [], False, True),             # the design, CytoSet and Set Transformer
    ("12_simulation", "12_simulation.py", "scrat", ["--scrat"], False, True),   # ScRAT in its environment
    ("12_simulation", "12_simulation.py", "main", ["--readout"], False, False),  # read-outs and the criterion, after both
    ("12a_properties", "12a_properties.py", "main", [], False, False),
]
STAGE_NAMES = list(dict.fromkeys(stage for stage, *_ in STAGES))
# Stages that pool every pair: run once, without --pair, and skipped under --pair (a one-pair run can't form them)
POOLED = {"08a_alpha_edges", "11_statistics", "11b_secondary"}


def dry_run(stage, script, pair):
    """Print how many of the stage's recorded outputs for the pair carry a code hash other than
    the current tree's."""
    current = codehash.stage_hash(PIPELINE / script, pair, codehash.extra(stage, pair))
    root = settings.OUTPUTS / stage if pair is None else settings.OUTPUTS / stage / pair   # a pooled stage: every record
    records = [json.loads(p.read_text()) for p in root.rglob("*.manifest.json")]
    stale = sum(r["settings"].get("code") != current for r in records)
    print(f"{stage} {pair or 'pooled'}: {len(records)} outputs, {stale} with a different code hash", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", choices=STAGE_NAMES)
    parser.add_argument("--only", choices=STAGE_NAMES)
    parser.add_argument("--pair", choices=PAIR_ORDER)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--until-readout", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    names = STAGE_NAMES
    if args.start:
        names = names[names.index(args.start):]
    if args.only:
        names = [args.only]

    env = dict(os.environ)
    # thread pools of libraries loaded after start-up too (sc2bulk.determinism pins the rest)
    env.update({name: str(settings.N_THREADS) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    if args.smoke:
        env["SC2BULK_SMOKE"] = "1"
    pairs = [args.pair] if args.pair else PAIR_ORDER

    gated = set(STAGE_NAMES[STAGE_NAMES.index("08_readout"):])   # from bulk read-outs on
    for stage, script, environment, stage_args, needs_labels, _ in STAGES:
        if stage not in names or (args.until_readout and stage in gated):
            continue
        if not (PIPELINE / script).exists():
            print(f"== {stage}: {script} not written yet; skipped", flush=True)
            continue
        if stage in POOLED and args.pair:
            print(f"== {stage}: pools every pair; skipped under --pair", flush=True)
            continue
        for pair in [None] if stage in POOLED else pairs:
            if args.dry_run:
                dry_run(stage, script, pair)
                continue
            if settings.FULL_RUN:
                manifest.check_environments(settings.OUTPUTS)
            command = ["conda", "run", "--no-capture-output", "-n", ENVIRONMENTS[environment],
                       "python", str(PIPELINE / script), *stage_args, *(["--pair", pair] if pair else [])]
            print(f"== {stage} ({ENVIRONMENTS[environment]}): {' '.join(command[5:])}", flush=True)
            subprocess.run(command, env=env, check=True)
    if settings.FULL_RUN and not args.dry_run:
        manifest.check_environments(settings.OUTPUTS)


if __name__ == "__main__":
    sys.exit(main())
