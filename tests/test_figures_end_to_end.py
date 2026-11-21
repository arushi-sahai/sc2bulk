"""Every figure end to end.

A synthetic outputs tree for all six pairs: stage 11's and 11b's tables are made by the real, unedited stage 11 and 11b
main() (inputs from tests/test_statistics_driver.build and tests/test_statistics_secondary.fabricate, plus the files
those don't make: Fig 2B's within predictions, one-type within rows, one-change search candidates, Crohn's learning
curve), and every label-free and stage 10c file the figures read is written here. labels.bulk_labels is patched with
the fixture's synthetic labels; manifests are stubbed (provenance is tested on synthetic manifests in
tests/test_figures_firewall.py; stage 06's row A records get synthetic cost manifests for Table S11). Then make_all draws
every figure, and the tests check the files, the values against the stage tables, byte-reproducibility, the data files'
rule, the blinded mode, the conditional elements and Additional file 2's tables (figures/tables.py). Slow (stage 11b
on six pairs): the tree is built once per module."""
import gzip
import json
import re
import shutil
import sys
import types

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from sc2bulk import folds, labels, properties, schemas, training_inputs
from sc2bulk.config import alignment as A
from sc2bulk.config import configurations, deconvolution, fig1e, search_space, settings
from sc2bulk.config import geometry as G
from sc2bulk.config import figures as F
from sc2bulk.config import properties as P12
from sc2bulk.config import simulation as SIM
from sc2bulk.config import style as S
from sc2bulk.config import training as T
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS

import test_statistics_driver as driver
import test_statistics_secondary as secondary_test

sys.path.insert(0, str(settings.REPO / "figures"))
import common  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import fig4  # noqa: E402
import make_all  # noqa: E402
import plots  # noqa: E402
import style as figure_style  # noqa: E402

LABELLED = settings.LABELLED
ENCODERS = ("cytoset", "set_transformer", "scrat")
NORMALIZATION = {"norm_population": "cell_normalization", "norm_per_sample": "cell_normalization"}
ONE_CHANGE = {"cytoset": {"out_max": "out_pool", "out_sum": "out_pool", "out_attention": "out_pool", **NORMALIZATION},
              "set_transformer": NORMALIZATION, "scrat": NORMALIZATION}   # candidate -> its group in the real grid
LABEL_GENES = {"hammond": ("Lgals3", "Igf1"), "tms": ("C1qa", "C1qb"), "soundlife": None, "lupus": None, "crohns": None, "pineda": None}


def write(kind, frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    schemas.write(kind, schemas.with_neutral_keys(frame.reset_index(drop=True)), path)


def keys_of(M, key, **change) -> dict:
    return M.reference_keys(key, **change) if key in configurations.REFERENCE_ROWS else M.config_keys(key, **change)


def single_cell_samples(pair) -> pd.DataFrame:
    """Richer single-cell sample tables than fabricate's (QC counts, the S2-S7 extra columns); Sound Life's donors are
    the bulk side's (cross-fitting by donor) and so are its sample ids (as in the real pair, distinct within each unit),
    mouse organs' mice give two organs each."""
    if pair == "soundlife":
        donors = [f"d{d}" for d in range(32)]
        table = pd.DataFrame({"sample_id": [f"{d}_v0" for d in donors], "donor_id": donors, "holdout_group": donors,
                              "group": [d % 2 for d in range(32)], "cmv": ["positive", "negative"] * 16})
    else:
        ids = [f"sc{i}" for i in range(8)]
        donors = [f"m{i // 2}" for i in range(8)] if pair == "tms" else ids
        table = pd.DataFrame({"sample_id": ids, "donor_id": donors, "holdout_group": donors,
                              "group": [(i // 2) % 2 for i in range(8)] if pair == "tms" else [i % 2 for i in range(8)]})
    n = len(table)
    table = table.assign(cells_before_qc=1200, cells_after_qc=1100 + np.arange(n), cells_stored=1000, one_type_stored=True, drop_reason=np.nan)
    if pair == "tms":
        table["organ"] = ["Kidney", "Lung"] * (n // 2)
    if pair == "crohns":
        table = table.assign(**{proxy: ["lot1", "lot2"] * (n // 2) for proxy in F.CROHNS_BATCH_PROXIES})
    if pair == "lupus":   # the batch-4 restriction's dropped samples stay in stage 02's table (S5H)
        dropped = pd.DataFrame({"sample_id": ["x0", "x1", "x2"], "donor_id": ["x0", "x1", "x2"], "holdout_group": ["x0", "x1", "x2"],
                                "drop_reason": [F.LUPUS_RESTRICTION, F.LUPUS_RESTRICTION, "ImmVar control (excluded, as by the authors)"],
                                "label_value": [0, 1, 0]})
        table = pd.concat([table.assign(label_value=table["group"]), dropped], ignore_index=True)
    if pair == "pineda":
        table = table.assign(sex=["female", "male"] * (n // 2), age_at_death=60.0 + np.arange(n), pmi_hours=10.0 + np.arange(n),
                             run=["r1", "r2"] * (n // 2))
    return table


def scoring_unit(pair, samples) -> np.ndarray:
    return samples["donor_id" if PAIRS[pair]["scoring_unit"] == "donor" else "sample_id"].to_numpy()


def prediction_rows(M, pair, unit, key, side, samples, seeds, rng, **change) -> pd.DataFrame:
    frames = []
    for e in seeds:
        for r in range(10):
            score = samples["group"].to_numpy() * 1.2 + rng.normal(0, 1.0, len(samples))
            frames.append(pd.DataFrame({**keys_of(M, key, **change), "pair": pair, "fold_set": unit[0], "fold": unit[1], "encoder_seed": e,
                                        "readout_seed": r, "side": side, "sample_id": samples["sample_id"].to_numpy(),
                                        "scoring_unit_id": scoring_unit(pair, samples), "score": score, "probability": np.clip(score, 0, 1)}))
    return pd.concat(frames, ignore_index=True)


def held_out(pair, unit, samples) -> pd.DataFrame:
    donor_folds = pd.read_csv(folds.FOLDS / pair / "donor_folds.csv", dtype={"donor_id": str})
    scored = donor_folds[(donor_folds["fold_set"] == unit[0]) & (donor_folds["fold"] == unit[1]) & (donor_folds["side"] == "scored")]
    return samples[samples["donor_id"].isin(scored["donor_id"])]


def strengthen_pseudobulk(root, truth):
    """Pseudobulk's bulk scores made more separating outside the ceiling pair, so the tree's pooled rows are not all
    "better" (the fixture's alternatives are otherwise stronger than pseudobulk) and its claims look like a real run's."""
    for pair in PAIR_ORDER[1:]:
        group = truth[pair].set_index("sample_id")["group"]
        for path in sorted((root / "08_readout" / pair).glob("*/pseudobulk_predictions.parquet")):
            table = schemas.read("predictions", path)
            schemas.write("predictions", table.assign(score=table["score"] + 0.5 * group.loc[table["sample_id"]].to_numpy()), path)


def extend(root, M, truth):
    """Every file the figures read that the stage 11/11b fixtures don't make."""
    rng = np.random.default_rng(58)
    shutil.rmtree(root / "08b_secondary" / "hammond" / "all_0" / "one_type")   # mouse microglia has no one-type store
    for pair in PAIR_ORDER:
        samples = single_cell_samples(pair)
        samples.to_csv(root / "02_preprocess" / pair / "single_cell_samples.csv", index=False)
        samples = samples[samples["drop_reason"].isna()].astype({"group": int})
        if pair != "soundlife":   # stage 03's donor folds (Sound Life's come from the stage 11 fixture)
            (root / "03_splits" / pair).mkdir(parents=True)
            folds.donor_folds(pair, samples).to_csv(root / "03_splits" / pair / "donor_folds.csv", index=False)
        bulk = pd.read_csv(root / "02_preprocess" / pair / "bulk_samples.csv", dtype={"sample_id": str, "donor_id": str})
        bulk = bulk.assign(library_size=2_000_000 + 100_000 * np.arange(len(bulk)), genes_detected=15000 + 300 * np.arange(len(bulk)))   # ranges with no axis offset
        if pair == "hammond":
            bulk["condition"] = ["ELS", "control"] * (len(bulk) // 2)
        if pair == "crohns":
            bulk["age"] = 30.0 + np.arange(len(bulk))
        bulk.to_csv(root / "02_preprocess" / pair / "bulk_samples.csv", index=False)
        units = M.primary_units(pair)
        primary_set = units[0][0]
        for unit in units:
            name = folds.unit_name(*unit)
            heldout = primary_set == "crossfit"
            within_samples = held_out(pair, unit, samples) if heldout else samples
            side = "single_cell_heldout" if heldout else "single_cell_validation"
            for key in M.KEYS:
                seeds = (M.seeds_for(key, range(10)) if heldout else range(10))
                write("predictions", prediction_rows(M, pair, unit, key, side, within_samples, seeds, rng),
                      root / "08_readout" / pair / name / f"{key}_within_predictions.parquet")
                if "one_type" in PAIRS[pair]:
                    seeds = M.seeds_for(key, range(5)) if heldout else range(5)
                    write("predictions", prediction_rows(M, pair, unit, key, side, within_samples, seeds, rng, cell_subset="one_type"),
                          root / "08b_secondary" / pair / name / "one_type" / f"{key}_within_predictions.parquet")
            if pair == "crohns":   # fabricate's learning curve covers its own fixture pairs only
                for key in M.KEYS:
                    t = schemas.read("predictions", root / "08_readout" / pair / name / f"{key}_predictions.parquet")
                    curve = pd.concat([t.assign(encoder_seed=e) for e in range(5)]) if key in M.NO_ENCODER_LEVEL else secondary_test.derived(t, range(5))
                    write("predictions", pd.concat([curve.assign(variant=f"training_fraction_{f}") for f in (0.25, 0.5)]),
                          root / "08b_secondary" / pair / name / "learning_curve" / f"{key}_predictions.parquet")
            base = schemas.read("predictions", root / "08_readout" / pair / name / "cytoset_cells_predictions.parquet")
            for encoder in ENCODERS:
                rows = []
                for i in range(5):
                    rows.append({"key": "base", "change_group": "base", "change_value": "{}", "encoder_seed": i})
                    rows += [{"key": c, "change_group": g, "change_value": json.dumps(search_space.candidates(encoder, pair)[g][c]), "encoder_seed": i}
                             for c, g in ONE_CHANGE[encoder].items()]
                table = pd.DataFrame(rows).assign(pair=pair, fold_set=unit[0], fold=unit[1], row_kind="candidate", model=encoder, input="cells",
                                                  n_cells=500, pooling="native", validation_bce=0.6 + 0.01 * rng.random(len(rows)), did_train=True,
                                                  did_train_reason="", selected=False, merged=False)
                for i in range(5):
                    write("candidates", table[table["encoder_seed"] == i], root / "05_search" / pair / name / encoder / f"candidates_s{i}.parquet")
                    merged = table[(table["encoder_seed"] == i) & (table["key"] == "base")].assign(key="merged", change_group="merged", merged=True)
                    write("candidates", merged, root / "05_search" / pair / name / encoder / f"candidates_merged_s{i}.parquet")
                for j, candidate in enumerate(ONE_CHANGE[encoder]):
                    shifted = secondary_test.derived(base, range(5), row_kind="candidate", key=candidate, model=encoder)
                    write("predictions", shifted.assign(score=shifted["score"] + 0.05 * rng.normal(size=len(shifted))),
                          root / "08b_secondary" / pair / name / "search_candidates" / f"{encoder}__{candidate}_predictions.parquet")
                for e in range(10):
                    record = {"did_train": e != 3, "did_train_reason": "" if e != 3 else "collapse",
                              "margin_d_star": 0.05 + 0.01 * e, "margin_p": 0.01, "epochs": 40 + e, "stop_epoch": 20 + e}
                    path = root / "06_train" / pair / name / "A" / encoder / f"record_s{e}.json"
                    path.write_text(json.dumps(record))
                    write_cost(path, f"06 {pair} {name} {encoder} {e}", e, 100.0 + e, row="A", model=encoder)   # Table S11
                selection = root / "05_search" / pair / name / encoder / "selection.json"   # stage 05's decision (Table S4C):
                made = json.loads(selection.read_text())   # the base kept, stored as JSON (tuples become lists)
                winners = {"winners": {}, "kept": {}, "gains": {}, "merged_result": None}
                frozen = {**search_space.base(encoder), **made["frozen"]}
                selection.write_text(json.dumps({**made, "frozen": frozen, "how": "base", "winners": winners}))
                smoke = root / "smoke" / "06_train" / pair / name / "A" / encoder / "record_s0.json.manifest.json"   # not counted
                smoke.parent.mkdir(parents=True, exist_ok=True)
                write_cost(smoke.with_name("record_s0.json"), "smoke", 0, 1e6)
            for i in range(5):
                path = root / "05v_scvi_search" / pair / name / f"elbo_s{i}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"candidate": ["base", "latent20", "layers2"], "group": ["base", "n_latent", "n_layers"], "config": "{}",
                              "validation_elbo": 1000 + rng.normal(0, 5, 3), "epochs": 100}).to_csv(path, index=False)
        geometry(root, M, pair, samples, bulk, rng)
        attribution(root, M, pair, rng)
        label_free(root, pair, samples, bulk, rng)
        property_r2(root, M, pair, rng)
    simulation(root, M, rng)
    plant_costs(root)


def write_cost(output, unit, seed, seconds, **record):
    """A synthetic manifest carrying a work unit's cost (Table S11): the settings hash names the unit."""
    manifest = {"settings_hash": unit, "settings": record, "seed": seed, "host": "synthetic",
                "cost": {"seconds": seconds, "device": "cuda:0", "gpu": "synthetic GPU"}}
    output.with_name(output.name + ".manifest.json").write_text(json.dumps(manifest))


def plant_costs(root):
    """Table S11's two hard cases: two work units writing into one directory (stage 09, one per configuration) and one
    unit writing into two directories (stage 12a, one per pair across its folds)."""
    shared = root / "09_alignment" / "lupus" / "all_0"
    for key, seconds in (("cytoset_cells", 3600.0), ("scgpt_cells", 7200.0)):
        write_cost(shared / f"{key}_predictions.parquet", f"09 lupus {key}", None, seconds, configuration=key)
    for fold, seconds in ((0, 1800.0), (1, 3600.0)):   # outputs carry the unit's cost so far: the unit counts 3600 s
        folder = root / "12a_properties" / "soundlife" / f"crossfit_{fold}"
        write_cost(folder / "pseudobulk_property_r2.parquet", "12a soundlife", None, seconds)


def geometry(root, M, pair, samples, bulk, rng):
    units = M.primary_units(pair)
    for unit in units:
        train = samples if unit[0] != "crossfit" else samples[~samples["sample_id"].isin(held_out(pair, unit, samples)["sample_id"])]
        scored = folds.scored_bulk(pair, *unit, bulk[bulk["drop_reason"].isna()])
        for key in configurations.CONFIGURATION_ORDER:
            frames = []
            for e in M.seeds_for(key, range(10)):
                row = {**M.config_keys(key), "pair": pair, "fold_set": unit[0], "fold": unit[1], "encoder_seed": e, "readout": "none", "readout_seed": -1}
                frames.append(pd.DataFrame([{**row, "alignment": "none", "measure": "silhouette_raw", "sample_id": "all", "value": rng.random()}]))
                for a in ("none", *A.ALIGNED):
                    frames.append(pd.DataFrame([{**row, "alignment": a, "measure": "silhouette", "sample_id": "all", "value": rng.random()}]))
                    seed = A.GEOMETRY_ENCODER_SEED if key not in M.NO_ENCODER_LEVEL else -1
                    if key in A.GEOMETRY_CONFIGURATIONS and e == seed and unit[1] == 0:
                        ids = [*train["sample_id"], *scored["sample_id"]]
                        frames += [pd.DataFrame({**row, "alignment": a, "measure": m, "sample_id": ids, "value": rng.normal(size=len(ids))})
                                   for m in ("pc1", "pc2")]
            write("alignment_geometry", pd.concat(frames, ignore_index=True), root / "09_alignment" / pair / folds.unit_name(*unit) / f"{key}_geometry.parquet")
        if unit[1] == 0:   # stage 09c: the Fig 3A / S13A-E coordinates
            ids = [*train["sample_id"], *scored["sample_id"]]
            for key in G.CONFIGURATIONS:
                row = {**M.config_keys(key), "pair": pair, "fold_set": unit[0], "fold": 0, "readout": "none", "readout_seed": -1,
                       "encoder_seed": -1 if key in M.NO_ENCODER_LEVEL else A.GEOMETRY_ENCODER_SEED}
                frames = [pd.DataFrame({**row, "alignment": a, "measure": m, "sample_id": ids, "value": rng.normal(size=len(ids))})
                          for a in ("none", *A.ALIGNED) for m in ("pc1", "pc2")]
                write("alignment_geometry", pd.concat(frames, ignore_index=True), root / "09c_geometry" / pair / folds.unit_name(*unit) / f"{key}_geometry.parquet")


def genes_of(pair) -> list:
    named = LABEL_GENES[pair]
    if named is None:
        labelled = pd.read_csv(settings.REPO / "figures" / "signature_labels.tsv", sep="\t", comment="#")
        named = tuple(labelled.loc[labelled["pair"] == pair, "gene"][:2])
    return [*named, *(f"G{i}" for i in range(118))]


def attribution(root, M, pair, rng):
    """Stage 10's rankings, checks, inclusion, modality summary and level differences, and stage 10c's tables."""
    units = M.primary_units(pair)
    genes = genes_of(pair)
    out = root / "10_attribution" / pair
    fold = -1 if len(units) > 1 else units[0][1]
    modality = ("cytoset_cells", "set_transformer_cells", "scrat_cells", "scvi_cells", "scgpt_cells", "geneformer_cells")
    checks, summary = [], []
    for key in configurations.CONFIGURATION_ORDER:
        frames = []
        targets = [("ridge", "phenotype"), ("lasso", "phenotype")] + ([("none", "modality_direction")] if key in modality else [])
        for readout, target in targets:
            score = rng.random(len(genes))
            rank = pd.Series(score).rank(ascending=False, method="first").astype(int).to_numpy()
            frames.append(pd.DataFrame({**M.config_keys(key), "pair": pair, "fold_set": units[0][0], "fold": fold, "encoder_seed": -1, "readout_seed": -1,
                                        "readout": readout, "target": target, "method": "deletion" if key.startswith("geneformer") else "gradient_x_input",
                                        "gene": genes, "attribution": score, "rank": rank, **{f"top{k}": rank <= k for k in (25, 50, 100)}}))
            checks += [{"pair": pair, "fold_set": units[0][0], "fold": fold, "configuration": key, "readout": readout, "target": target, "check": c,
                        "half": "", "value": rng.random(), "threshold": None, "passed": None} for c in ("stability_seeds", "stability_donors")]
        write("attributions", pd.concat(frames), out / f"{key}_ranking.parquet")
        if key.startswith("geneformer"):
            checks.append({"pair": pair, "fold_set": units[0][0], "fold": fold, "configuration": key, "readout": "ridge", "target": "phenotype",
                           "check": "additivity_beta_k50", "half": "", "value": 0.5, "threshold": "[0.5, 2.0]",
                           "passed": key == "geneformer_cells"})
        if key in modality:
            for unit in units + ([(units[0][0], -1)] if len(units) > 1 else []):
                summary.append({"pair": pair, "fold_set": unit[0], "fold": unit[1], "configuration": key, "share_large": rng.random(),
                                "share_marker": rng.random(), "share_either": rng.random(), "base_large": 0.05, "base_marker": 0.1, "base_either": 0.14,
                                "markers_available": True})
    pd.DataFrame(checks).to_csv(out / "checks.csv", index=False)
    pd.DataFrame([{"pair": pair, "configuration": k, "included": k == "geneformer_cells",
                   "failed_checks": "" if k == "geneformer_cells" else "additivity_beta_k50"}
                  for k in ("geneformer_cells", "geneformer_pseudobulk")]).to_csv(out / "inclusion.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "modality_summary.csv", index=False)
    for unit in units:
        (out / folds.unit_name(*unit)).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"gene": genes, "delta": rng.normal(size=len(genes)), "smd": rng.normal(size=len(genes)), "large": False,
                      "marker": False}).to_csv(out / folds.unit_name(*unit) / "level_difference.csv", index=False)
    labelled = pd.read_csv(settings.REPO / "figures" / "signature_labels.tsv", sep="\t", comment="#")
    direction = dict(zip(labelled.loc[labelled["pair"] == pair, "gene"], labelled.loc[labelled["pair"] == pair, "direction"]))
    symbol = {"hammond": str.upper, "tms": str.upper}.get(pair, lambda g: g)
    keys = {"pair": pair, "fold_set": units[0][0], "fold": fold, "encoder_seed": -1, "readout_seed": -1}
    shifts = pd.DataFrame({**M.config_keys("pseudobulk"), **keys, "readout": "none", "gene": genes, "single_cell_shift": rng.normal(size=len(genes)),
                           "bulk_shift": rng.normal(size=len(genes)), "signature_direction": [direction.get(symbol(g), "") for g in genes],
                           "pseudobulk_top50": np.arange(len(genes)) < 50})
    write("gene_shifts", shifts, root / LABELLED / "10c_concordance" / pair / f"{LABELLED}_gene_shifts.parquet")
    rows = [{**M.config_keys(key), **keys, "readout": readout, "k": k, "n_genes": k, "shared_share": rng.uniform(0.35, 0.95), "base_all": 0.5, "base_matched": 0.55,
             "rho": 0.3, "included": key != "geneformer_pseudobulk"}
            for key in configurations.CONFIGURATION_ORDER for readout in ("ridge", "lasso") for k in (25, 50, 100)]
    write("concordance", pd.DataFrame(rows), root / LABELLED / "10c_concordance" / pair / f"{LABELLED}_concordance.parquet")
    if pair == "tms":
        for unit in units:
            (root / "04_panel" / pair / folds.unit_name(*unit)).mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"gene": genes, "human_symbol": [g.upper() for g in genes]}).to_csv(
                root / "04_panel" / pair / folds.unit_name(*unit) / "orthologs.csv", index=False)


def label_free(root, pair, samples, bulk, rng):
    """Stages 02 (per-cell QC), 04c and 04d."""
    cells = pd.DataFrame({"sample_id": np.repeat(samples["sample_id"].to_numpy(), 30), "cell_type": np.tile(["T", "B", "NK"], 10 * len(samples)),
                          "genes_detected": rng.integers(150, 3000, 30 * len(samples)), "total_counts": rng.integers(500, 9000, 30 * len(samples)),
                          "percent_mito": rng.random(30 * len(samples)) * 25})
    cells["passed_qc"] = (cells["genes_detected"] >= 200) & (cells["percent_mito"] < 20)
    with gzip.open(root / "02_preprocess" / pair / "cell_qc.csv.gz", "wt") as handle:
        cells.to_csv(handle, index=False)
    out = root / "04c_qc_embed" / pair
    out.mkdir(parents=True)
    cells.assign(cell_id=[f"c{i}" for i in range(len(cells))], umap1=rng.normal(size=len(cells)), umap2=rng.normal(size=len(cells)))[
        ["cell_id", "sample_id", "cell_type", "umap1", "umap2"]].to_parquet(out / "umap.parquet")
    kept = bulk[bulk["drop_reason"].isna()]
    pd.DataFrame({"sample_id": kept["sample_id"], "pc1": rng.normal(size=len(kept)), "pc2": rng.normal(size=len(kept))}).to_parquet(out / "bulk_pca.parquet")
    out = root / "04d_characteristics" / pair
    out.mkdir(parents=True)
    states = ["not determinable" if (c == "Batch confounded with label" and fig1e.BATCH[pair] is None) else
              fig1e.FIG1E[pair].get(c, ("no", ""))[0] if c == "Independent cohorts" else ("yes" if i % 2 else "no")
              for i, c in enumerate(fig1e.FIG1E_COLUMNS)]
    pd.DataFrame({"pair": pair, "column": fig1e.FIG1E_COLUMNS, "state": states, "value": np.nan,
                  "annotation": ["a short annotation" if s != "no" else "" for s in states]}).to_csv(out / "characteristics.csv", index=False)
    composition = cells.groupby(["sample_id", "cell_type"]).size().rename("cells").reset_index()
    composition["fraction"] = composition["cells"] / composition.groupby("sample_id")["cells"].transform("sum")
    composition.insert(1, "level", "cell_type")
    composition.to_csv(out / "composition.csv", index=False)
    pd.DataFrame({"level": "sample", "batch": ["b1", "b1", "b2", "b2"], "group": [0, 1, 0, 1], "count": [3, 1, 1, 3]}).to_csv(out / "batch_label.csv", index=False)
    pd.DataFrame([{"fold_set": fs, "fold": f, "training_samples": 8, "training_donors": 8, "scored_bulk_samples": 10} for fs, f in folds.units(pair)]).to_csv(
        out / "unit_sizes.csv", index=False)
    if deconvolution.LEVELS[pair] is not None:   # stage 04b: training samples' true and scored bulk's estimated fractions
        types = ["T", "B", "NK", "a long cell-type name past the legend's cut"] + ([f"rare {i}" for i in range(8)] if pair == "pineda" else [])
        for unit in folds.units(pair):
            unit_dir = root / "04b_deconvolve" / pair / folds.unit_name(*unit)
            unit_dir.mkdir(parents=True)
            parts = []
            for side, kind, ids in (("single_cell", "true", samples["sample_id"]), ("single_cell", "estimated", samples["sample_id"]),
                                    ("bulk", "estimated", kept["sample_id"])):
                wide = pd.DataFrame(rng.dirichlet(np.ones(len(types)), len(ids)), index=ids.to_numpy(), columns=types)
                parts.append(wide.rename_axis("sample_id").reset_index().melt(id_vars="sample_id", var_name="type", value_name="fraction")
                             .assign(residual=0.1, stratum="all", side=side, kind=kind))
            pd.concat(parts, ignore_index=True).to_parquet(unit_dir / "fractions.parquet", index=False)
    if pair in fig1e.TISSUE_MARKERS:
        pd.DataFrame([{"side": side, "marker": m, "organ": organ, "gene": g, "mean_log1p_cpm": rng.random() * 5, "present": True}
                      for side in ("single_cell", "bulk") for m, (genes, organ) in fig1e.TISSUE_MARKERS[pair].items() for g in genes]).to_csv(
            out / "markers.csv", index=False)


def property_r2(root, M, pair, rng):
    """Stage 12a's tables (mitochondrial fraction absent for mouse organs, composition for mouse microglia)."""
    units = M.primary_units(pair)
    for unit in units:
        for key in [*configurations.CONFIGURATION_ORDER, "pseudobulk_pca64"]:
            seeds = (M.seeds_for(key, range(10)) if key != "pseudobulk_pca64" else [-1]) if unit[0] == "crossfit" else range(10)
            rows = []
            for e in seeds:
                for r in range(10):
                    for prop in [p for p in ("depth", "mito_fraction", "composition_pc1", "phenotype") if (pair, p) not in F.NOT_APPLICABLE["Fig 6A"]]:
                        value = rng.uniform(0.6, 0.9) if prop == "phenotype" else rng.uniform(-0.3, -0.1) if key == "scvi_cells" else rng.uniform(-0.2, 0.6)
                        rows.append({**keys_of(M, key), "pair": pair, "fold_set": unit[0], "fold": unit[1], "encoder_seed": e, "readout_seed": r,
                                     "property": prop, "cv_r2": value, "null_mean": 0.0, "null_q95": 0.1})
            write("property_r2", pd.DataFrame(rows), root / "12a_properties" / pair / folds.unit_name(*unit) / f"{key}_property_r2.parquet")


def write_licence(root):
    files = sorted((root / "12a_properties").rglob("*_property_r2.parquet"))
    values = properties.split_mean(pd.concat([schemas.read("property_r2", p) for p in files], ignore_index=True))
    present = [k for k in configurations.CONFIGURATION_ORDER if k != "pseudobulk"]
    (root / "12a_properties" / "licence.json").write_text(json.dumps(properties.licence(values, present), indent=1))


def simulation(root, M, rng):
    unit = SIM.UNIT
    rows = []
    for key in ["pseudobulk", "mean_log_cells", *(f"{m}_cells" for m in SIM.ENCODERS)]:
        keys = M.reference_keys(key) if key == "mean_log_cells" else {**M.config_keys(key), "pooling": "native"}
        for scenario in SIM.SCENARIOS:
            for e in SIM.SPLITS:
                for r in range(10):
                    rows.append({**keys, "pair": SIM.PAIR, "fold_set": unit[0], "fold": unit[1], "encoder_seed": e, "readout_seed": r, "scenario": scenario,
                                 "within_sc_auroc": rng.uniform(0.5, 0.9), "simulated_bulk_auroc": rng.uniform(0.5, 0.9)})
    out = root / "12_simulation" / SIM.PAIR / folds.unit_name(*unit)
    write("simulation", pd.DataFrame(rows), out / "simulation.parquet")
    (out / "criterion.json").write_text(json.dumps({"set_encoders": list(SIM.ENCODERS), "margin": SIM.SUCCESS_MARGIN,
                                                    "delta": {m: 0.12 for m in (*SIM.ENCODERS, "mean_log_cells")},
                                                    "passes": {m: True for m in (*SIM.ENCODERS, "mean_log_cells")},
                                                    "control_passes": True, "structure_readings": {m: "detects structure" for m in SIM.ENCODERS},
                                                    "normalization": {m: "none" for m in SIM.ENCODERS},
                                                    "did_train_passed": {f"{m}_cells {s}": "3/5" for m in SIM.ENCODERS
                                                                         for s in SIM.SCENARIOS}}))


def stub_manifest():
    return types.SimpleNamespace(input_hashes=lambda paths: {}, stage_code_hash=lambda *a, **k: "x", begin=lambda *a, **k: {},
                                 temporary=lambda p: p, finish=lambda *a, **k: None)


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """The synthetic outputs tree with stage 11/11b's tables, and make_all run once on it."""
    root = tmp_path_factory.mktemp("outputs")
    mp = pytest.MonkeyPatch()
    mp.setattr(settings, "OUTPUTS", root)
    mp.setattr(settings, "LABELLED_OUTPUTS", root / LABELLED)
    mp.setattr(settings, "BLINDED", None)
    mp.setattr(folds, "FOLDS", root / "03_splits")
    mp.setattr(folds, "SAMPLES", root / "02_preprocess")
    mp.setattr(driver, "PAIRS", PAIR_ORDER)
    mp.setattr(T, "N_PERMUTED", 3)
    truth = driver.build(root, {"hammond"})
    strengthen_pseudobulk(root, truth)
    truth["crohns"] = truth["crohns"].assign(inflammation=["inflamed", "uninflamed"] * (len(truth["crohns"]) // 2))
    mp.setattr(labels, "require_access", lambda: None)
    mp.setattr(labels, "bulk_labels", lambda pair: truth[pair])
    mp.setattr(driver.S, "HAMMOND_SCHEME", "samples")
    M = secondary_test.stage11b()
    secondary_test.fabricate(root, M)
    extend(root, M, truth)
    write_licence(root)
    D = driver.driver()
    mp.setattr(D, "assert_frozen", lambda: None)
    mp.setattr(D, "determinism", types.SimpleNamespace(configure=lambda: None))
    mp.setattr(D, "manifest", stub_manifest())
    D.main()
    mp.setattr(M, "B", 200)
    mp.setattr(M.C, "SUBGROUP_COLUMNS", {"tms": "organ"})
    rng = np.random.default_rng(0)
    mp.setattr(training_inputs, "bulk_profiles", lambda pair, unit, ids: types.SimpleNamespace(X=scipy.sparse.csr_matrix(
        rng.normal(size=(len(ids), 4)) + np.array([truth[pair].set_index("sample_id").loc[s, "group"] for s in ids])[:, None])))
    mp.setattr(M, "determinism", types.SimpleNamespace(configure=lambda: None))
    mp.setattr(M, "preflight", lambda: [])
    mp.setattr(M, "manifest", stub_manifest())
    M.main()
    mp.setattr(common, "sources", lambda paths, repository=(): {"inputs": [], "repository": [], "commits": []})
    mp.setattr(common, "WATERMARK", figure_style.SYNTHETIC_WATERMARK)
    # Two layouts that move with the synthetic data alone (the real build checks every figure strictly): S7g's synthetic PCA
    # puts a tick label under the letter, and S11e's synthetic subgroups (organs) change its row height
    mp.setattr(F, "LAYOUT_PENDING", {"Fig. S7", "Fig. S11"})
    layout, check_layout = {}, common.check_layout   # item -> its layout report (common.check_layout prints it)
    mp.setattr(common, "check_layout", lambda fig, item, panel_axes, rows: layout.setdefault(item, check_layout(fig, item, panel_axes, rows)))
    make_all.main([])
    yield types.SimpleNamespace(root=root, truth=truth, monkeypatch=mp, layout=layout)
    mp.undo()


def figures_root(tree):
    return tree.root / LABELLED / "figures"


def test_every_panel_and_figure_written(tree):
    out = figures_root(tree)
    for item, spec in F.ITEMS.items():
        if spec["script"] not in make_all.SCRIPTS:
            continue
        stem = common.stem(item)
        for suffix in ("pdf", "png", "svg"):
            assert (out / stem / f"{stem}.{suffix}").exists(), item
        assert (out / stem / f"{stem}.pdf").stat().st_size <= S.MAX_FIGURE_BYTES
        for panel in F.PANELS[item]:
            assert (out / stem / f"{common.panel_name(item, panel)}.csv").exists(), (item, panel)
        assert json.loads((out / stem / "sources.json").read_text())["item"] == item
    text = json.loads((out / "figure_text.json").read_text())
    assert not text["partial"]
    claims = json.loads((tree.root / LABELLED / "11_statistics" / "pooled" / f"{LABELLED}_claims.json").read_text())
    forms = F.FIG2_TITLES["pooled"]   # K >= 2 in the tree
    if not claims["pooled_better"]:
        k = len(claims["inconclusive_at_margin"])
        assert text["items"]["Fig 2"]["title"] == (forms["within_margin"].format(margin="0.05") if claims["within_margin_all"] else
                                                   forms["inconclusive"].format(k=k, n=9, verb=F.FIG2_VERB.get(k, "are"), margin="0.05"))
    assert text["items"]["Fig 4"]["title"] and text["items"]["Fig 6"]["title"] == F.FIG6_TITLE


def test_values_match_the_stage_tables(tree):
    out = figures_root(tree)
    stats = schemas.read("statistics", tree.root / LABELLED / "11_statistics" / "pooled" / f"{LABELLED}_statistics.parquet")
    fig2a = pd.read_csv(out / "Fig2" / "Fig2A.csv")
    rows = fig2a[fig2a["row_type"] == "configuration"].set_index(["column", "key"])["delta"]
    per_pair = stats[(stats["pool_set"] == "per_pair") & (stats["readout"] == "ridge")].set_index(["pair", "key"])["delta"]
    assert np.allclose(rows.loc[list(per_pair.index)].to_numpy(), per_pair.to_numpy(), rtol=0, atol=1e-12)
    seeds = fig2a[fig2a["row_type"] == "seed"].groupby(["column", "key"])["delta_seed"].mean()
    assert np.allclose(seeds.loc[list(per_pair.index)].to_numpy(), per_pair.to_numpy(), rtol=0, atol=1e-9)   # dots' mean = delta
    fig2b = pd.read_csv(out / "Fig2" / "Fig2B.csv")
    strip = fig2a[fig2a["row_type"] == "pseudobulk_auroc"].set_index("column")   # the strip is Fig 2B's pseudobulk bulk value,
    pseudobulk = fig2b[(fig2b["key"] == "pseudobulk") & fig2b["within_auroc"].notna()].set_index("pair")["bulk_auroc"]   # and ♦ = pooled
    assert np.allclose(strip["auroc"].loc[list(pseudobulk.index)].to_numpy(), pseudobulk.to_numpy(), rtol=0, atol=1e-12)
    claims = json.loads((tree.root / LABELLED / "11_statistics" / "pooled" / f"{LABELLED}_claims.json").read_text())
    assert {c for c, r in strip.iterrows() if r["pooled"]} == {p for p in strip.index if p not in claims["excluded_at_ceiling"]}
    fig3c = pd.read_csv(out / "Fig3" / "Fig3C.csv")
    none = fig3c[fig3c["alignment"] == "none"].set_index(["pair", "key"])["value"]
    bulk = fig2b.dropna(subset=["within_auroc"]).set_index(["pair", "key"])["bulk_auroc"]
    assert bulk.index.is_unique, "Fig 2B: one summary row per pair and configuration"
    assert np.allclose(none.to_numpy(), bulk.loc[list(none.index)].to_numpy())   # Fig 3C's None = Fig 2B's bulk
    fig4a = pd.read_csv(out / "Fig4" / "Fig4A.csv")
    fig4b = pd.read_csv(out / "Fig4" / "Fig4B.csv")
    rings = fig4b[fig4b["kind"] == "ring"].dropna(subset=["delta"]).set_index(["pair", "key"])["delta"]
    point = fig4a[(fig4a["n_cells"] == "500") & fig4a["seed"].isna()].set_index(["pair", "key"])["delta"]
    assert np.allclose(rings.to_numpy(), point.loc[list(rings.index)].to_numpy())   # the ring = 4A's n = 500 point
    fig6a = pd.read_csv(out / "Fig6" / "Fig6A.csv")
    assert (fig6a["r2"] < 0).any()   # negative R² kept raw in the CSV (drawn as 0)
    assert not ((fig6a["pair"] == "tms") & (fig6a["property"] == "mito_fraction")).any()
    fig5b = pd.read_csv(out / "Fig5" / "Fig5B.csv", keep_default_na=False, na_values=[""])
    assert (fig5b.loc[fig5b["key"] == "geneformer_pseudobulk", "note"] == F.FIGURE_TEXT["not_applicable"]).all()   # included False
    fig6c = pd.read_csv(out / "Fig6" / "Fig6C.csv")
    assert set(fig6c["pair"]) == {p for p in PAIR_ORDER if "one_type" in PAIRS[p]}


def test_fig6c_smoke_skipped_pair(tree, monkeypatch, tmp_path):
    """Fig 6C under smoke draws only the pairs with one-type scores (08b skips a unit's one-type rows where its store
    falls short; tms all_0 in the smoke run); the real run still asserts every pair with a one-type store has them."""
    import fig6
    scores = common.scores()
    monkeypatch.setattr(common, "scores", lambda: scores[~((scores["pair"] == "tms") & (scores["cell_subset"] == "one_type"))])
    monkeypatch.setattr(common, "directory", lambda item: tmp_path)
    panels = {"Fig 6": tuple(F.PANELS["Fig 6"])}
    with pytest.raises(AssertionError, match="Fig 6C: one-type scores"):
        fig6.main(list(PAIR_ORDER), panels)
    monkeypatch.setattr(fig6, "settings", types.SimpleNamespace(SMOKE=True))
    fig6.main(list(PAIR_ORDER), panels)
    assert set(pd.read_csv(tmp_path / "Fig6C.csv")["pair"]) == {p for p in PAIR_ORDER if "one_type" in PAIRS[p]} - {"tms"}


def test_s16_frozen_85_rows(tree):
    """Fig. S16 has a single-cell-moments column (11b's per-pair D3 differences); the leave-one-pair-out rows
    (pooled only, unstreamed) are left out."""
    s16 = pd.read_csv(figures_root(tree) / "S16Fig" / "S16AFig.csv")
    sec = schemas.read("secondary_statistics", tree.root / LABELLED / "11b_secondary" / "pooled" / f"{LABELLED}_secondary_statistics.parquet")
    d3 = sec[(sec["row"] == "single_cell_moments") & (sec["pair"] != "pooled") & sec["key"].isin(common.alternatives())]
    shown = s16[s16["column"] == "single_cell_moments"].set_index(["pair", "key"])["delta"]
    assert len(shown) == len(d3) and np.allclose(shown.loc[list(zip(d3["pair"], d3["key"]))].to_numpy(), d3["delta"].to_numpy())
    assert "leave_one_pair_out" in set(sec["row"]) and not s16["column"].str.startswith("leave_one_pair_out").any()


def test_rerun_is_byte_identical(tree):
    out = figures_root(tree)
    before = {p: p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    make_all.main([])
    after = {p: p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    assert before.keys() == after.keys()
    assert [p for p in before if before[p] != after[p]] == []


def test_s_data(tree):
    out = figures_root(tree) / "data"
    import zipfile
    for i in range(3, 7):
        with zipfile.ZipFile(out / f"Additional_file_{i}.zip") as archive:
            readme = pd.read_csv(archive.open("README.csv"))
            members = {n: pd.read_csv(archive.open(n), low_memory=False) for n in archive.namelist() if n != "README.csv"}
        assert list(readme["file"]) == list(members)
        for name, frame in members.items():
            sensitive = [c for c in frame.columns if c in F.SENSITIVE_COLUMNS]
            if i == 3:
                cited = readme.set_index("file").loc[name, "item"]   # "Fig. S4i": the panel letter last, lower case
                assert set(sensitive) <= set(F.PANEL_SENSITIVE.get(cited[:-1] + cited[-1].upper(), ())), (name, sensitive)
            else:
                assert not sensitive, (name, sensitive)
            if "group" in frame:
                assert set(frame["group"].dropna()) <= {0, 1}
        if i == 3:   # the index = the registered panels with a CSV, in registry order
            registered = [f"{common.panel_name(item, p)}.csv" for item, panels in F.PANELS.items() for p in panels
                          if F.ITEMS[item]["script"] in make_all.SCRIPTS]
            assert list(readme["file"]) == registered
        if i == 6:   # seed-averaged predictions equal the mean of stage 08's per-seed predictions
            frame = members["predictions_lupus.csv"]
            primary = schemas.read("predictions", tree.root / "08_readout" / "lupus" / "all_0" / "scgpt_cells_predictions.parquet")
            primary = primary[(primary["readout"] == "ridge") & (primary["side"] == "bulk")]
            expected = primary.groupby("sample_id")["score"].mean()
            mine = frame[(frame["key"] == "scgpt_cells") & (frame["side"] == "bulk")].set_index("sample_id")["score"]
            assert np.allclose(mine.loc[expected.index], expected)
    assert (figures_root(tree) / "additional_files.csv").exists()


def test_conditional_elements(tree, monkeypatch):
    """Fig 4A's 07d flag and the age-similarity n.e. both ways; a missing configuration row fails loudly."""
    monkeypatch.setattr(F, "FIG4A_FM_ALL_AVAILABLE", ("scgpt",))
    with pytest.raises(AssertionError, match="FIG4A_FM_ALL_AVAILABLE"):
        fig4.points_a("lupus")
    monkeypatch.setattr(F, "FIG4A_FM_ALL_AVAILABLE", ("geneformer", "scgpt"))
    real = common.read_json
    lost = ("crossfit", 1)
    def read_json(stage, pair, name, unit=None):
        made = real(stage, pair, name, unit)
        return {**made, "frozen": {"age_similarity_loss": True}} if (pair, unit, name) == ("soundlife", lost, "cytoset/selection.json") else made
    monkeypatch.setattr(common, "read_json", read_json)
    with pytest.raises(AssertionError, match="frozen §60"):   # the predicate holds but the rows are there
        fig4.points_a("soundlife")
    scores = common.scores()
    dropped = scores[~((scores["pair"] == "soundlife") & (scores["fold"] == lost[1]) & (scores["key"] == "cytoset_cells")
                       & (scores["cell_subset"] == "all_available"))]
    monkeypatch.setattr(common, "scores", lambda: dropped)
    points = fig4.points_a("soundlife")
    assert not [p for p in points if p[0] == "cytoset_cells" and p[1] == fig4.ALL]   # n.e.: no CytoSet "all" point
    assert [p for p in points if p[0] == "set_transformer_cells" and p[1] == fig4.ALL]
    monkeypatch.setattr(common, "scores", lambda: scores[scores["key"] != "scrat_cells"])
    with pytest.raises(AssertionError, match="scrat_cells"):
        common.aurocs("lupus", "scrat_cells", **common.headline_conditions("scrat_cells"))


def test_blinded_mode(tree, monkeypatch):
    """A blinded run draws only the panels a blinded run can (BLINDED_PANELS), under the blinded root, stamped, and
    builds no data files; the others carry the stated note."""
    blinded = tree.root / "blinded_1"
    shutil.copytree(tree.root / LABELLED / "10c_concordance", blinded / "10c_concordance")
    monkeypatch.setattr(settings, "BLINDED", 1)
    monkeypatch.setattr(settings, "LABELLED_OUTPUTS", blinded)
    monkeypatch.setattr(common, "CACHE", {})
    monkeypatch.setattr(common, "WRITTEN", set())
    stamps = []
    real = common.check_figure
    monkeypatch.setattr(common, "check_figure", lambda fig, item: (stamps.append(any(t.get_text() == figure_style.BLINDED_WATERMARK["text"] for t in fig.texts)),
                                                                     real(fig, item)))
    make_all.main([])
    assert stamps and all(stamps)
    drawn = {(i, p) for i, ps in F.BLINDED_PANELS.items() for p in ps}
    assert common.WRITTEN == drawn
    assert not (blinded / "figures" / "data").exists() and not (blinded / "figures" / "Fig2").exists()
    assert "Fig 2" not in json.loads((blinded / "figures" / "figure_text.json").read_text())["items"]   # no stage 11 claims


def test_fig2_pooled_column_at_k1_and_k0(tree, monkeypatch, tmp_path):
    """Fig 2A/2C's pooled column: K = 1 is headed by the single pair; K = 0 has no pooled column, its place
    carries the sentence, and 11b's pooled reference rows must be absent."""
    import fig2
    claims, statistics, secondary = common.claims(), common.statistics(), common.secondary()
    monkeypatch.setattr(common, "directory", lambda item: tmp_path)
    titles_seen = []
    real = common.check_figure
    monkeypatch.setattr(common, "check_figure", lambda fig, item: (titles_seen.append({" ".join(a.get_title().split()) for a in fig.axes}), real(fig, item)))
    panels = {"Fig 2": tuple(F.PANELS["Fig 2"])}
    excluded = {"value": [p for p in PAIR_ORDER if p != "lupus"]}
    monkeypatch.setattr(common, "claims", lambda: {**claims, "excluded_at_ceiling": excluded["value"]})
    monkeypatch.setattr(common, "pseudobulk_auroc", lambda pair: 0.9 if pair not in excluded["value"] else 0.97)   # the strip's rule
    fig2.main(list(PAIR_ORDER), panels)
    assert " ".join(F.FIGURE_TEXT["pooled_header_k1"].format(mark=F.FIGURE_TEXT["pooled_mark"]).split()) in titles_seen[-1]
    assert set(pd.read_csv(tmp_path / "Fig2A.csv")["column"]) == {*PAIR_ORDER, "pooled"}
    excluded["value"] = list(PAIR_ORDER)
    monkeypatch.setattr(common, "claims", lambda: {**claims, "excluded_at_ceiling": excluded["value"]})
    monkeypatch.setattr(common, "statistics", lambda: statistics[statistics["pair"] != "pooled"])
    monkeypatch.setattr(common, "secondary", lambda: secondary[secondary["pair"] != "pooled"])
    fig2.main(list(PAIR_ORDER), panels)
    assert set(pd.read_csv(tmp_path / "Fig2A.csv")["column"]) == set(PAIR_ORDER)
    assert set(pd.read_csv(tmp_path / "Fig2C.csv")["column"]) == set(PAIR_ORDER)
    monkeypatch.setattr(common, "secondary", lambda: secondary)   # K = 0 with 11b's pooled reference rows: fails
    with pytest.raises(AssertionError):
        fig2.main(list(PAIR_ORDER), panels)


def test_strip_refuses_a_pooling_that_disagrees_with_stage11(tree, monkeypatch):
    """The strip's pseudobulk AUROC must agree with stage 11's pooling (below 1 - X iff pooled)."""
    claims = common.claims()
    pair = next(p for p in PAIR_ORDER if p not in claims["excluded_at_ceiling"])
    common.pseudobulk_auroc(pair)
    monkeypatch.setattr(common, "claims", lambda: {**claims, "excluded_at_ceiling": [*claims["excluded_at_ceiling"], pair]})
    with pytest.raises(AssertionError, match="stage 11's pooling"):
        common.pseudobulk_auroc(pair)


def test_only_keeps_other_items(tree):
    """--only redraws the named items and touches no other figure, figure_text.json or data file."""
    out = figures_root(tree)
    others = {p: p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file() and not p.is_relative_to(out / "S2Fig")}
    make_all.main(["--only", "Fig. S2"])
    assert {p: p.read_bytes() for p in others} == others


def test_s10c_mse_row(tree):
    """Fig. S10c's fourth row: CytoSet's MSE objective from 11b's best-candidate rows, aging pairs only, never
    pooled."""
    csv = pd.read_csv(figures_root(tree) / "S10Fig" / "S10CFig.csv")
    mse = csv[csv["key"] == "objective_mse"].set_index("pair")
    assert set(mse.index) == set(T.AGE_LOSS_PAIRS) and (mse["model"] == "cytoset").all()
    sec = schemas.read("secondary_statistics", tree.root / LABELLED / "11b_secondary" / "pooled" / f"{LABELLED}_secondary_statistics.parquet")
    best = sec[(sec["row"] == "best_candidate") & (sec["key"] == "objective_mse")].set_index("pair")
    assert np.allclose(mse.loc[list(best.index), ["delta", "ci_low", "ci_high"]].to_numpy(), best[["delta", "ci_low", "ci_high"]].to_numpy(),
                       equal_nan=True)
    assert set(csv.loc[csv["key"] != "objective_mse", "key"]) == {f"{e}_own_recipe" for e in ENCODERS}


def test_geneformer_pending(tree, monkeypatch, tmp_path):
    """From a gather run without the Geneformer deletion (10c then has no Geneformer rows), Fig 6B and S14
    print "n/a (pending)" for it, not the triage text, the figure carries the pending stamp and sources.json records it;
    a figure that read no stage 10 inclusion file has neither."""
    import fig2
    import fig6
    import s14
    from matplotlib.text import Text
    from sc2bulk import paths
    geneformer = ("geneformer_cells", "geneformer_pseudobulk")
    cache = dict(common.CACHE)
    for pair in PAIR_ORDER:
        cache[paths.output("10_attribution", pair, "inclusion.csv")] = pd.DataFrame(
            [{"pair": pair, "configuration": k, "included": False, "failed_checks": "pending"} for k in geneformer])
        path = paths.output("10_attribution", pair, "modality_summary.csv")
        summary = pd.read_csv(path)
        cache[path] = summary[~summary["configuration"].isin(geneformer)]
        path = paths.output("10c_concordance", pair, f"{common.LABELLED}concordance.parquet")
        concordance = schemas.read("concordance", path)
        cache[path] = concordance[~concordance["key"].isin(geneformer)]
    monkeypatch.setattr(common, "CACHE", cache)
    items = ("Fig 6", "Fig. S14", "Fig 2")
    for item in items:
        (tmp_path / common.stem(item)).mkdir()
    monkeypatch.setattr(common, "directory", lambda item: tmp_path / common.stem(item))
    texts = {}
    real = common.check_figure
    monkeypatch.setattr(common, "check_figure", lambda fig, item: (texts.update({item: {t.get_text() for t in fig.findobj(Text)}}),
                                                                     real(fig, item)))
    for module in (fig6, s14, fig2):
        module.main(list(PAIR_ORDER), {module.FIGURE: tuple(F.PANELS[module.FIGURE])})
    recorded = {item: json.loads((tmp_path / common.stem(item) / "sources.json").read_text())["geneformer_pending"] for item in items}
    for item in ("Fig 6", "Fig. S14"):
        assert figure_style.PENDING_WATERMARK["text"] in texts[item] and F.FIGURE_TEXT["pending"] in texts[item]
        assert F.FIGURE_TEXT["not_affordable"] not in texts[item] and recorded[item]
    assert figure_style.PENDING_WATERMARK["text"] not in texts["Fig 2"] and not recorded["Fig 2"]


def test_tables(tree):
    """Additional file 2 (figures/tables.py): every table and sub-table builds non-empty with its legend, the workbook has a
    sheet per sub-table, each sheet's CSV and each table's preview are written, no number carries a thousands separator,
    and Table S2A's differences are stage 11's."""
    import openpyxl
    import tables
    built = tables.main()
    assert list(built) == tables.ITEMS
    out = tree.root / LABELLED / "tables"
    book = openpyxl.load_workbook(out / "AdditionalFile2.xlsx")
    for item, (sheets, lines) in built.items():
        assert sheets and lines, item
        directory = out / item.replace(" ", "")
        preview = (directory / f"{directory.name}.md").read_text()
        assert not re.search(r"\d,\d{3}(?!\d)", preview), item
        for name, (_, frame, _) in sheets.items():
            assert len(frame) and len(frame.columns), (item, name)
            assert f"Table {name}" in book.sheetnames and (directory / f"Table{name}.csv").exists(), (item, name)
    assert json.loads((out / "sources.json").read_text())["item"] == "Additional file 2"
    stats = schemas.read("statistics", tree.root / LABELLED / "11_statistics" / "pooled" / f"{LABELLED}_statistics.parquet")
    stats = stats[(stats["pool_set"] == "per_pair") & (stats["readout"] == "ridge")]
    s2a = built["Table S2"][0]["S2A"][1].dropna(subset=["Δ bulk AUROC"]).set_index(["Pair", "Configuration"])["Δ bulk AUROC"]
    expected = stats.set_index([stats["pair"].map(lambda p: PAIRS[p]["name"]), stats["key"].map(common.label)])["delta"]
    assert len(s2a) == len(expected) and np.allclose(s2a.loc[expected.index].to_numpy(), expected.to_numpy())
    costs = common.costs()   # Table S11: pipeline stages only (no smoke tree), and S11B = the sum over unique work units
    stages = {p.stem for p in (settings.REPO / "pipeline").glob("[0-9]*.py")}
    assert len(costs) and set(costs["stage"]) <= stages
    units = costs.assign(seed=costs["seed"].astype(str)).groupby(["stage", "settings_hash", "seed"])["seconds"].max()
    hours = units.groupby("stage").sum() / 3600
    hours.index = hours.index.map(tables.STAGE_NAMES)
    totals = built["Table S11"][0]["S11B"][1].set_index("Analysis step")["Wall-clock hours"]
    assert sorted(totals.index) == sorted(hours.index) and np.allclose(totals.loc[hours.index].to_numpy(), hours.to_numpy())
    s11a = built["Table S11"][0]["S11A"][1].set_index(["Analysis step", "Pair", "Configuration"])
    assert s11a.loc[("Alignment", "Human blood, lupus", "CytoSet, cells"), "Wall-clock hours"] == 1.0   # one directory,
    assert s11a.loc[("Alignment", "Human blood, lupus", "scGPT, cells"), "Wall-clock hours"] == 2.0      # two units
    spanning = s11a.loc[("Sample properties", "Human blood, aging", "shared")]   # one unit, two directories
    assert spanning["Runs"] == 1 and spanning["Wall-clock hours"] == 1.0 and spanning["Outputs"] == 2
    selections = built["Table S4"][0]["S4C"][1]   # no adopted setting where the base was kept
    assert (selections.loc[selections["Kept its base"] == "yes", "Settings adopted (changes from the base)"] == "none").all()


def test_layout_checked_on_every_figure(tree):
    """Every drawn figure's layout is measured (common.check_layout); a figure out of config/figures.LAYOUT_PENDING has no
    problem left (report-only warnings aside)."""
    assert set(tree.layout) == {i for i, spec in F.ITEMS.items() if spec["script"] in make_all.SCRIPTS}
    for item, problems in tree.layout.items():
        assert all(re.match(r"(warn: )?[LWX]\d: ", p) for p in problems), item
        assert item in F.LAYOUT_PENDING or all(p.startswith("warn:") for p in problems), item


# ---- the layout checks, each on a small constructed figure that violates it ----

def blank(height_mm=80):
    figure_style.apply()
    return plt.figure(figsize=(S.WIDTH_MM["double"] / 25.4, height_mm / 25.4))


def codes(fig, panel_axes, rows=None) -> set:
    """The checks a figure fails (report-only ones included)."""
    problems = common.layout_problems(fig, panel_axes, rows or [(p,) for p in panel_axes])
    plt.close(fig)
    return {p.removeprefix("warn: ").split(":")[0] for p in problems}


def two_rows(hspace=0.5, letters=(0, 0)):
    fig = blank()
    grid = fig.add_gridspec(2, 1, top=0.85, bottom=0.12, left=0.1, right=0.97, hspace=hspace)
    a, b = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    figure_style.panel_label(a, "A", at_left=True, left_mm=letters[0])
    figure_style.panel_label(b, "B", at_left=True, left_mm=letters[1])
    return fig, {"A": [a], "B": [b]}


def test_layout_l1_letters_share_a_left_edge():
    assert "L1" not in codes(*two_rows())
    assert "L1" in codes(*two_rows(letters=(0, 3)))


def test_layout_l2_letter_level_with_its_panel():
    fig = blank()
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.6))
    figure_style.panel_label(ax, "A", at_left=True, y=1.3)
    assert "L2" in codes(fig, {"A": [ax]})


def test_layout_w1_row_gaps():
    assert "W1" in codes(*two_rows(hspace=0.05))
    assert "W1" in codes(*two_rows(hspace=1.5))


def test_layout_w2_empty_bands_and_margins():
    fig = blank()
    a, b = fig.add_axes((0.1, 0.75, 0.85, 0.2)), fig.add_axes((0.1, 0.1, 0.85, 0.2))   # ~30 mm apart
    assert "W2" in codes(fig, {"A": [a], "B": [b]})
    fig = blank()
    ax = fig.add_axes((0.3, 0.1, 0.65, 0.8))   # ~45 mm left of the axes
    assert "W2" in codes(fig, {"A": [ax]})


def test_layout_w3_gap_inside_a_panel():
    fig = blank()
    axes = [fig.add_axes((0.1, 0.1, 0.2, 0.8)), fig.add_axes((0.6, 0.1, 0.35, 0.8))]
    assert "W3" in codes(fig, {"A": axes})
    fig = blank()   # the spacer before a pooled column is deliberate
    axes = plots.columns(fig, fig.add_gridspec(1, 1, left=0.1, right=0.97)[0], 3, gap=2)
    for ax in axes:
        ax.set_yticks([])
    assert "W3" not in codes(fig, {"A": axes})


def test_layout_x1_text_outside():
    fig = blank()
    ax = fig.add_axes((0.1, 0.1, 0.85, 0.8))
    fig.text(0.99, 0.5, "past the right edge")
    assert "X1" in codes(fig, {"A": [ax]})


def test_layout_x2_overlaps():
    fig = blank()
    ax = fig.add_axes((0.1, 0.1, 0.4, 0.8))
    ax.text(0.5, 0.5, "one")
    ax.text(0.5, 0.5, "two")
    problems = common.layout_problems(fig, {"A": [ax]}, [("A",)])
    assert any(p.startswith("X2: 'one' overlaps 'two'") or p.startswith("X2: 'two' overlaps 'one'") for p in problems)
    fig = blank()
    a, b = fig.add_axes((0.1, 0.1, 0.4, 0.8)), fig.add_axes((0.52, 0.1, 0.4, 0.8))   # b's tick labels inside a
    figure_style.panel_label(b, "B", x=0.1, y=0.5)   # a letter on its own axes
    problems = common.layout_problems(fig, {"A": [a], "B": [b]}, [("A", "B")])
    assert any(p.startswith("X2:") and "ytick text of panel B overlap another axes of panel A" in p for p in problems)
    assert any(p.startswith("X2: letter b overlaps") for p in problems)
    plt.close("all")


def test_layout_x3_matching_columns_aligned():
    fig = blank(120)
    grid = fig.add_gridspec(2, 1, top=0.9, bottom=0.1, left=0.1, right=0.97, hspace=0.5)
    a = plots.columns(fig, grid[0], 3)
    b = plots.columns(fig, grid[1], 3, wspace=0.8)
    assert "X3" in codes(fig, {"A": a, "B": b})


def test_layout_x4_sibling_axes():
    fig = blank()
    axes = plots.columns(fig, fig.add_gridspec(1, 1, left=0.1, right=0.97)[0], 3)
    axes[1].set_box_aspect(0.3)
    assert "X4" in codes(fig, {"A": axes})


def test_layout_x5_font_roles():
    fig = blank()
    ax = fig.add_axes((0.1, 0.1, 0.85, 0.8))
    ax.tick_params(axis="x", labelsize=8)   # y stays at 6: two sizes
    ax.text(0.5, 0.5, "large", fontsize=12)
    problems = common.layout_problems(fig, {"A": [ax]}, [("A",)])
    assert any(p.startswith("X5: tick labels at 6 pt") and "8 pt" in p for p in problems)
    assert any(p.startswith("warn: X5:") and "at 12 pt" in p for p in problems)
    fig = blank()
    ax = fig.add_axes((0.1, 0.1, 0.85, 0.8))
    ax.text(0, 1, "a", gid="panel_label", fontsize=S.PANEL_LABEL["size"])   # not bold
    assert "X5" in codes(fig, {"A": [ax]})
    fig = blank()
    ax = fig.add_axes((0.1, 0.1, 0.85, 0.8))
    ax.tick_params(labelsize=7)   # one size, the house 7 pt: fine
    assert "X5" not in codes(fig, {"A": [ax]})
    fig = blank()
    ax = fig.add_axes((0.1, 0.1, 0.85, 0.8))
    ax.tick_params(labelsize=5)   # one size, below the floor
    assert "X5" in codes(fig, {"A": [ax]})


def test_layout_x6_axis_labels():
    fig = blank()
    ax = fig.add_axes((0.2, 0.1, 0.75, 0.8))
    ax.set_ylabel("far", labelpad=30)
    assert "X6" in codes(fig, {"A": [ax]})
    fig = blank()
    axes = plots.columns(fig, fig.add_gridspec(1, 1, left=0.1, right=0.97, bottom=0.2)[0], 3)
    axes[0].set_xlabel("shared")   # under the first column, not the middle
    assert "X6" in codes(fig, {"A": axes})
    for n, gap in ((3, 2), (4, 3), (5, None)):   # plots.centered_xlabel's placement, a pooled column included
        fig = blank()
        axes = plots.columns(fig, fig.add_gridspec(1, 1, left=0.1, right=0.97, bottom=0.2)[0], n, gap=gap)
        plots.centered_xlabel(axes, "shared")
        assert "X6" not in codes(fig, {"A": axes}), (n, gap)


def test_layout_x7_canonical_names():
    fig = blank()
    ax = fig.add_axes((0.1, 0.1, 0.85, 0.7))
    ax.set_title(common.pair_title("hammond").replace(" ", "-"))   # a punctuation difference
    assert "X7" in codes(fig, {"A": [ax]})
    fig = blank()
    ax = fig.add_axes((0.1, 0.1, 0.85, 0.7))
    ax.set_title(common.pair_title("hammond").replace("microglia", "brain microglia"))   # a word inserted
    assert "X7" in codes(fig, {"A": [ax]})
    fig = blank()
    ax = fig.add_axes((0.1, 0.1, 0.85, 0.7))
    ax.set_title(f"{plots.narrow(common.pair_title('hammond'), 10)} {F.FIGURE_TEXT['pooled_mark']}")   # rewrapped and marked: the same words
    assert "X7" not in codes(fig, {"A": [ax]})
    fig = blank()
    ax = fig.add_axes((0.1, 0.1, 0.85, 0.7))
    ax.set_title(f"{common.pair_title('hammond')} p = 0.96")   # a name with a value after it
    assert "X7" not in codes(fig, {"A": [ax]})


def test_check_layout_asserts_outside_the_pending_set(monkeypatch):
    fig, panel_axes = two_rows(letters=(0, 3))
    monkeypatch.setattr(F, "LAYOUT_PENDING", {"pending"})   # names the tree fixture's recording wrapper never saw
    assert any(p.startswith("L1:") for p in common.check_layout(fig, "pending", panel_axes, [("A",), ("B",)]))
    with pytest.raises(AssertionError, match="L1"):
        common.check_layout(fig, "fixed", panel_axes, [("A",), ("B",)])
    plt.close(fig)
