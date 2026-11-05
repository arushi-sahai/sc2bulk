"""Stage 04d: dataset characteristics (Fig 1E's computed columns, S8 Fig, S1 Table).

Descriptive and label-free on the bulk side: only single-cell labels are read (bulk label
distributions are joined in the figure scripts, behind the firewall). The rules and per-pair
settings are pre-declared in config/fig1e.py.

Cell-level values come from stage 02's per-cell QC table (every cell before QC, with its
annotation). The stored cells are reproduced from it exactly (each sample's post-QC cells taken
in preprocess.cell_order, the first MAX_STORED_CELLS; counts asserted against stage 02), so no
stored expression is loaded. Computed columns use the pair's primary unit's training samples
(folds.training_samples). No stage consumes these outputs as model features: they describe all
donors, outside the fold restriction.

Outputs, per pair, in <outputs>/04d_characteristics/<pair>/:
  characteristics.csv  every Fig 1E column: state (yes / no / not determinable), value, annotation
  composition.csv      per kept single-cell sample and annotated type: post-QC cells, fraction
  qc_steps.csv         per single-cell sample: cells before QC, failing each threshold, passing
  batch_label.csv      primary unit's training samples: batch x 0/1 group counts (and per donor
                       for human blood, aging)
  unit_sizes.csv       per unit: training samples and donors, scored bulk samples
  markers.csv          pairs whose tissue or preparation differs: mean log1p CPM of the marker
                       genes on each side

  python pipeline/04d_characteristics.py [--pair PAIR]
"""
import argparse

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2_contingency

from sc2bulk import determinism, folds, manifest, paths, preprocess
from sc2bulk.config import fig1e, settings, sources
from sc2bulk.config.pairs import PAIR_ORDER, PAIRS
from sc2bulk.data import lupus

STAGE = "04d_characteristics"
SAMPLES = settings.OUTPUTS / "02_preprocess"
OUTPUTS = ("characteristics.csv", "composition.csv", "qc_steps.csv", "batch_label.csv", "unit_sizes.csv", "markers.csv")


def input_paths(pair) -> list:
    return folds.inputs(pair) + [SAMPLES / pair / name for name in ("cell_qc.csv.gz", "bulk.h5ad", "bulk_samples.csv")] \
        + [settings.OUTPUTS / "03_splits" / pair / "donor_folds.csv", paths.output("04_panel", pair, "shared_genes.csv")]


def record_for(pair) -> dict:
    raw = {n: spec[-1] for (d, n), spec in sources.DOWNLOADS.items() if pair == "lupus" and n == sources.LUPUS_SINGLE_CELL}
    return {"stage": STAGE, "pair": pair, "smoke": settings.SMOKE,
            "code": manifest.stage_code_hash(settings.REPO / "pipeline" / f"{STAGE}.py", pair),
            "inputs": manifest.input_hashes(input_paths(pair)), "raw_inputs": raw}


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013); 0 when either variable has one level."""
    table = pd.crosstab(a, b).to_numpy()
    r, k = table.shape
    if min(r, k) < 2:
        return 0.0
    n = table.sum()
    phi2 = chi2_contingency(table, correction=False)[0] / n
    phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r, k = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / min(k - 1, r - 1))) if min(k, r) > 1 else 0.0


def lupus_pools(samples: pd.DataFrame) -> pd.Series:
    """Per kept lupus sample, its pool: the connected component of the sample-library graph over
    the sample's processing-cohort-4 cells (the cells stage 02 keeps)."""
    obs = lupus._single_cell(settings.RAW).obs[["sample_uuid", "library_uuid", "Processing_Cohort"]].astype(str)
    obs = obs[(obs["Processing_Cohort"] == lupus.COHORT) & obs["sample_uuid"].isin(samples["sample_id"])]
    return pools(obs.rename(columns={"sample_uuid": "sample", "library_uuid": "library"}), samples)


def pools(cells: pd.DataFrame, samples: pd.DataFrame) -> pd.Series:
    """Each sample's pool: the connected component of the sample-library graph (cells: one row per
    cell, columns sample and library), named pool_0, pool_1, ... in order of the smallest sample id."""
    edges = cells.drop_duplicates(["sample", "library"]).rename(columns={"sample": "sample_uuid", "library": "library_uuid"})
    ids = pd.Index(sorted(set(edges["sample_uuid"]))).append(pd.Index(sorted(set(edges["library_uuid"]))))
    graph = sp.coo_matrix((np.ones(len(edges)), (ids.get_indexer(edges["sample_uuid"]), ids.get_indexer(edges["library_uuid"]))),
                          shape=(len(ids), len(ids)))
    _, component = connected_components(graph, directed=False)
    pool = pd.Series(component, index=ids).loc[samples["sample_id"]]
    # pools numbered in the order of their smallest sample id, so the names don't depend on graph order
    order = pool.index.to_series().groupby(pool.to_numpy()).min().sort_values()
    return pool.map({c: f"pool_{i}" for i, c in enumerate(order.index)}).set_axis(samples.index)


def many_cell_types(pair, cells: pd.DataFrame, train: pd.DataFrame) -> tuple:
    """(state, value, annotation) for "Many cell types" on the primary unit's training cells."""
    level = fig1e.ANNOTATION[pair]
    cells = cells[cells["sample_id"].isin(train["sample_id"])]

    def qualifying(c):
        share = c[level].value_counts(normalize=True)
        return int((share >= fig1e.MANY_TYPES_FRACTION).sum())

    if pair == "tms":
        organ = cells["sample_id"].map(train.set_index("sample_id")["organ"])
        per_organ = {o: qualifying(c) for o, c in cells.groupby(organ)}
        passing = sum(n >= fig1e.MANY_TYPES_MIN for n in per_organ.values())
        state = "yes" if passing > len(per_organ) / 2 else "no"
        return state, passing, "; ".join(f"{o}: {n} types" for o, n in sorted(per_organ.items()))
    n = qualifying(cells)
    return ("yes" if n >= fig1e.MANY_TYPES_MIN else "no"), n, f"{level}: {n} types >= {fig1e.MANY_TYPES_FRACTION:.0%}"


def stored(pair, qc: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """The stored cells of the kept samples: each sample's post-QC rows (in stage 02's order), in
    preprocess.cell_order, the first MAX_STORED_CELLS."""
    passed = qc[qc["passed_qc"]]
    parts = []
    for sample_id, rows in passed.groupby("sample_id", sort=True):
        if sample_id in set(samples["sample_id"]):
            parts.append(rows.iloc[np.sort(preprocess.stored_cells(pair, sample_id, len(rows)))])
    cells = pd.concat(parts)
    counts = cells.groupby("sample_id").size()
    expected = samples.set_index("sample_id")["cells_stored"]
    assert counts.reindex(expected.index).eq(expected).all(), "reproduced stored cells differ from stage 02's counts"
    return cells


def batch_column(pair, samples: pd.DataFrame) -> pd.Series:
    return lupus_pools(samples) if fig1e.BATCH[pair] == "pool" else samples[fig1e.BATCH[pair]].astype(str)


def tissue_markers(pair) -> pd.DataFrame:
    """Mean log1p CPM of each marker gene over the pair's single-cell pseudobulk and bulk samples,
    CPM over the shared genes on both sides (stage 04), so the denominators match."""
    rows = []
    shared = list(pd.read_csv(paths.output("04_panel", pair, "shared_genes.csv"), keep_default_na=False)["gene"])
    bulk = anndata.read_h5ad(SAMPLES / pair / "bulk.h5ad")   # label-free bulk expression
    for side, adata in (("single_cell", folds.all_pseudobulk(pair)[:, shared]), ("bulk", bulk[:, shared])):
        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        cpm = np.log1p(X / X.sum(axis=1, keepdims=True) * 1e6)
        for marker, (genes, organ) in fig1e.TISSUE_MARKERS[pair].items():
            rows_in = np.ones(adata.n_obs, bool) if organ is None else (adata.obs["organ"].astype(str) == organ).to_numpy()
            for gene in genes:
                present = gene in adata.var_names
                value = float(cpm[rows_in, adata.var_names.get_loc(gene)].mean()) if present else np.nan
                rows.append({"side": side, "marker": marker, "organ": organ, "gene": gene, "mean_log1p_cpm": value,
                             "present": present})
    return pd.DataFrame(rows)


def characteristics(pair, cells, train, batch) -> pd.DataFrame:
    rows = []
    typed = fig1e.FIG1E[pair]
    for column in fig1e.FIG1E_COLUMNS:
        if column == "Many cell types":
            state, value, note = many_cell_types(pair, cells, train)
        elif column == "Batch confounded with label":
            if fig1e.BATCH[pair] is None:
                state, value, note = "not determinable", np.nan, fig1e.NOT_DETERMINABLE[pair]
            else:
                value = cramers_v(batch, train["group"])
                state, note = ("yes" if value >= fig1e.BATCH_V else "no"), f"{fig1e.BATCH[pair]}: Cramer's V {value:.2f}"
        elif column == "Few training donors (< 100)":
            value = train["donor_id"].nunique()
            state, note = ("yes" if value < fig1e.FEW_TRAINING_DONORS else "no"), f"{value} training donors, {len(train)} samples"
        else:
            state, note = typed.get(column, ("no", ""))
            value = np.nan
        rows.append({"pair": pair, "column": column, "state": state, "value": value, "annotation": note})
    return pd.DataFrame(rows)


def qc_steps(pair, cells: pd.DataFrame) -> pd.DataFrame:
    """Per sample: cells before QC, failing each threshold (not exclusive: a cell can fail both),
    passing (asserted to match stage 02), and whether stage 02 kept the sample."""
    ceiling = PAIRS[pair].get("qc_max_percent_mito", settings.QC_MAX_PERCENT_MITO)
    genes_ok = cells["genes_detected"] >= settings.QC_MIN_GENES_PER_CELL
    mito_ok = cells["percent_mito"] < ceiling
    assert (cells["passed_qc"] == (genes_ok & mito_ok)).all(), "stage 02's QC flags differ from the thresholds"
    table = pd.DataFrame({"sample_id": cells["sample_id"], "before_qc": 1, "fail_min_genes": ~genes_ok,
                          "fail_mito_ceiling": ~mito_ok, "passed_qc": cells["passed_qc"]})
    table = table.groupby("sample_id", sort=True).sum().astype(int).reset_index()
    kept = folds.single_cell_samples(pair)["sample_id"]
    table["sample_kept"] = table["sample_id"].isin(kept)
    return table


def write_csv(frame, path, record, unit, inputs):
    frame.to_csv(manifest.temporary(path), index=False)
    manifest.finish(path, record, unit, inputs=inputs)


def main():
    determinism.configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=PAIR_ORDER)
    args = parser.parse_args()
    for pair in [args.pair] if args.pair else PAIR_ORDER:
        record = record_for(pair)
        outputs = [paths.output(STAGE, pair, name) for name in OUTPUTS]
        if manifest.is_done(outputs, record):
            print(f"{pair}: done", flush=True)
            continue
        outputs[0].parent.mkdir(parents=True, exist_ok=True)
        unit = manifest.begin()
        level = fig1e.ANNOTATION[pair]
        qc_cells = pd.read_csv(SAMPLES / pair / "cell_qc.csv.gz", dtype={"sample_id": str},
                                usecols=["sample_id", level, "genes_detected", "percent_mito", "passed_qc"])
        samples = folds.single_cell_samples(pair)
        cells = stored(pair, qc_cells, samples)
        primary = folds.units(pair)[0]
        train = folds.training_samples(pair, *primary)
        batch = batch_column(pair, train) if fig1e.BATCH[pair] is not None else None

        table = characteristics(pair, cells, train, batch)
        composition = (cells.groupby(["sample_id", level]).size().rename("cells").reset_index()
                       .rename(columns={level: "cell_type"}))
        composition["fraction"] = composition["cells"] / composition.groupby("sample_id")["cells"].transform("sum")
        composition.insert(1, "level", level)
        batch_label = pd.DataFrame(columns=["level", "batch", "group", "count"])
        if batch is not None:
            batch_label = pd.crosstab(batch, train["group"]).stack().rename("count").reset_index()
            batch_label.columns = ["batch", "group", "count"]
            batch_label.insert(0, "level", "sample")
            if pair == "soundlife":   # one row per donor: its modal batch
                donors = train.assign(batch=batch).groupby("donor_id").agg(
                    batch=("batch", lambda b: b.value_counts().idxmax()), group=("group", "first"))
                per_donor = pd.crosstab(donors["batch"], donors["group"]).stack().rename("count").reset_index()
                per_donor.columns = ["batch", "group", "count"]
                batch_label = pd.concat([batch_label, per_donor.assign(level="donor")], ignore_index=True)
                v = cramers_v(donors["batch"], donors["group"])
                table.loc[table["column"] == "Batch confounded with label", "annotation"] += f" (sample level, bias-corrected); per donor (modal batch) V {v:.2f}"
        bulk_samples = pd.read_csv(SAMPLES / pair / "bulk_samples.csv", dtype={"sample_id": str, "donor_id": str})
        bulk_samples = bulk_samples[bulk_samples["drop_reason"].isna()]
        sizes = pd.DataFrame([{"fold_set": fs, "fold": f, "training_samples": len(t), "training_donors": t["donor_id"].nunique(),
                               "scored_bulk_samples": len(folds.scored_bulk(pair, fs, f, bulk_samples))}
                              for fs, f in folds.units(pair) for t in [folds.training_samples(pair, fs, f)]])
        markers = tissue_markers(pair) if pair in fig1e.TISSUE_MARKERS else pd.DataFrame(
            columns=["side", "marker", "organ", "gene", "mean_log1p_cpm", "present"])

        inputs = input_paths(pair)
        for frame, path in zip((table, composition, qc_steps(pair, qc_cells), batch_label, sizes, markers), outputs):
            write_csv(frame, path, record, unit, inputs)
        print(f"{pair}: " + "; ".join(f"{r.column}: {r.state}" for r in table.itertuples()), flush=True)


if __name__ == "__main__":
    main()
