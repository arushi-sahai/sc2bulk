"""The data files (Additional files 3-6) and the Additional-file mapping. Real runs only: a blinded run builds no data files (its per-sample files are
label-equivalent).

Each data file is one zip of CSVs plus README.csv (file, item, description, columns), at most 20 MB (Genome
Biology's limit per additional file), written byte-reproducibly to <labelled outputs>/figures/data/:
  Additional file 3  source data: every registered panel's CSV, in registry order (the README is the index);
  Additional file 4  gene rankings, per pair: stage 10's rankings (configuration x read-out x target: gene, attribution, rank, top-k flags),
           Geneformer's inclusion and the attribution checks, and Fig 5A's gene shifts (stage 10c);
  Additional file 5  search candidates, per pair: every search candidate per unit and search seed (stage 05: change, validation BCE, did-it-train),
           with its bulk AUROC (stage 11b's scores), and scVI's search (stage 05v's ELBO);
  Additional file 6  per-sample predictions, per pair: per bulk sample, each configuration's and reference row's score and probability averaged over
           encoder and read-out seeds (ridge, primary), with its fold, scoring unit and 0/1 group; and the same for the
           within-single-cell predictions (Fig 2B's samples). Per-seed predictions are stage 08's files (Zenodo).
The donor-level rule: Additional files 4-6 carry no column of config/figures.SENSITIVE_COLUMNS; in
Additional file 3 only the panels config/figures.PANEL_SENSITIVE names may (asserted on writing; tests/test_s_data.py).
additional_files.csv: item, Additional file number, file name, title, from the citation plan's first main-body citations
(every supplementary figure in Additional file 1, every supplementary table in Additional file 2)."""
import io
import zipfile

import pandas as pd

import citation_order
import common
from sc2bulk.config import figures as F
from sc2bulk.config import style as S
from sc2bulk.config.pairs import PAIRS

PREDICTION_KEYS = ["pseudobulk_pca64", "mean_log_cells", "per_cell_classifier", "cytoset_untrained", "deconvolution_clr",
                   "deconvolution_true", "known_signature"]


def sensitive(frame, allowed=()) -> list:
    return [c for c in frame.columns if c in F.SENSITIVE_COLUMNS and c not in allowed]


def write_zip(name, files: dict, descriptions: dict):
    """files: CSV name -> frame; a README.csv indexes them. Byte-reproducible (fixed timestamps, sorted members)."""
    readme = pd.DataFrame([{"file": f, "item": descriptions[f][0], "description": descriptions[f][1], "columns": ";".join(map(str, frame.columns))}
                           for f, frame in files.items()])
    members = {"README.csv": readme, **files}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for member, frame in members.items():
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, frame.to_csv(index=False))
    data = buffer.getvalue()
    assert len(data) <= S.MAX_ADDITIONAL_FILE_BYTES, f"{name}: {len(data)} bytes, over Genome Biology's 20 MB"
    out = common.root() / "data"
    out.mkdir(exist_ok=True)
    common.write_bytes(out / f"{name.replace(' ', '_')}.zip", lambda p: p.write_bytes(data))
    return readme


def source_data() -> pd.DataFrame:
    """Every registered panel's CSV (written by the figure scripts), in registry order."""
    files, descriptions = {}, {}
    for item, panels in F.PANELS.items():
        for panel, description in panels.items():
            if (item, panel) not in common.WRITTEN:
                continue
            name = f"{common.panel_name(item, panel)}.csv"
            frame = common.read_written(item, panel)
            allowed = F.PANEL_SENSITIVE.get(common.citation(item, panel), ())
            assert not sensitive(frame, allowed), f"Additional file 3 {name}: donor-level columns {sensitive(frame, allowed)} (frozen §18)"
            files[name], descriptions[name] = frame, (common.citation(item, common.panel_letter(panel)), description)
    return write_zip("Additional file 3", files, descriptions)


def gene_rankings(pairs):
    files, descriptions = {}, {}
    for pair in pairs:
        rankings = []
        for key in [k for k in common.kept() if k not in common.dropped(pair)]:
            table = common.read_kind("attributions", "10_attribution", pair, f"{key}_ranking.parquet")
            rankings.append(table[["key", "readout", "target", "method", "gene", "attribution", "rank", "top25", "top50", "top100"]])
        files[f"rankings_{pair}.csv"] = pd.concat(rankings, ignore_index=True)
        descriptions[f"rankings_{pair}.csv"] = (PAIRS[pair]["name"], "gene rankings per configuration, readout and target (stage 10)")
        for name in ("inclusion.csv", "checks.csv"):
            files[f"{name[:-4]}_{pair}.csv"] = common.read_csv("10_attribution", pair, name, keep_default_na=False)
            descriptions[f"{name[:-4]}_{pair}.csv"] = (PAIRS[pair]["name"], f"stage 10's {name}")
        shifts = common.read_kind("gene_shifts", "10c_concordance", pair, f"{common.LABELLED}gene_shifts.parquet")
        files[f"gene_shifts_{pair}.csv"] = shifts[["gene", "single_cell_shift", "bulk_shift", "signature_direction", "pseudobulk_top50"]]
        descriptions[f"gene_shifts_{pair}.csv"] = (PAIRS[pair]["name"], "Fig 5A's phenotype shifts (stage 10c)")
    for name, frame in files.items():
        assert not sensitive(frame), f"Additional file 4 {name}: {sensitive(frame)}"
    return write_zip("Additional file 4", files, descriptions)


def search_candidates(pairs):
    files, descriptions = {}, {}
    for pair in pairs:
        frames = []
        for unit in common.primary_units(pair):
            for encoder in ("cytoset", "set_transformer", "scrat"):
                for name in common.names("05_search", pair, f"{encoder}/candidates*_s*.parquet", unit):
                    frames.append(common.read_kind("candidates", "05_search", pair, name, unit))
        candidates = pd.concat(frames, ignore_index=True)
        scores = common.pick(common.scores(), pair=pair, row_kind="candidate", side="bulk", bulk_subset="all",
                             fold_set=common.primary_units(pair)[0][0])
        auroc = scores.groupby(["model", "fold", "key", "encoder_seed"], as_index=False)["auroc"].mean().rename(columns={"auroc": "bulk_auroc"})
        table = candidates.merge(auroc, on=["model", "fold", "key", "encoder_seed"], how="left", validate="one_to_one")
        files[f"candidates_{pair}.csv"] = table[["fold", "model", "key", "change_group", "change_value", "encoder_seed", "validation_bce",
                                                 "did_train", "did_train_reason", "selected", "merged", "bulk_auroc"]]
        descriptions[f"candidates_{pair}.csv"] = (PAIRS[pair]["name"], "every search candidate per search seed (stage 05) with its bulk AUROC")
        elbo = [common.read_csv("05v_scvi_search", pair, f"elbo_s{i}.csv", unit).rename(columns={"group": "change_group"}).assign(fold=unit[1], search_seed=i)
                for unit in common.primary_units(pair) for i in common.search_seeds()]
        files[f"scvi_search_{pair}.csv"] = pd.concat(elbo, ignore_index=True)
        descriptions[f"scvi_search_{pair}.csv"] = (PAIRS[pair]["name"], "scVI's search (stage 05v): validation ELBO per candidate and seed")
    for name, frame in files.items():
        assert not sensitive(frame), f"Additional file 5 {name}: {sensitive(frame)}"
    return write_zip("Additional file 5", files, descriptions)


def averaged(table: pd.DataFrame) -> pd.DataFrame:
    """Per sample, the score and probability averaged over encoder and read-out seeds."""
    columns = ["key", "fold", "side", "sample_id", "scoring_unit_id"]
    return table.groupby(columns, as_index=False)[["score", "probability"]].mean()


def predictions(pairs):
    files, descriptions = {}, {}
    for pair in pairs:
        frames = []
        for unit in common.primary_units(pair):
            for key in common.kept():
                for suffix in ("predictions", "within_predictions"):
                    table = common.read_kind("predictions", "08_readout", pair, f"{key}_{suffix}.parquet", unit)
                    frames.append(averaged(common.pick(table, encoder_seed=tuple(common.seeds_of(key) if suffix == "predictions" or
                                                                                    common.within_side(pair) == "single_cell_heldout" else common.encoder_seeds()),
                                                       **common.headline_conditions(key))))
            for key in [k for k in PREDICTION_KEYS if k in reference_files(pair, unit)]:
                table = common.read_kind("predictions", "08b_secondary", pair, f"reference_rows/{key}_predictions.parquet", unit)
                frames.append(averaged(common.pick(table, neutral=False, side="bulk", variant="primary", row_kind="reference")))
        table = pd.concat(frames, ignore_index=True)
        bulk = common.bulk_labels(pair).set_index("sample_id")["group"]
        single = common.single_cell_samples(pair).set_index("sample_id")["group"]
        table["group"] = [bulk.get(s) if side == "bulk" else single.get(s) for s, side in zip(table["sample_id"], table["side"])]
        assert table["group"].isin([0, 1]).all(), f"Additional file 6 {pair}: a sample without a 0/1 group"
        files[f"predictions_{pair}.csv"] = table.astype({"group": int})
        descriptions[f"predictions_{pair}.csv"] = (PAIRS[pair]["name"], "per-sample predictions averaged over encoder and readout seeds (ridge)")
    for name, frame in files.items():
        assert not sensitive(frame), f"Additional file 6 {name}: {sensitive(frame)}"
    return write_zip("Additional file 6", files, descriptions)


def reference_files(pair, unit) -> set:
    """The reference rows stage 08b formed for the unit (deconvolution: not mouse microglia)."""
    return {n.split("/")[-1].removesuffix("_predictions.parquet") for n in common.names("08b_secondary", pair, "reference_rows/*_predictions.parquet", unit)}


def additional() -> pd.DataFrame:
    mapping = citation_order.additional_files(F.CITATIONS)
    table = pd.DataFrame([{"item": item, "additional_file": n, "file": f"Additional file {n}", "title": F.TITLES.get(item, "")}
                          for item, n in mapping.items()])
    common.write_bytes(common.root() / "additional_files.csv", lambda p: table.to_csv(p, index=False))
    return table


def main(pairs):
    common.start()
    source_data()
    gene_rankings(pairs)
    search_candidates(pairs)
    predictions(pairs)
    additional()
