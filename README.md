# sc2bulk

Code for *A benchmark of learned single-cell sample representations against pseudobulk for predicting phenotypes in bulk RNA-seq cohorts* (Arushi Sahai, Beatriz Campillo, Luvna Dhawka, and Natalie Stanley) (preprint forthcoming).

The paper benchmarks how a sample's single cells should be represented when a phenotype predictor trained on single-cell RNA-seq is applied to bulk RNA-seq. It compares pseudobulk (a sample's counts summed) with nine learned configurations: three supervised set encoders (CytoSet, Set Transformer, ScRAT), an unsupervised generative model (scVI), and two frozen foundation models (Geneformer, scGPT), with CytoSet and the foundation models also applied to each sample's pseudobulk profile. For each of six single-cell → bulk dataset pairs, a ridge classifier trained on the single-cell samples classifies the bulk samples, and performance is measured by the area under the ROC curve (AUROC).

## What this reproduces

Every figure and table in the paper, from the public raw data, with one command. The learned models used about 2,100 GPU hours on NVIDIA RTX 3090 GPUs (the paper's Table S11); pseudobulk needs no GPU.

## Installation

- Python 3.10, conda, and a CUDA GPU for encoder training and the foundation-model steps.
- The main environment:
  ```
  git clone https://github.com/arushi-sahai/sc2bulk && cd sc2bulk
  conda create -n sc2bulk python=3.10 && conda activate sc2bulk
  pip install -r requirements.txt && pip install -e .
  ```
- ScRAT, scVI, Geneformer, and scGPT run in their own environments (`envs/scrat.txt`, `envs/scvi.txt`, `envs/fm-geneformer.txt`, `envs/fm-scgpt.txt`), each built as described at the top of its file; `pipeline/run_all.py` runs each stage in the environment it needs (default names `sc2bulk`, `scrat`, `scvi`, `fm-geneformer`, `fm-scgpt`; `SC2BULK_ENV_<NAME>` overrides one).
- Three external models are fetched, never redistributed, and found through environment variables:
  - `SC2BULK_SCRAT`: a clone of ScRAT (https://github.com/yuzhenmao/ScRAT) at commit cbf0772 (`envs/scrat.txt`).
  - `SC2BULK_GENEFORMER`: a clone of Geneformer (https://huggingface.co/ctheodoris/Geneformer) at commit 1f7fbae; the V2-104M checkpoint's files are checked against pinned SHA-256s at load.
  - `SC2BULK_SCGPT`: the scGPT whole-human checkpoint directory (`best_model.pt`, `vocab.json`, `args.json`; `envs/fm-scgpt.txt` gives its source and MD5).
- Check the installation with the unit tests (no data needed) and, once the data are downloaded (below), the smoke run, which runs every stage on a tiny subset in its own `outputs/smoke/`:
  ```
  pytest -q tests
  python pipeline/run_all.py --smoke
  ```

Other environment variables: `SC2BULK_OUTPUTS` (where every stage writes; default `outputs/`), `SC2BULK_THREADS` (BLAS, OpenMP, and torch threads; default 16), and `SC2BULK_PERMUTE_BULK_LABELS=<seed>` (a test run with randomly permuted bulk labels, whose label-dependent outputs go to `outputs/blinded_<seed>/`).

## Data

| Pair | Single-cell side | Bulk side |
|---|---|---|
| Mouse microglia, aging | Hammond et al. 2019, mouse microglia (GEO GSE121654) | Reemst et al. 2022, hippocampal microglia (GEO GSE207067) |
| Mouse organs, aging | Tabula Muris Senis, 4 organs (CELLxGENE) | Schaum et al. 2020, the same organs (GEO GSE132040) |
| Human blood, aging | AIFI Sound Life PBMC (doi:10.57785/bn17-3h16) | Sound Life whole blood, same donors (cross-fitted by donor) |
| Human blood, lupus | Perez et al. 2022, PBMC (CELLxGENE) | Hung et al. 2015, whole blood (SRP062966, via recount3) |
| Human ileum, Crohn's | IBDverse, terminal ileum (CELLxGENE) | MSCCR, ileal biopsies (GEO GSE193677) |
| Human motor cortex, ALS | Pineda et al. 2024, motor cortex snRNA-seq (Synapse) | NYGC ALS Consortium motor cortex (GEO GSE153960) |

The first pipeline stage downloads everything it can and checks each file against its checksum. Two inputs need manual steps:

- **Sound Life single-cell.** The AIFI portal serves its files to a browser only. Download the eight "Sound Life subject group" scRNA-seq files (~248 GB) and run `python pipeline/00_soundlife_slims.py <directory>`. It keeps a fixed random 3,000 cells per sample (the shipped barcode lists in `src/sc2bulk/data/slim_barcodes/`) and checks the result against a pinned content checksum. This rebuild was run from start to finish and reproduced every file's recorded checksum. Also download `sound-life_whole-blood_no-stim.h5ad` (bulk).
- **Pineda single-nucleus.** Stage 01 downloads it from Synapse project syn51105515, which is open to anyone with a free Synapse account: create a personal access token and put it in `~/.synapseConfig` before running stage 01. Every file is pinned by its Synapse id, version, and checksum (`src/sc2bulk/config/pineda_synapse_files.tsv`).

The Sound Life barcode lists are derived from the AIFI Sound Life data (doi:10.57785/bn17-3h16), openly downloadable from the AIFI Human Immune System Explorer with no account needed; no license is stated.

## Reproduce

The run order:

1. The stages that do not read bulk labels (01 to 07: download, preprocessing, splits, panels, searches, training, and embeddings):
   ```
   python pipeline/run_all.py --until-readout
   ```
2. The foundation-model attribution pilots, by hand, in their environments (they size stage 10's runs):
   ```
   python pipeline/10_attribution.py --pair hammond --model scgpt --pilot
   python pipeline/10_attribution.py --pair hammond --model geneformer --pilot
   ```
   The Geneformer gene-deletion step, the most expensive, is skipped unless `SC2BULK_GENEFORMER_GO=1` is set.
3. Give the label-reading stages access to the bulk labels. Stages 10c, 11, and 11b read them only when `outputs/analysis_fixed.json` names an annotated git tag in the checkout's history. Create both:
   ```
   git tag -a analysis-fixed -m "analysis choices fixed"
   echo "{\"tag\": \"analysis-fixed\", \"commit\": \"$(git rev-parse 'analysis-fixed^{commit}')\"}" > outputs/analysis_fixed.json
   ```
   This repository does not include the authors' tag.
4. Everything else:
   ```
   python pipeline/run_all.py --from 08_readout
   ```

Outputs go to `outputs/` (or wherever `SC2BULK_OUTPUTS` points). A rerun resumes: finished work is skipped. `--from <stage>`, `--only <stage>`, and `--pair <pair>` run part of the pipeline; `--dry-run` counts, per stage and pair, the outputs whose recorded code hash differs from the current code (they would rerun).

One stage 06 job cannot be trained: scVI on mouse organs at attribution seed 14 (a final training minibatch of one cell, which scvi-tools' batch normalization rejects). Stages 08 and 10 skip that seed.

When run sequentially, stage 06 stops at that job: `06_train.py --pair tms --scvi` stops before the remaining mouse-organ scVI jobs, and a full `run_all.py` stops there too, before the scVI jobs of the later pairs. Our runs used `scripts/schedule.py`, a GPU scheduler written for the lab's server (its paths, memory caps, and GPU classes are that machine's), which retries the job once, marks it failed, and continues. To run stage 06 by hand instead: run `run_all.py --only 06_train --pair <pair>` for every pair but mouse organs; for mouse organs run `06_train.py --pair tms` and `06_train.py --pair tms --scrat`, then, in the scVI environment, every job that `06_train.py --pair tms --list` reports for model scvi except row J seed 14; then run `run_all.py --from 07_embed`.

## Figures and tables

Once the pipeline has finished, every figure, its panel data, and Additional files 3–6 are drawn into `outputs/figures/`, and the tables into `outputs/tables/`:

```
python figures/make_all.py
python figures/table1.py
python figures/tables.py
```

Each figure script reads only finished stage outputs, and each output's recorded code hash must match the current code (an output made by other code fails as stale). `--only <item>` redraws one item (e.g. `--only "Fig 2"`). The figure and table legends and the hand-written parts of Tables S1 and S12 are read from `figures/text/legends.md` and `figures/text/tables.md`. A Zenodo record of the intermediate outputs will be added with the preprint so the figures can be redrawn without retraining.

## Pipeline stages

`pipeline/run_all.py` runs these in order:

- `01_download`: raw data, checksummed
- `02_preprocess`: QC, normalization, pseudobulk
- `02b_all_available`: the all-available-cells store (Fig 4a's "up to 3,000 cells" point)
- `03_splits`: the folds (donor cross-fitting where donors overlap), the encoder train/validation splits and the read-out CV folds
- `03b_attribution_splits`: encoder splits for the attribution seeds
- `04_panel`: the 5,000-gene panel per fold, from training single-cell data only
- `04b_deconvolve`: the deconvolution reference rows
- `04c_qc_embed`: QC embeddings (Figs. S2–S7)
- `04d_characteristics`: dataset characteristics (Fig 1e, Fig. S8, Table S1)
- `05_search`: the supervised set encoders' configuration searches (equal budgets) and the selected configurations
- `05v_scvi_search`: scVI's small grid
- `06_train`: the supervised set encoders at their selected configurations, and scVI, with their control, learning-curve and recipe rows
- `07_embed`: Geneformer and scGPT embeddings
- `07d_embed_all_available`: foundation-model embeddings of the all-available cells
- `08_readout`: ridge and LASSO read-outs for pseudobulk, the nine learned configurations, and the reference rows
- `08a_alpha_edges`: a check of ridge penalties chosen at the edge of the grid
- `08b_secondary`: the secondary read-outs
- `09_alignment`: ComBat, CORAL and Harmony
- `10_attribution`: gene attribution
- `10c_concordance`: concordance of the gene rankings with the bulk phenotype (Fig 5)
- `11_statistics`: the primary tests and tables
- `11b_secondary`: the secondary statistics
- `12_simulation`: the planted-signal simulation, a positive control
- `12a_properties`: sample properties predictable from each representation (Fig 6a)

Three more scripts run on their own: `00_soundlife_slims` (the Sound Life rebuild, see Data), `05s_eligibility_simulation` (a simulation checking the configuration search's eligibility rule) and `05t_timing` (the timing test that set the training budget).

## Configuration

Every setting is in `src/sc2bulk/config/`: the pairs, pseudobulk and the nine learned configurations, the supervised set encoders' search space and scVI's grid, every pinned number, the smoke run and the figure style. Every output has a manifest beside it recording the code version, settings, seed and machine that produced it. GPU training is not bit-identical across hardware.

## Vendored code

`third_party/cytoset/` is CytoSet (Yi & Stanley, ACM-BCB 2021; https://github.com/CompCy-lab/cytoset) at commit 830f5a2, unchanged. Every change this project makes is in its own code and listed in `third_party/cytoset/CHANGES.txt`.

`third_party/set_transformer/` is the Set Transformer's `modules.py` (Lee et al., ICML 2019; https://github.com/juho-lee/set_transformer) at commit 73432c6, unchanged (MIT license, kept). How this project assembles and trains it is listed in `third_party/set_transformer/CHANGES.txt`.

## Tests

`pytest -q tests` runs the unit tests on synthetic data (`tests/test_figures_end_to_end.py` draws every figure on a synthetic outputs tree and takes longer). `tests/run_level/` holds checks run against a finished run's outputs.

## Use of AI tools

Arushi Sahai used Claude (Claude Opus 5.5, Anthropic), through the Claude Code command-line tool, as an assistant under her direction. It wrote code to her specifications, ran and monitored jobs on the laboratory server, drafted text from her outlines and the analysis outputs, and carried out the audits she set up. Every reported number comes from the outputs of the released code, figures were drawn by that code, and none contains AI-generated imagery. The authors take full responsibility for the work. Her contributions are listed in [AI_USE.md](AI_USE.md).

## License and citation

GPL-2.0 (`LICENSE`), as required by CytoSet's license. Please cite the paper (`CITATION.cff`), as well as CytoSet (Yi and Stanley, ACM BCB 2021), Geneformer (Theodoris et al., Nature 2023), and scGPT (Cui et al., Nature Methods 2024).
