# Hand-written supplementary tables

Read by figures/tables.py: each '## <item>' section, its pipe tables and the notes around them (Tables S1A-S1C, S1E, the titles of S1D and S1F, and S12).

## Table S1

**S1A. Pair-selection criteria (no transfer result was computed before a pair was accepted).**

| # | Criterion | Kind |
|---|---|---|
| 1 | Phenotype: a binary sample-level label, defined identically on both sides (or by a stated mapping), from metadata, not expression | Required |
| 2 | Two sides: single-cell (scRNA-seq or snRNA-seq) and bulk RNA-seq samples carrying the label | Required |
| 3 | Raw counts on both sides: integer UMI or read counts per gene, downloadable (not normalized-only, not microarray) | Required |
| 4 | Size: single-cell ≥ 40 samples, ≥ 15 per group, after QC (≥ 100 cells per sample); bulk ≥ 40 samples, ≥ 15 per group | Required |
| 5 | Gene identifiers: symbols or Ensembl ids, mappable to a shared gene axis | Required |
| 6 | Access: open, or controlled access whose license permits the analysis and release of derived results | Required |
| 7 | Documentation: a published paper or data descriptor describing cohorts, label, and processing | Required |
| 8 | Independent cohorts on the two sides (if donors are shared, a donor split is mandatory) | Preferred |
| 9 | Same tissue or compartment on both sides (otherwise the difference is disclosed) | Preferred |
| 10 | Fills a gap in the grid: non-aging phenotype, new tissue, human, large n | Preferred |
| 11 | Low technical confounding: batch or sequencing run vs. label, Cramér's V < 0.5, where metadata record it | Preferred |
| 12 | Cell-type annotation on the single-cell side | Preferred |
| 13 | Larger single-cell n | Preferred |

Ties on the required criteria are broken by the preferred criteria in the order listed, then by single-cell n. Criteria 1–7 were verified from metadata and file listings before download; the preprocessing asserts them after download.

**S1B. Accepted pairs and criteria-evaluated candidates: phenotype, sides, and donor relationship.**

| Pair | Single-cell side (assay; tissue or preparation) | Bulk side (assay; tissue) | Phenotype, single-cell (0 vs. 1) | Phenotype, bulk (0 vs. 1) | Donor relationship | Decision |
|---|---|---|---|---|---|---|
| Mouse microglia, aging | Hammond et al. 2019 [Hammond 2019] (scRNA-seq; whole-brain microglia, FACS-sorted) | Reemst et al. 2022 [Reemst 2022] (bulk RNA-seq; hippocampal microglia, FACS-sorted) | E14.5–P30 vs. P100–P540 (days) | P9 vs. P200 (days) | Independent studies; no age occurs on both sides | Accepted (pre-criteria; re-evaluated: fails criterion 4 on the bulk side, kept and disclosed) |
| Mouse organs, aging | Tabula Muris Senis [Tabula Muris Consortium 2020], 10x droplet (scRNA-seq; marrow, lung, kidney, limb muscle, dissociated) | Schaum et al. 2020 [Schaum 2020] (bulk RNA-seq; the same four organs, whole) | 1–3 vs. 18–30 months | ≤ 6 vs. 18–30 months (9–15 months excluded) | Same colony, different mice; bulk mice not linkable across organs | Accepted (pre-criteria; re-evaluated) |
| Human blood, aging | Sound Life, Gong et al. 2025 [Gong 2025] (scRNA-seq; PBMC) | Sound Life, Gong et al. 2025 [Gong 2025] (bulk RNA-seq; whole blood, unstimulated) | 25–37 vs. 55–67 years | 25–37 vs. 55–67 years | The same 96 donors; 4-fold donor cross-fitting (no donor's bulk is scored by a model trained on that donor) | Accepted (pre-criteria; re-evaluated) |
| Human blood, lupus | Perez et al. 2022 [Perez 2022], processing cohort 4 (scRNA-seq, 10x 3′ v2; PBMC) | Hung et al. 2015 [Hung 2015] via recount3 (bulk RNA-seq, 50 bp single-end; whole blood) | Healthy control vs. SLE (managed disease; ImmVar controls excluded) | Healthy control vs. SLE (active, trial entry) | Independent cohorts (California vs. a multicenter trial) | Accepted under the criteria |
| Human ileum, Crohn's | IBDverse [Alegbe 2026; Krzak 2026] (scRNA-seq, 10x 3′ v3; terminal ileum, dissociated) | MSCCR, Argmann et al. 2023 [Argmann 2023] (bulk RNA-seq; ileal biopsy) | Non-IBD control vs. Crohn's disease (inflamed or uninflamed; inflamed sample first) | Non-IBD control vs. Crohn's disease (inflamed or uninflamed; inflamed sample first) | Independent cohorts (UK vs. US) | Accepted under the criteria |
| Human motor cortex, ALS | Pineda et al. 2024 [Pineda 2024] (snRNA-seq; motor cortex BA4) | Prudencio et al. 2020 [Prudencio 2020], NYGC ALS Consortium + Target ALS (bulk RNA-seq; motor cortex) | Pathologically normal vs. ALS (sporadic or C9orf72) | Non-neurological control vs. ALS spectrum MND | Independent studies | Accepted (pre-criteria; re-evaluated) |
| Candidate: ulcerative colitis | Smillie et al. 2019 [Smillie 2019], SCP259 (scRNA-seq; colon biopsies) | MSCCR, GSE193677 (bulk RNA-seq; rectal biopsy) | Healthy vs. UC | Non-IBD control vs. UC | Independent cohorts | Rejected: criterion 4 (12 healthy donors, 30 in total; the paired inflamed vs. non-inflamed alternative has 36 samples) |
| Candidate: lupus via GSE112087 | Perez et al. 2022 [Perez 2022] (scRNA-seq; PBMC) | Oon et al. 2019 [Oon 2019] / Figgett et al. 2019 [Figgett 2019], GSE112087 (bulk RNA-seq; whole blood) | Healthy control vs. SLE | Healthy vs. SLE | Independent cohorts (California vs. Australia) | Rejected: criterion 11 (bulk lane × label V = 0.90, not removable by restriction; library names separate the groups, suggesting a different control source) |

Studies, accessions, and counts: Table 1. Pre-criteria pairs were re-evaluated under S1A after acceptance (Methods). The rejected GSE112087 candidate's single-cell side, restricted to processing cohort 4, is the single-cell side of Human blood, lupus.

**S1C. Criteria met per pair.** ● met; ○ not met; — not applicable.

| # | Mouse microglia, aging | Mouse organs, aging | Human blood, aging | Human blood, lupus | Human ileum, Crohn's | Human motor cortex, ALS | UC candidate | Lupus via GSE112087 |
|---|---|---|---|---|---|---|---|---|
| 1–2 | ● | ● | ● | ● | ● | ● | ● | ● |
| 3 | ● | ● | ● | ●^a | ●^b | ●^c | ● (single-cell integer content unverified) | ● |
| 4 | **○** (41; 38: 25 vs. 13): kept as a pre-criteria pair and disclosed | ● (65; 145: 69 vs. 76) | ● | ● (84; 117) | ● (359; 427) | ● (48; 260 samples, 170 donors) | **○** (12 healthy donors) | ● |
| 5 | ● | ● | ● | ● | ● | ● | unverified | ● |
| 6 | ● | ● | ● | ● | ● | ● (Synapse: a free account, no access requirements) | unknown (no SCP license) | ● |
| 7 | ● | ● | ● | ● | ● | ● | ● | ● |
| 8 | ● | ○ (same colony) | ○ (same donors; cross-fitted) | ● | ● | ● | ● | ● |
| 9 | ○ (brain region differs) | ● | ○ (PBMC vs. whole blood) | ○ (PBMC vs. whole blood) | ● | ● | partly (colon vs. rectum) | ○ |
| 10 | — | — | — | ● | ● | — | ● | ● |
| 11 | ○ (chemistry V 0.61) | ○ (release cohort V 0.80) | ○ (processing batch ≈ age group) | ● after the cohort-4 restriction (cohort V 0; bulk flowcell V 0.34) | ● on proxies (single-cell lots ≤ 0.48; bulk ≤ 0.12; flowcells unrecoverable) | ○ (run V 0.96) | unknown | **○** (single-cell cohort V 0.69; bulk lane V 0.90) |
| 12 | ○ (microglia only) | ● | ● | ● | ● | ● | ● | ● |
| 13 | 41 | 65 | 868 | 84 | 359 | 48 | 30 donors | 211 donors |

Criterion 4 entries give single-cell samples; bulk samples (group 0 vs. group 1 where shown). Mouse microglia is the one pair that fails a required criterion: its bulk side has 38 samples, 13 in one group; it was kept as a pre-criteria pair and is disclosed in the Methods. Criterion 10 is not applicable to the four pre-criteria pairs, which define the grid. V values in criterion 11 are from the pre-acceptance checks; the final bias-corrected values over the training samples are in S1D. ^a recount3 coverage sums ÷ 50 bp read length, rounded (recount3's documented transform); per-sample totals asserted to be 0.8–1.0 × SRA spot counts. ^b Bulk samples containing any half-integer count are dropped, never rounded. ^c NYGC RSEM expected counts are rounded to integers (12.4% of nonzero values non-integer; total change 4 × 10⁻⁵).

**S1D. Characteristics of the pairs (Fig 1e and Additional file 1: Fig. S8a).** ● has it; ○ not determinable; blank, does not have it.

| Characteristic | Definition and threshold | Fig 1d | Mouse microglia, aging | Mouse organs, aging | Human blood, aging | Human blood, lupus | Human ileum, Crohn's | Human motor cortex, ALS |
|---|---|---|---|---|---|---|---|---|
| Many cell types | ≥ 3 annotated types, each ≥ 5% of the primary unit's post-QC training cells, at one fixed annotation level per pair; mouse organs counted per organ, "has it" if more than half the organs qualify | Yes | (1 type) | ● (4 of 4 organs; 5–7 types each) | ● (4 types) | ● (5 types) | ● (6 types) | ● (7 types) |
| Tissue or preparation differs between sides | The sides come from different tissues (anatomical region included), or the bulk is whole tissue while the single-cell side is dissociated cells; isolation protocols are not part of the rule | Yes | ● whole brain vs. hippocampus | ● whole organ vs. dissociated cells | ● PBMC vs. whole blood | ● PBMC vs. whole blood | ● biopsy vs. dissociated cells | (same tissue; nuclei, not dissociated cells) |
| Independent cohorts | No donor or animal on both sides | Yes | ● | (same colony) | (same donors, cross-fitted) | ● | ● | ● |
| Batch confounded with label | Bias-corrected Cramér's V [Bergsma 2013] between the single-cell processing batch and the 0/1 group over the primary unit's training samples ≥ 0.5; ○ where no single-cell batch variable is recorded | Yes | ● chemistry, V = 0.59 | ● release cohort, V = 0.78 | ● processing batch, V = 0.57^a | pool (connected component of the donor–library graph), V = 0.00 | ○ no batch variable; reagent lots are proxies | ● sequencing run, V = 0.92 |
| Few training donors | < 100 single-cell training donors (mice for the mouse pairs) | Yes | ● (41) | ● (22 mice; 65 samples) | ● (72 per fold) | ● (84) | (359) | ● (48) |
| Single-nucleus → bulk | Single-cell side is snRNA-seq (nuclear vs. whole-cell RNA) | No | | | | | | ● |
| Non-age phenotype | The label is a disease status | No | | | | ● SLE | ● Crohn's | ● ALS |
| Several tissues | The pair spans more than one tissue | No | | ● 4 organs | | | | |
| No shared label values | No label value occurs on both sides | No | ● | | | | | |

Values in parentheses: training donors (Few training donors); annotated types at ≥ 5% (Many cell types; mouse organs: kidney 6, limb muscle 5, lung 7, marrow 7); V, bias-corrected Cramér's V over the primary unit's training samples (Batch confounded with label). ^a Sample level; with each donor's modal batch, V = 0.49. Human blood, lupus has pool V = 0 after the processing-cohort-4 restriction. Values per pair, cell-type compositions, and batch × group counts: Additional file 3.

**S1E. Quality control and selection per pair.**

| Pair | Single-cell: cell QC | Single-cell: sample selection and exclusions | Bulk: selection and exclusions |
|---|---|---|---|
| Mouse microglia, aging | < 200 genes or ≥ 20% mitochondrial dropped | P100 LPC-demyelination and saline (injury) samples excluded (47 → 41); P5 Percoll pairs (GSM3442047–52) share a holdout group; GSM3442038–40 from the injury batch kept, disclosed | LPS-challenged animals excluded (38: 25 P9, 13 P200); litter recorded |
| Mouse organs, aging | Quality-controlled by the provider (≥ 229 genes per cell; no mitochondrial genes, so the ceiling is inert); < 200 genes dropped | Four organs: marrow, lung, kidney, limb muscle (65 samples) | 9–15-month mice excluded (145) |
| Human blood, aging | Quality-controlled by the provider; cells drawn from 3000-cell pre-QC subsamples, all of which pass < 200 genes / ≥ 20% mitochondrial | All 96 donors, every visit (868 samples); scored per donor | Unstimulated whole blood; kits sequenced twice have their two libraries summed (836) |
| Human blood, lupus | < 200 genes or ≥ 20% mitochondrial dropped | ImmVar controls excluded; processing-cohort-4 cells only; per donor, the sample with the most cohort-4 cells (84: 44 control, 40 SLE; smallest 454 post-QC cells) | Read-equivalent counts from recount3 (S1C ^a); one sample per donor (117: 18 control, 99 SLE); sex and age not recorded |
| Human ileum, Crohn's | < 200 genes or **≥ 50% mitochondrial** dropped (the provider's ileum ceiling; 20% removed 39% of epithelial cells and 0% of immune and stromal cells; label-blind, chosen after seeing pass rates) | Terminal ileum; rectum and blood excluded; one sample per donor, inflamed first; two samples with fewer than 100 cells, excluded by identifier (17 and 14 cells, one per group) (359: 242 control, 117 Crohn's) | Ileum; UC excluded; samples with half-integer counts dropped (51 in ileum; 12% of Crohn's vs. 7% of controls); one per donor, inflamed first (427: 113 control, 314 Crohn's) |
| Human motor cortex, ALS | Quality-controlled by the provider (≥ 651 genes, ≤ 19.9% mitochondrial; our QC removes none) | BA4, one library per donor; FTLD and BA9 excluded; donor 102 excluded as a failed library (195 nuclei, 55% vascular), decided before any result (48: 16 control, 32 ALS) | Motor cortex (lateral, medial, unspecified); other diagnoses excluded; 7 samples without counts (260 samples, 170 donors); 9.2% of counts fall outside the shared annotation |

Every pair: genes keyed by symbol, duplicate symbols summed; no doublet removal by us; ≥ 100 post-QC cells per sample, checked to remove no sample (except Crohn's two samples with fewer than 100 cells, excluded by identifier); at most 1000 post-QC cells stored per sample (seeded), pseudobulk is their sum; bulk libraries with < 1 million counts dropped. Counts per step: S1F. Batch × group counts, confounder summaries, and provider full cell counts: Additional file 3. QC panels: Additional file 1: Figs. S2–S7.

**S1F. Samples and cells removed at each selection and QC step.**

| Pair | Single-cell samples: listed → kept (exclusions) | Cells before QC | Removed: < 200 genes | Removed: mitochondrial ceiling | Cells after QC | Smallest sample after QC | Cells stored | Bulk samples: listed → kept (exclusions) |
|---|---|---|---|---|---|---|---|---|
| Mouse microglia, aging | 47 → 41 (6 injury experiment) | 108532 | 155 | 12 | 108365 | 900 | 40900 | 50 → 38 (12 LPS-challenged) |
| Mouse organs, aging | 65 → 65 | 115274 | 0 | 0 | 115274 | 331 | 61083 | 933 → 145 (715 other organs; 70 aged 9–15 months; 3 below 1 million counts) |
| Human blood, aging | 868 → 868 | 2586999^a | 0 | 0 | 2586999 | 984 | 867984 | 836 → 836^b |
| Human blood, lupus | 274 → 84 (50 ImmVar controls; 139 whose donor's majority processing cohort is not 4; 1 donor's second sample) | 336887^c | 214 | 274 | 336423 | 454 | 83454 | 117 → 117 |
| Human ileum, Crohn's | 361 → 359 (2 samples below 100 cells, excluded by identifier) | 913538 | 0 | 1 | 913537 | 115 | 342466 | 672 → 427 (188 ulcerative colitis; 51 non-integer counts; 6 a donor's second ileal biopsy) |
| Human motor cortex, ALS | 49 → 48 (1 failed library) | 191922 | 0 | 0 | 191922 | 673 | 47480 | 267 → 260 (7 without counts) |

Cell counts are summed over kept samples. A cell failing both QC thresholds is counted in both columns (24 cells, all in human blood, lupus). Cells stored: at most 1000 per sample, drawn with a fixed seed. ^a The provider's 3000-cell pre-QC subsamples (all cells where a sample had fewer). ^b The two libraries of each kit sequenced twice are summed. ^c Processing-cohort-4 cells only.

## Table S12

| Method | What it produces | Embeds a bulk profile? | Public code? | Trainable at our sizes? | Decision and reason |
|---|---|---|---|---|---|
| CytoSet [Yi 2021] | Supervised set encoder: a sample embedding from a bag of cells (permutation-equivariant layers + pooling) | Yes, as a one-element set | Yes (CompCy-lab/cytoset, GPL-2.0) | Yes | **Included** (supervised set encoders; cells and pseudobulk input) |
| Set Transformer [Lee 2019] | Supervised set encoder: cell–cell self-attention + attention pooling | Yes, as a one-element set | Yes (MIT) | Yes | **Included** (supervised set encoders; cells) |
| ScRAT [Mao 2024] | Supervised set encoder: cell–cell attention for sample phenotype prediction | Yes, as a one-element set | Yes (GitHub; no license file, so fetched at a pinned commit, never redistributed) | Yes | **Included** (supervised set encoders; cells) |
| scVI [Lopez 2018] | Unsupervised generative model: a per-cell latent, averaged per sample | Yes: the profile is thinned to the median training-cell library size and encoded (10 draws, averaged) | Yes (scvi-tools) | Yes | **Included** (unsupervised generative model; cells) |
| Geneformer [Theodoris 2023; Chen 2026] | Frozen foundation model: a per-cell embedding, averaged per sample | Yes, the profile is embedded directly | Yes (Hugging Face) | No training needed | **Included** (frozen foundation models; cells and pseudobulk input) |
| scGPT [Cui 2024] | Frozen foundation model: a per-cell embedding, averaged per sample | Yes, the profile is embedded directly | Yes | No training needed | **Included** (frozen foundation models; cells and pseudobulk input) |
| MrVI [Boyeau 2025] | Unsupervised generative model with sample-level latent effects; sample representations and distances for the training samples | No: each training sample has a learned embedding, so a new sample needs its own cells and a refit | Yes (scvi-tools) | Yes | Excluded: no bulk embedding |
| scPoli sample embeddings [De Donno 2023] | Learned condition (sample) embeddings in a conditional autoencoder | No: a new sample's embedding is fit to its cells by reference mapping | Yes (scArches) | Yes | Excluded: no bulk embedding |
| PILOT [Joodaki 2024] | Sample-by-sample Wasserstein distances between samples' cluster or cell-type proportions | No: needs a sample's cells | Yes | Yes (unsupervised) | Excluded: no bulk embedding (distances, not an embedding) |
| GloScope [Wang 2024] | Sample-by-sample divergences between per-sample densities of cell embeddings | No: needs a sample's cells | Yes (R package) | Yes (unsupervised) | Excluded: no bulk embedding (distances, not an embedding) |
| CloudPred [He 2022] | Supervised: a sample as its cells' mean Gaussian-mixture responsibilities, then a polynomial classifier | Degenerate: a one-element set gives one point's responsibilities, close to a single component | Yes (GitHub; no license file) | Yes (published on 142–169 samples) | Excluded: no usable bulk embedding |
| ProtoCell4P [Xiong 2023] | Supervised: each cell's similarity to learned prototypes, weighted by a relevance scorer into per-cell class scores that are summed per sample | No sample embedding: the only sample-level quantity is the class score itself | Yes (MIT) | Yes (published on 42–169 samples) | Excluded: no sample embedding for the shared readout^a |
| PaSCient [Liu 2026] | Supervised: attention-pooled patient embedding; pretrained on 24.3 million human cells | Yes, as a one-element set | Yes (Apache-2.0); pretrained weights on request | From scratch, yes | Excluded: from scratch, attention pooling over a linear cell encoder, covered by CytoSet's attention-pooling candidate; pretrained, it requires 28,231 human genes, so it fits neither the shared panel nor the mouse pairs. Neither reason is among the three criteria^b |
| Fine-tuned foundation models [Theodoris 2023; Cui 2024] | Geneformer or scGPT with weights updated on the phenotype | Yes | Yes | No: 10⁷–10⁸ weights (Geneformer V2: 1.04 × 10⁸) against 41–868 sample-level labels | Excluded: not trainable at our sizes^a; the frozen models are included |
