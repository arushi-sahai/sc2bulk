# Figure and table titles and legends

Read by figures/common.py (each main figure's legend height) and figures/tables.py (Tables S1-S12's legends and footnotes): an item's bold title, then its legend lines.

---

## Main figures

**Fig 1. Six single-cell-to-bulk dataset pairs and nine learned configurations compared with pseudobulk for phenotype transfer.**

(a) A ridge readout on a sample representation predicts a binary phenotype. It is trained on single-cell RNA sequencing (RNA-seq) samples, applied unchanged to bulk RNA-seq samples, and scored by the area under the receiver operating characteristic curve (AUROC) between the two bulk groups.
(b) Pseudobulk sums the cells' counts, and the supervised set encoders (CytoSet, Set Transformer, and ScRAT) are trained on the single-cell labels. The unsupervised generative model (scVI) and the frozen foundation models (Geneformer and scGPT) embed each cell without phenotype labels, and the cell embeddings are averaged.
(c) Each model embeds bags of 500 cells, and CytoSet, Geneformer, and scGPT also embed the summed (pseudobulk) profile (hatched), giving nine learned configurations.
(d) The dataset pairs, with each side's tissue or preparation, assay, and compared groups. The connector marks the same donors (solid), the same colony (dotted), or independent cohorts (dashed). Studies, accessions, and counts: Table 1. Quality control: Additional file 1: Figs. S2–S7.
(e) Characteristics that differ between pairs (●: has it; ○: not determinable): many cell types (≥ 3 types, each ≥ 5% of training cells); tissue or preparation differs between sides; independent cohorts; batch confounded with label on the single-cell side (Cramér's V ≥ 0.5); and fewer than 100 training donors (mice, in the mouse pairs). All characteristics: Additional file 1: Fig. S8.

Source data: Additional file 3.

**Fig 2. Under the main readout, no configuration outperformed pseudobulk, with every pooled upper bound below 0.05.**

(a) Difference in bulk area under the receiver operating characteristic curve (AUROC) from pseudobulk, per pair and pooled with equal weights over the ♦ pairs. Ridge readout trained on every labeled training sample. Top row: pseudobulk's bulk AUROC (bar from 0.5, chance), and only pairs below the gray dashed line at 0.95 (♦), which allow a 0.05 advantage, are pooled. Whiskers: two-sided 95% bootstrap confidence intervals (unadjusted). The one-sided upper bound u lies below each whisker's upper end. Arrowheads: values beyond the axis. Shading: beyond the 0.05 margin. Dash-dot: within-bulk reference. Gray ✕, below the line: reference and control rows (Pseudobulk, PCA-64; mean of log-normalized cells; per-cell classifier; untrained CytoSet; deconvolution; label-permuted pseudobulk; and known-signature score). ↑ and ↓: significantly better or worse than pseudobulk (one-sided, Holm-adjusted over the nine configurations, per pair and pooled). Right column: the pooled difference under the out-of-fold readout, trained only on samples the encoders did not train on (about 30%). Ticks on CytoSet rows: CytoSet under the out-of-fold readout against pseudobulk under the main readout.
(b) Within-single-cell AUROC (left, on held-out single-cell samples, per fold for human blood, aging) and bulk AUROC (right). Dashed blue: pseudobulk's bulk AUROC. Dash-dot as in a. Axis: 0.5 (chance) to 1, with arrowheads for means below it.
(c) The difference from pseudobulk decomposed, per pair and pooled: from 0 to the model on the pseudobulk profile (open marker), then to the model on cells (filled). Set Transformer, ScRAT, and scVI use cells only and are not decomposed. No intervals in c.

Points: means over 10 encoder seeds × 10 readout seeds. Faint dots: encoder seeds, or readout seeds for the frozen foundation models (label-permuted row: 20 permutations). Bootstrap: B = 2,000, resampling bulk scoring units and encoder seeds. Scoring units per pair: Table 1. Values, including the out-of-fold readout: Additional file 2: Table S2. Source data: Additional file 3.

**Fig 3. Per-assay scaling removed most of the single-cell–bulk gap; label-free alignment did not reduce it further.**

(a) Single-cell (yellow) and bulk (purple) samples of human ileum, Crohn's in the first two principal components, for the gene profiles and "CytoSet, cells" embeddings (one encoder seed). For display only, each is shown with one scaler fit over both sides together and after ComBat. Filled markers are Crohn's samples and open markers controls. Panel titles give each representation's modality silhouette.
(b) Modality silhouette (silhouette of the side label on the top 10 principal components; 0: sides mixed; 1: sides separate) per pair, for pseudobulk and each configuration on cells (identified by marker). Each is shown under four conditions: per-side (no alignment), ComBat, correlation alignment (CORAL), and Harmony. Trained models give the median over 10 encoder seeds or scVI fits, pseudobulk and the frozen foundation models give one value, and human blood, aging also takes the median over folds.
(c) Bulk area under the receiver operating characteristic curve (AUROC) under the same conditions, mean over seeds as in Fig 2a, axis as in Fig 2b. Dashed blue: pseudobulk with no alignment.

In b and c, "per-side (no alignment)" is the per-side standardization of Fig 2, and no intervals are drawn. Alignment methods are fit without labels on the single-cell samples and the scored bulk samples. Other pairs and the pseudobulk-input configurations: Additional file 1: Fig. S13. Values: Additional file 2: Table S7. Source data: Additional file 3.

**Fig 4. No design choice of the supervised set encoders significantly outperformed pseudobulk in any pair.**

Difference in bulk area under the receiver operating characteristic curve (AUROC) from full pseudobulk, per dataset pair, as means over 5 seeds (faint dots in b and c: seeds), with no intervals drawn. Rings mark the configuration selected on single-cell validation loss, which may combine changes (in b and c, after the dotted divider). Axis and shading (beyond the 0.05 margin) are as in Fig 2a, with arrowheads for values beyond the axis (seed dots omitted).
(a) Cells per bag at each supervised set encoder's selected configuration. "All" is every available cell after quality control (≤ 3,000), whereas full pseudobulk sums the ≤ 1,000 stored cells. Rings sit at 500, the training bag size, and open marks at the published sizes (CytoSet 200, ScRAT 500, and Set Transformer 1,000). Under each pair's name: the best-candidate test's p, unadjusted across pairs (Holm-adjusted values: Additional file 2: Table S8).
(b) Output-level pooling, one change from each encoder's base configuration (its published defaults): CytoSet's options, including attention pooling added in this study, Set Transformer's published pooling by multihead attention (PMA), and ScRAT's published mean pooling. scVI and the frozen foundation models are shown with mean or max pooling, as labeled.
(c) Input normalization, one change from the base configuration: none (the original CytoSet), standardization against the training population, or standardization within each sample (Methods).

The best-candidate test is a one-sided max-statistic bootstrap per pair, conservative in simulation, and the pooled comparison is Fig 2a. Source data: Additional file 3. All candidates: Additional file 5.

**Fig 5. Single-cell–bulk gene-shift agreement varied by pair; supervised set encoders detected planted structure, not mean shifts.**

(a) Each panel gene's phenotype shift (group-mean difference divided by the within-group standard deviation) in single-cell pseudobulk against bulk, per pair, with Spearman's ρ under each title. Blue: pseudobulk's 50 top-ranked genes. Triangles: genes of the published signature behind the known-signature score (Methods), pointing in the signature's direction.
(b) Same-sign share: the share of each configuration's 50 top-ranked genes whose bulk shift has the same sign as their single-cell shift. References: all panel genes (vertical line) and genes matched on single-cell shift size (triangle under each point). Genes are ranked by attribution of the ridge readout, averaged over seeds (Methods), and for Geneformer by gene deletion (Additional file 1: Fig. S14d).
(c) Same-sign share against bulk area under the receiver operating characteristic curve (AUROC), one point per configuration, shown without a statistical test.
(d) A positive-control simulation in human ileum, Crohn's (Methods). For the encoders' training on the real pairs, see the training check (Additional file 1: Fig. S10d) and the within-single-cell AUROCs (Fig 2b). A random phenotype is planted in 50 genes, either as a shift in mean expression or as structure only, which moves counts between a sample's cells and leaves every per-gene sum unchanged. Points give the AUROC within single-cell data and on simulated bulk (each sample's cells summed) for pseudobulk, the mean of log-normalized cells (✕), and each supervised set encoder, as means over 5 splits (dots), with no intervals drawn. Gray marks: values that hold by construction.

Values: Additional file 2: Table S9. Gene lists: Additional file 4. Source data: Additional file 3.

**Fig 6. Frozen foundation-model cell embeddings encoded depth and other technical properties beyond pseudobulk in every pair.**

(a) Cross-validated R² for predicting each sample property (columns: sequencing depth, mitochondrial fraction, cell-type composition, and phenotype) from each representation (rows), per pair, on held-out single-cell samples over the 10 splits of Fig 2b, with no intervals drawn. The phenotype column shows the area under the receiver operating characteristic curve (AUROC) on a 0.5–1 scale, and negative R² is drawn as 0. Reference rows (above the line): pseudobulk and Pseudobulk, PCA-64 (up to 64 principal components). The gene panel holds mitochondrial genes (5–13 mitochondrial genes; none in mouse organs). Bottom row: chance, the largest permutation 95th percentile. n/a: mitochondrial fraction for mouse organs and composition for mouse microglia.
(b) Modality-direction genes have the largest attribution to the direction from the single-cell to the bulk mean embedding. Left: the 8 top such genes of "CytoSet, cells" on human ileum, Crohn's, annotated with their bulk-minus-single-cell-pseudobulk expression differences. Right: for each configuration on cells, the share of its top modality-direction genes that are large bulk-vs-single-cell difference genes or cell-type markers (vertical line: all panel genes; n/a: Geneformer's checks failed, Additional file 1: Fig. S14d). Other configurations: Additional file 1: Fig. S15.
(c) Within-single-cell and bulk AUROC as in Fig 2b (configurations dodged), with every configuration restricted to one abundant cell type per pair, named per column. Samples need ≥ 50 cells of the type, and the supervised set encoders are retrained with 5 seeds. Bulk inputs are not restricted and contain all cell types.

Values: Additional file 2: Table S10. Source data: Additional file 3.

---

## Additional file 1: supplementary figures

One PDF, Figs. S1–S16 in order, each with its title and legend.

**Fig. S1. Supervised set-encoder architectures and the embedding paths of the unsupervised generative and frozen foundation models.**
(a–c) Each supervised set encoder maps a bag of cells (500 cells × panel genes) to one embedding for the ridge readout, with a classification head used only in training. Each is drawn at its base configuration, its authors' published defaults, and elements varied by the configuration search are dashed. Hyperparameter values, search options, and per-pair selections: Additional file 2: Table S4.
(a) CytoSet: permutation-equivariant blocks, with pooling within each block and mean pooling over cells.
(b) Set Transformer: induced set attention blocks (ISAB) and pooling by multihead attention (PMA).
(c) ScRAT: a transformer encoder over cells, on principal components of the panel genes (principal component analysis, PCA).
(d) How the unsupervised generative model (scVI) and the frozen foundation models embed a cell, or a bulk profile treated as one cell. scVI, trained here without labels, encodes counts per gene as a latent mean. Geneformer reads genes ranked by expression and gives its <cls> output, and scGPT reads gene tokens with binned expression values and gives its normalized <cls> output. A sample's embedding is the mean of its cells' embeddings.

Source data: Additional file 3.

**Fig. S2. Quality control: mouse microglia, aging.**
Single-cell side (a–d): whole-brain microglia single-cell RNA sequencing of Hammond et al. [Hammond 2019]. Bulk side (e–h): hippocampal microglia bulk RNA sequencing of Reemst et al. [Reemst 2022].
(a) Each single-cell sample's median genes detected, counts, and mitochondrial percentage per cell, by group, with the per-cell quality-control (QC) thresholds (dotted lines at ≥ 200 genes and < 20% mitochondrial).
(b) Cells per sample after QC, by group.
(c) Uniform manifold approximation and projection (UMAP) of the single-cell side, colored by annotated cell type (rare types grouped as "other").
(d) The same UMAP, one plot per phenotype group (younger, older): that group's cells in yellow, the other group's in gray.
(e) Bulk samples per group.
(f) Bulk library size (millions) and genes detected (thousands).
(g) Principal component analysis (PCA) of the bulk samples by group.
(h) Bulk samples by early-life-stress group and age group (the stress condition is used only here and in Fig. S11e).

Younger: E14.5–P30 (single-cell), P9 (bulk); older: P100–P540, P200. Lipopolysaccharide-challenged animals are excluded from the bulk side. Open markers and bars: younger. Filled: older. Counts per step: Additional file 2: Table S1. Source data: Additional file 3.

**Fig. S3. Quality control: mouse organs, aging.**
Single-cell side (a–d): 10x droplet single-cell RNA sequencing of four organs from Tabula Muris Senis [Tabula Muris Consortium 2020]. Bulk side (e–g): whole-organ bulk RNA sequencing of the same colony [Schaum 2020]. Panels h and i show both sides.
(a) Each single-cell sample's median genes detected, counts, and mitochondrial percentage per cell, by group, with the per-cell quality-control (QC) thresholds (dotted). The provider's matrix holds no mitochondrial genes, so the mitochondrial threshold removes no cells.
(b) Cells per sample after QC, by group.
(c) Uniform manifold approximation and projection (UMAP) of the single-cell side, colored by annotated cell type (rare types grouped as "other").
(d) The same UMAP, one plot per phenotype group (younger, older): that group's cells in yellow, the other group's in gray.
(e) Bulk samples per group.
(f) Bulk library size (millions) and genes detected (thousands).
(g) Principal component analysis (PCA) of the bulk samples by group.
(h) Samples per organ (marrow, lung, kidney, limb muscle) on each side, by group. (i) Single-cell mice by the number of organs they contribute, by group.

Younger mice are at most 6 months old and older mice at least 18 months. Open markers and bars: younger. Filled: older. Counts per step: Additional file 2: Table S1. Source data: Additional file 3.

**Fig. S4. Quality control: human blood, aging.**
Single-cell side (a–d): peripheral blood mononuclear cell single-cell RNA sequencing. Bulk side (e–g): whole-blood bulk RNA sequencing, from the same donors of the Sound Life cohort [Gong 2025; Smithmyer 2025]. Panels h and i show both sides.
(a) Each single-cell sample's median genes detected, counts, and mitochondrial percentage per cell, by group, with the per-cell quality-control (QC) thresholds (dotted), computed on the provider's 3,000-cell subsamples before QC.
(b) Cells per sample after QC, by group.
(c) Uniform manifold approximation and projection (UMAP) of the single-cell side, colored by annotated cell type (rare types grouped as "other").
(d) The same UMAP, one plot per phenotype group (younger, older): that group's cells in yellow, the other group's in gray.
(e) Bulk samples per group.
(f) Bulk library size (millions) and genes detected (thousands).
(g) Principal component analysis (PCA) of the bulk samples by group.
(h) Scored donors per cross-fitting fold, by age group (four donor folds). (i) Cytomegalovirus (CMV) serostatus by age group.

Younger donors are 25–37 years old and older donors 55–67. Open markers and bars: younger. Filled: older. Counts per step: Additional file 2: Table S1. Source data: Additional file 3.

**Fig. S5. Quality control: human blood, lupus.**
Single-cell side (a–d, h): peripheral blood mononuclear cell single-cell RNA sequencing of Perez et al. [Perez 2022]. Bulk side (e–g): whole-blood bulk RNA sequencing of Hung et al. [Hung 2015].
(a) Each single-cell sample's median genes detected, counts, and mitochondrial percentage per cell, by group, with the per-cell quality-control (QC) thresholds (dotted lines at ≥ 200 genes and < 20% mitochondrial).
(b) Cells per sample after QC, by group.
(c) Uniform manifold approximation and projection (UMAP) of the single-cell side, colored by annotated cell type (rare types grouped as "other").
(d) The same UMAP, one plot per phenotype group (control, SLE): that group's cells in yellow, the other group's in gray.
(e) Bulk samples per group.
(f) Bulk library size (millions) and genes detected (thousands), as recount3 read-equivalent counts.
(g) Principal component analysis (PCA) of the bulk samples by group.
(h) Single-cell samples kept and dropped by the restriction to processing cohort 4 (or dropped for other reasons), by disease status.

SLE: systemic lupus erythematosus. The bulk flowcell is not recorded in any processed output. Open markers and bars: control. Filled: SLE. Counts per step: Additional file 2: Table S1. Source data: Additional file 3.

**Fig. S6. Quality control: human ileum, Crohn's disease.**
Single-cell side (a–d, h): terminal-ileum single-cell RNA sequencing of IBDverse [Alegbe 2026; Krzak 2026]. Bulk side (e–g, i–j): ileal-biopsy bulk RNA sequencing of the Mount Sinai Crohn's and Colitis Registry [Argmann 2023].
(a) Each single-cell sample's median genes detected, counts, and mitochondrial percentage per cell, by group, with the per-cell quality-control (QC) thresholds (dotted lines at ≥ 200 genes and < 50% mitochondrial, the provider's ileum ceiling).
(b) Cells per sample after QC, by group.
(c) Uniform manifold approximation and projection (UMAP) of the single-cell side, colored by annotated cell type (rare types grouped as "other").
(d) The same UMAP, one plot per phenotype group (control, Crohn's): that group's cells in yellow, the other group's in gray.
(e) Bulk samples per group.
(f) Bulk library size (millions) and genes detected (thousands).
(g) Principal component analysis (PCA) of the bulk samples by group.
(h) Single-cell samples per reagent-lot level (bead, chip, gel-bead-in-emulsion, and enzyme lots), by group, then counts by chip and bead version. (i) Donor age by group (bulk). (j) Inflamed and uninflamed bulk biopsies, both counted as Crohn's.

Controls have no inflammatory bowel disease. Bulk flowcells are not recoverable from public data (Methods). Open markers and bars: control. Filled: Crohn's. Counts per step: Additional file 2: Table S1. Source data: Additional file 3.

**Fig. S7. Quality control: human motor cortex, ALS.**
Single-cell side (a–d, h): motor cortex single-nucleus RNA sequencing of Pineda et al. [Pineda 2024], with nuclei counted as cells. Bulk side (e–g): motor cortex bulk RNA sequencing of the New York Genome Center ALS Consortium and Target ALS [Prudencio 2020].
(a) Each single-cell sample's median genes detected, counts, and mitochondrial percentage per cell, by group, with the per-cell quality-control (QC) thresholds (dotted). The provider had already filtered the nuclei, so our QC removes none.
(b) Cells per sample after QC, by group.
(c) Uniform manifold approximation and projection (UMAP) of the single-cell side, colored by annotated cell type (rare types grouped as "other").
(d) The same UMAP, one plot per phenotype group (control, ALS): that group's cells in yellow, the other group's in gray.
(e) Bulk samples per group.
(f) Bulk library size (millions) and genes detected (thousands).
(g) Principal component analysis (PCA) of the bulk samples by group.
(h) Potential confounders on the single-nucleus side, by group: sex, age at death, postmortem interval, and sequencing run.

ALS: amyotrophic lateral sclerosis. Open markers and bars: control. Filled: ALS. Counts per step: Additional file 2: Table S1. Source data: Additional file 3.

**Fig. S8. Characteristics of the dataset pairs.**
(a) Every characteristic per pair (●: has it; ○: not determinable; blank: does not have it): the five of Fig 1e and four more (single-nucleus to bulk, non-age phenotype, several tissues, and no shared label values).
(b) Cell-type composition per pair, as stacked bars colored by cell type: the single-cell training samples' annotated fractions and the scored bulk's fractions estimated by deconvolution (one color per type name across pairs; rare types grouped as "other"; human blood, aging: mean over its four folds). Mouse microglia has one cell type, so its bulk has no deconvolution (n/a).
(c) Tissue-difference markers, as mean log(1 + counts per million) on each side: neutrophil, erythrocyte, and platelet genes (blood); stromal and muscularis genes (ileum); and myofiber and lung-epithelium genes (mouse organs). Mouse microglia has none, because its cells are microglia on both sides.
(d) Single-cell training samples per batch level, by group, for the pairs with a recorded batch variable.
(e) Samples per side and group (younger and older, or control and case), per pair.
(f) Samples per organ and side (mouse organs).

Characteristic and phenotype definitions, thresholds, donor relationships, and the single-nucleus vs. bulk comparison: Additional file 2: Table S1. Source data: Additional file 3.

**Fig. S9. Transfer under alternative readouts, as differences from pseudobulk.**
Difference in bulk area under the receiver operating characteristic curve (AUROC) from pseudobulk under the same readout, per pair, in Fig 2a's format: means with two-sided 95% bootstrap confidence intervals (unadjusted), dashed blue at 0, and shading beyond +0.05. Arrowheads at the axis edge mark values beyond it. ♦ marks a pair in the pooled average (pseudobulk bulk AUROC below 0.95), and the last column, "Pooled (♦ pairs)", weights these pairs equally (n/a in a). (a) Least absolute shrinkage and selection operator (LASSO), reported per pair only. (b) Elastic net. (c) Partial least squares. (d) k-nearest neighbors. (e) Multilayer perceptron. (f) CytoSet's own classification head ("CytoSet, cells" only). Values: Additional file 2: Table S3. Source data: Additional file 3.

**Fig. S10. Configuration searches and training checks for the three supervised set encoders.**
Columns are pairs. For human blood, aging, folds are shown side by side in a, d, and e, and combined in b.
(a) Single-cell validation binary cross-entropy (BCE) of every search candidate of CytoSet, Set Transformer, and ScRAT (mean over 5 search seeds), on a log scale (one candidate diverged). Dotted: chance (ln 2, a constant prediction of 0.5).
(b) Bulk area under the receiver operating characteristic curve (AUROC) against validation BCE for every candidate and search seed (faint dots). Dotted: chance BCE. "Best on bulk" marks the candidate best on bulk, for reference only and never selected.
(c) Outside the search: each encoder's own published training procedure, and CytoSet trained on age with a mean-squared-error (MSE) objective (aging pairs, 5 search seeds). Difference in bulk AUROC from pseudobulk with its two-sided 95% confidence interval (unadjusted), per pair and pooled over the ♦ pairs (not formed for the MSE objective), on Fig 2a's axis.
(d) Training checks per encoder seed of the selected configuration: the loss improvement over a constant prediction, with the pass threshold dotted. Seeds failing the check are gray.
(e) scVI's search: validation evidence lower bound (ELBO) of each candidate relative to the base candidate (Δ ELBO, mean over search seeds).

Values and budgets: Additional file 2: Table S4. All candidates: Additional file 5. Source data: Additional file 3.

**Fig. S11. Robustness of the comparison with pseudobulk across seeds, normalizations, panel sizes, and subgroups.**
(a) Each bulk sample's predicted score for every configuration (ridge readout, mean over encoder and readout seeds), with control or younger samples left and case or older samples right. Per-sample values: Additional file 6.
(b) Bulk area under the receiver operating characteristic curve (AUROC) per encoder seed (faint dots) and their mean (marker) for each trained configuration, with the seed-ensemble AUROC of predictions averaged over encoder seeds (short horizontal bar).
(c) Pseudobulk normalized by trimmed mean of M-values (TMM) or a variance-stabilizing transformation (VST), as differences from library-size-normalized pseudobulk with two-sided 95% confidence intervals, per pair and pooled over the ♦ pairs.
(d) Pseudobulk on 1,000- and 2,000-gene panels and on every shared gene, as in c.
(e) Difference in bulk AUROC from pseudobulk within the bulk subgroups of each pair (Additional file 2: Table S5). CMV: cytomegalovirus. NYGC: New York Genome Center.
(f) Left: bulk AUROC differences in human ileum, Crohn's with the score adjusted for donor age. Right: within-single-cell AUROC in human motor cortex, ALS before and after adjustment for age at death.

In c–e and f (left), the axis is that of Fig 2a, with arrowheads at its edge for values beyond it. All intervals are unadjusted for multiple comparisons. Values: Additional file 2: Table S5. Source data: Additional file 3.

**Fig. S12. Full gene sets, native mouse genes, and more training donors, compared with pseudobulk.**
Difference in bulk area under the receiver operating characteristic curve (AUROC) from pseudobulk as in Fig 2a, per pair, with arrowheads at the axis edge for values beyond it.
(a) Geneformer and scGPT on their full gene sets, and pseudobulk restricted to the panel genes with one-to-one human orthologs (mouse pairs only, n/a for human pairs), with two-sided 95% confidence intervals (unadjusted), per pair and pooled over the ♦ pairs.
(b) Learning curves: each configuration's difference at 25, 50, and 100% of training donors (5 seeds), for the pairs with enough training donors to subsample (human blood, aging; human blood, lupus; and human ileum, Crohn's).

Values: Additional file 2: Table S6. Source data: Additional file 3.

**Fig. S13. The modality gap in every dataset pair.**
(a–e) Fig 3a for the other pairs, in pair order: (a) mouse microglia, aging; (b) mouse organs, aging; (c) human blood, aging (fold 0); (d) human blood, lupus; (e) human motor cortex, ALS. Each shows single-cell (yellow) and bulk (purple) samples in the first two principal components, with one scaler for both sides for display (left) and after ComBat (right), for the gene profiles (top) and "CytoSet, cells" embeddings (bottom). Filled markers show case or older samples and open markers control or younger samples, with the modality silhouette below each plot.
(f) Fig 3b (top) and 3c (bottom) for pseudobulk and the pseudobulk-input configurations ("CytoSet, pseudobulk", "Geneformer, pseudobulk", and "scGPT, pseudobulk"), under per-side (no alignment), ComBat, correlation alignment (CORAL), Harmony, and one scaler for both sides (light). The bottom row shows bulk area under the receiver operating characteristic curve (AUROC), axis as in Fig 2b.

Values: Additional file 2: Table S7. Source data: Additional file 3.

**Fig. S14. Robustness of the gene lists behind Fig 5.**
(a) Stability of each configuration's top 50 genes (overlap, 0–1) across seeds, seeds 0–9 vs. 10–19 (nine attribution seeds for scVI on mouse organs, aging), donor halves, and, for human blood, aging, folds. Each row has four marks, in that order, for the ridge readout of the phenotype.
(b) The same-sign share of Fig 5b with the top 25 (top) and top 100 (bottom) genes.
(c) The same share with the top 50 genes of the least absolute shrinkage and selection operator (LASSO).
In b and c, the references are all panel genes (light vertical line) and genes matched on single-cell shift size (open gray triangle under each point).
(d) Attribution checks per pair and configuration: the number of the four inclusion checks passed or failed (Geneformer's deletion checks decide its inclusion in Fig 5b).

Values: Additional file 2: Table S9. Source data: Additional file 3.

**Fig. S15. Modality-direction genes for each learned representation.**
Fig 6b (left) for each configuration on cells other than "CytoSet, cells", per pair (columns): the 10 genes with the largest attribution to the single-cell-to-bulk direction, each annotated with its bulk-minus-single-cell-pseudobulk expression difference. (a) Set Transformer. (b) ScRAT. (c) scVI. (d) "Geneformer, cells" (by gene deletion; n/a where its checks failed, Fig. S14d). (e) "scGPT, cells". Values: Additional file 2: Table S10. Source data: Additional file 3.

**Fig. S16. Summary of all comparisons with pseudobulk.**
Difference in bulk area under the receiver operating characteristic curve (AUROC) from pseudobulk (diverging color scale centered at 0: blue below, red above; colors saturate beyond ±0.3) for every pair × configuration (rows, grouped by pair). Columns are the main analysis (ridge and the least absolute shrinkage and selection operator, LASSO) and every sensitivity analysis that forms a per-configuration difference. Hatched cells are not formed. Values: Additional file 2: Tables S2, S3, S5, and S6. Source data: Additional file 3.

---

## Main table

**Table 1. Dataset pairs.**
Legend (below the table): Two rows per pair, single-cell then bulk, in Fig 1d's order. Columns: the study and accession; the tissue or preparation; the assay (scRNA-seq, single-cell RNA sequencing; snRNA-seq, single-nucleus RNA sequencing); the groups compared, with n per group; samples, with donors in parentheses where a donor gives more than one sample; and cells per sample after quality control, median (range). Bulk group counts include only the bulk samples that were scored. Selection and quality-control steps: Additional file 2: Table S1.
ᵃ n per group counts donors in the pairs scored by donor (human blood, aging; human blood, lupus; human ileum, Crohn's; and human motor cortex, ALS) and samples in the others.
ᵇ Donors or mice are given where a donor or mouse contributes more than one sample.
ᶜ Cells per sample after quality control; at most 1000 per sample were stored for modeling. In human blood, aging, cells are counted in each sample's random 3000-cell subsample, drawn before quality control (Additional file 1), hence the maximum of 3000.
ᵈ In mouse organs, aging, the age groups differ between sides: 1–3 vs. 18–30 months on the single-cell side and ≤ 6 vs. 18–30 months on the bulk side.

---

## Additional file 2: supplementary tables

One .xlsx workbook, one sheet per sub-table, each sheet headed by its title and closed by its legend and footnotes.

**Table S1. Selection criteria, characteristics, and quality control of the dataset pairs.**
Legend: Six sub-tables. S1A, the pair-selection criteria (no transfer result was computed before a pair was accepted). S1B, the accepted pairs and the candidates formally evaluated under the criteria (ulcerative colitis; lupus via GSE112087): the phenotype on each side, the sides, and the donor relationship, with each decision. S1C, the criteria met per pair (● met; ○ not met; —, not applicable), including mouse microglia's bulk sample size (kept as a pair analyzed before the criteria were set; Methods). S1D, every characteristic of Fig 1e and Additional file 1: Fig. S8a, with its definition, threshold, and value per pair (● has it; ○ not determinable; blank, does not have it). S1E, cell and sample quality control and selection per pair and side. S1F, samples and cells removed at each selection and quality-control step, and the sex composition per side. Studies, accessions, and counts: Table 1. Footnotes as in each sub-table.

**Table S2. Transfer performance of pseudobulk, the nine learned configurations, and reference analyses.**
Legend: Per pair × configuration, with training and bulk inputs: n per group; bulk area under the receiver operating characteristic curve (AUROC), mean ± SD over seeds; the difference from pseudobulk with its two-sided 95% bootstrap confidence interval and one-sided 95% upper bound (u); one-sided "better" and "worse" p values, Holm-adjusted within each family; the Brier score; area under the precision–recall curve; seeds passing the training check; within-single-cell AUROC; the label-permuted control (20 permutations); the decomposition; and the summary rank (the rank of each difference within its pair, largest first and pseudobulk included at 0, and its mean over pairs). Per pair and pooled: the within-bulk reference; per pair: the a priori minimum detectable differenceᵃ. Pooled estimates: equal weights (main), inverse-variance, Hartung–Knapp random-effects, and logit-scale, with heterogeneity; the equal-weight estimate without each pooled pair; and the pooled decomposition. Reference analyses: Fig 2a's ✕ rows and their variants, including deconvolution from annotated fractions. The pool of the criteria-accepted pairs (human blood, lupus; human ileum, Crohn's). Sensitivity analyses: human blood, aging with predictions combined over folds and with a 67/29 donor split; passing seeds only; the out-of-fold readout, trained only on samples the encoders were not trained on (about 30%); bulk scaled on moments of unscored bulk, with the single-cell training moments, or with one scaler for both sides; each supervised set encoder's own published training procedure; best candidate on bulkᵇ; bulk as 10 bags of thinned pseudo-cellsᶜ; and scVI with the observed bulk library size.
ᵃ Computed from group sizes before bulk labels were used (assumptions: Methods).
ᵇ Chosen with bulk labels, for reference only.
ᶜ Tested, with Holm adjustment within its own families: the supervised set encoders and the unsupervised generative model (scVI), on cells.

**Table S3. Transfer performance under alternative readouts.**
Legend: Table S2's layout for each alternative readout: LASSO, elastic net, partial least squares, k-nearest neighbors, and a multilayer perceptron, CytoSet's own classification head, and a readout on numeric age (aging pairs), with the Brier score and pseudobulk's AUROC and Brier score under each readout. Only the ridge readout is Holm-adjusted (Table S2); LASSO is reported per pair only; the other readouts are pooled with equal weights over the ♦ pairs (pseudobulk bulk AUROC below 0.95) and are unadjustedᵃ.
ᵃ Unadjusted analyses are reported with intervals and never called significant.

**Table S4. Configuration searches and tuning budgets.**
Legend: For each supervised set encoder (CytoSet, Set Transformer, and ScRAT): every candidate's single-cell validation binary cross-entropy and bulk area under the receiver operating characteristic curve (AUROC), the adopted settings, and the test of the combined configuration; the tuning budget (candidate trainings × seeds: the base and 26 candidates at 5 seeds, 135 trainings per unit, for every encoder, except CytoSet's 25 candidates and 130 trainings on the three non-aging pairs, which have no age-similarity loss; plus 5 trainings of the combined configuration wherever it was tested); the per-fold selections for human blood, aging, and whether each unit kept its base; stop-epoch distributions; training-check results per seed; the candidate best on bulk; and each encoder's own published training procedure. scVI's small grid is reported separately and is not counted in the budget. All candidates per seed: Additional file 5.

**Table S5. Robustness of the comparison with pseudobulk.**
Legend: Differences in bulk area under the receiver operating characteristic curve (AUROC) from pseudobulk, with 95% confidence intervals, within bulk subgroups (per pair, with group sizes and estimability) and with age-adjusted bulk scores (Crohn's); ALS within-single-nucleus AUROC before and after age adjustment; encoder- and readout-seed variance and the seed ensemble; pseudobulk under TMM and VST normalization; and gene panels of 1000 and 2000 genes and every shared gene. All analyses in this table are unadjusted for multiple comparisonsᵃ.
ᵃ Reported with intervals and never called significant.

**Table S6. Full gene sets, native mouse genes, and learning curves.**
Legend: Differences in bulk area under the receiver operating characteristic curve (AUROC) from pseudobulk as in Fig 2a, with 95% confidence intervals: Geneformer and scGPT on their full gene sets; pseudobulk on panel genes with one-to-one human orthologs (mouse pairs); and each configuration at 25, 50, and 100% of training donors (human blood, aging; human blood, lupus; human ileum, Crohn's; 5 seeds), as differences from pseudobulk at the same fraction.

**Table S7. Modality gap and transfer before and after alignment.**
Legend: Per pair × configuration: the modality silhouette (raw and after per-side standardization) and bulk area under the receiver operating characteristic curve (AUROC) under per-side (no alignment), ComBat, correlation alignment (CORAL), and Harmony, and one scaler for both sides (no alignment). Trained models: median silhouette and mean AUROC over seeds. Each configuration's difference in AUROC from pseudobulk aligned the same way, per pair and pooled, with 95% confidence intervals and u. Alignment differences are descriptive, with no statistical test.

**Table S8. Transfer under shared supervised set-encoder design choices.**
Legend: Fig 4's values per supervised set encoder and pair: each candidate's change from the base, its difference in bulk area under the receiver operating characteristic curve (AUROC) from full pseudobulk (mean over 5 seeds) and the selected configuration; each pair's best-candidate max-statistic test (p, Holm-adjusted p across pairs, the candidate with the maximum, and whether it is the selected configuration)ᵃ; candidates listed but not testedᵇ; and the observed encoder-seed SD of bulk AUROC (median and maximum over candidates, and the selected configuration's)ᶜ.
ᵃ In simulation, the test rejected a true null hypothesis in 1.6–3.1% of data sets (nominal 5%); the acceptable range for that rate was revised after the simulation was run.
ᵇ Candidates with zero bootstrap standard error.
ᶜ Above about 0.045, the test may reject a true null hypothesis more often than 5%.

**Table S9. Genes behind transfer and the simulation.**
Legend: Per pair: the concordance of single-cell and bulk phenotype shifts (Spearman ρ); each configuration's shared-gene share for its top 25, 50, and 100 genes under ridge and LASSO, with the two reference shares; gene-list stability; the attribution checks and Geneformer's inclusion; and the simulation's within-single-cell and simulated-bulk area under the receiver operating characteristic curve (AUROC) per scenario and model, with the seeds passing the training check and the criterion's resultᵃ. Gene lists: Additional file 4.
ᵃ Values that hold by construction are marked and are not results.

**Table S10. Sample properties captured by each configuration.**
Legend: Per pair × representation (pseudobulk, the nine learned configurations, and pseudobulk reduced to up to 64 principal components): the cross-validated R² for sequencing depth, mitochondrial fraction, and cell-type composition, and the area under the receiver operating characteristic curve (AUROC) for the phenotype, with permutation nulls (mean and 95th percentile) and, per pair, the number of the nine learned configurations (every configuration but pseudobulk) exceeding both pseudobulk baselines and whether the pair meets the criterion; the modality-direction genes and the cross-model summary; and the one-cell-type within-single-cell and bulk AUROC.

**Table S11. Computational cost.**
Legend: Wall-clock time and graphics-processing-unit (GPU) hours per configuration and analysis step, per pair, with the hardware used.

**Table S12. Sample-level methods considered and the inclusion criteria.**
Legend: Criteria: (i) embeds a bulk profile, a sample with no cells; (ii) public code; (iii) trainable at the available sample sizes (41–868 single-cell training samples). Included: the six models of Fig 1b. Excluded, with reasons: MrVI, PILOT, GloScope, scPoli sample embeddings, PaSCient, fine-tuned foundation models, ProtoCell4P, and CloudPred. Pseudobulk is the reference representation, not a candidate.
ᵃ An exclusion reason beyond the inclusion criteria.
ᵇ The one exclusion that fails none of the three criteria (PaSCient).

## Additional files

**Additional file 1.** Format: PDF (.pdf). Title: Supplementary figures. Description: Figs. S1–S16 with their legends: the supervised set-encoder architectures and the embedding paths of the unsupervised generative and frozen foundation models (Fig. S1); quality control for each dataset pair (Figs. S2–S7); the characteristics of the dataset pairs (Fig. S8); transfer under alternative readouts, the configuration searches and training checks, robustness analyses, and full gene sets, native mouse genes, and learning curves (Figs. S9–S12); the modality gap in every pair (Fig. S13); robustness of the gene lists (Fig. S14); modality-direction genes for each learned representation (Fig. S15); and a summary of all comparisons with pseudobulk (Fig. S16).

**Additional file 2.** Format: Excel workbook (.xlsx). Title: Supplementary tables. Description: Tables S1–S12, one sheet per sub-table: the selection criteria, characteristics, and quality control of the dataset pairs (Table S1); transfer performance of pseudobulk, the nine learned configurations, and reference analyses, under alternative readouts, and across robustness and further analyses (Tables S2, S3, S5, and S6); the configuration searches and tuning budgets (Table S4); the modality gap and transfer before and after alignment (Table S7); shared supervised set-encoder design choices (Table S8); the genes behind transfer and the simulation (Table S9); sample properties captured by each configuration (Table S10); computational cost (Table S11); and the sample-level methods considered with the inclusion criteria (Table S12).

**Additional file 3.** Format: ZIP archive of CSV files (.zip). Title: Source data for every figure panel, including per-seed values. Description: one CSV per figure panel (main and supplementary), with per-seed values, and a README.csv indexing each file by its panel ("Fig 2a", "Fig. S9a"), description, and columns.

**Additional file 4.** Format: ZIP archive of CSV files (.zip). Title: Gene rankings, top-50 lists, and attributions. Description: per dataset pair, every configuration's gene ranking per readout and target (attribution, rank, and top-25/50/100 membership), Geneformer's inclusion and the attribution checks, and Fig 5a's per-gene phenotype shifts; README.csv indexes the files.

**Additional file 5.** Format: ZIP archive of CSV files (.zip). Title: Every configuration-search candidate, per seed. Description: per dataset pair, every search candidate of the three supervised set encoders per unit and search seed (change from the base, validation loss, training check, selection) with its bulk AUROC, and the unsupervised generative model's search (scVI; validation ELBO per candidate and seed); README.csv indexes the files.

**Additional file 6.** Format: ZIP archive of CSV files (.zip). Title: Per-sample predictions. Description: per dataset pair, each bulk and held-out single-cell sample's score and probability for every configuration and reference analysis, averaged over encoder and readout seeds (ridge readout), with its fold, donor or sample, and group; README.csv indexes the files. Per-seed predictions are deposited on Zenodo rather than in this file.
