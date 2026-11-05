# Contributions and use of AI

Arushi Sahai conceived and led this study. Her main contributions:

**The question and the benchmark design**
- Framed the question, whether models built on learned single-cell sample representations predict phenotypes in bulk RNA-seq more accurately than pseudobulk, out of her earlier work applying single-cell phenotype models to bulk cohorts.
- Designed the benchmark that answers it: a shared ridge classifier trained on single-cell samples and applied unchanged to bulk samples, so that differences come from the representation; pseudobulk as the baseline against nine learned configurations; and two readouts, one trained on every labeled sample and one only on samples the encoders were not trained on.
- Set the evaluation: bulk AUROC as the metric, a margin of 0.05 AUROC as the smallest advantage that would justify a learned representation, a pooled comparison over the pairs where pseudobulk leaves room for that margin, one-sided upper bounds, and Holm-adjusted tests.

**The datasets**
- Chose the six single-cell-to-bulk dataset pairs across two species, aging, and three diseases, and wrote the criteria a pair had to meet.
- Decided how each pair's problems were handled: splitting the Sound Life donors so that no model scores the bulk of a donor whose cells it saw, restricting lupus to one processing cohort to remove batch confounding, keeping and disclosing a pair whose bulk side falls short of the size criterion, excluding a problem donor, and setting tissue-appropriate quality-control thresholds.

**The models and a fair comparison**
- Chose the models and organized them into families: supervised set encoders (CytoSet, Set Transformer, ScRAT), an unsupervised generative model (scVI), and frozen foundation models (Geneformer, scGPT), with CytoSet and the foundation models also applied to pseudobulk profiles.
- Required equal tuning budgets for the three set encoders, set after a timing test, kept the foundation models frozen and scVI at its published defaults, and disclosed the asymmetry.
- Contributed CytoSet's training additions from her preliminary work and decided which models were out of scope.

**The analyses**
- Designed the analyses that explain the main result: the single-cell-to-bulk gap and its decomposition, the reference and control rows (within-bulk reference, permuted labels, cell-type composition, published signatures, simple cell aggregations), label-free alignment, the set encoders' design choices, gene-level agreement, a planted-signal positive control, technical properties of the embeddings, and cell-type restriction.
- Kept the bulk labels unreadable by the pipeline until the analysis was final, and had the full pipeline run end to end on scrambled labels first to catch bugs.

**How the work was run and checked**
- Ran the project as a specified, audited process: each pipeline stage had a written specification and an independent audit of the specification and of its code before it ran, and every decision went into a numbered log of more than 100 entries.
- Planned and oversaw the computation on a shared laboratory GPU server (about 2,100 GPU hours), including smoke runs, failure recovery, and full rebuilds.
- Reviewed the code and its outputs, and set up independent audits of the tables, figures, and text.

**Results, figures, and writing**
- Interpreted the results and decided the claims and how they are qualified, including reporting the out-of-fold readout beside the main result and comparing each approach with every label available to it.
- Directed the figures and tables, and wrote and revised the paper with her co-authors.

Natalie Stanley supervised the project.

**Use of AI tools.** Arushi used Claude (Claude Opus 5.5, Anthropic), through the Claude Code command-line tool, as an assistant under her direction. It wrote code to her specifications, ran and monitored jobs on the laboratory server, drafted text from her outlines and the analysis outputs, and carried out the audits she set up. Every reported number comes from the outputs of the released code, figures were drawn by that code, and none contains AI-generated imagery. The authors take full responsibility for the work.
