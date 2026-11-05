"""The six single-cell -> bulk pairs, in the paper's order.

Per pair:
  name            the descriptive display name, "<Species> <tissue>, <phenotype>"
  title           the two-line panel title (tissue / phenotype), the same everywhere
  studies         the study mapping, given once at first mention, in Table 1 and Fig 1D
  species         "mouse" or "human"
  single_cell     the training side: source and accession
  bulk            the transfer target: source and accession
  label_value     what the raw label is ("age" in the pair's unit, or "als" = 1 for ALS,
                  0 for control); samples carry it as `label_value` and their 0/1 `group`
  groups          the pre-declared 0/1 group on each side, as inclusive ranges of the
                  label (0 = younger / control, 1 = older / ALS): CytoSet's BCE target,
                  the read-out target, and the bulk AUROC direction
  bulk_label_columns  bulk metadata columns that encode the label (checked against the 0/1
                  group on kept samples, 2026-10-04); with label_value and group they go to
                  stage 02's labels file only, behind the bulk-label firewall (sc2bulk.labels)
  scoring_unit    the unit bulk predictions are averaged over before scoring
  holdout_group   what a split keeps together (Fig 2B holdout, encoder train/validation).
                  Splits group on each sample's `holdout_group` column (stage 02's sample
                  table), which the readers set; this entry describes it
  expected        sample counts stage 02 must reach after every filter (asserted)
  fold_sets       how single-cell training and bulk scoring are partitioned (sc2bulk.folds):
                  {name: number of folds}, the primary set first. "all" = one fold, every
                  single-cell sample trains and every bulk sample is scored (no donor is on
                  both sides). Where donors overlap, the pair is cross-fitted by donor
                  ("cross-fit exactly when donors overlap")
  qc_max_percent_mito  optional per-pair mitochondrial ceiling (default settings.QC_MAX_PERCENT_MITO)
  one_type        Fig 6C's single cell type, fixed in advance: the per-cell column and the
                  labels counted as the type ("by": a sample column the type depends on,
                  e.g. TMS's organ, or None). Stage 02 asserts it is the most abundant type
                  by pooled cells and by mean per-sample fraction

Source files and their checksums are in sources.py.
"""
import importlib

PAIR_ORDER = ("hammond", "tms", "soundlife", "lupus", "crohns", "pineda")   # aging first (mouse, then human), then disease

# Each pair's configuration is its own module (config/pairs/<pair>.py), hashed only into that
# pair's records. Loaded by name: the code hash of a stage includes config/pairs/<pair>.py for the
# pair it runs (sc2bulk.codehash), not the other pairs' files.
PAIRS = {pair: importlib.import_module(f"sc2bulk.config.pairs.{pair}").PAIR for pair in PAIR_ORDER}
