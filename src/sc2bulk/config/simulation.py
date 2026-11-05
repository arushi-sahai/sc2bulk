"""The planted-signal simulation, a positive control (stage 12_simulation, Fig 5D). Every number here is pre-declared, before any run.

Hashed only into the stages that import it (sc2bulk.codehash), so its own file.
"""
from sc2bulk.config import settings, smoke, triage

PAIR = "crohns"                        # human ileum, Crohn's: one sample per donor
UNIT = ("all", 0)
SPLITS = tuple(range(smoke.N_ENCODER_SEEDS if settings.SMOKE else 5))   # encoder splits e = 0-4 (stage 03's)
# The set encoders kept after triage: an encoder whose cells configuration is in CUT_CONFIGURATIONS at this stage's
# first real run (as the licence's n is pinned) is neither trained nor judged, and "every set encoder passes"
# means every kept one (criterion.json lists the cut ones)
SET_ENCODERS = ("cytoset", "set_transformer", "scrat")
ENCODERS = tuple(m for m in SET_ENCODERS if f"{m}_cells" not in triage.CUT_CONFIGURATIONS)
assert ENCODERS, "the positive control needs at least one set encoder"
SCENARIOS = ("mean_shift", "structure_only")

# The planted genes G: N_PLANTED panel genes, uniformly from those detected in >= DETECTION_MIN_FRACTION of the unit's
# training cells and not mitochondrial (label-free); the mean scenario splits G at random into halves G+ and G-
N_PLANTED = 50
DETECTION_MIN_FRACTION = 0.10
MITO_PREFIXES = ("MT-", "mt-")

# Strength: per scenario, the weakest level (grids in order, weakest first) at which the oracle reaches ORACLE_TARGET
# AUROC over all the unit's samples; if none does, the strongest, reported
RHO_GRID = (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5)              # mean shift: binomial keep probability
Q_GRID = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)        # structure only: the label-1 subset's fraction of cells
GRIDS = {"mean_shift": RHO_GRID, "structure_only": Q_GRID}
ORACLE_TARGET = 0.90

# The success criterion: each set encoder's within-single-cell AUROC on half B exceeds pseudobulk's
# by >= SUCCESS_MARGIN, averaged over e and r, in the structure-only scenario; mean_log_cells is judged by the same rule
#
SUCCESS_MARGIN = 0.10
MARGIN_TOLERANCE = 1e-12               # inclusive, compared with a 1e-12 tolerance (the ceiling rule's)
CRITERION_SCENARIO = "structure_only"

# Seed streams: `simulation` (id 18; drawn nowhere else), unit-keyed. Per sample, count draws use
# SeedSequence([base, sample_seed % 2**32, role]): role ORDER_ROLE = the thinning uniforms (mean shift) or the cell
# ordering (structure only), shared across levels so levels nest; LABEL0_ROLE = label 0's reallocation (no level);
# LABEL1_ROLE + level index = label 1's reallocation at that level
GENES_INDEX = 0
LABELS_INDEX = 1
DRAWS_INDEX = {"mean_shift": 2, "structure_only": 3}
HALF_OFFSET = 10                       # half A / half B of split e: simulation index HALF_OFFSET + e
ORDER_ROLE, LABEL0_ROLE, LABEL1_ROLE = 0, 1, 2

# Halves of each validation side: >= 2 holdout groups per simulated label in each half (did-it-train needs 2
# in half A). The smoke run's validation sides hold ~5 groups, too few to halve: there half A = half B = the whole
# validation side, and the simulated labels are also stratified by split 0's side, so it holds >= 2 per label (wiring only)
MIN_HALF_GROUPS = 2
