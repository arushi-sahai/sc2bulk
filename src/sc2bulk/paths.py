"""Where every output lives: outputs/<stage>/<pair>/<unit>/<name> (unit = <fold_set>_<fold>), or
outputs/<stage>/<pair>/<name> for pair-level files. Every stage and figure script finds files
through output()."""
from pathlib import Path

from sc2bulk import folds
from sc2bulk.config import settings


def output(stage: str, pair: str, name: str, unit: tuple | None = None) -> Path:
    """The path of an output file; unit = (fold_set, fold) or None for a pair-level file.
    Label-bearing outputs (named settings.LABELLED + "_") go under settings.LABELLED_OUTPUTS, so a
    blinded run's can't be mistaken for real ones."""
    labelled = name.startswith(settings.LABELLED + "_")
    root = (settings.LABELLED_OUTPUTS if labelled else settings.OUTPUTS) / stage / pair
    return root / name if unit is None else root / folds.unit_name(*unit) / name
