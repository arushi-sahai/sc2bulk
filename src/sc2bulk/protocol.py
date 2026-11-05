"""The encoder protocol: one interface for every sample
encoder (CytoSet, Set Transformer, ScRAT, scVI) and for outside users of the released
benchmark. The contract is fit / embed (+ an optional predict for native heads). The harness
draws every bag (sc2bulk.bags) and applies the one normalization; encoders never normalize.
Frozen foundation models implement embed_cells / embed_profiles only (their fit does nothing)
and run in their own environments; the embeddings file is the interface between environments."""
from dataclasses import dataclass
from typing import Literal, Optional, Protocol

import numpy as np
import pandas as pd

InputSpace = Literal["log_panel", "counts_panel", "counts_all_genes"]


@dataclass
class Cells:
    """A unit's cells in an encoder's input space: X (cells x genes), per-cell cell_id, sample_id,
    cell type and library_size (total counts over all genes, the same definition as
    Profiles.library_size: scVI's observed size factor); per-sample 0/1 group and holdout_group
    (samples: index = sample_id). Bags (sc2bulk.bags) are cell ids; the harness maps them to rows
    once (row_of)."""
    X: object
    genes: list
    cell_id: np.ndarray
    sample_id: np.ndarray
    cell_type: np.ndarray
    library_size: np.ndarray
    samples: pd.DataFrame

    def row_of(self, cell_ids) -> np.ndarray:
        """Row indices of the given cell ids (the id index is built once per Cells)."""
        if not hasattr(self, "_index"):
            self._index = pd.Index(self.cell_id)
        rows = self._index.get_indexer(cell_ids)
        assert (rows >= 0).all(), "a bag names a cell that isn't in these Cells"
        return rows


@dataclass
class Profiles:
    """One-profile inputs (bulk, pseudobulk) in an encoder's input space; they carry no group.
    library_size is each profile's total counts over all genes (scVI's observed library size)."""
    X: object
    genes: list
    sample_id: np.ndarray
    library_size: np.ndarray


@dataclass
class TrainingRecord:
    """Every field is required, so a failed training can't pass as a default. stop_epoch and kept_step
    are the kept checkpoint's; margin_d and margin_p are "did it train?"'s statistic and its
    permutation p-value, margin_d_star and margin_temperature the collapse screen's scale-invariant d*
    and its temperature (sc2bulk.harness.did_it_train); on a split with too few
    labellings to test (testable False), did_train is the collapse screen alone and margin_p is NaN."""
    epochs: int
    stop_epoch: int
    validation_curve: list
    did_train: bool
    did_train_reason: str
    seconds: float
    steps: int
    kept_step: int
    margin_d: float
    margin_d_star: float
    margin_temperature: float
    margin_p: float
    testable: bool
    peak_gpu_bytes: Optional[int]


class Encoder(Protocol):
    input_space: InputSpace

    def fit(self, train: Cells, validation: Cells, seed: int) -> TrainingRecord: ...

    def embed_cells(self, cells: Cells) -> Optional[np.ndarray]: ...

    def embed_bags(self, cells: Cells, bags: list) -> np.ndarray: ...
    # bags: a list of cell-id arrays (from sc2bulk.bags); returns one embedding per bag, in order

    def embed_profiles(self, profiles: Profiles) -> np.ndarray: ...

    def predict(self, cells: Cells, bags: list) -> Optional[np.ndarray]: ...
    # the native head's score per bag, in order (CytoSet's BCE head; ScRAT's majority vote), or None
