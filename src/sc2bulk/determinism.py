"""The settings every stage runs under, in one place.
configure() is called at the top of every stage script, before any numerical or CUDA work."""
import os

import torch
from threadpoolctl import threadpool_limits

from sc2bulk.config import settings

_limits = None


def configure():
    """Deterministic algorithms, the cuBLAS workspace (set before the first CUDA call), no cuDNN
    autotuning, and pinned thread counts for BLAS/OpenMP (threadpoolctl; environment variables
    set after numpy's import have no effect) and torch."""
    global _limits
    assert not torch.cuda.is_initialized(), "configure() must run before any CUDA work"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    _limits = threadpool_limits(settings.N_THREADS)
    torch.set_num_threads(settings.N_THREADS)
