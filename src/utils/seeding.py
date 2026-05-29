"""Punto único para fijar semillas y determinismo de PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Fija ``PYTHONHASHSEED``, ``random``, ``numpy`` y ``torch`` desde un único punto.

    Args:
        seed: Semilla no negativa; las semillas por surrogate están
            documentadas en ``docs/metodologia.md``.
        deterministic_torch: Si ``True``, fuerza ``cudnn`` a modo
            determinista (más lento pero reproducible).
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
