from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs from one entry point."""

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
