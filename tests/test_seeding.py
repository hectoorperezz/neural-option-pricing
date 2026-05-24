import random

import numpy as np
import pytest
import torch

from src.utils import set_global_seed


def test_set_global_seed_makes_rngs_reproducible() -> None:
    set_global_seed(123)
    first = (random.random(), np.random.random(), torch.rand(1).item())

    set_global_seed(123)
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert second == pytest.approx(first)


def test_set_global_seed_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        set_global_seed(-1)
