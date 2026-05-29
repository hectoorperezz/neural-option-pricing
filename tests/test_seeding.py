"""Test de reproducibilidad global.

El README y los logs documentan que fijando la semilla se reproducen
los resultados bit a bit. Sin este test, si ``set_global_seed`` deja
de resetear alguno de los tres generadores (``random``, ``numpy``,
``torch``), dos ejecuciones nominalmente idénticas darían pesos
distintos y nadie podría reproducir las cifras del paper.
"""

import random

import numpy as np
import pytest
import torch

from src.utils import set_global_seed


def test_set_global_seed_makes_rngs_reproducible() -> None:
    """Dos llamadas con la misma semilla devuelven idéntica secuencia."""
    set_global_seed(123)
    first = (random.random(), np.random.random(), torch.rand(1).item())

    set_global_seed(123)
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert second == pytest.approx(first)
