"""Test del veredicto pre-registrado de E3 (sampling).

``docs/metodologia.md §E3`` clasifica el resultado en
``positivo_fuerte``, ``positivo_debil`` o ``negativo`` según dos
umbrales fijos: la mejora local en los bins críticos ATM y la
degradación global tolerada. Si el código aplicara otros umbrales, el
veredicto reportado en el paper no sería defendible. Este test fuerza
los tres escenarios y comprueba que la función decisoria devuelve la
etiqueta esperada.
"""

import pytest

from src.experiments.sampling_study import decide_verdict


@pytest.mark.parametrize(
    ("local_improvement", "global_degradation", "expected"),
    [
        (0.15, 0.05, "positivo_fuerte"),
        (0.05, 0.0, "positivo_debil"),
        (0.0, 0.0, "negativo"),
        (0.30, 0.25, "negativo"),
    ],
)
def test_decide_verdict_follows_preregistered_thresholds(
    local_improvement: float,
    global_degradation: float,
    expected: str,
) -> None:
    """Los umbrales 10% local y 10% global producen la clasificación documentada."""
    assert decide_verdict(local_improvement, global_degradation) == expected
