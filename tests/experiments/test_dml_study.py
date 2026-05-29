"""Test del veredicto pre-registrado de E5 (differential ML).

``docs/metodologia.md §E5`` clasifica el resultado en
``positivo_fuerte``, ``positivo_debil`` o ``negativo`` según la
mejora de ``MAE_Delta`` (umbral 20%) y la tolerancia a degradación de
``MAE(C/K)`` (umbral 10%). Si el código aplicara otros umbrales, el
veredicto del paper no sería defendible. Este test fija los cuatro
escenarios canónicos.
"""

import pytest

from src.experiments.dml_study import decide_verdict


@pytest.mark.parametrize(
    ("delta_improvement", "price_degradation", "expected"),
    [
        (0.25, 0.05, "positivo_fuerte"),
        (0.25, -0.05, "positivo_fuerte"),
        (0.10, 0.05, "positivo_debil"),
        (0.30, 0.15, "negativo"),
    ],
)
def test_decide_verdict_follows_preregistered_thresholds(
    delta_improvement: float,
    price_degradation: float,
    expected: str,
) -> None:
    """Los umbrales 20% (Delta) y 10% (precio) producen la clasificación documentada."""
    assert decide_verdict(delta_improvement, price_degradation) == expected
