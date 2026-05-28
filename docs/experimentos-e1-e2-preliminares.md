# Conclusiones preliminares de E1 y E2

**Autores:** Ángel Fernández Sánchez, Jorge Alfageme Sotillos, Héctor Pérez Ledesma

## Motivación

---

Con la infraestructura de evaluación cerrada en `src/evaluation/` y las dos primeras clases de experimentos materializadas en `src/experiments/` (`PriceVsIVStudy` y `ActivationStudy`), los dos primeros experimentos del proyecto ya tienen su entregable cuantitativo sobre el test balanced documentado en `tasks.md` (5k puntos por bin, 25 bins, 125k puntos por familia de modelo). Este documento resume las observaciones que se desprenden de esa primera ronda. Es deliberadamente corto y respeta la doble restricción que `metodologia.md` impone sobre ambos experimentos: E1 reporta diagnóstico sin clasificación, E2 reporta el patrón observado sin emitir veredicto fuerte/débil. Los surrogates evaluados son los entrenados sobre el régimen 50x descrito en `experimento-escalado-datos.md`; el cambio de tamaño de dataset solo afecta a la calidad de la red, no al protocolo de evaluación, que sigue al pie de la letra los 125k por bin que fijan `tasks.md` y `metodologia.md`.

## E1 — Precio frente a volatilidad implícita

---

Sujetos: BS-3 y H-3, los surrogates baseline de cada familia que `tasks.md` §E1 designa explícitamente. Métrica primaria: discrepancia por bin entre `MAE(C/K)` y `MAE_IV`, materializada en la columna `iv_to_price_ratio` del CSV (cociente entre las dos medias por bin). Diagnóstico complementario: tasa de fallos de inversión IV por bin y un Vega proxy definido como `BlackScholesSolver.call_vega` evaluada en la IV recuperada del precio verdadero del solver, promediada por bin.

Los cinco bins de mayor discrepancia, leídos directamente de `results/metrics/e1_table_baseline.csv`:

| Surrogate | Bin | `iv_to_price_ratio` | `price_mae_mean` | `iv_mae_mean` |
|---|---|---|---|---|
| H-3 | deep_otm_weekly | 1444 | 1.09e-03 | 1.57 |
| H-3 | deep_otm_short | 1136 | 6.85e-04 | 7.77e-01 |
| BS-3 | deep_otm_weekly | 700 | 2.69e-03 | 1.89 |
| BS-3 | deep_otm_short | 619 | 1.68e-03 | 1.04 |
| H-3 | deep_otm_medium_short | 465 | 4.09e-04 | 1.90e-01 |

La tasa media de fallos de inversión IV por bin es del 24.11% para BS-3 y del 14.50% para H-3, concentrada en los mismos bins de cola en los dos casos.

La lectura es la que `metodologia.md` §E1 anticipaba como justificación del propio experimento: en regiones de Vega baja, errores absolutamente pequeños en precio se traducen en errores operativamente grandes en IV. En `deep_otm_weekly` los dos surrogates tienen `MAE(C/K)` del orden de `1e-3` y, sin embargo, sus `MAE_IV` superan la unidad, tres órdenes de magnitud por encima. La consecuencia metodológica es directa: IV debe formar parte del protocolo de evaluación aunque no sea el target de entrenamiento, porque el `MAE(C/K)` global puede ocultar errores relevantes en la escala que el mercado usa. Como exige el documento, no se emite clasificación fuerte/débil para E1; los heatmaps separados de precio e IV por surrogate quedan archivados en `results/figures/e1_baseline/`.

## E2 — Activaciones y calidad de Delta

---

Sujetos: BS-1 a BS-4 (Black-Scholes con ReLU, Softplus, Swish y tanh) y H-1 a H-4 (los mismos en Heston). Métrica primaria: `MAE_Delta` por bin. Control: `MAE(C/K)` por bin. El test es el balanced documentado de 125k puntos por familia. Las cifras se leen de `results/metrics/e2_bs.csv` y `results/metrics/e2_heston.csv`.

| Familia | Activación | `MAE_Delta` medio | `p95` peor bin | `MAE(C/K)` medio |
|---|---|---|---|---|
| BS | Swish | 1.78e-02 | 1.87e-01 | 1.80e-03 |
| BS | Softplus | 2.32e-02 | 2.11e-01 | 4.27e-03 |
| BS | tanh | 2.59e-02 | 2.52e-01 | 2.78e-03 |
| BS | ReLU | 2.69e-02 | 1.57e-01 | 1.97e-03 |
| Heston | Swish | 1.11e-02 | 1.76e-01 | 1.28e-03 |
| Heston | Softplus | 1.58e-02 | 1.82e-01 | 2.63e-03 |
| Heston | ReLU | 1.64e-02 | 1.22e-01 | 1.27e-03 |
| Heston | tanh | 1.72e-02 | 2.07e-01 | 1.91e-03 |

Se sostienen tres observaciones, sin clasificación formal.

La primera es que Swish gana en `MAE_Delta` medio por bin en las dos familias. La mejora respecto a ReLU es del 34% en Black-Scholes y del 32% en Heston. La activación que `metodologia.md` y el paper de Chen identifican como suave es la que mejor aproxima la derivada, lo que es exactamente la hipótesis arquitectónica que E2 fue diseñado para contrastar.

La segunda es que el control de precio se cumple. El `MAE(C/K)` medio se mantiene entre `1.3e-3` y `4.3e-3` en Black-Scholes y entre `1.3e-3` y `2.6e-3` en Heston, sin diferencias de orden entre activaciones. Ninguna activación gana en Delta a costa de sacrificar el nivel del precio, que era la condición que la metodología exigía para que la elección de activación tuviera sentido.

La tercera matiza la lectura anterior. El `p95` del peor bin invierte el ranking en cola: ReLU es la menos extrema en `p95` (`1.57e-01` en Black-Scholes, `1.22e-01` en Heston) aunque sea la peor en media. Es decir, ReLU comete menos errores excepcionales pero falla sistemáticamente más en el grueso de los puntos; Swish invierte esa relación. Esto importa porque la elección operativa entre las dos activaciones depende de si una aplicación posterior penaliza el error típico o el error máximo. La métrica primaria de E2 sigue siendo `MAE_Delta`, pero el patrón en colas se reporta como diagnóstico.

Como exige `metodologia.md` §E2, no se emite clasificación fuerte/débil; los heatmaps de precio y Delta por surrogate quedan archivados en `results/figures/e2/`.

## Trabajo pendiente

---

E3 (muestreo uniforme frente a enfocado), E4 (eficiencia del surrogate frente al solver) y E5 (differential ML con Delta) siguen abiertos. La infraestructura para E3 y E5 reutilizará las clases ya creadas en `src/experiments/` y la pieza `SurrogateInput.labels` que se introdujo para E2; E4 requiere implementar el `TimingBenchmark` que `architecture.md` §Evaluation prevé pero que aún no existe.
