# Metodología experimental

**Autores:** Ángel Fernández Sánchez, Jorge Alfageme Sotillos, Héctor Pérez Ledesma

## Visión

---

Este documento es la referencia metodológica del proyecto. Fija qué se entrena, qué se mantiene constante, qué métricas se reportan y bajo qué condiciones una comparación se considera limpia. La idea central es combinar dos líneas de la literatura: de Chen, Didisheim y Scheidegger tomamos la lógica de deep surrogates entrenados con datos sintéticos; de Huge y Savine tomamos, de forma acotada, el aprendizaje diferencial con Delta.

La regla principal es que cada experimento varía una sola dimensión. Si se cambia la activación, no se cambia el dataset. Si se cambia el sampler, no se cambia la arquitectura. Si se cambia la pérdida en E5, se mantiene el resto de la comparación tan controlado como sea posible. Esta disciplina permite atribuir las diferencias observadas a la variable experimental y no a decisiones accidentales.

## Pipeline experimental

---

Todos los experimentos siguen el mismo flujo. Primero se define un dominio de inputs y una distribución de muestreo. Después se genera un dataset sintético evaluando el solver verdadero. A continuación se entrena un surrogate con configuración cerrada. Finalmente se evalúa sobre un test independiente y balanceado por bins de moneyness y vencimiento.

La separación entre generación, entrenamiento y evaluación evita contaminar conclusiones. El sampler decide dónde preguntamos al modelo verdadero; el solver produce los targets; el trainer decide cómo aprende la red; y el evaluador mide el error. Una comparación deja de ser interpretable si cambia más de una de esas piezas al mismo tiempo.

## Experimentos

---

| ID | Pregunta | Lo que varía | Surrogates | Métrica primaria |
|---|---|---|---|---|
| **E1** | ¿Un error bajo en precio implica un error bajo en IV? | Métrica de evaluación | BS-3, H-3 | Discrepancia entre `MAE(C/K)` y `MAE_IV` por bin |
| **E2** | ¿Qué activación produce mejores Greeks? | Activación | BS-1..4, H-1..4 | `MAE_Delta` por bin |
| **E3** | ¿El muestreo enfocado mejora regiones difíciles? | Distribución de muestreo | H-3, H-5 | `MAE_IV` en ATM weekly, short y medium-short |
| **E4** | ¿Qué speedup ofrece el surrogate frente al solver? | Tamaño de lote y dispositivo | H-3 | `tiempo_solver / tiempo_surrogate` |
| **E5** | ¿Entrenar con Delta mejora la eficiencia muestral? | Target de entrenamiento | H-3-small, H-6-small, H-3 | Mejora de `MAE_Delta` de H-6-small frente a H-3-small |
| **E6** | ¿Ampliar la profundidad o añadir un scheduler de LR mejoran el baseline sin más datos? | Profundidad de red y scheduler de LR | H-3, H-7-shallow, H-8-deep, H-9-lr-schedule | `MAE(C/K)` medio por bin |

E1 es diagnóstico y no entrena nada nuevo. E2 contrasta el efecto de la suavidad de la activación sobre las derivadas. E3 evalúa si conviene gastar más muestras cerca de ATM y vencimientos cortos. E4 mide utilidad práctica. E5 comprueba si añadir Delta como target diferencial mejora la forma local aprendida por la red. E6, añadido al final para ampliar el estudio, prueba si dos palancas de arquitectura y optimización, mayor profundidad o un scheduler de learning rate, mejoran el baseline H-3 sin aumentar el volumen de datos.

E3 se considera positivo fuerte si H-5 mejora al menos un 10% en los bins críticos y el `MAE_IV` global no empeora más de un 10%. E5 se considera positivo fuerte si H-6-small mejora Delta al menos un 20% frente a H-3-small y el precio no empeora más de un 10%. E1, E2, E4 y E6 no necesitan veredicto fuerte/débil: son diagnósticos, mediciones o lecturas observacionales que se ordenan por reducción frente al baseline.

## Surrogates y datos

---

El diseño usa catorce surrogates. H-3 es el baseline central de Heston y aparece como referencia en E1, E2, E3, E4, E5 y E6.

| ID | Modelo | Activación | Pérdida | Rol |
|---|---|---|---|---|
| BS-1 | Black-Scholes | ReLU | precio | E2 |
| BS-2 | Black-Scholes | Softplus | precio | E2 |
| BS-3 | Black-Scholes | Swish | precio | E1, E2 |
| BS-4 | Black-Scholes | tanh | precio | E2 |
| H-1 | Heston | ReLU | precio | E2 |
| H-2 | Heston | Softplus | precio | E2 |
| H-3 | Heston | Swish | precio | E1, E2, E3, E4, E5, E6 baseline |
| H-4 | Heston | tanh | precio | E2 |
| H-5 | Heston | Swish | precio | E3, sampler enfocado |
| H-3-small | Heston | Swish | precio | E5, baseline pequeño |
| H-6-small | Heston | Swish | precio + Delta | E5, aprendizaje diferencial |
| H-7-shallow | Heston | Swish | precio | E6, red poco profunda (2×128) |
| H-8-deep | Heston | Swish | precio | E6, red profunda (6×128) |
| H-9-lr-schedule | Heston | Swish | precio | E6, scheduler ReduceLROnPlateau |

El diseño inicial fijaba 200k muestras para Black-Scholes, 500k para Heston y 100k para las variantes small. Tras el diagnóstico de escalado documentado en `docs/experiments/data-scaling.md`, los checkpoints usados en los resultados finales se entrenan en régimen 50x:

| Grupo | Diseño inicial | Régimen final |
|---|---:|---:|
| BS-1..4 | 200k uniforme + Delta | 10M uniforme + Delta |
| H-1..4 | 500k uniforme | 25M uniforme |
| H-5 | 500k enfocado | 25M enfocado |
| H-3-small | 100k uniforme | 5M uniforme + Delta, usando solo precio |
| H-6-small | 100k uniforme + Delta | 5M uniforme + Delta |
| Validación BS | 50k uniforme + Delta | 2.5M uniforme + Delta |
| Validación Heston | 50k uniforme | 2.5M uniforme |

El test formal no escala a 50x. Se mantiene en 125k puntos por familia, con 5k observaciones por bin, para evitar que la inversión IV domine el coste de evaluación. En E5, H-3-small y H-6-small comparten la misma muestra small final y la misma semilla de entrenamiento; H-3-small ignora la columna de Delta porque su pérdida solo usa precio.

## Targets y normalización

---

El target principal de entrenamiento es el precio normalizado por strike,

```text
y = C / K
```

y el input financiero principal es la moneyness simple,

```text
m = S / K
```

Esta elección elimina una escala redundante: si spot y strike se multiplican por la misma constante, el precio de una call europea escala con el strike. Además deja limpia la relación de Delta:

```text
Delta = dC/dS = d(C/K)/d(S/K) = dy/dm.
```

No usamos la moneyness normalizada de Chen et al. porque nuestro objetivo no es replicar la escala empírica de SPX, sino construir un entorno controlado donde precio normalizado y Delta estén directamente relacionados. También fijamos `q = 0`; el dividend yield queda fuera de las preguntas centrales y eliminarlo reduce la dimensión del problema.

Todos los inputs se normalizan a `[0, 1]` antes de entrar en la red. Cuando se calcula la Delta del surrogate, se aplica la regla de la cadena:

```text
m_norm = (m - m_min) / (m_max - m_min)
Delta_hat = (d y_hat / d m_norm) / (m_max - m_min).
```

La Delta del surrogate se obtiene con autograd de PyTorch. En evaluación se usa `create_graph=False`; en E5, durante entrenamiento diferencial, se usa `create_graph=True` porque la pérdida de Delta debe retropropagarse hasta los pesos.

La volatilidad implícita Black-Scholes no es target de entrenamiento. Se usa como métrica de evaluación porque en mercado un error pequeño en precio puede ser grande en escala IV, especialmente en regiones de Vega baja.

## Solvers y validación

---

Black-Scholes funciona como caso de control. Precio, Delta, Vega e inversión IV son cerrados o numéricamente estables, así que cualquier fallo ahí señala un problema del pipeline.

Heston es el caso principal. El precio se calcula con una formulación semi-cerrada de Fourier:

```text
C = S e^{-qT} P1 - K e^{-rT} P2.
```

La Delta de una call europea bajo esta formulación es

```text
Delta = e^{-qT} P1.
```

Como fijamos `q = 0`, la Delta usada como target diferencial en E5 es `P1`. La implementación se valida contra diferencias finitas centrales y contra QuantLib antes de generar datasets grandes. La inversión IV se usa solo en evaluación y debe reportar fallos de convergencia o precios fuera de límites de arbitraje.

## Dominio y muestreo

---

El dominio es amplio y orientado a trading. Incluye opciones semanales, wings profundas y volatilidades de estrés, porque esas son las regiones donde precio, IV y Delta pueden contar historias distintas.

| Variable | Rango | Uso |
|---|---|---|
| `m = S/K` | `[0.4, 2.0]` | Black-Scholes y Heston |
| `T` | `[7/365, 2.0]` | Black-Scholes y Heston |
| `r` | `[0.00, 0.075]` | Black-Scholes y Heston |
| `q` | `0` | Black-Scholes y Heston |
| `sigma` | `[0.03, 1.00]` | Black-Scholes |

Para Heston se añade el hipercubo de parámetros:

| Variable Heston | Rango | Interpretación |
|---|---|---|
| `v0` | `[0.0009, 1.00]` | `sqrt(v0)` entre 3% y 100% |
| `theta` | `[0.0009, 1.00]` | `sqrt(theta)` entre 3% y 100% |
| `kappa` | `[0.10, 10.00]` | Reversión lenta a rápida |
| `xi` | `[0.10, 3.00]` | Volatilidad de la varianza moderada a extrema |
| `rho` | `[-0.95, -0.05]` | Skew negativo típico de equity |

No imponemos la condición de Feller `2 kappa theta >= xi^2` como restricción dura. Muchas calibraciones reales la violan, así que se registra como diagnóstico y solo se descartan puntos con fallos numéricos claros o precios fuera de cotas de no arbitraje.

El muestreo baseline es uniforme en la escala financiera declarada. En Heston, `sqrt(v0)` y `sqrt(theta)` se muestrean uniformemente en `[0.03, 1.00]` y después se elevan al cuadrado, para evitar que una uniforme directa en varianza sobrepondere volatilidades extremas.

El sampler enfocado de H-5 es una mezcla `50/50`. La mitad de las muestras sigue el baseline uniforme completo. La otra mitad usa:

```text
m ~ TruncNormal(1.0, 0.15, [0.7, 1.3])
T ~ LogUniform(7/365, 0.25)
```

El resto de variables se muestrea como en el baseline. Así E3 concentra datos en ATM y vencimientos cortos sin abandonar la cobertura global.

## Entrenamiento

---

| Parámetro | Valor |
|---|---|
| Arquitectura | MLP, 4 capas ocultas, 128 unidades |
| Activación baseline | Swish |
| Optimizador | Adam, `lr=1e-3` fijo |
| Épocas | 100 |
| Batch size operativo | 32768 en modelos grandes, 16384 en small |
| Regularización | ninguna |
| Early stopping | no; se conserva el mejor checkpoint por validación |
| Selección de checkpoint | menor `MAE(C/K)` en validación |

No usamos scheduler, dropout, weight decay ni early stopping. Los datos son sintéticos y deterministas; el objetivo es estudiar el efecto de las variables experimentales, no añadir otra capa de sintonización. La selección por `MAE(C/K)` se mantiene también para H-6-small aunque su pérdida incluya Delta, de forma que E5 cambie la información usada durante el entrenamiento y no el criterio de selección del modelo final.

La pérdida de H-6-small es

```text
L = MAE(C/K) + MAE(Delta).
```

Los pesos 1:1 son deliberadamente simples: `C/K` y Delta viven en escalas comparables, y E5 debe medir el efecto de añadir información diferencial, no el efecto de ajustar pesos de loss.

## Evaluación por bins

---

La evaluación por bins es obligatoria porque el error global puede ocultar fallos sistemáticos en regiones operativamente importantes. La partición es `5 x 5`:

| Moneyness | Vencimiento |
|---|---|
| Deep OTM `[0.4, 0.7)` | Weekly `[7/365, 14/365)` |
| OTM `[0.7, 0.9)` | Short `[14/365, 1/12)` |
| ATM `[0.9, 1.1)` | Medium-short `[1/12, 0.25)` |
| ITM `[1.1, 1.3)` | Medium `[0.25, 1.0)` |
| Deep ITM `[1.3, 2.0]` | Long `[1.0, 2.0]` |

Cada bin incluye su extremo inferior y excluye su extremo superior, salvo el último bin de cada eje, que incluye también el extremo superior del dominio. Así, `m = 1.1` cae en ITM y `m = 2.0` cae en deep ITM.

Cada test set tiene 5k opciones por bin, 25 bins y 125k puntos en total. Se reportan medias y percentiles altos por bin. En E1 y E3 se reportan también fallos de inversión IV; en E5 y E6 no se invoca IV porque sus preguntas se centran en Delta y precio.

## Criterios de validez

---

Una comparación solo es válida si cambia una dimensión relevante. En E2 cambia la activación. En E3 cambia el sampler. En E5 cambia la pérdida y se controlan la muestra small final y la semilla de entrenamiento. Cuando una corrida preliminar rompe esa regla, se documenta y se corrige antes de usarla como resultado final.

El proyecto no intenta calibrar a datos reales de mercado, estudiar opciones exóticas ni construir un surrogate industrial a escala de Chen et al. El objetivo es demostrar, en un entorno controlado y reproducible, cuándo una red puede aproximar un solver de pricing con precisión, rapidez y derivadas útiles.
