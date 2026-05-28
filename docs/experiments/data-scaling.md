# Data scaling

**Autores:** Ángel Fernández Sánchez, Jorge Alfageme Sotillos, Héctor Pérez Ledesma

## Nota de alcance

---

Este documento se conserva como registro diagnóstico del run de escalado. No sustituye al protocolo formal de evaluación de E1-E5, que sigue usando los tests balanceados de 125k puntos definidos en `metodologia.md`. Los artefactos específicos del run 50x no forman parte de la versión final del repositorio; las cifras relevantes quedan resumidas aquí para justificar las decisiones tomadas durante el proyecto.

## Motivación

---

Tras cerrar la primera ronda de entrenamiento de los once surrogates definidos en la metodología sobre los datasets baseline (200k para Black-Scholes y 500k para Heston), observamos que todos ellos se estabilizaban con un `MAE(C/K)` de validación del orden de `1e-2`, lejos del objetivo nominal que esperábamos para BS-3. En las últimas veinte épocas el error oscilaba en torno al mínimo en lugar de descender de forma sostenida, sugiriendo que el entrenamiento había encontrado una región estable pero todavía lejos del nivel de precisión buscado.

Antes de cuestionar la arquitectura acordada (MLP 4x128, decisión cerrada en las constantes del pipeline) o cambiar el resto de hiperparámetros, convenía aislar primero la fuente de ese estancamiento. La hipótesis natural era la de aproximación universal: con una MLP de cincuenta mil parámetros y una función de pricing tan suave como la de Heston, ¿estaba saturando la capacidad de la red o simplemente faltaban datos? El paper de Chen, Didisheim y Scheidegger entrena su deep surrogate con mil millones de muestras en un espacio de trece dimensiones; nuestro baseline operaba en un régimen aproximado de diez muestras por parámetro, mientras que ellos están en torno a las mil. Esa diferencia de dos órdenes de magnitud justificó plantear un experimento de escalado de datos como diagnóstico previo a cualquier modificación del modelo.

El experimento consistió en regenerar los datasets de entrenamiento y validación multiplicando por cincuenta el número de muestras, reentrenar los once surrogates manteniendo idénticos arquitectura, optimizador, función de pérdida, número de épocas y semilla de cada surrogate, y comparar el `MAE(C/K)` resultante con el baseline. Si el límite observado venía principalmente del volumen de datos, esperábamos ver una reducción sustancial del error. Si la red ya estaba saturada, esperábamos ver una reducción marginal o nula, y entonces tendría sentido abrir la conversación con el grupo para escalar arquitectura.

## Configuración del experimento

---

Se mantuvieron todos los elementos del pipeline acordado salvo el tamaño de los datasets. Las semillas se preservaron por surrogate (101-104 para BS, 201-207 para Heston) para que cualquier diferencia con el baseline fuese atribuible al volumen de datos y no a un sorteo distinto. El batch size en el orquestador paralelo fue de 32768 para BS y los Heston grandes, 16384 para los smalls. La arquitectura siguió siendo MLP 4x128 con Swish por defecto, Adam con `lr=1e-3`, sin scheduler ni regularización, cien épocas, selección de checkpoint por menor `MAE(C/K)` en validación.

| Dataset | Baseline | Escalado diagnóstico | Factor |
|---|---|---|---|
| BS train | 200k uniforme + Delta | 10M uniforme + Delta | 50x |
| BS validación | 50k uniforme + Delta | 2.5M uniforme + Delta | 50x |
| BS test balanced | 125k (5k por bin) | generado solo como diagnóstico | -- |
| Heston train uniforme | 500k | 25M | 50x |
| Heston train focused | 500k (E3) | 25M (E3) | 50x |
| Heston small uniforme | 100k (E5) | 5M (E5) | 50x |
| Heston small uniforme + Delta | 100k (E5) | 5M (E5) | 50x |
| Heston validación | 50k uniforme | 2.5M uniforme | 50x |
| Heston test balanced + Delta | 125k (5k por bin) | generado solo como diagnóstico | -- |

La generación de los datasets escalados llevó 8h 20min con paralelismo de 32 workers en CPU (un i9 14900K) y 5.16 GB de disco. La tasa global de rechazo fue 0.0186%, indistinguible de la del baseline. El entrenamiento paralelo posterior de los once surrogates llevó 63.3 minutos con `--parallel 2 --preload-to-device` en una RTX 4060 de 8 GB de VRAM, limitado a dos trainings simultáneos por el coste de mantener los Heston-25M cargados íntegros en memoria de GPU.

## Resultados cuantitativos

---

Todos los surrogates mejoraron entre un 37% y un 91% en `MAE(C/K)` de validación al pasar al régimen 50x. El detalle por surrogate fue:

| ID | Activación | Pérdida | Baseline `MAE(C/K)` | Escalado 50x | Reducción |
|---|---|---|---|---|---|
| BS-1 | ReLU | precio | 4.076e-03 | 2.542e-03 | -37.6% |
| BS-2 | Softplus | precio | 3.599e-02 | 4.123e-03 | -88.5% |
| BS-3 | Swish | precio | 9.879e-03 | **1.104e-03** | **-88.8%** |
| BS-4 | tanh | precio | 1.372e-02 | 1.789e-03 | -87.0% |
| H-1 | ReLU | precio | 4.725e-03 | 1.481e-03 | -68.6% |
| H-2 | Softplus | precio | 2.590e-02 | 2.833e-03 | -89.1% |
| H-3 | Swish | precio | 1.154e-02 | **1.045e-03** | **-90.9%** |
| H-4 | tanh | precio | 1.147e-02 | 1.170e-03 | -89.8% |
| H-5 | Swish (focused) | precio | 1.112e-02 | **9.888e-04** | **-91.1%** |
| H-3-small | Swish | precio | 1.273e-02 | 1.858e-03 | -85.4% |
| H-6-small | Swish | diferencial | 1.094e-02 | 1.169e-03 | -89.3% |

El surrogate H-5, que combina Swish con muestreo enfocado en zonas de alta curvatura, fue el primero en bajar de `1e-3` en `MAE(C/K)` global. H-3, el baseline central de los experimentos E1, E2, E4 y E5, se situó muy cerca, en `1.045e-03`. El criterio nominal de Fase 1 (`MAE_precio < 1e-4`) siguió sin cumplirse, pero la distancia se redujo en un orden de magnitud completo: pasamos de estar a un factor cien del objetivo a estar aproximadamente a un factor diez.

Las curvas de entrenamiento mostraron además que varios surrogates seguían descendiendo al final de las cien épocas. En H-1 la pendiente de las últimas veinte épocas era de un 98% de reducción del `MAE(C/K)` entre el primer y el último valor del bloque, en H-5 de un 56%, en H-6-small de un 21%. Esto sugiere que ampliar el presupuesto de épocas o introducir un scheduler de learning rate podría obtener mejoras adicionales sin tocar arquitectura ni dataset, aunque esa modificación no forma parte del protocolo actual.

## Lectura por experimento

---

### E2 — Activaciones

El cambio más llamativo fue el reordenamiento del ranking de activaciones en precio. En el baseline 500k, ReLU producía el mejor `MAE` de precio tanto en Black-Scholes como en Heston, contradiciendo la hipótesis del paper de Chen de que las activaciones suaves son superiores. En el régimen 50x el orden se invirtió y se alineó con lo predicho por Chen:

| Posición | BS (10M) | Heston (25M) |
|---|---|---|
| Mejor | Swish (1.104e-03) | Swish (1.045e-03) |
| Segunda | tanh (1.789e-03) | tanh (1.170e-03) |
| Tercera | ReLU (2.542e-03) | ReLU (1.481e-03) |
| Peor | Softplus (4.123e-03) | Softplus (2.833e-03) |

La interpretación natural es que ReLU aprende rápido cuando hay pocos datos porque su no linealidad por tramos encaja superficialmente bien con el kink del payoff, pero su discontinuidad en la derivada acaba siendo un techo cuando el dataset es lo bastante grande como para que la red pueda aprovechar una activación realmente suave. El E2 formal, sin embargo, se cierra con `MAE_Delta` por bin sobre el test balanceado de 125k, no con esta métrica global de validación.

### E3 — Muestreo uniforme frente a enfocado

H-5 (Swish, dataset focused 25M) se situó en `9.888e-04` frente a H-3 (Swish, dataset uniforme 25M) en `1.045e-03`. La ventaja global del muestreo enfocado en validación uniforme fue del 5.4%. Este número era solo una señal preliminar: la conclusión formal de E3 debía hacerse por bins, especialmente sobre los bins ATM y de vencimiento corto donde se concentra el sampling enfocado. Esa evaluación formal se reporta en `docs/experiments/e3.md`.

### E5 — Differential Machine Learning

La comparación canónica de E5 enfrenta H-6-small (5M de muestras con precio y Delta, pérdida diferencial) con H-3-small (5M de muestras con solo precio) y con el baseline H-3 (25M de muestras con solo precio). En `MAE(C/K)`:

| Surrogate | Dataset | Pérdida | `MAE(C/K)` |
|---|---|---|---|
| H-3-small | 5M uniforme | precio | 1.858e-03 |
| H-6-small | 5M uniforme + Delta | precio + Delta | 1.169e-03 |
| H-3 | 25M uniforme | precio | 1.045e-03 |

A igualdad de datos, H-6-small redujo el `MAE` de precio en un 37% respecto a H-3-small, lo que confirma en este régimen la tesis central de Huge y Savine: añadir gradientes verdaderos a la pérdida actúa como una forma potente de data augmentation. La métrica primaria de E5, no obstante, es `MAE_Delta`; por tanto esta lectura es preliminar y no reemplaza el experimento formal.
