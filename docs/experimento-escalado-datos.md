# Experimento de escalado de datos

**Autores:** Ángel Fernández Sánchez, Jorge Alfageme Sotillos, Héctor Pérez Ledesma

## Motivación

---

Tras cerrar la primera ronda de entrenamiento de los once surrogates definidos en `tasks.md` sobre los datasets baseline (200k para Black-Scholes y 500k para Heston), observamos que todos ellos convergían a un plateau de `MAE(C/K)` en validación del orden de `1e-2`, sin acercarse al criterio de salida de Fase 1 que exige `MAE_precio < 1e-4` para BS-3. El estancamiento aparecía además en las últimas veinte épocas como oscilación en torno al mínimo y no como descenso continuo, sugiriendo que el entrenamiento sí había encontrado un óptimo local pero ese óptimo estaba lejos del objetivo nominal del proyecto.

Antes de cuestionar la arquitectura acordada (MLP 4x128, decisión cerrada en las constantes del pipeline) o cambiar el resto de hiperparámetros, conviene aislar primero la fuente del plateau. La hipótesis natural a contrastar es la de aproximación universal: con una MLP de cincuenta mil parámetros y una función de pricing tan suave como la de Heston, ¿está saturando la capacidad de la red o simplemente le faltan datos? El paper de Chen, Didisheim y Scheidegger entrena su deep surrogate con mil millones de muestras en un espacio de trece dimensiones; nuestro baseline operaba en un régimen aproximado de diez muestras por parámetro, mientras que ellos están en torno a las mil. Esa diferencia de dos órdenes de magnitud justifica plantear un experimento de escalado de datos como diagnóstico previo a cualquier modificación del modelo.

El experimento consiste en regenerar todos los datasets multiplicando por cincuenta el número de muestras, reentrenar los once surrogates manteniendo idénticos arquitectura, optimizador, función de pérdida, número de épocas y semilla, y comparar el `MAE(C/K)` resultante con el baseline. Si el plateau es data-bound, esperamos ver una reducción sustancial del error. Si la red ya estaba saturada, esperamos ver una reducción marginal o nula, y entonces tendría sentido abrir la conversación con el grupo para escalar arquitectura. El experimento se diseña precisamente para que el resultado sea binario y accionable.

## Configuración del experimento

---

Se mantuvieron todos los elementos del pipeline acordado salvo el tamaño de los datasets. Las semillas se preservaron por surrogate (101-104 para BS, 201-207 para Heston) para que cualquier diferencia con el baseline sea atribuible al volumen de datos y no a un sorteo distinto. El batch size en el orquestador paralelo es de 32768 para BS y los Heston grandes, 16384 para los smalls, manteniendo la decisión previa del orquestador `max`. La arquitectura sigue siendo MLP 4x128 con Swish por defecto, Adam con `lr=1e-3`, sin scheduler ni regularización, cien épocas, selección de checkpoint por menor `MAE(C/K)` en validación.

| Dataset | Baseline (500k) | Escalado (50x) | Factor |
|---|---|---|---|
| BS train | 200k uniforme + Delta | 10M uniforme + Delta | 50x |
| BS validación | 50k uniforme + Delta | 2.5M uniforme + Delta | 50x |
| BS test balanced | 125k (5k por bin) | 6.25M (250k por bin) | 50x |
| Heston train uniforme | 500k | 25M | 50x |
| Heston train focused | 500k (E3) | 25M (E3) | 50x |
| Heston small uniforme | 100k (E5) | 5M (E5) | 50x |
| Heston small uniforme + Delta | 100k (E5) | 5M (E5) | 50x |
| Heston validación | 50k uniforme | 2.5M uniforme | 50x |
| Heston test balanced + Delta | 125k (5k por bin) | 6.25M (250k por bin) | 50x |

La generación de los 87.5M de muestras llevó 8h 20min con paralelismo de 32 workers en CPU (un i9 14900K) y 5.16 GB de disco. La tasa global de rechazo fue 0.0186%, indistinguible de la del baseline. El entrenamiento paralelo posterior de los once surrogates llevó 63.3 minutos con `--parallel 2 --preload-to-device` en una RTX 4060 de 8 GB de VRAM, limitado a dos trainings simultáneos por el coste de mantener los Heston-25M cargados íntegros en memoria de GPU.

## Resultados cuantitativos

---

El plateau es **data-bound**. Todos los surrogates mejoraron entre un 37% y un 91% en `MAE(C/K)` de validación al pasar al régimen 50x. El detalle por surrogate es:

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

El surrogate H-5, que combina Swish con muestreo enfocado en zonas de alta curvatura, es el primero en bajar de `1e-3` en `MAE(C/K)` global. H-3, el baseline central de los experimentos E1, E2, E4 y E5, se sitúa muy cerca, en `1.045e-03`. El criterio nominal de Fase 1 (`MAE_precio < 1e-4`) sigue sin cumplirse, pero la distancia se ha reducido en un orden de magnitud completo: estamos a un factor diez del objetivo, no a un factor cien como en el baseline.

Las curvas de entrenamiento muestran además que varios surrogates seguían descendiendo al final de las cien épocas. En H-1 la pendiente de las últimas veinte épocas era de un 98% de reducción del `MAE(C/K)` entre el primer y el último valor del bloque, en H-5 de un 56%, en H-6-small de un 21%. Esto sugiere que ampliar el presupuesto de épocas o introducir un scheduler de learning rate podría obtener mejoras adicionales sin tocar arquitectura ni dataset.

## Lectura por experimento

---

### E2 — Activaciones

El cambio más llamativo es el reordenamiento del ranking de activaciones. En el baseline 500k, la ReLU producía el mejor `MAE` de precio tanto en Black-Scholes como en Heston, contradiciendo la hipótesis del paper de Chen de que las activaciones suaves son superiores. En el régimen 50x el orden se invierte y se alinea con lo predicho por Chen:

| Posición | BS (10M) | Heston (25M) |
|---|---|---|
| Mejor | Swish (1.104e-03) | Swish (1.045e-03) |
| Segunda | tanh (1.789e-03) | tanh (1.170e-03) |
| Tercera | ReLU (2.542e-03) | ReLU (1.481e-03) |
| Peor | Softplus (4.123e-03) | Softplus (2.833e-03) |

La interpretación natural es que ReLU aprende rápido cuando hay pocos datos porque su no linealidad por tramos encaja superficialmente bien con el kink del payoff, pero su discontinuidad en la derivada acaba siendo un techo cuando el dataset es lo bastante grande como para que la red pueda aprovecharse de una activación realmente suave. Esto deja parcialmente cuestionada la lectura preliminar que hicimos de los baseline, en la que aparentemente ReLU dominaba; era un efecto del régimen de pocos datos. El experimento E2 sigue requiriendo `MAE_Delta` por bin como métrica primaria para concluir formalmente, pero al menos en `MAE` de precio la hipótesis Chen ya se confirma.

### E3 — Muestreo uniforme frente a enfocado

H-5 (Swish, dataset focused 25M) se sitúa en `9.888e-04` frente a H-3 (Swish, dataset uniforme 25M) en `1.045e-03`. La ventaja global del muestreo enfocado es del 5.4%, por debajo del umbral fuerte del 10% que `tasks.md` exige para concluir mejora positiva. Sin embargo este número se calcula sobre el conjunto de validación uniforme, que cubre todo el dominio incluyendo zonas en las que H-5 no entrenó con tanta densidad. Para concluir limpiamente sobre E3 hace falta la evaluación por bins, en particular sobre los bins ATM y de vencimiento corto donde se concentra el sampling enfocado. Es plausible que la mejora local en esos bins sea muy superior al 10% incluso si el promedio global queda por debajo.

### E5 — Differential Machine Learning

La comparación canónica de E5 enfrenta H-6-small (5M de muestras con precio y Delta, pérdida diferencial) con H-3-small (5M de muestras con solo precio) y con el baseline H-3 (25M de muestras con solo precio). En `MAE(C/K)`:

| Surrogate | Dataset | Pérdida | `MAE(C/K)` |
|---|---|---|---|
| H-3-small | 5M uniforme | precio | 1.858e-03 |
| H-6-small | 5M uniforme + Delta | precio + Delta | 1.169e-03 |
| H-3 | 25M uniforme | precio | 1.045e-03 |

A igualdad de datos, H-6-small reduce el `MAE` de precio en un 37% respecto a H-3-small, lo que confirma en este régimen la tesis central de Huge y Savine: añadir gradientes verdaderos a la pérdida actúa como una forma muy potente de data augmentation. Comparado con H-3, sin embargo, H-6-small queda ligeramente por detrás en precio, lo que era esperable: H-3 dispone de cinco veces más datos, y la ventaja de la información de Delta no llega a compensar del todo esa diferencia.

La métrica primaria de E5, no obstante, es el `MAE_Delta`. Es probable que en esa métrica H-6-small bata a H-3 con holgura, porque H-3 nunca vio Deltas verdaderas durante el entrenamiento y solo puede inferirlas por autograd implícitamente. Sin la evaluación de Greeks por bin no podemos cerrar la pregunta de E5 todavía, pero la señal preliminar es la esperada.

## Diagnóstico del plateau

---

El experimento responde de forma inequívoca a la pregunta diagnóstica que lo motivaba: el plateau a `1e-2` que observábamos en el baseline 500k era data-bound, no model-bound. La MLP 4x128 que el grupo había fijado como constante del pipeline absorbió un volumen de datos cincuenta veces mayor sin saturar su capacidad, reduciendo el error de validación entre un factor tres y un factor diez según el surrogate. Esto descarta la lectura intuitiva, que también barajamos, de que con cincuenta mil parámetros la red estaba aproximando todo lo que podía aproximar y el cuello era arquitectónico.

La consecuencia metodológica directa es que el tamaño de los datasets baseline definidos en `tasks.md` (200k para BS, 500k para Heston) probablemente está infradimensionado para los objetivos de precisión que el propio documento se fija. Es razonable interpretar que el criterio `MAE_precio < 1e-4` de Fase 1 se escribió como una aspiración alineada con los números que aparecen en el paper de Chen, pero los baselines elegidos quedaron en una escala muy alejada de las condiciones que en ese paper hacían posible alcanzar dicha precisión. Llegar al `1e-4` parece posible sin tocar arquitectura, basta con escalar lo suficiente la cantidad de datos, aunque el factor cincuenta no es todavía suficiente y queda abierto a cuánto más habría que escalar.

Conviene además registrar que varios surrogates seguían descendiendo en `MAE` de validación al cerrar las cien épocas, lo cual implica que parte de la mejora pendiente podría obtenerse simplemente entrenando más tiempo o con un scheduler de learning rate, dos modificaciones que se mantienen dentro del pipeline acordado y no requerirían renegociar las constantes. Una versión más exigente del experimento podría reentrenar los mismos surrogates 50x con doscientas o trescientas épocas y un decay coseno para ver cuánto del techo restante se mueve.

## Implicaciones para el plan acordado

---

Este experimento abre tres temas que el grupo debería discutir antes de la siguiente fase del trabajo. El primero es si formalizar el régimen 50x como nuevo baseline del proyecto, sustituyendo los tamaños de `tasks.md`. La ventaja es obvia: todos los experimentos E1 a E5 se sostienen mejor con surrogates más precisos, las conclusiones serán más robustas frente al ruido de aproximación, y nos acercamos al rango de errores que tiene sentido reportar en la memoria final. El coste, ocho horas de generación y una hora de entrenamiento, se ha demostrado asumible una sola vez.

El segundo tema es si el criterio numérico de Fase 1 (`MAE_precio < 1e-4`) debe ajustarse al rango realmente alcanzable, situarse en `MAE_precio < 1e-3` o conservarse y considerarse no cumplido durante la primera entrega, dejando abierta la opción de seguir escalando datos o capacidad. La elección debería basarse en qué nivel de precisión necesitamos efectivamente para que las conclusiones de E1 a E5 sean cualitativamente interpretables, y no en una cifra heredada de una primera estimación.

El tercer tema es de orden expositivo y aprovecha que ya tenemos resultados comparables en dos regímenes. Mostrar la curva `MAE` vs tamaño de dataset es exactamente el tipo de figura que las secciones de métodos numéricos suelen pedir: una validación empírica de que el error decrece de forma controlada y predecible al aumentar las muestras, lo cual conecta con los resultados teóricos de Barron y Grohs que citamos en la revisión bibliográfica. Esta figura puede entrar en la memoria final como uno de los hallazgos transversales del trabajo, independiente de los cinco experimentos formales.

## Trabajo pendiente

---

Los resultados de este experimento son preliminares en el sentido de que la métrica reportada es `MAE(C/K)` global, mientras que cuatro de los cinco experimentos formales (E1, E2, E3, E5) exigen métricas distintas o evaluación por bins. Antes de extraer conclusiones definitivas hay que implementar el módulo `binning.py` y el módulo `metrics.py` que la Fase 2 prevé, para poder calcular `MAE_Delta` y `MAE_IV` por bin sobre los checkpoints ya entrenados. Esta es la tarea inmediatamente siguiente.

El experimento E4 (medición de speedup) no se ve afectado por el escalado de datos porque mide el tiempo de inferencia, no el de entrenamiento. Su implementación depende del módulo `timing.py` previsto para la Fase 4 y puede realizarse sobre el checkpoint actual de H-3 sin esperar a más entrenamientos.

Por último, en función de lo que decida el grupo sobre el primer tema de la sección anterior, queda abierta la posibilidad de un experimento posterior con escalado adicional (por ejemplo 500x o 1000x) o con LR scheduler y más épocas sobre los datasets 50x ya generados.

Los datasets, checkpoints, logs de entrenamiento y resumen cuantitativo de esta ronda viven respectivamente en `data/`, `results/checkpoints/`, `results/logs/` y `results/metrics/data_scaling_summary.csv`. Solo el último se versiona; el resto queda gitignored por convención de proyecto y es reproducible desde semilla mediante los scripts `scripts/generate_all_50x_progress.py` y `scripts/train_all_parallel_50x.py`.
