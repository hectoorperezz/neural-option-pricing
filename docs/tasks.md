# Tareas y experimentos

**Autores:** Ángel Fernández Sánchez, Jorge Alfageme Sotillos, Héctor Pérez Ledesma

## Visión

---

Este documento concreta los experimentos que vamos a ejecutar y las tareas necesarias para llegar a ellos. Es la referencia única para saber qué se entrena, qué se mide y qué se entrega en cada fase. Cada decisión técnica relevante va acompañada de su justificación porque, en un proyecto donde casi todo lo que decidimos hoy condiciona el código que escribimos mañana, conviene tener por escrito el porqué de cada elección para no reabrir debates a mitad de camino. Está deliberadamente acotado: cada surrogate aparece una sola vez en el plan y cada experimento se sostiene sobre comparaciones limpias entre los surrogates entrenados.

## Experimentos

---

Cinco experimentos cubren los cinco elementos diferenciales del trabajo. La elección de exactamente cinco no es arbitraria, es la lectura directa del bloque de elementos diferenciales recogidos en `primera-entrega.md`, traducido a comparaciones experimentales concretas. Cada experimento responde a una pregunta única, varía una sola dimensión y mantiene todo lo demás fijo. Esta rigidez es lo que permite atribuir cualquier diferencia observada a la dimensión que estamos variando y no a otras variables que se nos hayan colado por accidente.

| ID | Pregunta | Lo que varía | Surrogates implicados | Resultado esperado |
|---|---|---|---|---|
| **E1** | ¿Un error bajo en precio implica un error bajo en volatilidad implícita? | Métrica de evaluación | BS-3, H-3 | Bins donde el MAE de precio es bajo pero el MAE de IV no lo es. Si aparecen, la elección de métrica importa. |
| **E2** | ¿Qué función de activación produce mejores Greeks sin degradar el precio? | Activación (ReLU, Softplus, Swish, tanh) | BS-1..4, H-1..4 | MAE de precio similar entre todas las activaciones, MAE de Delta sustancialmente mayor en ReLU. |
| **E3** | ¿El muestreo enfocado en zonas de mayor curvatura reduce el error en bins críticos? | Distribución del muestreo (uniforme vs enfocado) | H-3 vs H-5 | H-5 con menor MAE en bins ATM y vencimiento corto, posiblemente peor en bins extremos. |
| **E4** | ¿Qué speedup ofrece el surrogate frente al solver original? | Tamaño del lote de evaluación | H-3 | Speedup creciente con el tamaño del lote, especialmente notable en lotes grandes. |
| **E5** | ¿Entrenar con Delta como target diferencial mejora la eficiencia muestral del surrogate? | Targets de entrenamiento (precio vs precio + Delta) | H-3-small, H-6-small, H-3 | H-6-small con mejor Delta y precisión de precio comparable o superior a H-3-small; idealmente se acerca a H-3 usando menos datos. |

E1 es observacional. No requiere entrenar nada nuevo, simplemente recalcula métricas sobre un surrogate que ya está entrenado para otros experimentos. La razón por la que merece llamarse experimento aparte es que la pregunta que responde es metodológica: nos dice si dos surrogates con MAE de precio aparentemente similares pueden tener errores muy distintos en la métrica que de verdad importa en el mercado, que es la volatilidad implícita. Si la respuesta es afirmativa, la elección de métrica de validación deja de ser una decisión técnica menor y pasa a ser parte del diseño del proyecto.

E2 es el experimento más cargado en surrogates porque la activación es la única decisión arquitectónica que afecta directamente la calidad de las derivadas. Entrenar las cuatro activaciones sobre Black-Scholes nos sirve como control en un entorno donde conocemos la solución exacta y las Deltas analíticas, y entrenarlas sobre Heston confirma o desmiente el patrón en el caso real. La hipótesis, alineada con Chen et al., es que las activaciones suaves dan Deltas mucho mejores que ReLU aunque el precio sea parecido, y queremos comprobar si esa diferencia es robusta entre ambos modelos.

E3 ataca una idea clásica de métodos numéricos transferida al dominio de los surrogates: concentrar el esfuerzo computacional en las regiones donde la función es más difícil de aproximar. La función de pricing tiene más curvatura cerca del at-the-money y en vencimientos cortos, así que sembrar más densidad de muestreo ahí debería traducirse en errores más bajos en esos bins. La comparación es limpia porque H-3 y H-5 difieren solo en la distribución de muestreo: misma arquitectura, mismo número total de puntos, misma pérdida y mismo tamaño total del dataset.

E4 es de medición, no de comparación. Cronometra evaluaciones de H-3 contra el solver original de Heston en lotes de tamaño creciente. La razón para hacerlo con lotes y no con evaluaciones unitarias es que las redes amortiguan costes cuando procesan muchos puntos en paralelo, así que el speedup real solo se ve a partir de cierto tamaño. Reportar el speedup como función del tamaño de lote es más honesto que dar un único número que dependería arbitrariamente del lote elegido.

E5 prueba la hipótesis central del paper de Huge y Savine en nuestro setting de forma acotada: si además de precios generamos Delta verdadera, ¿puede el surrogate aprender mejor la forma local de la función de pricing con menos datos? Delta es la Greek más limpia para este primer experimento porque, usando `y = C/K` y `m = S/K`, se cumple `Delta = ∂y/∂m`. Vega, Gamma, Theta y Rho quedan como métricas diagnósticas o extensiones posteriores, pero no forman parte de la pérdida principal de E5. Las sensibilidades respecto a parámetros internos de Heston (`kappa`, `theta`, `xi`, `rho`) quedan fuera del alcance principal porque son riesgos de modelo/calibración, no Greeks operativas de trading.

Cada experimento tendrá una métrica primaria alineada con su pregunta y métricas secundarias como control. Esto evita que el resultado se reduzca a una tabla grande sin conclusión clara.

| ID | Métrica primaria | Controles y diagnóstico |
|---|---|---|
| E1 | Discrepancia entre `MAE(C/K)` y `MAE_IV` por bin | Diagnóstico sin umbral fuerte/débil; Vega/proxy, percentiles altos y fallos de IV |
| E2 | `MAE_Delta` por bin | Sin umbral fuerte/débil; se evalúa robustez de activaciones suaves vs ReLU y `MAE(C/K)` como control |
| E3 | `MAE_IV` en bins ATM con weekly, short y medium-short | Positivo fuerte si mejora >=10% y `MAE_IV` global empeora <=10%; `MAE(C/K)` como control |
| E4 | `speedup = tiempo_solver / tiempo_surrogate` por tamaño de lote | Lotes `10^2`..`10^5`, 3 warmups, 10 repeticiones, mediana y CPU/GPU separado |
| E5 | Mejora de `MAE_Delta` de H-6-small frente a H-3-small | Positivo fuerte si Delta mejora >=20% y precio empeora <=10%; distancia frente a H-3 |

## Surrogates a entrenar

---

Once surrogates en total. La cifra sale de trabajar hacia atrás desde los cinco experimentos: cada uno necesita un conjunto mínimo de surrogates entrenados para sostener su comparación, y muchos de esos surrogates se solapan entre experimentos. H-3 en particular, el surrogate de Heston con configuración baseline, aparece como referencia en los cinco experimentos. Esa centralidad evita que el coste computacional total se dispare y justifica que dediquemos especial cuidado a su entrenamiento.

| ID | Modelo | Activación | Pérdida | Dataset | Aporta a |
|---|---|---|---|---|---|
| BS-1 | Black-Scholes | ReLU | L1(precio) | 200k uniforme | E2 |
| BS-2 | Black-Scholes | Softplus | L1(precio) | 200k uniforme | E2 |
| **BS-3** | **Black-Scholes** | **Swish** | **L1(precio)** | **200k uniforme** | **E1, E2** |
| BS-4 | Black-Scholes | tanh | L1(precio) | 200k uniforme | E2 |
| H-1 | Heston | ReLU | L1(precio) | 500k uniforme | E2 |
| H-2 | Heston | Softplus | L1(precio) | 500k uniforme | E2 |
| **H-3** | **Heston** | **Swish** | **L1(precio)** | **500k uniforme** | **E1, E2, E3, E4, E5 baseline** |
| H-4 | Heston | tanh | L1(precio) | 500k uniforme | E2 |
| H-5 | Heston | Swish | L1(precio) | 500k **enfocado** | E3 |
| H-3-small | Heston | Swish | L1(precio) | 100k uniforme | E5 |
| H-6-small | Heston | Swish | L1(precio) + L1(Δ), pesos 1:1 | 100k uniforme | E5 |

Todos los surrogates comparados dentro de un mismo experimento comparten arquitectura, optimizador, hiperparámetros de entrenamiento y semilla. Esta rigidez es deliberada y es lo que permite atribuir las diferencias observadas a la dimensión que cada experimento varía. En E5 sí varía deliberadamente el tamaño del dataset porque la pregunta no es solo si DML mejora las Greeks, sino si permite alcanzar precisión comparable con menos muestras. Si dejásemos variar la arquitectura entre surrogates, la comparación de activaciones quedaría contaminada con el efecto del cambio arquitectónico, por mencionar el caso más obvio.

## Constantes del pipeline

---

Para evitar reabrir discusiones durante la ejecución, fijamos los valores siguientes en código y no los movemos salvo que un hallazgo experimental lo justifique. Cada uno tiene una razón concreta que documentamos a continuación para que el porqué no se pierda con el tiempo.

| Parámetro | Valor |
|---|---|
| Arquitectura | MLP, 4 capas ocultas, 128 unidades |
| Activación por defecto | Swish |
| Optimizador | Adam, `lr=1e-3`, scheduler reduce-on-plateau |
| Batch size | 1024 |
| Épocas máximas | 100 |
| Regularización | ninguna (datos sintéticos) |
| Moneyness | `m = S/K` |
| Normalización de inputs | min-max a `[0, 1]` |
| Output | precio normalizado `C/K` |
| Dividend yield | `q = 0` |
| Recorte de bordes | 5% en cada cara del hipercubo |
| Semilla | fijada por configuración |

Elegimos un MLP feed-forward simple porque el teorema universal de aproximación garantiza que basta para funciones suaves en dimensión moderada como las nuestras, y Chen et al. confirman empíricamente que funciona en Bates 13D. Della Corte y coautores comparan arquitecturas más sofisticadas como highway o variantes de DGM y encuentran mejoras marginales a cambio de un coste de implementación y depuración que no se justifica para un proyecto de este alcance. Cuatro capas ocultas y 128 unidades por capa dan del orden de cincuenta mil parámetros, lo que para datasets de cien mil a quinientas mil muestras supone entre dos y diez muestras por parámetro, una ratio cómoda. Chen usa redes más profundas pero su problema tiene trece dimensiones y mil millones de muestras; nosotros estamos en una escala menor y escalamos abajo de forma proporcional.

Swish entra como activación de los surrogates baseline porque, según Chen, es la decisión clave para que las Deltas obtenidas por autograd sean estables. ReLU tiene un kink en cero, su derivada es discontinua y eso ensucia las Deltas que la red produce. Swish, en cambio, es suave en todo el dominio, se comporta como ReLU para `x` grande y como una transición suave cerca del origen. Softplus y tanh son también suaves pero tienen problemas de saturación o de gradiente vanishing en redes profundas que las hacen menos atractivas como defaults. Como E2 compara las cuatro de forma sistemática, usar Swish en el resto de surrogates como baseline tiene sentido porque es nuestra hipótesis de mejor opción y queremos que los experimentos no relacionados con la activación se hagan sobre el candidato más fuerte.

Adam con `lr=1e-3` es el default de PyTorch y converge sin sintonización en la inmensa mayoría de problemas de regresión. Alternativas como L-BFGS pueden funcionar mejor en aproximación supervisada pura pero son menos robustas en redes profundas y requieren más cuidado al sintonizarlas. Batch size de 1024 es potencia de dos para alineación de memoria en GPU y da entre 195 y 488 updates por época según el tamaño del dataset, suficiente para que el optimizador progrese sin que el ruido del gradiente domine la dinámica. El scheduler reduce-on-plateau es lo más simple que funciona: cuando la pérdida de validación se estanca, divide el learning rate por dos. No requiere conocer a priori cuándo decaer, a diferencia de cosine decay o step schedulers que requieren un cronograma fijado de antemano.

La ausencia de regularización es la decisión metodológica más importante y viene directamente de Chen. En aprendizaje automático clásico la regularización existe porque los datos llevan ruido y la red tendería a memorizarlo, lo que produce overfitting. En un surrogate los targets vienen del propio solver, son deterministas y no hay nada que memorizar. La pérdida de validación nunca empeora por overfitting, solo se estanca cuando la red llega al límite de su capacidad expresiva. Por eso no usamos early stopping, dropout ni weight decay. El tope de cien épocas funciona como límite práctico para que el entrenamiento no se eternice si la pérdida deja de bajar antes de tiempo.

La selección de checkpoint será común para todos los surrogates: guardamos el estado con menor `MAE(C/K)` en validación. Esta regla se mantiene también para H-6-small, aunque su pérdida incluya Delta. La razón es experimental: E5 debe cambiar solo la información usada durante el entrenamiento, no el criterio con el que elegimos el modelo final. Si H-6-small se seleccionase con una métrica combinada de precio y Delta, no sabríamos si la mejora viene de la pérdida diferencial o de haber escogido otro punto de la trayectoria de entrenamiento. El test final sí reportará precio, IV y Delta por separado.

Min-max a `[0, 1]` es la normalización determinista natural cuando conocemos los límites del hipercubo de muestreo de antemano, que es nuestro caso. Z-score también funcionaría pero pierde la interpretabilidad de "estoy en el dominio". El output es precio normalizado por strike, `C/K`, en lugar de la volatilidad implícita que usa Chen. Esta es la única divergencia significativa con el paper de referencia y tiene una justificación experimental concreta: para que E1 (precio vs IV como métrica) tenga sentido, conviene entrenar en una de las dos escalas y evaluar en ambas. Si entrenásemos en IV, sesgaríamos la comparación a su favor. Adicionalmente, `C/K` es invariante bajo escalado simultáneo de spot y strike, lo que reduce la dimensionalidad efectiva, y evita el paso de inversión a IV durante la generación de datos, donde más fallos numéricos aparecen.

Por último, las redes aproximan peor cerca de la frontera del dominio porque hay menos vecinos a un lado para que la red aprenda la estructura local. Si evaluásemos en el dominio completo, los errores de frontera dominarían las métricas globales y enmascararían el comportamiento real en el interior. El cinco por ciento es el recorte que aplica Chen y que nosotros replicamos sin más discusión por consistencia y porque funciona.

## Dominio de entrenamiento

---

Usamos un dominio contractual amplio orientado a trading, no un dominio estrecho diseñado solo para que la red entrene fácil. La moneyness simple `m = S/K` permite separar de forma directa opciones deep OTM, OTM, ATM, ITM y deep ITM, y mantiene limpia la relación `Delta = d(C/K)/dm` que necesitamos en E5. Fijamos `q = 0` porque los dividendos no forman parte de las preguntas experimentales y porque esta simplificación reduce la dimensión del problema.

| Variable | Rango | Uso |
|---|---|---|
| `m = S/K` | `[0.4, 2.0]` | Black-Scholes y Heston |
| `T` | `[7/365, 2.0]` | Black-Scholes y Heston |
| `r` | `[0.00, 0.075]` | Black-Scholes y Heston |
| `q` | `0` | Black-Scholes y Heston |
| `sigma` | `[0.03, 1.00]` | Black-Scholes |

Este dominio incluye opciones semanales, wings profundas y volatilidades de estrés. Es más exigente que un dominio académico centrado en ATM, pero sigue siendo manejable con evaluación por bins. A diferencia de Chen et al., no copiamos su moneyness normalizada ni su escala industrial de entrenamiento; copiamos la idea metodológica de declarar un hipercubo, muestrear de forma controlada y validar explícitamente las regiones difíciles.

Para Heston añadimos un dominio de parámetros amplio, pero no tan extremo como el de Chen et al. en Bates, porque nuestro objetivo es estudiar el surrogate y no cubrir todos los casos patológicos de calibración.

| Variable Heston | Rango | Interpretación |
|---|---|---|
| `v0` | `[0.0009, 1.00]` | `sqrt(v0)` entre 3% y 100% |
| `theta` | `[0.0009, 1.00]` | `sqrt(theta)` entre 3% y 100% |
| `kappa` | `[0.10, 10.00]` | Reversión lenta a rápida |
| `xi` | `[0.10, 3.00]` | Volatilidad de la varianza moderada a extrema |
| `rho` | `[-0.95, -0.05]` | Skew negativo típico de equity |

No imponemos la condición de Feller `2 kappa theta >= xi^2` como restricción dura. Muchas calibraciones reales de Heston la violan y excluirlas estrecharía artificialmente el problema. La registramos como diagnóstico, reportamos qué porcentaje del dataset la cumple y descartamos únicamente puntos con fallos numéricos claros o precios fuera de cotas de no arbitraje.

El muestreo baseline será uniforme en la escala financiera declarada. En Black-Scholes se muestrean directamente `m`, `T`, `r` y `sigma`. En Heston se muestrean uniformemente `sqrt(v0)` y `sqrt(theta)` en `[0.03, 1.00]` y después se elevan al cuadrado; `kappa`, `xi` y `rho` se muestrean uniformemente en sus rangos. Esta regla mantiene el baseline alineado con Chen et al., pero evita que una uniforme directa en varianza concentre artificialmente demasiados puntos en volatilidades extremas.

La comparación de métodos de muestreo queda concentrada en E3. Todos los demás experimentos usan el baseline uniforme para no mezclar efectos. En E3, H-3 representa el muestreo uniforme y H-5 representa el muestreo enfocado, manteniendo fija la arquitectura, la pérdida, el tamaño del dataset y el protocolo de evaluación.

El sampler enfocado de H-5 será una mezcla `50/50`. La mitad de las muestras seguirá el baseline uniforme completo. La otra mitad usará `m ~ TruncNormal(1.0, 0.15, [0.7, 1.3])` y `T ~ LogUniform(7/365, 0.25)`, manteniendo `r` y los parámetros de Heston con el mismo muestreo baseline. Esto concentra datos en ATM y vencimientos cortos sin abandonar la cobertura global del dominio.

## Particiones de datos

---

Separaremos explícitamente entrenamiento, validación y test. El conjunto de entrenamiento depende de cada surrogate: `BS-*` usa 200k puntos, `H-*` usa 500k puntos y las variantes `small` de E5 usan 100k puntos. El conjunto de validación será independiente, uniforme y de 50k puntos por familia de modelo. Se usará para monitorizar convergencia, ajustar el scheduler y seleccionar el checkpoint por una regla fija, pero no para detener el entrenamiento antes del presupuesto definido.

El test final será independiente y balanceado por bins. Para cada familia de modelo generaremos 5k puntos por cada combinación de moneyness y vencimiento, es decir, 25 bins y 125k puntos en total. Este test no se usa durante entrenamiento ni selección de checkpoint. En particular, H-3 y H-5 se evalúan sobre el mismo test balanceado, aunque H-5 haya entrenado con muestreo enfocado. Así E3 mide mejora o deterioro por región sin sesgar la evaluación hacia la distribución de entrenamiento de H-5.

## Bins de evaluación

---

Evaluamos por bins porque las métricas globales como el MAE promedio esconden lo que de verdad importa: en qué regiones del dominio el surrogate funciona mejor o peor. La superficie de pricing tiene rasgos muy distintos según moneyness y vencimiento, así que un MAE promedio puede ser engañosamente bajo si la mayoría de las opciones del test set caen en regiones fáciles. Particionar el dominio nos permite ver el comportamiento por regiones y descubrir si hay bins donde el surrogate falla sistemáticamente, que es exactamente la información que importa para una aplicación práctica.

| Moneyness | Vencimiento |
|---|---|
| Deep OTM `[0.4, 0.7)` | Weekly `[7/365, 14/365)` |
| OTM `[0.7, 0.9)` | Short `[14/365, 1/12)` |
| ATM `[0.9, 1.1]` | Medium-short `[1/12, 0.25)` |
| ITM `(1.1, 1.3]` | Medium `[0.25, 1.0)` |
| Deep ITM `(1.3, 2.0]` | Long `[1.0, 2.0]` |

Cinco niveles de moneyness por cinco de vencimiento dan veinticinco bins. Esta partición es más granular que la de Chen porque queremos separar explícitamente wings profundas y opciones semanales, dos regiones relevantes para trading y numéricamente más delicadas. Con cinco mil opciones por bin, el test set completo tendría ciento veinticinco mil puntos, un tamaño asumible para evaluación y suficiente para reportar percentiles cincuenta, noventa y cinco y noventa y nueve con confianza. Reportar percentiles altos en lugar de solo el MAE es importante porque la cola de la distribución de errores es donde aparecen los problemas operativos: una opción con error grande puntual puede arruinar un cálculo de cobertura aunque el promedio de errores esté bien.

## Tareas por fase

---

El proyecto se divide en cuatro fases ordenadas. Cada una no empieza hasta que la anterior está cerrada y verificada contra un criterio de salida concreto. Esta secuencialidad es estricta porque las fases tienen dependencias reales: no podemos entrenar redes sobre datos que no hemos generado, no podemos generar datos sin solvers que funcionen y no podemos validar un experimento sin métricas implementadas. Intentar paralelizar fases provoca exactamente el tipo de problema que queremos evitar, que es descubrir a mitad del proyecto que algo de la fase anterior estaba mal y tener que volver atrás.

Cada tarea y cada criterio de salida lleva un cuadro `☐` en la columna de estado que iremos cambiando a `☑` conforme se cierre el entregable, de forma que el documento sirva también como tablero de progreso vivo.

### Fase 0 — Infraestructura (días 1–3)

| Estado | Tarea | Entregable |
|---|---|---|
| ☑ | Estructura del repo, `.gitignore`, `requirements.txt`, semillas centralizadas | Repo navegable, `pytest` corriendo en vacío |
| ☑ | BS solver: precio, Delta y Vega cerradas | `src/solvers/black_scholes.py` con tests |
| ☑ | IV inverter: Newton sobre BS | `src/solvers/iv.py` con tests |
| ☑ | Heston solver: Fourier semi-cerrado con `P1`, `P2` y Delta `P1` | `src/solvers/heston.py` con tests |
| ☑ | Validación de Delta Heston contra diferencias finitas centrales | Test en grid representativa |
| ☑ | Validación cruzada con QuantLib | Grid de precio, Delta y Vega BS; precio Heston; Delta Heston por diferencias finitas QuantLib |

**Criterio de salida:** ☑ Black-Scholes y Heston reproducen valores de referencia externos con error inferior a `1e-6` en precio y `1e-4` en Delta; el inversor de IV recupera la volatilidad de referencia en casos cerrados de Black-Scholes.

La razón por la que esta fase es la primera y se cierra contra valores tabulados conocidos es que un fallo silencioso en el solver contaminaría todo el resto del proyecto sin que nos enteremos. Si la red está aprendiendo de targets erróneos, ningún entrenamiento posterior tiene sentido. La inversión inicial en validar los solvers compensa con creces el coste de descubrir el problema más tarde, y por eso el criterio de salida es estricto y verificable, no una métrica cualitativa.

### Fase 1 — Pipeline de datos y entrenamiento (días 4–7)

| Estado | Tarea | Entregable |
|---|---|---|
| ☑ | `sampler.py`: muestreo uniforme y enfocado del hipercubo | `src/datasets/sampler.py` |
| ☑ | `generator.py`: produce `(x, precio, Delta opcional)` | `src/datasets/generator.py` |
| ☑ | Script reproducible de generación | `scripts/generate_dataset.py` con salida `.npz`, metadatos `.json` y modo test balanceado |
| ☐ | Datasets train BS-200k y Heston-500k uniformes | Ficheros en `data/`, gitignored |
| ☐ | Datasets validation 50k y test balanceado 125k por familia | Ficheros reproducibles en `data/` |
| ☑ | `mlp.py`: MLP parametrizable por activación | `src/models/mlp.py` |
| ☑ | `greeks.py`: Delta del surrogate por autograd y regla de cadena | `src/models/greeks.py` con tests |
| ☑ | `trainer.py`: loop con pérdida de precio y término de Delta opcional | `src/training/trainer.py` |
| ☐ | Smoke test: entrenar BS-3 hasta convergencia | Reporte con curvas de pérdida |

**Criterio de salida:** ☐ BS-3 converge a `MAE_precio < 1e-4` en validación.

El smoke test sobre BS-3 al final de esta fase no es opcional. Es la primera vez que el pipeline completo se ejecuta de principio a fin: sampler, generador, modelo, entrenador y evaluación. Si BS-3 no converge a un error pequeño cuando le pasamos datos de una fórmula cerrada conocida, hay algo roto en el pipeline antes de tocar Heston y este es el momento de descubrirlo. Una vez cerrada esta fase, el resto del proyecto se reduce a entrenar variantes de la misma configuración base sobre datos generados con la misma maquinaria.

### Fase 2 — Experimentos sobre Black-Scholes (días 8–12)

| Estado | Tarea | Entregable |
|---|---|---|
| ☐ | Entrenar BS-1, BS-2, BS-3, BS-4 | Cuatro checkpoints en `results/bs/` |
| ☐ | `binning.py` y `metrics.py` | Módulo de evaluación con tests |
| ☐ | Tablas y heatmaps por bin para cada surrogate | Figuras y CSV en `results/bs/` |
| ☐ | Análisis E1 (precio vs IV) sobre BS-3 | Tablas y heatmaps precio/IV por bin |
| ☐ | Análisis E2 (activaciones) sobre BS-1..4 | Sección redactable de la memoria |

**Criterio de salida:** ☐ el análisis Black-Scholes reporta `MAE_Delta` y `MAE(C/K)` por activación y por bin, y permite comprobar si el patrón de activaciones suaves frente a ReLU aparece en un entorno con Delta analítica.

El motivo de hacer Black-Scholes antes que Heston es que Black-Scholes tiene fórmulas cerradas para precio, Delta, Vega e IV, lo que permite validar todas las piezas del pipeline contra valores exactos. Si E1 y E2 dan resultados coherentes sobre Black-Scholes, podemos extender el mismo análisis a Heston con la confianza de que el pipeline está bien montado. No fijamos un umbral numérico de éxito para E2 en esta fase; lo importante es que el reporte permita detectar si el patrón señalado por Chen sobre activaciones suaves y Deltas aparece, y que cualquier resultado contrario pueda diagnosticarse antes de llegar a Heston, donde no tenemos una referencia analítica tan cómoda.

### Fase 3 — Experimentos sobre Heston (días 13–22)

| Estado | Tarea | Entregable |
|---|---|---|
| ☐ | Dataset train Heston enfocado para H-5 | Fichero en `data/` |
| ☐ | Dataset Heston 100k uniforme con precio y Delta para E5 | Fichero en `data/` |
| ☐ | Entrenar H-1, H-2, H-3, H-4 | Cuatro checkpoints |
| ☐ | Entrenar H-5 (muestreo enfocado) | Un checkpoint |
| ☐ | Entrenar H-3-small y H-6-small para E5 | Dos checkpoints |
| ☐ | Análisis E2 (activaciones) sobre H-1..4 | Comparación por bins sin umbral fijo |
| ☐ | Análisis E3 (muestreo) sobre H-3 vs H-5 | Clasificación fuerte/débil/negativa |
| ☐ | Análisis E5 (DML) sobre H-3-small, H-6-small y H-3 | Clasificación fuerte/débil/negativa |

**Criterio de salida:** ☐ H-3 alcanza `MAE_IV < 1e-3` en el percentil noventa y cinco del error para opciones con `T > 0.1`.

Heston es el bloque más grande del proyecto y consume las dos semanas centrales. Aquí entran tres de los cinco experimentos y siete de los once surrogates. La razón de exigir `MAE_IV < 1e-3` como criterio de salida es que ese es el orden de magnitud que reporta Chen sobre Bates 13D, y dado que Heston es menos exigente que Bates por tener menos dimensiones y carecer de saltos, no llegar a ese nivel indicaría un problema en el pipeline o en el solver más que una limitación intrínseca del enfoque. El umbral aplica solo a opciones con `T > 0.1` porque los vencimientos muy cortos son intrínsecamente más difíciles y conviene reportarlos por separado.

### Fase 4 — Eficiencia y cierre (días 23–28)

| Estado | Tarea | Entregable |
|---|---|---|
| ☐ | `timing.py`: protocolo de cronometraje con warmup y diez repeticiones por lote | Módulo con tests |
| ☐ | Análisis E4 (eficiencia) sobre H-3 | Tabla de medianas, p25/p75 y speedups por lote |
| ☐ | Figuras finales y consistencia de estilo | Carpeta `results/figures/` |
| ☐ | Redacción de la memoria y revisión cruzada | Documento final |

**Criterio de salida:** ☐ speedup documentado para lotes de `10²`, `10³`, `10⁴` y `10⁵` opciones.

La eficiencia se mide al final del proyecto porque depende de tener el surrogate ya entrenado y validado, no antes. Hacerla en una fase previa carecería de sentido porque no podríamos comparar con un surrogate que aún no existe. El cierre del documento ocurre en paralelo durante esta fase porque la mayor parte del trabajo experimental ya está hecho y solo queda integrar resultados, generar figuras finales y revisar la consistencia del texto antes de la entrega.
