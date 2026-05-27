# Metodología experimental

**Autores:** Ángel Fernández Sánchez, Jorge Alfageme Sotillos, Héctor Pérez Ledesma

## Visión

---

Este documento concreta la metodología que conecta las preguntas experimentales de `tasks.md` con la implementación que describe `architecture.md`. Su función no es añadir más tareas, sino fijar cómo se ejecuta cada experimento, qué magnitudes se entrenan, qué métricas se reportan y bajo qué condiciones consideramos que una comparación es limpia. En un proyecto de surrogates, la parte delicada no es solo entrenar redes, sino asegurar que el dato sintético, la normalización, la métrica y la evaluación responden exactamente a la pregunta que se quiere contestar.

La idea central del proyecto es combinar dos líneas de la literatura. De Chen, Didisheim y Scheidegger tomamos la lógica de deep surrogates: construir una red que aproxime un solver financiero caro a partir de datos sintéticos generados por el propio modelo. De Huge y Savine tomamos, de forma acotada, la idea de differential machine learning: si además del valor de la función conocemos una derivada económicamente relevante, podemos enseñar a la red la forma local de la función y no solo su nivel. En nuestro caso, esa derivada será la Delta.

## Pipeline experimental

---

Todos los experimentos siguen el mismo flujo base. Primero se define un dominio de inputs y una distribución de muestreo. Después se genera un dataset sintético evaluando el solver verdadero en cada punto. A continuación se entrena un surrogate con una configuración cerrada y versionada. Finalmente se evalúa el surrogate sobre un conjunto independiente, separando los resultados por bins de moneyness y vencimiento.

La separación entre generación, entrenamiento y evaluación es importante porque evita contaminar las conclusiones. El sampler decide dónde preguntamos al modelo verdadero; el solver decide cuáles son los targets; el trainer decide cómo aprende la red; y el evaluador decide cómo medimos el error. Si una comparación experimental cambia más de una de estas piezas al mismo tiempo, deja de estar claro qué causa la diferencia observada.

## Targets, normalización y métricas

---

El target principal de entrenamiento será el precio normalizado por strike,

```text
y = C / K
```

y el input financiero principal será la moneyness simple,

```text
m = S / K
```

Esta decisión elimina una escala redundante del problema: si spot y strike se multiplican por la misma constante, el precio de una call europea escala con el strike. Trabajar con `C/K` y `S/K` permite que la red aprenda una función más estable y facilita la interpretación de la Delta, porque

```text
Delta = dC/dS = d(C/K)/d(S/K) = dy/dm.
```

No usamos la moneyness normalizada de Chen et al. porque nuestro objetivo no es replicar la escala empírica de SPX, sino construir un entorno controlado donde la relación entre precio normalizado y Delta sea directa. También fijamos `q = 0` en todos los experimentos. El dividend yield no forma parte de las preguntas centrales del trabajo, y eliminarlo reduce la dimensión del dominio sin afectar a las comparaciones entre métricas, activaciones, muestreo, eficiencia y aprendizaje diferencial.

Además de esta normalización financiera, todos los inputs se normalizarán numéricamente a `[0, 1]` antes de entrar en la red. Esta normalización no cambia el problema matemático, pero mejora la estabilidad del optimizador porque evita que dimensiones como `T`, `rho`, `v0` o `kappa` vivan en escalas muy distintas. Cuando se calculen derivadas de la red, se aplicará la regla de la cadena para volver a la escala financiera. En particular, si

```text
m_norm = (m - m_min) / (m_max - m_min),
```

entonces

```text
Delta_hat = d y_hat / d m = (d y_hat / d m_norm) / (m_max - m_min).
```

La Delta del surrogate se calculará con autograd de PyTorch sobre la red, no con diferencias finitas. La Delta verdadera del dataset viene del solver (`N(d1)` en Black-Scholes y `P1` en Heston), mientras que la Delta predicha se obtiene derivando `y_hat` respecto a `m_norm` y aplicando la corrección anterior. En evaluación, como E2, se usará `create_graph=False` porque solo se mide la derivada. En E5, durante entrenamiento, se usará `create_graph=True` porque la pérdida contiene `MAE(Delta)` y PyTorch necesita derivar esa pérdida respecto a los pesos de la red. Esto incrementa coste y memoria, pero es el coste esperado del entrenamiento diferencial.

La volatilidad implícita Black-Scholes no será el target principal de entrenamiento. Se usará como métrica de evaluación. Esta elección mantiene el entrenamiento más directo, porque el solver genera precios, y evita introducir fallos de inversión de volatilidad implícita durante la generación del dataset. Al mismo tiempo, evaluar también en IV permite comprobar si un error pequeño en precio es realmente pequeño en la escala que se usa en mercado.

Las métricas mínimas de evaluación serán `MAE(C/K)`, error absoluto de precio, `MAE_IV` y `MAE_Delta`. Para cada una se reportarán promedios y percentiles por bin, con especial atención al percentil 95, porque los errores de cola son los que revelan si el surrogate falla en regiones operativamente importantes.

## Solvers y validación numérica

---

Black-Scholes funciona como entorno de control. El precio, la Delta, la Vega y la inversión a volatilidad implícita tienen fórmulas cerradas o procedimientos numéricos muy estables, así que cualquier fallo en Black-Scholes apunta a un error del pipeline y no a una dificultad intrínseca del modelo.

Heston será el caso principal. El precio se calculará mediante una formulación semi-cerrada de Fourier, preferiblemente expresada en términos de las probabilidades `P1` y `P2`,

```text
C = S e^{-qT} P1 - K e^{-rT} P2.
```

Esta formulación tiene una ventaja metodológica para E5: la Delta de una call europea se obtiene como

```text
Delta = e^{-qT} P1.
```

Si fijamos `q = 0`, entonces `Delta = P1`. Esta Delta será el target diferencial de E5. Antes de generar datasets con Delta, la implementación se validará contra diferencias finitas centrales en una grid pequeña de puntos representativos del dominio. La validación no busca que las diferencias finitas sean el método principal, sino detectar errores de implementación, escalado o inestabilidad numérica. El criterio de aceptación será error absoluto medio inferior a `1e-4` en Delta; los errores extremos se revisarán manualmente porque pueden señalar problemas de integración, vencimientos demasiado cortos o puntos cercanos a frontera.

La inversión a volatilidad implícita se usará solo en evaluación. El inverter deberá controlar precios fuera de límites de arbitraje, zonas de Vega baja y fallos de convergencia. Los fallos no deben ocultarse: se reportarán como parte del diagnóstico, especialmente en vencimientos muy cortos o regiones muy fuera del dinero.

## Generación de datos

---

Los datasets se generarán muestreando un hipercubo de inputs. Esta decisión sigue la lógica de Chen et al.: tratamos los parámetros del modelo como pseudo-estados para que una única red aprenda la función de pricing en muchas parametrizaciones posibles. La distribución base será uniforme, porque cubre el dominio completo y no replica únicamente las regiones más frecuentes del mercado.

El dominio contractual común será amplio y orientado a trading:

| Variable | Rango | Uso |
|---|---|---|
| `m = S/K` | `[0.4, 2.0]` | Black-Scholes y Heston |
| `T` | `[7/365, 2.0]` | Black-Scholes y Heston |
| `r` | `[0.00, 0.075]` | Black-Scholes y Heston |
| `q` | `0` | Black-Scholes y Heston |
| `sigma` | `[0.03, 1.00]` | Black-Scholes |

Este dominio incluye opciones semanales, wings profundas y volatilidades de estrés. Es deliberadamente más exigente que un dominio centrado en `m ∈ [0.6, 1.4]` y vencimientos cómodos, porque esas regiones extremas son relevantes en trading y son precisamente donde precio, IV y Delta pueden contar historias distintas. La contrapartida es que la evaluación debe ser más granular para no mezclar zonas fáciles con zonas difíciles.

Para Heston, el dominio contractual anterior se combina con un hipercubo propio de parámetros del proceso:

| Variable Heston | Rango | Interpretación |
|---|---|---|
| `v0` | `[0.0009, 1.00]` | `sqrt(v0)` entre 3% y 100% |
| `theta` | `[0.0009, 1.00]` | `sqrt(theta)` entre 3% y 100% |
| `kappa` | `[0.10, 10.00]` | Reversión lenta a rápida |
| `xi` | `[0.10, 3.00]` | Volatilidad de la varianza moderada a extrema |
| `rho` | `[-0.95, -0.05]` | Skew negativo típico de equity |

La elección de `v0` y `theta` se expresa en varianza, pero se interpreta a través de su raíz cuadrada porque esa es la escala financiera natural. El rango cubre desde volatilidades bajas de mercado hasta episodios de estrés severo sin forzar al solver a vivir permanentemente en casos extremos. `kappa` permite tanto reversión lenta como rápida a la media, pero evita el rango muy agresivo de Chen et al. porque nuestro dataset será varios órdenes de magnitud menor. `xi` mantiene suficiente variabilidad para generar smiles pronunciadas, aunque acota los casos donde la integración de Heston se vuelve frágil. `rho` se restringe a valores negativos porque el proyecto se centra en el comportamiento típico de equity, donde la correlación negativa entre spot y varianza es la fuente principal del skew.

No imponemos la condición de Feller `2 kappa theta >= xi^2` como restricción de muestreo. Tratarla como filtro obligatorio haría el problema más limpio, pero menos representativo de calibraciones reales. En su lugar, se guarda como variable diagnóstica: el informe debe indicar qué porcentaje de puntos la cumple y separar los descartes debidos a fallos numéricos o violaciones evidentes de cotas de no arbitraje.

El muestreo baseline será uniforme en la escala financiera elegida. Para Black-Scholes, esto significa uniforme directo sobre `m`, `T`, `r` y `sigma`. Para Heston, `sqrt(v0)` y `sqrt(theta)` se muestrearán uniformemente en `[0.03, 1.00]` y después se convertirán a varianza; `kappa`, `xi` y `rho` se muestrearán uniformemente en sus rangos. Esta decisión conserva la simplicidad del esquema de Chen et al., pero evita una distorsión importante: una uniforme directa sobre varianza sobreponderaría volatilidades muy altas.

La comparación de métodos de muestreo se concentra únicamente en E3. El resto de experimentos usa el baseline uniforme para que las diferencias observadas se puedan atribuir a la variable que cada experimento modifica. En E3 se añadirá un sampler enfocado, con más masa cerca del at-the-money y de vencimientos cortos, donde la función de pricing tiene mayor curvatura y donde los errores suelen ser más relevantes. La comparación con el sampler uniforme solo será válida si se mantienen fijos el tamaño total del dataset, la arquitectura, la pérdida, la semilla y el protocolo de evaluación.

El sampler enfocado será una mezcla. Con probabilidad `0.5` se genera una muestra uniforme en todo el dominio, exactamente igual que en H-3. Con probabilidad `0.5` se genera una muestra enfocada: `m` se obtiene de una normal truncada con media `1.0`, desviación `0.15` y soporte `[0.7, 1.3]`; `T` se obtiene de una log-uniforme en `[7/365, 0.25]`. El resto de variables se muestrea igual que en el baseline. Esta construcción hace que H-5 responda a una pregunta concreta: si mantenemos la misma familia de modelos Heston, pero colocamos más contratos en zonas de mayor curvatura, ¿mejora el error en los bins operativamente críticos?

La partición de datos tendrá tres niveles. El entrenamiento usa la distribución propia de cada surrogate. La validación usa una muestra independiente uniforme de 50k puntos por familia de modelo y sirve para monitorizar convergencia y seleccionar el checkpoint mediante una regla fijada antes del experimento. Esta selección no cambia el presupuesto de entrenamiento: no hacemos early stopping, no activamos scheduler y solo conservamos el mejor estado observado bajo la misma métrica para todos los surrogates comparables.

La regla de selección será siempre el menor `MAE(C/K)` en validación. Mantenerla fija es importante para que las comparaciones sean limpias. En particular, H-6-small se entrenará con precio y Delta, pero se seleccionará por precio normalizado igual que H-3-small. Así E5 mide si añadir Delta a la pérdida mejora la función aprendida, sin introducir una segunda diferencia metodológica en la elección del checkpoint.

El test final será independiente, balanceado por bins y compartido por todos los surrogates de una misma familia de modelo. Cada uno de los 25 bins tendrá 5k puntos, para un total de 125k observaciones. Esta decisión es especialmente importante en E3: si evaluásemos H-5 en una muestra con la misma concentración con la que entrena, confundiríamos precisión local con mejora global. El test balanceado permite ver exactamente dónde gana y dónde pierde el sampler enfocado.

Las métricas principales se reportarán en un dominio interior que recorta un 5% de cada dimensión del hipercubo. Las redes tienden a aproximar peor en fronteras porque no tienen vecinos a ambos lados de la función. Separar el dominio interior de la frontera evita que el promedio global quede dominado por errores de borde y hace que las conclusiones sobre el comportamiento en el interior sean más limpias.

## Evaluación por bins

---

La evaluación por bins es obligatoria porque los errores medios globales son insuficientes. Una red puede tener buen MAE agregado y, aun así, fallar sistemáticamente en opciones ATM de vencimiento corto o en regiones OTM donde la Vega es baja. Por eso se usará una partición `5 × 5` por moneyness y vencimiento.

Los bins de moneyness serán deep OTM, OTM, ATM, ITM y deep ITM. Los bins de vencimiento serán weekly, short, medium-short, medium y long. Dentro de cada bin se calcularán las mismas métricas, de forma que cada experimento pueda distinguir entre mejora global y mejora localizada. Esta distinción es especialmente importante para E3, donde esperamos que el muestreo enfocado mejore regiones difíciles a costa de perder algo de precisión en regiones menos muestreadas.

| Moneyness | Vencimiento |
|---|---|
| Deep OTM `[0.4, 0.7)` | Weekly `[7/365, 14/365)` |
| OTM `[0.7, 0.9)` | Short `[14/365, 1/12)` |
| ATM `[0.9, 1.1)` | Medium-short `[1/12, 0.25)` |
| ITM `[1.1, 1.3)` | Medium `[0.25, 1.0)` |
| Deep ITM `[1.3, 2.0]` | Long `[1.0, 2.0]` |

La convención de fronteras es semiabierta: cada bin incluye su extremo inferior y excluye su extremo superior, salvo el último bin de cada eje, que incluye también el extremo superior del dominio. Así, por ejemplo, `m = 1.1` cae en ITM y `m = 2.0` cae en deep ITM.

## Experimento E1 — Precio frente a volatilidad implícita

---

E1 responde a una pregunta metodológica: si un surrogate tiene poco error en precio, ¿podemos asumir que también tiene poco error en volatilidad implícita? La respuesta no tiene por qué ser afirmativa. La transformación de precio a IV depende de la Vega, y en regiones de Vega baja pequeños errores de precio pueden convertirse en errores de IV relevantes.

El experimento no entrena redes adicionales. Usa los surrogates baseline, especialmente BS-3 y H-3, y recalcula sus errores en dos escalas: precio normalizado e IV. La comparación se hará por bins. Si aparecen zonas donde el error de precio es bajo pero el error de IV no lo es, quedará justificado que IV debe formar parte del protocolo de evaluación aunque no sea el target de entrenamiento.

La métrica primaria de E1 no será un único promedio, sino la discrepancia por bin entre `MAE(C/K)` y `MAE_IV`. La justificación es que IV no es una transformación homogénea del precio: cuando la Vega es baja, errores pequeños en precio pueden convertirse en errores relevantes de IV. Como proxy de Vega se usará la Vega Black-Scholes evaluada en la IV objetivo recuperada desde el precio verdadero, agregada por bin. Los percentiles altos y los fallos de inversión de IV se reportarán como diagnóstico para distinguir error del surrogate de fragilidad numérica del inverter.

No fijamos clasificación fuerte/débil para E1. El objetivo no es decidir si un surrogate gana o pierde, sino comprobar si la métrica de precio puede ocultar errores relevantes en la escala de mercado. El entregable debe incluir tabla por bin con `MAE(C/K)`, `MAE_IV`, Vega media o proxy de Vega, percentiles altos y tasa de fallos de inversión IV, además de heatmaps separados para precio e IV.

## Experimento E2 — Activaciones y calidad de Delta

---

E2 compara ReLU, Softplus, Swish y tanh manteniendo fijo todo lo demás. La pregunta no es solo qué activación da mejor precio, sino cuál produce mejores derivadas. Esta distinción viene directamente de Chen et al.: una red con ReLU puede aproximar bien el nivel de la función, pero producir Deltas menos estables por la falta de suavidad de la activación.

En Black-Scholes, la Delta verdadera es analítica y permite validar el análisis en un entorno controlado. En Heston, la Delta se obtendrá del solver de referencia. El resultado esperado no es necesariamente que Swish tenga un MAE de precio muy inferior, sino que produzca una Delta más limpia y con menos errores extremos. Por eso el análisis debe reportar precio y Delta por separado.

La métrica primaria de E2 será `MAE_Delta` por bin. El precio normalizado se mantiene como control, porque una activación solo será preferible si mejora las derivadas sin degradar de forma relevante el nivel de la función. Esta decisión sigue la motivación de Chen et al.: ReLU puede ajustar precios aceptables, pero su falta de suavidad puede producir Greeks pobres.

No fijamos umbrales fuerte/débil para E2. A diferencia de E3 y E5, aquí no se evalúa una intervención práctica con una frontera natural de éxito, sino un patrón arquitectónico: si las activaciones suaves producen Deltas más estables que ReLU. La conclusión se tomará tras observar la robustez del patrón en Black-Scholes y Heston, por bins y sin perder de vista el error de precio.

## Experimento E3 — Muestreo uniforme frente a enfocado

---

E3 evalúa si conviene gastar más muestras en las zonas donde la función es más difícil. La hipótesis es que el muestreo enfocado reducirá el error en bins ATM y vencimientos cortos, porque ahí la superficie de pricing tiene mayor curvatura y las sensibilidades cambian más rápido.

La comparación principal será H-3 frente a H-5. Ambos surrogates deben compartir arquitectura, pérdida, optimizador, tamaño de dataset y protocolo de evaluación. La única diferencia será la distribución de muestreo contractual en `m` y `T`; los parámetros de Heston se mantienen con el mismo muestreo baseline para no mezclar el efecto del sampler con el efecto de cambiar el régimen de volatilidad. Si H-5 mejora los bins críticos pero empeora regiones extremas, el resultado seguirá siendo informativo: mostrará el tradeoff entre precisión local y cobertura uniforme del dominio.

La métrica primaria de E3 será `MAE_IV` en los bins ATM combinados con weekly, short y medium-short. Son las regiones donde el sampler enfocado coloca más masa y donde esperamos mayor curvatura de la función de pricing. Como control, se reportarán `MAE(C/K)`, el deterioro en wings, deep ITM/OTM y vencimientos largos. Esta doble lectura es necesaria porque un sampler enfocado puede mejorar la precisión local sacrificando cobertura global.

Fijamos antes de entrenar tres niveles de interpretación. E3 será positivo fuerte si H-5 reduce al menos un 10% el `MAE_IV` promedio en los bins críticos frente a H-3 y el `MAE_IV` global no empeora más de un 10%. Los bins críticos son ATM combinado con weekly, short y medium-short. Será positivo débil si H-5 mejora esos bins pero menos de un 10%, o si la mejora local viene acompañada de un deterioro global moderado. Será negativo si no mejora los bins críticos o si la mejora local exige sacrificar de forma excesiva la cobertura global. Usamos `MAE_IV` para la clasificación porque E3 está motivado por precisión práctica en la escala de mercado; `MAE(C/K)` se reporta como control porque sigue siendo el target de entrenamiento.

## Experimento E4 — Eficiencia computacional

---

E4 mide el beneficio práctico del surrogate frente al solver original. La comparación se hará por tamaño de lote, no con una única evaluación aislada, porque la red aprovecha mejor el paralelismo cuando procesa muchos puntos a la vez. El resultado esperado es que el speedup crezca con el tamaño del lote, especialmente si se usa GPU.

El protocolo debe fijar hardware, precisión numérica, número de repeticiones, warmup, modo de evaluación de PyTorch y si el tiempo incluye conversiones de datos. Sin ese protocolo, los tiempos serían difíciles de interpretar. La métrica principal será el ratio

```text
speedup = tiempo_solver / tiempo_surrogate.
```

La métrica primaria de E4 será ese speedup por tamaño de lote. Reportarlo como curva es más informativo que reportar un único número, porque el surrogate aprovecha mejor el paralelismo cuando evalúa lotes grandes y el solver numérico no escala igual. Los tiempos solo serán comparables si se documentan warmup, repeticiones, hardware, modo CPU/GPU y si se incluyen conversiones de datos.

El protocolo concreto usará lotes de `10^2`, `10^3`, `10^4` y `10^5` opciones. Para cada tamaño se harán tres ejecuciones de warmup que no se miden y diez repeticiones medidas. Se reportará la mediana del tiempo y el rango intercuartílico `p25/p75`, porque la mediana es más robusta ante ejecuciones aisladas lentas. El surrogate se evaluará con `model.eval()` y `torch.no_grad()`. Si hay GPU disponible, se reportarán por separado CPU y GPU para no mezclar mejora algorítmica con diferencia de hardware.

## Experimento E5 — Differential ML con Delta

---

E5 combina la idea de deep surrogates con la idea central de differential machine learning, pero de forma deliberadamente acotada. El objetivo no es entrenar con todas las Greeks ni con todas las sensibilidades del modelo, sino comprobar si añadir Delta como target diferencial mejora la capacidad de aprendizaje del surrogate.

La comparación principal será

```text
H-3-small: precio solo, 100k muestras
H-6-small: precio + Delta, 100k muestras
H-3: precio solo, 500k muestras
```

Esta estructura responde a la pregunta de eficiencia muestral. Si H-6-small supera claramente a H-3-small en Delta y mantiene un error de precio comparable, entonces la información diferencial aporta valor. Si además se acerca a H-3 usando muchas menos muestras, el resultado conecta directamente con la tesis de Huge y Savine: las derivadas enseñan a la red la forma local de la función y reducen la cantidad de datos necesaria.

La métrica primaria de E5 será la mejora de `MAE_Delta` de H-6-small frente a H-3-small. La restricción de validez es que `MAE(C/K)` se mantenga comparable; una mejora de Delta que destruya el precio no sería útil. H-3 funciona como referencia de muestra grande: si H-6-small se acerca a H-3 usando solo 100k puntos, la conclusión será de eficiencia muestral, no solo de calidad de Greeks.

Fijamos antes de entrenar tres niveles de interpretación. E5 será positivo fuerte si `MAE_Delta(H-6-small)` mejora al menos un 20% frente a `MAE_Delta(H-3-small)` y `MAE(C/K)(H-6-small)` no empeora más de un 10% frente a `MAE(C/K)(H-3-small)`. Será positivo débil si Delta mejora pero menos de un 20%, siempre que el precio siga dentro de ese margen del 10%. Será negativo si Delta no mejora o si la mejora de Delta exige sacrificar más de un 10% de precisión en precio. Estos umbrales no son leyes financieras; son una regla experimental previa para evitar interpretar resultados de forma oportunista.

La pérdida de H-6-small será

```text
L = MAE(C/K) + MAE(Delta).
```

Esto equivale a fijar `alpha = 1` y `beta = 1`. La decisión es deliberadamente simple: `C/K` y Delta viven en escalas comparables y E5 debe medir el efecto de añadir información diferencial, no el efecto de sintonizar pesos de loss. Durante el entrenamiento se registrarán `MAE(C/K)` y `MAE(Delta)` por separado para comprobar que ningún término domina completamente la optimización. Si apareciese dominancia fuerte, se reportará como limitación o extensión, pero no se cambiará la configuración principal de E5.

Vega, Gamma, Theta y Rho quedan fuera de la pérdida principal. Pueden calcularse como diagnóstico si el solver y el surrogate lo permiten, pero no deben convertirse en condiciones de éxito de E5. Esta restricción mantiene el experimento limpio y evita mezclar la pregunta principal con problemas numéricos adicionales.

## Criterios de validez y limitaciones

---

Cada experimento debe cerrarse con una comparación que cambie una sola dimensión relevante. Si se cambia activación, no se cambia dataset. Si se cambia sampler, no se cambia arquitectura. Si se cambia loss en E5, se mantiene el mismo tamaño pequeño de dataset para H-3-small y H-6-small. Esta disciplina es lo que permite atribuir los resultados a la variable experimental y no a decisiones accidentales de implementación.

El proyecto tiene limitaciones deliberadas. No intentamos calibrar a datos reales de mercado, ni estudiar opciones exóticas, ni construir un surrogate industrial a escala de Chen et al. El objetivo es demostrar, en un entorno controlado y reproducible, cuándo una red puede aproximar un solver de pricing con precisión, rapidez y derivadas útiles. La contribución está en la comparación ordenada de métricas, activaciones, muestreo, eficiencia y aprendizaje diferencial, no en maximizar complejidad del modelo a cualquier coste.
