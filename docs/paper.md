# Redes neuronales como aproximadores numéricos para pricing de opciones

**Autores:** Ángel Fernández Sánchez, Jorge Alfageme Sotillos, Héctor Pérez Ledesma

**Asignatura:** Métodos numéricos — proyecto final de investigación

---

## Resumen

Este trabajo estudia bajo qué condiciones una red neuronal profunda puede actuar como un *surrogate* preciso, rápido y diferenciable de una función de pricing de opciones, entendida como sustituto numérico de un solver matemático ya conocido. La motivación no es predecir nada sobre el mercado sino abaratar la evaluación de modelos cuya solución analítica no existe o es cara: una vez entrenada de forma offline a partir de datos sintéticos generados por el propio solver, la red puede sustituir al modelo original en tareas que exigen millones de evaluaciones, como la calibración diaria, el cálculo de Greeks o la simulación masiva de escenarios. Tomamos como referencia central el trabajo de Chen, Didisheim y Scheidegger sobre *deep surrogates* y como complemento metodológico el de Huge y Savine sobre *differential machine learning*. Nuestro objeto de estudio son las opciones europeas vainilla bajo dos modelos, Black-Scholes como caso de validación con solución cerrada y Heston como caso principal, donde la integración por Fourier semi-cerrada es lo suficientemente costosa como para justificar el surrogate. El trabajo se articula en torno a cinco experimentos comparativos que varían cada uno una sola dimensión del problema: la elección de métrica de evaluación, la activación de la red, la distribución del muestreo, la eficiencia computacional y el uso de información diferencial en la pérdida. Este documento describe la revisión bibliográfica que sustenta el planteamiento, la metodología experimental, el diseño concreto de cada comparación y la estructura del código que la implementa.

---

## 1. Introducción

El objetivo general de este trabajo es estudiar cómo se pueden usar las redes neuronales como herramienta numérica para aproximar modelos de pricing de opciones. Es decir, no nos interesa la red como un sistema que aprende del mercado, sino como un aproximador funcional capaz de imitar a un solver matemático ya conocido. La distinción puede parecer sutil pero cambia completamente lo que se está haciendo: no estamos prediciendo nada, estamos sustituyendo un cálculo costoso por una aproximación rápida y diferenciable.

En finanzas cuantitativas hay muchos modelos de pricing de opciones, desde el clásico Black-Scholes hasta modelos modernos con volatilidad estocástica, saltos o memoria. Todos ellos tienen una estructura común: dado un conjunto de inputs, que pueden ser características del contrato y parámetros del modelo, devuelven un precio o una volatilidad implícita. El problema es que conforme los modelos ganan realismo también se vuelven más caros computacionalmente. Lo que en Black-Scholes es una fórmula cerrada se convierte en Heston en una integral que hay que resolver por Fourier, y en Bates o en *rough volatility* en algo aún más pesado. Cuando estos modelos hay que evaluarlos miles o millones de veces, por ejemplo en calibración diaria, en cálculo de Greeks o en simulación de escenarios, el coste se vuelve un cuello de botella real.

La idea de los *deep surrogates* es resolver ese cuello de botella entrenando una red neuronal a partir de datos sintéticos generados por el propio modelo, de forma que la red aprenda la función de pricing. Si llamamos `f : X ⊂ R^d → R` a la función de pricing del modelo verdadero, lo que hacemos es entrenar una red `f_θ` con parámetros `θ` para que aproxime `f_θ(x) ≈ f(x)` sobre el dominio `X`. Una vez entrenada, evaluarla es órdenes de magnitud más rápido que llamar al solver original. El esquema general es por tanto

```text
modelo financiero  →  datos sintéticos  →  red neuronal  →  evaluación rápida
```

La red no reemplaza la teoría financiera, simplemente aprende a imitar al solver. El objetivo no es aplicar las técnicas de *deep learning* a datos de mercado ruidosos, sino utilizar estas técnicas para resolver un problema clásico de aproximación numérica con la ventaja añadida de que el resultado es diferenciable por construcción.

En Black-Scholes la fórmula cerrada ya es muy rápida y un surrogate no aporta valor práctico, pero es ideal como caso de validación, porque conocemos la solución exacta y podemos medir errores de precio, Delta y volatilidad implícita con precisión. El objetivo del proyecto es verificar si se pueden utilizar surrogates con modelos más complejos sin perder precisión, y para ello combinamos Black-Scholes como entorno controlado con Heston como caso principal de estudio. La pregunta de investigación que estructura todo el trabajo se formula así:

> *Bajo qué condiciones una red neuronal profunda puede actuar como un surrogate preciso, rápido y diferenciable para funciones de pricing de opciones.*

Para responderla de forma operativa, descomponemos la pregunta en cinco preguntas más concretas que se traducen en cinco experimentos comparativos. Cada uno varía una sola dimensión del problema mientras mantiene todo lo demás fijo, lo que permite atribuir cualquier diferencia observada a la dimensión que estamos variando y no a otras variables que se hayan colado por accidente.

El resto del documento se organiza como sigue. La sección 2 sintetiza la revisión bibliográfica que ha guiado las decisiones metodológicas. La sección 3 analiza con detalle el paper de Chen, Didisheim y Scheidegger por ser nuestra referencia central. La sección 4 explica cómo enfocamos nuestro propio trabajo a partir de esa base. La sección 5 fija la metodología experimental: targets, normalización, solvers, generación de datos y evaluación por bins. La sección 6 describe los cinco experimentos y los once surrogates que los sostienen. La sección 7 documenta la arquitectura del código y las decisiones de reproducibilidad. La sección 8 presenta el plan de ejecución por fases. La sección 9 discute alcance, limitaciones y trabajo en curso.

---

## 2. Revisión bibliográfica

La primera fase de nuestro trabajo ha consistido en una investigación bibliográfica profunda para tener una visión global de los temas relevantes y situar nuestra propuesta dentro del panorama existente. Hay cinco artículos que hemos trabajado a fondo y que forman el núcleo de la revisión: los tres primeros eran una recomendación de los profesores como punto de partida, un cuarto, el de *deep surrogates*, lo encontramos buscando trabajos más recientes en la línea de aproximación numérica de modelos de pricing, y un quinto, el de *differential machine learning* de Huge y Savine, lo hemos incorporado como complemento al anterior porque ofrece una pieza metodológica que conecta directamente con nuestras preocupaciones sobre la calidad de las derivadas aprendidas.

### 2.1. Hutchinson, Lo y Poggio (1994): el punto de partida histórico

El punto de partida histórico es el trabajo de Hutchinson, Lo y Poggio. Estos autores proponen un método no paramétrico para estimar la fórmula de pricing de un derivado mediante *learning networks*, en una época en la que la literatura financiera estaba dominada por enfoques paramétricos basados en Black-Scholes y argumentos de no arbitraje. La idea central es invertir la lógica habitual: en lugar de partir de un modelo estocástico para el subyacente y derivar analíticamente la fórmula del precio, dejan que la red aprenda directamente la función que mapea variables económicas observables, como el precio del subyacente normalizado por el strike y el tiempo a vencimiento, al precio de la opción, sin imponer supuestos de lognormalidad ni continuidad de trayectoria. Comparan cuatro arquitecturas (redes de funciones de base radial, perceptrones multicapa, *projection pursuit regression* y mínimos cuadrados ordinarios como baseline), y atacan a la vez los problemas de pricing y delta-hedging, lo cual es importante porque la cobertura exige que la red aproxime bien no solo el precio sino también su derivada respecto al subyacente. Validan primero con datos sintéticos generados por Monte Carlo bajo Black-Scholes y después con datos reales de opciones sobre futuros del S\&P 500 entre 1987 y 1991, donde la red llega a superar a Black-Scholes en *tracking error* de la cartera replicante.

Su relevancia histórica reside en que es el primer trabajo que demuestra de forma sistemática que una red neuronal puede aprender una fórmula de pricing y usarse operativamente para cubrir, abriendo la línea de investigación que conecta con el enfoque actual de *deep surrogates*. La diferencia crucial con nuestro proyecto es que aquí la red aprende del mercado, mientras que un surrogate moderno aprende la salida de un pricer ya conocido para acelerarlo: la red pasa de competir con Black-Scholes a aproximarlo de manera eficiente.

### 2.2. Becker, Cheridito y Jentzen (2019): deep optimal stopping

El segundo paper es el de Becker, Cheridito y Jentzen sobre *deep optimal stopping*. Aquí los autores proponen un método de *deep learning* para resolver problemas de parada óptima en tiempo discreto, es decir, problemas en los que hay que decidir en cada instante si parar o continuar un proceso de Markov para maximizar la esperanza de un pago. La idea central es que cualquier tiempo de parada admisible se puede descomponer en una sucesión de decisiones binarias, una por cada paso temporal, y que cada una de esas decisiones puede aprenderse como una función del estado actual mediante una red neuronal *feedforward*. Sustituyen la indicadora dura por una sigmoide, lo que permite entrenar los parámetros con ascenso de gradiente estocástico sobre trayectorias Monte Carlo, y proceden por inducción hacia atrás desde el último instante hasta el primero. El método se aplica al pricing de opciones bermudas tipo *max-call* sobre cestas de hasta quinientos activos en un Black-Scholes multidimensional, a un *callable multi barrier reverse convertible* y al problema no markoviano de parar óptimamente un movimiento browniano fraccionario, con precisión alta y tiempos cortos en dimensiones donde los métodos tradicionales sufren la maldición de la dimensionalidad.

Para nuestra revisión es importante porque ilustra una línea claramente distinta de la nuestra: ellos aprenden directamente la política de ejercicio para derivados con derecho de parada anticipada, mientras que en nuestro proyecto la red actuará como surrogate de la función de pricing de opciones europeas. Son enfoques complementarios dentro del uso del *deep learning* en finanzas cuantitativas, pero no resuelven el mismo problema, y conviene tener esa distinción clara para evitar comparaciones que no aplican.

### 2.3. Ruf y Wang (2020): mapa del campo

El tercer paper recomendado es la revisión bibliográfica de Ruf y Wang, que funciona como mapa general del campo. Los autores compilan y comparan más de ciento cincuenta artículos en una tabla detallada que cataloga cada estudio según seis dimensiones, entre ellas las variables de entrada, las salidas de la red, los modelos *benchmark*, las métricas de rendimiento, el método de partición de los datos y los activos subyacentes. Cubren tanto el enfoque puramente no paramétrico, donde la red aprende directamente el precio a partir de variables como subyacente, strike, vencimiento y volatilidad, como los enfoques híbridos que combinan fórmulas paramétricas tipo Black-Scholes con correcciones aprendidas, además de variantes más recientes orientadas a calibración, resolución de PDEs, aproximación de funciones de valor en problemas de control óptimo y *deep hedging*.

Lo más útil de este paper son las lecciones que extrae del repaso exhaustivo: la importancia de usar inputs estacionarios como la moneyness en lugar de precio y strike por separado, la necesidad de elegir *benchmarks* que no trivialicen la comparación y, sobre todo, el problema del *data leakage* que aparece cuando la partición train/test se hace de forma aleatoria sobre datos con estructura temporal, rompiendo el orden cronológico, contaminando el test e infraestimando sistemáticamente el error de generalización. Para nuestro trabajo es una referencia útil porque ofrece un panorama de la disciplina, permite situar nuestra propuesta dentro de las distintas líneas históricas y nos advierte de errores típicos que conviene evitar, aunque la mayor parte de la bibliografía que analiza versa sobre el uso de técnicas de *machine learning* directamente sobre datos de mercado y no sobre el problema de aproximar un solver conocido.

### 2.4. Chen, Didisheim y Scheidegger (2025): referencia central

El cuarto paper, y el que ha terminado convenciendonos como referencia central, es el de Chen, Didisheim y Scheidegger sobre *deep surrogates* en finanzas. Este artículo propone los *deep surrogates* como una clase de aproximadores numéricos para modelos estructurales costosos, sustituyendo la función original por una red neuronal profunda entrenada con muestras simuladas que toma exactamente los mismos inputs y devuelve la misma salida a un coste computacional drásticamente menor, además de proporcionar gradientes a un coste que no escala con el número de inputs.

Una idea crucial del paper es que la maldición de la dimensionalidad queda mitigada por dos factores que se refuerzan entre sí. Por un lado, los datos de entrenamiento se generan de manera sintética y determinista a partir del propio modelo, sin ruido estadístico, lo que elimina la necesidad de cubrir el espacio de inputs con una densidad de muestras que permita promediar errores aleatorios. Por otro lado, la función de pricing en modelos afines como Heston o Bates es analíticamente suave, lo que controla el problema de aproximación y permite a una red profunda representar el mapa de inputs a precio con un número de parámetros que escala de forma manejable con la dimensión. Estas dos propiedades juntas son las que justifican que un *deep surrogate* sea una herramienta de aproximación numérica plausible en problemas de dimensión moderada-alta, no solo un truco que funciona empíricamente.

En el paper emplean una versión del modelo de Bates en un espacio aumentado de dimensión trece, mapean los precios a volatilidad implícita Black-Scholes para que los errores sean comparables entre opciones de distinta moneyness y vencimiento, eligen Swish frente a ReLU para que las Deltas sean estables, validan por *bins* de moneyness y madurez y demuestran que el surrogate reduce el coste de la calibración diaria del modelo de las ciento veinticinco horas que necesita la inversión FFT a poco más de dos horas en CPU y unas decenas de minutos en GPU. Esa reducción de tiempos abre la puerta a aplicaciones empíricas que antes eran inviables, como el índice diario *TailRisk* propuesto para predecir crashes utilizando la información del mercado de opciones o la medida de inestabilidad de parámetros vinculada a la liquidez de mercado y a la competitividad de las *quotes* de los *market makers*.

Dedicamos la sección 3 a un análisis más detallado de este paper porque las decisiones que toman son las que guían las nuestras y conviene desempaquetarlas con cuidado.

### 2.5. Huge y Savine (2020): differential machine learning

El quinto paper es el de Huge y Savine sobre *differential machine learning*. Originalmente lo teníamos como uno más del panorama, pero al pensar en cómo podríamos innovar respecto al paper de *deep surrogates* nos hemos dado cuenta de que ofrece una pieza metodológica muy potente y muy alineada con nuestras preocupaciones. La idea es que cuando se entrena un surrogate con datos sintéticos, normalmente la red solo ve el valor de la función en cada punto, por ejemplo el precio de la opción, y aprende a predecirlo. Sin embargo, si el modelo verdadero es diferenciable, también se conoce la pendiente de la función en cada punto, es decir, las *Greeks*. *Differential machine learning* aprovecha esa información añadiendo los gradientes verdaderos al conjunto de entrenamiento y modificando la pérdida para que la red no solo estime el precio, sino también su derivada respecto a los inputs. El resultado es que se consiguen aproximaciones mucho más precisas, especialmente en términos de *Greeks*, con datasets sustancialmente más pequeños.

Para nuestro trabajo es una referencia clave por dos razones. La primera es que conecta directamente con la necesidad de validar *Greeks* que ya teníamos heredada del paper de Chen et al. La segunda es que tiene una motivación matemática sólida desde el punto de vista de aproximación de funciones, análoga a la que justifica la interpolación de Hermite frente a la clásica: disponer del valor y de la derivada de una función en cada punto muestreado permite reconstruirla con muchos menos puntos. Esta motivación es lo que nos lleva a incluir un experimento dedicado a estudiar el efecto de añadir Delta como *target* diferencial.

### 2.6. Otras líneas complementarias

Más allá de estos cinco pilares, hemos revisado otros trabajos que ayudan a entender que la aplicación de redes neuronales a derivados ha avanzado en varias direcciones complementarias. Una primera línea ataca de frente el problema de resolver las ecuaciones diferenciales parciales que aparecen en el pricing de derivados de alta dimensión, donde los métodos de mallado tradicionales chocan con la maldición de la dimensionalidad: en esa familia se inscribe el *Deep Galerkin Method* de Sirignano y Spiliopoulos, que entrena una red profunda para satisfacer el operador diferencial sobre puntos muestreados al azar y prescinde por completo de la malla, así como el *Deep BSDE Solver* de Han, Jentzen y E, que reformula la PDE como una ecuación diferencial estocástica retrógrada y aproxima el gradiente de la solución mediante redes neuronales, demostrando viabilidad incluso en problemas de cien dimensiones.

Una segunda línea busca acelerar la valoración aprendiendo directamente la función de precios, y aquí el trabajo de Ferguson y Green muestra que una red profunda entrenada con datos sintéticos generados por Monte Carlo puede valorar opciones *basket* millones de veces más rápido que el modelo subyacente. Una tercera vertiente aborda la cobertura de carteras como un problema de aprendizaje por refuerzo, como en el *deep hedging* de Buehler, Gonon, Teichmann y Wood, que parametriza la estrategia de cobertura mediante redes neuronales y la optimiza directamente bajo medidas de riesgo convexas, incorporando de forma natural costes de transacción y otras fricciones de mercado sin necesidad de *Greeks*. La calibración de modelos estocásticos también se ha beneficiado de estas técnicas: trabajos recientes aplican *differential ML* al modelo de Heston para reducir drásticamente el tiempo de calibración.

Después de revisar todo este material, la conclusión a la que hemos llegado es que el paper de *deep surrogates* es el que mejor encaja con nuestros objetivos. La razón principal es que aborda un problema que en la práctica de la industria es muy importante, abaratar el coste computacional de los modelos de pricing y reducir los tiempos de calibración y estimación de parámetros, y son temas muy actuales en mesas de derivados y en *research* cuantitativo a medida que los modelos se vuelven más realistas y por tanto más caros de evaluar. Además, desde el punto de vista de métodos numéricos, el enfoque de *surrogates* es el más completo e interesante porque entrelaza muestreo, aproximación de funciones, validación numérica y cálculo de derivadas.

---

## 3. Análisis del paper de referencia

Como el paper de Chen, Didisheim y Scheidegger es la referencia central del trabajo, conviene dedicarle un análisis detallado. Repasamos primero qué proponen y a qué modelo lo aplican, después por qué eligen redes neuronales frente a otras alternativas de aproximación numérica y las decisiones técnicas concretas que toman al construir el surrogate, y por último cómo validan la precisión del aproximador.

### 3.1. Qué proponen

El paper aplica la idea de *deep surrogate* al modelo de Bates, que combina volatilidad estocástica al estilo Heston con saltos en el precio del activo, donde los saltos siguen una distribución doble exponencial asimétrica en lugar de la normal del Bates original. Esto permite capturar fenómenos que Black-Scholes no captura bien, como el *leverage effect* (cuando el mercado cae la volatilidad suele subir) o la asimetría entre saltos positivos y negativos, algo importante porque en la realidad las caídas extremas y las subidas extremas no se comportan igual. La distribución doble exponencial permite modelar colas más pesadas que la normal y separar el comportamiento de saltos hacia arriba y hacia abajo.

Cuando llevan Bates al *framework* del surrogate, acaban con un input de trece dimensiones: cinco variables de estado (moneyness normalizada, tiempo a vencimiento, varianza instantánea, tipo de interés y *dividend yield*) y ocho parámetros estructurales (velocidad de reversión a la media de la volatilidad, varianza de largo plazo, volatilidad de la varianza, correlación entre precio y volatilidad, intensidad de saltos, tamaños medios de saltos positivos y negativos y probabilidad de que un salto sea positivo). La red aprende una función que recibe todos estos inputs y devuelve la volatilidad implícita Black-Scholes generada por Bates.

Hay un dato que conviene tener presente: el tamaño final de la muestra de entrenamiento que usan es de mil millones de observaciones. Aunque parezca un número muy elevado, los autores recalcan que en dimensión trece eso sigue siendo una muestra muy dispersa. La observación es relevante porque demuestra que la red no está memorizando una tabla densa sino aprendiendo una estructura funcional. Aquí entra una de las justificaciones teóricas más fuertes del enfoque: para ciertos modelos afines como Heston o Bates, hay resultados que muestran que el número de parámetros entrenables y el tamaño de la muestra necesarios para aproximar bien la solución crecen polinómicamente con la dimensión del input, no exponencialmente. Esta propiedad se sustenta en el trabajo clásico de Barron sobre aproximación de funciones con norma de Barron acotada y, más recientemente, en los resultados de Grohs, Hornung, Jentzen y von Wurstemberger, que demuestran formalmente cómo las redes neuronales profundas escapan a la maldición de la dimensionalidad en la aproximación de soluciones de la ecuación de Black-Scholes en alta dimensión.

### 3.2. Por qué redes neuronales

El paper compara explícitamente las redes neuronales profundas con otros métodos de aproximación que serían candidatos naturales, como polinomios, *splines*, *sparse grids* y *Gaussian processes*. Los polinomios funcionan razonablemente en funciones suaves pero les cuesta capturar rasgos locales fuertes como *kinks*, y eso en opciones es un problema serio porque el *payoff* tiene un *kink* claro en el strike; si se intenta aproximar con polinomios globales aparecen oscilaciones no deseadas, el fenómeno de Runge. Los *splines* capturan bien rasgos locales porque trabajan por tramos, pero escalan muy mal en alta dimensión. Las *sparse grids* resuelven parcialmente el problema de las mallas cartesianas, pero funcionan peor en dominios irregulares, y los modelos financieros muchas veces tienen restricciones entre parámetros que generan dominios complicados. Los *Gaussian processes* son potentes y elegantes pero escalan fatal con muchos datos porque las operaciones matriciales son del orden de `n^3`, así que con millones de observaciones se vuelven inviables. Las redes neuronales cumplen los cuatro criterios de forma razonable: funcionan en alta dimensión, capturan no linealidades, admiten dominios irregulares y escalan bien con muchos datos.

Hay además otra ventaja que conecta directamente con el espíritu de los métodos numéricos: el jacobiano de la red está disponible a un coste muy reducido mediante diferenciación automática. Esto es crítico en finanzas porque muchas magnitudes que importan no son el precio en sí sino sus derivadas. Si se trabaja con el modelo original, muchas veces hay que calcular derivadas mediante diferencias finitas, lo que eleva el coste computacional. Con una red neuronal, tanto el cálculo de *Greeks* como el proceso de calibración son mucho más baratos. De hecho, el paper documenta que estimar los parámetros con el método tradicional vía FFT lleva alrededor de ciento veinticinco horas, mientras que con el surrogate lleva poco más de dos horas en CPU y unas decenas de minutos en GPU. No es una mejora marginal sino una reducción de órdenes de magnitud.

### 3.3. Decisiones clave de diseño

Hay cuatro decisiones técnicas en el paper que nos parecen especialmente importantes y que incorporamos a nuestro propio enfoque.

La primera es la **activación suave en lugar de ReLU**. Intuitivamente uno pensaría que ReLU es la elección natural porque se parece al *payoff* de una *call*, con esa forma de cero hasta cierto punto y luego crecimiento lineal. Sin embargo los autores eligen Swish, que se puede ver como una versión suave de ReLU. La razón es que ReLU no es diferenciable en cero y, aunque eso puede no afectar mucho al precio aproximado, sí afecta a las derivadas que la red produce. En el paper se hace un experimento muy revelador donde entrenan dos surrogates idénticos salvo por la activación y encuentran que los errores de pricing son comparables pero los errores de Delta son sustancialmente mayores con ReLU. La lección que heredamos es que para un surrogate financiero no basta con mirar el error en precio, también hay que mirar la calidad de las derivadas, y la suavidad de la activación importa más para los *Greeks* que para el precio.

La segunda es la **volatilidad implícita como métrica de evaluación**. El paper convierte los precios a volatilidad implícita Black-Scholes y mide errores en esa escala. Comparar errores en precio puede ser muy engañoso porque un error de veinte céntimos no significa lo mismo en una opción que vale medio euro que en una que vale veinticinco, ni en una corta que en una larga, ni en una *in-the-money* que en una *out-of-the-money*. La volatilidad implícita actúa como unidad común que pone todas las opciones en una escala más homogénea. Es la unidad estándar del mercado y es una decisión metodológica que copiamos tal cual, aunque, como veremos en la sección 5, optamos por entrenar en escala de precio y evaluar también en IV para que la elección de métrica sea una variable explícita de comparación y no una decisión a priori.

La tercera es la **generación de datos sobre un hipercubo con muestreo flexible**. Para la generación del dataset se define el espacio de inputs como un hipercubo fijando mínimo y máximo para cada variable, se generan puntos aleatorios dentro y para cada punto se evalúa el modelo verdadero, formando así la muestra sintética. La distribución de muestreo no tiene por qué ser uniforme: puede ser uniforme simple, basada en *priors* jerárquicos, o concentrarse en zonas donde la función es más no lineal o donde la red está cometiendo más error, lo cual conecta con *active learning*. Hay también un detalle técnico importante: las redes suelen aproximar peor cerca de los bordes del dominio, así que los autores generan los datos en un rango ligeramente reducido, quitando un cinco por ciento por cada lado, para evitar que la evaluación quede dominada por errores de frontera.

La cuarta es la **lógica de entrenamiento distinta a la del ML clásico**. En *machine learning* normal se suele usar *early stopping*, *dropout* y otras técnicas de regularización para evitar *overfitting*. En un surrogate no es necesario porque los datos son sintéticos y casi sin ruido, ya que el *target* viene del propio modelo; si la red empieza a fallar en alguna zona siempre se pueden generar más datos en esos entornos. No estamos intentando aprender patrones inciertos a partir de datos limitados y ruidosos, sino aproximando una función determinista conocida pero cara de evaluar. Esa diferencia conceptual es la que justifica que los hiperparámetros y la lógica de entrenamiento sean distintos a los de un problema típico de ML.

### 3.4. Cómo validar la precisión del surrogate

La validación que hace el paper está separada en dos niveles. En el primer nivel suponen que los parámetros verdaderos son conocidos y comparan precio del modelo Bates resuelto con FFT contra precio del surrogate. Para hacerlo de forma seria dividen las opciones en doce grupos según vencimiento y moneyness, con cuatro categorías de tiempo a vencimiento y tres niveles de moneyness, y dentro de cada grupo generan veinte mil opciones aleatorias. Esa estructura por *bins* permite ver si el surrogate funciona bien en distintas zonas del mercado, no solo en promedio. Los resultados son muy buenos: el percentil noventa y cinco del error absoluto en volatilidad implícita y en Delta está por debajo de `8 × 10^-4` para opciones con más de siete días a vencimiento. Los errores son algo mayores en opciones muy cortas, lo cual tiene sentido porque ahí el *payoff* está más cerca, hay más no linealidad y las sensibilidades son más bruscas, pero incluso ahí los errores siguen siendo pequeños en términos relativos.

En el segundo nivel de validación, una vez probado que la red aproxima bien los precios para parámetros dados, los autores estudian si esos errores de aproximación se traducen en errores de estimación de parámetros cuando el surrogate se usa dentro de un procedimiento de calibración. La idea es que, en la práctica real, la estimación de parámetros del modelo se hace a partir de datos de mercado y ese paso introduce una nueva fuente de error. Para aislar este efecto se realiza un experimento de simulación controlado: se fija un conjunto de parámetros verdaderos, se simula con el solver original una superficie de precios de opciones que actúa como datos observados y se aplica el método generalizado de momentos usando el surrogate como evaluador del modelo dentro del bucle de optimización para recuperar una estimación de esos parámetros. Una vez obtenida la estimación, miden el error de pricing fuera de muestra comparando los precios que salen al evaluar el solver original con los parámetros estimados frente a los precios que salen al evaluarlo con los parámetros verdaderos, lo que les permite cuantificar el coste económico atribuible exclusivamente al hecho de haber calibrado con la red en lugar de con el solver. La distinción entre error de aproximación y error de estimación es metodológicamente importante y la mantenemos en mente al diseñar nuestras propias métricas.

---

## 4. Enfoque del trabajo

Después de digerir esta bibliografía, la decisión más importante es cómo enfocar nuestro propio trabajo para complementar y aportar valor al material existente dentro de nuestras limitaciones temporales y computacionales. La estructura más coherente es trabajar en dos niveles. El primer nivel es Black-Scholes como caso de validación. Black-Scholes en la práctica no necesita un surrogate porque la fórmula cerrada ya es muy rápida, pero precisamente por eso es ideal como entorno de prueba: conocemos la solución exacta, conocemos las Deltas analíticas y podemos comparar precio, volatilidad implícita y *Greeks* de forma directa antes de complicar el problema. El segundo nivel es extender a Heston. Heston introduce volatilidad estocástica, es bastante más realista y su evaluación requiere métodos numéricos como Fourier semi-cerrado, así que la motivación de tener un surrogate empieza a cobrar sentido real. Es un paso intermedio entre Black-Scholes y Bates que conserva la esencia del paper de referencia sin obligarnos a implementar saltos doble exponenciales.

A partir de ahí, hemos identificado cinco puntos en los que centrar nuestro proyecto de investigación. Los enumeramos aquí en forma de preguntas y los desarrollamos en la sección 6 como experimentos formales.

El primero es comparar de forma sistemática el error en precio frente al error en volatilidad implícita. Planteamos como pregunta metodológica explícita hasta qué punto puede un surrogate tener un error en precio aparentemente bajo pero un error relevante en *implied volatility* en ciertas zonas del dominio. Esto acerca el trabajo a la práctica real del mercado, donde la volatilidad implícita es la unidad natural de comparación.

El segundo elemento diferencial es comparar funciones de activación con foco en la calidad de los *Greeks*. Entrenaremos redes con la misma arquitectura cambiando solo la activación, probando ReLU, *Softplus*, *Swish/SiLU* y *tanh*, y compararemos el MAE de precio, el MAE de Delta y el MAE de volatilidad implícita en cada caso. La hipótesis, alineada con Chen et al., es que las activaciones suaves producen derivadas más estables aunque los errores de precio sean parecidos. Mostrar visualmente que dos redes con MAE de precio similar pueden tener Deltas muy distintas es una conclusión potente para un trabajo de métodos numéricos.

El tercer elemento es comparar muestreo uniforme frente a muestreo enfocado. Generaremos dos *datasets*, uno con muestreo uniforme en todo el dominio y otro con más densidad cerca del *at-the-money* y de vencimientos cortos, que es donde sabemos por adelantado que la función de pricing tiene más curvatura. Evaluaremos ambos surrogates con la misma división por *bins* y analizaremos si el muestreo enfocado reduce el error en regiones críticas. Esto conecta directamente con ideas clásicas de métodos numéricos como la elección adaptativa de puntos de malla.

El cuarto elemento es la medición seria de eficiencia computacional. En Black-Scholes la ganancia será probablemente pequeña porque la fórmula cerrada ya es rapidísima, pero el experimento sirve para validar el pipeline. En Heston mediremos el tiempo necesario para valorar grandes volúmenes de opciones con el solver original frente al surrogate, en función del tamaño del lote, porque la red aprovecha mejor el paralelismo cuando procesa muchos puntos a la vez y reportar un único número aislado sería engañoso.

El quinto elemento, central en nuestro planteamiento, es estudiar el efecto de la pérdida en la calidad del surrogate con especial atención al enfoque de *differential machine learning* de Huge y Savine. En lugar de entrenar la red solo con precios, modificamos la pérdida para que también penalice la diferencia entre la Delta verdadera y la Delta que sale por *autograd* de la red. De este modo, la pérdida queda como una combinación ponderada del error de precio y del error de la derivada y la red aprende simultáneamente a acertar el nivel y la pendiente de la función. El planteamiento tiene una motivación matemática sólida, análoga a la que justifica la interpolación de Hermite frente a la clásica: disponer del valor y de la derivada en cada punto muestreado permite reconstruir la función con muchos menos puntos. La hipótesis asociada es que el *differential machine learning* ofrece una ventaja sustancial frente al entrenamiento con pérdidas estándar en regímenes de *datasets* pequeños y que esa ventaja se atenúa conforme crece el tamaño de la muestra. Esto nos permite cuantificar bajo qué condiciones compensa el coste adicional de calcular gradientes verdaderos durante la generación del *dataset*.

---

## 5. Metodología

Esta sección concreta cómo se traducen las preguntas anteriores en un pipeline experimental ejecutable. Su función no es añadir más tareas sino fijar cómo se ejecuta cada experimento, qué magnitudes se entrenan, qué métricas se reportan y bajo qué condiciones consideramos que una comparación es limpia. En un proyecto de *surrogates*, la parte delicada no es solo entrenar redes, sino asegurar que el dato sintético, la normalización, la métrica y la evaluación responden exactamente a la pregunta que se quiere contestar.

La idea central del proyecto es combinar dos líneas de la literatura. De Chen, Didisheim y Scheidegger tomamos la lógica de *deep surrogates*: construir una red que aproxime un solver financiero caro a partir de datos sintéticos generados por el propio modelo. De Huge y Savine tomamos, de forma acotada, la idea de *differential machine learning*: si además del valor de la función conocemos una derivada económicamente relevante, podemos enseñar a la red la forma local de la función y no solo su nivel. En nuestro caso, esa derivada será la Delta.

### 5.1. Pipeline experimental

Todos los experimentos siguen el mismo flujo base. Primero se define un dominio de inputs y una distribución de muestreo. Después se genera un *dataset* sintético evaluando el solver verdadero en cada punto. A continuación se entrena un surrogate con una configuración cerrada y versionada. Finalmente se evalúa el surrogate sobre un conjunto independiente, separando los resultados por *bins* de moneyness y vencimiento. La separación entre generación, entrenamiento y evaluación es importante porque evita contaminar las conclusiones: el *sampler* decide dónde preguntamos al modelo verdadero, el solver decide cuáles son los *targets*, el *trainer* decide cómo aprende la red y el evaluador decide cómo medimos el error. Si una comparación experimental cambia más de una de estas piezas al mismo tiempo, deja de estar claro qué causa la diferencia observada.

### 5.2. Targets, normalización y métricas

El *target* principal de entrenamiento es el precio normalizado por strike, `y = C/K`, y el input financiero principal es la moneyness simple, `m = S/K`. Esta decisión elimina una escala redundante del problema: si spot y strike se multiplican por la misma constante, el precio de una *call* europea escala con el strike. Trabajar con `C/K` y `S/K` permite que la red aprenda una función más estable y facilita la interpretación de la Delta, porque

```text
Delta = dC/dS = d(C/K)/d(S/K) = dy/dm.
```

No usamos la moneyness normalizada de Chen et al. porque nuestro objetivo no es replicar la escala empírica del S\&P 500, sino construir un entorno controlado donde la relación entre precio normalizado y Delta sea directa. También fijamos `q = 0` en todos los experimentos: el *dividend yield* no forma parte de las preguntas centrales del trabajo y eliminarlo reduce la dimensión del dominio sin afectar a las comparaciones entre métricas, activaciones, muestreo, eficiencia y aprendizaje diferencial.

Además de esta normalización financiera, todos los inputs se normalizan numéricamente a `[0, 1]` antes de entrar en la red. Esta normalización no cambia el problema matemático pero mejora la estabilidad del optimizador porque evita que dimensiones como `T`, `rho`, `v0` o `kappa` vivan en escalas muy distintas. Cuando se calculan derivadas de la red se aplica la regla de la cadena para volver a la escala financiera. En particular, si `m_norm = (m - m_min) / (m_max - m_min)`, entonces

```text
Delta_hat = d y_hat / d m = (d y_hat / d m_norm) / (m_max - m_min).
```

La Delta del surrogate se calcula con *autograd* sobre la red, no con diferencias finitas. La Delta verdadera del *dataset* viene del solver (`N(d1)` en Black-Scholes y `P1` en Heston), mientras que la Delta predicha se obtiene derivando `y_hat` respecto a `m_norm` y aplicando la corrección anterior. En evaluación se usa `create_graph=False` porque solo se mide la derivada. En el experimento que entrena con pérdida diferencial se usa `create_graph=True` porque la pérdida contiene `MAE(Delta)` y PyTorch necesita derivar esa pérdida respecto a los pesos de la red. Esto incrementa coste y memoria pero es el coste esperado del entrenamiento diferencial.

La volatilidad implícita Black-Scholes no será el *target* principal de entrenamiento, sino una métrica de evaluación. Esta elección mantiene el entrenamiento más directo, porque el solver genera precios, y evita introducir fallos de inversión de volatilidad implícita durante la generación del *dataset*. Al mismo tiempo, evaluar también en IV permite comprobar si un error pequeño en precio es realmente pequeño en la escala que se usa en mercado.

Las métricas mínimas de evaluación son `MAE(C/K)`, el error absoluto en precio normalizado, `MAE_IV` y `MAE_Delta`. Para cada una se reportan promedios y percentiles por *bin*, con especial atención al percentil 95, porque los errores de cola son los que revelan si el surrogate falla en regiones operativamente importantes.

### 5.3. Solvers y validación numérica

Black-Scholes funciona como entorno de control. El precio, la Delta, la Vega y la inversión a volatilidad implícita tienen fórmulas cerradas o procedimientos numéricos muy estables, así que cualquier fallo en Black-Scholes apunta a un error del *pipeline* y no a una dificultad intrínseca del modelo.

Heston es el caso principal. El precio se calcula mediante una formulación semi-cerrada de Fourier, expresada en términos de las probabilidades `P1` y `P2`,

```text
C = S e^{-qT} P1 - K e^{-rT} P2.
```

Esta formulación tiene una ventaja metodológica importante para el experimento de DML: la Delta de una *call* europea se obtiene como `Delta = e^{-qT} P1`. Si fijamos `q = 0`, entonces `Delta = P1`. Esta Delta es el *target* diferencial del experimento E5. Antes de generar *datasets* con Delta, la implementación se valida contra diferencias finitas centrales en una *grid* pequeña de puntos representativos del dominio. La validación no busca que las diferencias finitas sean el método principal, sino detectar errores de implementación, escalado o inestabilidad numérica. El criterio de aceptación es error absoluto medio inferior a `10^-4` en Delta; los errores extremos se revisan manualmente porque pueden señalar problemas de integración, vencimientos demasiado cortos o puntos cercanos a frontera.

Más allá de los tests internos, la implementación se valida también de forma cruzada con QuantLib sobre una *grid* representativa de inputs. Esta validación externa cubre precio y Delta en Black-Scholes y en Heston, y es lo que cierra la primera fase del proyecto antes de generar *datasets* grandes. Mantener esta capa de validación es lo que permite descartar que cualquier patrón posterior se deba a un error silencioso en el solver, escenario que contaminaría todo el resto del trabajo.

La inversión a volatilidad implícita se usa solo en evaluación. El inversor debe controlar precios fuera de límites de arbitraje, zonas de Vega baja y fallos de convergencia. Los fallos no se ocultan: se reportan como parte del diagnóstico, especialmente en vencimientos muy cortos o regiones muy fuera del dinero.

### 5.4. Dominio y muestreo

Los *datasets* se generan muestreando un hipercubo de inputs. Esta decisión sigue la lógica de Chen et al.: tratamos los parámetros del modelo como pseudo-estados para que una única red aprenda la función de pricing en muchas parametrizaciones posibles. La distribución base es uniforme, porque cubre el dominio completo y no replica únicamente las regiones más frecuentes del mercado.

El dominio contractual común es amplio y orientado a *trading*:

| Variable | Rango | Uso |
|---|---|---|
| `m = S/K` | `[0.4, 2.0]` | Black-Scholes y Heston |
| `T` | `[7/365, 2.0]` | Black-Scholes y Heston |
| `r` | `[0.00, 0.075]` | Black-Scholes y Heston |
| `q` | `0` | Black-Scholes y Heston |
| `sigma` | `[0.03, 1.00]` | Black-Scholes |

Este dominio incluye opciones semanales, *wings* profundas y volatilidades de estrés. Es deliberadamente más exigente que un dominio centrado en `m ∈ [0.6, 1.4]` y vencimientos cómodos porque esas regiones extremas son relevantes en *trading* y son precisamente donde precio, IV y Delta pueden contar historias distintas. La contrapartida es que la evaluación debe ser más granular para no mezclar zonas fáciles con zonas difíciles.

Para Heston, el dominio contractual se combina con un hipercubo propio de parámetros del proceso:

| Variable Heston | Rango | Interpretación |
|---|---|---|
| `v0` | `[0.0009, 1.00]` | `sqrt(v0)` entre 3% y 100% |
| `theta` | `[0.0009, 1.00]` | `sqrt(theta)` entre 3% y 100% |
| `kappa` | `[0.10, 10.00]` | Reversión lenta a rápida |
| `xi` | `[0.10, 3.00]` | Volatilidad de la varianza moderada a extrema |
| `rho` | `[-0.95, -0.05]` | *Skew* negativo típico de *equity* |

La elección de `v0` y `theta` se expresa en varianza pero se interpreta a través de su raíz cuadrada porque esa es la escala financiera natural. El rango cubre desde volatilidades bajas hasta episodios de estrés severo sin forzar al solver a vivir permanentemente en casos extremos. `kappa` permite tanto reversión lenta como rápida a la media, pero evita el rango muy agresivo de Chen et al. porque nuestro *dataset* será varios órdenes de magnitud menor. `xi` mantiene suficiente variabilidad para generar *smiles* pronunciadas, aunque acota los casos donde la integración de Heston se vuelve frágil. `rho` se restringe a valores negativos porque el proyecto se centra en el comportamiento típico de *equity*, donde la correlación negativa entre *spot* y varianza es la fuente principal del *skew*.

No imponemos la condición de Feller `2 kappa theta >= xi^2` como restricción de muestreo. Tratarla como filtro obligatorio haría el problema más limpio pero menos representativo de calibraciones reales, donde muchas violaciones aparecen en condiciones de mercado normales. En su lugar, se guarda como variable diagnóstica: el informe debe indicar qué porcentaje de puntos la cumple y separar los descartes debidos a fallos numéricos o violaciones evidentes de cotas de no arbitraje.

El muestreo *baseline* es uniforme en la escala financiera elegida. Para Black-Scholes, esto significa uniforme directo sobre `m`, `T`, `r` y `sigma`. Para Heston, `sqrt(v0)` y `sqrt(theta)` se muestrean uniformemente en `[0.03, 1.00]` y después se elevan al cuadrado; `kappa`, `xi` y `rho` se muestrean uniformemente en sus rangos. Esta decisión conserva la simplicidad del esquema de Chen et al. pero evita una distorsión importante: una uniforme directa sobre varianza sobreponderaría volatilidades muy altas.

La comparación de métodos de muestreo se concentra únicamente en el experimento E3. El resto de experimentos usa el *baseline* uniforme para que las diferencias observadas se puedan atribuir a la variable que cada experimento modifica. En E3 se añade un *sampler* enfocado, con más masa cerca del *at-the-money* y de vencimientos cortos, donde la función de pricing tiene mayor curvatura. La construcción concreta es una mezcla 50/50: la mitad de las muestras sigue el *baseline* uniforme completo y la otra mitad usa `m ~ TruncNormal(1.0, 0.15, [0.7, 1.3])` y `T ~ LogUniform(7/365, 0.25)`, manteniendo `r` y los parámetros de Heston con el mismo muestreo *baseline*. Esto concentra datos en ATM y vencimientos cortos sin abandonar la cobertura global del dominio.

### 5.5. Particiones y evaluación por bins

La partición de datos tiene tres niveles. El entrenamiento usa la distribución propia de cada surrogate. La validación usa una muestra independiente uniforme de cincuenta mil puntos por familia de modelo y sirve para monitorizar convergencia, activar el *scheduler* y seleccionar el *checkpoint*. La regla de selección es siempre el menor `MAE(C/K)` en validación, incluso cuando la pérdida de entrenamiento incluye Delta. Mantener la regla fija es importante para que las comparaciones sean limpias: si el surrogate entrenado con DML se seleccionase con una métrica combinada de precio y Delta, no sabríamos si la mejora viene de la pérdida diferencial o de haber escogido otro punto de la trayectoria de entrenamiento.

El test final es independiente y balanceado por *bins*, y se comparte por todos los surrogates de una misma familia de modelo. Cada uno de los veinticinco *bins* tiene cinco mil puntos, para un total de ciento veinticinco mil observaciones. Esta decisión es especialmente importante en E3: si evaluásemos H-5 en una muestra con la misma concentración con la que entrena, confundiríamos precisión local con mejora global. El test balanceado permite ver exactamente dónde gana y dónde pierde el *sampler* enfocado.

Las métricas principales se reportan en un dominio interior que recorta un cinco por ciento de cada dimensión del hipercubo. Las redes tienden a aproximar peor en fronteras porque no tienen vecinos a ambos lados de la función. Separar el dominio interior de la frontera evita que el promedio global quede dominado por errores de borde y hace que las conclusiones sobre el comportamiento en el interior sean más limpias.

La evaluación por *bins* es obligatoria porque los errores medios globales son insuficientes. Una red puede tener buen MAE agregado y, aun así, fallar sistemáticamente en opciones ATM de vencimiento corto o en regiones OTM donde la Vega es baja. Por eso usamos una partición `5 × 5` por moneyness y vencimiento.

| Moneyness | Vencimiento |
|---|---|
| Deep OTM `[0.4, 0.7)` | Weekly `[7/365, 14/365)` |
| OTM `[0.7, 0.9)` | Short `[14/365, 1/12)` |
| ATM `[0.9, 1.1]` | Medium-short `[1/12, 0.25)` |
| ITM `(1.1, 1.3]` | Medium `[0.25, 1.0)` |
| Deep ITM `(1.3, 2.0]` | Long `[1.0, 2.0]` |

Cinco niveles de moneyness por cinco de vencimiento dan veinticinco *bins*. Esta partición es más granular que la de Chen et al. porque queremos separar explícitamente *wings* profundas y opciones semanales, dos regiones relevantes para *trading* y numéricamente más delicadas. Reportar percentiles altos en lugar de solo el MAE es importante porque la cola de la distribución de errores es donde aparecen los problemas operativos: una opción con error puntual grande puede arruinar un cálculo de cobertura aunque el promedio esté bien.

### 5.6. Constantes del pipeline

Para evitar reabrir discusiones durante la ejecución, fijamos los siguientes valores en código y no los movemos salvo que un hallazgo experimental lo justifique. Cada uno tiene una razón concreta que documentamos a continuación para que el porqué no se pierda con el tiempo.

| Parámetro | Valor |
|---|---|
| Arquitectura | MLP, 4 capas ocultas, 128 unidades |
| Activación por defecto | Swish |
| Optimizador | Adam, `lr=10^-3`, *scheduler reduce-on-plateau* |
| *Batch size* | 1024 (*baseline*) |
| Épocas máximas | 100 |
| Regularización | ninguna (datos sintéticos) |
| Moneyness | `m = S/K` |
| Normalización de inputs | min-max a `[0, 1]` |
| *Output* | precio normalizado `C/K` |
| *Dividend yield* | `q = 0` |
| Recorte de bordes | 5% en cada cara del hipercubo |
| Semilla | fijada por configuración |

Un MLP *feed-forward* simple es suficiente porque el teorema universal de aproximación garantiza que basta para funciones suaves en dimensión moderada, y Chen et al. confirman empíricamente que funciona en Bates en dimensión trece. Cuatro capas ocultas y 128 unidades por capa dan del orden de cincuenta mil parámetros, lo que para *datasets* de cien mil a quinientas mil muestras supone entre dos y diez muestras por parámetro, una ratio cómoda. Chen et al. usan redes más profundas pero su problema tiene trece dimensiones y mil millones de muestras; nosotros estamos en una escala menor y escalamos abajo de forma proporcional. Swish entra como activación por defecto porque, según Chen, es la decisión clave para que las Deltas obtenidas por *autograd* sean estables. Adam con `lr=10^-3` es el *default* de PyTorch y converge sin sintonización en la inmensa mayoría de problemas de regresión, y el *scheduler reduce-on-plateau* es lo más simple que funciona: cuando la pérdida de validación se estanca, divide el *learning rate* por dos.

La ausencia de regularización es la decisión metodológica más importante y viene directamente de Chen. En aprendizaje automático clásico la regularización existe porque los datos llevan ruido y la red tendería a memorizarlo, lo que produce *overfitting*. En un surrogate, los *targets* vienen del propio solver, son deterministas y no hay nada que memorizar; la pérdida de validación nunca empeora por *overfitting*, solo se estanca cuando la red llega al límite de su capacidad expresiva. Por eso no usamos *early stopping*, *dropout* ni *weight decay*. El tope de cien épocas funciona como límite práctico para que el entrenamiento no se eternice si la pérdida deja de bajar antes de tiempo.

---

## 6. Diseño experimental

Cinco experimentos cubren los cinco elementos diferenciales descritos en la sección 4. La elección de exactamente cinco no es arbitraria: es la lectura directa del bloque de elementos diferenciales, traducido a comparaciones experimentales concretas. Cada experimento responde a una pregunta única, varía una sola dimensión y mantiene todo lo demás fijo. Esta rigidez es lo que permite atribuir cualquier diferencia observada a la dimensión que estamos variando y no a otras variables que se nos hayan colado por accidente.

| ID | Pregunta | Lo que varía | Surrogates implicados | Resultado esperado |
|---|---|---|---|---|
| **E1** | ¿Un error bajo en precio implica un error bajo en volatilidad implícita? | Métrica de evaluación | BS-3, H-3 | *Bins* donde el MAE de precio es bajo pero el MAE de IV no lo es. Si aparecen, la elección de métrica importa. |
| **E2** | ¿Qué función de activación produce mejores *Greeks* sin degradar el precio? | Activación (ReLU, *Softplus*, Swish, *tanh*) | BS-1..4, H-1..4 | MAE de precio similar entre todas las activaciones, MAE de Delta sustancialmente mayor en ReLU. |
| **E3** | ¿El muestreo enfocado en zonas de mayor curvatura reduce el error en *bins* críticos? | Distribución del muestreo | H-3 vs H-5 | H-5 con menor MAE en *bins* ATM y vencimiento corto, posiblemente peor en *bins* extremos. |
| **E4** | ¿Qué *speedup* ofrece el surrogate frente al solver original? | Tamaño del lote de evaluación | H-3 | *Speedup* creciente con el tamaño del lote, especialmente notable en lotes grandes. |
| **E5** | ¿Entrenar con Delta como *target* diferencial mejora la eficiencia muestral del surrogate? | *Targets* de entrenamiento | H-3-small, H-6-small, H-3 | H-6-small con mejor Delta y precisión de precio comparable a H-3-small; idealmente se acerca a H-3 usando menos datos. |

E1 es observacional. No requiere entrenar nada nuevo, simplemente recalcula métricas sobre un surrogate que ya está entrenado para otros experimentos. La razón por la que merece llamarse experimento aparte es que la pregunta que responde es metodológica: nos dice si dos surrogates con MAE de precio aparentemente similares pueden tener errores muy distintos en la métrica que de verdad importa en el mercado, que es la volatilidad implícita. Si la respuesta es afirmativa, la elección de métrica de validación deja de ser una decisión técnica menor y pasa a ser parte del diseño del proyecto.

E2 es el experimento más cargado en surrogates porque la activación es la única decisión arquitectónica que afecta directamente la calidad de las derivadas. Entrenar las cuatro activaciones sobre Black-Scholes nos sirve como control en un entorno donde conocemos la solución exacta y las Deltas analíticas, y entrenarlas sobre Heston confirma o desmiente el patrón en el caso real. La hipótesis, alineada con Chen et al., es que las activaciones suaves dan Deltas mucho mejores que ReLU aunque el precio sea parecido.

E3 ataca una idea clásica de métodos numéricos transferida al dominio de los *surrogates*: concentrar el esfuerzo computacional en las regiones donde la función es más difícil de aproximar. La función de pricing tiene más curvatura cerca del *at-the-money* y en vencimientos cortos, así que sembrar más densidad de muestreo ahí debería traducirse en errores más bajos en esos *bins*. La comparación es limpia porque H-3 y H-5 difieren solo en la distribución de muestreo: misma arquitectura, mismo número total de puntos, misma pérdida y mismo tamaño total del *dataset*.

E4 es de medición, no de comparación. Cronometra evaluaciones de H-3 contra el solver original de Heston en lotes de tamaño creciente. La razón para hacerlo con lotes y no con evaluaciones unitarias es que las redes amortiguan costes cuando procesan muchos puntos en paralelo, así que el *speedup* real solo se ve a partir de cierto tamaño. Reportar el *speedup* como función del tamaño de lote es más honesto que dar un único número que dependería arbitrariamente del lote elegido.

E5 prueba la hipótesis central del paper de Huge y Savine en nuestro *setting* de forma acotada: si además de precios generamos Delta verdadera, ¿puede el surrogate aprender mejor la forma local de la función de pricing con menos datos? Delta es la *Greek* más limpia para este primer experimento porque, usando `y = C/K` y `m = S/K`, se cumple `Delta = dy/dm`. Vega, Gamma, Theta y Rho quedan como métricas diagnósticas o extensiones posteriores, pero no forman parte de la pérdida principal de E5. Las sensibilidades respecto a parámetros internos de Heston quedan fuera del alcance principal porque son riesgos de modelo y calibración, no *Greeks* operativas de *trading*.

Cada experimento tiene una métrica primaria alineada con su pregunta y métricas secundarias como control. Esto evita que el resultado se reduzca a una tabla grande sin conclusión clara.

| ID | Métrica primaria | Controles y diagnóstico |
|---|---|---|
| E1 | Discrepancia entre `MAE(C/K)` y `MAE_IV` por *bin* | Sin umbral fuerte/débil; Vega/proxy, percentiles altos y fallos de IV |
| E2 | `MAE_Delta` por *bin* | Sin umbral; robustez de activaciones suaves vs ReLU y `MAE(C/K)` como control |
| E3 | `MAE_IV` en *bins* ATM con weekly, short y medium-short | Positivo fuerte si mejora ≥10% y `MAE_IV` global empeora ≤10%; `MAE(C/K)` como control |
| E4 | `speedup = tiempo_solver / tiempo_surrogate` por tamaño de lote | Lotes `10^2..10^5`, 3 *warmups* + 10 repeticiones, mediana y CPU/GPU separado |
| E5 | Mejora de `MAE_Delta` de H-6-small frente a H-3-small | Positivo fuerte si Delta mejora ≥20% y precio empeora ≤10%; distancia frente a H-3 |

### 6.1. Once surrogates

Once surrogates en total. La cifra sale de trabajar hacia atrás desde los cinco experimentos: cada uno necesita un conjunto mínimo de surrogates entrenados para sostener su comparación y muchos se solapan. H-3, el surrogate de Heston con configuración *baseline*, aparece como referencia en los cinco experimentos. Esa centralidad evita que el coste computacional total se dispare y justifica que dediquemos especial cuidado a su entrenamiento.

| ID | Modelo | Activación | Pérdida | *Dataset* | Aporta a |
|---|---|---|---|---|---|
| BS-1 | Black-Scholes | ReLU | L1(precio) | 200k uniforme | E2 |
| BS-2 | Black-Scholes | *Softplus* | L1(precio) | 200k uniforme | E2 |
| **BS-3** | **Black-Scholes** | **Swish** | **L1(precio)** | **200k uniforme** | **E1, E2** |
| BS-4 | Black-Scholes | *tanh* | L1(precio) | 200k uniforme | E2 |
| H-1 | Heston | ReLU | L1(precio) | 500k uniforme | E2 |
| H-2 | Heston | *Softplus* | L1(precio) | 500k uniforme | E2 |
| **H-3** | **Heston** | **Swish** | **L1(precio)** | **500k uniforme** | **E1, E2, E3, E4, E5 baseline** |
| H-4 | Heston | *tanh* | L1(precio) | 500k uniforme | E2 |
| H-5 | Heston | Swish | L1(precio) | 500k enfocado | E3 |
| H-3-small | Heston | Swish | L1(precio) | 100k uniforme | E5 |
| H-6-small | Heston | Swish | L1(precio) + L1(Δ), pesos 1:1 | 100k uniforme con Delta | E5 |

Todos los surrogates comparados dentro de un mismo experimento comparten arquitectura, optimizador, hiperparámetros de entrenamiento y semilla. Esta rigidez es deliberada y es lo que permite atribuir las diferencias observadas a la dimensión que cada experimento varía. En E5 sí varía deliberadamente el tamaño del *dataset* porque la pregunta no es solo si DML mejora las *Greeks*, sino si permite alcanzar precisión comparable con menos muestras. Si dejásemos variar la arquitectura entre surrogates, la comparación de activaciones quedaría contaminada con el efecto del cambio arquitectónico, por mencionar el caso más obvio.

### 6.2. Criterios de validez

Cada experimento se cierra con una comparación que cambia una sola dimensión relevante. Si se cambia activación, no se cambia *dataset*. Si se cambia *sampler*, no se cambia arquitectura. Si se cambia *loss* en E5, se mantiene el mismo tamaño pequeño de *dataset* para H-3-small y H-6-small. Esta disciplina es lo que permite atribuir los resultados a la variable experimental y no a decisiones accidentales de implementación. Para E3 y E5, en los que sí buscamos una clasificación operativa del resultado, los umbrales *positivo fuerte/débil/negativo* se fijan antes de ver los datos. Esa regla previa es lo que evita interpretar resultados de forma oportunista cuando llegue el momento de redactar conclusiones.

---

## 7. Arquitectura del sistema y reproducibilidad

Esta sección recoge las decisiones de diseño que estructuran el código del proyecto. La idea es que cualquiera que abra el repositorio dentro de tres meses pueda entender por qué los ficheros están donde están, por qué las clases tienen los nombres que tienen y por qué algunas decisiones aparentemente menores son las que evitan que el proyecto se vuelva inmantenible a mitad de camino.

### 7.1. Principios de diseño

Tres principios guían toda la arquitectura. El primero es la separación estricta entre código reutilizable, configuración por experimento, *scripts* de entrada y salidas. El código vive en `src/` y se importa como librería; la configuración vive como ficheros versionados; los *scripts* solo orquestan llamadas sin contener lógica reutilizable; los datos y resultados quedan en `data/` y `results/` respectivamente. Esta separación es lo que evita que un cambio en una parte del *pipeline* rompa partes que no estábamos tocando.

El segundo principio es el uso de orientación a objetos donde aporta polimorfismo real. Cuando un mismo concepto tiene varias implementaciones que se intercambian entre experimentos (*solvers*, *samplers*, pérdidas, experimentos), creamos una clase base abstracta que define la interfaz y clases concretas que la implementan. El resto del *pipeline* depende de la interfaz, no de la implementación. Eso es lo que permite que un evaluador por *bins* evalúe *surrogates* de Black-Scholes y de Heston sin enterarse de qué modelo hay por debajo, simplemente recibe un *pricer* por inyección. Donde solo hay una implementación posible o el código es puramente algorítmico, usamos funciones libres y no creamos clases por crear.

El tercer principio es la reproducibilidad por configuración. Cada uno de los once *surrogates* está completamente descrito por una configuración pequeña y versionada. Reentrenar un *surrogate* dentro de seis meses es cargar la configuración y volver a llamar al entrenador. Las semillas están centralizadas, las dependencias congeladas en `requirements.txt` y los hiperparámetros viven en código bajo control de versiones. Si un resultado no se puede reproducir desde un *commit*, hay un *bug* en la arquitectura.

### 7.2. Estructura del repositorio

La organización del repositorio refleja esos principios:

```text
proyecto-final-metodos-numericos/
├── docs/                      Documentación del proyecto
├── papers/                    Bibliografía consultada
├── src/                       Librería del proyecto, importable como paquete
│   ├── solvers/               Pricers analíticos y numéricos
│   ├── datasets/              Muestreo y generación de datasets sintéticos
│   ├── models/                Arquitectura de la red y Greeks del surrogate
│   ├── training/              Bucle de entrenamiento, pérdidas, configuración
│   ├── evaluation/            Binning, métricas, eficiencia
│   ├── experiments/           Lógica de cada uno de los cinco experimentos
│   └── utils/                 Semillas, entrada/salida
├── configs/                   Una config por surrogate
├── scripts/                   Puntos de entrada ejecutables
├── data/                      Datasets generados, gitignored
├── results/                   Checkpoints, métricas y figuras versionadas
├── tests/                     pytest
├── requirements.txt
└── README.md
```

La distinción entre `src/` y `scripts/` es deliberada. Todo lo que es código reutilizable, testeable y que importará el resto del proyecto vive bajo `src/`. Los *scripts* son los puntos donde el usuario interactúa con el sistema y solo contienen orquestación: leer una configuración, llamar a las clases de `src/`, escribir resultados. Si dentro de un *script* aparece lógica que merezca ser reutilizada por otro *script*, se promociona a `src/`.

### 7.3. Diseño orientado a objetos por módulo

El módulo `src/solvers/` define la jerarquía que sostiene todo el proyecto. La clase base abstracta `OptionPricer` declara la interfaz mínima que cualquier *pricer* debe cumplir: devolver precio, devolver Delta cuando esté disponible y reportar su dimensión de entrada. Las dos implementaciones concretas son `BlackScholesSolver`, con fórmulas cerradas para precio, Delta y Vega, y `HestonSolver`, que implementa la formulación semi-cerrada de Fourier con `P1` y `P2` mediante integración numérica. El inversor de volatilidad implícita vive como clase aparte porque su lógica es ortogonal al *pricing* y se invoca tanto durante la evaluación como dentro de algunos análisis. El polimorfismo del *solver* es lo que permite que el generador de datos, el evaluador y el *benchmark* de eficiencia reutilicen el mismo código entre Black-Scholes y Heston.

El muestreo del hipercubo se modela como una jerarquía `Sampler` con varias implementaciones: `UniformSampler`, `FocusedSampler` y `BalancedBinSampler`. Un `Domain` (dataclass) encapsula los rangos del hipercubo, transformaciones como muestrear `sqrt(v0)` y `sqrt(theta)` antes de convertir a varianza, el recorte de bordes y diagnósticos como la condición de Feller, y se inyecta tanto en el *sampler* como en el evaluador. La clase `DatasetGenerator` compone un `OptionPricer` y un `Sampler` para producir `(x, precio, Delta opcional)`, filtra fallos numéricos claros del solver y materializa el resultado como un `OptionDataset` heredero de `torch.utils.data.Dataset`. Esta composición por inyección de dependencias es lo que hace que añadir un nuevo *solver* o un nuevo *sampler* no requiera tocar el generador.

La arquitectura de la red vive en una sola clase `MLP` que hereda de `nn.Module`. Su constructor recibe la dimensión de entrada, el ancho de capa oculta, el número de capas y la activación como *string*, que se resuelve internamente a través de un diccionario que mapea nombres a clases de PyTorch. Esto permite que E2 cambie el comportamiento del modelo con una sola entrada en la configuración, sin tocar el código de la red. No hay jerarquía de modelos porque no la necesitamos: usamos un único tipo de red y solo varían sus hiperparámetros.

El entrenamiento se divide en tres piezas. La jerarquía de pérdidas tiene `SurrogateLoss` como base abstracta y dos implementaciones concretas, `PriceLoss` para el régimen base y `DifferentialLoss` que combina `MAE(C/K)` y `MAE(Delta)` con pesos 1:1. La Delta predicha por el surrogate se calcula en una utilidad única que deriva el *output* de la red respecto a `m_norm` y divide por `(m_max - m_min)`. Una *dataclass* `TrainConfig` serializa toda la información necesaria para reproducir un entrenamiento, y un `Trainer` orquesta el bucle, recibe modelo, pérdida, optimizador, *scheduler* y configuración, y expone los métodos `fit`, `save` y `load`. El `Trainer` no sabe nada sobre el problema concreto que está resolviendo; es código que ejecutaría el mismo bucle si en lugar de *surrogates* financieros estuviésemos entrenando otra cosa.

El módulo de evaluación encapsula todo el protocolo de medida. `BinPartition` define la rejilla `5 × 5` y sabe asignar cada punto a su *bin* correspondiente. `BinEvaluator` recibe una partición y un `OptionPricer` y expone un método `evaluate` que toma un surrogate y un test set y devuelve un *Report* con MAE de precio, MAE de IV y MAE de Delta, además de percentiles por *bin*. Una clase `TimingBenchmark` implementa el protocolo de E4: recibe un surrogate y un solver, ejecuta tres *warmups* y diez repeticiones medidas por cada lote `10^2`, `10^3`, `10^4` y `10^5`, y devuelve mediana, p25/p75 y *speedup* por lote, separando CPU y GPU cuando aplique.

Por último, cada uno de los cinco experimentos del proyecto se materializa como una clase concreta que hereda de `Experiment`. La clase base define el método `run`, que es lo único que los *scripts* de análisis llaman. Las cinco clases concretas reciben los *surrogates* ya entrenados como dependencias y producen un *Report* específico. La uniformidad de la interfaz es lo que hace que un *script* de análisis se reduzca a tres líneas: instanciar el experimento, llamar `run`, volcar el *Report*.

### 7.4. Estrategia de testing

El proyecto usa `pytest` con tres capas de tests. La primera capa son tests unitarios de los *solvers* contra valores tabulados de referencia, validados de forma cruzada con QuantLib sobre una *grid* representativa. Estos tests son los más críticos del proyecto porque un fallo silencioso en un *solver* contaminaría todo el resto del trabajo sin que nos enterásemos. La segunda capa son tests unitarios de las piezas auxiliares: el *sampler* debe respetar los límites del hipercubo, el inversor de IV debe converger en casos conocidos, la Delta numérica debe validarse contra diferencias finitas en una *grid* pequeña, y la partición por *bins* debe asignar correctamente los puntos. La tercera capa son tests de integración del *pipeline* completo sobre un *mini-dataset* sintético de Black-Scholes con cien puntos. El objetivo aquí no es validar la calidad del entrenamiento sino verificar que las piezas encajan. La inyección de dependencias que practicamos en todo el código es lo que hace estos tests posibles: cada clase recibe sus dependencias por constructor, lo que permite testearlas con *mocks* o con implementaciones triviales.

---

## 8. Plan de ejecución por fases

El proyecto se divide en cuatro fases ordenadas. Cada una no empieza hasta que la anterior está cerrada y verificada contra un criterio de salida concreto. Esta secuencialidad es estricta porque las fases tienen dependencias reales: no podemos entrenar redes sobre datos que no hemos generado, no podemos generar datos sin *solvers* que funcionen y no podemos validar un experimento sin métricas implementadas. Intentar paralelizar fases provoca exactamente el tipo de problema que queremos evitar, que es descubrir a mitad del proyecto que algo de la fase anterior estaba mal y tener que volver atrás.

La **fase 0** cubre la infraestructura. Incluye la estructura del repositorio, semillas centralizadas, *solvers* de Black-Scholes y Heston, inversor de volatilidad implícita y la validación cruzada con QuantLib. Su criterio de salida es que Black-Scholes y Heston reproduzcan valores de referencia externos con error inferior a `10^-6` en precio y `10^-4` en Delta, y que el inversor de IV recupere la volatilidad de referencia en casos cerrados de Black-Scholes. Esta fase se cierra contra valores tabulados conocidos porque un fallo silencioso en el *solver* contaminaría todo el resto del proyecto sin que nos enteremos.

La **fase 1** monta el *pipeline* de datos y entrenamiento. Implementa el *sampler*, el generador, el *script* reproducible de generación, el modelo, las utilidades de *Greeks*, el *trainer* y el *script* de entrenamiento. Genera los *datasets* base de entrenamiento, validación y test balanceado por *bins* para ambas familias de modelo. Su criterio de salida es un *smoke test* sobre BS-3 que converja a un MAE de precio razonablemente pequeño en validación. Es la primera vez que el *pipeline* completo se ejecuta de principio a fin: si BS-3 no converge cuando le pasamos datos de una fórmula cerrada conocida, hay algo roto antes de tocar Heston y este es el momento de descubrirlo.

La **fase 2** ejecuta los experimentos sobre Black-Scholes. Entrena BS-1, BS-2, BS-3 y BS-4, implementa los módulos de *binning* y métricas, produce tablas y *heatmaps* por *bin* para cada *surrogate* y completa los análisis E1 (precio vs IV) y E2 (activaciones) sobre Black-Scholes. El motivo de hacer Black-Scholes antes que Heston es que Black-Scholes tiene fórmulas cerradas para precio, Delta, Vega e IV, lo que permite validar todas las piezas del *pipeline* contra valores exactos antes de extender el mismo análisis a Heston.

La **fase 3** ejecuta los experimentos sobre Heston, que es el bloque más grande del proyecto. Genera los *datasets* específicos de E3 y E5, entrena H-1, H-2, H-3, H-4, H-5, H-3-small y H-6-small, repite el análisis E2 sobre Heston y cierra E3 (muestreo) y E5 (DML) con su clasificación operativa fuerte/débil/negativa.

La **fase 4** cubre eficiencia y cierre. Implementa el protocolo de cronometraje con *warmup* y diez repeticiones por lote, ejecuta E4 sobre H-3, produce las figuras finales y redacta la memoria. La eficiencia se mide al final porque depende de tener el *surrogate* ya entrenado y validado; hacerla en una fase previa carecería de sentido porque no podríamos comparar con un *surrogate* que aún no existe.

---

## 9. Discusión y trabajo futuro

El alcance del proyecto está deliberadamente acotado. No intentamos calibrar a datos reales de mercado, ni estudiar opciones exóticas, ni construir un *surrogate* industrial a escala de Chen et al. El objetivo es demostrar, en un entorno controlado y reproducible, cuándo una red puede aproximar un *solver* de pricing con precisión, rapidez y derivadas útiles. La contribución está en la comparación ordenada de métricas, activaciones, muestreo, eficiencia y aprendizaje diferencial, no en maximizar complejidad del modelo a cualquier coste.

Quedan abiertas varias extensiones naturales que merecen mención. Extender el conjunto de *Greeks* incorporadas a la pérdida diferencial (Vega, Gamma, Theta, Rho) permitiría medir hasta qué punto el *differential machine learning* escala más allá del caso unidimensional que estudiamos en E5, aunque introduce dificultades numéricas adicionales porque algunas de esas derivadas son sensibles a la inestabilidad del *solver* de Heston en zonas extremas. Pasar de opciones europeas a opciones americanas o exóticas pondría a prueba la generalización del enfoque a *payoffs* con derecho de ejercicio anticipado y conectaría nuestro trabajo con la línea de Becker, Cheridito y Jentzen. Aplicar el *surrogate* dentro de un procedimiento de calibración a datos reales —en línea con el segundo nivel de validación de Chen et al.— mediría el coste económico del error de aproximación de la red. Por último, escalar la arquitectura o el tamaño del *dataset* permitiría explorar si la reducción de error es lineal o se satura, una pregunta especialmente relevante a la luz de los resultados que estamos observando en la fase actual del proyecto.

A día de hoy, las fases 0 y 1 están prácticamente cerradas: los *solvers* están validados contra QuantLib y el *pipeline* de generación y entrenamiento está operativo. Las fases 2 a 4 se ejecutarán en las semanas siguientes según el plan descrito en la sección 8. La estructura del trabajo está diseñada para que los resultados experimentales se puedan integrar de forma incremental en la versión final del paper sin reescribir las secciones metodológicas, que ya están cerradas.

---

## Referencias

Las referencias se citan informalmente en este borrador y se formalizarán en la versión final. Los PDFs de los cinco pilares están disponibles en `papers/` del repositorio del proyecto, y los trabajos complementarios en `papers/others/`.

1. Hutchinson, J. M., Lo, A. W., y Poggio, T. (1994). *A nonparametric approach to pricing and hedging derivative securities via learning networks*. *Journal of Finance*, 49(3), 851–889.
2. Becker, S., Cheridito, P., y Jentzen, A. (2019). *Deep optimal stopping*. *Journal of Machine Learning Research*, 20, 1–25.
3. Ruf, J., y Wang, W. (2020). *Neural networks for option pricing and hedging: a literature review*. *Journal of Computational Finance*, 24(1), 1–46.
4. Chen, H., Didisheim, A., y Scheidegger, S. (2025). *Deep surrogates for finance: with an application to option pricing*. *Manuscript / SSRN preprint*.
5. Huge, B., y Savine, A. (2020). *Differential machine learning*. *Risk Magazine* y SSRN.
6. Sirignano, J., y Spiliopoulos, K. (2018). *DGM: a deep learning algorithm for solving partial differential equations*. *Journal of Computational Physics*, 375, 1339–1364.
7. Han, J., Jentzen, A., y E, W. (2018). *Solving high-dimensional partial differential equations using deep learning*. *Proceedings of the National Academy of Sciences*, 115(34), 8505–8510.
8. Ferguson, R., y Green, A. (2018). *Deeply learning derivatives*. *arXiv:1809.02233*.
9. Buehler, H., Gonon, L., Teichmann, J., y Wood, B. (2019). *Deep hedging*. *Quantitative Finance*, 19(8), 1271–1291.
10. Barron, A. R. (1993). *Universal approximation bounds for superpositions of a sigmoidal function*. *IEEE Transactions on Information Theory*, 39(3), 930–945.
11. Grohs, P., Hornung, F., Jentzen, A., y von Wurstemberger, P. (2018). *A proof that artificial neural networks overcome the curse of dimensionality in the numerical approximation of Black-Scholes partial differential equations*. *arXiv:1809.02362*.
