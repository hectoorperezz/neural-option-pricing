# Arquitectura del código

**Autores:** Ángel Fernández Sánchez, Jorge Alfageme Sotillos, Héctor Pérez Ledesma

## Visión

---

Este documento recoge las decisiones de diseño que estructuran el código del proyecto. La idea es que cualquiera que abra el repositorio dentro de tres meses pueda entender por qué los ficheros están donde están, por qué las clases tienen los nombres que tienen y por qué algunas decisiones aparentemente menores (como tener un único entrenador parametrizado en lugar de un script por surrogate) son las que evitan que el proyecto se vuelva inmantenible a mitad de camino. Es complementario a `tasks.md`: aquel define qué entregamos y este define cómo está organizado el código que produce esos entregables.

## Principios de diseño

---

Tres principios guían toda la arquitectura. El primero es la separación estricta entre código reutilizable, configuración por experimento, scripts de entrada y salidas. El código vive en `src/` y se importa como librería; la configuración vive en `configs/` como archivos YAML versionados; los scripts de `scripts/` solo orquestan llamadas sin contener lógica reutilizable; los datos y resultados quedan en `data/` y `results/` respectivamente. Esta separación es lo que evita que un cambio en una parte del pipeline rompa partes que no estábamos tocando.

El segundo principio es el uso de orientación a objetos donde aporta polimorfismo real. Cuando un mismo concepto tiene varias implementaciones que se intercambian entre experimentos (solvers, samplers, pérdidas, experimentos), creamos una clase base abstracta que define la interfaz y clases concretas que la implementan. El resto del pipeline depende de la interfaz, no de la implementación. Eso es lo que permite que `BinEvaluator` evalúe surrogates de Black-Scholes y de Heston sin enterarse de qué modelo hay por debajo, simplemente recibe un `OptionPricer` por inyección. Donde solo hay una implementación posible o el código es puramente algorítmico, usamos funciones libres y no creamos clases por crear.

El tercer principio es la reproducibilidad por configuración. Cada uno de los once surrogates está completamente descrito por un fichero YAML pequeño en `configs/`. Reentrenar un surrogate dentro de seis meses es cargar el YAML y volver a llamar al entrenador. Las semillas están centralizadas, las dependencias congeladas en `requirements.txt` y los hiperparámetros viven en código bajo control de versiones. Si un resultado no se puede reproducir desde un commit, hay un bug en la arquitectura.

## Estructura del repositorio

---

```
proyecto-final-metodos-numericos/
├── docs/                          Documentación del proyecto
├── papers/                        Bibliografía consultada
├── src/                           Librería del proyecto, importable como paquete
│   ├── solvers/                   Pricers analíticos y numéricos
│   ├── datasets/                  Muestreo y generación de datasets sintéticos
│   ├── models/                    Arquitectura de la red y Greeks del surrogate
│   ├── training/                  Bucle de entrenamiento, pérdidas, configuración
│   ├── evaluation/                Binning, métricas, eficiencia
│   ├── experiments/               Lógica de cada uno de los cinco experimentos
│   └── utils/                     Semillas, entrada/salida
├── configs/                       Una config YAML por surrogate
├── scripts/                       Puntos de entrada ejecutables
├── notebooks/                     Exploración y figuras finales
├── data/                          Datasets generados, gitignored
├── results/                       Checkpoints, métricas y figuras versionadas
├── tests/                         pytest
├── .gitignore
├── requirements.txt
├── pyproject.toml                 Instalación del paquete src/
└── README.md
```

La distinción entre `src/` y `scripts/` es deliberada. Todo lo que es código reutilizable, testeable y que importará el resto del proyecto vive bajo `src/`. Los scripts son los puntos donde el usuario interactúa con el sistema y solo contienen orquestación: leer una config, llamar a las clases de `src/`, escribir resultados. Si dentro de un script aparece lógica que merezca ser reutilizada por otro script, se promociona a `src/`.

## Diseño orientado a objetos por módulo

---

### Solvers

El módulo `src/solvers/` define la jerarquía que sostiene todo el proyecto. La clase base abstracta `OptionPricer` declara la interfaz mínima que cualquier pricer debe cumplir: devolver precio, devolver Delta cuando esté disponible y reportar su dimensión de entrada. Las dos implementaciones concretas son `BlackScholesSolver`, que tiene fórmulas cerradas para precio, Delta y Vega, y `HestonSolver`, que implementa una formulación semi-cerrada de Fourier con las probabilidades `P1` y `P2` mediante integración numérica con `scipy.integrate.quad`. En Heston, la Delta principal se obtiene como `exp(-qT) * P1`; como fijamos `q = 0`, el target diferencial de E5 será directamente `P1`, validado contra diferencias finitas centrales antes de generar datasets con Delta. El inversor de volatilidad implícita vive como clase aparte, `ImpliedVolatilityInverter`, porque su lógica es ortogonal al pricing y se invoca tanto durante la evaluación como dentro de algunos análisis de los experimentos. El polimorfismo del solver es lo que permite que el generador de datos, el evaluador y el benchmark de eficiencia reutilicen el mismo código entre Black-Scholes y Heston.

### Datasets

El muestreo del hipercubo se modela como una jerarquía `Sampler` con dos implementaciones, `UniformSampler` y `FocusedSampler`. La diferencia entre ellas es la distribución de probabilidad sobre el hipercubo: el primero uniforme en la parametrización definida por el `Domain`, el segundo mezcla un componente uniforme global con un componente enfocado cerca del at-the-money y de vencimientos cortos. Esta separación es lo que hace que el experimento E3 (muestreo) se reduzca a cambiar una sola línea en la configuración del surrogate H-5 frente a H-3. El `Domain` es un dataclass que encapsula los rangos del hipercubo, transformaciones como muestrear `sqrt(v0)` y `sqrt(theta)` antes de convertir a varianza, el recorte de bordes y diagnósticos como la condición de Feller en Heston, y se inyecta tanto en el sampler como en el evaluador. La config de `FocusedSampler` fija el peso de mezcla, la normal truncada de `m` y la log-uniforme de `T`; el resto de variables se delega al muestreo baseline del `Domain`. La clase `DatasetGenerator` compone un `OptionPricer` y un `Sampler` para producir `(x, precio, Delta opcional)`, filtra fallos numéricos claros del solver y materializa el resultado como un `OptionDataset` heredero de `torch.utils.data.Dataset`. El generador debe poder crear tres particiones reproducibles: train con el sampler del surrogate, validation uniforme de 50k puntos y test balanceado por bins de 125k puntos. Para E5, Delta es el único target diferencial principal; Vega, Gamma, Theta y Rho se pueden calcular después como métricas diagnósticas si el cálculo numérico es estable. Esta composición por inyección de dependencias es lo que hace que añadir un nuevo solver o un nuevo sampler no requiera tocar el generador.

### Models

La arquitectura de la red vive en una sola clase, `MLP`, que hereda de `nn.Module`. Su constructor recibe la dimensión de entrada, el ancho de capa oculta, el número de capas y la activación como string. La activación se resuelve internamente a través de un diccionario `ACTIVATIONS` que mapea nombres a clases de PyTorch. Esto es lo que permite que el experimento E2 (activaciones) cambie el comportamiento del modelo con una única entrada en la configuración, sin tocar el código de la red. No hay jerarquía de modelos porque no la necesitamos: usamos un único tipo de red y solo varían sus hiperparámetros. Si en algún momento añadiésemos redes residuales o convolucionales, esa sería la situación que justifica introducir una clase base.

### Training

El entrenamiento se divide en tres piezas. La primera es la jerarquía de pérdidas, con `SurrogateLoss` como base abstracta y dos implementaciones concretas, `PriceLoss` para el régimen base que solo penaliza error de precio, y `DifferentialLoss` que combina `MAE(C/K)` y `MAE(Delta)` con pesos 1:1. La Delta predicha por el surrogate se calcula en una utilidad única, `src/models/greeks.py`, derivando el output de la red respecto a `m_norm` y dividiendo por `(m_max - m_min)`. En evaluación se llama con `create_graph=False`; en `DifferentialLoss` se llama con `create_graph=True` para que el término de Delta pueda retropropagarse hasta los pesos. Esta separación es la que hace que el experimento E5 (DML) se reduzca a cambiar la pérdida de la configuración del surrogate H-6-small respecto a H-3-small, manteniendo arquitectura, optimizador y semilla constantes.

La segunda pieza es `TrainConfig`, un dataclass que serializa toda la información necesaria para reproducir un entrenamiento: tipo de modelo, activación, clase de pérdida y sus parámetros, ruta del dataset, número de épocas, batch size, learning rate y semilla. Esta config es la unidad de reproducibilidad del proyecto.

La tercera pieza es la clase `Trainer`, que orquesta el bucle de entrenamiento. Recibe un modelo, una pérdida, un optimizador, un scheduler y la config, y expone los métodos `fit`, `save` y `load`. Tiene también un constructor de clase `from_config` que toma una `TrainConfig` y ensambla todas las dependencias automáticamente. En cada época calcula `MAE(C/K)` sobre validación y conserva el checkpoint con menor valor de esa métrica, incluso cuando la pérdida de entrenamiento incluya Delta. El `Trainer` no sabe nada sobre el problema concreto que está resolviendo; es el código que ejecutaría el mismo bucle si en lugar de surrogates financieros estuviésemos entrenando otra cosa.

### Evaluation

El módulo de evaluación encapsula todo el protocolo de medida del proyecto. La clase `BinPartition` define la rejilla `5 × 5` de moneyness por vencimiento y sabe asignar cada punto del test set a su bin correspondiente. La clase `BalancedTestSampler` genera exactamente 5k puntos por bin para construir el test común de cada familia de modelo. La clase `BinEvaluator` recibe una partición y un `OptionPricer` y expone un método `evaluate` que toma un surrogate y un test set y devuelve un `Report` con MAE de precio, MAE de IV y MAE de Delta, además de percentiles cincuenta, noventa y cinco y noventa y nueve por bin. Vega, Gamma, Theta y Rho pueden añadirse como columnas diagnósticas si se calculan de forma estable. El `Report` es un dataclass que sabe serializarse a CSV y a heatmap, y cada clase de experimento añade una capa de interpretación que marca la métrica primaria y los controles relevantes para su pregunta. La clase `TimingBenchmark` implementa el protocolo de eficiencia del experimento E4: recibe un surrogate y un solver, ejecuta tres warmups y diez repeticiones medidas por cada lote `10^2`, `10^3`, `10^4` y `10^5`, y devuelve mediana, p25/p75 y speedup por lote, separando CPU y GPU cuando aplique.

### Experiments

Cada uno de los cinco experimentos del proyecto se materializa como una clase concreta que hereda de `Experiment`. La clase base define el método `run`, que es lo único que los scripts de análisis llaman. Las cinco clases concretas (`PriceVsIVStudy`, `ActivationStudy`, `SamplingStudy`, `EfficiencyStudy`, `DMLStudy`) reciben los surrogates ya entrenados como dependencias y producen un reporte específico. La uniformidad de la interfaz es lo que hace que un script de análisis se reduzca a tres líneas: instanciar el experimento, llamar `run`, volcar el reporte. La lógica concreta de qué se compara y cómo se reporta vive dentro de cada clase, pero la forma de invocarlos desde fuera es siempre la misma.

## Composición de un experimento

---

Para ilustrar cómo se compone todo, un script de análisis del experimento E2 sobre Heston queda así:

```python
from src.solvers import HestonSolver
from src.evaluation import BinPartition, BinEvaluator
from src.experiments import ActivationStudy
from src.utils.io import load_checkpoint

solver = HestonSolver()
evaluator = BinEvaluator(BinPartition.default(), solver)
surrogates = [load_checkpoint(f"results/checkpoints/h_{i}.pt") for i in [1, 2, 3, 4]]

study = ActivationStudy(surrogates, evaluator)
report = study.run()
report.to_csv("results/metrics/e2_heston.csv")
report.to_heatmap("results/figures/e2_heston.png")
```

El script no contiene lógica reutilizable. Toda la inteligencia está en las clases que importa. Sustituir Heston por Black-Scholes en este experimento es cambiar dos líneas: la primera es la clase del solver y la segunda es la lista de surrogates que carga. El resto del código es idéntico porque depende solo de las interfaces abstractas.

## Configuración y reproducibilidad

---

Cada surrogate se describe por un fichero YAML en `configs/`. La estructura del archivo es la serialización directa de un `TrainConfig`:

```yaml
# configs/h_6_small.yaml
name: h_6_small
model:
  cls: MLP
  hidden_dim: 128
  n_layers: 4
  activation: swish
loss:
  cls: DifferentialLoss
  alpha_price: 1.0
  beta_delta: 1.0
dataset: data/heston_100k_uniform_delta.npz
solver: HestonSolver
n_epochs: 100
batch_size: 1024
lr: 1.0e-3
seed: 42
```

Entrenar el surrogate H-6-small es invocar `python scripts/train.py --config configs/h_6_small.yaml`. La config queda comiteada y el checkpoint resultante hereda el nombre. Esto es lo que permite que el documento `tasks.md` y los ficheros del repositorio estén siempre en correspondencia: un surrogate documentado allí es un YAML aquí, y un YAML aquí es un checkpoint reproducible. Las semillas son centralizadas a través de `src/utils/seeds.py`, que configura las semillas de Python, NumPy y PyTorch en una sola llamada al inicio de cada script.

## Estrategia de testing

---

El proyecto usa `pytest` con tres capas de tests. La primera capa son tests unitarios de los solvers contra valores tabulados de referencia, que son los que dan la garantía de que la fase 0 está cerrada correctamente. Estos tests son lo más crítico del proyecto porque un fallo silencioso en un solver contaminaría todo el resto del trabajo sin que nos enterásemos.

La segunda capa son tests unitarios de las piezas auxiliares: el sampler debe respetar los límites del hipercubo, el inversor de IV debe converger en casos conocidos, la Delta numérica debe validarse contra diferencias finitas en una grid pequeña, y la partición por bins debe asignar correctamente los puntos a sus bins. Estos tests son baratos de escribir, rápidos de ejecutar y atrapan la mayoría de los bugs que aparecen al refactorizar.

La tercera capa son tests de integración del pipeline completo sobre un mini-dataset sintético de Black-Scholes con cien puntos. El objetivo aquí no es validar la calidad del entrenamiento sino verificar que las piezas encajan: que el sampler produce datos que el generador consume, que el generador produce datasets que el entrenador acepta, que el entrenador produce checkpoints que el evaluador carga. Es un smoke test que se ejecuta en segundos y nos avisa si alguna refactorización rompe la integración entre capas.

La inyección de dependencias que practicamos en todo el código es lo que hace estos tests posibles. Cada clase recibe sus dependencias por constructor, lo que permite testearlas con mocks o con implementaciones triviales. El `BinEvaluator` se testea con un solver fake que devuelve valores conocidos; el `Trainer` se testea con una pérdida fake que solo cuenta llamadas. Esa testabilidad es lo que justifica el sobrecoste arquitectónico de tener clases en lugar de funciones.
