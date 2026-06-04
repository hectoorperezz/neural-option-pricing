# Arquitectura del código

**Autores:** Ángel Fernández Sánchez, Jorge Alfageme Sotillos, Héctor Pérez Ledesma

## Visión

---

Este documento describe cómo está organizado el código que sostiene los experimentos E1-E6. Es complementario a `metodologia.md`: aquel fija qué se mide y bajo qué reglas; este explica dónde vive cada pieza, qué responsabilidades tiene y cómo se conectan los scripts con la librería interna.

## Principios de diseño

---

Tres principios guían la arquitectura actual. El primero es separar código reutilizable, scripts de entrada y artefactos generados. La lógica vive en `src/` y se importa como librería. Los scripts de `scripts/` solo parsean argumentos, llaman a `src/` y escriben resultados. Los datasets, checkpoints y logs son artefactos generados; las métricas CSV y figuras finales quedan en `results/`.

El segundo principio es usar orientación a objetos solo donde aporta polimorfismo real. Solvers, samplers, pérdidas y experimentos tienen interfaces comunes porque se intercambian entre configuraciones. Donde el código es puramente algorítmico, usamos funciones libres.

El tercer principio es reproducibilidad por comando y metadatos. Los datasets `.npz` se acompañan de un `.json` con dominio, semillas y contadores. Cada entrenamiento escribe `config.json`, `history.csv`, `history.json` y `checkpoint.pt`. No dependemos todavía de YAMLs versionados por surrogate; la orquestación completa está en `scripts/train/train_all_parallel.py` y en los `config.json` que produce `scripts/train/train_surrogate.py`.

## Estructura del repositorio

---

```text
proyecto-final-metodos-numericos/
├── docs/                          Documentación técnica, metodología y resultados
├── papers/                        Bibliografía consultada
├── src/                           Librería importable del proyecto
│   ├── solvers/                   Black-Scholes, Heston e inversión IV
│   ├── datasets/                  Dominios, samplers y datasets sintéticos
│   ├── models/                    MLP y cálculo de Greeks del surrogate
│   ├── training/                  Pérdidas y bucle de entrenamiento
│   ├── evaluation/                Bins, métricas, reportes y timing
│   ├── experiments/               Lógica de E1-E6
│   └── utils/                     Semillas, carga de artefactos y helpers comunes
├── scripts/                       Puntos de entrada ejecutables
│   ├── data/                      Generación de datasets (.py, .sh, .bat)
│   ├── train/                     Entrenamiento de surrogates
│   ├── experiments/               Runners de E1-E6 y evaluación
│   └── figures/                   Figuras del paper y estilo común
├── data/                          Datasets generados, no versionados
├── results/                       Métricas, figuras, checkpoints y logs
├── tests/                         Suite pytest
├── requirements.txt
├── requirements-gpu.txt
└── README.md
```

La regla práctica es simple: si una función se reutiliza o merece tests unitarios, vive en `src/`; si solo coordina una ejecución concreta, vive en `scripts/`.

## Módulos principales

---

### Solvers

`src/solvers/` contiene la interfaz `OptionPricer` y las implementaciones de referencia. `BlackScholesSolver` da precio, Delta y Vega cerradas. `HestonSolver` usa una formulación semi-cerrada de Fourier con las probabilidades `P1` y `P2`; como `q = 0`, la Delta de Heston usada en E5 es `P1`. `ImpliedVolatilityInverter` vive separado porque la inversión IV se usa como métrica de evaluación, no como parte del solver principal.

### Datasets

`src/datasets/` define el hipercubo de entrenamiento mediante `Domain` y los samplers `UniformSampler`, `FocusedSampler` y `BalancedBinSampler`. El sampler uniforme cubre el dominio completo. El enfocado mezcla cobertura global con concentración ATM y vencimientos cortos para E3. El balanceado genera tests con 5k puntos por bin para que la evaluación no dependa de la densidad de entrenamiento.

`DatasetGenerator` compone un solver y un sampler, filtra fallos numéricos claros y produce arrays normalizados. El formato persistente es `.npz`: `features`, `raw_inputs`, `prices`, `input_names` y, cuando aplica, `deltas` y `bin_id`.

### Models

`src/models/mlp.py` implementa una única MLP parametrizable por anchura, profundidad y activación (`relu`, `softplus`, `swish`, `tanh`). No hay jerarquía de modelos porque el proyecto compara variantes de la misma arquitectura. `src/models/greeks.py` calcula la Delta del surrogate con autograd y aplica la regla de la cadena para pasar de `m_norm` a `m`.

### Training

`src/training/losses.py` define `PriceLoss` y `DifferentialLoss`. La primera optimiza `MAE(C/K)`. La segunda suma `MAE(C/K) + MAE(Delta)` con pesos 1:1 para E5. `Trainer` ejecuta el bucle de entrenamiento, evalúa `MAE(C/K)` en validación al final de cada época y conserva el estado con mejor validación.

El punto de entrada real es `scripts/train/train_surrogate.py`. Recibe rutas de train/validation, activación, pérdida, batch size, learning rate, semilla y dispositivo. Para lanzar los once surrogates pre-registrados de forma reproducible existe `scripts/train/train_all_parallel.py`, que contiene la tabla `SURROGATES` con identificadores, datasets, seeds y pérdidas. Las tres variantes de E6 (H-7-shallow, H-8-deep, H-9-lr-schedule) se entrenan aparte con `scripts/train/train_e6.bat`.

### Evaluation

`src/evaluation/` encapsula el protocolo de medida. `BinPartition` define la rejilla `5 x 5` de moneyness y vencimiento. `BinEvaluator` calcula errores de precio, IV y Delta por bin. `Report` serializa tablas CSV. `TimingBenchmark` implementa E4 con warmups, repeticiones medidas, medianas, percentiles y speedup frente al solver.

La inversión IV se ejecuta solo cuando el experimento la necesita. E1 y E3 la usan como métrica principal o diagnóstico; E5 y E6 no la invocan porque sus preguntas se centran en Delta y precio.

### Experiments

`src/experiments/` contiene una clase por experimento: `PriceVsIVStudy`, `ActivationStudy`, `SamplingStudy`, `EfficiencyStudy`, `DMLStudy` y `ArchitectureStudy` (E6). Cada clase recibe surrogates ya entrenados, evaluadores y datasets, ejecuta la comparación y devuelve una tabla interpretable. Los scripts `run_experiment_e1.py` a `run_experiment_e6.py` son envoltorios finos sobre esas clases.

## Flujo de ejecución

---

La generación de datos se hace con un único script:

```bash
python scripts/data/generate_dataset.py \
  --family heston \
  --sampler balanced \
  --samples-per-bin 5000 \
  --include-delta \
  --output data/heston_test_125k_balanced_delta.npz
```

El workflow baseline está en `scripts/data/generate_all_datasets.sh` y su equivalente `.bat`. Crea train, validation y test para Black-Scholes y Heston con las semillas fijadas en la metodología.

El entrenamiento individual se lanza así:

```bash
python scripts/train/train_surrogate.py \
  --train data/heston_train_25M_uniform.npz \
  --validation data/heston_validation_2500k_uniform.npz \
  --experiment-id H-3 \
  --output-dir results/checkpoints/H-3 \
  --loss price \
  --activation swish \
  --epochs 100 \
  --batch-size 32768 \
  --learning-rate 1e-3 \
  --seed 203
```

Los experimentos finales se ejecutan con los scripts `scripts/experiments/run_experiment_e*.py`, que escriben CSV en `results/metrics/` y figuras en `results/figures/`.

## Estrategia de testing

---

El proyecto usa `pytest` con tres capas. La primera valida solvers contra fórmulas cerradas y QuantLib; es la capa crítica porque un solver mal implementado contaminaría todos los datasets. La segunda cubre samplers, binning, métricas, inversión IV, pérdidas y cálculo de Greeks. La tercera ejecuta scripts y recorridos de integración pequeños para comprobar que los módulos encajan.

La suite completa contiene 283 tests y pasa en entorno normal con:

```bash
python -m pytest -q
```

Los tests que usan `ProcessPoolExecutor` pueden fallar dentro de sandboxes restrictivos por permisos de memoria compartida; fuera del sandbox pasan junto con el resto de la suite.
