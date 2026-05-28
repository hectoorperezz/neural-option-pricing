# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es este proyecto

Trabajo de investigación universitario (Métodos Numéricos) de Ángel Fernández Sánchez, Jorge Alfageme Sotillos y Héctor Pérez Ledesma. **No es un TFM.** Idioma: español para toda comunicación, comentarios y documentos; identificadores y rutas en inglés.

Pregunta central: *bajo qué condiciones una red neuronal profunda puede actuar como un surrogate preciso, rápido y diferenciable para funciones de pricing de opciones*. La red no aprende del mercado; aprende a imitar a un solver ya conocido y caro (Heston por Fourier semi-cerrado) para sustituirlo en evaluaciones masivas (calibración diaria, Greeks, simulación de escenarios). Black-Scholes funciona solo como caso de validación porque su fórmula cerrada permite errores exactos.

Referencias centrales (no opcionales para entender decisiones de diseño):
- **Chen, Didisheim y Scheidegger 2025** — deep surrogates. De aquí salen Swish como activación por defecto, evaluación por bins, recorte del 5% de bordes y la ausencia de regularización.
- **Huge y Savine 2020** — differential ML. Inspira E5: añadir Delta como target diferencial debería reducir la cantidad de datos necesarios.
- `papers/` contiene los cinco pilares (Hutchinson 1994, Becker-Cheridito-Jentzen 2019, Ruf-Wang 2020, Chen 2025, Huge-Savine 2020) y `papers/others/` complementarios.

**Lecturas obligatorias antes de cualquier cambio no trivial:**
- `docs/primera-entrega.md` — revisión bibliográfica y por qué el proyecto se enfoca en deep surrogates.
- `docs/metodologia.md` — targets, normalización, métricas, bins, criterios de validez de cada experimento.
- `docs/tasks.md` — los 5 experimentos (E1–E5), los 11 surrogates y las cuatro fases del proyecto con criterios de salida.
- `docs/architecture.md` — principios de diseño y jerarquías OO por módulo.
- `docs/experimento-escalado-datos.md` — registro diagnóstico del run 50x; no sustituye al protocolo formal de evaluación.
- `docs/contexto-generacion-datasets.md` — playbook operativo (entorno, benchmark, tmux, formato `.npz`/`.json`).

## Comandos

Entorno: **Python 3.11 obligatorio** (versiones fijadas para evitar colisiones NumPy/SciPy/PyTorch).
- CPU/macOS: `pip install -r requirements.txt`
- Linux + CUDA 12.1 (PC de oficina con RTX 4060): `pip install -r requirements-gpu.txt`

No hay `pyproject.toml`; los scripts insertan `REPO_ROOT` en `sys.path` para importar `src/`.

Tests:
```bash
pytest                                                    # toda la suite (~60 passed)
pytest tests/solvers/test_quantlib_reference.py           # validación cruzada con QuantLib (~26 passed)
pytest tests/solvers/                                     # solo solvers
pytest tests/test_seeding.py -k name_of_test              # un test concreto
```
**La validación QuantLib es load-bearing** (criterio de salida de Fase 0: error `<1e-6` en precio, `<1e-4` en Delta). Ejecutarla antes de regenerar datasets grandes.

Generación de datasets (un único script parametrizado):
```bash
python scripts/generate_dataset.py \
  --family {black_scholes|heston} \
  --sampler {uniform|focused|balanced} \
  --n-samples N | --samples-per-bin N         # balanced → 25*N puntos totales
  --batch-size 10000 --seed S --include-delta \
  --output data/foo.npz
```
- Baseline completo (200k BS / 500k Heston): `bash scripts/generate_all_datasets.sh` (`.bat` en Windows).

Entrenamiento (un único script parametrizado por flags, no por YAML aún):
```bash
python scripts/train_surrogate.py \
  --train data/foo_train.npz --validation data/foo_val.npz \
  --experiment-id BS-3 --output-dir results/checkpoints/BS-3 \
  --loss {price|differential} --activation {relu|softplus|swish|tanh} \
  --hidden-width 128 --hidden-layers 4 \
  --epochs 100 --batch-size 4096 --learning-rate 1e-3 --seed 103 \
  --device {auto|cpu|cuda} \
  --preload-to-device       # carga todo el dataset en VRAM, evita transferencias por batch
  --compile                 # torch.compile (Inductor); en Windows requiere Triton
```

Orquestadores que entrenan los 11 surrogates en paralelo sobre una GPU (cada surrogate es un `subprocess`; `ThreadPoolExecutor` controla concurrencia):
- `python scripts/train_all_parallel.py --parallel 4` — baseline.

Los identificadores de surrogate, datasets, semillas, batch size y pérdida están **hardcodeados** en la lista `SURROGATES` dentro de los orquestadores. `configs/` está vacío salvo `.gitkeep` aunque `docs/architecture.md` planifica un YAML por surrogate — esa migración aún no se ha hecho.

## Surrogates y experimentos

11 surrogates fijos. **Todos los comparados dentro de un mismo experimento comparten arquitectura, optimizador, hiperparámetros y semilla** — solo varía la dimensión que ese experimento estudia:

| ID | Modelo | Activación | Pérdida | Dataset | Aporta a |
|---|---|---|---|---|---|
| BS-1..4 | Black-Scholes | ReLU/Softplus/Swish/tanh | L1(precio) | 200k uniforme | E2 (BS-3 también E1) |
| H-1..4 | Heston | ReLU/Softplus/Swish/tanh | L1(precio) | 500k uniforme | E2 (H-3 también E1, E3, E4, E5 baseline) |
| H-5 | Heston | Swish | L1(precio) | 500k **focused** | E3 |
| H-3-small | Heston | Swish | L1(precio) | 100k uniforme | E5 |
| H-6-small | Heston | Swish | L1(precio) + L1(Δ), pesos 1:1 | 100k uniforme | E5 |

H-3 es el surrogate central (aparece en los cinco experimentos). Semillas: 101-104 para BS, 201-207 para Heston.

Cada experimento varía **una sola dimensión**:

| ID | Pregunta | Métrica primaria | Notas |
|---|---|---|---|
| E1 | ¿Error bajo en precio ⇒ error bajo en IV? | Discrepancia por bin entre `MAE(C/K)` y `MAE_IV` | **Observacional**, no entrena nada nuevo; recalcula sobre BS-3 y H-3 |
| E2 | ¿Qué activación da mejores Greeks? | `MAE_Delta` por bin | Sin umbral fuerte/débil; precio como control |
| E3 | ¿Sampler enfocado mejora bins difíciles? | `MAE_IV` en ATM × {weekly, short, medium-short} | Fuerte si mejora ≥10% y global no empeora >10% |
| E4 | ¿Speedup del surrogate vs solver? | `tiempo_solver / tiempo_surrogate` por tamaño de lote | Lotes `10²..10⁵`, 3 warmups + 10 repes, mediana, CPU/GPU separado |
| E5 | ¿DML mejora eficiencia muestral? | Mejora de `MAE_Delta` de H-6-small vs H-3-small | Fuerte si Delta mejora ≥20% y precio no empeora >10% |

## Convenciones invariantes del pipeline

Las decisiones de `docs/metodologia.md` y `docs/tasks.md` están cerradas y no se mueven salvo evidencia experimental contraria:

- **Target principal**: `y = C/K` (precio normalizado por strike). **Input principal**: `m = S/K` (moneyness simple). Esto da `Delta = dC/dS = dy/dm` directo, y elimina una escala redundante (si spot y strike se multiplican por la misma constante, el precio escala con el strike).
- **No usamos** la moneyness normalizada `m/(F·sqrt(T))` de Chen et al. — nuestro objetivo no es replicar SPX sino mantener `Delta = dy/dm` directa.
- **`q = 0` en todos los experimentos** (dividend yield fuera del problema). Por eso en Heston `Delta = exp(-qT)·P1 = P1`, lo que convierte la primera probabilidad de Fourier en el target diferencial limpio de E5.
- **Inputs normalizados min-max a `[0,1]`** antes de entrar a la red. La Delta del surrogate se obtiene por autograd derivando respecto a `m_norm` y reescalando: `Delta_hat = (∂ŷ/∂m_norm) / (m_max - m_min)`. En E2 (evaluación) `create_graph=False`; en E5 (entrenamiento con DML) `create_graph=True` porque la pérdida contiene `MAE(Delta)` y PyTorch debe retropropagar.
- **Métrica de selección de checkpoint: `MAE(C/K)` en validación, siempre**. También para H-6-small aunque su loss incluya Delta — E5 debe cambiar solo la información usada durante el entrenamiento, no el criterio de selección. El test final sí reporta precio, IV y Delta por separado.
- **Recorte 5%** en cada cara del hipercubo al evaluar (las redes aproximan peor en frontera). Las métricas reportadas son sobre el dominio interior.
- **Sin regularización, sin early stopping, sin dropout, sin weight decay**. Los datos son sintéticos sin ruido; el solver es determinista. La pérdida de validación se estanca pero no empeora por overfitting.

Dominios fijos (más exigentes que un dominio académico ATM-céntrico; deliberadamente incluyen weeklies, wings profundas y volatilidades de estrés):
- Contractual: `m ∈ [0.4, 2.0]`, `T ∈ [7/365, 2.0]`, `r ∈ [0.00, 0.075]`, `sigma_BS ∈ [0.03, 1.00]`.
- Heston: `v0, theta ∈ [0.0009, 1.00]` (muestreados como `sqrt` en `[0.03, 1.00]`), `kappa ∈ [0.10, 10]`, `xi ∈ [0.10, 3.0]`, `rho ∈ [-0.95, -0.05]` (skew negativo típico de equity). **Feller no se impone**; se reporta como diagnóstico.
- Sampler enfocado H-5 (E3): mezcla 50/50. 50% baseline uniforme + 50% `m ~ TruncNormal(1.0, 0.15, [0.7, 1.3])`, `T ~ LogUniform(7/365, 0.25)`, resto baseline.
- Particiones: train (sampler del surrogate), validación uniforme 50k, test balanced 5k/bin → 125k. Test balanceado **compartido** por familia: H-3 y H-5 se evalúan sobre el mismo test, aunque H-5 haya entrenado focused — eso es lo que permite ver dónde gana y dónde pierde el sampler enfocado.
- Bins 5×5: moneyness {Deep OTM `[0.4,0.7)`, OTM `[0.7,0.9)`, ATM `[0.9,1.1]`, ITM `(1.1,1.3]`, Deep ITM `(1.3,2.0]`} × vencimiento {Weekly `[7/365,14/365)`, Short `[14/365,1/12)`, Medium-short `[1/12,0.25)`, Medium `[0.25,1.0)`, Long `[1.0,2.0]`}.
- Arquitectura por defecto: MLP 4 capas × 128 unidades (~50k parámetros), Swish, Adam `lr=1e-3` fijo, batch size definido por script, 100 épocas máximo.

## Arquitectura

Separación estricta en cuatro capas (ver `docs/architecture.md` §Principios):
- `src/` — librería importable. Toda la lógica reutilizable, testeable.
- `configs/` — YAML por surrogate (planificado, aún no poblado).
- `scripts/` — solo orquestación: parsear args, llamar `src/`, escribir a `data/` o `results/`. Si aparece lógica reutilizable en un script, se promociona a `src/`.
- `data/`, `results/` — outputs (gitignored salvo `.gitkeep` y CSVs de métricas).

Polimorfismo donde aporta, funciones libres donde no:
- `src/solvers/` — `OptionPricer` (ABC) con `BlackScholesSolver` (precio, Delta `N(d1)`, Vega cerradas) y `HestonSolver` (Fourier semi-cerrado con `P1` y `P2` por `scipy.integrate.quad`; `Delta = exp(-qT)·P1`). `ImpliedVolatilityInverter` aparte porque su lógica es ortogonal al pricing.
- `src/datasets/` — `Sampler` ABC con `UniformSampler`, `FocusedSampler`, `BalancedBinSampler`. `Domain` (dataclass) encapsula rangos, transformaciones `sqrt(v0)/sqrt(theta)`, recortes y Feller como diagnóstico. `DatasetGenerator` compone solver + sampler, filtra fallos numéricos, materializa `OptionDataset` (`torch.utils.data.Dataset`). **La diferencia entre H-3 y H-5 (E3) es solo cambiar el sampler inyectado.**
- `src/models/` — una sola clase `MLP` con activación resuelta vía dict `ACTIVATIONS`. No hay jerarquía de modelos: solo varían hiperparámetros, introducir clases base sería abstracción prematura. `src/models/greeks.py` calcula Delta del surrogate por autograd con la regla de la cadena.
- `src/training/` — `SurrogateLoss` ABC con `PriceLoss` y `DifferentialLoss` (combina `MAE(C/K)` + `MAE(Delta)`). `TrainConfig` dataclass + `Trainer` con `fit/save/load` y `from_config`. Selecciona checkpoint con menor `MAE(C/K)` de validación.
- `src/evaluation/` y `src/experiments/` — evaluación por bins, reportes CSV/heatmaps y experimentos E1-E3 implementados como clases (`PriceVsIVStudy`, `ActivationStudy`, `SamplingStudy`). E4/E5 quedan pendientes.
- `src/utils/seeding.py` — `set_global_seed` cubre `random`, `numpy`, `torch`. Cada script debe llamarla al inicio.

## Estado del proyecto

Fases con criterios de salida concretos (ver `docs/tasks.md` §Tareas por fase):
- **Fase 0 — Infraestructura**: ☑ cerrada. Solvers BS y Heston validados contra QuantLib (`tests/solvers/test_quantlib_reference.py`), Delta Heston verificada con diferencias finitas.
- **Fase 1 — Pipeline**: prácticamente cerrada. Quedan datasets en `data/` (gitignored) y el smoke test BS-3 hasta `MAE_precio < 1e-4`.
- **Fases 2-3**: E1, E2 y E3 ya tienen scripts, CSVs y heatmaps archivados en `results/`. E4 y E5 quedan pendientes.

## Cuando edites el código

- **Antes de añadir lógica a un script, comprueba que no debería vivir en `src/`.** Regla de oro de `architecture.md`: scripts solo orquestan.
- **Cada experimento cambia una sola variable.** Si alguien te pide "comparar X variando Y", asegúrate de que el resto (arquitectura, sampler, dataset, semilla, pérdida) permanece idéntico. Esta disciplina es lo que permite atribuir diferencias a Y.
- Cualquier cambio en `src/solvers/` o `src/utils/seeding.py` toca el cimiento del proyecto: la suite contra QuantLib y la reproducibilidad son load-bearing. Un fallo silencioso ahí contamina todos los datasets generados después.
- Compatibilidad Windows: `torch.compile` requiere Triton, que no tiene wheel oficial en Windows. `scripts/train_surrogate.py` ya hace probe con `importlib` y degrada limpiamente — no introduzcas dependencias del backend Inductor sin proteger el path Windows. Los orquestadores tienen variantes `.bat` paralelas a los `.sh`.
- **No commitees `data/`** (gitignored) ni cambies rangos del dominio, semillas o nombres de archivo sin documentarlo: rompería la comparabilidad de cualquier resultado nuevo con los baselines existentes.
- Para entrenamientos largos en el PC de oficina, usar `tmux` (ver `docs/contexto-generacion-datasets.md` §Ejecución Larga).
