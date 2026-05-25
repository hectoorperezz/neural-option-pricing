# Contexto para Generación de Datasets

## Objetivo

Este documento es para un agente o desarrollador que ejecute la generación de datasets en el PC de oficina. El proyecto construye surrogates neuronales para pricing de opciones europeas bajo Black-Scholes y Heston. Los datasets se generan con solvers propios ya validados contra QuantLib.

## Estado del Proyecto

La generación debe hacerse después de traer la última versión del repositorio. Antes de generar datos grandes, comprobar que existen:

- `scripts/generate_dataset.py`
- `src/solvers/heston.py`
- `src/datasets/sampler.py`
- `tests/solvers/test_quantlib_reference.py`

La suite local validada actualmente pasa con:

```bash
python -m pytest
```

Resultado esperado aproximado: `58 passed`.

La validación externa de solvers debe pasar antes de lanzar datasets grandes:

```bash
python -m pytest tests/solvers/test_quantlib_reference.py
```

Resultado esperado aproximado: `26 passed`.

## Entorno Recomendado

Usar Python 3.11. Las versiones están fijadas en `requirements.txt` para evitar colisiones NumPy/SciPy/PyTorch.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests/solvers/test_quantlib_reference.py
```

Si el equipo solo tiene Python 3.12, usar `pyenv`, `conda` o Docker con Python 3.11. No actualizar versiones sin repetir los tests contra QuantLib.

## Capacidades del Script

El script `scripts/generate_dataset.py` cubre los tres tipos de muestreo necesarios:

- `--sampler uniform`: entrenamiento y validación uniforme.
- `--sampler focused`: entrenamiento Heston enfocado para E3.
- `--sampler balanced`: test final balanceado por bins de moneyness y vencimiento.

El modo balanceado usa 25 bins (`5 x 5`) y exige `--samples-per-bin`. Por ejemplo, `--samples-per-bin 5000` genera `125000` muestras.

## Consideraciones de Hardware

La generación Heston actual es principalmente CPU-bound, porque usa integración Fourier con `scipy.integrate.quad`. La GPU no es crítica para generar datos, aunque sí será útil para entrenar redes después. En un i9 con 64 GB de RAM, usar batches de `5_000` o `10_000` es razonable, pero primero hay que hacer benchmark.

## Benchmark Obligatorio

Ejecutar primero un dataset pequeño para estimar throughput y tasa de rechazo:

```bash
python scripts/generate_dataset.py \
  --family heston \
  --sampler uniform \
  --n-samples 5000 \
  --batch-size 1000 \
  --seed 42 \
  --include-delta \
  --output data/heston_benchmark_5k_delta.npz
```

Revisar el `.json` generado:

```bash
cat data/heston_benchmark_5k_delta.npz.json
```

Campos importantes: `throughput_samples_per_second`, `rejection_rate`, `elapsed_seconds`.

## Datasets de Entrenamiento

Black-Scholes train uniforme para BS-1..4:

```bash
python scripts/generate_dataset.py \
  --family black_scholes \
  --sampler uniform \
  --n-samples 200000 \
  --batch-size 20000 \
  --seed 45 \
  --include-delta \
  --output data/bs_train_200k_uniform_delta.npz
```

Heston train uniforme para H-1..4 y H-3 baseline:

```bash
python scripts/generate_dataset.py \
  --family heston \
  --sampler uniform \
  --n-samples 500000 \
  --batch-size 5000 \
  --seed 42 \
  --output data/heston_train_500k_uniform.npz
```

Heston small con Delta para H-6-small en E5:

```bash
python scripts/generate_dataset.py \
  --family heston \
  --sampler uniform \
  --n-samples 100000 \
  --batch-size 5000 \
  --seed 43 \
  --include-delta \
  --output data/heston_train_100k_uniform_delta.npz
```

Heston small sin Delta para H-3-small en E5:

```bash
python scripts/generate_dataset.py \
  --family heston \
  --sampler uniform \
  --n-samples 100000 \
  --batch-size 5000 \
  --seed 48 \
  --output data/heston_train_100k_uniform.npz
```

Heston enfocado para H-5 en E3:

```bash
python scripts/generate_dataset.py \
  --family heston \
  --sampler focused \
  --n-samples 500000 \
  --batch-size 5000 \
  --seed 44 \
  --output data/heston_train_500k_focused.npz
```

## Datasets de Validación

Validation Black-Scholes uniforme:

```bash
python scripts/generate_dataset.py \
  --family black_scholes \
  --sampler uniform \
  --n-samples 50000 \
  --batch-size 20000 \
  --seed 49 \
  --include-delta \
  --output data/bs_validation_50k_uniform_delta.npz
```

Validation Heston uniforme:

```bash
python scripts/generate_dataset.py \
  --family heston \
  --sampler uniform \
  --n-samples 50000 \
  --batch-size 5000 \
  --seed 50 \
  --output data/heston_validation_50k_uniform.npz
```

## Datasets de Test

Test Heston balanceado por bins:

```bash
python scripts/generate_dataset.py \
  --family heston \
  --sampler balanced \
  --samples-per-bin 5000 \
  --batch-size 1000 \
  --seed 46 \
  --include-delta \
  --output data/heston_test_125k_balanced_delta.npz
```

Test Black-Scholes balanceado por bins:

```bash
python scripts/generate_dataset.py \
  --family black_scholes \
  --sampler balanced \
  --samples-per-bin 5000 \
  --batch-size 20000 \
  --seed 47 \
  --include-delta \
  --output data/bs_test_125k_balanced_delta.npz
```

## Ejecución Larga

Usar `tmux` para que el proceso no muera al cerrar SSH:

```bash
tmux new -s generate_heston
source .venv/bin/activate
python scripts/generate_dataset.py ...
```

Para salir sin matar el proceso: `Ctrl-b`, luego `d`. Para volver:

```bash
tmux attach -t generate_heston
```

Para generar toda la tanda de datasets en una sola ejecución, usar:

```bash
tmux new -s generate_all
source .venv/bin/activate
bash scripts/generate_all_datasets.sh
```

El script genera archivos separados en `data/` y logs separados en `results/logs/`. No sobrescribe por defecto. Para relanzar sobrescribiendo:

```bash
OVERWRITE=1 bash scripts/generate_all_datasets.sh
```

También se pueden cambiar rutas sin tocar el script:

```bash
DATA_DIR=/mnt/datasets LOG_DIR=/mnt/logs bash scripts/generate_all_datasets.sh
```

## Formato de Salida

Cada `.npz` contiene:

- `features`: inputs normalizados min-max.
- `raw_inputs`: inputs financieros originales.
- `prices`: precio normalizado `C/K`.
- `deltas`: solo si se usa `--include-delta`.
- `input_names`: nombres de columnas.
- `bin_id`, `moneyness_bin`, `maturity_bin`: solo si se usa `--sampler balanced`.

Cada `.npz` tiene un `.json` asociado con metadatos, dominio, semilla, batch size, tiempo, throughput y tasa de rechazo.

## Checklist Operativo

1. Clonar o actualizar el repositorio.
2. Crear `.venv` con Python 3.11.
3. Instalar `requirements.txt`.
4. Ejecutar `python -m pytest tests/solvers/test_quantlib_reference.py`.
5. Ejecutar benchmark Heston 5k con Delta.
6. Revisar throughput y tasa de rechazo.
7. Lanzar datasets grandes en `tmux`.
8. Guardar logs de consola y `.json` de cada dataset.

## Reglas Importantes

No commitear `data/`. Está gitignored. Si una generación falla, conservar el `.json` y el log de consola para diagnosticar. Si la tasa de rechazo es alta o el throughput es muy bajo, parar y revisar antes de lanzar 500k muestras. No cambiar rangos del dominio, semillas ni nombres de archivo sin documentarlo.
