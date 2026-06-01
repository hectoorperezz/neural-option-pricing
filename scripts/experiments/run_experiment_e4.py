"""Ejecuta E4 sobre el surrogate Heston documentado.

Este script solo orquesta. Carga H-3, construye ``TimingBenchmark`` con el
test balanced compartido, ejecuta ``EfficiencyStudy`` en CPU y, si existe, en
CUDA, y escribe el CSV por ``(device, batch_size)`` junto con la figura de
``speedup`` frente a ``batch_size``.

Uso típico::

    python scripts/experiments/run_experiment_e4.py \\
        --checkpoint  results/checkpoints/H-3 \\
        --test        data/heston_test_125k_balanced_delta.npz \\
        --output      results/metrics/e4_table.csv \\
        --plot        results/figures/e4/speedup_vs_batch.png

Por defecto usa CPU y CUDA si hay GPU visible. ``--devices cpu`` fuerza solo
CPU; ``--devices cpu cuda`` fuerza ambos.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import TimingBenchmark
from src.evaluation.timing import default_solver_workers
from src.experiments import EfficiencyResult, EfficiencyStudy
from src.solvers import HestonSolver
from src.utils import load_mlp_checkpoint, load_npz_features_and_raw_inputs


def parse_args() -> argparse.Namespace:
    """Parser CLI; los ``help`` describen cada flag."""
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta E4, benchmark temporal de H-3 frente al solver Heston, "
            "y escribe CSV por device/lote más figura de speedup."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Directorio del checkpoint Heston a cronometrar; sujeto previsto: H-3.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help=(
            "Test set Heston cuyos primeros N puntos alimentan cada lote. "
            "Por defecto: data/heston_test_125k_balanced_delta.npz."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV de destino, una fila por par (device, batch_size).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="PNG opcional para la figura de speedup frente a tamaño de lote.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help=(
            "Devices a cronometrar. Por defecto: cpu y cuda si está disponible. "
            "Usa '--devices cpu' para forzar solo CPU."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Sobrescribe los tamaños de lote del protocolo. Por defecto: "
            "100 1000 10000 100000, como está pre-registrado en tasks.md §E4."
        ),
    )
    parser.add_argument(
        "--n-warmups",
        type=int,
        default=None,
        help="Sobrescribe el número de warmups. Por defecto: 3.",
    )
    parser.add_argument(
        "--n-repetitions",
        type=int,
        default=None,
        help="Sobrescribe el número de repeticiones medidas. Por defecto: 10.",
    )
    parser.add_argument(
        "--solver-workers",
        type=int,
        default=None,
        help=(
            "Procesos usados para paralelizar el solver Heston. Por defecto: "
            "cpu_count - 2, igual que en el generador de datasets. Usa 1 "
            "para forzar ejecución serie en el proceso principal."
        ),
    )
    return parser.parse_args()


def resolve_devices(requested: list[str] | None) -> tuple[str, ...]:
    """Devuelve los devices a cronometrar; añade ``cuda`` si está disponible."""
    if requested is not None:
        return tuple(requested)
    if torch.cuda.is_available():
        return ("cpu", "cuda")
    return ("cpu",)


def main() -> EfficiencyResult:
    """Entrada del script: corre ``EfficiencyStudy`` y vuelca CSV + plot."""
    args = parse_args()
    devices = resolve_devices(args.devices)

    print("E4 — Eficiencia computacional (H-3 vs Heston solver)")
    print(f"checkpoint   : {args.checkpoint}")
    print(f"test         : {args.test}")
    print(f"output       : {args.output}")
    if args.plot is not None:
        print(f"plot         : {args.plot}")
    print(f"devices      : {', '.join(devices)}")

    features, raw_inputs, input_names = load_npz_features_and_raw_inputs(args.test)
    surrogate, _config = load_mlp_checkpoint(args.checkpoint)

    benchmark_kwargs: dict[str, Any] = {
        "pricer": HestonSolver(),
        "raw_inputs": raw_inputs,
        "features": features,
        "input_names": input_names,
    }
    if args.batch_sizes is not None:
        benchmark_kwargs["batch_sizes"] = tuple(args.batch_sizes)
    if args.n_warmups is not None:
        benchmark_kwargs["n_warmups"] = args.n_warmups
    if args.n_repetitions is not None:
        benchmark_kwargs["n_repetitions"] = args.n_repetitions
    solver_workers = (
        args.solver_workers if args.solver_workers is not None
        else default_solver_workers()
    )
    benchmark_kwargs["solver_workers"] = solver_workers
    benchmark = TimingBenchmark(**benchmark_kwargs)

    print(
        "protocol     : batch_sizes={}, n_warmups={}, n_repetitions={}, "
        "solver_workers={}".format(
            benchmark.batch_sizes,
            benchmark.n_warmups,
            benchmark.n_repetitions,
            benchmark.solver_workers,
        )
    )

    study = EfficiencyStudy(
        benchmark=benchmark,
        surrogate=surrogate,
        surrogate_id=args.checkpoint.name,
        devices=devices,
    )

    started_at = time.perf_counter()
    result = study.run()
    elapsed = time.perf_counter() - started_at

    result.to_csv(args.output)
    print(f"\nCSV written: {args.output}  (elapsed {elapsed:.2f}s)")
    if args.plot is not None:
        result.to_plot(args.plot)
        print(f"Plot written: {args.plot}")

    print(result.summary)
    return result


if __name__ == "__main__":
    main()
