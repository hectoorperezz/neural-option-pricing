"""Ejecuta E1 sobre los surrogates baseline documentados.

Este script solo orquesta. Carga BS-3 y/o H-3, construye el
``BinEvaluator`` de cada familia, ejecuta ``PriceVsIVStudy`` y escribe el CSV
por bin junto con los heatmaps exigidos en la metodología::

    "El entregable debe incluir tabla por bin con MAE(C/K), MAE_IV,
    Vega media o proxy de Vega, percentiles altos y tasa de fallos de
    inversión IV, además de heatmaps separados para precio e IV."

Uso típico con ambas familias::

    python scripts/experiments/run_experiment_e1.py \\
        --bs-checkpoint     results/checkpoints/BS-3 \\
        --bs-test           data/bs_test_125k_balanced_delta.npz \\
        --heston-checkpoint results/checkpoints/H-3 \\
        --heston-test       data/heston_test_125k_balanced_delta.npz \\
        --output            results/metrics/e1_table.csv \\
        --figures-dir       results/figures/e1

Se puede omitir una familia; el script ejecuta solo los surrogates
proporcionados y valida las rutas al inicio.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import BinEvaluator, BinPartition
from src.experiments import PriceVsIVStudy, SurrogateInput
from src.solvers import BlackScholesSolver, HestonSolver
from src.utils import load_mlp_checkpoint, load_option_dataset_npz, resolve_torch_device


def parse_args() -> argparse.Namespace:
    """Parser CLI; los ``help`` describen cada flag."""
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta E1 sobre los baseline BS-3 y/o H-3 y escribe el CSV "
            "por bin junto con heatmaps de precio e IV."
        )
    )
    parser.add_argument("--bs-checkpoint", type=Path, default=None)
    parser.add_argument("--bs-test", type=Path, default=None)
    parser.add_argument("--heston-checkpoint", type=Path, default=None)
    parser.add_argument("--heston-test", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True, help="Ruta del CSV de salida; debe terminar en .csv.")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directorio opcional para escribir los heatmaps PNG de precio e IV.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help=(
            "Número de workers CPU para la inversión de IV Black-Scholes, "
            "la fase lenta de E1. Por defecto usa los cores disponibles "
            "menos dos."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Desactiva la barra tqdm durante la inversión de IV.",
    )
    return parser.parse_args()


def _build_input(
    surrogate_id: str,
    checkpoint_dir: Path,
    test_path: Path,
    pricer: Any,
    device: str,
    batch_size: int,
    iv_workers: int,
    iv_progress: bool,
) -> SurrogateInput:
    """Carga modelo + dataset y empaqueta el ``SurrogateInput`` para el estudio."""
    model, _ = load_mlp_checkpoint(checkpoint_dir)
    dataset, bin_id = load_option_dataset_npz(test_path)
    evaluator = BinEvaluator(
        partition=BinPartition.default(),
        pricer=pricer,
        device=device,
        batch_size=batch_size,
        iv_workers=iv_workers,
        iv_progress=iv_progress,
    )
    return SurrogateInput(
        surrogate_id=surrogate_id,
        model=model,
        dataset=dataset,
        evaluator=evaluator,
        bin_id=bin_id,
    )


def main() -> None:
    """Entrada del script: corre ``PriceVsIVStudy`` y vuelca CSV + heatmaps."""
    args = parse_args()
    if args.output.suffix.lower() != ".csv":
        raise ValueError("--output must end with .csv")

    bs_paths_set = (args.bs_checkpoint is not None) or (args.bs_test is not None)
    heston_paths_set = (args.heston_checkpoint is not None) or (args.heston_test is not None)
    if not bs_paths_set and not heston_paths_set:
        raise ValueError(
            "provide at least one of (--bs-checkpoint + --bs-test) or "
            "(--heston-checkpoint + --heston-test)"
        )
    if bs_paths_set and (args.bs_checkpoint is None or args.bs_test is None):
        raise ValueError("both --bs-checkpoint and --bs-test must be provided together")
    if heston_paths_set and (
        args.heston_checkpoint is None or args.heston_test is None
    ):
        raise ValueError(
            "both --heston-checkpoint and --heston-test must be provided together"
        )

    device = resolve_torch_device(args.device)
    iv_workers = max(1, int(args.workers))
    iv_progress = not args.no_progress
    inputs: list[SurrogateInput] = []
    if bs_paths_set:
        inputs.append(
            _build_input(
                surrogate_id=args.bs_checkpoint.name,
                checkpoint_dir=args.bs_checkpoint,
                test_path=args.bs_test,
                pricer=BlackScholesSolver(),
                device=device,
                batch_size=args.batch_size,
                iv_workers=iv_workers,
                iv_progress=iv_progress,
            )
        )
    if heston_paths_set:
        inputs.append(
            _build_input(
                surrogate_id=args.heston_checkpoint.name,
                checkpoint_dir=args.heston_checkpoint,
                test_path=args.heston_test,
                pricer=HestonSolver(),
                device=device,
                batch_size=args.batch_size,
                iv_workers=iv_workers,
                iv_progress=iv_progress,
            )
        )

    print(f"E1 — surrogates: {', '.join(item.surrogate_id for item in inputs)}")
    print(f"device         : {device}")
    print(f"iv_workers     : {iv_workers}")
    print(f"iv_progress    : {iv_progress}")

    study = PriceVsIVStudy(inputs=tuple(inputs))
    started_at = time.perf_counter()
    result = study.run()
    elapsed = time.perf_counter() - started_at

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output)
    print(f"\nCSV written: {args.output}")

    if args.figures_dir is not None:
        figures = result.to_heatmaps(args.figures_dir)
        print(f"Heatmaps:    {len(figures)} PNGs in {args.figures_dir}")

    print(f"\nElapsed: {elapsed:.2f}s")
    print()
    print(result.summary)


if __name__ == "__main__":
    main()
