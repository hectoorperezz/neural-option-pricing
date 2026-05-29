"""Evalúa un surrogate entrenado sobre un test set y escribe un CSV.

Este script solo orquesta, siguiendo ``docs/architecture.md``: los scripts no
deben contener lógica reutilizable. Las decisiones numéricas y estructurales
se delegan en:

* ``src.models.MLP`` para reconstruir la red desde ``config.json``;
* ``BinPartition`` para la rejilla 5x5;
* ``BinEvaluator`` para la evaluación completa;
* ``Report`` para serializar a CSV.

Uso típico::

    python scripts/evaluate_surrogate.py \\
        --checkpoint results/checkpoints/BS-3 \\
        --test       data/bs_test_125k_balanced_delta.npz \\
        --output     results/metrics/BS-3_eval.csv

``--no-iv`` omite la inversión de IV Black-Scholes, que puede ser órdenes de
magnitud más lenta. No se oculta en silencio: si se usa esa opción, las
columnas de IV quedan vacías.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import BinEvaluator, BinPartition, Report
from src.utils import (
    load_mlp_checkpoint,
    load_option_dataset_npz,
    resolve_pricer,
    resolve_torch_device,
)


def parse_args() -> argparse.Namespace:
    """Parser CLI; los ``help`` describen cada flag."""
    parser = argparse.ArgumentParser(
        description="Evalúa un surrogate entrenado sobre un test set y escribe CSV por bin."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Directorio que contiene checkpoint.pt y config.json, tal como los escribe train_surrogate.py.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help="Ruta del archivo .npz de test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Ruta del CSV de salida; debe terminar en .csv.",
    )
    parser.add_argument(
        "--solver",
        choices=("auto", "black_scholes", "heston"),
        default="auto",
        help="Familia del pricer de referencia. 'auto' la infiere por input_names: 4 -> Black-Scholes, 8 -> Heston.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda' o cualquier device válido de Torch.",
    )
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument(
        "--no-iv",
        action="store_true",
        help="Omite la inversión de IV; el CSV deja vacías las columnas de IV.",
    )
    parser.add_argument("--moneyness-min", type=float, default=0.4)
    parser.add_argument("--moneyness-max", type=float, default=2.0)
    return parser.parse_args()


def _top_n_finite(arr: np.ndarray, n: int) -> np.ndarray:
    """Devuelve los ``n`` índices de mayor valor finito en ``arr``."""
    finite_indices = np.where(np.isfinite(arr))[0]
    if finite_indices.size == 0:
        return np.array([], dtype=np.int64)
    order = finite_indices[np.argsort(-arr[finite_indices])]
    return order[:n]


def print_summary(report: Report, elapsed: float) -> None:
    """Imprime resumen global y los tres peores bins por métrica."""
    print()
    print("=== Summary ===")
    print(f"  surrogate_id    {report.surrogate_id}")
    print(f"  n_samples       {report.n_samples:,}")
    print(f"  elapsed         {elapsed:.2f}s")

    def safe_nanmean(arr: np.ndarray | None) -> float:
        if arr is None:
            return float("nan")
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        return float(finite.mean())

    print()
    print("Global nan-mean across bins:")
    print(f"  price MAE       {safe_nanmean(report.price['mean']):.4e}")
    if report.delta is not None:
        print(f"  delta MAE       {safe_nanmean(report.delta['mean']):.4e}")
    if report.iv is not None:
        print(f"  IV MAE          {safe_nanmean(report.iv['mean']):.4e}")
        print(f"  IV failure rate {safe_nanmean(report.iv_failure_rate_per_bin):.2%}")

    def show_worst(name: str, agg: dict[str, np.ndarray] | None) -> None:
        if agg is None:
            return
        worst = _top_n_finite(agg["mean"], n=3)
        if worst.size == 0:
            return
        print()
        print(f"Worst 3 bins by {name} MAE:")
        for bin_index in worst:
            moneyness_idx = int(bin_index) % report.partition.n_moneyness_bins
            maturity_idx = int(bin_index) // report.partition.n_moneyness_bins
            label = report.partition.bin_label(moneyness_idx, maturity_idx)
            print(
                f"  {label:<22} "
                f"mean={agg['mean'][bin_index]:.3e}  "
                f"p95={agg['p95'][bin_index]:.3e}"
            )

    show_worst("price", report.price)
    show_worst("delta", report.delta)
    show_worst("IV", report.iv)


def main() -> None:
    """Entrada del script: evalúa el surrogate y vuelca CSV + resumen."""
    args = parse_args()

    if args.output.suffix.lower() != ".csv":
        raise ValueError("--output must end with .csv")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be strictly positive")
    if args.moneyness_max <= args.moneyness_min:
        raise ValueError("--moneyness-max must be greater than --moneyness-min")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"--checkpoint not found: {args.checkpoint}")
    if not args.test.exists():
        raise FileNotFoundError(f"--test not found: {args.test}")

    device = resolve_torch_device(args.device)
    model, _ = load_mlp_checkpoint(args.checkpoint)
    dataset, bin_id = load_option_dataset_npz(args.test)
    pricer = resolve_pricer(args.solver, dataset.input_names)

    print(f"checkpoint  {args.checkpoint}")
    print(f"test        {args.test}")
    print(f"n_samples   {dataset.features.shape[0]:,}")
    print(f"input_dim   {dataset.features.shape[1]}")
    print(f"deltas      {'yes' if dataset.deltas is not None else 'no'}")
    print(f"bin_id      {'from file' if bin_id is not None else 'computed on the fly'}")
    print(f"device      {device}")
    print(f"solver      {type(pricer).__name__}")
    print(f"compute_iv  {not args.no_iv}")

    evaluator = BinEvaluator(
        partition=BinPartition.default(),
        pricer=pricer,
        device=device,
        batch_size=args.batch_size,
        moneyness_range=(args.moneyness_min, args.moneyness_max),
    )

    started_at = time.perf_counter()
    report = evaluator.evaluate(
        surrogate=model,
        dataset=dataset,
        bin_id=bin_id,
        compute_iv=not args.no_iv,
        surrogate_id=args.checkpoint.name,
        test_path=str(args.test),
    )
    elapsed = time.perf_counter() - started_at

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output)
    print()
    print(f"CSV written: {args.output}")
    print_summary(report, elapsed)


if __name__ == "__main__":
    main()
