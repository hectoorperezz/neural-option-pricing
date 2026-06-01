"""Ejecuta E5 sobre los surrogates Heston documentados.

Este script solo orquesta. Carga H-3-small y H-6-small, opcionalmente H-3
como baseline grande, comparte el mismo test balanced Heston, ejecuta
``DMLStudy`` y escribe el CSV largo por bin junto con heatmaps de precio y
Delta.

Uso típico::

    python scripts/experiments/run_experiment_e5.py \\
        --small-price-checkpoint results/checkpoints/H-3-small \\
        --small-dml-checkpoint   results/checkpoints/H-6-small \\
        --baseline-checkpoint    results/checkpoints/H-3 \\
        --test                   data/heston_test_125k_balanced_delta.npz \\
        --output                 results/metrics/e5_table.csv \\
        --figures-dir            results/figures/e5

``--baseline-checkpoint`` es opcional. Sin él, el experimento mantiene su
veredicto pre-registrado, que solo depende de H-3-small y H-6-small, pero no
incluye la distancia frente a H-3 en el resumen.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.generator import OptionDataset
from src.evaluation import BinEvaluator, BinPartition
from src.experiments import DMLStudy, ExperimentResult, SurrogateInput
from src.solvers import HestonSolver
from src.utils import load_mlp_checkpoint, load_option_dataset_npz, resolve_torch_device


def parse_args() -> argparse.Namespace:
    """Parser CLI; los ``help`` describen cada flag."""
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta E5: H-3-small frente a H-6-small, con H-3 opcional "
            "como baseline, y escribe CSV por bin más heatmaps."
        )
    )
    parser.add_argument(
        "--small-price-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint del surrogate pequeño entrenado solo con precio, por ejemplo H-3-small.",
    )
    parser.add_argument(
        "--small-dml-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint del surrogate pequeño DML, precio más Delta, por ejemplo H-6-small.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint opcional del baseline grande, por ejemplo H-3. "
            "Si se proporciona, el resumen reporta la distancia del DML "
            "frente a este baseline."
        ),
    )
    parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help=(
            "Test set Heston con Delta, compartido por todos los surrogates "
            "según tasks.md §E5."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="CSV de destino en formato largo, una fila por surrogate y bin.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directorio opcional para heatmaps PNG de precio y Delta.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    return parser.parse_args()


def _build_input(
    *,
    checkpoint_dir: Path,
    role: str,
    dataset: OptionDataset,
    bin_id: np.ndarray | None,
    evaluator: BinEvaluator,
) -> SurrogateInput:
    """Carga el checkpoint y lo etiqueta con ``role`` y ``loss`` para DMLStudy."""
    model, config = load_mlp_checkpoint(checkpoint_dir)
    labels = {
        "role": role,
        "loss": str(config.get("loss", "")),
    }
    return SurrogateInput(
        surrogate_id=checkpoint_dir.name,
        model=model,
        dataset=dataset,
        evaluator=evaluator,
        bin_id=bin_id,
        labels=labels,
    )


def _run(
    *,
    small_price_checkpoint: Path,
    small_dml_checkpoint: Path,
    baseline_checkpoint: Path | None,
    test_path: Path,
    output_csv: Path,
    figures_dir: Path | None,
    device: str,
    batch_size: int,
) -> ExperimentResult:
    """Corre ``DMLStudy`` y vuelca CSV + heatmaps de precio y Delta."""
    dataset, bin_id = load_option_dataset_npz(test_path, require_delta=True)
    evaluator = BinEvaluator(
        partition=BinPartition.default(),
        pricer=HestonSolver(),
        device=device,
        batch_size=batch_size,
    )
    inputs: list[SurrogateInput] = [
        _build_input(
            checkpoint_dir=small_price_checkpoint,
            role="small_price",
            dataset=dataset,
            bin_id=bin_id,
            evaluator=evaluator,
        ),
        _build_input(
            checkpoint_dir=small_dml_checkpoint,
            role="small_dml",
            dataset=dataset,
            bin_id=bin_id,
            evaluator=evaluator,
        ),
    ]
    if baseline_checkpoint is not None:
        inputs.append(
            _build_input(
                checkpoint_dir=baseline_checkpoint,
                role="baseline_large",
                dataset=dataset,
                bin_id=bin_id,
                evaluator=evaluator,
            )
        )

    ids = ", ".join(item.surrogate_id for item in inputs)
    print(f"\n=== E5: {ids} ===")
    study = DMLStudy(inputs=tuple(inputs))
    started_at = time.perf_counter()
    result = study.run()
    elapsed = time.perf_counter() - started_at

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv)
    print(f"CSV written: {output_csv}  (elapsed {elapsed:.2f}s)")

    if figures_dir is not None:
        figures = result.to_heatmaps(figures_dir, metrics=("price", "delta"))
        print(f"Heatmaps:    {len(figures)} PNGs in {figures_dir}")

    print(result.summary)
    return result


def main() -> None:
    """Entrada del script: ejecuta E5 sobre los pequeños (+ baseline opcional)."""
    args = parse_args()
    device = resolve_torch_device(args.device)
    print("E5 — Differential ML con Delta")
    print(f"device       : {device}")
    print(f"test         : {args.test}")
    print(f"small_price  : {args.small_price_checkpoint}")
    print(f"small_dml    : {args.small_dml_checkpoint}")
    if args.baseline_checkpoint is not None:
        print(f"baseline     : {args.baseline_checkpoint}")
    print(f"output       : {args.output}")
    if args.figures_dir is not None:
        print(f"figures_dir  : {args.figures_dir}")

    _run(
        small_price_checkpoint=args.small_price_checkpoint,
        small_dml_checkpoint=args.small_dml_checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        test_path=args.test,
        output_csv=args.output,
        figures_dir=args.figures_dir,
        device=device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
