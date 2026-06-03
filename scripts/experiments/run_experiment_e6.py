"""Ejecuta E6 sobre los surrogates Heston documentados.

Este script solo orquesta. Carga H-3 como baseline (4x128 con `lr`
fijo), H-7-shallow (2x128, `lr` fijo), H-8-deep (6x128, `lr` fijo) y
H-9-lr-schedule (4x128 con `ReduceLROnPlateau`), comparte el mismo test
balanced Heston, ejecuta ``ArchitectureStudy`` y escribe el CSV largo
por bin junto con heatmaps de precio (y Delta cuando el test los trae).

Uso típico::

    python scripts/experiments/run_experiment_e6.py \\
        --baseline-checkpoint  results/checkpoints/H-3 \\
        --shallow-checkpoint   results/checkpoints/H-7-shallow \\
        --deep-checkpoint      results/checkpoints/H-8-deep \\
        --scheduler-checkpoint results/checkpoints/H-9-lr-schedule \\
        --test                 data/heston_test_125k_balanced_delta.npz \\
        --output               results/metrics/e6_table.csv \\
        --figures-dir          results/figures/e6

Los cuatro checkpoints son opcionales individualmente, pero al menos
dos deben proporcionarse para que la comparación tenga sentido. El
ranking del resumen es por ``MAE(C/K)`` medio por bin; si el test
incluye Delta, también se reporta ``MAE_Delta`` como diagnóstico.
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
from src.experiments import ArchitectureStudy, ExperimentResult, SurrogateInput
from src.solvers import HestonSolver
from src.utils import load_mlp_checkpoint, load_option_dataset_npz, resolve_torch_device


def parse_args() -> argparse.Namespace:
    """Parser CLI; los ``help`` describen cada flag."""
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta E6: H-3 frente a sus variantes de profundidad "
            "(H-7-shallow, H-8-deep) y de scheduler (H-9-lr-schedule), "
            "y escribe CSV por bin más heatmaps."
        )
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint del baseline 4x128 con `lr` fijo, por ejemplo H-3.",
    )
    parser.add_argument(
        "--shallow-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint de la variante con menor profundidad, por ejemplo H-7-shallow.",
    )
    parser.add_argument(
        "--deep-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint de la variante con mayor profundidad, por ejemplo H-8-deep.",
    )
    parser.add_argument(
        "--scheduler-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint de la variante con scheduler, por ejemplo H-9-lr-schedule.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help=(
            "Test set Heston balanced compartido por todos los surrogates, "
            "según lo documentado en metodologia.md."
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
        help="Directorio opcional para heatmaps PNG de precio (y Delta si el test los trae).",
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
    """Carga el checkpoint y lo etiqueta con ``role`` y ``architecture``."""
    model, config = load_mlp_checkpoint(checkpoint_dir)
    architecture = _architecture_label_from_config(config)
    return SurrogateInput(
        surrogate_id=checkpoint_dir.name,
        model=model,
        dataset=dataset,
        evaluator=evaluator,
        bin_id=bin_id,
        labels={"role": role, "architecture": architecture},
    )


def _architecture_label_from_config(config: dict) -> str:
    """Construye una etiqueta corta a partir de la profundidad y el scheduler."""
    layers = int(config.get("hidden_layers", 0))
    width = int(config.get("hidden_width", 0))
    scheduler = str(config.get("scheduler", "none")).lower()
    base = f"{layers}x{width}"
    if scheduler != "none":
        return f"{base}+{scheduler}"
    return base


def _run(
    *,
    baseline_checkpoint: Path | None,
    shallow_checkpoint: Path | None,
    deep_checkpoint: Path | None,
    scheduler_checkpoint: Path | None,
    test_path: Path,
    output_csv: Path,
    figures_dir: Path | None,
    device: str,
    batch_size: int,
) -> ExperimentResult:
    """Corre ``ArchitectureStudy`` y vuelca CSV + heatmaps."""
    dataset, bin_id = load_option_dataset_npz(test_path)
    evaluator = BinEvaluator(
        partition=BinPartition.default(),
        pricer=HestonSolver(),
        device=device,
        batch_size=batch_size,
    )

    inputs: list[SurrogateInput] = []
    for checkpoint, role in (
        (baseline_checkpoint, "baseline"),
        (shallow_checkpoint, "shallow"),
        (deep_checkpoint, "deep"),
        (scheduler_checkpoint, "lr_schedule"),
    ):
        if checkpoint is None:
            continue
        inputs.append(
            _build_input(
                checkpoint_dir=checkpoint,
                role=role,
                dataset=dataset,
                bin_id=bin_id,
                evaluator=evaluator,
            )
        )

    if len(inputs) < 2:
        raise ValueError(
            "E6 requires at least two checkpoints to compare; pass two or "
            "more of --baseline-checkpoint, --shallow-checkpoint, "
            "--deep-checkpoint and --scheduler-checkpoint."
        )

    ids = ", ".join(item.surrogate_id for item in inputs)
    print(f"\n=== E6: {ids} ===")
    study = ArchitectureStudy(inputs=tuple(inputs))
    started_at = time.perf_counter()
    result = study.run()
    elapsed = time.perf_counter() - started_at

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv)
    print(f"CSV written: {output_csv}  (elapsed {elapsed:.2f}s)")

    if figures_dir is not None:
        metrics: tuple[str, ...]
        if dataset.deltas is not None:
            metrics = ("price", "delta")
        else:
            metrics = ("price",)
        figures = result.to_heatmaps(figures_dir, metrics=metrics)
        print(f"Heatmaps:    {len(figures)} PNGs in {figures_dir}")

    print(result.summary)
    return result


def main() -> None:
    """Entrada del script: ejecuta E6 sobre las variantes documentadas."""
    args = parse_args()
    device = resolve_torch_device(args.device)
    print("E6 — Profundidad de red y learning-rate scheduling")
    print(f"device       : {device}")
    print(f"test         : {args.test}")
    if args.baseline_checkpoint is not None:
        print(f"baseline     : {args.baseline_checkpoint}")
    if args.shallow_checkpoint is not None:
        print(f"shallow      : {args.shallow_checkpoint}")
    if args.deep_checkpoint is not None:
        print(f"deep         : {args.deep_checkpoint}")
    if args.scheduler_checkpoint is not None:
        print(f"lr_schedule  : {args.scheduler_checkpoint}")
    print(f"output       : {args.output}")
    if args.figures_dir is not None:
        print(f"figures_dir  : {args.figures_dir}")

    _run(
        baseline_checkpoint=args.baseline_checkpoint,
        shallow_checkpoint=args.shallow_checkpoint,
        deep_checkpoint=args.deep_checkpoint,
        scheduler_checkpoint=args.scheduler_checkpoint,
        test_path=args.test,
        output_csv=args.output,
        figures_dir=args.figures_dir,
        device=device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
