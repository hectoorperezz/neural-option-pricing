"""Ejecuta E3 sobre los surrogates Heston documentados.

Este script solo orquesta. Carga H-3 como baseline uniforme y H-5 como
candidato enfocado, comparte el mismo test balanced, etiqueta cada input con
su sampler, ejecuta ``SamplingStudy`` y escribe el CSV largo por bin junto
con heatmaps de precio e IV.

Uso típico::

    python scripts/run_experiment_e3.py \\
        --uniform-checkpoint results/checkpoints/H-3 \\
        --focused-checkpoint results/checkpoints/H-5 \\
        --test               data/heston_test_125k_balanced_delta.npz \\
        --output             results/metrics/e3_table.csv \\
        --figures-dir        results/figures/e3

La métrica primaria del resumen es ``MAE_IV`` medio en los tres bins críticos
ATM: weekly, short y medium-short. El veredicto sigue la clasificación
pre-registrada en ``metodologia.md``.
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

from src.datasets.generator import OptionDataset
from src.evaluation import BinEvaluator, BinPartition
from src.experiments import ExperimentResult, SamplingStudy, SurrogateInput
from src.solvers import HestonSolver
from src.utils import load_mlp_checkpoint, load_option_dataset_npz, resolve_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta E3 (H-3 uniforme frente a H-5 enfocado) y escribe "
            "CSV por bin junto con heatmaps de precio e IV."
        )
    )
    parser.add_argument(
        "--uniform-checkpoint",
        type=Path,
        required=True,
        help="Directorio del checkpoint baseline uniforme, por ejemplo H-3.",
    )
    parser.add_argument(
        "--focused-checkpoint",
        type=Path,
        required=True,
        help="Directorio del checkpoint candidato enfocado, por ejemplo H-5.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help=(
            "Test set Heston balanced compartido por ambos surrogates, "
            "según lo documentado en tasks.md §E3."
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
        help="Directorio opcional para heatmaps PNG de precio e IV.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument(
        "--iv-workers",
        type=int,
        default=1,
        help=(
            "Procesos para la inversión de IV. Con 1 se mantiene en serie; "
            "valores mayores reparten el trabajo con ProcessPoolExecutor."
        ),
    )
    parser.add_argument(
        "--iv-progress",
        action="store_true",
        help="Muestra una barra tqdm durante la inversión de IV.",
    )
    return parser.parse_args()


def _build_input(
    *,
    checkpoint_dir: Path,
    sampler: str,
    dataset: OptionDataset,
    bin_id: np.ndarray | None,
    evaluator: BinEvaluator,
) -> SurrogateInput:
    model, _config = load_mlp_checkpoint(checkpoint_dir)
    return SurrogateInput(
        surrogate_id=checkpoint_dir.name,
        model=model,
        dataset=dataset,
        evaluator=evaluator,
        bin_id=bin_id,
        labels={"sampler": sampler},
    )


def _run(
    *,
    uniform_checkpoint: Path,
    focused_checkpoint: Path,
    test_path: Path,
    output_csv: Path,
    figures_dir: Path | None,
    device: str,
    batch_size: int,
    iv_workers: int,
    iv_progress: bool,
) -> ExperimentResult:
    dataset, bin_id = load_option_dataset_npz(test_path)
    evaluator = BinEvaluator(
        partition=BinPartition.default(),
        pricer=HestonSolver(),
        device=device,
        batch_size=batch_size,
        iv_workers=iv_workers,
        iv_progress=iv_progress,
    )
    inputs = (
        _build_input(
            checkpoint_dir=uniform_checkpoint,
            sampler="uniform",
            dataset=dataset,
            bin_id=bin_id,
            evaluator=evaluator,
        ),
        _build_input(
            checkpoint_dir=focused_checkpoint,
            sampler="focused",
            dataset=dataset,
            bin_id=bin_id,
            evaluator=evaluator,
        ),
    )
    print(
        f"\n=== E3: uniform={inputs[0].surrogate_id} vs "
        f"focused={inputs[1].surrogate_id} ==="
    )
    study = SamplingStudy(inputs=inputs)
    started_at = time.perf_counter()
    result = study.run()
    elapsed = time.perf_counter() - started_at

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv)
    print(f"CSV written: {output_csv}  (elapsed {elapsed:.2f}s)")

    if figures_dir is not None:
        figures = result.to_heatmaps(figures_dir, metrics=("price", "iv"))
        print(f"Heatmaps:    {len(figures)} PNGs in {figures_dir}")

    print(result.summary)
    return result


def main() -> None:
    args = parse_args()
    device = resolve_torch_device(args.device)
    print("E3 — Sampling Study (uniform vs focused)")
    print(f"device       : {device}")
    print(f"test         : {args.test}")
    print(f"uniform      : {args.uniform_checkpoint}")
    print(f"focused      : {args.focused_checkpoint}")
    print(f"output       : {args.output}")
    if args.figures_dir is not None:
        print(f"figures_dir  : {args.figures_dir}")

    _run(
        uniform_checkpoint=args.uniform_checkpoint,
        focused_checkpoint=args.focused_checkpoint,
        test_path=args.test,
        output_csv=args.output,
        figures_dir=args.figures_dir,
        device=device,
        batch_size=args.batch_size,
        iv_workers=args.iv_workers,
        iv_progress=args.iv_progress,
    )


if __name__ == "__main__":
    main()
