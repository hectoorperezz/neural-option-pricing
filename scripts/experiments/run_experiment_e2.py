"""Ejecuta E2 sobre los surrogates de activaciones documentados.

Este script solo orquesta. Carga BS-1..4 y/o H-1..4, construye el
``BinEvaluator`` de cada familia, etiqueta cada checkpoint con su activación
desde ``config.json``, ejecuta ``ActivationStudy`` y escribe el CSV largo por
bin junto con heatmaps de precio y Delta.

Uso típico con ambas familias::

    python scripts/experiments/run_experiment_e2.py \\
        --bs-checkpoints   results/checkpoints/BS-1 results/checkpoints/BS-2 \\
                           results/checkpoints/BS-3 results/checkpoints/BS-4 \\
        --bs-test          data/bs_test_125k_balanced_delta.npz \\
        --heston-checkpoints results/checkpoints/H-1 results/checkpoints/H-2 \\
                             results/checkpoints/H-3 results/checkpoints/H-4 \\
        --heston-test      data/heston_test_125k_balanced_delta.npz \\
        --output-dir       results/metrics \\
        --figures-dir      results/figures/e2

Se puede omitir una familia; basta con proporcionar checkpoints y test de la
familia que se quiera evaluar.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import BinEvaluator, BinPartition
from src.experiments import ActivationStudy, ExperimentResult, SurrogateInput
from src.solvers import BlackScholesSolver, HestonSolver
from src.utils import load_mlp_checkpoint, load_option_dataset_npz, resolve_torch_device


def parse_args() -> argparse.Namespace:
    """Parser CLI; los ``help`` describen cada flag."""
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta E2 sobre BS-1..4 y/o H-1..4 y escribe CSVs por bin "
            "junto con heatmaps de precio y Delta."
        )
    )
    parser.add_argument(
        "--bs-checkpoints",
        type=Path,
        nargs="+",
        default=None,
        help="Uno o varios directorios de checkpoints BS.",
    )
    parser.add_argument("--bs-test", type=Path, default=None)
    parser.add_argument(
        "--heston-checkpoints",
        type=Path,
        nargs="+",
        default=None,
        help="Uno o varios directorios de checkpoints Heston.",
    )
    parser.add_argument("--heston-test", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directorio para los CSV por familia (e2_bs.csv, e2_heston.csv).",
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


def _build_family_inputs(
    checkpoint_dirs: list[Path],
    test_path: Path,
    pricer: Any,
    device: str,
    batch_size: int,
) -> list[SurrogateInput]:
    """Construye un ``SurrogateInput`` por checkpoint, etiquetado con su activación."""
    dataset, bin_id = load_option_dataset_npz(test_path)
    evaluator = BinEvaluator(
        partition=BinPartition.default(),
        pricer=pricer,
        device=device,
        batch_size=batch_size,
    )
    inputs: list[SurrogateInput] = []
    for checkpoint_dir in checkpoint_dirs:
        model, config = load_mlp_checkpoint(checkpoint_dir)
        labels = {"activation": str(config.get("activation", ""))}
        inputs.append(
            SurrogateInput(
                surrogate_id=checkpoint_dir.name,
                model=model,
                dataset=dataset,
                evaluator=evaluator,
                bin_id=bin_id,
                labels=labels,
            )
        )
    return inputs


def _run_family(
    family_label: str,
    inputs: list[SurrogateInput],
    output_csv: Path,
    figures_dir: Path | None,
    figures_subdir_name: str,
) -> ExperimentResult:
    """Corre ``ActivationStudy`` para una familia y persiste CSV + heatmaps."""
    print(
        f"\n=== {family_label}: {len(inputs)} surrogates "
        f"({', '.join(item.surrogate_id for item in inputs)}) ==="
    )
    study = ActivationStudy(inputs=tuple(inputs), family_label=family_label)
    started_at = time.perf_counter()
    result = study.run()
    elapsed = time.perf_counter() - started_at

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv)
    print(f"CSV written: {output_csv}  (elapsed {elapsed:.2f}s)")

    if figures_dir is not None:
        subdir = figures_dir / figures_subdir_name
        figures = result.to_heatmaps(subdir, metrics=("price", "delta"))
        print(f"Heatmaps:    {len(figures)} PNGs in {subdir}")

    print(result.summary)
    return result


def main() -> None:
    """Entrada del script: ejecuta E2 por familia (BS y/o Heston)."""
    args = parse_args()

    bs_has_checkpoints = args.bs_checkpoints is not None
    bs_has_test = args.bs_test is not None
    heston_has_checkpoints = args.heston_checkpoints is not None
    heston_has_test = args.heston_test is not None

    if bs_has_checkpoints != bs_has_test:
        raise ValueError(
            "--bs-checkpoints and --bs-test must be provided together (or both omitted)"
        )
    if heston_has_checkpoints != heston_has_test:
        raise ValueError(
            "--heston-checkpoints and --heston-test must be provided together "
            "(or both omitted)"
        )
    if not bs_has_checkpoints and not heston_has_checkpoints:
        raise ValueError(
            "provide at least one family (--bs-checkpoints + --bs-test or "
            "--heston-checkpoints + --heston-test)"
        )

    device = resolve_torch_device(args.device)
    print(f"E2 — Activations and Delta quality")
    print(f"device       : {device}")
    print(f"output_dir   : {args.output_dir}")
    if args.figures_dir is not None:
        print(f"figures_dir  : {args.figures_dir}")

    if bs_has_checkpoints:
        bs_inputs = _build_family_inputs(
            checkpoint_dirs=list(args.bs_checkpoints),
            test_path=args.bs_test,
            pricer=BlackScholesSolver(),
            device=device,
            batch_size=args.batch_size,
        )
        _run_family(
            family_label="Black-Scholes",
            inputs=bs_inputs,
            output_csv=args.output_dir / "e2_bs.csv",
            figures_dir=args.figures_dir,
            figures_subdir_name="e2_bs",
        )

    if heston_has_checkpoints:
        heston_inputs = _build_family_inputs(
            checkpoint_dirs=list(args.heston_checkpoints),
            test_path=args.heston_test,
            pricer=HestonSolver(),
            device=device,
            batch_size=args.batch_size,
        )
        _run_family(
            family_label="Heston",
            inputs=heston_inputs,
            output_csv=args.output_dir / "e2_heston.csv",
            figures_dir=args.figures_dir,
            figures_subdir_name="e2_heston",
        )


if __name__ == "__main__":
    main()
