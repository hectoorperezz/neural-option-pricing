"""Run E3 (Sampling Study) on the documented Heston surrogates.

Orchestration only. The script loads the uniform baseline (``H-3``) and
the focused candidate (``H-5``) — the two subjects that
``docs/tasks.md`` §E3 and ``docs/metodologia.md`` §"Experimento E3 —
Muestreo uniforme frente a enfocado" designate — constructs a single
:class:`BinEvaluator` (Heston solver) backed by the shared balanced test
set, builds :class:`SurrogateInput` instances labelled with the sampler
each surrogate was trained on (``"uniform"`` for the baseline checkpoint
and ``"focused"`` for the candidate), hands them to
:class:`SamplingStudy`, and writes the per-bin long-format CSV plus the
price and IV heatmaps that ``docs/tasks.md`` §"Fase 3" mandates as the
deliverable.

Typical invocation::

    python scripts/run_experiment_e3.py \\
        --uniform-checkpoint results/checkpoints/H-3 \\
        --focused-checkpoint results/checkpoints/H-5 \\
        --test               data/heston_test_125k_balanced_delta.npz \\
        --output             results/metrics/e3_table.csv \\
        --figures-dir        results/figures/e3

The metric primary printed in the summary is the per-bin ``MAE_IV``
averaged over the three critical ATM bins (weekly, short, medium-short),
and the verdict is the pre-registered fuerte/débil/negativo
classification from ``metodologia.md`` §E3.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.generator import OptionDataset
from src.evaluation import BinEvaluator, BinPartition
from src.experiments import ExperimentResult, SamplingStudy, SurrogateInput
from src.models import MLP
from src.solvers import HestonSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run E3 (uniform H-3 vs focused H-5) and write per-bin CSV "
            "plus price/IV heatmaps."
        )
    )
    parser.add_argument(
        "--uniform-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory for the uniform baseline (e.g. H-3).",
    )
    parser.add_argument(
        "--focused-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory for the focused candidate (e.g. H-5).",
    )
    parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help=(
            "Balanced Heston test set shared by both surrogates "
            "(documented in tasks.md §E3 as the same balanced test)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV (long-format, one row per surrogate x bin).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Optional directory for price/IV heatmap PNGs.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument(
        "--iv-workers",
        type=int,
        default=1,
        help=(
            "Worker processes for IV inversion. 1 keeps it serial; "
            "values above 1 fan out via ProcessPoolExecutor."
        ),
    )
    parser.add_argument(
        "--iv-progress",
        action="store_true",
        help="Show a tqdm bar during IV inversion (long jobs).",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_checkpoint(checkpoint_dir: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    config_path = checkpoint_dir / "config.json"
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config.json at {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint.pt at {checkpoint_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = MLP(
        input_dim=int(config["input_dim"]),
        hidden_width=int(config["hidden_width"]),
        hidden_layers=int(config["hidden_layers"]),
        activation=str(config["activation"]),
    )
    state_dict = checkpoint.get("best_state_dict") or checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def load_test_dataset(path: Path) -> tuple[OptionDataset, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(f"test set not found at {path}")
    data = np.load(path, allow_pickle=False)
    features = np.asarray(data["features"], dtype=np.float32)
    raw_inputs = np.asarray(data["raw_inputs"], dtype=np.float32)
    prices = np.asarray(data["prices"], dtype=np.float32).reshape(-1, 1)
    deltas: np.ndarray | None = None
    if "deltas" in data.files:
        deltas = np.asarray(data["deltas"], dtype=np.float32).reshape(-1, 1)
    input_names = tuple(str(name) for name in np.asarray(data["input_names"]).tolist())
    bin_id: np.ndarray | None = None
    if "bin_id" in data.files:
        bin_id = np.asarray(data["bin_id"], dtype=np.int64)
    return (
        OptionDataset(
            features=torch.from_numpy(features),
            prices=torch.from_numpy(prices),
            deltas=None if deltas is None else torch.from_numpy(deltas),
            raw_inputs=torch.from_numpy(raw_inputs),
            input_names=input_names,
        ),
        bin_id,
    )


def _build_input(
    *,
    checkpoint_dir: Path,
    sampler: str,
    dataset: OptionDataset,
    bin_id: np.ndarray | None,
    evaluator: BinEvaluator,
) -> SurrogateInput:
    model, _config = load_checkpoint(checkpoint_dir)
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
    dataset, bin_id = load_test_dataset(test_path)
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
    device = resolve_device(args.device)
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
