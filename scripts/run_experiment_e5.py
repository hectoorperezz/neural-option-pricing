"""Run E5 (Differential ML) on the documented Heston surrogates.

Orchestration only. The script loads the two small surrogates that
``docs/tasks.md`` §E5 designates as the subjects (``H-3-small`` trained
with price-only loss and ``H-6-small`` trained with the differential
price+Delta loss), optionally loads the large baseline ``H-3`` as a
reference, builds a :class:`BinEvaluator` backed by the shared balanced
Heston test set, hands the inputs to :class:`DMLStudy`, and writes the
per-bin long-format CSV plus the price/Delta heatmaps that
``docs/tasks.md`` §"Fase 3" mandates as the deliverable.

Typical invocation::

    python scripts/run_experiment_e5.py \\
        --small-price-checkpoint results/checkpoints/H-3-small \\
        --small-dml-checkpoint   results/checkpoints/H-6-small \\
        --baseline-checkpoint    results/checkpoints/H-3 \\
        --test                   data/heston_test_125k_balanced_delta.npz \\
        --output                 results/metrics/e5_table.csv \\
        --figures-dir            results/figures/e5

``--baseline-checkpoint`` is optional; without it the experiment still
emits its pre-registered verdict (which only depends on ``H-3-small``
and ``H-6-small``) but skips the "distance vs baseline" diagnostic
lines of the summary.
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
from src.experiments import DMLStudy, ExperimentResult, SurrogateInput
from src.models import MLP
from src.solvers import HestonSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run E5 (DML: H-3-small vs H-6-small, optionally with H-3 as "
            "baseline) and write per-bin CSV plus price/Delta heatmaps."
        )
    )
    parser.add_argument(
        "--small-price-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint of the small price-only surrogate (e.g. H-3-small).",
    )
    parser.add_argument(
        "--small-dml-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint of the small DML (price+Delta) surrogate (e.g. H-6-small).",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint of the large baseline surrogate (e.g. H-3). "
            "If provided, the summary reports the distance from the DML "
            "surrogate to this baseline as a diagnostic."
        ),
    )
    parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help=(
            "Heston test set with Delta. Shared across all surrogates per "
            "tasks.md §E5."
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
        help="Optional directory for price/Delta heatmap PNGs.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
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
    if "deltas" not in data.files:
        raise ValueError(
            "test set lacks the `deltas` array; E5 needs it as ground truth. "
            "Regenerate with `--include-delta` if necessary."
        )
    deltas = np.asarray(data["deltas"], dtype=np.float32).reshape(-1, 1)
    input_names = tuple(str(name) for name in np.asarray(data["input_names"]).tolist())
    bin_id: np.ndarray | None = None
    if "bin_id" in data.files:
        bin_id = np.asarray(data["bin_id"], dtype=np.int64)
    return (
        OptionDataset(
            features=torch.from_numpy(features),
            prices=torch.from_numpy(prices),
            deltas=torch.from_numpy(deltas),
            raw_inputs=torch.from_numpy(raw_inputs),
            input_names=input_names,
        ),
        bin_id,
    )


def _build_input(
    *,
    checkpoint_dir: Path,
    role: str,
    dataset: OptionDataset,
    bin_id: np.ndarray | None,
    evaluator: BinEvaluator,
) -> SurrogateInput:
    model, config = load_checkpoint(checkpoint_dir)
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
    dataset, bin_id = load_test_dataset(test_path)
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
    args = parse_args()
    device = resolve_device(args.device)
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
