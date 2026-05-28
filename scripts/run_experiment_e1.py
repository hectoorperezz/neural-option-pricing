"""Run E1 (Price vs Implied Volatility) on the documented baseline surrogates.

Orchestration only. The script loads BS-3 and/or H-3 (the surrogates that
``docs/tasks.md`` §E1 explicitly designates as the subjects of the
experiment), builds the matching :class:`BinEvaluator` for each family,
hands them to :class:`PriceVsIVStudy`, and writes the per-bin CSV plus
the heatmaps that ``docs/metodologia.md`` §E1 requires::

    "El entregable debe incluir tabla por bin con MAE(C/K), MAE_IV,
    Vega media o proxy de Vega, percentiles altos y tasa de fallos de
    inversión IV, además de heatmaps separados para precio e IV."

Typical invocation (both families)::

    python scripts/run_experiment_e1.py \\
        --bs-checkpoint     results/checkpoints/BS-3 \\
        --bs-test           data/bs_test_125k_balanced_delta.npz \\
        --heston-checkpoint results/checkpoints/H-3 \\
        --heston-test       data/heston_test_125k_balanced_delta.npz \\
        --output            results/metrics/e1_table.csv \\
        --figures-dir       results/figures/e1

Either family can be omitted; the script runs the surrogates that are
provided. Any missing path is reported up front.
"""

from __future__ import annotations

import argparse
import json
import os
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
from src.experiments import PriceVsIVStudy, SurrogateInput
from src.models import MLP
from src.solvers import BlackScholesSolver, HestonSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run E1 on the baseline surrogates (BS-3 and/or H-3) and "
            "write per-bin CSV plus price/IV heatmaps."
        )
    )
    parser.add_argument("--bs-checkpoint", type=Path, default=None)
    parser.add_argument("--bs-test", type=Path, default=None)
    parser.add_argument("--heston-checkpoint", type=Path, default=None)
    parser.add_argument("--heston-test", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path (must end with .csv).")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Optional directory to write price/IV heatmap PNGs into.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
        help=(
            "Number of parallel CPU workers for the BS implied-volatility "
            "inversion (the slow stage of E1). Defaults to all physical "
            "cores minus two."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar shown during IV inversion.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_checkpoint(checkpoint_dir: Path) -> torch.nn.Module:
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
    return model


def load_test_dataset(path: Path) -> tuple[OptionDataset, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(f"--test set not found at {path}")
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
    surrogate_id: str,
    checkpoint_dir: Path,
    test_path: Path,
    pricer: Any,
    device: str,
    batch_size: int,
    iv_workers: int,
    iv_progress: bool,
) -> SurrogateInput:
    model = load_checkpoint(checkpoint_dir)
    dataset, bin_id = load_test_dataset(test_path)
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

    device = resolve_device(args.device)
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
