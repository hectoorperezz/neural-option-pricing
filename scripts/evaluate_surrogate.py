"""Evaluate a single trained surrogate over a test set and write a CSV.

Orchestration only (per ``docs/architecture.md`` §Principios: "los scripts
solo orquestan llamadas sin contener lógica reutilizable"). Every numeric
or structural decision delegates to:

* :class:`src.models.MLP` to rebuild the network from its training config
* :class:`src.evaluation.BinPartition` for the 5x5 grid
* :class:`src.evaluation.BinEvaluator` for the full evaluation pipeline
* :class:`src.evaluation.Report` for CSV serialization

Typical invocation::

    python scripts/evaluate_surrogate.py \\
        --checkpoint results/checkpoints/BS-3 \\
        --test       data/bs_test_6250k_balanced_delta.npz \\
        --output     results/metrics/BS-3_eval.csv

Pass ``--no-iv`` to skip the BS implied-volatility inversion (orders of
magnitude faster, since the inverter is scalar and runs once per test
point). The methodology document allows IV to be reported as a separate
metric; we never omit it silently — when ``--no-iv`` is used the CSV
simply leaves the IV columns blank.
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
from src.evaluation import BinEvaluator, BinPartition, Report
from src.models import MLP
from src.solvers import BlackScholesSolver, HestonSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained surrogate over a test set, write per-bin CSV."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Directory containing checkpoint.pt and config.json "
             "(as written by scripts/train_surrogate.py).",
    )
    parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help="Path to the test .npz file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path. Must end with .csv.",
    )
    parser.add_argument(
        "--solver",
        choices=("auto", "black_scholes", "heston"),
        default="auto",
        help="Reference pricer family. 'auto' infers from input_names length "
             "(4 -> Black-Scholes, 8 -> Heston).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or any torch device.",
    )
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument(
        "--no-iv",
        action="store_true",
        help="Skip implied volatility inversion. The CSV will have blank IV columns.",
    )
    parser.add_argument("--moneyness-min", type=float, default=0.4)
    parser.add_argument("--moneyness-max", type=float, default=2.0)
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


def load_test_npz(path: Path) -> tuple[OptionDataset, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(f"--test not found: {path}")

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

    dataset = OptionDataset(
        features=torch.from_numpy(features),
        prices=torch.from_numpy(prices),
        deltas=None if deltas is None else torch.from_numpy(deltas),
        raw_inputs=torch.from_numpy(raw_inputs),
        input_names=input_names,
    )
    return dataset, bin_id


def resolve_solver(
    requested: str,
    input_names: tuple[str, ...],
) -> BlackScholesSolver | HestonSolver:
    if requested == "black_scholes":
        return BlackScholesSolver()
    if requested == "heston":
        return HestonSolver()
    # auto: infer from input dimensionality
    if len(input_names) == 4:
        return BlackScholesSolver()
    if len(input_names) == 8:
        return HestonSolver()
    raise ValueError(
        f"cannot auto-detect solver from {len(input_names)} input names: {input_names}. "
        "Pass --solver black_scholes or --solver heston explicitly."
    )


def _top_n_finite(arr: np.ndarray, n: int) -> np.ndarray:
    finite_indices = np.where(np.isfinite(arr))[0]
    if finite_indices.size == 0:
        return np.array([], dtype=np.int64)
    order = finite_indices[np.argsort(-arr[finite_indices])]
    return order[:n]


def print_summary(report: Report, elapsed: float) -> None:
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

    device = resolve_device(args.device)
    model, _ = load_checkpoint(args.checkpoint)
    dataset, bin_id = load_test_npz(args.test)
    pricer = resolve_solver(args.solver, dataset.input_names)

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
