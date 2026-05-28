"""Run E4 (Efficiency study) on the documented Heston surrogate.

Orchestration only. The script loads ``H-3`` (the surrogate that
``docs/tasks.md`` §E4 designates as the subject of the timing
benchmark), builds a :class:`TimingBenchmark` backed by the shared
balanced test set, runs :class:`EfficiencyStudy` on CPU and (if
available) on CUDA, writes the per-(device, batch_size) CSV and the
``speedup`` vs ``batch_size`` line plot that ``docs/tasks.md`` §"Fase
4" mandates as the deliverable.

Typical invocation::

    python scripts/run_experiment_e4.py \\
        --checkpoint  results/checkpoints/H-3 \\
        --test        data/heston_test_125k_balanced_delta.npz \\
        --output      results/metrics/e4_table.csv \\
        --plot        results/figures/e4/speedup_vs_batch.png

By default the script runs on CPU plus CUDA if a device is visible.
Pass ``--devices cpu`` to force CPU-only, or ``--devices cpu cuda`` to
force both.
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

from src.evaluation import TimingBenchmark
from src.experiments import EfficiencyResult, EfficiencyStudy
from src.models import MLP
from src.solvers import HestonSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run E4 (timing benchmark of H-3 vs Heston solver) and write "
            "per-(device, batch_size) CSV plus a speedup-vs-batch PNG."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory for the Heston surrogate to benchmark "
        "(default subject: H-3).",
    )
    parser.add_argument(
        "--test",
        type=Path,
        required=True,
        help=(
            "Heston test set whose first N points feed each batch "
            "(default: data/heston_test_125k_balanced_delta.npz)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV (one row per (device, batch_size) pair).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional destination PNG for the speedup-vs-batch line plot.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help=(
            "Devices to benchmark. Default: 'cpu' plus 'cuda' if available. "
            "Pass e.g. '--devices cpu' to force CPU-only."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Override the protocol batch sizes. Default: 100 1000 10000 100000 "
            "(as pre-registered in tasks.md §E4)."
        ),
    )
    parser.add_argument(
        "--n-warmups",
        type=int,
        default=None,
        help="Override warmup count. Default: 3 (as pre-registered).",
    )
    parser.add_argument(
        "--n-repetitions",
        type=int,
        default=None,
        help="Override repetition count. Default: 10 (as pre-registered).",
    )
    return parser.parse_args()


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


def load_test_dataset(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not path.exists():
        raise FileNotFoundError(f"test set not found at {path}")
    data = np.load(path, allow_pickle=False)
    features = np.asarray(data["features"], dtype=np.float32)
    raw_inputs = np.asarray(data["raw_inputs"], dtype=np.float32)
    input_names = tuple(str(name) for name in np.asarray(data["input_names"]).tolist())
    return features, raw_inputs, input_names


def resolve_devices(requested: list[str] | None) -> tuple[str, ...]:
    if requested is not None:
        return tuple(requested)
    if torch.cuda.is_available():
        return ("cpu", "cuda")
    return ("cpu",)


def main() -> EfficiencyResult:
    args = parse_args()
    devices = resolve_devices(args.devices)

    print("E4 — Eficiencia computacional (H-3 vs Heston solver)")
    print(f"checkpoint   : {args.checkpoint}")
    print(f"test         : {args.test}")
    print(f"output       : {args.output}")
    if args.plot is not None:
        print(f"plot         : {args.plot}")
    print(f"devices      : {', '.join(devices)}")

    features, raw_inputs, input_names = load_test_dataset(args.test)
    surrogate, _config = load_checkpoint(args.checkpoint)

    benchmark_kwargs: dict[str, Any] = {
        "pricer": HestonSolver(),
        "raw_inputs": raw_inputs,
        "features": features,
        "input_names": input_names,
    }
    if args.batch_sizes is not None:
        benchmark_kwargs["batch_sizes"] = tuple(args.batch_sizes)
    if args.n_warmups is not None:
        benchmark_kwargs["n_warmups"] = args.n_warmups
    if args.n_repetitions is not None:
        benchmark_kwargs["n_repetitions"] = args.n_repetitions
    benchmark = TimingBenchmark(**benchmark_kwargs)

    print(
        "protocol     : batch_sizes={}, n_warmups={}, n_repetitions={}".format(
            benchmark.batch_sizes, benchmark.n_warmups, benchmark.n_repetitions
        )
    )

    study = EfficiencyStudy(
        benchmark=benchmark,
        surrogate=surrogate,
        surrogate_id=args.checkpoint.name,
        devices=devices,
    )

    started_at = time.perf_counter()
    result = study.run()
    elapsed = time.perf_counter() - started_at

    result.to_csv(args.output)
    print(f"\nCSV written: {args.output}  (elapsed {elapsed:.2f}s)")
    if args.plot is not None:
        result.to_plot(args.plot)
        print(f"Plot written: {args.plot}")

    print(result.summary)
    return result


if __name__ == "__main__":
    main()
