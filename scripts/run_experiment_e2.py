"""Run E2 (Activations and Delta quality) on the documented surrogates.

Orchestration only. The script loads BS-1..4 and/or H-1..4 (the
activations slate that ``docs/tasks.md`` §E2 designates as the subjects),
constructs a :class:`BinEvaluator` per family, builds
:class:`SurrogateInput` instances labelled with each checkpoint's
activation (read from ``config.json``), hands them to
:class:`ActivationStudy`, and writes the per-bin long-format CSV plus the
price and Delta heatmaps that ``docs/tasks.md`` §"Fase 2" mandates as the
deliverable.

Typical invocation (both families)::

    python scripts/run_experiment_e2.py \\
        --bs-checkpoints   results/checkpoints/BS-1 results/checkpoints/BS-2 \\
                           results/checkpoints/BS-3 results/checkpoints/BS-4 \\
        --bs-test          data/bs_test_125k_balanced_delta.npz \\
        --heston-checkpoints results/checkpoints/H-1 results/checkpoints/H-2 \\
                             results/checkpoints/H-3 results/checkpoints/H-4 \\
        --heston-test      data/heston_test_125k_balanced_delta.npz \\
        --output-dir       results/metrics \\
        --figures-dir      results/figures/e2

Either family can be omitted; the script processes whichever side
provides both ``--<family>-checkpoints`` and ``--<family>-test``.
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
from src.experiments import ActivationStudy, ExperimentResult, SurrogateInput
from src.models import MLP
from src.solvers import BlackScholesSolver, HestonSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run E2 on the activation slate (BS-1..4 and/or H-1..4) and "
            "write per-bin CSV plus price/Delta heatmaps."
        )
    )
    parser.add_argument(
        "--bs-checkpoints",
        type=Path,
        nargs="+",
        default=None,
        help="One or more BS checkpoint directories.",
    )
    parser.add_argument("--bs-test", type=Path, default=None)
    parser.add_argument(
        "--heston-checkpoints",
        type=Path,
        nargs="+",
        default=None,
        help="One or more Heston checkpoint directories.",
    )
    parser.add_argument("--heston-test", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the per-family CSVs (e2_bs.csv, e2_heston.csv).",
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


def _build_family_inputs(
    checkpoint_dirs: list[Path],
    test_path: Path,
    pricer: Any,
    device: str,
    batch_size: int,
) -> list[SurrogateInput]:
    dataset, bin_id = load_test_dataset(test_path)
    evaluator = BinEvaluator(
        partition=BinPartition.default(),
        pricer=pricer,
        device=device,
        batch_size=batch_size,
    )
    inputs: list[SurrogateInput] = []
    for checkpoint_dir in checkpoint_dirs:
        model, config = load_checkpoint(checkpoint_dir)
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

    device = resolve_device(args.device)
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
