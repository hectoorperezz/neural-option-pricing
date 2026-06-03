"""Training script for a pricing surrogate from `.npz` datasets.

Loads train + validation sets, builds an :class:`~src.models.MLP`, trains it
with :class:`~src.training.Trainer`, and saves ``checkpoint.pt``,
``config.json`` and training histories (CSV + JSON) into ``--output-dir``.

Supported losses:
- ``price``
- ``differential``

Supported loading modes:
- classic ``DataLoader``
- ``--preload-to-device`` to keep tensors resident in VRAM
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader


def _torch_compile_available() -> bool:
    """Check whether ``torch.compile`` can be used on this platform.

    Inductor needs Triton. On Windows there is no official Triton wheel, so we
    check the import before paying the compilation cost and failing at runtime.
    """
    if sys.platform == "win32":
        try:
            import importlib

            importlib.import_module("triton")
            return True
        except ImportError:
            return False
    return True


class GPUBatchIterator:
    """Batch iterator for tensors already loaded onto the target ``device``.

    Avoids ``DataLoader`` and host->device transfer at each batch when the full
    dataset fits in VRAM. Yields dictionaries with the same keys as
    ``OptionDataset.__getitem__``.
    """

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        batch_size: int,
        shuffle: bool,
        device: str,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be strictly positive")

        self.tensors = {name: value.to(device) for name, value in tensors.items()}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.n = int(next(iter(self.tensors.values())).shape[0])
        self._generator = torch.Generator(device=device).manual_seed(seed)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.shuffle:
            indices = torch.randperm(self.n, device=self.device, generator=self._generator)
        else:
            indices = torch.arange(self.n, device=self.device)

        for start in range(0, self.n, self.batch_size):
            idx = indices[start : start + self.batch_size]
            yield {name: value[idx] for name, value in self.tensors.items()}

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import MLP
from src.training import DifferentialLoss, PriceLoss, Trainer
from src.utils import load_option_dataset_npz, resolve_torch_device, set_global_seed


def parse_args() -> argparse.Namespace:
    """CLI parser."""
    parser = argparse.ArgumentParser(description="Train a pricing surrogate from NPZ data.")

    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)

    parser.add_argument("--loss", choices=("price", "differential"), default="price")
    parser.add_argument(
        "--activation",
        choices=("relu", "softplus", "swish", "tanh"),
        default="swish",
    )
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    parser.add_argument(
        "--scheduler",
        choices=("none", "plateau"),
        default="none",
        help="Learning-rate scheduler. 'plateau' uses ReduceLROnPlateau on validation_price_mae.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="Multiplicative LR factor when ReduceLROnPlateau triggers.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=3,
        help="Number of epochs without improvement before reducing LR.",
    )
    parser.add_argument(
        "--scheduler-min-lr",
        type=float,
        default=1e-5,
        help="Lower bound for LR when a scheduler is enabled.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda' or any valid Torch device string.",
    )
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--price-weight", type=float, default=1.0)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--moneyness-min", type=float, default=0.4)
    parser.add_argument("--moneyness-max", type=float, default=2.0)

    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Wrap the model with torch.compile() to fuse kernels. "
            "This reduces per-step overhead at the cost of compile time."
        ),
    )
    parser.add_argument(
        "--preload-to-device",
        action="store_true",
        help=(
            "Load train and validation tensors fully onto the target device and "
            "use a native GPU iterator. Avoids per-batch transfers if data fits in VRAM."
        ),
    )

    return parser.parse_args()


def make_loss(args: argparse.Namespace) -> torch.nn.Module:
    """Return ``PriceLoss`` or ``DifferentialLoss`` based on ``--loss``."""
    if args.loss == "price":
        return PriceLoss()

    return DifferentialLoss(
        price_weight=args.price_weight,
        delta_weight=args.delta_weight,
        moneyness_range=(args.moneyness_min, args.moneyness_max),
    )


def write_history(output_dir: Path, history: list[dict[str, float]]) -> None:
    """Persist training history to ``history.json`` and ``history.csv``."""
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if not history:
        return

    fieldnames = list(history[0].keys())
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_config(output_dir: Path, config: dict[str, Any]) -> None:
    """Persist run configuration to ``config.json``."""
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    """Main training entrypoint."""
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be strictly positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be strictly positive")
    if args.hidden_width <= 0 or args.hidden_layers <= 0:
        raise ValueError("--hidden-width and --hidden-layers must be strictly positive")
    if args.scheduler_factor <= 0.0 or args.scheduler_factor >= 1.0:
        raise ValueError("--scheduler-factor must be in the interval (0, 1)")
    if args.scheduler_patience < 0:
        raise ValueError("--scheduler-patience must be non-negative")
    if args.scheduler_min_lr <= 0.0:
        raise ValueError("--scheduler-min-lr must be strictly positive")

    set_global_seed(args.seed)
    device = resolve_torch_device(args.device, require_cuda=True)

    train_dataset, _ = load_option_dataset_npz(
        args.train,
        require_delta=args.loss == "differential",
        dataset_label="train dataset",
    )
    validation_dataset, _ = load_option_dataset_npz(
        args.validation,
        require_delta=args.loss == "differential",
        dataset_label="validation dataset",
    )

    if train_dataset.features.shape[1] != validation_dataset.features.shape[1]:
        raise ValueError("train and validation input dimensions do not match")

    pin_memory = device.startswith("cuda")

    if args.preload_to_device:
        train_tensors: dict[str, torch.Tensor] = {
            "features": train_dataset.features,
            "price": train_dataset.prices,
        }
        if train_dataset.deltas is not None:
            train_tensors["delta"] = train_dataset.deltas

        validation_tensors: dict[str, torch.Tensor] = {
            "features": validation_dataset.features,
            "price": validation_dataset.prices,
        }
        if validation_dataset.deltas is not None:
            validation_tensors["delta"] = validation_dataset.deltas

        train_loader = GPUBatchIterator(
            train_tensors,
            args.batch_size,
            shuffle=True,
            device=device,
            seed=args.seed,
        )
        validation_loader = GPUBatchIterator(
            validation_tensors,
            args.batch_size,
            shuffle=False,
            device=device,
            seed=args.seed + 1,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    model = MLP(
        input_dim=train_dataset.features.shape[1],
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
    )

    if args.compile:
        if _torch_compile_available():
            model = torch.compile(model)
        else:
            print(
                "warning: --compile requested but torch.compile backend is unavailable "
                "on this platform (Triton not installed). Running uncompiled.",
                flush=True,
            )
            args.compile = False

    loss_fn = make_loss(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
        )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment_id": args.experiment_id,
        "train": str(args.train),
        "validation": str(args.validation),
        "loss": args.loss,
        "activation": args.activation,
        "hidden_width": args.hidden_width,
        "hidden_layers": args.hidden_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": device,
        "num_workers": args.num_workers,
        "input_dim": train_dataset.features.shape[1],
        "input_names": train_dataset.input_names,
        "price_weight": args.price_weight,
        "delta_weight": args.delta_weight,
        "moneyness_range": [args.moneyness_min, args.moneyness_max],
        "compile": args.compile,
        "preload_to_device": args.preload_to_device,
        "scheduler": args.scheduler,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_min_lr": args.scheduler_min_lr,
    }
    write_config(args.output_dir, config)

    def on_epoch_end(record: dict[str, float]) -> None:
        if scheduler is not None:
            scheduler.step(record["validation_price_mae"])

        current_lr = optimizer.param_groups[0]["lr"]
        record["learning_rate"] = float(current_lr)

        metrics = " ".join(
            f"{name}={value:.6g}" for name, value in record.items() if name != "epoch"
        )
        print(f"epoch={int(record['epoch'])} {metrics}", flush=True)

    history = trainer.fit(args.epochs, on_epoch_end=on_epoch_end)
    trainer.load_best()
    write_history(args.output_dir, history)

    checkpoint = {
        "experiment_id": args.experiment_id,
        "model_state_dict": trainer.model.state_dict(),
        "best_state_dict": trainer.best_state_dict,
        "best_validation_price_mae": trainer.best_validation_mae,
        "config": config,
        "history": history,
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")

    print(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "best_validation_price_mae": trainer.best_validation_mae,
                "checkpoint": str(args.output_dir / "checkpoint.pt"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()