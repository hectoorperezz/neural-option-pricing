from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from src.datasets.generator import OptionDataset
from src.models import MLP

if TYPE_CHECKING:
    from src.solvers import BlackScholesSolver, HestonSolver


def resolve_torch_device(device: str, *, require_cuda: bool = False) -> str:
    """Resuelve ``auto`` a CPU o CUDA y valida CUDA si se exige."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if require_cuda and device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def load_mlp_checkpoint(checkpoint_dir: Path) -> tuple[nn.Module, dict[str, Any]]:
    """Carga un checkpoint de entrenamiento y reconstruye su MLP."""
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


def load_option_dataset_npz(
    path: Path,
    *,
    require_delta: bool = False,
    dataset_label: str = "test set",
) -> tuple[OptionDataset, np.ndarray | None]:
    """Carga un ``.npz`` de opciones como ``OptionDataset`` y ``bin_id``."""
    if not path.exists():
        raise FileNotFoundError(f"{dataset_label} not found at {path}")

    data = np.load(path, allow_pickle=False)
    features = np.asarray(data["features"], dtype=np.float32)
    raw_inputs = np.asarray(data["raw_inputs"], dtype=np.float32)
    prices = np.asarray(data["prices"], dtype=np.float32).reshape(-1, 1)
    deltas: np.ndarray | None = None
    if "deltas" in data.files:
        deltas = np.asarray(data["deltas"], dtype=np.float32).reshape(-1, 1)
    elif require_delta:
        raise ValueError(
            f"{dataset_label} lacks the `deltas` array; regenerate with --include-delta"
        )

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


def load_npz_features_and_raw_inputs(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Carga solo ``features``, ``raw_inputs`` e ``input_names`` desde NPZ."""
    if not path.exists():
        raise FileNotFoundError(f"test set not found at {path}")

    data = np.load(path, allow_pickle=False)
    features = np.asarray(data["features"], dtype=np.float32)
    raw_inputs = np.asarray(data["raw_inputs"], dtype=np.float32)
    input_names = tuple(str(name) for name in np.asarray(data["input_names"]).tolist())
    return features, raw_inputs, input_names


def resolve_pricer(
    requested: str,
    input_names: tuple[str, ...],
) -> "BlackScholesSolver | HestonSolver":
    """Devuelve el solver de referencia pedido o lo infiere por dimensión."""
    from src.solvers import BlackScholesSolver, HestonSolver

    if requested == "black_scholes":
        return BlackScholesSolver()
    if requested == "heston":
        return HestonSolver()
    if len(input_names) == 4:
        return BlackScholesSolver()
    if len(input_names) == 8:
        return HestonSolver()
    raise ValueError(
        f"cannot auto-detect solver from {len(input_names)} input names: {input_names}. "
        "Pass --solver black_scholes or --solver heston explicitly."
    )
