import json
import runpy
import sys

import numpy as np
import torch


def write_npz(path, n_samples: int = 32, include_delta: bool = True) -> None:
    rng = np.random.default_rng(123)
    features = rng.uniform(0.0, 1.0, size=(n_samples, 4)).astype(np.float32)
    raw_inputs = features.copy()
    prices = (0.5 * features[:, 0] + 0.25 * features[:, 1]).astype(np.float32)
    payload = {
        "features": features,
        "raw_inputs": raw_inputs,
        "prices": prices,
        "input_names": np.asarray(["moneyness", "maturity", "rate", "volatility"]),
    }
    if include_delta:
        payload["deltas"] = np.full(n_samples, 0.5 / 1.6, dtype=np.float32)
    np.savez(path, **payload)


def run_train_script(monkeypatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["scripts/train_surrogate.py", *args])
    runpy.run_path("scripts/train_surrogate.py", run_name="__main__")


def test_train_surrogate_script_writes_checkpoint_and_history(tmp_path, monkeypatch) -> None:
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    output_dir = tmp_path / "run"
    write_npz(train, n_samples=32)
    write_npz(validation, n_samples=16)

    run_train_script(
        monkeypatch,
        [
            "--train",
            str(train),
            "--validation",
            str(validation),
            "--output-dir",
            str(output_dir),
            "--experiment-id",
            "test-price",
            "--loss",
            "price",
            "--activation",
            "swish",
            "--hidden-width",
            "8",
            "--hidden-layers",
            "1",
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--learning-rate",
            "0.01",
        ],
    )

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu")
    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))

    assert checkpoint["experiment_id"] == "test-price"
    assert checkpoint["best_validation_price_mae"] < float("inf")
    assert len(history) == 2
    assert config["activation"] == "swish"
    assert (output_dir / "history.csv").exists()


def test_train_surrogate_script_supports_differential_loss(tmp_path, monkeypatch) -> None:
    train = tmp_path / "train_delta.npz"
    validation = tmp_path / "validation.npz"
    output_dir = tmp_path / "run_delta"
    write_npz(train, n_samples=24, include_delta=True)
    write_npz(validation, n_samples=12, include_delta=False)

    run_train_script(
        monkeypatch,
        [
            "--train",
            str(train),
            "--validation",
            str(validation),
            "--output-dir",
            str(output_dir),
            "--experiment-id",
            "test-dml",
            "--loss",
            "differential",
            "--hidden-width",
            "8",
            "--hidden-layers",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "8",
        ],
    )

    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location="cpu")
    assert checkpoint["config"]["loss"] == "differential"
