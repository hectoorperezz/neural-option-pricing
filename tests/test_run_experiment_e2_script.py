"""Test extremo a extremo de ``scripts/experiments/run_experiment_e2.py``.

Monta cuatro checkpoints (uno por activación) y comprueba que el
script genera ``e2_bs.csv``/``e2_heston.csv`` con la columna de
activación propagada desde ``config.json``.
"""

import csv
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import MLP
from src.solvers import BlackScholesSolver


def _write_bs_checkpoint(checkpoint_dir: Path, activation: str) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_id": checkpoint_dir.name,
        "input_dim": 4,
        "hidden_width": 8,
        "hidden_layers": 1,
        "activation": activation,
        "loss": "price",
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.001,
        "seed": 0,
        "device": "cpu",
        "num_workers": 0,
        "input_names": ["moneyness", "maturity", "rate", "volatility"],
    }
    model = MLP(input_dim=4, hidden_width=8, hidden_layers=1, activation=activation)
    (checkpoint_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    torch.save(
        {
            "experiment_id": checkpoint_dir.name,
            "model_state_dict": model.state_dict(),
            "best_state_dict": model.state_dict(),
            "best_validation_price_mae": 0.001,
            "config": config,
            "history": [],
        },
        checkpoint_dir / "checkpoint.pt",
    )


def _write_bs_test_npz(path: Path, n_samples: int = 60) -> None:
    rng = np.random.default_rng(123)
    moneyness = rng.uniform(0.6, 1.6, size=n_samples)
    maturity = rng.uniform(0.1, 1.5, size=n_samples)
    rate = rng.uniform(0.0, 0.05, size=n_samples)
    sigma = rng.uniform(0.05, 0.5, size=n_samples)
    bs = BlackScholesSolver()
    prices = np.asarray(
        bs.call_price(
            spot=moneyness, strike=1.0, maturity=maturity,
            rate=rate, volatility=sigma, dividend_yield=0.0,
        ),
        dtype=np.float32,
    )
    deltas = np.asarray(
        bs.call_delta(
            spot=moneyness, strike=1.0, maturity=maturity,
            rate=rate, volatility=sigma, dividend_yield=0.0,
        ),
        dtype=np.float32,
    )
    raw_inputs = np.stack([moneyness, maturity, rate, sigma], axis=1).astype(np.float32)
    features = raw_inputs.copy()
    features[:, 0] = (features[:, 0] - 0.4) / (2.0 - 0.4)
    features[:, 1] = (features[:, 1] - 7.0 / 365.0) / (2.0 - 7.0 / 365.0)
    features[:, 2] = features[:, 2] / 0.075
    features[:, 3] = (features[:, 3] - 0.03) / (1.0 - 0.03)
    np.savez(
        path,
        features=features,
        raw_inputs=raw_inputs,
        prices=prices,
        deltas=deltas,
        input_names=np.asarray(["moneyness", "maturity", "rate", "volatility"]),
    )


def _run_script(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["scripts/experiments/run_experiment_e2.py", *args])
    runpy.run_path("scripts/experiments/run_experiment_e2.py", run_name="__main__")


def test_script_writes_csv_for_bs_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E2 sobre BS produce 50 filas (2 surrogates × 25 bins) en el CSV."""
    bs_dir = tmp_path / "ckpts"
    _write_bs_checkpoint(bs_dir / "BS-1", activation="relu")
    _write_bs_checkpoint(bs_dir / "BS-3", activation="swish")
    bs_test = tmp_path / "bs_test.npz"
    _write_bs_test_npz(bs_test)
    output_dir = tmp_path / "metrics"

    _run_script(
        monkeypatch,
        [
            "--bs-checkpoints", str(bs_dir / "BS-1"), str(bs_dir / "BS-3"),
            "--bs-test", str(bs_test),
            "--output-dir", str(output_dir),
            "--device", "cpu",
            "--batch-size", "32",
        ],
    )

    csv_path = output_dir / "e2_bs.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 50  # 2 surrogates x 25 bins
    activations = {row["activation"] for row in rows}
    assert activations == {"relu", "swish"}

