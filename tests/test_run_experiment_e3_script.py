"""Test extremo a extremo de ``scripts/run_experiment_e3.py``.

Monta los dos checkpoints (uniform/focused) sobre un test Heston de
juguete y comprueba el CSV resultante junto con las etiquetas de
sampler.
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


def _write_heston_like_checkpoint(checkpoint_dir: Path) -> None:
    """Write a tiny MLP checkpoint that mimics the Heston surrogate layout.

    The script doesn't care about the family of the checkpoint — it only
    reads ``config.json`` and ``checkpoint.pt`` — so we reuse a 4-input
    Swish MLP. The Heston solver is never invoked because IV inversion is
    against Black-Scholes regardless of the originating solver, and the
    test set we feed below is BS-priced.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_id": checkpoint_dir.name,
        "input_dim": 4,
        "hidden_width": 8,
        "hidden_layers": 1,
        "activation": "swish",
        "loss": "price",
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.001,
        "seed": 0,
        "device": "cpu",
        "num_workers": 0,
        "input_names": ["moneyness", "maturity", "rate", "volatility"],
    }
    model = MLP(input_dim=4, hidden_width=8, hidden_layers=1, activation="swish")
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


def _write_test_npz(path: Path, n_samples: int = 60) -> None:
    """Write a small BS-priced test set with raw_inputs in the Heston shape.

    The Heston solver expects 8 raw inputs, but BinEvaluator only reads
    the first three columns (moneyness, maturity, rate). We pad the rest
    with plausible values so the file structure matches what
    ``scripts/generate_dataset.py`` would produce for Heston.
    """
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
    monkeypatch.setattr(sys, "argv", ["scripts/run_experiment_e3.py", *args])
    runpy.run_path("scripts/run_experiment_e3.py", run_name="__main__")


def test_script_writes_csv_and_heatmaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E3 (uniforme vs focused) produce CSV largo con la columna ``sampler`` y heatmaps."""
    ckpts = tmp_path / "ckpts"
    _write_heston_like_checkpoint(ckpts / "H-3")
    _write_heston_like_checkpoint(ckpts / "H-5")
    test_path = tmp_path / "heston_test.npz"
    _write_test_npz(test_path)
    output_csv = tmp_path / "metrics" / "e3_table.csv"
    figures_dir = tmp_path / "figures"

    _run_script(
        monkeypatch,
        [
            "--uniform-checkpoint", str(ckpts / "H-3"),
            "--focused-checkpoint", str(ckpts / "H-5"),
            "--test", str(test_path),
            "--output", str(output_csv),
            "--figures-dir", str(figures_dir),
            "--device", "cpu",
            "--batch-size", "32",
        ],
    )

    assert output_csv.exists()
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert len(rows) == 50  # 2 surrogates x 25 bins
    samplers = {row["sampler"] for row in rows}
    assert samplers == {"uniform", "focused"}

    surrogates_present = {row["surrogate_id"] for row in rows}
    assert surrogates_present == {"H-3", "H-5"}

    figures = list(figures_dir.glob("*.png"))
    assert len(figures) == 4  # 2 surrogates x 2 metrics (price + iv)

