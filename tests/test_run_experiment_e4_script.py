"""Test extremo a extremo de ``scripts/experiments/run_experiment_e4.py``.

Construye un checkpoint sintético y reduce el protocolo (lotes y
repeticiones) para que el script termine en segundos. Comprueba el
CSV de timing.
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
from src.solvers import BlackScholesSolver, HestonSolver


def _write_heston_checkpoint(checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_id": checkpoint_dir.name,
        "input_dim": 8,
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
        "input_names": [
            "moneyness", "maturity", "rate",
            "v0", "theta", "kappa", "xi", "rho",
        ],
    }
    model = MLP(input_dim=8, hidden_width=8, hidden_layers=1, activation="swish")
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


def _write_heston_test_npz(path: Path, n_samples: int = 120) -> None:
    """Write a tiny BS-priced test set with the Heston raw_inputs layout.

    The E4 script only needs raw_inputs for the solver call and features
    for the surrogate forward. We price with BS for speed and pad the
    Heston-specific columns with realistic values so HestonSolver.call_price
    can also be called (the test_script_runs_with_heston_solver test uses
    it). The Heston solver is slow so we keep n_samples small.
    """
    rng = np.random.default_rng(123)
    moneyness = rng.uniform(0.8, 1.2, size=n_samples)
    maturity = rng.uniform(0.1, 0.5, size=n_samples)
    rate = rng.uniform(0.0, 0.05, size=n_samples)
    v0 = rng.uniform(0.02, 0.1, size=n_samples)
    theta = rng.uniform(0.02, 0.1, size=n_samples)
    kappa = rng.uniform(0.5, 3.0, size=n_samples)
    xi = rng.uniform(0.2, 1.0, size=n_samples)
    rho = rng.uniform(-0.9, -0.1, size=n_samples)
    sigma = np.sqrt(v0)
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
    raw_inputs = np.stack(
        [moneyness, maturity, rate, v0, theta, kappa, xi, rho], axis=1
    ).astype(np.float32)
    features = raw_inputs.copy()
    np.savez(
        path,
        features=features,
        raw_inputs=raw_inputs,
        prices=prices,
        deltas=deltas,
        input_names=np.asarray([
            "moneyness", "maturity", "rate",
            "v0", "theta", "kappa", "xi", "rho",
        ]),
    )


def _run_script(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["scripts/experiments/run_experiment_e4.py", *args])
    runpy.run_path("scripts/experiments/run_experiment_e4.py", run_name="__main__")


def test_script_writes_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E4 con protocolo reducido produce el CSV de timing."""
    ckpt = tmp_path / "ckpts" / "H-3"
    _write_heston_checkpoint(ckpt)
    test_path = tmp_path / "heston_test.npz"
    _write_heston_test_npz(test_path)
    output_csv = tmp_path / "metrics" / "e4_table.csv"

    _run_script(
        monkeypatch,
        [
            "--checkpoint", str(ckpt),
            "--test", str(test_path),
            "--output", str(output_csv),
            "--devices", "cpu",
            "--batch-sizes", "10", "50",
            "--n-warmups", "1",
            "--n-repetitions", "2",
            "--solver-workers", "1",
        ],
    )

    assert output_csv.exists()
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert len(rows) == 2  # 1 device x 2 batch sizes
    assert rows[0]["device"] == "cpu"
    assert rows[0]["surrogate_id"] == "H-3"

