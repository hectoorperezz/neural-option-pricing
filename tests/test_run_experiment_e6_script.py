"""Test extremo a extremo de ``scripts/experiments/run_experiment_e6.py``.

Monta cuatro checkpoints sintéticos (con `hidden_layers` y
`scheduler` distintos) sobre un test Heston con Delta y comprueba que
el script emite el CSV largo con las columnas de ``role`` y
``architecture``.
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


def _write_heston_like_checkpoint(
    checkpoint_dir: Path,
    *,
    hidden_layers: int = 4,
    scheduler: str = "none",
) -> None:
    """Write a small MLP checkpoint that mimics the Heston surrogate layout.

    The script only reads ``config.json`` and ``checkpoint.pt`` to label
    the surrogate by architecture (hidden_layers + scheduler), so a tiny
    model is enough.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_id": checkpoint_dir.name,
        "input_dim": 4,
        "hidden_width": 8,
        "hidden_layers": hidden_layers,
        "activation": "swish",
        "loss": "price",
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.001,
        "scheduler": scheduler,
        "seed": 0,
        "device": "cpu",
        "num_workers": 0,
        "input_names": ["moneyness", "maturity", "rate", "volatility"],
    }
    model = MLP(
        input_dim=4,
        hidden_width=8,
        hidden_layers=hidden_layers,
        activation="swish",
    )
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


def _write_test_npz_with_deltas(path: Path, n_samples: int = 60) -> None:
    """Write a small BS-priced test set with the column layout the script reads."""
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
    monkeypatch.setattr(
        sys, "argv", ["scripts/experiments/run_experiment_e6.py", *args]
    )
    runpy.run_path(
        "scripts/experiments/run_experiment_e6.py", run_name="__main__"
    )


def test_script_writes_csv_four_surrogates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E6 sobre baseline + shallow + deep + lr_schedule produce el CSV largo con roles y arquitecturas."""
    ckpts = tmp_path / "ckpts"
    _write_heston_like_checkpoint(ckpts / "H-3", hidden_layers=4, scheduler="none")
    _write_heston_like_checkpoint(ckpts / "H-7-shallow", hidden_layers=2, scheduler="none")
    _write_heston_like_checkpoint(ckpts / "H-8-deep", hidden_layers=6, scheduler="none")
    _write_heston_like_checkpoint(
        ckpts / "H-9-lr-schedule", hidden_layers=4, scheduler="plateau"
    )
    test_path = tmp_path / "heston_test.npz"
    _write_test_npz_with_deltas(test_path)
    output_csv = tmp_path / "metrics" / "e6_table.csv"

    _run_script(
        monkeypatch,
        [
            "--baseline-checkpoint", str(ckpts / "H-3"),
            "--shallow-checkpoint", str(ckpts / "H-7-shallow"),
            "--deep-checkpoint", str(ckpts / "H-8-deep"),
            "--scheduler-checkpoint", str(ckpts / "H-9-lr-schedule"),
            "--test", str(test_path),
            "--output", str(output_csv),
            "--device", "cpu",
            "--batch-size", "32",
        ],
    )

    assert output_csv.exists()
    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert len(rows) == 100  # 4 surrogates x 25 bins
    roles = {row["role"] for row in rows}
    assert roles == {"baseline", "shallow", "deep", "lr_schedule"}
    surrogates_present = {row["surrogate_id"] for row in rows}
    assert surrogates_present == {
        "H-3",
        "H-7-shallow",
        "H-8-deep",
        "H-9-lr-schedule",
    }
    architectures = {row["architecture"] for row in rows}
    assert architectures == {"4x8", "2x8", "6x8", "4x8+plateau"}
