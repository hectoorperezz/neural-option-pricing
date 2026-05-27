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


def _write_bs_checkpoint(checkpoint_dir: Path, input_dim: int = 4) -> None:
    """Create a checkpoint.pt and config.json mirroring scripts/train_surrogate.py."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_id": "test",
        "input_dim": input_dim,
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
        "input_names": (
            ["moneyness", "maturity", "rate", "volatility"]
            if input_dim == 4
            else ["moneyness", "maturity", "rate", "v0", "theta", "kappa", "xi", "rho"]
        ),
    }
    model = MLP(
        input_dim=input_dim,
        hidden_width=config["hidden_width"],
        hidden_layers=config["hidden_layers"],
        activation=config["activation"],
    )
    (checkpoint_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    torch.save(
        {
            "experiment_id": "test",
            "model_state_dict": model.state_dict(),
            "best_state_dict": model.state_dict(),
            "best_validation_price_mae": 0.001,
            "config": config,
            "history": [],
        },
        checkpoint_dir / "checkpoint.pt",
    )


def _write_bs_test_npz(
    path: Path, n_samples: int = 32, include_delta: bool = True, include_bin_id: bool = False
) -> None:
    """Build a small in-domain BS test set with reference prices."""
    rng = np.random.default_rng(123)
    moneyness = rng.uniform(0.5, 1.8, size=n_samples)
    maturity = rng.uniform(7.0 / 365.0, 1.5, size=n_samples)
    rate = rng.uniform(0.0, 0.07, size=n_samples)
    sigma = rng.uniform(0.05, 0.6, size=n_samples)

    bs = BlackScholesSolver()
    prices = np.asarray(
        bs.call_price(
            spot=moneyness,
            strike=1.0,
            maturity=maturity,
            rate=rate,
            volatility=sigma,
            dividend_yield=0.0,
        ),
        dtype=np.float32,
    )

    raw_inputs = np.stack([moneyness, maturity, rate, sigma], axis=1).astype(np.float32)
    features = raw_inputs.copy()
    features[:, 0] = (features[:, 0] - 0.4) / (2.0 - 0.4)
    features[:, 1] = (features[:, 1] - 7.0 / 365.0) / (2.0 - 7.0 / 365.0)
    features[:, 2] = features[:, 2] / 0.075
    features[:, 3] = (features[:, 3] - 0.03) / (1.0 - 0.03)

    payload: dict[str, np.ndarray] = {
        "features": features,
        "raw_inputs": raw_inputs,
        "prices": prices,
        "input_names": np.asarray(["moneyness", "maturity", "rate", "volatility"]),
    }
    if include_delta:
        deltas = np.asarray(
            bs.call_delta(
                spot=moneyness,
                strike=1.0,
                maturity=maturity,
                rate=rate,
                volatility=sigma,
                dividend_yield=0.0,
            ),
            dtype=np.float32,
        )
        payload["deltas"] = deltas
    if include_bin_id:
        payload["bin_id"] = np.zeros(n_samples, dtype=np.int64)

    np.savez(path, **payload)


def _run_script(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["scripts/evaluate_surrogate.py", *args])
    runpy.run_path("scripts/evaluate_surrogate.py", run_name="__main__")


def test_script_writes_csv_with_25_bins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    test_npz = tmp_path / "test.npz"
    output_csv = tmp_path / "report.csv"
    _write_bs_checkpoint(checkpoint_dir)
    _write_bs_test_npz(test_npz, n_samples=64)

    _run_script(
        monkeypatch,
        [
            "--checkpoint", str(checkpoint_dir),
            "--test", str(test_npz),
            "--output", str(output_csv),
            "--device", "cpu",
            "--batch-size", "32",
            "--no-iv",
        ],
    )

    assert output_csv.exists()
    rows = list(csv.reader(output_csv.open(encoding="utf-8")))
    assert len(rows) == 26  # 1 header + 25 bins
    header = rows[0]
    assert "price_mae_mean" in header
    assert "delta_mae_mean" in header
    assert "iv_mae_mean" in header


def test_script_with_no_iv_leaves_iv_columns_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    test_npz = tmp_path / "test.npz"
    output_csv = tmp_path / "report.csv"
    _write_bs_checkpoint(checkpoint_dir)
    _write_bs_test_npz(test_npz, n_samples=64)

    _run_script(
        monkeypatch,
        [
            "--checkpoint", str(checkpoint_dir),
            "--test", str(test_npz),
            "--output", str(output_csv),
            "--device", "cpu",
            "--no-iv",
        ],
    )

    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    assert all(row["iv_mae_mean"] == "" for row in rows)
    assert all(row["iv_failure_rate"] == "" for row in rows)


def test_script_respects_bin_id_from_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    test_npz = tmp_path / "test.npz"
    output_csv = tmp_path / "report.csv"
    _write_bs_checkpoint(checkpoint_dir)
    _write_bs_test_npz(test_npz, n_samples=32, include_bin_id=True)

    _run_script(
        monkeypatch,
        [
            "--checkpoint", str(checkpoint_dir),
            "--test", str(test_npz),
            "--output", str(output_csv),
            "--device", "cpu",
            "--no-iv",
        ],
    )

    rows = list(csv.DictReader(output_csv.open(encoding="utf-8")))
    # All points were assigned bin_id=0 in the npz; only the first bin should
    # have non-zero count, the rest must be empty.
    assert int(rows[0]["n_points"]) == 32
    for row in rows[1:]:
        assert int(row["n_points"]) == 0


def test_script_rejects_non_csv_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    test_npz = tmp_path / "test.npz"
    _write_bs_checkpoint(checkpoint_dir)
    _write_bs_test_npz(test_npz)

    with pytest.raises(ValueError, match="csv"):
        _run_script(
            monkeypatch,
            [
                "--checkpoint", str(checkpoint_dir),
                "--test", str(test_npz),
                "--output", str(tmp_path / "report.txt"),
                "--device", "cpu",
                "--no-iv",
            ],
        )


def test_script_rejects_missing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_npz = tmp_path / "test.npz"
    _write_bs_test_npz(test_npz)

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        _run_script(
            monkeypatch,
            [
                "--checkpoint", str(tmp_path / "does_not_exist"),
                "--test", str(test_npz),
                "--output", str(tmp_path / "report.csv"),
                "--device", "cpu",
                "--no-iv",
            ],
        )


def test_script_creates_missing_output_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    test_npz = tmp_path / "test.npz"
    output_csv = tmp_path / "nested" / "deep" / "report.csv"
    _write_bs_checkpoint(checkpoint_dir)
    _write_bs_test_npz(test_npz)

    _run_script(
        monkeypatch,
        [
            "--checkpoint", str(checkpoint_dir),
            "--test", str(test_npz),
            "--output", str(output_csv),
            "--device", "cpu",
            "--no-iv",
        ],
    )

    assert output_csv.exists()
