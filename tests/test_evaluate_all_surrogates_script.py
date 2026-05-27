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


def _write_bs_checkpoint(parent_dir: Path, name: str, activation: str = "swish") -> Path:
    checkpoint_dir = parent_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment_id": name,
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
            "experiment_id": name,
            "model_state_dict": model.state_dict(),
            "best_state_dict": model.state_dict(),
            "best_validation_price_mae": 0.001,
            "config": config,
            "history": [],
        },
        checkpoint_dir / "checkpoint.pt",
    )
    return checkpoint_dir


def _write_bs_test_npz(path: Path, n_samples: int = 32) -> None:
    rng = np.random.default_rng(123)
    moneyness = rng.uniform(0.5, 1.8, size=n_samples)
    maturity = rng.uniform(7.0 / 365.0, 1.5, size=n_samples)
    rate = rng.uniform(0.0, 0.07, size=n_samples)
    sigma = rng.uniform(0.05, 0.6, size=n_samples)

    bs = BlackScholesSolver()
    prices = np.asarray(
        bs.call_price(
            spot=moneyness, strike=1.0, maturity=maturity, rate=rate,
            volatility=sigma, dividend_yield=0.0,
        ),
        dtype=np.float32,
    )
    deltas = np.asarray(
        bs.call_delta(
            spot=moneyness, strike=1.0, maturity=maturity, rate=rate,
            volatility=sigma, dividend_yield=0.0,
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
    monkeypatch.setattr(sys, "argv", ["scripts/evaluate_all_surrogates.py", *args])
    runpy.run_path("scripts/evaluate_all_surrogates.py", run_name="__main__")


def test_runs_all_checkpoints_and_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints_dir = tmp_path / "ckpts"
    output_dir = tmp_path / "metrics"
    bs_test = tmp_path / "bs_test.npz"
    _write_bs_checkpoint(checkpoints_dir, "BS-1", activation="relu")
    _write_bs_checkpoint(checkpoints_dir, "BS-3", activation="swish")
    _write_bs_test_npz(bs_test)

    _run_script(
        monkeypatch,
        [
            "--checkpoints-dir", str(checkpoints_dir),
            "--bs-test", str(bs_test),
            "--heston-test", str(tmp_path / "heston_test.npz"),
            "--output-dir", str(output_dir),
            "--device", "cpu",
            "--batch-size", "32",
            "--no-iv",
        ],
    )

    assert (output_dir / "BS-1_eval.csv").exists()
    assert (output_dir / "BS-3_eval.csv").exists()
    summary_csv = output_dir / "all_surrogates_summary.csv"
    assert summary_csv.exists()

    rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    assert len(rows) == 2
    surrogate_ids = sorted(row["surrogate_id"] for row in rows)
    assert surrogate_ids == ["BS-1", "BS-3"]


def test_summary_carries_expected_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints_dir = tmp_path / "ckpts"
    output_dir = tmp_path / "metrics"
    bs_test = tmp_path / "bs_test.npz"
    _write_bs_checkpoint(checkpoints_dir, "BS-3")
    _write_bs_test_npz(bs_test)

    _run_script(
        monkeypatch,
        [
            "--checkpoints-dir", str(checkpoints_dir),
            "--bs-test", str(bs_test),
            "--heston-test", str(tmp_path / "heston_test.npz"),
            "--output-dir", str(output_dir),
            "--device", "cpu",
            "--batch-size", "32",
            "--no-iv",
        ],
    )

    summary_csv = output_dir / "all_surrogates_summary.csv"
    rows = list(csv.DictReader(summary_csv.open(encoding="utf-8")))
    expected = {
        "surrogate_id",
        "test_path",
        "n_points",
        "price_mae_mean",
        "price_mae_p95_max",
        "price_worst_bin",
        "delta_mae_mean",
        "delta_mae_p95_max",
        "delta_worst_bin",
        "iv_mae_mean",
        "iv_mae_p95_max",
        "iv_worst_bin",
        "iv_failure_rate_mean",
        "iv_failure_rate_max",
    }
    assert expected.issubset(set(rows[0].keys()))


def test_empty_checkpoints_dir_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints_dir = tmp_path / "ckpts"
    checkpoints_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="no checkpoints"):
        _run_script(
            monkeypatch,
            [
                "--checkpoints-dir", str(checkpoints_dir),
                "--bs-test", str(tmp_path / "bs_test.npz"),
                "--heston-test", str(tmp_path / "heston_test.npz"),
                "--output-dir", str(tmp_path / "metrics"),
                "--device", "cpu",
                "--no-iv",
            ],
        )


def test_missing_required_test_set_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints_dir = tmp_path / "ckpts"
    _write_bs_checkpoint(checkpoints_dir, "BS-1")

    with pytest.raises(FileNotFoundError, match="black_scholes"):
        _run_script(
            monkeypatch,
            [
                "--checkpoints-dir", str(checkpoints_dir),
                "--bs-test", str(tmp_path / "does_not_exist.npz"),
                "--heston-test", str(tmp_path / "heston_test.npz"),
                "--output-dir", str(tmp_path / "metrics"),
                "--device", "cpu",
                "--no-iv",
            ],
        )


def test_subprocess_failure_stops_with_non_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a checkpoint is malformed, the script exits non-zero."""
    checkpoints_dir = tmp_path / "ckpts"
    bs_test = tmp_path / "bs_test.npz"

    # Build a "broken" checkpoint: config.json present but checkpoint.pt is garbage
    bad = checkpoints_dir / "BAD"
    bad.mkdir(parents=True)
    (bad / "config.json").write_text(
        json.dumps(
            {
                "input_dim": 4,
                "hidden_width": 8,
                "hidden_layers": 1,
                "activation": "swish",
            }
        ),
        encoding="utf-8",
    )
    (bad / "checkpoint.pt").write_bytes(b"not a torch checkpoint")
    _write_bs_test_npz(bs_test)

    with pytest.raises(RuntimeError, match="evaluate_surrogate.py failed"):
        _run_script(
            monkeypatch,
            [
                "--checkpoints-dir", str(checkpoints_dir),
                "--bs-test", str(bs_test),
                "--heston-test", str(tmp_path / "heston_test.npz"),
                "--output-dir", str(tmp_path / "metrics"),
                "--device", "cpu",
                "--no-iv",
            ],
        )
