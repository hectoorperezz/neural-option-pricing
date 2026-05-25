import json
import subprocess
import sys

import numpy as np


def test_generate_dataset_script_writes_npz_and_metadata(tmp_path) -> None:
    output = tmp_path / "bs_smoke.npz"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_dataset.py",
            "--family",
            "black_scholes",
            "--sampler",
            "uniform",
            "--n-samples",
            "32",
            "--batch-size",
            "16",
            "--seed",
            "123",
            "--include-delta",
            "--output",
            str(output),
        ],
        check=True,
    )

    data = np.load(output)
    metadata = json.loads(output.with_suffix(".npz.json").read_text(encoding="utf-8"))

    assert data["features"].shape == (32, 4)
    assert data["raw_inputs"].shape == (32, 4)
    assert data["prices"].shape == (32,)
    assert data["deltas"].shape == (32,)
    assert metadata["accepted_count"] == 32
    assert metadata["family"] == "black_scholes"


def test_generate_dataset_script_writes_balanced_bins(tmp_path) -> None:
    output = tmp_path / "heston_balanced_smoke.npz"

    subprocess.run(
        [
            sys.executable,
            "scripts/generate_dataset.py",
            "--family",
            "heston",
            "--sampler",
            "balanced",
            "--samples-per-bin",
            "2",
            "--batch-size",
            "2",
            "--seed",
            "123",
            "--include-delta",
            "--output",
            str(output),
        ],
        check=True,
    )

    data = np.load(output)
    metadata = json.loads(output.with_suffix(".npz.json").read_text(encoding="utf-8"))

    assert data["features"].shape == (50, 8)
    assert data["raw_inputs"].shape == (50, 8)
    assert data["prices"].shape == (50,)
    assert data["deltas"].shape == (50,)
    assert data["bin_id"].shape == (50,)
    assert np.unique(data["bin_id"], return_counts=True)[1].tolist() == [2] * 25
    assert metadata["sampler"] == "balanced"
    assert metadata["samples_per_bin"] == 2
