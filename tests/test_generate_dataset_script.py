"""Test extremo a extremo de ``scripts/data/generate_dataset.py``.

Invoca el script como subproceso y verifica que escribe ``.npz`` +
``.json`` de metadata con los campos esperados.
"""

import json
import subprocess
import sys

import numpy as np


def test_generate_dataset_script_writes_npz_and_metadata(tmp_path) -> None:
    """Tras un run BS uniforme pequeño, el ``.npz`` y la metadata son consistentes."""
    output = tmp_path / "bs_smoke.npz"

    subprocess.run(
        [
            sys.executable,
            "scripts/data/generate_dataset.py",
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


