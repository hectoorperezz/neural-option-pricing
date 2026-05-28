"""Evaluate every checkpoint under a directory and emit a consolidated summary.

This is the orchestrator of the orchestrator: it walks
``results/checkpoints/``, detects each surrogate's family from its
``config.json`` (``input_dim == 4`` -> Black-Scholes, ``input_dim == 8``
-> Heston), routes each checkpoint to its matching test set, and invokes
``scripts/evaluate_surrogate.py`` once per checkpoint via ``subprocess``.

By deferring to ``evaluate_surrogate.py`` the script keeps every numeric
or structural decision in a single place (``docs/architecture.md``
§Principios: "los scripts solo orquestan llamadas sin contener lógica
reutilizable"; ``CLAUDE.md`` lo refuerza). Si Héctor mejora
``evaluate_surrogate.py``, este script hereda la mejora sin un solo
cambio.

Outputs in ``results/metrics/``:

* ``<surrogate>_eval.csv`` — one per surrogate, 25 rows (one per bin),
  produced verbatim by ``evaluate_surrogate.py``.
* ``all_surrogates_summary.csv`` — one row per surrogate with the
  globally-aggregated metrics (mean of per-bin means, worst-bin
  ``p95``, worst-bin label) so the team can scan all 11 at once in
  Excel without opening 11 files.

Sequential by default. A failure in any subprocess stops the whole run
with a clear error; we never emit a partial summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_SURROGATE_SCRIPT = REPO_ROOT / "scripts" / "evaluate_surrogate.py"

DEFAULT_CHECKPOINTS_DIR = REPO_ROOT / "results" / "checkpoints"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "metrics"
DEFAULT_BS_TEST = REPO_ROOT / "data" / "bs_test_125k_balanced_delta.npz"
DEFAULT_HESTON_TEST = REPO_ROOT / "data" / "heston_test_125k_balanced_delta.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every checkpoint under a directory and emit a summary CSV."
    )
    parser.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS_DIR)
    parser.add_argument("--bs-test", type=Path, default=DEFAULT_BS_TEST)
    parser.add_argument("--heston-test", type=Path, default=DEFAULT_HESTON_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument(
        "--no-iv",
        action="store_true",
        help="Forward --no-iv to every evaluation (skips IV inversion, much faster).",
    )
    parser.add_argument(
        "--summary-name",
        default="all_surrogates_summary.csv",
        help="Filename for the consolidated summary CSV (written inside --output-dir).",
    )
    return parser.parse_args()


def find_checkpoints(checkpoints_dir: Path) -> list[Path]:
    """Return every immediate subdir of ``checkpoints_dir`` carrying a checkpoint."""
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"checkpoints directory not found: {checkpoints_dir}")
    candidates = [
        path
        for path in sorted(checkpoints_dir.iterdir())
        if path.is_dir()
        and (path / "checkpoint.pt").exists()
        and (path / "config.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"no checkpoints found under {checkpoints_dir}")
    return candidates


def detect_family(checkpoint_dir: Path) -> str:
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    input_dim = int(config["input_dim"])
    if input_dim == 4:
        return "black_scholes"
    if input_dim == 8:
        return "heston"
    raise ValueError(
        f"unexpected input_dim={input_dim} in {checkpoint_dir / 'config.json'}; "
        "expected 4 for Black-Scholes or 8 for Heston"
    )


def evaluate_one(
    python_exe: str,
    checkpoint_dir: Path,
    test_path: Path,
    output_csv: Path,
    device: str,
    batch_size: int,
    no_iv: bool,
) -> int:
    """Invoke ``evaluate_surrogate.py`` for a single checkpoint."""
    cmd = [
        python_exe,
        str(EVALUATE_SURROGATE_SCRIPT),
        "--checkpoint", str(checkpoint_dir),
        "--test", str(test_path),
        "--output", str(output_csv),
        "--device", device,
        "--batch-size", str(batch_size),
    ]
    if no_iv:
        cmd.append("--no-iv")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def summarize_one(
    surrogate_id: str,
    test_path: Path,
    eval_csv: Path,
) -> dict[str, Any]:
    """Compute the consolidated row for a single surrogate from its per-bin CSV."""
    rows = list(csv.DictReader(eval_csv.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"{eval_csv} is empty")

    labels = [row["bin_label"] for row in rows]
    n_points = sum(int(row["n_points"]) for row in rows)

    summary: dict[str, Any] = {
        "surrogate_id": surrogate_id,
        "test_path": str(test_path),
        "n_points": n_points,
    }
    for metric in ("price", "delta", "iv"):
        means = [_parse_cell(row[f"{metric}_mae_mean"]) for row in rows]
        p95s = [_parse_cell(row[f"{metric}_mae_p95"]) for row in rows]
        worst_value, worst_label = _argmax_finite(means, labels)
        summary[f"{metric}_mae_mean"] = _format(_nan_mean(means))
        summary[f"{metric}_mae_p95_max"] = _format(_nan_max(p95s))
        summary[f"{metric}_worst_bin"] = worst_label

    failures = [_parse_cell(row["iv_failure_rate"]) for row in rows]
    summary["iv_failure_rate_mean"] = _format(_nan_mean(failures))
    summary["iv_failure_rate_max"] = _format(_nan_max(failures))
    return summary


def write_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("nothing to summarize")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_cell(raw: str) -> float:
    if raw == "":
        return math.nan
    return float(raw)


def _nan_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def _nan_max(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    return max(finite)


def _argmax_finite(values: list[float], labels: list[str]) -> tuple[float, str]:
    best_value = -math.inf
    best_label = ""
    for value, label in zip(values, labels):
        if math.isfinite(value) and value > best_value:
            best_value = value
            best_label = label
    if best_value == -math.inf:
        return math.nan, ""
    return best_value, best_label


def _format(value: float) -> Any:
    if math.isfinite(value):
        return value
    return ""


def _resolve_test_path(family: str, bs_test: Path, heston_test: Path) -> Path:
    test_path = bs_test if family == "black_scholes" else heston_test
    if not test_path.exists():
        raise FileNotFoundError(
            f"test set for family={family} not found at {test_path}; "
            "pass --bs-test / --heston-test to override the default"
        )
    return test_path


def main() -> None:
    args = parse_args()
    checkpoints = find_checkpoints(args.checkpoints_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(checkpoints)} checkpoints in {args.checkpoints_dir}")
    print(f"Outputs will land in {args.output_dir}")

    started_at = time.perf_counter()
    summary_rows: list[dict[str, Any]] = []

    for index, checkpoint_dir in enumerate(checkpoints, start=1):
        family = detect_family(checkpoint_dir)
        test_path = _resolve_test_path(family, args.bs_test, args.heston_test)
        eval_csv = args.output_dir / f"{checkpoint_dir.name}_eval.csv"

        print(
            f"\n[{index}/{len(checkpoints)}] {checkpoint_dir.name} "
            f"({family}) -> {eval_csv.name}"
        )
        per_started = time.perf_counter()
        returncode = evaluate_one(
            sys.executable,
            checkpoint_dir,
            test_path,
            eval_csv,
            args.device,
            args.batch_size,
            args.no_iv,
        )
        per_elapsed = time.perf_counter() - per_started
        if returncode != 0:
            raise RuntimeError(
                f"evaluate_surrogate.py failed for {checkpoint_dir.name} "
                f"after {per_elapsed:.1f}s with exit code {returncode}"
            )
        print(f"  done in {per_elapsed:.1f}s")

        summary_rows.append(summarize_one(checkpoint_dir.name, test_path, eval_csv))

    summary_path = args.output_dir / args.summary_name
    write_summary(summary_rows, summary_path)

    total_elapsed = time.perf_counter() - started_at
    print()
    print(f"=== Done in {total_elapsed:.1f}s ===")
    print(f"Per-surrogate CSVs: {len(summary_rows)} files in {args.output_dir}")
    print(f"Summary CSV:        {summary_path}")


if __name__ == "__main__":
    main()
