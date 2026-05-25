"""50x dataset pipeline orchestrator with an overall progress bar.

Replaces ``generate_all_datasets_50x.bat`` for long-running 50x runs where
seeing live progress matters. Each subprocess writes its full stdout to a
per-dataset log under ``results/logs/``; the main process shows a single
live progress line (overall percent, current dataset percent, elapsed, ETA)
so the terminal stays readable for ~10 hours.

Estimates per dataset are seeded from the throughput measured in the prior
500k run (1446-2400 samples/s for Heston) and they self-adjust as datasets
complete (the overall ETA reflects actual vs estimated elapsed).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
LOG_DIR = REPO_ROOT / "results" / "logs"
PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
GEN_SCRIPT = REPO_ROOT / "scripts" / "generate_dataset.py"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    estimated_seconds: int
    args: tuple[str, ...]


def heston_uniform(name: str, n_samples: int, seed: int, include_delta: bool, batch_size: int = 5000) -> DatasetSpec:
    args = [
        "--family", "heston", "--sampler", "uniform",
        "--n-samples", str(n_samples), "--batch-size", str(batch_size),
        "--seed", str(seed),
        "--output", str(DATA_DIR / f"{name}.npz"),
    ]
    if include_delta:
        args.append("--include-delta")
    # ~1900 samples/s for Heston uniform with 24 workers (measured)
    return DatasetSpec(name, estimated_seconds=max(60, int(n_samples / 1900)), args=tuple(args))


WORKERS = int(os.environ.get("WORKERS", "24"))

DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        "bs_train_10M_uniform_delta",
        estimated_seconds=90,
        args=(
            "--family", "black_scholes", "--sampler", "uniform",
            "--n-samples", "10000000", "--batch-size", "100000",
            "--seed", "45", "--include-delta",
            "--output", str(DATA_DIR / "bs_train_10M_uniform_delta.npz"),
        ),
    ),
    DatasetSpec(
        "bs_validation_2500k_uniform_delta",
        estimated_seconds=30,
        args=(
            "--family", "black_scholes", "--sampler", "uniform",
            "--n-samples", "2500000", "--batch-size", "100000",
            "--seed", "49", "--include-delta",
            "--output", str(DATA_DIR / "bs_validation_2500k_uniform_delta.npz"),
        ),
    ),
    DatasetSpec(
        "bs_test_6250k_balanced_delta",
        estimated_seconds=90,
        args=(
            "--family", "black_scholes", "--sampler", "balanced",
            "--samples-per-bin", "250000", "--batch-size", "100000",
            "--seed", "47", "--include-delta",
            "--output", str(DATA_DIR / "bs_test_6250k_balanced_delta.npz"),
        ),
    ),
    DatasetSpec(
        "heston_train_25M_uniform",
        estimated_seconds=11364,  # ~3h 9min @ 2200 samples/s
        args=(
            "--family", "heston", "--sampler", "uniform",
            "--n-samples", "25000000", "--batch-size", "5000",
            "--seed", "42",
            "--output", str(DATA_DIR / "heston_train_25M_uniform.npz"),
        ),
    ),
    DatasetSpec(
        "heston_train_25M_focused",
        estimated_seconds=10417,  # ~2h 53min @ 2400 samples/s
        args=(
            "--family", "heston", "--sampler", "focused",
            "--n-samples", "25000000", "--batch-size", "5000",
            "--seed", "44",
            "--output", str(DATA_DIR / "heston_train_25M_focused.npz"),
        ),
    ),
    DatasetSpec(
        "heston_train_5M_uniform",
        estimated_seconds=2632,  # ~44min
        args=(
            "--family", "heston", "--sampler", "uniform",
            "--n-samples", "5000000", "--batch-size", "5000",
            "--seed", "48",
            "--output", str(DATA_DIR / "heston_train_5M_uniform.npz"),
        ),
    ),
    DatasetSpec(
        "heston_train_5M_uniform_delta",
        estimated_seconds=2700,  # ~45min
        args=(
            "--family", "heston", "--sampler", "uniform",
            "--n-samples", "5000000", "--batch-size", "5000",
            "--seed", "43", "--include-delta",
            "--output", str(DATA_DIR / "heston_train_5M_uniform_delta.npz"),
        ),
    ),
    DatasetSpec(
        "heston_validation_2500k_uniform",
        estimated_seconds=1316,  # ~22min
        args=(
            "--family", "heston", "--sampler", "uniform",
            "--n-samples", "2500000", "--batch-size", "5000",
            "--seed", "50",
            "--output", str(DATA_DIR / "heston_validation_2500k_uniform.npz"),
        ),
    ),
    DatasetSpec(
        "heston_test_6250k_balanced_delta",
        estimated_seconds=7423,  # ~2h 04min @ 842 samples/s
        args=(
            "--family", "heston", "--sampler", "balanced",
            "--samples-per-bin", "250000", "--batch-size", "1000",
            "--seed", "46", "--include-delta",
            "--output", str(DATA_DIR / "heston_test_6250k_balanced_delta.npz"),
        ),
    ),
]


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds - 60 * m)
        return f"{m}m{s:02d}s"
    h = int(seconds // 3600)
    m = int((seconds - 3600 * h) // 60)
    return f"{h}h{m:02d}m"


def render_bar(percent: float, width: int = 40) -> str:
    percent = max(0.0, min(100.0, percent))
    filled = int(percent / 100 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    total_estimated = sum(d.estimated_seconds for d in DATASETS)
    overall_start = time.time()
    completed_real = 0.0
    completed_estimated = 0
    use_carriage = sys.stdout.isatty()
    last_logged = 0.0

    print(f"=== 50x dataset pipeline ===")
    print(f"datasets         : {len(DATASETS)}")
    print(f"workers/dataset  : {WORKERS}")
    print(f"estimated total  : {fmt_duration(total_estimated)}")
    print(f"start            : {datetime.now().isoformat(timespec='seconds')}")
    print()

    for idx, spec in enumerate(DATASETS, 1):
        log_file = LOG_DIR / f"generate_{spec.name}.log"
        ds_start = time.time()

        print(
            f"[{idx}/{len(DATASETS)}] {spec.name} (est {fmt_duration(spec.estimated_seconds)})",
            flush=True,
        )

        cmd = [
            str(PY), str(GEN_SCRIPT),
            *spec.args,
            "--workers", str(WORKERS),
            "--overwrite",
        ]

        with log_file.open("w", encoding="utf-8") as handle:
            handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] START {spec.name}\n")
            handle.flush()
            try:
                proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
            except FileNotFoundError as exc:
                print(f"  ERROR launching {spec.name}: {exc}")
                return 1

            try:
                while proc.poll() is None:
                    now = time.time()
                    elapsed_ds = now - ds_start
                    elapsed_total = now - overall_start
                    ds_pct = min(elapsed_ds / spec.estimated_seconds * 100, 99.0)
                    total_so_far = completed_estimated + (ds_pct / 100) * spec.estimated_seconds
                    overall_pct = total_so_far / total_estimated * 100
                    eta = max(0.0, total_estimated - total_so_far)
                    line = (
                        f"  {render_bar(overall_pct)} {overall_pct:5.1f}%  "
                        f"ds {ds_pct:5.1f}%  elapsed {fmt_duration(elapsed_total)}  ETA {fmt_duration(eta)}"
                    )
                    if use_carriage:
                        sys.stdout.write("\r" + line + "    ")
                        sys.stdout.flush()
                        time.sleep(2)
                    else:
                        if now - last_logged >= 60:
                            print(line, flush=True)
                            last_logged = now
                        time.sleep(5)
            except KeyboardInterrupt:
                print("\n  interrupted; terminating subprocess...", flush=True)
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] INTERRUPTED\n")
                return 130

            ds_elapsed = time.time() - ds_start
            handle.write(
                f"[{datetime.now().isoformat(timespec='seconds')}] DONE {spec.name} "
                f"exit={proc.returncode} elapsed={ds_elapsed:.1f}s\n"
            )

        if use_carriage:
            sys.stdout.write("\n")
            sys.stdout.flush()

        if proc.returncode != 0:
            print(f"  ERROR: {spec.name} failed with exit {proc.returncode}")
            print(f"  See log: {log_file}")
            return 1

        completed_estimated += spec.estimated_seconds
        completed_real += ds_elapsed
        actual_vs_est = ds_elapsed / spec.estimated_seconds * 100
        print(
            f"  done in {fmt_duration(ds_elapsed)} ({actual_vs_est:.0f}% of estimate, "
            f"throughput in log)",
            flush=True,
        )

    total_elapsed = time.time() - overall_start
    print()
    print(
        f"=== ALL 50x DATASETS GENERATED in {fmt_duration(total_elapsed)} "
        f"(estimated {fmt_duration(total_estimated)}) ==="
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
