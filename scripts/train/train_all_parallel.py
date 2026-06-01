"""Orquestador paralelo para entrenar los 11 surrogates en una sola GPU.

Cada surrogate corre en su propio ``subprocess`` y un ``ThreadPoolExecutor``
controla cuántos entrenamientos se lanzan a la vez. Con la MLP pequeña
usada en el proyecto, un único proceso infrautiliza la GPU; varios procesos
en paralelo aumentan el throughput.

Los valores por defecto están pensados para una RTX 4060 de 8 GB:
``--parallel 4`` mantiene la VRAM en un rango razonable y
``--preload-to-device`` se activa salvo que se pase ``--no-preload``.
"""

from __future__ import annotations

import argparse
import platform
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_RESULTS = REPO_ROOT / "results" / "checkpoints"
DEFAULT_LOGS = REPO_ROOT / "results" / "logs"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train" / "train_surrogate.py"


@dataclass(frozen=True)
class SurrogateSpec:
    """Configuración mínima de un surrogate para el lanzador paralelo."""

    experiment_id: str
    train_file: str
    validation_file: str
    loss: str
    activation: str
    batch_size: int
    seed: int


SURROGATES: list[SurrogateSpec] = [
    SurrogateSpec("BS-1",      "bs_train_200k_uniform_delta.npz",        "bs_validation_50k_uniform_delta.npz",     "price",        "relu",     32768, 101),
    SurrogateSpec("BS-2",      "bs_train_200k_uniform_delta.npz",        "bs_validation_50k_uniform_delta.npz",     "price",        "softplus", 32768, 102),
    SurrogateSpec("BS-3",      "bs_train_200k_uniform_delta.npz",        "bs_validation_50k_uniform_delta.npz",     "price",        "swish",    32768, 103),
    SurrogateSpec("BS-4",      "bs_train_200k_uniform_delta.npz",        "bs_validation_50k_uniform_delta.npz",     "price",        "tanh",     32768, 104),
    SurrogateSpec("H-1",       "heston_train_500k_uniform.npz",          "heston_validation_50k_uniform.npz",       "price",        "relu",     32768, 201),
    SurrogateSpec("H-2",       "heston_train_500k_uniform.npz",          "heston_validation_50k_uniform.npz",       "price",        "softplus", 32768, 202),
    SurrogateSpec("H-3",       "heston_train_500k_uniform.npz",          "heston_validation_50k_uniform.npz",       "price",        "swish",    32768, 203),
    SurrogateSpec("H-4",       "heston_train_500k_uniform.npz",          "heston_validation_50k_uniform.npz",       "price",        "tanh",     32768, 204),
    SurrogateSpec("H-5",       "heston_train_500k_focused.npz",          "heston_validation_50k_uniform.npz",       "price",        "swish",    32768, 205),
    SurrogateSpec("H-3-small", "heston_train_100k_uniform.npz",          "heston_validation_50k_uniform.npz",       "price",        "swish",    16384, 206),
    SurrogateSpec("H-6-small", "heston_train_100k_uniform_delta.npz",    "heston_validation_50k_uniform.npz",       "differential", "swish",    16384, 207),
]


def build_command(
    spec: SurrogateSpec,
    args: argparse.Namespace,
) -> list[str]:
    """Compone el comando ``python train_surrogate.py ...`` para ``spec``."""
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--experiment-id", spec.experiment_id,
        "--output-dir", str(args.results_dir / spec.experiment_id),
        "--train", str(args.data_dir / spec.train_file),
        "--validation", str(args.data_dir / spec.validation_file),
        "--loss", spec.loss,
        "--activation", spec.activation,
        "--epochs", str(args.epochs),
        "--batch-size", str(spec.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--hidden-width", str(args.hidden_width),
        "--hidden-layers", str(args.hidden_layers),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--seed", str(spec.seed),
    ]
    if args.compile:
        cmd.append("--compile")
    if args.preload_to_device:
        cmd.append("--preload-to-device")
    return cmd


def run_surrogate(spec: SurrogateSpec, args: argparse.Namespace) -> tuple[str, int, float]:
    """Lanza un entrenamiento como subproceso y devuelve (id, exit_code, segundos)."""
    log_file = args.log_dir / f"train_{spec.experiment_id}.log"
    started = datetime.now()
    with log_file.open("w", encoding="utf-8") as handle:
        handle.write(f"[{started.isoformat(timespec='seconds')}] START {spec.experiment_id}\n")
        handle.flush()
        result = subprocess.run(
            build_command(spec, args),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        ended = datetime.now()
        handle.write(f"[{ended.isoformat(timespec='seconds')}] DONE {spec.experiment_id} exit={result.returncode}\n")
    elapsed = (ended - started).total_seconds()
    return spec.experiment_id, result.returncode, elapsed


def main() -> int:
    """Lanza los 11 entrenamientos con ``--parallel`` workers concurrentes."""
    parser = argparse.ArgumentParser(description="Entrena los 11 surrogates en paralelo sobre una sola GPU.")
    parser.add_argument("--parallel", type=int, default=4, help="Número de subprocesses de entrenamiento concurrentes.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0, help="Workers de DataLoader por proceso; no aplica con --preload-to-device.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda' o cualquier device válido de Torch.")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Activa torch.compile() por proceso. Requiere Triton; se ignora en Windows.",
    )
    parser.add_argument("--no-preload", action="store_true", help="Desactiva --preload-to-device, activo por defecto.")
    args = parser.parse_args()

    if args.compile and platform.system() == "Windows":
        print(
            "[parallel] torch.compile() needs Triton, which has no official Windows wheel. "
            "Ignoring --compile.",
            flush=True,
        )
        args.compile = False
    args.preload_to_device = not args.no_preload

    if args.parallel <= 0:
        raise SystemExit("--parallel must be strictly positive")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[parallel] launching {len(SURROGATES)} surrogates with {args.parallel} concurrent workers",
        flush=True,
    )
    print(
        f"[parallel] device={args.device} compile={args.compile} preload={args.preload_to_device} "
        f"epochs={args.epochs} num_workers={args.num_workers}",
        flush=True,
    )
    print(
        f"[parallel] logs: {args.log_dir.relative_to(REPO_ROOT)}",
        flush=True,
    )

    started_at = datetime.now()
    failures: list[tuple[str, int]] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(run_surrogate, spec, args): spec.experiment_id
            for spec in SURROGATES
        }
        for future in as_completed(futures):
            experiment_id, code, elapsed = future.result()
            completed += 1
            status = "OK" if code == 0 else f"FAIL(exit={code})"
            print(
                f"[parallel] {completed}/{len(SURROGATES)} {experiment_id} {status} elapsed={elapsed:.1f}s",
                flush=True,
            )
            if code != 0:
                failures.append((experiment_id, code))

    total_elapsed = (datetime.now() - started_at).total_seconds()
    if failures:
        print(f"[parallel] FAILED: {failures}", flush=True)
        print(f"[parallel] total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", flush=True)
        return 1

    print(f"[parallel] ALL SURROGATES TRAINED in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
