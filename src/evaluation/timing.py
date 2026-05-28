"""End-to-end timing benchmark for E4 (efficiency study).

Materializes the protocol pre-registered in ``docs/tasks.md`` §E4 and
``docs/metodologia.md`` §"Experimento E4 — Eficiencia computacional":

    "El protocolo concreto usará lotes de 10^2, 10^3, 10^4 y 10^5
    opciones. Para cada tamaño se harán tres ejecuciones de warmup que
    no se miden y diez repeticiones medidas. Se reportará la mediana
    del tiempo y el rango intercuartílico p25/p75, porque la mediana es
    más robusta ante ejecuciones aisladas lentas. El surrogate se
    evaluará con `model.eval()` y `torch.no_grad()`. Si hay GPU
    disponible, se reportarán por separado CPU y GPU para no mezclar
    mejora algorítmica con diferencia de hardware."

    "Los tiempos solo serán comparables si se documentan warmup,
    repeticiones, hardware, modo CPU/GPU y si se incluyen conversiones
    de datos."

And the contract declared in ``docs/architecture.md`` §Evaluation:

    "La clase `TimingBenchmark` implementa el protocolo de eficiencia
    del experimento E4: recibe un surrogate y un solver, ejecuta tres
    warmups y diez repeticiones medidas por cada lote 10^2, 10^3, 10^4
    y 10^5, y devuelve mediana, p25/p75 y speedup por lote, separando
    CPU y GPU cuando aplique."

The benchmark therefore:

* Includes the cost of moving the input batch from NumPy to ``torch``
  and onto the target device on every surrogate measurement. The
  methodology document explicitly asks to disclose whether conversions
  are included; we include them because they are part of what the user
  actually pays in a calibration loop.
* Calls ``torch.cuda.synchronize`` before stopping the clock on CUDA
  devices so the recorded time reflects the kernel finishing, not the
  launch returning.
* Measures the Heston solver **once** per batch size and reuses those
  timings across every device under benchmark; the solver runs on CPU
  regardless of where the surrogate lives, so measuring it twice would
  be wasted wall-clock time without adding information.
* Optionally fans the solver out across multiple processes with
  ``solver_workers``; this matches what a calibration loop user would
  actually do in production (the in-process ``scipy.integrate.quad``
  per point is serial by construction) and avoids one core saturating
  while the other 23 sit idle.
* Emits one log line per measured cell (with wall-clock delta, batch
  size, device and current speedup) so a long run is observable in
  real time instead of going silent for minutes.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from src.solvers.black_scholes import BlackScholesSolver
from src.solvers.heston import HestonSolver

OptionPricer = BlackScholesSolver | HestonSolver


# Default protocol constants — pre-registered in ``tasks.md`` §E4 /
# ``metodologia.md`` §E4. Exposed so the script and docs cite them
# verbatim and tests can override them with smaller values.
DEFAULT_BATCH_SIZES: tuple[int, ...] = (100, 1_000, 10_000, 100_000)
DEFAULT_N_WARMUPS: int = 3
DEFAULT_N_REPETITIONS: int = 10


@dataclass(frozen=True)
class TimingResult:
    """Raw timings for one ``(device, batch_size)`` cell."""

    device: str
    batch_size: int
    n_repetitions: int
    solver_times_s: tuple[float, ...]
    surrogate_times_s: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.solver_times_s) != self.n_repetitions:
            raise ValueError(
                "solver_times_s length "
                f"({len(self.solver_times_s)}) does not match "
                f"n_repetitions ({self.n_repetitions})"
            )
        if len(self.surrogate_times_s) != self.n_repetitions:
            raise ValueError(
                "surrogate_times_s length "
                f"({len(self.surrogate_times_s)}) does not match "
                f"n_repetitions ({self.n_repetitions})"
            )

    @property
    def solver_median_s(self) -> float:
        return float(np.median(self.solver_times_s))

    @property
    def solver_p25_s(self) -> float:
        return float(np.percentile(self.solver_times_s, 25))

    @property
    def solver_p75_s(self) -> float:
        return float(np.percentile(self.solver_times_s, 75))

    @property
    def surrogate_median_s(self) -> float:
        return float(np.median(self.surrogate_times_s))

    @property
    def surrogate_p25_s(self) -> float:
        return float(np.percentile(self.surrogate_times_s, 25))

    @property
    def surrogate_p75_s(self) -> float:
        return float(np.percentile(self.surrogate_times_s, 75))

    @property
    def speedup_median(self) -> float:
        if self.surrogate_median_s <= 0.0:
            return float("inf")
        return self.solver_median_s / self.surrogate_median_s


@dataclass(frozen=True)
class TimingBenchmark:
    """Time the surrogate vs the solver on identical input slices."""

    pricer: OptionPricer
    raw_inputs: np.ndarray
    features: np.ndarray
    input_names: tuple[str, ...]
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES
    n_warmups: int = DEFAULT_N_WARMUPS
    n_repetitions: int = DEFAULT_N_REPETITIONS
    solver_workers: int = 1

    def __post_init__(self) -> None:
        if self.raw_inputs.ndim != 2:
            raise ValueError(
                f"raw_inputs must be 2D; got shape {self.raw_inputs.shape}"
            )
        if self.features.ndim != 2:
            raise ValueError(
                f"features must be 2D; got shape {self.features.shape}"
            )
        if self.raw_inputs.shape[0] != self.features.shape[0]:
            raise ValueError(
                "raw_inputs and features must share the first dimension "
                f"({self.raw_inputs.shape[0]} vs {self.features.shape[0]})"
            )
        if not self.batch_sizes:
            raise ValueError("batch_sizes must contain at least one value")
        if any(b <= 0 for b in self.batch_sizes):
            raise ValueError("every batch size must be strictly positive")
        if max(self.batch_sizes) > self.raw_inputs.shape[0]:
            raise ValueError(
                f"largest batch size ({max(self.batch_sizes)}) exceeds the "
                f"pool of available points ({self.raw_inputs.shape[0]})"
            )
        if self.n_warmups < 0:
            raise ValueError("n_warmups must be non-negative")
        if self.n_repetitions <= 0:
            raise ValueError("n_repetitions must be strictly positive")
        if self.solver_workers < 1:
            raise ValueError("solver_workers must be >= 1")
        if len(self.input_names) != self.raw_inputs.shape[1]:
            raise ValueError(
                "input_names length does not match raw_inputs columns "
                f"({len(self.input_names)} vs {self.raw_inputs.shape[1]})"
            )

    def run(
        self,
        surrogate: nn.Module,
        devices: tuple[str, ...],
        logger: Callable[[str], None] | None = None,
    ) -> tuple[TimingResult, ...]:
        """Benchmark surrogate vs solver on every (device, batch_size) cell.

        The solver is timed exactly once per batch size (its work does
        not depend on the surrogate device) and the resulting timings
        are reused for each ``TimingResult`` that shares that batch
        size. ``logger`` defaults to ``print`` with ``flush=True``;
        pass ``logger=lambda _: None`` to suppress output (tests do).
        """
        if not devices:
            raise ValueError("at least one device must be provided")
        log = logger if logger is not None else _print_flush
        wall_start = time.perf_counter()

        log(
            f"[{_elapsed(wall_start)}] benchmark start: "
            f"batch_sizes={list(self.batch_sizes)}, devices={list(devices)}, "
            f"warmups={self.n_warmups}, repetitions={self.n_repetitions}, "
            f"solver_workers={self.solver_workers}"
        )

        surrogate.eval()

        results: list[TimingResult] = []
        with _maybe_pool(self.solver_workers) as pool:
            for batch_size in self.batch_sizes:
                log(
                    f"[{_elapsed(wall_start)}] === batch_size={batch_size} ==="
                )
                solver_times = _measure_solver(
                    pricer=self.pricer,
                    raw_inputs=self.raw_inputs[:batch_size],
                    input_names=self.input_names,
                    n_warmups=self.n_warmups,
                    n_repetitions=self.n_repetitions,
                    pool=pool,
                    n_workers=self.solver_workers,
                )
                solver_median = float(np.median(solver_times))
                log(
                    f"[{_elapsed(wall_start)}]   solver done. "
                    f"median={solver_median:.4f}s, "
                    f"throughput={batch_size / max(solver_median, 1e-12):.1f} opt/s"
                )

                for device in devices:
                    surr_times = _measure_surrogate(
                        surrogate=surrogate,
                        features=self.features[:batch_size],
                        device=device,
                        n_warmups=self.n_warmups,
                        n_repetitions=self.n_repetitions,
                    )
                    surrogate_median = float(np.median(surr_times))
                    speedup = (
                        solver_median / surrogate_median
                        if surrogate_median > 0
                        else float("inf")
                    )
                    log(
                        f"[{_elapsed(wall_start)}]   surrogate[{device}] done. "
                        f"median={surrogate_median:.4e}s, "
                        f"speedup=x{speedup:.1f}"
                    )
                    results.append(
                        TimingResult(
                            device=device,
                            batch_size=batch_size,
                            n_repetitions=self.n_repetitions,
                            solver_times_s=tuple(solver_times),
                            surrogate_times_s=tuple(surr_times),
                        )
                    )

        log(f"[{_elapsed(wall_start)}] benchmark complete.")
        return tuple(results)


# ---------------------------------------------------------------------------
# Solver timing (single-process or pool-fanned)
# ---------------------------------------------------------------------------


def _measure_solver(
    *,
    pricer: OptionPricer,
    raw_inputs: np.ndarray,
    input_names: tuple[str, ...],
    n_warmups: int,
    n_repetitions: int,
    pool: ProcessPoolExecutor | None,
    n_workers: int,
) -> list[float]:
    if pool is None or n_workers <= 1 or raw_inputs.shape[0] < n_workers:
        return _measure_solver_serial(
            pricer=pricer,
            raw_inputs=raw_inputs,
            input_names=input_names,
            n_warmups=n_warmups,
            n_repetitions=n_repetitions,
        )
    return _measure_solver_parallel(
        pricer=pricer,
        raw_inputs=raw_inputs,
        input_names=input_names,
        n_warmups=n_warmups,
        n_repetitions=n_repetitions,
        pool=pool,
        n_workers=n_workers,
    )


def _measure_solver_serial(
    *,
    pricer: OptionPricer,
    raw_inputs: np.ndarray,
    input_names: tuple[str, ...],
    n_warmups: int,
    n_repetitions: int,
) -> list[float]:
    kwargs = _solver_kwargs(pricer, raw_inputs, input_names)
    for _ in range(n_warmups):
        pricer.call_price(**kwargs)
    times: list[float] = []
    for _ in range(n_repetitions):
        start = time.perf_counter()
        pricer.call_price(**kwargs)
        times.append(time.perf_counter() - start)
    return times


def _measure_solver_parallel(
    *,
    pricer: OptionPricer,
    raw_inputs: np.ndarray,
    input_names: tuple[str, ...],
    n_warmups: int,
    n_repetitions: int,
    pool: ProcessPoolExecutor,
    n_workers: int,
) -> list[float]:
    """Fan the solver out across processes by splitting ``raw_inputs``.

    Each worker receives a contiguous chunk of the input batch and runs
    the solver on it. The wall-clock time recorded is the time from
    submitting the first chunk to receiving the last result, which is
    the number a calibration-loop user would feel. The pool itself is
    created once by the caller and reused across batch sizes.
    """
    chunks = _split_chunks(raw_inputs, n_workers)
    args = [(pricer, chunk, input_names) for chunk in chunks]
    for _ in range(n_warmups):
        list(pool.map(_solver_worker, args))
    times: list[float] = []
    for _ in range(n_repetitions):
        start = time.perf_counter()
        list(pool.map(_solver_worker, args))
        times.append(time.perf_counter() - start)
    return times


def _solver_worker(args: tuple[Any, np.ndarray, tuple[str, ...]]) -> np.ndarray:
    pricer, raw_inputs, input_names = args
    kwargs = _solver_kwargs(pricer, raw_inputs, input_names)
    return np.asarray(pricer.call_price(**kwargs))


def _split_chunks(
    raw_inputs: np.ndarray, n_workers: int
) -> list[np.ndarray]:
    chunks = np.array_split(raw_inputs, n_workers)
    return [c for c in chunks if c.shape[0] > 0]


# ---------------------------------------------------------------------------
# Surrogate timing (per device)
# ---------------------------------------------------------------------------


def _measure_surrogate(
    *,
    surrogate: nn.Module,
    features: np.ndarray,
    device: str,
    n_warmups: int,
    n_repetitions: int,
) -> list[float]:
    surrogate.to(device)
    is_cuda = _is_cuda_device(device)
    for _ in range(n_warmups):
        with torch.no_grad():
            tensor = torch.from_numpy(features).to(device)
            _ = surrogate(tensor)
            if is_cuda:
                torch.cuda.synchronize(device)
    times: list[float] = []
    for _ in range(n_repetitions):
        start = time.perf_counter()
        with torch.no_grad():
            tensor = torch.from_numpy(features).to(device)
            _ = surrogate(tensor)
            if is_cuda:
                torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    return times


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solver_kwargs(
    pricer: OptionPricer,
    raw_inputs: np.ndarray,
    input_names: tuple[str, ...],
) -> dict[str, Any]:
    """Build the keyword arguments expected by ``pricer.call_price``."""
    columns = {
        name: raw_inputs[:, idx].astype(np.float64)
        for idx, name in enumerate(input_names)
    }
    base_kwargs: dict[str, Any] = {
        "spot": columns["moneyness"],
        "strike": 1.0,
        "maturity": columns["maturity"],
        "rate": columns["rate"],
        "dividend_yield": 0.0,
    }
    if isinstance(pricer, BlackScholesSolver):
        base_kwargs["volatility"] = columns["volatility"]
    elif isinstance(pricer, HestonSolver):
        for name in ("v0", "theta", "kappa", "xi", "rho"):
            base_kwargs[name] = columns[name]
    else:
        raise TypeError(f"unsupported pricer type: {type(pricer).__name__}")
    return base_kwargs


def _is_cuda_device(device: str) -> bool:
    return device.startswith("cuda")


def _print_flush(message: str) -> None:
    print(message, flush=True)


def _elapsed(start: float) -> str:
    delta = time.perf_counter() - start
    minutes = int(delta // 60)
    seconds = delta - minutes * 60
    return f"+{minutes:02d}m{seconds:05.2f}s"


class _NullPool:
    """Sentinel used in place of a real pool when ``solver_workers==1``."""

    def __enter__(self) -> "_NullPool":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def _maybe_pool(n_workers: int) -> ProcessPoolExecutor | _NullPool:
    if n_workers <= 1:
        return _NullPool()
    return ProcessPoolExecutor(max_workers=n_workers)


def default_solver_workers() -> int:
    """Sensible default for ``solver_workers`` based on CPU count.

    Mirrors the convention used elsewhere in the project: ``cpu_count - 2``
    leaves room for the OS and the surrogate while still feeding the
    solver fan-out, and clamps to at least 1 on tiny boxes.
    """
    return max(1, (os.cpu_count() or 2) - 2)
