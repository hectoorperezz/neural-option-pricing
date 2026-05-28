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
* Leaves the Heston solver serial — it is single-process by
  construction (``scipy.integrate.quad`` per point) and the experiment
  is meant to measure the solver "as is".
* Does not touch ``torch.set_num_threads``; PyTorch CPU concurrency is
  whatever the user gets by default in production, and that is the
  reference for the speedup we report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from src.solvers.black_scholes import BlackScholesSolver
from src.solvers.heston import HestonSolver

OptionPricer = BlackScholesSolver | HestonSolver


# Default protocol constants — these are the values pre-registered in
# ``tasks.md`` §E4 / ``metodologia.md`` §E4. Exposed as module-level
# constants so the script and the docs can cite them verbatim and so
# tests can override them with smaller values when needed.
DEFAULT_BATCH_SIZES: tuple[int, ...] = (100, 1_000, 10_000, 100_000)
DEFAULT_N_WARMUPS: int = 3
DEFAULT_N_REPETITIONS: int = 10


@dataclass(frozen=True)
class TimingResult:
    """Raw timings for one ``(device, batch_size)`` cell.

    ``solver_times_s`` and ``surrogate_times_s`` contain exactly
    ``n_repetitions`` wall-clock seconds (warmups are discarded before
    populating the tuples). All statistics are derived properties so the
    result stays a pure data object.
    """

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
        """``solver_median / surrogate_median``. ``inf`` if surrogate is 0."""
        if self.surrogate_median_s <= 0.0:
            return float("inf")
        return self.solver_median_s / self.surrogate_median_s


@dataclass(frozen=True)
class TimingBenchmark:
    """Time the surrogate vs the solver on identical input slices.

    The benchmark stores the full pool of points and slices the first
    ``batch_size`` of them per measurement so that the solver and the
    surrogate see the same option specifications at every batch size.
    The protocol constants (``n_warmups``, ``n_repetitions``,
    ``batch_sizes``) default to the values pre-registered in
    ``tasks.md`` §E4 and ``metodologia.md`` §E4.
    """

    pricer: OptionPricer
    raw_inputs: np.ndarray
    features: np.ndarray
    input_names: tuple[str, ...]
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES
    n_warmups: int = DEFAULT_N_WARMUPS
    n_repetitions: int = DEFAULT_N_REPETITIONS

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
        if len(self.input_names) != self.raw_inputs.shape[1]:
            raise ValueError(
                "input_names length does not match raw_inputs columns "
                f"({len(self.input_names)} vs {self.raw_inputs.shape[1]})"
            )

    def run(self, surrogate: nn.Module, device: str) -> tuple[TimingResult, ...]:
        """Run the benchmark on ``device`` and return one result per batch."""
        surrogate = surrogate.to(device)
        surrogate.eval()

        results: list[TimingResult] = []
        for batch_size in self.batch_sizes:
            raw_slice = self.raw_inputs[:batch_size]
            feature_slice = self.features[:batch_size]
            solver_times = _measure_solver(
                pricer=self.pricer,
                raw_inputs=raw_slice,
                input_names=self.input_names,
                n_warmups=self.n_warmups,
                n_repetitions=self.n_repetitions,
            )
            surrogate_times = _measure_surrogate(
                surrogate=surrogate,
                features=feature_slice,
                device=device,
                n_warmups=self.n_warmups,
                n_repetitions=self.n_repetitions,
            )
            results.append(
                TimingResult(
                    device=device,
                    batch_size=batch_size,
                    n_repetitions=self.n_repetitions,
                    solver_times_s=tuple(solver_times),
                    surrogate_times_s=tuple(surrogate_times),
                )
            )
        return tuple(results)


def _measure_solver(
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


def _measure_surrogate(
    *,
    surrogate: nn.Module,
    features: np.ndarray,
    device: str,
    n_warmups: int,
    n_repetitions: int,
) -> list[float]:
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


def _solver_kwargs(
    pricer: OptionPricer,
    raw_inputs: np.ndarray,
    input_names: tuple[str, ...],
) -> dict[str, Any]:
    """Build the keyword arguments expected by ``pricer.call_price``.

    The dataset generator (``scripts/generate_dataset.py``) writes
    ``raw_inputs`` with one column per non-fixed input — for Black-Scholes
    that is ``(moneyness, maturity, rate, volatility)``, for Heston it is
    ``(moneyness, maturity, rate, v0, theta, kappa, xi, rho)``. Strike
    and dividend yield are pipeline-wide constants (``strike=1``,
    ``q=0``) per the conventions documented in ``metodologia.md``.
    """
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
