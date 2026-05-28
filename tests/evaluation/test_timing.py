import numpy as np
import pytest
import torch
from torch import nn

from src.evaluation.timing import (
    DEFAULT_BATCH_SIZES,
    DEFAULT_N_REPETITIONS,
    DEFAULT_N_WARMUPS,
    TimingBenchmark,
    TimingResult,
    default_solver_workers,
)
from src.solvers import BlackScholesSolver


# ---------------------------------------------------------------------------
# Helpers — tests use the Black-Scholes solver because it is closed-form and
# fast enough to keep the suite sub-second, even though E4 itself benchmarks
# the Heston solver.
# ---------------------------------------------------------------------------


class _ConstantPriceModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


def _silent(_msg: str) -> None:
    return None


def _build_bs_pool(n: int = 200) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(7)
    moneyness = rng.uniform(0.6, 1.6, size=n).astype(np.float32)
    maturity = rng.uniform(0.1, 1.0, size=n).astype(np.float32)
    rate = rng.uniform(0.0, 0.05, size=n).astype(np.float32)
    sigma = rng.uniform(0.1, 0.5, size=n).astype(np.float32)
    raw_inputs = np.stack([moneyness, maturity, rate, sigma], axis=1)

    features = raw_inputs.copy()
    features[:, 0] = (features[:, 0] - 0.4) / (2.0 - 0.4)
    features[:, 1] = (features[:, 1] - 7.0 / 365.0) / (2.0 - 7.0 / 365.0)
    features[:, 2] = features[:, 2] / 0.075
    features[:, 3] = (features[:, 3] - 0.03) / (1.0 - 0.03)
    input_names = ("moneyness", "maturity", "rate", "volatility")
    return features, raw_inputs, input_names


def _make_benchmark(
    *,
    batch_sizes: tuple[int, ...] = (10, 50),
    n_warmups: int = 1,
    n_repetitions: int = 3,
    solver_workers: int = 1,
) -> TimingBenchmark:
    features, raw_inputs, input_names = _build_bs_pool()
    return TimingBenchmark(
        pricer=BlackScholesSolver(),
        raw_inputs=raw_inputs,
        features=features,
        input_names=input_names,
        batch_sizes=batch_sizes,
        n_warmups=n_warmups,
        n_repetitions=n_repetitions,
        solver_workers=solver_workers,
    )


# ---------------------------------------------------------------------------
# Protocol defaults are the values pre-registered in tasks.md §E4
# ---------------------------------------------------------------------------


def test_protocol_defaults_match_methodology() -> None:
    assert DEFAULT_BATCH_SIZES == (100, 1_000, 10_000, 100_000)
    assert DEFAULT_N_WARMUPS == 3
    assert DEFAULT_N_REPETITIONS == 10


def test_default_solver_workers_is_positive() -> None:
    assert default_solver_workers() >= 1


# ---------------------------------------------------------------------------
# TimingResult statistics
# ---------------------------------------------------------------------------


def test_timing_result_statistics_match_numpy() -> None:
    solver_times = (0.10, 0.12, 0.11, 0.20, 0.13)
    surrogate_times = (0.001, 0.002, 0.0015, 0.0011, 0.0013)
    result = TimingResult(
        device="cpu",
        batch_size=100,
        n_repetitions=5,
        solver_times_s=solver_times,
        surrogate_times_s=surrogate_times,
    )

    assert result.solver_median_s == float(np.median(solver_times))
    assert result.surrogate_median_s == float(np.median(surrogate_times))
    assert result.solver_p25_s == float(np.percentile(solver_times, 25))
    assert result.solver_p75_s == float(np.percentile(solver_times, 75))
    expected_speedup = float(np.median(solver_times)) / float(np.median(surrogate_times))
    assert result.speedup_median == pytest.approx(expected_speedup)


def test_timing_result_rejects_wrong_length_arrays() -> None:
    with pytest.raises(ValueError, match="solver_times_s"):
        TimingResult(
            device="cpu",
            batch_size=10,
            n_repetitions=3,
            solver_times_s=(0.1, 0.2),
            surrogate_times_s=(0.01, 0.02, 0.03),
        )


def test_timing_result_speedup_is_inf_on_zero_surrogate_time() -> None:
    result = TimingResult(
        device="cpu",
        batch_size=10,
        n_repetitions=3,
        solver_times_s=(0.1, 0.1, 0.1),
        surrogate_times_s=(0.0, 0.0, 0.0),
    )
    assert result.speedup_median == float("inf")


# ---------------------------------------------------------------------------
# TimingBenchmark construction
# ---------------------------------------------------------------------------


def test_benchmark_rejects_batch_larger_than_pool() -> None:
    features, raw_inputs, names = _build_bs_pool(n=50)
    with pytest.raises(ValueError, match="exceeds the pool"):
        TimingBenchmark(
            pricer=BlackScholesSolver(),
            raw_inputs=raw_inputs,
            features=features,
            input_names=names,
            batch_sizes=(10, 100),
        )


def test_benchmark_rejects_empty_batch_sizes() -> None:
    features, raw_inputs, names = _build_bs_pool()
    with pytest.raises(ValueError, match="at least one"):
        TimingBenchmark(
            pricer=BlackScholesSolver(),
            raw_inputs=raw_inputs,
            features=features,
            input_names=names,
            batch_sizes=(),
        )


def test_benchmark_rejects_input_names_mismatch() -> None:
    features, raw_inputs, _names = _build_bs_pool()
    with pytest.raises(ValueError, match="input_names"):
        TimingBenchmark(
            pricer=BlackScholesSolver(),
            raw_inputs=raw_inputs,
            features=features,
            input_names=("moneyness", "maturity"),
            batch_sizes=(10,),
        )


def test_benchmark_rejects_non_positive_repetitions() -> None:
    features, raw_inputs, names = _build_bs_pool()
    with pytest.raises(ValueError, match="n_repetitions"):
        TimingBenchmark(
            pricer=BlackScholesSolver(),
            raw_inputs=raw_inputs,
            features=features,
            input_names=names,
            batch_sizes=(10,),
            n_repetitions=0,
        )


def test_benchmark_rejects_zero_solver_workers() -> None:
    features, raw_inputs, names = _build_bs_pool()
    with pytest.raises(ValueError, match="solver_workers"):
        TimingBenchmark(
            pricer=BlackScholesSolver(),
            raw_inputs=raw_inputs,
            features=features,
            input_names=names,
            batch_sizes=(10,),
            solver_workers=0,
        )


# ---------------------------------------------------------------------------
# TimingBenchmark.run — structural guarantees
# ---------------------------------------------------------------------------


def test_run_rejects_empty_devices() -> None:
    benchmark = _make_benchmark()
    with pytest.raises(ValueError, match="at least one device"):
        benchmark.run(_ConstantPriceModel(), (), logger=_silent)


def test_run_returns_one_result_per_device_and_batch_size() -> None:
    benchmark = _make_benchmark(batch_sizes=(10, 50, 100))
    results = benchmark.run(_ConstantPriceModel(), ("cpu",), logger=_silent)
    assert len(results) == 3
    assert tuple(r.batch_size for r in results) == (10, 50, 100)


def test_run_records_exactly_n_repetitions_per_result() -> None:
    benchmark = _make_benchmark(batch_sizes=(20,), n_repetitions=4, n_warmups=2)
    (result,) = benchmark.run(_ConstantPriceModel(), ("cpu",), logger=_silent)
    assert len(result.solver_times_s) == 4
    assert len(result.surrogate_times_s) == 4
    assert result.n_repetitions == 4


def test_run_marks_device_on_every_result() -> None:
    benchmark = _make_benchmark(batch_sizes=(10, 20))
    results = benchmark.run(_ConstantPriceModel(), ("cpu",), logger=_silent)
    assert all(r.device == "cpu" for r in results)


def test_run_produces_finite_positive_timings() -> None:
    benchmark = _make_benchmark(batch_sizes=(30,))
    (result,) = benchmark.run(_ConstantPriceModel(), ("cpu",), logger=_silent)
    for t in result.solver_times_s + result.surrogate_times_s:
        assert np.isfinite(t)
        assert t >= 0.0


def test_run_includes_data_conversion_in_surrogate_time() -> None:
    """Smoke test that surrogate timing is end-to-end (not just forward)."""
    benchmark = _make_benchmark(batch_sizes=(50,), n_repetitions=5, n_warmups=1)
    (result,) = benchmark.run(_ConstantPriceModel(), ("cpu",), logger=_silent)
    assert result.surrogate_median_s > 0.0


def test_run_reuses_solver_times_across_devices() -> None:
    """Solver runs once per batch, its times must be identical across devices."""
    benchmark = _make_benchmark(batch_sizes=(20,))
    results = benchmark.run(
        _ConstantPriceModel(), ("cpu", "cpu"), logger=_silent
    )
    assert len(results) == 2
    # Same batch on two device entries => the same solver timing tuple
    # must be shared (the solver only ran once).
    assert results[0].solver_times_s == results[1].solver_times_s


def test_run_logs_per_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom logger receives at least one message per (device, batch) cell."""
    benchmark = _make_benchmark(batch_sizes=(10, 20))
    messages: list[str] = []
    benchmark.run(_ConstantPriceModel(), ("cpu",), logger=messages.append)
    joined = "\n".join(messages)
    # Both batch sizes must show up in the log
    assert "batch_size=10" in joined
    assert "batch_size=20" in joined
    # And both phases (solver + surrogate) must be reported
    assert "solver done" in joined
    assert "surrogate[cpu]" in joined


def test_run_with_pool_workers_succeeds() -> None:
    """Solver with workers>1 returns the same shape as the serial path."""
    benchmark = _make_benchmark(
        batch_sizes=(40,), n_repetitions=2, n_warmups=1, solver_workers=2
    )
    (result,) = benchmark.run(_ConstantPriceModel(), ("cpu",), logger=_silent)
    assert len(result.solver_times_s) == 2
    assert len(result.surrogate_times_s) == 2
    for t in result.solver_times_s:
        assert t > 0.0
