"""Pointwise and per-bin error metrics for surrogate evaluation.

The functions in this module follow the contract laid out in
``docs/metodologia.md`` (section "Targets, normalización y métricas") and in
``docs/architecture.md`` (section "Evaluation"). They are deliberately
written as free functions rather than methods on a class because the logic
is purely algorithmic: no polymorphism is needed.

Three metrics are produced for every surrogate, all of them expressed as
absolute errors so they can be aggregated per bin without further
transformation:

* ``MAE(C/K)`` — absolute error on the normalized call price ``y = C/K``.
* ``MAE_Delta`` — absolute error on ``Delta = dy/dm``, obtained from the
  surrogate by autograd and from the solver as ``N(d1)`` (BS) or ``P1``
  (Heston, with ``q = 0``).
* ``MAE_IV`` — absolute error on the Black-Scholes implied volatility
  recovered by inverting the predicted price. Inversion failures are
  reported through ``ok_mask`` rather than silently dropped; the
  methodology document mandates that this diagnostic is preserved.

Per-bin aggregation reports the mean and the 50/95/99 percentiles, with
the methodology document highlighting ``p95`` as the column that exposes
operationally relevant tail errors.

Conventions inherited from the dataset generator (``src/datasets/generator.py``):

* ``strike = 1.0``, so the predicted output of the network is directly the
  call price ``C`` and ``spot = moneyness``.
* ``dividend_yield = 0`` in every experiment.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn

from src.models.greeks import surrogate_price_and_delta
from src.solvers.iv import ImpliedVolatilityInverter


def absolute_errors(
    predicted: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Element-wise absolute error ``|predicted - target|`` as ``np.ndarray``.

    Accepts NumPy arrays or PyTorch tensors interchangeably; the result is
    always a float64 NumPy array detached from any autograd graph.
    """
    pred_array = _to_numpy(predicted)
    target_array = _to_numpy(target)
    if pred_array.shape != target_array.shape:
        raise ValueError(
            "predicted and target must have the same shape "
            f"(got {pred_array.shape} and {target_array.shape})"
        )
    return np.abs(pred_array.astype(np.float64) - target_array.astype(np.float64))


def aggregate_by_bin(
    values: np.ndarray,
    bin_id: np.ndarray,
    n_bins: int,
    percentiles: Iterable[int] = (50, 95, 99),
) -> dict[str, np.ndarray]:
    """Aggregate per-point values into per-bin summary statistics.

    Returns a dict with one entry per statistic, each entry an array of shape
    ``(n_bins,)``: ``mean``, ``count``, plus one entry per requested
    percentile (e.g. ``p50``, ``p95``, ``p99``). ``count`` is the number of
    finite values that contributed to the aggregates of that bin; ``NaN``
    inputs are ignored. Bins that receive no finite value get ``count = 0``
    and ``NaN`` for every aggregate, so heatmaps can show empty cells
    cleanly.
    """
    values_arr = np.asarray(values, dtype=np.float64)
    bin_id_arr = np.asarray(bin_id, dtype=np.int64)
    if values_arr.shape != bin_id_arr.shape:
        raise ValueError(
            "values and bin_id must have the same shape "
            f"(got {values_arr.shape} and {bin_id_arr.shape})"
        )
    if n_bins <= 0:
        raise ValueError("n_bins must be strictly positive")
    if bin_id_arr.size > 0 and (bin_id_arr.min() < 0 or bin_id_arr.max() >= n_bins):
        raise ValueError(
            f"bin_id values must lie in [0, {n_bins})"
        )

    percentiles_tuple = tuple(percentiles)
    mean = np.full(n_bins, np.nan, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.int64)
    pct_arrays = {p: np.full(n_bins, np.nan, dtype=np.float64) for p in percentiles_tuple}

    for k in range(n_bins):
        bin_mask = bin_id_arr == k
        if not bin_mask.any():
            continue
        bin_values = values_arr[bin_mask]
        finite_values = bin_values[np.isfinite(bin_values)]
        if finite_values.size == 0:
            continue
        count[k] = finite_values.size
        mean[k] = float(finite_values.mean())
        for p in percentiles_tuple:
            pct_arrays[p][k] = float(np.percentile(finite_values, p))

    result: dict[str, np.ndarray] = {"mean": mean, "count": count}
    for p in percentiles_tuple:
        result[f"p{p}"] = pct_arrays[p]
    return result


def predict_surrogate_prices_and_deltas(
    model: nn.Module,
    features: np.ndarray | torch.Tensor,
    batch_size: int = 32768,
    device: str = "cpu",
    moneyness_range: tuple[float, float] = (0.4, 2.0),
    moneyness_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the surrogate on the full test set in batches and return arrays.

    Both the price ``y_hat = C/K`` and the Delta ``dy/dm`` (recovered by
    autograd with the chain-rule correction defined in
    ``src/models/greeks.py``) are computed in the same forward pass.
    ``create_graph=False`` because evaluation does not need to backpropagate
    through the gradient term. The model is placed in eval mode but autograd
    is left enabled so that Delta can be obtained.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be strictly positive")
    if moneyness_range[1] <= moneyness_range[0]:
        raise ValueError("moneyness_range must satisfy min < max")

    if isinstance(features, np.ndarray):
        features_tensor = torch.from_numpy(features).float()
    else:
        features_tensor = features.detach().float()

    n_samples = int(features_tensor.shape[0])
    model = model.to(device)
    model.eval()
    price_parts: list[np.ndarray] = []
    delta_parts: list[np.ndarray] = []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = features_tensor[start:end].to(device)
        prices, deltas = surrogate_price_and_delta(
            model,
            batch,
            moneyness_range=moneyness_range,
            moneyness_index=moneyness_index,
            create_graph=False,
        )
        price_parts.append(prices.detach().cpu().numpy().reshape(-1))
        delta_parts.append(deltas.detach().cpu().numpy().reshape(-1))

    if not price_parts:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    return (
        np.concatenate(price_parts).astype(np.float64, copy=False),
        np.concatenate(delta_parts).astype(np.float64, copy=False),
    )


def invert_implied_volatility_call(
    prices: np.ndarray,
    moneyness: np.ndarray,
    maturity: np.ndarray,
    rate: np.ndarray,
    inverter: ImpliedVolatilityInverter | None = None,
    initial_guess: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert call prices to Black-Scholes implied volatility, point by point.

    The project convention fixes ``strike = 1.0`` and ``dividend_yield = 0``
    (see ``src/datasets/generator.py``), so the inputs only need to carry
    moneyness, maturity and rate alongside the predicted price ``C/K``.

    Returns ``(iv, ok)`` where ``iv`` is the recovered volatility per point
    and ``ok`` is a boolean mask of successful inversions. Failed inversions
    keep ``iv = NaN`` and ``ok = False``; failure modes include prices
    outside the no-arbitrage band, non-convergence of Newton-Raphson with
    bisection fallback, and any other ``ValueError`` raised by the inverter.
    The methodology document mandates that these failures are reported as
    diagnostic rather than silently dropped.
    """
    prices_arr = np.asarray(prices, dtype=np.float64)
    moneyness_arr = np.asarray(moneyness, dtype=np.float64)
    maturity_arr = np.asarray(maturity, dtype=np.float64)
    rate_arr = np.asarray(rate, dtype=np.float64)

    shape = prices_arr.shape
    if not (
        shape == moneyness_arr.shape == maturity_arr.shape == rate_arr.shape
    ):
        raise ValueError(
            "prices, moneyness, maturity and rate must all share the same shape"
        )

    if inverter is None:
        inverter = ImpliedVolatilityInverter()

    n_samples = int(prices_arr.size)
    iv = np.full(n_samples, np.nan, dtype=np.float64)
    ok = np.zeros(n_samples, dtype=bool)

    flat_prices = prices_arr.reshape(-1)
    flat_moneyness = moneyness_arr.reshape(-1)
    flat_maturity = maturity_arr.reshape(-1)
    flat_rate = rate_arr.reshape(-1)

    for i in range(n_samples):
        if not (
            np.isfinite(flat_prices[i])
            and np.isfinite(flat_moneyness[i])
            and np.isfinite(flat_maturity[i])
            and np.isfinite(flat_rate[i])
        ):
            continue
        try:
            iv[i] = inverter.solve_call(
                price=float(flat_prices[i]),
                spot=float(flat_moneyness[i]),
                strike=1.0,
                maturity=float(flat_maturity[i]),
                rate=float(flat_rate[i]),
                dividend_yield=0.0,
                initial_guess=initial_guess,
            )
            ok[i] = True
        except (ValueError, RuntimeError, FloatingPointError):
            pass

    return iv.reshape(shape), ok.reshape(shape)


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
