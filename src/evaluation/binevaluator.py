"""Per-bin evaluation orchestrator for trained surrogates.

:class:`BinEvaluator` is the piece that glues together the three other
modules under :mod:`src.evaluation`:

* :class:`src.evaluation.binning.BinPartition` decides which bin each test
  point belongs to.
* The free functions in :mod:`src.evaluation.metrics` compute pointwise
  absolute errors, per-bin aggregates, surrogate outputs and the BS
  implied-volatility inversion.
* :class:`src.evaluation.report.Report` packages the resulting numbers
  and knows how to serialize them to CSV.

Given a trained surrogate, an :class:`src.datasets.generator.OptionDataset`
and the standard 5x5 partition, :meth:`BinEvaluator.evaluate` returns a
fully populated :class:`Report`. This matches the contract described in
``docs/architecture.md`` §"Evaluation": the evaluator receives the
partition and an ``OptionPricer`` by injection so the same code path
serves Black-Scholes and Heston surrogates without branching.

In this version the injected ``pricer`` is not invoked: the reference
prices and deltas come from the dataset itself (already produced by the
solver during dataset generation). The field is kept for forward
compatibility with the timing benchmark planned for E4 and to honour the
constructor signature documented in ``architecture.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from torch import nn

from src.datasets.generator import OptionDataset
from src.evaluation.binning import BinPartition
from src.evaluation.metrics import (
    absolute_errors,
    aggregate_by_bin,
    invert_implied_volatility_call,
    predict_surrogate_prices_and_deltas,
)
from src.evaluation.report import Report
from src.solvers.black_scholes import BlackScholesSolver
from src.solvers.heston import HestonSolver
from src.solvers.iv import ImpliedVolatilityInverter


OptionPricer = Union[BlackScholesSolver, HestonSolver]


@dataclass(frozen=True)
class BinEvaluator:
    """Evaluate a surrogate over a test set and return a :class:`Report`.

    The evaluator is configured once and can be reused across surrogates of
    the same family; calling :meth:`evaluate` does not mutate any field of
    the evaluator. The frozen dataclass mirrors the convention used by
    every other component in :mod:`src.evaluation`.
    """

    partition: BinPartition
    pricer: OptionPricer
    iv_inverter: ImpliedVolatilityInverter | None = None
    device: str = "auto"
    batch_size: int = 32768
    moneyness_range: tuple[float, float] = (0.4, 2.0)
    moneyness_index: int = 0
    iv_workers: int = 1
    iv_progress: bool = False

    def evaluate(
        self,
        surrogate: nn.Module,
        dataset: OptionDataset,
        bin_id: np.ndarray | None = None,
        compute_iv: bool = True,
        surrogate_id: str = "",
        test_path: str = "",
    ) -> Report:
        """Run the full per-bin evaluation pipeline.

        ``bin_id`` may be provided when the test set already carries it
        (the ``.npz`` files produced by ``BalancedBinSampler`` do). When it
        is ``None`` the partition computes it on the fly from ``raw_inputs``.

        ``compute_iv=False`` short-circuits the implied-volatility inversion,
        which is by far the slowest stage of the pipeline (it loops over the
        test set point by point through the scalar inverter). The resulting
        report will carry ``iv = None`` and ``iv_failure_rate_per_bin = None``.
        """
        device = self._resolve_device()
        inverter = self.iv_inverter if self.iv_inverter is not None else ImpliedVolatilityInverter()

        raw_inputs = self._raw_inputs_as_numpy(dataset)
        moneyness = raw_inputs[:, 0]
        maturity = raw_inputs[:, 1]
        rate = raw_inputs[:, 2]

        bin_id_array = self._resolve_bin_id(bin_id, moneyness, maturity)
        n_samples = int(dataset.features.shape[0])

        pred_prices, pred_deltas = predict_surrogate_prices_and_deltas(
            surrogate,
            dataset.features,
            batch_size=self.batch_size,
            device=device,
            moneyness_range=self.moneyness_range,
            moneyness_index=self.moneyness_index,
        )

        target_prices = dataset.prices.detach().cpu().numpy().reshape(-1).astype(np.float64)
        price_err = absolute_errors(pred_prices, target_prices)
        price_agg = aggregate_by_bin(price_err, bin_id_array, self.partition.n_bins)

        delta_agg: dict[str, np.ndarray] | None = None
        if dataset.deltas is not None:
            target_deltas = dataset.deltas.detach().cpu().numpy().reshape(-1).astype(np.float64)
            delta_err = absolute_errors(pred_deltas, target_deltas)
            delta_agg = aggregate_by_bin(delta_err, bin_id_array, self.partition.n_bins)

        iv_agg: dict[str, np.ndarray] | None = None
        iv_failure_rate_per_bin: np.ndarray | None = None
        if compute_iv:
            pred_iv, ok_pred = invert_implied_volatility_call(
                prices=pred_prices,
                moneyness=moneyness,
                maturity=maturity,
                rate=rate,
                inverter=inverter,
                workers=self.iv_workers,
                progress=self.iv_progress,
            )
            target_iv, ok_target = invert_implied_volatility_call(
                prices=target_prices,
                moneyness=moneyness,
                maturity=maturity,
                rate=rate,
                inverter=inverter,
                workers=self.iv_workers,
                progress=self.iv_progress,
            )
            ok_both = ok_pred & ok_target
            iv_err = np.full(n_samples, np.nan, dtype=np.float64)
            iv_err[ok_both] = np.abs(pred_iv[ok_both] - target_iv[ok_both])
            iv_agg = aggregate_by_bin(iv_err, bin_id_array, self.partition.n_bins)
            iv_failure_rate_per_bin = self._failure_rate_per_bin(
                ok_both, bin_id_array, self.partition.n_bins
            )

        return Report(
            surrogate_id=surrogate_id,
            test_path=test_path,
            n_samples=n_samples,
            partition=self.partition,
            price=price_agg,
            delta=delta_agg,
            iv=iv_agg,
            iv_failure_rate_per_bin=iv_failure_rate_per_bin,
        )

    def _resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @staticmethod
    def _raw_inputs_as_numpy(dataset: OptionDataset) -> np.ndarray:
        raw_inputs = dataset.raw_inputs.detach().cpu().numpy()
        if raw_inputs.ndim != 2 or raw_inputs.shape[1] < 3:
            raise ValueError(
                "dataset.raw_inputs must have shape (N, >=3) carrying moneyness, "
                f"maturity and rate as the first three columns; got shape {raw_inputs.shape}"
            )
        return raw_inputs.astype(np.float64, copy=False)

    def _resolve_bin_id(
        self,
        bin_id: np.ndarray | None,
        moneyness: np.ndarray,
        maturity: np.ndarray,
    ) -> np.ndarray:
        if bin_id is None:
            assigned, _, _ = self.partition.assign(moneyness, maturity)
            return assigned
        bin_id_array = np.asarray(bin_id, dtype=np.int64)
        if bin_id_array.shape != moneyness.shape:
            raise ValueError(
                f"bin_id shape {bin_id_array.shape} does not match the dataset "
                f"({moneyness.shape[0]} points)"
            )
        if bin_id_array.size > 0 and (
            bin_id_array.min() < 0 or bin_id_array.max() >= self.partition.n_bins
        ):
            raise ValueError(
                f"bin_id values must lie in [0, {self.partition.n_bins})"
            )
        return bin_id_array

    @staticmethod
    def _failure_rate_per_bin(
        ok_mask: np.ndarray,
        bin_id: np.ndarray,
        n_bins: int,
    ) -> np.ndarray:
        rate = np.full(n_bins, np.nan, dtype=np.float64)
        for k in range(n_bins):
            mask = bin_id == k
            n_in_bin = int(mask.sum())
            if n_in_bin == 0:
                continue
            n_failed = int((~ok_mask[mask]).sum())
            rate[k] = n_failed / n_in_bin
        return rate
