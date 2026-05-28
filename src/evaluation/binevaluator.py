"""Orquestador de evaluación por bins para surrogates entrenados.

``BinEvaluator`` conecta la partición de bins, las métricas y el ``Report``.
Dado un surrogate entrenado, un ``OptionDataset`` y la partición 5x5, devuelve
un informe completo con errores de precio, Delta e IV cuando aplica.

El ``pricer`` se inyecta para mantener el contrato de arquitectura y para
compatibilidad con E4. En esta ruta no se usa directamente: los precios y
Deltas de referencia ya vienen calculados en el dataset.
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
    """Evalúa un surrogate sobre un test set y devuelve un ``Report``.

    La instancia se configura una vez y puede reutilizarse con surrogates de
    la misma familia. ``evaluate`` no muta el estado interno.
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
        """Ejecuta la evaluación completa por bins.

        ``bin_id`` puede venir del propio test set, como ocurre con los
        ``.npz`` generados por ``BalancedBinSampler``. Si no se proporciona,
        la partición lo calcula desde ``raw_inputs``.

        ``compute_iv=False`` evita la inversión de IV, que es la fase más
        lenta. En ese caso el informe deja ``iv`` e ``iv_failure_rate`` como
        ``None``.
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
