"""Inversión a volatilidad implícita Black-Scholes para calls europeas.

El proyecto evalúa los surrogates tanto en la escala de precio como en
la de IV (ver E1). Este inversor se usa solo en evaluación, nunca como
target de entrenamiento, y combina Newton sobre Vega con un respaldo
por bisección para los puntos donde la Vega cae a casi cero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.solvers.black_scholes import BlackScholesSolver


@dataclass(frozen=True)
class ImpliedVolatilityInverter:
    """Recupera la volatilidad implícita Black-Scholes a partir del precio.

    Estrategia: Newton sobre la diferencia ``modelo - precio`` mientras
    la Vega sea estable; en caso contrario, bisección sobre el intervalo
    de búsqueda.

    Attributes:
        solver: Pricer Black-Scholes usado en cada iteración.
        min_volatility: Cota inferior del intervalo de búsqueda.
        max_volatility: Cota superior del intervalo de búsqueda.
        tolerance: Tolerancia absoluta sobre la diferencia de precio.
        max_iterations: Número máximo de iteraciones por método.
    """

    solver: BlackScholesSolver = BlackScholesSolver()
    min_volatility: float = 1e-8
    max_volatility: float = 5.0
    tolerance: float = 1e-10
    max_iterations: int = 100

    def solve_call(
        self,
        price: float,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float = 0.0,
        initial_guess: float = 0.2,
    ) -> float:
        """Devuelve la IV que reproduce ``price`` bajo Black-Scholes.

        Comprueba primero las cotas de no arbitraje y, si el precio está
        en el valor intrínseco, devuelve ``min_volatility``. A partir de
        ``initial_guess`` itera Newton; si la Vega se vuelve degenerada,
        cae a bisección.
        """
        self._validate_inputs(price, spot, strike, maturity, rate, dividend_yield)

        lower = self._lower_bound(spot, strike, maturity, rate, dividend_yield)
        upper = spot * np.exp(-dividend_yield * maturity)
        if price < lower - self.tolerance or price > upper + self.tolerance:
            raise ValueError("price is outside no-arbitrage bounds for a call option")
        if abs(price - lower) <= self.tolerance:
            return self.min_volatility

        sigma = float(np.clip(initial_guess, self.min_volatility, self.max_volatility))
        for _ in range(self.max_iterations):
            model_price = self.solver.call_price(
                spot, strike, maturity, rate, sigma, dividend_yield
            )
            diff = model_price - price
            if abs(diff) <= self.tolerance:
                return sigma

            vega = self.solver.call_vega(spot, strike, maturity, rate, sigma, dividend_yield)
            if vega <= 1e-12:
                break

            next_sigma = sigma - diff / vega
            if not np.isfinite(next_sigma) or not (
                self.min_volatility <= next_sigma <= self.max_volatility
            ):
                break
            sigma = float(next_sigma)

        # Newton no converge o sale del intervalo: respaldo por bisección.
        return self._bisect(price, spot, strike, maturity, rate, dividend_yield)

    def _bisect(
        self,
        price: float,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
    ) -> float:
        """Bisección en ``[min_volatility, max_volatility]``."""
        low = self.min_volatility
        high = self.max_volatility

        high_price = self.solver.call_price(spot, strike, maturity, rate, high, dividend_yield)
        if high_price < price - self.tolerance:
            raise ValueError("price cannot be matched within volatility bounds")

        for _ in range(self.max_iterations):
            mid = 0.5 * (low + high)
            mid_price = self.solver.call_price(spot, strike, maturity, rate, mid, dividend_yield)
            diff = mid_price - price
            if abs(diff) <= self.tolerance:
                return mid
            if diff > 0.0:
                high = mid
            else:
                low = mid
        return 0.5 * (low + high)

    def _validate_inputs(
        self,
        price: float,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
    ) -> None:
        if not (
            np.isfinite(price)
            and np.isfinite(spot)
            and np.isfinite(strike)
            and np.isfinite(maturity)
            and np.isfinite(rate)
            and np.isfinite(dividend_yield)
        ):
            raise ValueError(
                "price, spot, strike, maturity, rate and dividend_yield must be finite"
            )
        if spot <= 0.0:
            raise ValueError("spot must be strictly positive")
        if strike <= 0.0:
            raise ValueError("strike must be strictly positive")
        if maturity <= 0.0:
            raise ValueError("maturity must be strictly positive")
        if price < 0.0:
            raise ValueError("price must be non-negative")

    def _lower_bound(
        self,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend_yield: float,
    ) -> float:
        """Valor intrínseco descontado, suelo teórico del precio de la call."""
        forward_intrinsic = (
            spot * np.exp(-dividend_yield * maturity) - strike * np.exp(-rate * maturity)
        )
        return max(forward_intrinsic, 0.0)
