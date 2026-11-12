"""Monte Carlo valuation of a fixed-rate mortgage pass-through under a prepayment model.

Steps: benchmark zero curve -> Hull-White short-rate paths (QuantLib) -> market mortgage
rate on every path -> monthly prepayment rate from a prepayment model -> investor cash
flows -> price, option-adjusted spread (OAS), zero-volatility spread, effective duration
and convexity.

Conventions
- One period is one month; cash flow k is paid at t = k/12 years, with no payment delay.
- Curve rates and spreads are decimals with continuous compounding.
- Mortgage rates and pool coupons are in percent, as in the modeling code.
- Prices are per 100 of current face.

Simplifications
- One homogeneous pool, represented by a single loan's characteristics.
- The market mortgage rate is the 10-year zero yield plus a constant spread.
- Voluntary prepayment only: no defaults, delinquencies or curtailments.
- The pool's rate history before the valuation date is unknown, so burnout starts at zero.
- A fitted model is never asked to extrapolate: its numeric inputs are held at the edge of
  the range seen in training (see fitted_hazard).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import QuantLib as ql

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features

# The model works in year fractions; the date only anchors the QuantLib curve.
REFERENCE_DATE = ql.Date(15, 1, 2026)
DAY_COUNTER = ql.Actual365Fixed()
BENCHMARK_YEARS = 10.0


@dataclass(frozen=True)
class Market:
    """Benchmark curve, mortgage spread and Hull-White parameters."""

    tenors_years: tuple[float, ...]  # must start at 0 and reach the pool maturity + 10 years
    zero_rates: tuple[float, ...]
    mortgage_spread: float = 1.80  # percentage points over the 10-year zero yield
    mean_reversion: float = 0.03
    volatility: float = 0.01


# Made-up inputs for the demonstration: not market data and not calibrated to anything.
ILLUSTRATIVE_MARKET = Market(
    tenors_years=(0.0, 0.25, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0),
    zero_rates=(0.040, 0.040, 0.039, 0.038, 0.039, 0.042, 0.045, 0.046, 0.046),
)


@dataclass(frozen=True)
class Pool:
    """A level-payment fixed-rate pool and the representative loan behind it."""

    wac: float = 6.0  # note rate paid by borrowers, percent
    coupon: float = 5.5  # pass-through rate paid to investors, percent
    remaining_months: int = 360
    loan_age: int = 0  # months since origination at the valuation date
    fico: float = 740.0
    original_ltv: float = 75.0
    original_dti: float = 34.0
    original_upb: float = 260_000.0
    occupancy: str = "P"
    loan_purpose: str = "P"
    channel: str = "R"
    property_type: str = "SF"
    state: str = "Other"
    first_time_homebuyer: str = "N"


@dataclass(frozen=True)
class RateScenarios:
    discount: np.ndarray  # (paths, months + 1): discount factor from 0 to k/12
    mortgage_rate: np.ndarray  # (paths, months + 1): market mortgage rate at k/12, percent
    curve_fit: float  # largest relative correction applied to reprice the curve


# A prepayment model maps (pool, mortgage rates at months 0..N-1) to the single monthly
# mortality (SMM) applied in months 1..N. The rate known at month k-1 drives month k.
PrepaymentModel = Callable[[Pool, np.ndarray], np.ndarray]


def _curve(market: Market, shift: float) -> ql.YieldTermStructureHandle:
    dates = [REFERENCE_DATE + int(round(365 * t)) for t in market.tenors_years]
    curve = ql.ZeroCurve(dates, [z + shift for z in market.zero_rates], DAY_COUNTER)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


def forward_scenario(market: Market, months: int, shift: float = 0.0) -> RateScenarios:
    """The zero-volatility path: every rate follows today's forward curve."""
    curve = _curve(market, shift)
    times = np.arange(months + 1) / 12
    discount = np.array([curve.discount(float(t), True) for t in times])
    later = np.array([curve.discount(float(t) + BENCHMARK_YEARS, True) for t in times])
    benchmark = -np.log(later / discount) / BENCHMARK_YEARS
    mortgage_rate = 100 * benchmark + market.mortgage_spread
    return RateScenarios(discount[None, :], mortgage_rate[None, :], 0.0)


def simulate_scenarios(
    market: Market, months: int, n_paths: int, seed: int, shift: float = 0.0
) -> RateScenarios:
    """Hull-White short-rate paths in antithetic pairs, rescaled to reprice the curve."""
    if n_paths < 2 or n_paths % 2:
        raise ValueError("n_paths must be a positive even number (antithetic pairs)")
    curve = _curve(market, shift)
    process = ql.HullWhiteProcess(curve, market.mean_reversion, market.volatility)
    uniforms = ql.UniformRandomSequenceGenerator(months, ql.UniformRandomGenerator(seed))
    generator = ql.GaussianPathGenerator(
        process, months / 12, months, ql.GaussianRandomSequenceGenerator(uniforms), False
    )
    short_rate = np.empty((n_paths, months + 1))
    for i in range(0, n_paths, 2):
        short_rate[i] = np.array(generator.next().value())
        short_rate[i + 1] = np.array(generator.antithetic().value())

    # Discount factor along each path: trapezoid rule on the monthly short rates.
    step_average = 0.5 * (short_rate[:, 1:] + short_rate[:, :-1])
    discount = np.exp(-np.cumsum(step_average, axis=1) / 12)
    discount = np.hstack([np.ones((n_paths, 1)), discount])

    # Sampling noise and the monthly time step leave the average path discount factor
    # slightly off the curve. Rescale every date so that the paths reprice it exactly.
    times = np.arange(months + 1) / 12
    target = np.array([curve.discount(float(t), True) for t in times])
    correction = target / discount.mean(axis=0)
    discount = discount * correction

    # In Hull-White a zero-coupon bond price is exp(lnA - B * short rate), so the 10-year
    # zero yield at any future date is a straight line in that date's short rate.
    model = ql.HullWhite(curve, market.mean_reversion, market.volatility)
    benchmark = np.empty_like(short_rate)
    for k, t in enumerate(times):
        log_p0 = np.log(model.discountBond(float(t), float(t) + BENCHMARK_YEARS, 0.0))
        log_p1 = np.log(model.discountBond(float(t), float(t) + BENCHMARK_YEARS, 0.01))
        slope = (log_p0 - log_p1) / 0.01
        benchmark[:, k] = -(log_p0 - slope * short_rate[:, k]) / BENCHMARK_YEARS
    mortgage_rate = 100 * benchmark + market.mortgage_spread
    return RateScenarios(discount, mortgage_rate, float(np.abs(correction - 1).max()))


def constant_cpr(cpr: float) -> PrepaymentModel:
    smm = 1 - (1 - cpr) ** (1 / 12)
    return lambda pool, mortgage_rate: np.full(mortgage_rate.shape, smm)


def psa(speed: float = 100.0) -> PrepaymentModel:
    """PSA benchmark: CPR rises 0.2% a month to 6% at month 30, times speed / 100."""

    def model(pool: Pool, mortgage_rate: np.ndarray) -> np.ndarray:
        age = pool.loan_age + 1 + np.arange(mortgage_rate.shape[1])
        cpr = np.minimum(age / 30, 1.0) * 0.06 * speed / 100
        return np.broadcast_to(1 - (1 - cpr) ** (1 / 12), mortgage_rate.shape).copy()

    return model


def training_limits(panel: pd.DataFrame) -> pd.DataFrame:
    """Smallest and largest value of every numeric feature in a training panel."""
    return panel[NUMERIC_FEATURES].agg(["min", "max"])


def fitted_hazard(pipeline, limits: pd.DataFrame | None = None) -> PrepaymentModel:
    """Use a fitted loan-month hazard model (see modeling.py) as the pool's SMM.

    A 30-year valuation reaches loan ages and rate levels the model never saw in training.
    With `limits` (from training_limits) every numeric input is held at the edge of the
    training range instead of being extrapolated.
    """

    def model(pool: Pool, mortgage_rate: np.ndarray) -> np.ndarray:
        features = loan_month_features(pool, mortgage_rate)
        if limits is not None:
            features[NUMERIC_FEATURES] = features[NUMERIC_FEATURES].clip(
                limits.loc["min"], limits.loc["max"], axis=1
            )
        return pipeline.predict_proba(features)[:, 1].reshape(mortgage_rate.shape)

    return model


def scheduled_factor(wac: float, term: int, age: np.ndarray) -> np.ndarray:
    """Balance of a level-payment loan after `age` payments, as a share of the original."""
    growth = 1 + wac / 1200
    return (growth**term - growth**age) / (growth**term - 1)


def loan_month_features(pool: Pool, mortgage_rate: np.ndarray) -> pd.DataFrame:
    """One representative loan per path, run through the same feature code as training."""
    n_paths, months = mortgage_rate.shape
    age = pool.loan_age + np.arange(months)
    factor = scheduled_factor(pool.wac, pool.loan_age + pool.remaining_months, age)
    panel = pd.DataFrame(
        {
            "loan_id": np.repeat(np.arange(n_paths), months),
            "period": np.tile(np.arange(months), n_paths),
            "market_rate": mortgage_rate.ravel(),
            "current_rate": pool.wac,
            "original_rate": pool.wac,
            "loan_age": np.tile(age, n_paths),
            "current_upb": np.tile(pool.original_upb * factor, n_paths),
            "original_upb": pool.original_upb,
            "estimated_ltv": np.tile(pool.original_ltv * factor, n_paths),
            "delinquency_status": "00",
            "fico": pool.fico,
            "original_dti": pool.original_dti,
            "original_ltv": pool.original_ltv,
            "original_cltv": pool.original_ltv,
            "occupancy": pool.occupancy,
            "loan_purpose": pool.loan_purpose,
            "channel": pool.channel,
            "property_type": pool.property_type,
            "state": pool.state,
            "first_time_homebuyer": pool.first_time_homebuyer,
            "modification_flag": "N",
        }
    )
    return engineer_features(panel)[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def project_cash_flows(pool: Pool, smm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Investor cash flow and principal by path and month, per 100 of current face."""
    n_paths, months = smm.shape
    note, coupon = pool.wac / 1200, pool.coupon / 1200
    balance = np.full(n_paths, 100.0)
    cash, principal = np.empty(smm.shape), np.empty(smm.shape)
    for k in range(months):
        remaining = pool.remaining_months - k
        payment = balance * note / (1 - (1 + note) ** -remaining)
        scheduled = payment - balance * note
        prepaid = smm[:, k] * (balance - scheduled)
        principal[:, k] = scheduled + prepaid
        cash[:, k] = balance * coupon + principal[:, k]
        balance = balance - principal[:, k]
    return cash, principal


def path_values(cash: np.ndarray, discount: np.ndarray, spread: float = 0.0) -> np.ndarray:
    """Present value of each path's cash flows at the path's rates plus a spread."""
    times = np.arange(1, cash.shape[1] + 1) / 12
    return (cash * discount[:, 1:] * np.exp(-spread * times)).sum(axis=1)


def price(cash: np.ndarray, discount: np.ndarray, spread: float = 0.0) -> float:
    return float(path_values(cash, discount, spread).mean())


def solve_spread(cash: np.ndarray, discount: np.ndarray, target_price: float) -> float:
    """The spread at which the average path value equals the target price."""
    return ql.Brent().solve(
        lambda s: price(cash, discount, s) - target_price, 1e-12, 0.0, -0.5, 1.0
    )


def weighted_average_life(principal: np.ndarray) -> np.ndarray:
    times = np.arange(1, principal.shape[1] + 1) / 12
    return (principal * times).sum(axis=1) / principal.sum(axis=1)


def annual_rate(smm: np.ndarray) -> float:
    """Annualized prepayment rate (CPR) over the months given."""
    months = len(smm)
    return float(1 - np.prod(1 - smm) ** (12 / months))


@dataclass(frozen=True)
class Projection:
    cash: np.ndarray  # (paths, months) investor cash flow
    principal: np.ndarray  # (paths, months) principal returned
    smm: np.ndarray  # (paths, months) prepayment rate applied
    scenarios: RateScenarios


@dataclass(frozen=True)
class Valuation:
    price: float
    oas: float
    oas_std_error: float
    zero_vol_spread: float
    option_cost: float  # zero-volatility spread minus OAS
    effective_duration: float
    effective_convexity: float
    static_duration: float  # forward-path cash flows held fixed
    static_convexity: float
    wal_mean: float
    wal_5th: float
    wal_95th: float
    wal_forward: float
    price_std_error: float
    curve_fit: float
    n_paths: int


class PoolValuer:
    """Bundles a pool, a market and a prepayment model so scenarios share one setup."""

    def __init__(
        self, pool: Pool, market: Market, prepayment: PrepaymentModel,
        n_paths: int = 2000, seed: int = 42,
    ) -> None:
        self.pool, self.market, self.prepayment = pool, market, prepayment
        self.n_paths, self.seed = n_paths, seed

    def _project(self, scenarios: RateScenarios) -> Projection:
        smm = self.prepayment(self.pool, scenarios.mortgage_rate[:, :-1])
        cash, principal = project_cash_flows(self.pool, smm)
        return Projection(cash, principal, smm, scenarios)

    def simulated(self, shift: float = 0.0) -> Projection:
        """Projection on Hull-White paths. Every shift reuses the same seed, so the
        difference between two shifts is not sampling noise."""
        return self._project(simulate_scenarios(
            self.market, self.pool.remaining_months, self.n_paths, self.seed, shift
        ))

    def forward(self, shift: float = 0.0) -> Projection:
        """Projection on the single zero-volatility path."""
        return self._project(forward_scenario(self.market, self.pool.remaining_months, shift))

    def price_at(self, oas: float, shift: float = 0.0) -> float:
        """Model price after a parallel curve shift, at a given OAS."""
        projection = self.simulated(shift)
        return price(projection.cash, projection.scenarios.discount, oas)

    def fixed_cash_flow_price(self, cash: np.ndarray, spread: float, shift: float) -> float:
        """Price of cash flows that do not react to rates, after a parallel curve shift."""
        discount = forward_scenario(self.market, self.pool.remaining_months, shift).discount
        return price(cash, discount, spread)

    def value(self, market_price: float, bump: float = 0.0025) -> Valuation:
        base = self.simulated()
        oas = solve_spread(base.cash, base.scenarios.discount, market_price)
        forward = self.forward()
        zero_vol_spread = solve_spread(forward.cash, forward.scenarios.discount, market_price)

        # Effective measures: move the curve, let prepayments respond, hold OAS fixed.
        down, up = self.price_at(oas, -bump), self.price_at(oas, bump)
        # Static measures: same curve moves, but the forward-path cash flows stay fixed.
        static_down = self.fixed_cash_flow_price(forward.cash, zero_vol_spread, -bump)
        static_up = self.fixed_cash_flow_price(forward.cash, zero_vol_spread, bump)

        # Antithetic pairs are independent of each other; paths within a pair are not.
        values = path_values(base.cash, base.scenarios.discount, oas)
        pair_means = values.reshape(-1, 2).mean(axis=1)
        price_std_error = float(pair_means.std(ddof=1) / np.sqrt(len(pair_means)))
        price_per_unit_spread = (
            price(base.cash, base.scenarios.discount, oas + 1e-4) - market_price
        ) / 1e-4
        wal = weighted_average_life(base.principal)
        return Valuation(
            price=market_price,
            oas=oas,
            oas_std_error=price_std_error / abs(price_per_unit_spread),
            zero_vol_spread=zero_vol_spread,
            option_cost=zero_vol_spread - oas,
            effective_duration=(down - up) / (2 * market_price * bump),
            effective_convexity=(down + up - 2 * market_price) / (market_price * bump**2),
            static_duration=(static_down - static_up) / (2 * market_price * bump),
            static_convexity=(static_down + static_up - 2 * market_price)
            / (market_price * bump**2),
            wal_mean=float(wal.mean()),
            wal_5th=float(np.percentile(wal, 5)),
            wal_95th=float(np.percentile(wal, 95)),
            wal_forward=float(weighted_average_life(forward.principal)[0]),
            price_std_error=price_std_error,
            curve_fit=base.scenarios.curve_fit,
            n_paths=self.n_paths,
        )

    def shift_table(self, valuation: Valuation, shifts_bp=(-100, -50, -25, 0, 25, 50, 100)):
        """Prices after parallel curve shifts, with OAS and zero-volatility spread held."""
        base_forward = self.forward()
        rows = []
        for bp in shifts_bp:
            shift = bp / 1e4
            forward = self.forward(shift)
            rows.append({
                "shift_bp": bp,
                "mortgage_rate": float(forward.scenarios.mortgage_rate[0, 0]),
                "model_price": self.price_at(valuation.oas, shift),
                "fixed_cash_flow_price": self.fixed_cash_flow_price(
                    base_forward.cash, valuation.zero_vol_spread, shift
                ),
                "forward_cpr_year2": annual_rate(forward.smm[0, 12:24]),
                "forward_wal": float(weighted_average_life(forward.principal)[0]),
            })
        return pd.DataFrame(rows)

    def speed_table(self, incentives=(-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0)):
        """Prepayment speed if the market mortgage rate stayed at WAC minus the incentive."""
        rows = []
        for incentive in incentives:
            rate_path = np.full((1, self.pool.remaining_months), self.pool.wac - incentive)
            smm = self.prepayment(self.pool, rate_path)[0]
            rows.append({
                "incentive": incentive,
                "cpr_year1": annual_rate(smm[:12]),
                "cpr_year3": annual_rate(smm[24:36]),
                "cpr_year8": annual_rate(smm[84:96]),
            })
        return pd.DataFrame(rows)
