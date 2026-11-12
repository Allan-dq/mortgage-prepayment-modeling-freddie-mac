"""Checks on the MBS valuation: known answers first, then the fitted model."""

import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("QuantLib")

from prepayment import valuation as v  # noqa: E402
from prepayment.features import engineer_features  # noqa: E402
from prepayment.modeling import run_backtest  # noqa: E402
from prepayment.synthetic import make_synthetic_panel  # noqa: E402

SCRIPT = Path(__file__).parents[1] / "scripts" / "business_translation.py"
spec = importlib.util.spec_from_file_location("business_translation", SCRIPT)
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)


def flat_market(monthly_compounded_yield: float) -> v.Market:
    """A flat curve whose one-month discount factor is 1 / (1 + yield / 12)."""
    rate = 12 * math.log(1 + monthly_compounded_yield / 12)
    return v.Market(tenors_years=(0.0, 40.0), zero_rates=(rate, rate))


def rate_sensitive(pool: v.Pool, mortgage_rate: np.ndarray) -> np.ndarray:
    """Toy prepayment rule: 0.5% a month, plus 2% a month per point in the money."""
    incentive = np.clip(pool.wac - mortgage_rate, 0.0, None)
    return np.clip(0.005 + 0.02 * incentive, 0.0, 1.0)


def test_scheduled_factor_runs_from_one_to_zero():
    assert v.scheduled_factor(6.0, 360, np.array([0]))[0] == pytest.approx(1.0)
    assert v.scheduled_factor(6.0, 360, np.array([360]))[0] == pytest.approx(0.0)


def test_cash_flows_without_prepayment_follow_the_amortization_schedule():
    pool = v.Pool(wac=6.0, coupon=6.0, remaining_months=360)
    _, principal = v.project_cash_flows(pool, np.zeros((1, 360)))
    balance = 100 - np.cumsum(principal[0])
    expected = 100 * v.scheduled_factor(6.0, 360, np.arange(1, 361))
    assert balance == pytest.approx(expected, abs=1e-9)


def test_psa_ramp():
    cpr = lambda smm: 1 - (1 - smm) ** 12  # noqa: E731
    smm = v.psa(100)(v.Pool(), np.zeros((1, 40)))[0]
    assert cpr(smm[0]) == pytest.approx(0.002)
    assert cpr(smm[29]) == pytest.approx(0.06)
    assert cpr(smm[39]) == pytest.approx(0.06)
    assert cpr(v.psa(200)(v.Pool(), np.zeros((1, 40)))[0][39]) == pytest.approx(0.12)


@pytest.mark.parametrize("prepayment", [v.constant_cpr(0.0), v.constant_cpr(0.25), v.psa(300)])
def test_pool_prices_at_par_when_the_curve_yield_equals_the_coupon(prepayment):
    # Interest at 5.5% on whatever balance is outstanding, discounted at 5.5%, is worth par
    # however fast the principal comes back. The 6% note rate only sets the amortization.
    pool = v.Pool(wac=6.0, coupon=5.5)
    forward = v.PoolValuer(pool, flat_market(0.055), prepayment).forward()
    assert v.price(forward.cash, forward.scenarios.discount) == pytest.approx(100.0, abs=1e-6)


def test_agrees_with_the_business_translation_script():
    pool = v.Pool(wac=6.0, coupon=6.0)
    forward = v.PoolValuer(pool, flat_market(0.05), v.constant_cpr(0.08)).forward()
    reference = bt.summarize(100.0, 0.06, 360, 0.08, [0.05])
    assert v.price(forward.cash, forward.scenarios.discount) == pytest.approx(
        reference["by_yield"][0.05]["price"], abs=1e-6
    )
    assert v.weighted_average_life(forward.principal)[0] == pytest.approx(
        reference["wal_years"], abs=1e-9
    )


def test_simulated_paths_reprice_the_curve():
    market = v.ILLUSTRATIVE_MARKET
    scenarios = v.simulate_scenarios(market, 360, n_paths=2000, seed=7)
    forward = v.forward_scenario(market, 360)
    # After the correction the average path discount factor is the curve's, at every date.
    assert scenarios.discount.mean(axis=0) == pytest.approx(forward.discount[0], rel=1e-12)
    # The correction itself is small: the raw simulation was already close to the curve.
    assert scenarios.curve_fit < 0.03
    # Every path starts from today's mortgage rate, which is the forward path's first value.
    assert scenarios.mortgage_rate[:, 0] == pytest.approx(forward.mortgage_rate[0, 0])
    assert forward.mortgage_rate[0, 0] == pytest.approx(6.0, abs=0.01)


def test_simulation_is_reproducible_and_shift_moves_the_mortgage_rate():
    market = v.ILLUSTRATIVE_MARKET
    first = v.simulate_scenarios(market, 120, n_paths=200, seed=3)
    again = v.simulate_scenarios(market, 120, n_paths=200, seed=3)
    assert np.array_equal(first.discount, again.discount)
    shifted = v.simulate_scenarios(market, 120, n_paths=200, seed=3, shift=0.01)
    assert shifted.mortgage_rate[0, 0] - first.mortgage_rate[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_cash_flows_that_ignore_rates_carry_no_option_cost():
    valuer = v.PoolValuer(v.Pool(), v.ILLUSTRATIVE_MARKET, v.constant_cpr(0.08), n_paths=400)
    result = valuer.value(100.0)
    assert result.option_cost == pytest.approx(0.0, abs=1e-9)
    assert result.effective_duration == pytest.approx(result.static_duration, rel=1e-6)
    assert result.effective_convexity == pytest.approx(result.static_convexity, rel=1e-4)
    assert result.wal_mean == pytest.approx(result.wal_forward)


def test_oas_reprices_the_pool():
    valuer = v.PoolValuer(v.Pool(), v.ILLUSTRATIVE_MARKET, rate_sensitive, n_paths=400)
    result = valuer.value(99.0)
    assert valuer.price_at(result.oas) == pytest.approx(99.0, abs=1e-8)


def test_rate_sensitive_prepayment_costs_the_investor():
    valuer = v.PoolValuer(v.Pool(), v.ILLUSTRATIVE_MARKET, rate_sensitive, n_paths=1000)
    result = valuer.value(100.0)
    # Borrowers refinance when it hurts the investor, so the option has a cost...
    assert result.option_cost > 0.0010
    # ...the price gains less when rates fall than fixed cash flows would...
    assert result.effective_convexity < 0 < result.static_convexity
    assert result.effective_duration < result.static_duration
    table = valuer.shift_table(result, shifts_bp=(-100, 0, 100)).set_index("shift_bp")
    model_gain = table.loc[-100, "model_price"] - table.loc[0, "model_price"]
    fixed_gain = table.loc[-100, "fixed_cash_flow_price"] - table.loc[0, "fixed_cash_flow_price"]
    assert 0 < model_gain < fixed_gain
    # ...and more rate volatility makes the option more valuable to the borrower.
    calmer = replace(v.ILLUSTRATIVE_MARKET, volatility=0.005)
    calm = v.PoolValuer(v.Pool(), calmer, rate_sensitive, n_paths=1000).value(100.0)
    assert calm.option_cost < result.option_cost


def test_prepayment_in_a_month_uses_the_rate_known_the_month_before():
    seen = {}

    def spy(pool, mortgage_rate):
        seen["rates"] = mortgage_rate
        return np.zeros(mortgage_rate.shape)

    valuer = v.PoolValuer(v.Pool(remaining_months=24), v.ILLUSTRATIVE_MARKET, spy, n_paths=10)
    projection = valuer.simulated()
    assert seen["rates"].shape == (10, 24)
    assert np.array_equal(seen["rates"], projection.scenarios.mortgage_rate[:, :-1])


def test_features_come_from_the_training_feature_code():
    # Flat market rate one point below the note rate: in the money every month.
    features = v.loan_month_features(v.Pool(), np.full((2, 6), 5.0))
    one_path = features.iloc[:6]
    assert list(one_path["refi_incentive"]) == pytest.approx([1.0] * 6)
    assert list(one_path["months_in_money_prior"]) == [0, 1, 2, 3, 4, 5]
    assert list(one_path["loan_age"]) == [0, 1, 2, 3, 4, 5]
    assert one_path["balance_factor"].iloc[0] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def synthetic_fit(tmp_path_factory):
    panel = engineer_features(make_synthetic_panel(3000, 42))
    model, _, _ = run_backtest(panel, tmp_path_factory.mktemp("fit"))
    return model, panel


def test_fitted_model_stays_inside_its_training_range(synthetic_fit):
    model, panel = synthetic_fit
    limits = v.training_limits(panel)
    assert limits.loc["max", "loan_age"] < 360  # the valuation runs past the training data
    flat = np.full((1, 360), 6.0)

    guarded = v.fitted_hazard(model, limits)(v.Pool(), flat)[0]
    assert np.all((guarded >= 0) & (guarded <= 1))
    # Held at the edge of the training range, late-life speeds stay ordinary...
    assert guarded[300] < 0.05
    # ...while the unguarded model extrapolates to an implausible monthly rate.
    unguarded = v.fitted_hazard(model)(v.Pool(), flat)[0]
    assert unguarded[300] > 0.5


def test_fitted_model_gives_a_negatively_convex_pool(synthetic_fit):
    model, panel = synthetic_fit
    prepayment = v.fitted_hazard(model, v.training_limits(panel))
    valuer = v.PoolValuer(v.Pool(), v.ILLUSTRATIVE_MARKET, prepayment, n_paths=400)
    result = valuer.value(100.0)
    assert result.option_cost > 0
    assert result.effective_convexity < 0 < result.static_convexity
    assert result.wal_5th < result.wal_mean < result.wal_95th
