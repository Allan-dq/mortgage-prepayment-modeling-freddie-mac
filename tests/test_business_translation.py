"""Sanity checks for the cash-flow arithmetic behind the README's business translation."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "business_translation.py"
spec = importlib.util.spec_from_file_location("business_translation", SCRIPT)
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)


def test_pool_prices_at_par_when_yield_equals_coupon():
    flows = bt.cash_flows(100.0, 0.06, 360, 0.08)
    assert bt.analytics(flows, 100.0, 0.06)["price"] == pytest.approx(100.0, abs=1e-9)


def test_principal_is_fully_repaid_and_level_payment_wal_matches_textbook():
    flows = bt.cash_flows(100.0, 0.06, 360, 0.0)
    assert sum(p for _, p in flows) == pytest.approx(100.0)
    assert bt.summarize(100.0, 0.06, 360, 0.0, [])["wal_years"] == pytest.approx(
        19.31, abs=0.01
    )


def test_faster_prepayment_shortens_life_and_hurts_premium_holders():
    slow = bt.summarize(100.0, 0.06, 360, 0.08, [0.05, 0.07])
    fast = bt.summarize(100.0, 0.06, 360, 0.09, [0.05, 0.07])
    assert fast["wal_years"] < slow["wal_years"]
    assert fast["by_yield"][0.05]["price"] < slow["by_yield"][0.05]["price"]
    assert fast["by_yield"][0.07]["price"] > slow["by_yield"][0.07]["price"]
