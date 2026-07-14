"""Synthetic loan-month generator for CI and demonstration only.

Synthetic output is not Freddie Mac data and synthetic metrics are not evidence
of real-world model performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_panel(n_loans: int = 3_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2014-01-01", "2025-12-01", freq="MS")
    market_curve = 4.1 + 1.25 * np.sin(np.arange(len(months)) / 15) + np.linspace(0, 1.4, len(months))
    market_curve += rng.normal(0, 0.12, len(months))
    market = dict(zip(months, market_curve))
    rows: list[dict] = []

    for i in range(n_loans):
        start_idx = int(rng.integers(0, 72))
        start = months[start_idx]
        term = int(rng.choice([180, 360], p=[0.18, 0.82]))
        original_rate = float(np.clip(market[start] + rng.normal(0.35, 0.55), 2.25, 9.5))
        original_upb = float(np.exp(rng.normal(np.log(260_000), 0.55)))
        fico = int(np.clip(rng.normal(742, 46), 550, 850))
        ltv = float(np.clip(rng.normal(76, 14), 20, 110))
        dti = float(np.clip(rng.normal(34, 9), 5, 65))
        state = str(rng.choice(["CA", "TX", "FL", "NY", "IL", "WA", "Other"], p=[.18,.14,.11,.10,.08,.07,.32]))
        purpose = str(rng.choice(["P", "C", "N"], p=[.48, .34, .18]))
        occupancy = str(rng.choice(["P", "I", "S"], p=[.87, .08, .05]))
        channel = str(rng.choice(["R", "B", "C", "T"], p=[.36, .26, .24, .14]))
        property_type = str(rng.choice(["SF", "CO", "PU", "MH"], p=[.74, .15, .09, .02]))
        first_time_homebuyer = str(rng.choice(["Y", "N"], p=[.22, .78]))
        monthly_rate = original_rate / 1200
        payment = original_upb * monthly_rate / (1 - (1 + monthly_rate) ** -term)
        balance = original_upb
        months_in_money = 0
        prepaid = False

        max_horizon = min(term, len(months) - start_idx - 1, 120)
        for age in range(max_horizon):
            period = months[start_idx + age]
            market_rate = float(market[period])
            incentive = original_rate - market_rate
            season = -0.0007 * (age - 38) ** 2 / 100
            logit = -5.45 + 0.88 * incentive + 0.55 * max(incentive, 0) + season
            logit += 0.000003 * original_upb + 0.006 * (fico - 700) - 0.012 * max(ltv - 80, 0)
            logit += 0.23 * (purpose in ["C", "N"]) - 0.18 * (occupancy == "I") - 0.018 * months_in_money
            logit += rng.normal(0, 0.18)
            probability = 1 / (1 + np.exp(-logit))
            event_next = bool(rng.random() < probability and age >= 2)
            delinquency = "01" if rng.random() < max(0.002, (700 - fico) / 8000) else "00"
            rows.append({
                "loan_id": f"SYN{i:07d}", "period": period, "current_upb": balance,
                "delinquency_status": delinquency, "loan_age": age,
                "remaining_term": term - age, "zero_balance_code": pd.NA,
                "current_rate": original_rate, "market_rate": market_rate,
                "fico": fico, "original_dti": dti, "original_ltv": ltv,
                "original_cltv": ltv, "original_upb": original_upb,
                "original_rate": original_rate, "occupancy": occupancy,
                "loan_purpose": purpose, "channel": channel,
                "property_type": property_type, "state": state,
                "first_time_homebuyer": first_time_homebuyer,
                "modification_flag": "N", "estimated_ltv": ltv * balance / original_upb,
                "response": int(event_next),
            })
            months_in_money += int(incentive > 0.50)
            if event_next:
                prepaid = True
                break
            interest = balance * monthly_rate
            balance = max(0.0, balance - max(payment - interest, 0))
        _ = prepaid
    return pd.DataFrame(rows)
