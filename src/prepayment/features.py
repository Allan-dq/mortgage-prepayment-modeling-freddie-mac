"""Economically motivated feature engineering using information available at t."""

from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "fico", "original_dti", "original_ltv", "original_cltv", "original_upb",
    "original_rate", "current_rate", "market_rate", "refi_incentive",
    "positive_incentive", "loan_age", "age_squared", "balance_factor",
    "estimated_ltv", "months_in_money_prior", "max_incentive_prior",
    "prior_delinquent_months", "rate_volatility_6m",
]

CATEGORICAL_FEATURES = [
    "occupancy", "loan_purpose", "channel", "property_type", "state",
    "first_time_homebuyer", "modification_flag", "delinquency_bucket",
]


def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    frame = panel.sort_values(["loan_id", "period"]).copy()
    grouped = frame.groupby("loan_id", sort=False)

    frame["refi_incentive"] = frame["current_rate"] - frame["market_rate"]
    frame["positive_incentive"] = frame["refi_incentive"].clip(lower=0)
    frame["age_squared"] = frame["loan_age"].clip(upper=360).pow(2) / 10_000
    frame["balance_factor"] = frame["current_upb"] / frame["original_upb"].replace(0, np.nan)

    # Burnout features are shifted so the current row cannot influence its own history.
    in_money = frame["refi_incentive"].gt(0.50).astype(int)
    frame["months_in_money_prior"] = (
        in_money.groupby(frame["loan_id"]).cumsum().groupby(frame["loan_id"]).shift(1).fillna(0)
    )
    frame["max_incentive_prior"] = (
        frame["refi_incentive"].groupby(frame["loan_id"]).cummax()
        .groupby(frame["loan_id"]).shift(1)
    )

    delinquent = ~frame["delinquency_status"].astype("string").fillna("XX").isin(["0", "00", "XX"])
    frame["prior_delinquent_months"] = (
        delinquent.astype(int).groupby(frame["loan_id"]).cumsum()
        .groupby(frame["loan_id"]).shift(1).fillna(0)
    )
    status = frame["delinquency_status"].astype("string")
    frame["delinquency_bucket"] = np.select(
        [status.isin(["0", "00"]), status.isin(["1", "01"]), status.isin(["2", "02"])],
        ["current", "30dpd", "60dpd"],
        default="90plus_or_other",
    )
    frame["rate_volatility_6m"] = (
        frame.groupby("loan_id")["market_rate"]
        .transform(lambda s: s.rolling(6, min_periods=3).std())
    )

    for col in NUMERIC_FEATURES:
        if col not in frame:
            frame[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        if col not in frame:
            frame[col] = "missing"
    return frame


def feature_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return frame[columns].copy(), frame["response"].astype(int)
