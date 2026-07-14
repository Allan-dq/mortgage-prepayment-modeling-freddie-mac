"""Data ingestion and target construction.

The modeling row is an active loan-month at time t. The response equals one
when the same loan terminates with Freddie Mac Zero Balance Code 01 in the
immediately following calendar month. All predictors are observable at t.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .schema import (
    NUMERIC_ORIGINATION,
    NUMERIC_PERFORMANCE,
    ORIGINATION_COLUMNS,
    PERFORMANCE_COLUMNS,
)


def _read_pipe(path: str | Path, columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=columns,
        dtype="string",
        na_values=["", " ", "NA", "NULL"],
        keep_default_na=True,
        low_memory=False,
    )
    if frame.shape[1] != len(columns):
        raise ValueError(f"Unexpected column count in {path}")
    return frame


def read_origination(path: str | Path) -> pd.DataFrame:
    frame = _read_pipe(path, ORIGINATION_COLUMNS)
    for col in NUMERIC_ORIGINATION:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["loan_id"] = frame["loan_id"].str.strip()
    frame["first_payment_date"] = pd.to_datetime(
        frame["first_payment_date"], format="%Y%m", errors="coerce"
    )
    return frame


def read_performance(path: str | Path) -> pd.DataFrame:
    frame = _read_pipe(path, PERFORMANCE_COLUMNS)
    for col in NUMERIC_PERFORMANCE:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["loan_id"] = frame["loan_id"].str.strip()
    frame["period"] = pd.to_datetime(frame["period"], format="%Y%m", errors="coerce")
    frame["zero_balance_code"] = frame["zero_balance_code"].str.zfill(2)
    return frame


def build_next_month_target(performance: pd.DataFrame) -> pd.DataFrame:
    """Create a leakage-safe, one-month voluntary prepayment target.

    Rows without a contiguous next observation are censored and removed; they
    are not silently labeled non-prepayments. The termination row itself is not
    a modeling row because its zero-balance code would reveal the outcome.
    """
    required = {"loan_id", "period", "current_upb", "zero_balance_code"}
    missing = required.difference(performance.columns)
    if missing:
        raise ValueError(f"Missing performance fields: {sorted(missing)}")

    frame = performance.sort_values(["loan_id", "period"]).copy()
    duplicate = frame.duplicated(["loan_id", "period"], keep=False)
    if duplicate.any():
        raise ValueError(f"Duplicate loan-month keys: {int(duplicate.sum())} rows")

    grouped = frame.groupby("loan_id", sort=False)
    frame["next_period"] = grouped["period"].shift(-1)
    frame["next_zero_balance_code"] = grouped["zero_balance_code"].shift(-1)
    expected_next = frame["period"] + pd.offsets.MonthBegin(1)
    frame["has_contiguous_next_month"] = frame["next_period"].eq(expected_next)
    frame["response"] = frame["next_zero_balance_code"].eq("01").fillna(False).astype("int8")

    active = frame["current_upb"].gt(0) & frame["zero_balance_code"].isna()
    eligible = active & frame["has_contiguous_next_month"]
    return frame.loc[eligible].drop(columns=["next_zero_balance_code"]).reset_index(drop=True)


def join_origination(panel: pd.DataFrame, origination: pd.DataFrame) -> pd.DataFrame:
    if origination["loan_id"].duplicated().any():
        raise ValueError("Origination file contains duplicate loan identifiers")
    matched = panel.merge(origination, on="loan_id", how="inner", validate="many_to_one")
    if matched.empty:
        raise ValueError(
            "Origination and performance files have no overlapping loan IDs. "
            "The public Release 47 format samples are schema examples and cannot be modeled together."
        )
    return matched


def add_macro_rates(panel: pd.DataFrame, macro_csv: str | Path | None) -> pd.DataFrame:
    """Join an optional monthly market mortgage rate without future information."""
    frame = panel.copy()
    if macro_csv is None:
        if "market_rate" not in frame:
            frame["market_rate"] = np.nan
        return frame
    macro = pd.read_csv(macro_csv)
    required = {"period", "market_rate"}
    if not required.issubset(macro.columns):
        raise ValueError("Macro CSV must contain period and market_rate")
    macro["period"] = pd.to_datetime(macro["period"])
    macro["market_rate"] = pd.to_numeric(macro["market_rate"], errors="coerce")
    return frame.merge(macro[list(required)], on="period", how="left", validate="many_to_one")
