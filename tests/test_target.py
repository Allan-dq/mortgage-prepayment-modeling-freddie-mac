import pandas as pd
import pytest

from prepayment.data import build_next_month_target, join_origination


def test_target_uses_following_contiguous_month_only():
    performance = pd.DataFrame({
        "loan_id": ["A", "A", "A", "B", "B"],
        "period": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-01-01", "2020-03-01"]),
        "current_upb": [100, 99, 0, 200, 0],
        "zero_balance_code": [pd.NA, pd.NA, "01", pd.NA, "01"],
    })
    panel = build_next_month_target(performance)
    assert panel[["loan_id", "period", "response"]].to_dict("records") == [
        {"loan_id": "A", "period": pd.Timestamp("2020-01-01"), "response": 0},
        {"loan_id": "A", "period": pd.Timestamp("2020-02-01"), "response": 1},
    ]


def test_duplicate_loan_month_is_rejected():
    performance = pd.DataFrame({
        "loan_id": ["A", "A"],
        "period": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "current_upb": [100, 100],
        "zero_balance_code": [pd.NA, pd.NA],
    })
    with pytest.raises(ValueError, match="Duplicate loan-month"):
        build_next_month_target(performance)


def test_nullable_string_codes_do_not_break_target_cast():
    performance = pd.DataFrame({
        "loan_id": pd.Series(["A", "A", "A"], dtype="string[pyarrow]"),
        "period": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        "current_upb": [100, 99, 98],
        "zero_balance_code": pd.Series([pd.NA, pd.NA, pd.NA], dtype="string[pyarrow]"),
    })
    panel = build_next_month_target(performance)
    assert panel["response"].tolist() == [0, 0]


def test_schema_sample_with_no_id_overlap_fails_loudly():
    panel = pd.DataFrame({"loan_id": ["PERF1"]})
    origination = pd.DataFrame({"loan_id": ["ORIG1"]})
    with pytest.raises(ValueError, match="no overlapping loan IDs"):
        join_origination(panel, origination)
