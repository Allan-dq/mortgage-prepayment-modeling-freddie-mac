import pandas as pd

from prepayment.modeling import choose_temporal_split, split_frame


def test_temporal_split_has_no_calendar_overlap():
    months = pd.date_range("2020-01-01", periods=30, freq="MS")
    frame = pd.DataFrame({
        "period": months.repeat(2),
        "response": [0, 1] * len(months),
    })
    split = choose_temporal_split(frame)
    parts = split_frame(frame, split)
    assert parts["train"]["period"].max() < parts["validation"]["period"].min()
    assert parts["validation"]["period"].max() < parts["test"]["period"].min()
