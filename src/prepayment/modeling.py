"""Temporal validation, model estimation, and credit-relevant diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, feature_matrix


@dataclass(frozen=True)
class TemporalSplit:
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def choose_temporal_split(frame: pd.DataFrame, train_share: float = 0.60, validation_share: float = 0.20) -> TemporalSplit:
    periods = np.array(sorted(pd.to_datetime(frame["period"].dropna().unique())))
    if len(periods) < 15:
        raise ValueError("At least 15 distinct months are required for a temporal backtest")
    train_idx = max(0, min(len(periods) - 3, int(len(periods) * train_share) - 1))
    valid_idx = max(train_idx + 1, min(len(periods) - 2, int(len(periods) * (train_share + validation_share)) - 1))
    return TemporalSplit(pd.Timestamp(periods[train_idx]), pd.Timestamp(periods[valid_idx]))


def split_frame(frame: pd.DataFrame, split: TemporalSplit) -> dict[str, pd.DataFrame]:
    period = pd.to_datetime(frame["period"])
    parts = {
        "train": frame.loc[period <= split.train_end].copy(),
        "validation": frame.loc[(period > split.train_end) & (period <= split.validation_end)].copy(),
        "test": frame.loc[period > split.validation_end].copy(),
    }
    for name, part in parts.items():
        if part.empty or part["response"].nunique() < 2:
            raise ValueError(f"{name} split is empty or contains only one response class")
    return parts


def _numeric_pipeline(scale: bool) -> Pipeline:
    steps: list[tuple] = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def logistic_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("numeric", _numeric_pipeline(scale=True), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=25, drop="first")),
        ]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(C=0.35, max_iter=1_000, solver="liblinear")),
    ])


def gradient_boosting_pipeline() -> Pipeline:
    preprocess = ColumnTransformer([
        ("numeric", _numeric_pipeline(scale=False), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.065,
            max_iter=160,
            max_leaf_nodes=24,
            min_samples_leaf=80,
            l2_regularization=1.5,
            random_state=42,
        )),
    ])


def score_predictions(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    y_array = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    order = np.argsort(-p)
    top_n = max(1, int(np.ceil(len(p) * 0.10)))
    base_rate = y_array.mean()
    top_rate = y_array[order[:top_n]].mean()
    return {
        "observations": int(len(y_array)),
        "events": int(y_array.sum()),
        "event_rate": float(base_rate),
        "roc_auc": float(roc_auc_score(y_array, p)),
        "pr_auc": float(average_precision_score(y_array, p)),
        "brier_score": float(brier_score_loss(y_array, p)),
        "log_loss": float(log_loss(y_array, p)),
        "top_decile_lift": float(top_rate / base_rate) if base_rate else float("nan"),
    }


def fit_candidates(parts: dict[str, pd.DataFrame]) -> tuple[str, dict[str, Pipeline], dict[str, dict]]:
    x_train, y_train = feature_matrix(parts["train"])
    x_valid, y_valid = feature_matrix(parts["validation"])
    models = {"logistic_regression": logistic_pipeline(), "hist_gradient_boosting": gradient_boosting_pipeline()}
    validation_metrics: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_valid)[:, 1]
        validation_metrics[name] = score_predictions(y_valid, probability)
    selected = max(validation_metrics, key=lambda name: validation_metrics[name]["roc_auc"])
    return selected, models, validation_metrics


def run_backtest(frame: pd.DataFrame, output_dir: str | Path) -> tuple[Pipeline, pd.DataFrame, dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = choose_temporal_split(frame)
    parts = split_frame(frame, split)
    selected, candidates, validation_metrics = fit_candidates(parts)

    development = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    x_development, y_development = feature_matrix(development)
    x_test, y_test = feature_matrix(parts["test"])
    final_model = candidates[selected]
    final_model.fit(x_development, y_development)
    test_probability = final_model.predict_proba(x_test)[:, 1]
    test_metrics = score_predictions(y_test, test_probability)

    prediction_frame = parts["test"][["loan_id", "period", "response", "current_upb"]].copy()
    prediction_frame["probability"] = test_probability
    metrics = {
        "target": "voluntary prepayment in the immediately following month",
        "selected_model": selected,
        "selection_metric": "validation ROC AUC",
        "split": {
            "train_end": split.train_end.strftime("%Y-%m"),
            "validation_end": split.validation_end.strftime("%Y-%m"),
            "test_start": (split.validation_end + pd.offsets.MonthBegin(1)).strftime("%Y-%m"),
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(final_model, output_dir / "selected_model.joblib")
    return final_model, prediction_frame, metrics
