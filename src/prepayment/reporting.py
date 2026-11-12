"""Aggregate visual diagnostics; never writes loan-level predictions."""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from .features import feature_matrix


COLORS = {"navy": "#102A43", "blue": "#2F6BFF", "teal": "#00A6A6", "gold": "#F2B134", "gray": "#6B7C93"}


def _style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 180, "font.size": 10, "axes.titleweight": "bold"})


def create_figures(predictions: pd.DataFrame, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _style()
    y = predictions["response"].to_numpy()
    p = predictions["probability"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, color=COLORS["blue"], lw=2.2, label=f"AUC = {auc(fpr, tpr):.3f}")
    axes[0].plot([0, 1], [0, 1], ls="--", color=COLORS["gray"])
    axes[0].set(title="Out-of-time discrimination", xlabel="False positive rate", ylabel="True positive rate")
    axes[0].legend(frameon=False)
    precision, recall, _ = precision_recall_curve(y, p)
    axes[1].plot(recall, precision, color=COLORS["teal"], lw=2.2)
    axes[1].axhline(y.mean(), ls="--", color=COLORS["gray"], label=f"Base rate = {y.mean():.2%}")
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "discrimination.png", bbox_inches="tight")
    plt.close(fig)

    true_rate, predicted_rate = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.plot(predicted_rate, true_rate, marker="o", color=COLORS["blue"], lw=2)
    bound = max(true_rate.max(initial=0), predicted_rate.max(initial=0)) * 1.08
    ax.plot([0, bound], [0, bound], ls="--", color=COLORS["gray"])
    ax.set(title="Out-of-time calibration", xlabel="Mean predicted probability", ylabel="Observed prepayment rate")
    fig.tight_layout()
    fig.savefig(output / "calibration.png", bbox_inches="tight")
    plt.close(fig)

    scored = predictions.copy()
    scored["score_decile"] = pd.qcut(scored["probability"].rank(method="first"), 10, labels=False) + 1
    lift = scored.groupby("score_decile", observed=True).agg(observed_rate=("response", "mean"), mean_score=("probability", "mean"))
    lift.to_csv(output / "decile_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(lift.index, lift["observed_rate"], color=COLORS["blue"], alpha=.88)
    ax.plot(lift.index, lift["mean_score"], color=COLORS["gold"], marker="o", lw=2, label="Mean prediction")
    ax.set(title="Risk ranking by score decile", xlabel="Score decile (10 = highest)", ylabel="Monthly prepayment rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "decile_lift.png", bbox_inches="tight")
    plt.close(fig)

    monthly = scored.groupby("period").agg(actual=("response", "mean"), predicted=("probability", "mean"), observations=("response", "size"))
    monthly.to_csv(output / "monthly_backtest.csv")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(monthly.index, monthly["actual"], color=COLORS["navy"], lw=1.8, label="Actual")
    ax.plot(monthly.index, monthly["predicted"], color=COLORS["gold"], lw=1.8, label="Predicted")
    ax.set(title="Out-of-time monthly stability", xlabel="Performance month", ylabel="Monthly prepayment rate")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "monthly_stability.png", bbox_inches="tight")
    plt.close(fig)


def create_driver_analysis(model, panel: pd.DataFrame, metrics: dict, output_dir: str | Path) -> None:
    """Write an aggregate driver chart for the fitted model.

    Logistic effects are standardized conditional log-odds coefficients. The
    nonlinear model uses out-of-time permutation AUC loss on a bounded sample.
    Neither measure is causal.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    test_start = pd.to_datetime(metrics["split"]["test_start"])
    test = panel.loc[pd.to_datetime(panel["period"]) >= test_start].copy()
    x_test, y_test = feature_matrix(test)
    model_name = metrics["selected_model"]

    if model_name == "logistic_regression":
        names = model.named_steps["preprocess"].get_feature_names_out()
        values = model.named_steps["model"].coef_[0]
        drivers = pd.DataFrame({"feature": names, "effect": values})
        drivers["feature"] = drivers["feature"].str.replace(r"^(numeric|categorical)__", "", regex=True)
        drivers = drivers.assign(abs_effect=lambda d: d["effect"].abs()).nlargest(18, "abs_effect")
        xlabel = "Standardized conditional log-odds coefficient"
        title = "Largest synthetic logistic effects"
    else:
        sample = x_test.sample(min(5000, len(x_test)), random_state=42)
        sampled_y = y_test.loc[sample.index]
        result = permutation_importance(model, sample, sampled_y, n_repeats=3, scoring="roc_auc", random_state=42)
        drivers = pd.DataFrame({"feature": sample.columns, "effect": result.importances_mean})
        drivers = drivers.assign(abs_effect=lambda d: d["effect"].abs()).nlargest(18, "abs_effect")
        xlabel = "Out-of-time ROC AUC decrease when permuted"
        title = "Largest synthetic permutation effects"

    drivers.drop(columns="abs_effect").to_csv(output / "driver_summary.csv", index=False)
    ordered = drivers.sort_values("effect")
    colors = [COLORS["teal"] if value >= 0 else COLORS["gold"] for value in ordered["effect"]]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(ordered["feature"], ordered["effect"], color=colors)
    ax.axvline(0, color=COLORS["navy"], lw=.8)
    ax.set(title=title, xlabel=xlabel, ylabel="")
    fig.tight_layout()
    fig.savefig(output / "driver_analysis.png", bbox_inches="tight")
    plt.close(fig)

def write_demo_disclaimer(metrics_path: str | Path, report_path: str | Path) -> None:
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    test = metrics["test"]
    selected = metrics["selected_model"].replace("_", " ").title()
    body = f"""# Demonstration results (synthetic data)

> **Important:** these metrics are generated from the repository's synthetic
> data generator. They validate software behavior and illustrate the reporting
> workflow; they are **not estimates of performance on Freddie Mac loans**.

## Demonstration backtest

- Selected model: **{selected}**
- Temporal split: train through {metrics['split']['train_end']}, validation through {metrics['split']['validation_end']}, test from {metrics['split']['test_start']}
- Test observations: {test['observations']:,}
- Test events: {test['events']:,} ({test['event_rate']:.2%})
- ROC AUC: {test['roc_auc']:.3f}
- PR AUC: {test['pr_auc']:.3f}
- Brier score: {test['brier_score']:.4f}
- Top-decile lift: {test['top_decile_lift']:.2f}x

For an empirical study, download licensed SFLLD annual files directly from
Freddie Mac, place them under `data/raw/`, and run `prepayment-model freddie`.
Review the resulting aggregate diagnostics before making any investment use.
"""
    Path(report_path).write_text(body, encoding="utf-8")


def create_valuation_report(valuer, valuation, model_note: str, output_dir: str | Path) -> None:
    """Write the valuation demonstration: one Markdown summary and one figure."""
    output = Path(output_dir)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    pool, market = valuer.pool, valuer.market
    shifts = valuer.shift_table(valuation)
    speeds = valuer.speed_table()
    base = shifts.loc[shifts["shift_bp"] == 0].iloc[0]
    bp = 1e4

    shift_rows = "\n".join(
        f"| {row.shift_bp:+.0f} | {row.mortgage_rate:.2f}% | {row.model_price:.3f} | "
        f"{row.model_price - base.model_price:+.3f} | {row.fixed_cash_flow_price:.3f} | "
        f"{row.fixed_cash_flow_price - base.fixed_cash_flow_price:+.3f} | "
        f"{row.forward_cpr_year2:.1%} | {row.forward_wal:.2f} |"
        for row in shifts.itertuples()
    )
    speed_rows = "\n".join(
        f"| {row.incentive:+.1f} | {row.cpr_year1:.1%} | {row.cpr_year3:.1%} | {row.cpr_year8:.1%} |"
        for row in speeds.itertuples()
    )
    curve_rows = " | ".join(f"{t:g}y: {z:.2%}" for t, z in zip(market.tenors_years, market.zero_rates))
    body = f"""# MBS valuation demonstration (synthetic model, illustrative market)

> **Important:** the prepayment model here is fitted on the repository's synthetic loans, the
> yield curve, mortgage spread and rate volatility are made-up inputs, and the price is an
> input rather than a quote. The numbers show how a prepayment model turns into a price, a
> spread and a hedge ratio. They are **not a valuation of any security**.

## Inputs

- Pool: {pool.wac:.2f}% note rate, {pool.coupon:.2f}% pass-through coupon, {pool.remaining_months} months remaining, loan age {pool.loan_age}; representative loan with FICO {pool.fico:.0f}, LTV {pool.original_ltv:.0f}, DTI {pool.original_dti:.0f}, balance ${pool.original_upb:,.0f}
- Zero curve (continuous compounding): {curve_rows}
- Market mortgage rate: 10-year zero yield + {market.mortgage_spread:.2f} points = {base.mortgage_rate:.2f}% today
- Hull-White short-rate model: mean reversion {market.mean_reversion:.3f}, volatility {market.volatility:.4f}
- Prepayment model: {model_note}
- Monte Carlo: {valuation.n_paths:,} paths in antithetic pairs, seed {valuer.seed}; paths rescaled to reprice the curve (largest correction {valuation.curve_fit:.2%})

## Results at a price of {valuation.price:.3f}

| Measure | Value |
|---|---:|
| Option-adjusted spread (OAS) | {valuation.oas * bp:.1f} bp (standard error about {valuation.oas_std_error * bp:.1f} bp) |
| Zero-volatility spread | {valuation.zero_vol_spread * bp:.1f} bp |
| Option cost (zero-volatility spread minus OAS) | {valuation.option_cost * bp:.1f} bp |
| Effective duration (prepayments respond to rates) | {valuation.effective_duration:.2f} |
| Effective convexity | {valuation.effective_convexity / 100:.2f} |
| Duration if cash flows were fixed | {valuation.static_duration:.2f} |
| Convexity if cash flows were fixed | {valuation.static_convexity / 100:.2f} |
| Weighted-average life on the forward path | {valuation.wal_forward:.2f} years |
| Weighted-average life across paths: mean (5th to 95th percentile) | {valuation.wal_mean:.2f} years ({valuation.wal_5th:.2f} to {valuation.wal_95th:.2f}) |

Durations and convexities come from moving the whole curve by 25 bp each way; convexity is
`(P_down + P_up - 2 P) / (P * shift^2) / 100`.

## Parallel curve shifts

Model price holds the OAS fixed and lets prepayments respond. The fixed-cash-flow price
keeps today's forward-path cash flows and holds the zero-volatility spread fixed.

| Shift (bp) | Mortgage rate today | Model price | Change | Fixed-cash-flow price | Change | Forward-path CPR, year 2 | Forward-path WAL (years) |
|---:|---:|---:|---:|---:|---:|---:|---:|
{shift_rows}

## What the prepayment model does

Prepayment speed (CPR) if the market mortgage rate stayed at the note rate minus the
incentive for the whole life of the pool:

| Incentive (points) | CPR in year 1 | CPR in year 3 | CPR in year 8 |
|---:|---:|---:|---:|
{speed_rows}
"""
    (output / "VALUATION_RESULTS.md").write_text(body, encoding="utf-8")
    shifts.to_csv(output / "figures" / "curve_shifts.csv", index=False)

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(shifts["shift_bp"], shifts["fixed_cash_flow_price"], marker="o", lw=2,
                 color=COLORS["gray"], label="Cash flows fixed")
    axes[0].plot(shifts["shift_bp"], shifts["model_price"], marker="o", lw=2.2,
                 color=COLORS["blue"], label="Prepayments respond to rates")
    axes[0].set(title="Price after a parallel curve shift", xlabel="Shift (bp)", ylabel="Price per 100")
    axes[0].legend(frameon=False)
    months = np.arange(1, 121)
    for shift_bp, color in ((-100, COLORS["gold"]), (0, COLORS["navy"]), (100, COLORS["teal"])):
        smm = valuer.forward(shift_bp / 1e4).smm[0, :120]
        axes[1].plot(months, 100 * (1 - (1 - smm) ** 12), lw=2, color=color, label=f"{shift_bp:+d} bp")
    axes[1].set(title="Projected speed on the forward path", xlabel="Month", ylabel="CPR (%)")
    axes[1].legend(frameon=False, title="Curve shift")
    fig.tight_layout()
    fig.savefig(output / "figures" / "price_and_speed.png", bbox_inches="tight")
    plt.close(fig)
