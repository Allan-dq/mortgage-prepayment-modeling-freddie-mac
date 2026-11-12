"""Command-line interface for synthetic demonstration and licensed SFLLD runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .data import add_macro_rates, build_next_month_target, join_origination, read_origination, read_performance
from .features import engineer_features
from .modeling import run_backtest
from .reporting import (
    create_driver_analysis,
    create_figures,
    create_valuation_report,
    write_demo_disclaimer,
)
from .synthetic import make_synthetic_panel


def run_demo(args: argparse.Namespace) -> None:
    root = Path(args.output)
    panel = engineer_features(make_synthetic_panel(args.loans, args.seed))
    model, predictions, metrics = run_backtest(panel, root / "private_artifacts")
    create_figures(predictions, root / "figures")
    create_driver_analysis(model, panel, metrics, root / "figures")
    write_demo_disclaimer(root / "private_artifacts" / "metrics.json", root / "DEMO_RESULTS.md")
    print(f"Synthetic demonstration completed: {root.resolve()}")


def run_freddie(args: argparse.Namespace) -> None:
    root = Path(args.output)
    origination = read_origination(args.origination)
    performance = read_performance(args.performance)
    panel = build_next_month_target(performance)
    panel = join_origination(panel, origination)
    panel = add_macro_rates(panel, args.macro)
    panel = engineer_features(panel)
    model, predictions, metrics = run_backtest(panel, root / "private_artifacts")
    create_figures(predictions, root / "figures")
    create_driver_analysis(model, panel, metrics, root / "figures")
    print(f"Licensed-data run completed. Selected model: {metrics['selected_model']}")
    print("Loan-level predictions remain under the ignored private_artifacts directory.")


def run_value(args: argparse.Namespace) -> None:
    # Imported here so the modeling commands work without QuantLib installed.
    from dataclasses import replace

    from . import valuation as v

    root = Path(args.output)
    panel = engineer_features(make_synthetic_panel(args.loans, args.seed))
    model, _, metrics = run_backtest(panel, root / "private_artifacts")
    prepayment = v.fitted_hazard(model, v.training_limits(panel))
    pool = v.Pool(wac=args.wac, coupon=args.coupon, remaining_months=args.months, loan_age=args.age)
    market = replace(
        v.ILLUSTRATIVE_MARKET,
        mortgage_spread=args.mortgage_spread,
        mean_reversion=args.mean_reversion,
        volatility=args.volatility,
    )
    valuer = v.PoolValuer(pool, market, prepayment, n_paths=args.paths, seed=args.seed)
    valuation = valuer.value(args.price)
    selected = metrics["selected_model"].replace("_", " ")
    note = (
        f"{selected} fitted on {args.loans:,} synthetic loans (seed {args.seed}); "
        "numeric inputs held within the range seen in training"
    )
    create_valuation_report(valuer, valuation, note, root)
    print(f"OAS {valuation.oas * 1e4:.1f} bp, zero-volatility spread "
          f"{valuation.zero_vol_spread * 1e4:.1f} bp, effective duration "
          f"{valuation.effective_duration:.2f} at a price of {valuation.price:.3f}")
    print(f"Valuation demonstration completed: {root.resolve()}")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run the reproducible synthetic-data demonstration")
    demo.add_argument("--loans", type=int, default=3000)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--output", default="reports/demo")
    demo.set_defaults(func=run_demo)
    freddie = sub.add_parser("freddie", help="Run with locally downloaded Freddie Mac Release 47 files")
    freddie.add_argument("--origination", required=True)
    freddie.add_argument("--performance", required=True)
    freddie.add_argument("--macro", required=True, help="Monthly CSV with period and market_rate")
    freddie.add_argument("--output", default="reports/freddie")
    freddie.set_defaults(func=run_freddie)
    value = sub.add_parser("value", help="Value an illustrative pool with the synthetic-data model")
    value.add_argument("--price", type=float, default=100.0, help="Price per 100 of face to solve the OAS for")
    value.add_argument("--wac", type=float, default=6.0, help="Note rate, percent")
    value.add_argument("--coupon", type=float, default=5.5, help="Pass-through rate, percent")
    value.add_argument("--months", type=int, default=360, help="Remaining term")
    value.add_argument("--age", type=int, default=0, help="Loan age in months")
    value.add_argument("--mortgage-spread", type=float, default=1.80)
    value.add_argument("--mean-reversion", type=float, default=0.03)
    value.add_argument("--volatility", type=float, default=0.01)
    value.add_argument("--paths", type=int, default=4000)
    value.add_argument("--loans", type=int, default=3000)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--output", default="reports/valuation")
    value.set_defaults(func=run_value)
    return command


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
