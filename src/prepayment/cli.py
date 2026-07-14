"""Command-line interface for synthetic demonstration and licensed SFLLD runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .data import add_macro_rates, build_next_month_target, join_origination, read_origination, read_performance
from .features import engineer_features
from .modeling import run_backtest
from .reporting import create_driver_analysis, create_figures, write_demo_disclaimer
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
    return command


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
