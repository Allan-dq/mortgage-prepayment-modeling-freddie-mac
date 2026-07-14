"""Download an auditable monthly 30-year mortgage-rate proxy from FRED.

Source series: MORTGAGE30US, Freddie Mac 30-Year Fixed Rate Mortgage Average in
the United States, distributed by the Federal Reserve Bank of St. Louis.
The weekly observations are averaged by calendar month. Run this script manually
and retain the dated raw download locally; data/raw is gitignored.
"""

from pathlib import Path
import pandas as pd

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
OUTPUT = Path("data/raw/market_rates.csv")


def main() -> None:
    weekly = pd.read_csv(URL)
    weekly.columns = ["date", "market_rate"]
    weekly["date"] = pd.to_datetime(weekly["date"], errors="raise")
    weekly["market_rate"] = pd.to_numeric(weekly["market_rate"], errors="coerce")
    monthly = (
        weekly.dropna()
        .set_index("date")["market_rate"]
        .resample("MS")
        .mean()
        .rename_axis("period")
        .reset_index()
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT, index=False)
    print(f"Saved {len(monthly):,} monthly observations to {OUTPUT}")


if __name__ == "__main__":
    main()
