# Reproducibility guide

## Synthetic smoke test

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
python -m pytest
prepayment-model demo --loans 3000 --seed 42 --output reports/demo
```

The seed controls data generation and the estimator random state is fixed.
Library versions are bounded in `requirements.txt`; a production study should
also commit a generated lock file and record hardware/runtime metadata.

## Empirical SFLLD run

1. Register and sign in to Freddie Mac Clarity.
2. Review and accept the terms yourself.
3. Download matching Release 47 Standard Dataset origination and performance files.
4. Keep all loan-level files in `data/raw/`; that tree is gitignored.
5. Run `python scripts/download_fred_mortgage_rate.py` or prepare an auditable
   monthly macro CSV with columns `period,market_rate`.
6. Run:

```bash
prepayment-model freddie \
  --origination data/raw/orig_2022Q1.txt \
  --performance data/raw/perf_2022Q1.txt \
  --macro data/raw/market_rates.csv \
  --output reports/freddie
```

The Release 47 website sample files are not an empirical sample: their
origination and performance identifiers do not overlap. The loader rejects that
case explicitly.
