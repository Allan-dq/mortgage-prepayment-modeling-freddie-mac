# Mortgage Prepayment Modeling with Public Freddie Mac Data

[![CI](https://github.com/Allan-dq/mortgage-prepayment-modeling-freddie-mac/actions/workflows/ci.yml/badge.svg)](https://github.com/Allan-dq/mortgage-prepayment-modeling-freddie-mac/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT (code only)](https://img.shields.io/badge/Code%20License-MIT-2EA44F.svg)](LICENSE)
[![Data: Freddie Mac SFLLD](https://img.shields.io/badge/Data-Freddie%20Mac%20SFLLD-102A43.svg)](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)
[![No raw data](https://img.shields.io/badge/Repository-No%20Loan--Level%20Data-F2B134.svg)](DATA_LICENSE.md)

An investment-oriented, leakage-aware framework for estimating the monthly
voluntary prepayment hazard of U.S. fixed-rate mortgages. The project separates
a transparent regularized logistic benchmark from a nonlinear gradient-boosting
challenger, validates them through calendar time, and emphasizes probability
calibration—not accuracy alone.

> **Project boundary:** this is an independent project built from public Freddie Mac
> documentation and user-obtained SFLLD files. It contains no proprietary data, code,
> results, or logic.

## At a glance

| | |
|---|---|
| Question | Will a performing fixed-rate loan pay off voluntarily next month (Freddie Mac zero-balance code `01`)? |
| Models | Regularized logistic benchmark vs. histogram gradient-boosting challenger |
| Validation | Non-overlapping calendar blocks (train / validate / test); ROC AUC, PR AUC, Brier score, calibration, decile lift |
| What +1pp CPR means | On an illustrative $100M 6% pool: life shortens 0.68 years; a premium pool (5% yield) loses 0.39 points, a discount pool (7% yield) gains 0.32 ([business translation](#business-translation-what-one-point-of-prepayment-speed-is-worth)) |
| MBS valuation | The model drives a QuantLib Hull-White Monte Carlo valuation of an illustrative 5.50% pass-through: OAS 58.5 bp vs. 136.9 bp zero-volatility spread (78 bp option cost); effective duration 3.67 vs. 5.22 with cash flows held fixed. Synthetic model and made-up market inputs ([details](#from-model-to-price-mbs-valuation-with-quantlib)) |
| Freddie Mac results | Not in this repository: SFLLD loan-level data are licensed and not redistributed. Run `prepayment-model freddie` on your own files ([steps](#running-on-locally-licensed-freddie-mac-files)) |
| Run it now | `pip install -e . && pytest && prepayment-model demo --loans 3000 --seed 42 --output reports/demo` uses synthetic data, no download needed ([details](#quick-start-fully-reproducible-synthetic-demonstration)) |

## Executive summary

Prepayment is an embedded borrower option. When prevailing mortgage rates fall
below a borrower's note rate, refinancing becomes valuable; home sales, equity,
seasoning, transaction costs, underwriting eligibility, and prior missed
opportunities (“burnout”) determine whether that option is exercised. For a
mortgage investor, the resulting negative convexity shortens assets precisely
when reinvestment yields are lower.

This repository estimates a one-month discrete-time hazard:

\[
P(Y_{i,t+1}=1 \mid X_{i,t}),\quad
Y_{i,t+1}=1\;\text{if Zero Balance Code}_{i,t+1}=01.
\]

Freddie Mac defines code `01` as *Prepaid or Matured (Voluntary Payoff)*. A row
is eligible only when the loan is active at `t` and has a contiguous `t+1`
record. The termination row is never exposed to the model.

## What this project demonstrates

- defensible loan-month target engineering with explicit censoring;
- strict no-leakage construction of burnout and payment-history features;
- calendar-time train/validation/test backtesting;
- interpretable logistic and nonlinear gradient-boosting models;
- ROC AUC, PR AUC, log loss, Brier score, calibration, lift, and time stability;
- reproducible command-line execution, unit tests, and a model card;
- data-license-aware publishing: no raw/processed loans or loan-level predictions.

## Repository map

```text
.
├── src/prepayment/
│   ├── data.py          # Release 47 parsing, integrity checks, target
│   ├── features.py      # incentive, seasoning, burnout, credit features
│   ├── modeling.py      # temporal split, two models, metrics
│   ├── reporting.py     # aggregate diagnostics
│   ├── valuation.py     # QuantLib rate paths, cash flows, OAS, duration
│   ├── schema.py        # July 2026 Release 47 schemas
│   └── synthetic.py     # deterministic CI/demo generator
├── scripts/             # FRED rate download, demo runners, business translation
├── tests/               # target and temporal-split invariants
├── docs/                # methodology, data dictionary, reproducibility
├── reports/demo/        # committed aggregate synthetic-demo outputs
├── data/                # all loan-level contents ignored by git
├── DATA_LICENSE.md      # data use and redistribution boundary
└── pyproject.toml
```

## Data source and responsible use

The intended empirical source is Freddie Mac's **Single-Family Loan-Level
Dataset (SFLLD)**. Freddie Mac reports that the Standard and Non-Standard data
cover roughly 55 million mortgages originated from 1999 through 2025, with
monthly performance through September 2025 as of the July 2026 publication.
Access requires registration and sign-in to Clarity and is subject to Freddie
Mac's terms.

This repository intentionally does **not** redistribute:

- raw Freddie Mac files, including website format samples;
- cleaned or sampled loan-month records;
- loan identifiers or loan-level model predictions;
- serialized models trained on Freddie Mac data.

The July 2026 Release 47 website sample is useful for checking the new schema,
but its origination and performance loan identifiers do not overlap. It is not a
valid empirical modeling sample. The code fails loudly if those files are paired.
See [data access and license notes](DATA_LICENSE.md).

## Quick start: fully reproducible synthetic demonstration

The synthetic mode tests the complete workflow without restricted data. Its
metrics are software demonstrations—not claims about Freddie Mac loans.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
python -m pytest
prepayment-model demo --loans 3000 --seed 42 --output reports/demo
```

![Synthetic out-of-time discrimination](reports/demo/figures/discrimination.png)

<p align="center">
  <img src="reports/demo/figures/calibration.png" width="48%" alt="Synthetic calibration" />
  <img src="reports/demo/figures/decile_lift.png" width="48%" alt="Synthetic decile lift" />
</p>

![Synthetic conditional driver analysis](reports/demo/figures/driver_analysis.png)

Driver signs are conditional associations, not causal effects. In logistic mode,
numeric features are standardized before estimation, so their coefficient sizes
are comparable within the fitted specification; categorical effects are relative
to the encoded reference structure. In nonlinear mode, the report uses
out-of-time permutation AUC loss instead.

The generated [demo results](reports/demo/DEMO_RESULTS.md) are clearly labeled
synthetic and are included so a reviewer can inspect expected outputs before
obtaining SFLLD access.

## Running on locally licensed Freddie Mac files

1. Open the [official SFLLD page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).
2. Register/sign in to Clarity and review the terms yourself.
3. Download matching Release 47 origination and monthly performance files.
4. Place them under `data/raw/` (already gitignored).
5. Create the monthly mortgage-rate file with
   `python scripts/download_fred_mortgage_rate.py`, or supply an auditable
   alternative CSV with `period,market_rate`.
6. Run:

```bash
prepayment-model freddie \
  --origination data/raw/orig_2022Q1.txt \
  --performance data/raw/perf_2022Q1.txt \
  --macro data/raw/market_rates.csv \
  --output reports/freddie
```

Current parsing targets the enhanced Release 47 layout effective July 2026.
Earlier SFLLD releases use a different layout and should be converted explicitly,
not positionally guessed.

## Modeling design

### Target and risk set

At month `t`, the model sees an active loan with positive UPB and no termination
code. It receives a positive response only when the same loan's immediately next
calendar-month record carries code `01`. Noncontiguous records are removed from
the risk set because their next-month outcome is unobserved. This prevents the
common error of labeling right-censored records as negatives.

### Economically motivated features

| Theme | Representative fields | Investment interpretation |
|---|---|---|
| Refinance option | note rate − market rate; positive incentive | Gross economic benefit of refinancing |
| Seasoning | loan age; nonlinear age term | Frictions at low age and burnout at high age |
| Burnout | prior in-the-money months; maximum prior incentive | Revealed nonresponse to prior refinance opportunities |
| Loan economics | UPB; balance factor; term | Closing-cost economics and remaining exposure |
| Credit / collateral | FICO; DTI; LTV/CLTV; estimated LTV | Refinance eligibility and available equity |
| Borrower / property | purpose; occupancy; property type; FTHB | Mobility and segment-specific behavior |
| Performance path | prior delinquent months; modification | Competing distress and workout pathways |
| Geography / channel | state; origination channel | Housing turnover, costs, and borrower acquisition |

Every cumulative feature is shifted so only history through `t-1` enters the row
at `t`. Optional market rates are joined contemporaneously by month.

### Models

1. **Regularized logistic regression** is the governance baseline. It is stable,
   auditable, and directionally interpretable.
2. **Histogram gradient boosting** captures thresholds and interactions, such as
   stronger incentive response among high-FICO, low-LTV, high-balance borrowers.

Both pipelines perform imputation and encoding internally. Candidate selection
uses validation ROC AUC; after selection, the winner is refit on train +
validation and evaluated once on the untouched test interval.

### Validation philosophy

Random loan-month splitting is inappropriate because adjacent records share the
same loan, rate regime, and macro environment. This project uses nonoverlapping
calendar blocks: the earliest 60% of months for training, next 20% for model
selection, and latest 20% for final testing. In a full research program I would
add rolling-origin folds, vintage holdouts, and loan-clustered bootstrap intervals.

### Why accuracy is omitted

Monthly prepayments are imbalanced. A classifier predicting “no prepayment” for
every loan can appear accurate while having no investment value. The core metrics
are:

- ROC AUC for rank ordering;
- PR AUC for minority-event retrieval;
- Brier score and log loss for probability quality;
- calibration by score band and month;
- top-decile lift for concentration;
- ultimately, UPB-weighted SMM/CPR and runoff forecast error.

## Interpreting results responsibly

High AUC alone does not guarantee useful cash-flow forecasts. Mortgage valuation
requires calibrated probabilities under the relevant rate path. Review score
deciles, time stability, vintage, servicer, geography, credit bands, and UPB
weighting. Large errors during a rate shock may be economically more important
than a small average improvement in discrimination.

The synthetic demonstration is designed with known nonlinear relationships, so
its achieved AUC is neither a benchmark nor a target for real SFLLD work. Real
performance depends on horizon, population, vintage, rate environment, leakage
controls, and whether the payoff definition includes scheduled maturity.

## Business translation: what one point of prepayment speed is worth

The model predicts a monthly payoff probability (SMM) for each loan. Weighted by unpaid balance
and annualized (`CPR = 1 - (1 - SMM)^12`), that becomes the prepayment speed that drives an
MBS holder's cash flows. For an illustrative $100M pool of 30-year, 6.00% level-payment loans,
moving from 8% to 9% CPR gives (`python scripts/business_translation.py`):

| | 8% CPR | 9% CPR | Change |
|---|---:|---:|---:|
| Weighted-average life | 8.96 years | 8.28 years | -0.68 years |
| Principal returned in year 1 | $9.13M | $10.12M | +$0.99M |
| Premium pool (valued at a 5.00% yield): price | 106.435 | 106.049 | -0.386 (-$0.39M) |
| Premium pool: modified duration | 6.40 | 6.02 | -0.38 years |
| Discount pool (valued at a 7.00% yield): price | 94.246 | 94.562 | +0.316 (+$0.32M) |
| Discount pool: modified duration | 5.78 | 5.46 | -0.32 years |

What it means for an investor:

- **Premium holders lose when prepayments speed up.** They paid above par, and each dollar
  repaid early comes back at par, so the premium is written off sooner and fewer above-market
  coupons are collected: about 0.39 points of value per point of CPR here.
- **Discount holders gain**, because principal bought below par is returned at par sooner.
- **Duration shortens either way**, by about 0.3 to 0.4 years per point of CPR. A hedge sized for
  8% CPR is too long if 9% materializes and has to be rebalanced.
- **Negative convexity.** Prepayments rise when rates fall, so duration shortens exactly when a
  holder wants it longer, and the returned principal is reinvested at lower yields.
- **Why calibration matters more than ranking.** A forecast that runs one point too slow on a
  $1B premium book misstates its value by about $3.9M under these assumptions. That is why this
  project reports calibration and Brier score next to AUC, and why UPB-weighted SMM/CPR is the
  next step on real data.

Assumptions: constant CPR, pass-through coupon equal to the note rate (no servicing or guarantee
fee), and duration measured at a fixed prepayment speed (static, not option-adjusted). The figures
illustrate the mechanics; they are not estimates from Freddie Mac data.

## From model to price: MBS valuation with QuantLib

`prepayment-model value` feeds the fitted hazard model into a Monte Carlo valuation of a
pass-through: Hull-White rate paths from QuantLib, a mortgage rate on every path, the model's
monthly prepayment rate, cash flows, then spread and risk measures. For an illustrative 5.50%
pass-through backed by 6.00% loans, priced at par
([full results](reports/valuation/VALUATION_RESULTS.md), [method](docs/VALUATION.md)):

| | Prepayments respond to rates | Cash flows held fixed |
|---|---:|---:|
| Spread over the curve | 58.5 bp (OAS, standard error about 1.6 bp) | 136.9 bp (zero-volatility) |
| Duration | 3.67 | 5.22 |
| Convexity | -2.70 | +0.49 |
| Price change for -100 bp / +100 bp | +2.23 / -4.88 | +5.48 / -4.99 |

- **Option cost: 78 bp.** Of the 137 bp the pool appears to earn over the curve, 78 bp pays for
  the borrowers' right to refinance.
- **Negative convexity, measured.** When rates fall 100 bp the pool gains 2.2 points, not 5.5.
- **Hedge ratio.** The duration to hedge is 3.7, not the 5.2 a fixed-cash-flow view gives.

![Price after a curve shift and projected prepayment speed](reports/valuation/figures/price_and_speed.png)

The prepayment model is fitted on synthetic loans, and the curve, mortgage spread and rate
volatility are made-up inputs, so these numbers show the mechanics and are not a valuation of
any security. The model's inputs are held inside the range it was trained on; the method note
explains why that matters over a 30-year horizon.

## Limitations and next research steps

1. **Competing risks:** estimate separate refinance/sale/maturity/default hazards,
   or use a multinomial/survival framework.
2. **Macro scenarios:** replace a single observed rate with a term-structure-aware
   refinance rate, house-price dynamics, turnover, and unemployment.
3. **Interpretability:** add out-of-time permutation importance, partial dependence,
   ICE, and constrained monotonicity where economics justify it.
4. **Uncertainty:** loan-clustered bootstrap confidence intervals and sensitivity
   to censoring, samples, feature lags, and calendar windows.
5. **Cash-flow translation:** aggregate SMM to CPR, weight by beginning UPB, and
   compare projected versus realized principal runoff.
6. **Production controls:** schema/version contracts, data-vintage manifests,
   drift monitoring, challenger governance, and locked environments.

See the full [methodology note](docs/METHODOLOGY.md) and [model card](docs/MODEL_CARD.md).

## References

- [Freddie Mac SFLLD overview](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset)
- [Freddie Mac General User Guide](https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf)
- [Freddie Mac Terms and Conditions](https://capitalmarkets.freddiemac.com/crt/docs/pdfs/fre_terms_conditions_sflld.pdf)
- [Freddie Mac Frequently Asked Questions](https://www.freddiemac.com/fmac-resources/research/pdf/faq.pdf)
- [FRED MORTGAGE30US series](https://fred.stlouisfed.org/series/MORTGAGE30US)

## Disclaimer

For research and educational use only. Not investment advice, a pricing model for any
security, or an official Freddie Mac publication. Freddie Mac does not endorse this work.
The author makes no representation that the model is suitable for trading, risk
limits, accounting, capital, or borrower decisions.
