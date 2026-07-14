# Methodology

## 1. Modeling estimand

The unit of observation is an active mortgage at month-end `t`. The response is
one if the loan's next contiguous monthly record (`t+1`) carries Zero Balance
Code `01`, and zero otherwise. Freddie Mac defines `01` as **Prepaid or Matured
(Voluntary Payoff)**. Rows without a contiguous next observation are censored,
not assumed to be non-events. The event row is never used as a predictor row.

This is a discrete-time hazard model. Its probability can be aggregated into
portfolio cash-flow assumptions, but a one-month probability is not itself CPR.
Under a constant monthly mortality assumption, `CPR = 1 - (1 - SMM)^12`.

## 2. Population and filters

The intended empirical population is the Freddie Mac Standard SFLLD population.
Recommended production filters are:

1. positive current UPB at `t`;
2. no zero-balance code at `t`;
3. a contiguous `t+1` performance record;
4. valid origination/performance identifier match;
5. performance months observed before the dataset cutoff;
6. optional exclusion or separate modeling of modified, seriously delinquent,
   HARP, and super-conforming loans where behavior is structurally different.

## 3. Information set and leakage controls

Every predictor is known at or before `t`. The target is derived with a grouped
one-row lead, but the lead fields are dropped before estimation. Burnout and
delinquency-history features are shifted one month. Imputation, scaling, and
category encoding are fit inside each model pipeline on development data only.
Model selection uses validation data; the test interval is touched once.

Calendar-time validation is deliberately stricter than a random loan-month
split. A random split leaks interest-rate regimes, seasoning patterns, and often
the same loan's adjacent months across folds. The chosen design asks the more
relevant investment question: does the relationship generalize into a later
rate environment?

## 4. Feature economics

| Feature family | Economic hypothesis | Expected relationship |
|---|---|---|
| Refinance incentive | Coupon minus prevailing mortgage rate approximates savings from refinancing | Positive, nonlinear |
| Seasoning | New loans face frictions; mobility/refinancing rises, then mature loans burn out | Hump-shaped |
| Burnout | Borrowers who repeatedly passed profitable refinance windows are less responsive | Negative conditional on incentive |
| Balance / UPB | Closing costs are more economical on larger balances | Generally positive |
| LTV / equity | High LTV constrains refinancing; equity also facilitates home sale | Lower LTV generally raises payoff propensity |
| FICO / DTI | Proxy for refinance eligibility and borrower capacity | Higher FICO positive; high DTI negative |
| Purpose / occupancy | Cash-out, investor, and second-home behavior differs from owner-occupied purchase loans | Segment-specific |
| Prior delinquency / modification | Distress and workout paths compete with voluntary payoff | Generally negative |
| Rate volatility | Option value and borrower attention change in volatile rate environments | Nonlinear |

The `market_rate` should be a contemporaneously available mortgage-rate series.
The included downloader uses FRED's `MORTGAGE30US` series (Freddie Mac 30-year
fixed-rate average) and averages weekly observations by calendar month. Modeling
accepts a local CSV rather than silently fetching data, preserving an auditable
vintage.

## 5. Models

The benchmark is regularized logistic regression. It is transparent, stable,
and gives directional effects, but additive log-odds can miss threshold behavior.
The challenger is histogram gradient boosting, which captures interactions among
incentive, age, balance, LTV, and borrower capacity. Candidate selection is based
on validation ROC AUC; both probability quality and ranking are reported.

## 6. Metrics

- **ROC AUC:** global rank discrimination; prevalence-insensitive but not enough alone.
- **PR AUC:** more informative when monthly prepayments are rare.
- **Brier score / log loss:** probability accuracy, important for cash-flow forecasts.
- **Calibration curve:** observed versus predicted hazards by score band.
- **Top-decile lift:** operational concentration of events in the highest-score loans.
- **Monthly stability:** regime drift and calibration through time.

For investment use, add UPB-weighted calibration, SMM/CPR rollups, dollar runoff
error, vintage/servicer/state slices, and confidence intervals clustered by loan.

## 7. Known limitations

- Voluntary payoff combines refinancing, home sale, and scheduled maturity.
- Public data mask or bucket some attributes and contain no borrower application,
  property liquidity, broker quote, or lock data.
- Market rates are aggregate offers, not borrower-specific executable refinance rates.
- Economic regimes change; 2020–2021 refinancing waves and later high-rate periods
  are not exchangeable.
- Competing risks matter: default, modification, sale, repurchase, and payoff should
  ideally be modeled jointly.
- AUC does not measure cash-flow materiality. Calibration and UPB-weighted errors
  can matter more for valuation and hedging.
- The included synthetic demonstration validates code only. Real-data claims must
  come from a licensed local SFLLD run.
