# Model card

## Intended use

Research estimation of one-month voluntary prepayment probabilities for active
fixed-rate mortgages represented in the Freddie Mac Standard SFLLD. Outputs can
support exploratory portfolio stratification and cash-flow model development.

## Out-of-scope use

The model is not approved for borrower eligibility, pricing, servicing actions,
fair-lending decisions, regulatory capital, accounting, or live investment use.
It should not be extrapolated to products or periods absent from development data.

## Target

`response = 1` when the immediately following contiguous monthly performance
record has Zero Balance Code `01`; otherwise `0`. The current zero-balance code
and all future information are excluded from model features.

## Data

Users obtain SFLLD directly from Freddie Mac and retain it locally. Freddie Mac
masks selected fields for privacy and describes the dataset as unaudited and
subject to change. This repository does not ship loan-level records.

## Evaluation

Nonoverlapping calendar-time train, validation, and test windows. Report ROC AUC,
PR AUC, Brier score, log loss, top-decile lift, calibration, and monthly stability.
Add UPB-weighted and segmented diagnostics before any portfolio application.

## Key risks

- data and concept drift across rate regimes;
- selection bias from SFLLD eligibility and missing records;
- conflation of refinance, sale, and scheduled maturity in code `01`;
- imperfect public proxies for executable borrower refinance rates;
- correlation across repeated loan-month observations;
- differential errors across geographic or borrower segments;
- overreliance on discrimination instead of probability calibration.

## Governance recommendations

Version data, schema, macro vintages, code, configuration, and random seeds.
Require independent target validation, leakage review, segmented backtests,
economic-sign review, challenger comparison, drift thresholds, and documented
approval before production use.
