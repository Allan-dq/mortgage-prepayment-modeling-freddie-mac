# Demonstration results (synthetic data)

> **Important:** these metrics are generated from the repository's synthetic
> data generator. They validate software behavior and illustrate the reporting
> workflow; they are **not estimates of performance on Freddie Mac loans**.

## Demonstration backtest

- Selected model: **Logistic Regression**
- Temporal split: train through 2021-01, validation through 2023-06, test from 2023-07
- Test observations: 16,138
- Test events: 33 (0.20%)
- ROC AUC: 0.710
- PR AUC: 0.005
- Brier score: 0.0020
- Top-decile lift: 3.03x

For an empirical study, download licensed SFLLD annual files directly from
Freddie Mac, place them under `data/raw/`, and run `prepayment-model freddie`.
Review the resulting aggregate diagnostics before making any investment use.
