# MBS valuation demonstration (synthetic model, illustrative market)

> **Important:** the prepayment model here is fitted on the repository's synthetic loans, the
> yield curve, mortgage spread and rate volatility are made-up inputs, and the price is an
> input rather than a quote. The numbers show how a prepayment model turns into a price, a
> spread and a hedge ratio. They are **not a valuation of any security**.

## Inputs

- Pool: 6.00% note rate, 5.50% pass-through coupon, 360 months remaining, loan age 0; representative loan with FICO 740, LTV 75, DTI 34, balance $260,000
- Zero curve (continuous compounding): 0y: 4.00% | 0.25y: 4.00% | 1y: 3.90% | 2y: 3.80% | 5y: 3.90% | 10y: 4.20% | 20y: 4.50% | 30y: 4.60% | 40y: 4.60%
- Market mortgage rate: 10-year zero yield + 1.80 points = 6.00% today
- Hull-White short-rate model: mean reversion 0.030, volatility 0.0100
- Prepayment model: logistic regression fitted on 3,000 synthetic loans (seed 42); numeric inputs held within the range seen in training
- Monte Carlo: 4,000 paths in antithetic pairs, seed 42; paths rescaled to reprice the curve (largest correction 0.53%)

## Results at a price of 100.000

| Measure | Value |
|---|---:|
| Option-adjusted spread (OAS) | 58.5 bp (standard error about 1.6 bp) |
| Zero-volatility spread | 136.9 bp |
| Option cost (zero-volatility spread minus OAS) | 78.4 bp |
| Effective duration (prepayments respond to rates) | 3.67 |
| Effective convexity | -2.70 |
| Duration if cash flows were fixed | 5.22 |
| Convexity if cash flows were fixed | 0.49 |
| Weighted-average life on the forward path | 7.05 years |
| Weighted-average life across paths: mean (5th to 95th percentile) | 6.00 years (1.72 to 11.12) |

Durations and convexities come from moving the whole curve by 25 bp each way; convexity is
`(P_down + P_up - 2 P) / (P * shift^2) / 100`.

## Parallel curve shifts

Model price holds the OAS fixed and lets prepayments respond. The fixed-cash-flow price
keeps today's forward-path cash flows and holds the zero-volatility spread fixed.

| Shift (bp) | Mortgage rate today | Model price | Change | Fixed-cash-flow price | Change | Forward-path CPR, year 2 | Forward-path WAL (years) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -100 | 5.00% | 102.232 | +2.232 | 105.476 | +5.476 | 28.2% | 4.76 |
| -50 | 5.50% | 101.491 | +1.491 | 102.673 | +2.673 | 20.8% | 4.47 |
| -25 | 5.75% | 100.834 | +0.834 | 101.321 | +1.321 | 14.9% | 5.82 |
| +0 | 6.00% | 100.000 | +0.000 | 100.000 | +0.000 | 11.2% | 7.05 |
| +25 | 6.25% | 98.997 | -1.003 | 98.710 | -1.290 | 9.0% | 8.07 |
| +50 | 6.50% | 97.836 | -2.164 | 97.449 | -2.551 | 7.3% | 8.92 |
| +100 | 7.00% | 95.119 | -4.881 | 95.011 | -4.989 | 5.6% | 10.22 |

## What the prepayment model does

Prepayment speed (CPR) if the market mortgage rate stayed at the note rate minus the
incentive for the whole life of the pool:

| Incentive (points) | CPR in year 1 | CPR in year 3 | CPR in year 8 |
|---:|---:|---:|---:|
| -2.0 | 3.2% | 3.7% | 5.2% |
| -1.0 | 5.0% | 5.7% | 8.1% |
| +0.0 | 10.4% | 11.9% | 16.6% |
| +0.5 | 20.4% | 23.3% | 31.4% |
| +1.0 | 33.8% | 25.1% | 13.2% |
| +2.0 | 81.7% | 70.6% | 45.8% |
| +3.0 | 99.5% | 98.3% | 88.4% |
