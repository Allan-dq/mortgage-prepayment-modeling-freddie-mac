# From prepayment model to price: the valuation module

`prepayment-model value` takes the fitted loan-month hazard model and uses it to value a
fixed-rate pass-through. Results: [`reports/valuation/VALUATION_RESULTS.md`](../reports/valuation/VALUATION_RESULTS.md).
Code: [`src/prepayment/valuation.py`](../src/prepayment/valuation.py).

**What this is not.** The model is fitted on synthetic loans and the curve, mortgage spread
and volatility are made-up inputs. The output shows the mechanics. It is not a valuation of
any security.

## Steps

1. **Curve.** A zero curve is built with QuantLib (`ZeroCurve`, continuous compounding).
2. **Rate paths.** QuantLib's `HullWhiteProcess` and `GaussianPathGenerator` simulate the
   short rate monthly for the life of the pool, in antithetic pairs. A path's discount
   factor is the exponential of minus the integrated short rate (trapezoid rule).
3. **Repricing the curve.** The average path discount factor is rescaled, date by date, to
   equal the curve's. Without this, sampling noise at long maturities would leak into every
   price. The largest correction is reported (0.53% with 4,000 paths).
4. **Mortgage rate.** On each path and month the market mortgage rate is the 10-year zero
   yield plus a constant spread. In Hull-White that yield is a straight line in the short
   rate, taken from QuantLib's `HullWhite.discountBond`.
5. **Prepayment.** One representative loan per path is run through `engineer_features`, the
   same code used in training, so refinancing incentive, burnout and rate volatility are
   computed the same way. The fitted model's monthly probability is the pool's SMM. The rate
   known at month k-1 drives the prepayment in month k, matching the model's target.
6. **Cash flows.** Scheduled principal from a level payment at the note rate, prepayment on
   the remaining balance, interest to the investor at the pass-through coupon.
7. **Spreads.** OAS is the constant spread over each path's rates at which the average path
   value equals the price. The zero-volatility spread is the same calculation on the single
   forward-curve path. Their difference is the option cost.
8. **Risk.** The curve is moved 25 bp each way with the same random numbers and the OAS held
   fixed; prepayments respond. That gives effective duration and convexity. Repeating it with
   the forward-path cash flows frozen gives the static figures for comparison.

## No extrapolation

The synthetic training panel has no loan older than 119 months and no market rate outside
3.4% to 6.7%, while a 30-year valuation needs both. Left alone, the logistic model
extrapolates to a monthly prepayment rate above 80% late in the pool's life. The valuation
therefore holds every numeric input at the edge of the range seen in training
(`training_limits`, `fitted_hazard`). This moved the OAS in the demonstration by about 10 bp.

## Checks (tests/test_valuation.py)

- A pool paying 5.5% priced on a flat 5.5% curve is worth par at any prepayment speed.
- Price and weighted-average life at a constant CPR match `scripts/business_translation.py`.
- Simulated paths reprice the curve; a fixed seed reproduces the paths exactly.
- Cash flows that ignore rates have zero option cost, and their effective duration equals
  their static duration.
- The OAS reprices the pool; rate-sensitive prepayment gives a positive option cost and
  negative convexity, and lower volatility lowers the option cost.
- Prepayment in a month uses only the rate known the month before.
- Eight bugs injected on purpose (look-ahead, missing curve correction, wrong coupon, wrong
  prepayment base, guard switched off, spread sign, yield slope sign, duration formula) were
  each caught by these tests.

## Limitations

- The synthetic model reaches speeds no real pool shows (over 80% CPR at two points in the
  money), which inflates the option cost. A model fitted on Freddie Mac data should replace it.
- Its burnout feature counts months more than 0.5 points in the money, so speeds can fall
  sharply once a pool has been in the money for a while; weighted-average life is therefore
  not monotone in the size of a rate fall.
- One representative loan stands for the pool; real pools mix loans, and that mix is itself a
  source of burnout.
- Hull-White parameters are not calibrated to swaptions; the mortgage rate is a fixed spread
  over the 10-year yield; no payment delay, defaults or curtailments.
- The standard error treats antithetic pairs as independent and ignores the curve correction.
