"""What a 1 percentage-point change in prepayment speed means for an MBS holder.

Projects the cash flows of an illustrative level-payment mortgage pool at two constant
prepayment speeds (CPR) and reports weighted-average life, first-year principal returned,
price and modified duration at a premium and a discount yield. Standard library only;
the numbers in the README's "Business translation" section come from this script.

Simplifications: the pass-through coupon equals the note rate (no servicing or guarantee
fee), CPR is constant over the life of the pool, and duration is measured at a fixed
prepayment speed (static duration, not option-adjusted effective duration).
"""

import argparse


def cash_flows(balance: float, note_rate: float, months: int, cpr: float):
    """Monthly (interest, principal) pairs under a constant CPR."""
    rate = note_rate / 12
    smm = 1 - (1 - cpr) ** (1 / 12)
    flows = []
    for month in range(months):
        if balance <= 1e-6:
            break
        remaining = months - month
        payment = balance * rate / (1 - (1 + rate) ** -remaining)
        interest = balance * rate
        scheduled = payment - interest
        prepaid = smm * (balance - scheduled)
        principal = scheduled + prepaid
        flows.append((interest, principal))
        balance -= principal
    return flows


def analytics(flows, original_balance: float, annual_yield: float) -> dict:
    discount = 1 + annual_yield / 12
    pv = [(i + p) / discount ** (t + 1) for t, (i, p) in enumerate(flows)]
    price = sum(pv) / original_balance * 100
    macaulay = sum((t + 1) / 12 * v for t, v in enumerate(pv)) / sum(pv)
    return {"price": price, "modified_duration": macaulay / discount}


def summarize(
    balance: float, note_rate: float, months: int, cpr: float, yields: list[float]
):
    flows = cash_flows(balance, note_rate, months, cpr)
    wal = sum((t + 1) * p for t, (_, p) in enumerate(flows)) / balance / 12
    first_year_principal = sum(p for _, p in flows[:12])
    return {
        "cpr": cpr,
        "wal_years": wal,
        "first_year_principal": first_year_principal,
        "by_yield": {y: analytics(flows, balance, y) for y in yields},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance", type=float, default=100_000_000)
    parser.add_argument("--note-rate", type=float, default=0.06)
    parser.add_argument("--months", type=int, default=360)
    parser.add_argument("--cpr", type=float, nargs=2, default=[0.08, 0.09])
    parser.add_argument("--yields", type=float, nargs="+", default=[0.05, 0.07])
    args = parser.parse_args()

    base, faster = (
        summarize(args.balance, args.note_rate, args.months, c, args.yields)
        for c in args.cpr
    )
    print(
        f"Pool: ${args.balance / 1e6:,.0f}M, {args.note_rate:.2%} note rate, "
        f"{args.months}-month level payment"
    )
    for row in (base, faster):
        print(
            f"\nCPR {row['cpr']:.0%}: WAL {row['wal_years']:.2f} years, "
            f"first-year principal ${row['first_year_principal'] / 1e6:,.2f}M"
        )
        for y, values in row["by_yield"].items():
            print(
                f"  yield {y:.2%}: price {values['price']:.3f}, "
                f"modified duration {values['modified_duration']:.2f}"
            )
    print("\nEffect of +1pp CPR:")
    print(f"  WAL {faster['wal_years'] - base['wal_years']:+.2f} years")
    extra = faster["first_year_principal"] - base["first_year_principal"]
    print(
        f"  first-year principal {extra / 1e6:+,.2f}M per ${args.balance / 1e6:,.0f}M"
    )
    for y in args.yields:
        b, f = base["by_yield"][y], faster["by_yield"][y]
        dollars = (f["price"] - b["price"]) / 100 * args.balance
        print(
            f"  at {y:.2%} yield: price {f['price'] - b['price']:+.3f} "
            f"({dollars / 1e6:+,.2f}M), modified duration "
            f"{f['modified_duration'] - b['modified_duration']:+.2f} years"
        )


if __name__ == "__main__":
    main()
