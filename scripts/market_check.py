"""Live diagnostic: does the whole read path actually work right now?

Walks the same route the agent takes on every run - clock, account, positions,
spot, option chain - and prints what it found. Run it before trusting a change,
and run it when something looks wrong in the journal.

    python scripts/market_check.py [SPY]
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from onenode.broker import AlpacaCLI, AlpacaCLIError  # noqa: E402
from onenode.portfolio import committed_risk, group_option_positions  # noqa: E402

TARGET_DELTA = 0.17


def main() -> int:
    underlying = (sys.argv[1] if len(sys.argv) > 1 else "SPY").upper()
    cli = AlpacaCLI()

    problems = cli.preflight()
    if problems:
        print("Preflight failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    clock = cli.clock()
    account = cli.account()
    equity = float(account["equity"])
    last_equity = float(account["last_equity"])

    print(f"market open        {clock.is_open}")
    if clock.is_open:
        print(f"minutes to close   {clock.minutes_to_close:.0f}")
    print(f"equity             ${equity:,.2f}")
    print(f"day P&L            {100 * (equity - last_equity) / last_equity:+.2f}%")

    groups = group_option_positions(cli.positions())
    risk = committed_risk(groups)
    print(f"open structures    {len(groups)}")
    if risk is None:
        print("committed risk     UNBOUNDED - something is open that we did not open")
    else:
        print(f"committed risk     ${risk:,.2f}")
    for group in groups:
        print(f"  {group.underlying} {group.expiry} x{group.contracts}: {group.remaining_risk}")

    spot = cli.latest_price(underlying)
    print(f"\n{underlying} spot          {spot:,.2f}")

    try:
        chain = cli.option_chain(
            underlying,
            option_type="put",
            strike_gte=spot * 0.96,
            strike_lte=spot * 1.01,
            limit=200,
        )
    except AlpacaCLIError as exc:
        print(f"chain fetch failed: {exc}")
        return 1

    with_delta = [q for q in chain.values() if q.delta is not None]
    print(f"tradeable puts     {len(chain)} ({len(with_delta)} with greeks)")

    if not with_delta:
        print("No contracts carried greeks - strike selection would have nothing to work with.")
        return 1

    print(f"\nclosest to {TARGET_DELTA:.2f} delta:")
    ranked = sorted(with_delta, key=lambda q: abs(abs(q.delta) - TARGET_DELTA))
    for quote in ranked[:5]:
        print(
            f"  {quote.symbol}  bid {quote.bid:6.2f}  ask {quote.ask:6.2f}  "
            f"delta {quote.delta:+.3f}  spread {quote.spread_pct_of_mid:5.1f}%  "
            f"age {quote.age_seconds():4.0f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
