"""Does the candidate builder actually find anything in the real chain?

Filters that look reasonable on paper can reject every trade in practice. This
sweeps the reward-to-risk floor and the delta band against a live chain and
prints what survives, so the thresholds in limits.py are set from data rather
than from taste.

    python scripts/calibrate.py [SPY]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from onenode.broker import AlpacaCLI  # noqa: E402
from onenode.risk.models import Right  # noqa: E402
from onenode.strategy import build_credit_spreads, size_position  # noqa: E402

EQUITY = 100_000.0


def main() -> int:
    underlying = (sys.argv[1] if len(sys.argv) > 1 else "SPY").upper()
    cli = AlpacaCLI()

    spot = cli.latest_price(underlying)
    print(f"{underlying} spot {spot:,.2f}\n")

    puts = cli.option_chain(
        underlying,
        option_type="put",
        strike_gte=spot * 0.93,
        strike_lte=spot * 1.00,
        limit=1000,
    )
    print(f"tradeable puts in band: {len(puts)}")

    quotes = list(puts.values())

    print("\nreward-to-risk floor sweep (delta 0.17 +/- 0.08, all widths):")
    for floor in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        found = build_credit_spreads(
            quotes, underlying=underlying, min_reward_to_risk=floor
        )
        best = found[0].reward_to_risk if found else 0.0
        print(f"  floor {floor:.2f} -> {len(found):4d} candidates, best r/r {best:.3f}")

    print("\ndelta band sweep (floor 0.05):")
    for target in (0.10, 0.15, 0.17, 0.20, 0.25, 0.30):
        found = build_credit_spreads(
            quotes,
            underlying=underlying,
            target_delta=target,
            delta_tolerance=0.03,
            min_reward_to_risk=0.05,
        )
        best = found[0].reward_to_risk if found else 0.0
        print(f"  delta {target:.2f} -> {len(found):4d} candidates, best r/r {best:.3f}")

    print("\nwidth sweep (delta 0.17 +/- 0.08, floor 0.05):")
    for width in (1.0, 2.0, 3.0, 5.0, 10.0):
        found = build_credit_spreads(
            quotes, underlying=underlying, widths=(width,), min_reward_to_risk=0.05
        )
        best = found[0].reward_to_risk if found else 0.0
        print(f"  width ${width:4.0f} -> {len(found):4d} candidates, best r/r {best:.3f}")

    everything = build_credit_spreads(
        quotes, underlying=underlying, min_reward_to_risk=0.05, widths=(1, 2, 3, 5, 10)
    )
    if not everything:
        print("\nNothing survives. The filters are too tight for this chain.")
        return 1

    by_expiry = Counter(str(c.expiry) for c in everything)
    print(f"\ncandidates by expiry: {dict(sorted(by_expiry.items()))}")

    print("\ntop 8 by reward-to-risk:")
    for candidate in everything[:8]:
        contracts = size_position(candidate, EQUITY, max_risk_pct=1.5, max_contracts=25)
        print(f"  {candidate.describe()} | size {contracts}")

    calls = cli.option_chain(
        underlying,
        option_type="call",
        strike_gte=spot * 1.00,
        strike_lte=spot * 1.07,
        limit=1000,
    )
    call_candidates = build_credit_spreads(
        list(calls.values()),
        underlying=underlying,
        right=Right.CALL,
        min_reward_to_risk=0.05,
        widths=(1, 2, 3, 5, 10),
    )
    print(f"\ncall-side candidates: {len(call_candidates)}")
    for candidate in call_candidates[:3]:
        print(f"  {candidate.describe()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
