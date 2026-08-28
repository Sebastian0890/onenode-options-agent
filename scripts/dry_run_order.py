"""Build a real order from the live chain and submit it with --dry-run.

Proves the order body is well-formed without placing a trade, which is the one
part of the agent that cannot be checked by unit tests alone: the broker has
the final say on whether a legs array is acceptable.

    python scripts/dry_run_order.py [SPY]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from onenode.broker import AlpacaCLI, AlpacaCLIError  # noqa: E402
from onenode.execution import (  # noqa: E402
    build_order_args,
    leg_payload,
    limit_price_for_credit,
    submit_spread,
)
from onenode.risk.gate import evaluate  # noqa: E402
from onenode.risk.limits import DEFAULT_LIMITS  # noqa: E402
from onenode.risk.models import AccountSnapshot  # noqa: E402
from onenode.strategy import build_credit_spreads, size_position  # noqa: E402


def main() -> int:
    underlying = (sys.argv[1] if len(sys.argv) > 1 else "SPY").upper()
    cli = AlpacaCLI()

    clock = cli.clock()
    account = cli.account()
    equity = float(account["equity"])

    spot = cli.latest_price(underlying)
    chain = cli.option_chain(
        underlying,
        option_type="put",
        strike_gte=spot * 0.94,
        strike_lte=spot * 1.00,
        limit=1000,
    )
    candidates = build_credit_spreads(list(chain.values()), underlying=underlying)
    if not candidates:
        print("No candidates right now - nothing to dry-run.")
        return 1

    candidate = candidates[0]
    contracts = size_position(
        candidate,
        equity,
        DEFAULT_LIMITS.max_risk_per_trade_pct,
        DEFAULT_LIMITS.max_contracts_per_order,
    )
    print(f"candidate  {candidate.describe()}")
    print(f"size       {contracts} contracts")

    trade = candidate.to_proposed_trade(contracts)
    snapshot = AccountSnapshot(
        equity=equity,
        last_equity=float(account["last_equity"]),
        open_positions=0,
        committed_risk=0.0,
    )
    decision = evaluate(trade, snapshot, clock, today=candidate.expiry.today())
    print(f"gate       {decision}")

    price = limit_price_for_credit(candidate)
    print(f"limit      ${price:.2f} net credit per share")
    print("\nlegs:")
    print(json.dumps(leg_payload(candidate), indent=2))

    print("\nargv the CLI receives:")
    for arg in build_order_args(leg_payload(candidate), contracts, price, dry_run=True):
        print(f"  {arg}")

    print("\n--- broker dry-run ---")
    try:
        result = submit_spread(cli, candidate, contracts, dry_run=True)
    except AlpacaCLIError as exc:
        print(f"REJECTED: {exc}")
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
