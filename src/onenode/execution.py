"""Turning an approved candidate into a multi-leg order.

Orders are always limit orders and always defined-risk structures submitted as
a single ``mleg`` order rather than as two separate legs. Legging into a spread
one contract at a time means that between the two fills the position is
briefly naked - which is precisely the state the hard gate exists to forbid.
Submitting both legs as one order makes that state unreachable.

The limit price is set below the mid the candidate was priced at, because the
candidate was already priced pessimistically at bid-ask. Paying a little for
certainty of fill is worth more than an extra dollar of credit on a position
whose whole thesis is that it expires worthless.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .broker.cli import AlpacaCLI, AlpacaCLIError
from .portfolio import PositionGroup
from .risk.models import OptionLeg, Side
from .strategy import SpreadCandidate

CONTRACT_MULTIPLIER = 100


def leg_payload(candidate: SpreadCandidate) -> list[dict[str, str]]:
    """The ``legs`` array Alpaca expects for an ``mleg`` order.

    ``position_intent`` is stated explicitly on both legs. Left implicit, a
    broker may net a new spread against an existing position and close
    something the agent still wanted open.
    """
    payload = []
    for leg in (candidate.short_leg, candidate.long_leg):
        opening = "sell_to_open" if leg.side is Side.SELL else "buy_to_open"
        payload.append(
            {
                "symbol": leg.symbol,
                "side": leg.side.value,
                "ratio_qty": str(leg.ratio_qty),
                "position_intent": opening,
            }
        )
    return payload


def closing_leg_payload(legs: Iterable[OptionLeg]) -> list[dict[str, str]]:
    """The same structure with every side reversed, to flatten a position.

    Takes legs rather than a candidate because positions are closed from what
    the broker reports as open, not from the candidate that opened them - by
    the time an exit rule fires, that candidate object is long gone.
    """
    payload = []
    for leg in legs:
        closing_side = "buy" if leg.side is Side.SELL else "sell"
        payload.append(
            {
                "symbol": leg.symbol,
                "side": closing_side,
                "ratio_qty": str(leg.ratio_qty),
                "position_intent": f"{closing_side}_to_close",
            }
        )
    return payload


def limit_price_for_credit(candidate: SpreadCandidate, slippage: float = 0.02) -> float:
    """Net credit to ask for, per share, shaded to make the fill likely.

    The candidate is priced at short-bid minus long-ask, already the worst
    realistic fill. Shading a further couple of cents buys fill probability on
    a position that earns its money by being opened at all.
    """
    credit_per_share = candidate.credit_per_contract / CONTRACT_MULTIPLIER
    return max(round(credit_per_share - slippage, 2), 0.01)


def build_order_args(
    legs: list[dict[str, str]],
    contracts: int,
    limit_price: float,
    *,
    dry_run: bool = False,
    client_order_id: str | None = None,
) -> list[str]:
    """Assemble the CLI argument list for a multi-leg limit order.

    Pure, so the argument shape is testable without a broker. ``time_in_force``
    is always ``day``: this agent should never wake up owning something it
    ordered yesterday and forgot about.
    """
    args = [
        "order",
        "submit",
        "--order-class",
        "mleg",
        "--qty",
        str(contracts),
        "--type",
        "limit",
        "--limit-price",
        f"{limit_price:.2f}",
        "--time-in-force",
        "day",
        "--legs",
        json.dumps(legs, separators=(",", ":")),
    ]
    if client_order_id:
        args += ["--client-order-id", client_order_id]
    if dry_run:
        args.append("--dry-run")
    return args


def submit_spread(
    cli: AlpacaCLI,
    candidate: SpreadCandidate,
    contracts: int,
    *,
    limit_price: float | None = None,
    dry_run: bool = False,
    client_order_id: str | None = None,
) -> dict[str, Any]:
    """Send the order. Raises :class:`AlpacaCLIError` if the broker refuses it."""
    price = limit_price if limit_price is not None else limit_price_for_credit(candidate)
    args = build_order_args(
        leg_payload(candidate),
        contracts,
        price,
        dry_run=dry_run,
        client_order_id=client_order_id,
    )
    result = cli._run(*args)
    if not isinstance(result, dict):
        raise AlpacaCLIError(f"unexpected order response: {result!r}")
    return result


def close_position(
    cli: AlpacaCLI,
    group: PositionGroup,
    limit_price: float,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Buy back an open structure, closing it.

    ``limit_price`` is the net debit paid to close - a positive number. It is
    set generously by the caller: an exit that does not fill is not an exit,
    and the whole reason for closing is that holding is the worse option.
    """
    args = build_order_args(
        closing_leg_payload(group.legs),
        group.contracts,
        limit_price,
        dry_run=dry_run,
    )
    result = cli._run(*args)
    if not isinstance(result, dict):
        raise AlpacaCLIError(f"unexpected order response: {result!r}")
    return result


def closing_limit_price(group: PositionGroup, cushion: float = 0.05) -> float:
    """Net debit to bid for the buy-back, per share.

    Derived from what the position is currently marked at, plus a cushion so a
    small adverse move between the quote and the fill does not leave the
    position open. Floored at a cent, since a limit of zero never fills.
    """
    per_share = abs(group.market_value) / (CONTRACT_MULTIPLIER * max(group.contracts, 1))
    return max(round(per_share + cushion, 2), 0.01)
