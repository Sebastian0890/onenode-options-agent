"""Turning a flat list of open option positions back into structures.

Alpaca reports positions one contract at a time. The risk ceiling, though, is a
property of structures rather than of legs: two legs of a vertical spread cap
each other's loss, and adding them up separately would overstate exposure badly
enough that the agent would refuse trades it can comfortably afford.

So positions are regrouped by underlying and expiry, the common contract count
is factored out, and the payoff engine measures each structure the same way it
measures a proposal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from math import gcd
from typing import Any

from .risk.models import OptionLeg, Side
from .risk.payoff import remaining_risk


@dataclass(frozen=True)
class PositionGroup:
    """Open legs on one underlying and expiry, treated as a single structure."""

    underlying: str
    expiry: date
    legs: tuple[OptionLeg, ...]
    contracts: int
    market_value: float

    @property
    def remaining_risk(self) -> float | None:
        """Additional dollars this structure can still lose, or ``None`` if unbounded."""
        return remaining_risk(self.legs, self.contracts, self.market_value)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def group_option_positions(positions: list[dict[str, Any]]) -> list[PositionGroup]:
    """Regroup raw Alpaca positions into structures.

    Non-option positions are ignored: this agent only ever holds options, so
    anything else is either a leftover or someone else's trade, and in both
    cases it is not ours to reason about.
    """
    buckets: dict[tuple[str, date], list[tuple[OptionLeg, float]]] = defaultdict(list)

    for position in positions:
        if position.get("asset_class") != "us_option":
            continue

        symbol = position.get("symbol")
        if not isinstance(symbol, str):
            continue

        quantity = abs(int(_to_float(position.get("qty"))))
        if quantity == 0:
            continue

        side = Side.SELL if str(position.get("side", "long")).lower() == "short" else Side.BUY
        try:
            leg = OptionLeg.from_symbol(symbol, side=side, ratio_qty=quantity)
        except ValueError:
            continue

        buckets[(leg.root, leg.expiry)].append((leg, _to_float(position.get("market_value"))))

    groups: list[PositionGroup] = []
    for (underlying, expiry), entries in sorted(buckets.items()):
        quantities = [leg.ratio_qty for leg, _ in entries]
        contracts = gcd(*quantities) if len(quantities) > 1 else quantities[0]

        # Factor the common contract count out of the leg ratios so the payoff
        # engine sees one unit of the structure repeated `contracts` times.
        legs = tuple(
            OptionLeg(
                symbol=leg.symbol,
                side=leg.side,
                ratio_qty=leg.ratio_qty // contracts,
                right=leg.right,
                strike=leg.strike,
                expiry=leg.expiry,
            )
            for leg, _ in entries
        )
        groups.append(
            PositionGroup(
                underlying=underlying,
                expiry=expiry,
                legs=legs,
                contracts=contracts,
                market_value=sum(value for _, value in entries),
            )
        )
    return groups


def committed_risk(groups: list[PositionGroup]) -> float | None:
    """Total dollars still at risk across all open structures.

    Returns ``None`` if any structure has unbounded downside. That should be
    impossible - the gate never approves one - so it means something is open
    that this agent did not put there, and the correct response is to stop
    trading rather than to guess at a number.
    """
    total = 0.0
    for group in groups:
        risk = group.remaining_risk
        if risk is None:
            return None
        total += risk
    return total
