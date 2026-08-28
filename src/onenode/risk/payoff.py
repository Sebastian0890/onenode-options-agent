"""Expiry-payoff analysis for multi-leg option positions.

The gate needs to know the worst case of a proposal before letting it through,
and it derives that number here rather than trusting whatever figure the model
attached to its proposal. An LLM that understates the risk of its own idea
cannot talk its way past a number the gate computed itself.

The method is the standard one for a portfolio of options on a single
underlying: profit at expiry is piecewise linear in the underlying price, with
kinks only at the strikes. Sampling the payoff at zero and at every strike
therefore finds the true minimum over the bounded region, and the slope as the
price grows without bound tells us whether a region of unbounded loss exists at
all.

Scope, stated plainly: this is a European-style analysis of value at expiry.
It does not model early assignment on short American legs, dividend-driven
early exercise, or pin risk at the strike. Those are real, and the agent avoids
them structurally instead - by trading index ETFs and by closing rather than
holding through expiry - not by modelling them here.
"""

from __future__ import annotations

from collections.abc import Iterable

from .models import CONTRACT_MULTIPLIER, OptionLeg, Right


def intrinsic_value(leg: OptionLeg, underlying_price: float) -> float:
    """Value of one share-equivalent of ``leg`` at expiry, ignoring its side."""
    if leg.right is Right.CALL:
        return max(underlying_price - leg.strike, 0.0)
    return max(leg.strike - underlying_price, 0.0)


def payoff_per_share(legs: Iterable[OptionLeg], underlying_price: float) -> float:
    """Net value of one unit of the structure at expiry, per share.

    Long legs contribute their intrinsic value, short legs subtract it.
    """
    return sum(leg.signed_ratio * intrinsic_value(leg, underlying_price) for leg in legs)


def slope_at_infinity(legs: Iterable[OptionLeg]) -> int:
    """Slope of the payoff as the underlying price grows without bound.

    Only calls matter: far above every strike, each long call gains a dollar
    per dollar of underlying and each short call loses one. A negative slope
    means losses grow forever - an undefined-risk position.
    """
    return sum(leg.signed_ratio for leg in legs if leg.right is Right.CALL)


def profit_at(
    legs: Iterable[OptionLeg],
    contracts: int,
    net_cash: float,
    underlying_price: float,
) -> float:
    """Total position P&L in dollars at expiry, for a given underlying price."""
    legs = tuple(legs)
    return net_cash + payoff_per_share(legs, underlying_price) * CONTRACT_MULTIPLIER * contracts


def worst_case_loss(
    legs: Iterable[OptionLeg],
    contracts: int,
    net_cash: float,
) -> float | None:
    """Largest possible loss at expiry, as a positive dollar figure.

    Returns ``None`` when the loss is unbounded, which the gate treats as an
    outright rejection rather than as a large number. A structure whose risk
    cannot be stated is not one this agent is allowed to hold.

    A structure that cannot lose money returns ``0.0``.
    """
    legs = tuple(legs)
    if not legs:
        raise ValueError("cannot evaluate an empty structure")

    if slope_at_infinity(legs) < 0:
        return None

    # Profit is piecewise linear with kinks only at strikes, so the minimum over
    # the bounded region is attained at zero or at one of the strikes. Beyond the
    # highest strike the slope is >= 0, so nothing lower lies out there.
    candidates = [0.0, *sorted({leg.strike for leg in legs})]
    worst_profit = min(profit_at(legs, contracts, net_cash, price) for price in candidates)

    return max(-worst_profit, 0.0)


def max_gain(
    legs: Iterable[OptionLeg],
    contracts: int,
    net_cash: float,
) -> float | None:
    """Best case at expiry in dollars, or ``None`` if unbounded upside.

    Used for the reward-to-risk sanity check, so the agent does not sell
    premium so thin that a single loss undoes a week of wins.
    """
    legs = tuple(legs)
    if not legs:
        raise ValueError("cannot evaluate an empty structure")

    if slope_at_infinity(legs) > 0:
        return None

    candidates = [0.0, *sorted({leg.strike for leg in legs})]
    return max(profit_at(legs, contracts, net_cash, price) for price in candidates)
