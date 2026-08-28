"""Data models for the risk layer.

Deliberately pure data with no I/O: the hard gate must be testable without a
network connection, an API key, or an LLM. Everything the gate needs to make a
decision arrives as one of these frozen structures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

CONTRACT_MULTIPLIER = 100
"""Shares per US equity option contract."""


class Right(str, Enum):
    CALL = "call"
    PUT = "put"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


# Alpaca uses the compact OCC symbol form, e.g. "SPY260904P00450000":
#   root (1-6 chars) | expiry YYMMDD | C/P | strike * 1000, zero padded to 8
_OCC_RE = re.compile(r"^(?P<root>[A-Z]{1,6})(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")


@dataclass(frozen=True)
class OptionLeg:
    """A single leg of a (possibly multi-leg) option order.

    Mirrors the shape Alpaca expects inside the ``legs`` array of an
    ``order_class="mleg"`` order, plus the parsed contract terms the gate needs
    in order to reason about the payoff itself.
    """

    symbol: str
    side: Side
    ratio_qty: int
    right: Right
    strike: float
    expiry: date

    def __post_init__(self) -> None:
        if self.ratio_qty <= 0:
            raise ValueError(f"ratio_qty must be positive, got {self.ratio_qty}")
        if self.strike <= 0:
            raise ValueError(f"strike must be positive, got {self.strike}")

    @classmethod
    def from_symbol(cls, symbol: str, side: Side, ratio_qty: int = 1) -> OptionLeg:
        """Build a leg by parsing an OCC option symbol.

        Parsing rather than trusting caller-supplied strike/right data is
        intentional: the symbol is what actually gets sent to the broker, so it
        is the only description of the contract that cannot drift out of sync
        with the order.
        """
        match = _OCC_RE.match(symbol.strip().upper())
        if match is None:
            raise ValueError(f"not a valid OCC option symbol: {symbol!r}")

        yy, mm, dd = (
            int(match["expiry"][0:2]),
            int(match["expiry"][2:4]),
            int(match["expiry"][4:6]),
        )
        return cls(
            symbol=symbol.strip().upper(),
            side=side,
            ratio_qty=ratio_qty,
            right=Right.CALL if match["right"] == "C" else Right.PUT,
            strike=int(match["strike"]) / 1000.0,
            expiry=date(2000 + yy, mm, dd),
        )

    @property
    def signed_ratio(self) -> int:
        """+ratio for long legs, -ratio for short legs."""
        return self.ratio_qty if self.side is Side.BUY else -self.ratio_qty


@dataclass(frozen=True)
class ProposedTrade:
    """A trade the agent wants to place, before any risk check has run.

    ``net_cash`` is the total dollar amount expected to hit the account on
    open: positive for a net credit, negative for a net debit. It covers the
    whole order (all contracts), not a single contract.
    """

    underlying: str
    legs: tuple[OptionLeg, ...]
    contracts: int
    net_cash: float
    # Quote quality context, needed because the free Alpaca data plan serves an
    # indicative feed rather than full OPRA - see MASTERPLAN section 6.3.
    quote_age_seconds: float
    worst_leg_spread_pct: float
    rationale: str = ""
    approvals: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("a trade must have at least one leg")
        if self.contracts <= 0:
            raise ValueError(f"contracts must be positive, got {self.contracts}")

    @property
    def strikes(self) -> tuple[float, ...]:
        return tuple(sorted({leg.strike for leg in self.legs}))

    @property
    def expiries(self) -> tuple[date, ...]:
        return tuple(sorted({leg.expiry for leg in self.legs}))


@dataclass(frozen=True)
class AccountSnapshot:
    """Account state as reported by Alpaca at the start of an agent run.

    ``committed_risk`` is the sum of the worst-case losses of every position
    already open, so the gate can enforce a portfolio-wide ceiling rather than
    only a per-trade one.
    """

    equity: float
    last_equity: float
    open_positions: int
    committed_risk: float

    @property
    def day_pnl(self) -> float:
        return self.equity - self.last_equity

    @property
    def day_pnl_pct(self) -> float:
        if self.last_equity <= 0:
            return 0.0
        return 100.0 * self.day_pnl / self.last_equity


@dataclass(frozen=True)
class MarketClock:
    """Trading-session context, sourced from Alpaca's /v2/clock endpoint.

    Taking this from the broker rather than computing it locally means market
    holidays and early closes are handled without a calendar of our own.
    """

    is_open: bool
    minutes_to_close: float


@dataclass(frozen=True)
class GateDecision:
    """The verdict of the hard gate. ``violations`` is empty iff approved."""

    approved: bool
    violations: tuple[str, ...]
    worst_case_loss: float | None

    @classmethod
    def allow(cls, worst_case_loss: float) -> GateDecision:
        return cls(approved=True, violations=(), worst_case_loss=worst_case_loss)

    @classmethod
    def block(cls, violations: list[str], worst_case_loss: float | None) -> GateDecision:
        return cls(
            approved=False,
            violations=tuple(violations),
            worst_case_loss=worst_case_loss,
        )

    def __str__(self) -> str:
        if self.approved:
            return f"APPROVED (worst case ${self.worst_case_loss:,.2f})"
        return "BLOCKED: " + "; ".join(self.violations)
