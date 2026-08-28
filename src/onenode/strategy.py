"""Building the menu of trades the agent is allowed to consider.

A language model asked to name an option contract will eventually name one that
does not exist, or quote a price nobody is offering. So it never gets to. This
module enumerates vertical credit spreads out of contracts that are really in
the chain, really quoted, and really tradeable, and the model's job is reduced
to choosing among them and explaining why.

That is a deliberate narrowing. It removes an entire class of failure - the
hallucinated symbol, the imagined fill - without removing the judgement that
makes the agent worth having.

Pricing here is deliberately pessimistic. A credit spread is priced as
``short.bid - long.ask``: what a marketable order would actually collect,
not the mid-to-mid figure that flatters every backtest. If a trade only looks
good at mid, it is not a trade.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from .broker.cli import ContractQuote
from .risk.models import CONTRACT_MULTIPLIER, OptionLeg, ProposedTrade, Right, Side

DEFAULT_WIDTHS: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)


@dataclass(frozen=True)
class SpreadCandidate:
    """A defined-risk vertical credit spread built from two live quotes."""

    underlying: str
    right: Right
    expiry: date
    short_leg: OptionLeg
    long_leg: OptionLeg
    short_quote: ContractQuote
    long_quote: ContractQuote

    @property
    def width(self) -> float:
        return abs(self.short_leg.strike - self.long_leg.strike)

    @property
    def credit_per_contract(self) -> float:
        """Conservative fill: sell the short at the bid, buy the long at the ask."""
        return (self.short_quote.bid - self.long_quote.ask) * CONTRACT_MULTIPLIER

    @property
    def max_loss_per_contract(self) -> float:
        return self.width * CONTRACT_MULTIPLIER - self.credit_per_contract

    @property
    def reward_to_risk(self) -> float:
        if self.max_loss_per_contract <= 0:
            return 0.0
        return self.credit_per_contract / self.max_loss_per_contract

    @property
    def short_delta(self) -> float:
        return abs(self.short_quote.delta or 0.0)

    @property
    def worst_leg_spread_pct(self) -> float:
        return max(self.short_quote.spread_pct_of_mid, self.long_quote.spread_pct_of_mid)

    def quote_age_seconds(self, now: datetime | None = None) -> float:
        return max(self.short_quote.age_seconds(now), self.long_quote.age_seconds(now))

    @property
    def key(self) -> str:
        """Stable identifier the model refers to when it picks a candidate."""
        return f"{self.short_leg.symbol}/{self.long_leg.symbol}"

    def describe(self) -> str:
        """One line, aimed at a model choosing between a few dozen of these."""
        return (
            f"{self.key} | {self.right.value} credit spread | exp {self.expiry} | "
            f"short {self.short_leg.strike:g} / long {self.long_leg.strike:g} | "
            f"width ${self.width:g} | credit ${self.credit_per_contract:.0f} | "
            f"risk ${self.max_loss_per_contract:.0f} | "
            f"r/r {self.reward_to_risk:.2f} | short delta {self.short_delta:.3f} | "
            f"worst leg spread {self.worst_leg_spread_pct:.1f}%"
        )

    def to_proposed_trade(
        self,
        contracts: int,
        rationale: str = "",
        now: datetime | None = None,
    ) -> ProposedTrade:
        """Hand this candidate to the hard gate for a verdict."""
        return ProposedTrade(
            underlying=self.underlying,
            legs=(self.short_leg, self.long_leg),
            contracts=contracts,
            net_cash=self.credit_per_contract * contracts,
            quote_age_seconds=self.quote_age_seconds(now),
            worst_leg_spread_pct=self.worst_leg_spread_pct,
            rationale=rationale,
        )


def _strike_key(strike: float) -> int:
    """Strikes as integer thousandths, so float equality never decides a lookup."""
    return round(strike * 1000)


def build_credit_spreads(
    quotes: Iterable[ContractQuote],
    *,
    underlying: str,
    right: Right = Right.PUT,
    target_delta: float = 0.17,
    delta_tolerance: float = 0.08,
    widths: Sequence[float] = DEFAULT_WIDTHS,
    max_spread_pct: float = 10.0,
    min_reward_to_risk: float = 0.10,
    max_days_to_expiry: int | None = 7,
    today: date | None = None,
) -> list[SpreadCandidate]:
    """Enumerate every tradeable credit spread that fits the strategy.

    The short strike is picked by delta rather than by implied volatility rank:
    the chain snapshot carries greeks but no IV field, so delta is the only
    probability-like measure actually available. At a short delta near 0.17 the
    contract expires worthless roughly five times in six, which is the shape of
    edge this strategy is built on.

    Returns candidates sorted by reward-to-risk, best first.
    """
    today = today or date.today()

    parsed: dict[tuple[date, int], tuple[OptionLeg, ContractQuote]] = {}
    for quote in quotes:
        try:
            leg = OptionLeg.from_symbol(quote.symbol, side=Side.SELL)
        except ValueError:
            continue
        if leg.right is not right or leg.root != underlying.upper():
            continue
        if not quote.is_tradeable:
            continue
        parsed[(leg.expiry, _strike_key(leg.strike))] = (leg, quote)

    # Puts are protected by a lower strike, calls by a higher one.
    direction = -1.0 if right is Right.PUT else 1.0

    candidates: list[SpreadCandidate] = []
    for (expiry, _), (short_leg, short_quote) in parsed.items():
        if max_days_to_expiry is not None and (expiry - today).days > max_days_to_expiry:
            continue
        if short_quote.delta is None:
            continue
        if abs(abs(short_quote.delta) - target_delta) > delta_tolerance:
            continue
        if short_quote.spread_pct_of_mid > max_spread_pct:
            continue

        for width in widths:
            protection = _strike_key(short_leg.strike + direction * width)
            found = parsed.get((expiry, protection))
            if found is None:
                continue

            long_leg_sell, long_quote = found
            if long_quote.spread_pct_of_mid > max_spread_pct:
                continue

            candidate = SpreadCandidate(
                underlying=underlying.upper(),
                right=right,
                expiry=expiry,
                short_leg=short_leg,
                long_leg=OptionLeg(
                    symbol=long_leg_sell.symbol,
                    side=Side.BUY,
                    ratio_qty=1,
                    right=long_leg_sell.right,
                    strike=long_leg_sell.strike,
                    expiry=long_leg_sell.expiry,
                ),
                short_quote=short_quote,
                long_quote=long_quote,
            )

            # A spread that collects nothing at a realistic fill is not a trade,
            # and one whose credit is a rounding error against its risk is worse:
            # it loses more in one bad week than it makes in a good month.
            if candidate.credit_per_contract <= 0:
                continue
            if candidate.reward_to_risk < min_reward_to_risk:
                continue

            candidates.append(candidate)

    return sorted(candidates, key=lambda c: (-c.reward_to_risk, c.max_loss_per_contract))


def size_position(
    candidate: SpreadCandidate,
    equity: float,
    max_risk_pct: float,
    max_contracts: int,
) -> int:
    """Largest contract count whose worst case still fits the per-trade budget.

    Sizing is derived from the risk budget rather than proposed by a model.
    Returns 0 when even one contract is too large, which the caller must treat
    as "no trade" rather than rounding up to one.
    """
    if candidate.max_loss_per_contract <= 0:
        return 0
    budget = equity * max_risk_pct / 100.0
    affordable = int(budget // candidate.max_loss_per_contract)
    return max(0, min(affordable, max_contracts))
