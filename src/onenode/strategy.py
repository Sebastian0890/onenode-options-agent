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
from .risk.payoff import worst_case_loss

DEFAULT_WIDTHS: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)

MAX_MENU = 25
"""How many candidates the Proposer is shown. Lives here rather than in the
proposer, because what the ranking does only matters over the slice that is
actually seen - and calibration has to measure that same slice."""


class Expectancy:
    """How a structure prices up against the odds the chain is quoting.

    These numbers are shown to the Proposer and written to the journal. They
    are deliberately **not** a gate, and the reason is worth stating plainly,
    because the obvious version of this check is wrong.

    Options are priced so that the risk-neutral expected value of any
    structure is about zero before costs. So a filter on "expected value above
    zero", built from delta, measures the bid-ask spread and nothing else - it
    would refuse every trade ever quoted, including the good ones. A negative
    ``edge`` here is not a discovery about the market; on the calibration chain
    it came to -5.0%, which is almost exactly what crossing the spread costs.

    What the seller is actually paid for does not appear in delta at all: it is
    the gap between implied and realised volatility, and delta is computed from
    implied. Measuring that needs history, not arithmetic on a snapshot.

    So what these are good for is comparison. Between two candidates, the one
    whose credit sits better against its own quoted odds is the better-priced
    trade, and that ordering is real even though its zero point is not.

    The gate that does work on a snapshot is ``execution_drag``, below: how much
    of the theoretical credit is handed to the spread on the way in. That one
    measures a cost that is certain rather than an edge that is not.
    """

    credit_per_contract: float
    mid_credit_per_contract: float
    reward_to_risk: float
    short_delta: float

    @property
    def break_even_win_rate(self) -> float:
        """Share of trades that must win, charging a full loss for every loser.

        ``p * credit = (1 - p) * max_loss`` solves to ``1 / (1 + r/r)``. Real
        losses are often partial, so this asks for more than the trade needs.
        """
        if self.credit_per_contract <= 0 or self.reward_to_risk <= 0:
            return 1.0
        return 1.0 / (1.0 + self.reward_to_risk)

    @property
    def implied_win_rate(self) -> float:
        """What the chain says the odds are, from the delta of the short legs."""
        return max(0.0, 1.0 - self.tested_probability)

    @property
    def tested_probability(self) -> float:
        """Chance that a short strike finishes in the money."""
        return self.short_delta

    @property
    def edge(self) -> float:
        """Implied win rate minus the win rate needed to break even.

        Comparative only - see the class docstring for why its zero point is
        not the line between a good trade and a bad one.
        """
        return self.implied_win_rate - self.break_even_win_rate

    @property
    def execution_cost_per_contract(self) -> float:
        """What crossing the bid-ask costs, in dollars, on the way in.

        The difference between the credit at mid and the credit a marketable
        order actually collects. It is paid with certainty on every fill, which
        is more than can be said for any of the numbers above.
        """
        return max(0.0, self.mid_credit_per_contract - self.credit_per_contract)

    @property
    def execution_drag(self) -> float:
        """Execution cost as a share of the credit at mid.

        A structure whose spread eats a third of its own premium has to be
        right far more often to end up in the same place. This is the one
        figure here that a snapshot can settle honestly, so it is the one the
        candidate filter uses.
        """
        if self.mid_credit_per_contract <= 0:
            return 1.0
        return self.execution_cost_per_contract / self.mid_credit_per_contract


@dataclass(frozen=True)
class SpreadCandidate(Expectancy):
    """A defined-risk vertical credit spread built from two live quotes."""

    underlying: str
    right: Right
    expiry: date
    short_leg: OptionLeg
    long_leg: OptionLeg
    short_quote: ContractQuote
    long_quote: ContractQuote

    @property
    def legs(self) -> tuple[OptionLeg, ...]:
        """Every leg of the structure, short side first.

        The generic accessor everything downstream uses, so an order builder or
        a prompt never has to know whether it is looking at two legs or four.
        """
        return (self.short_leg, self.long_leg)

    @property
    def quotes(self) -> tuple[ContractQuote, ...]:
        return (self.short_quote, self.long_quote)

    @property
    def structure(self) -> str:
        return f"{self.right.value} credit spread"

    @property
    def width(self) -> float:
        return abs(self.short_leg.strike - self.long_leg.strike)

    @property
    def credit_per_contract(self) -> float:
        """Conservative fill: sell the short at the bid, buy the long at the ask."""
        return (self.short_quote.bid - self.long_quote.ask) * CONTRACT_MULTIPLIER

    @property
    def mid_credit_per_contract(self) -> float:
        """The same spread priced mid-to-mid: what it is theoretically worth.

        Never used to size or to submit an order - only to measure what the
        conservative fill above is giving up.
        """
        return (self.short_quote.mid - self.long_quote.mid) * CONTRACT_MULTIPLIER

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
            f"needs {self.break_even_win_rate:.0%} wins, chain implies "
            f"{self.implied_win_rate:.0%} (edge {self.edge:+.1%}) | "
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
            legs=self.legs,
            contracts=contracts,
            net_cash=self.credit_per_contract * contracts,
            quote_age_seconds=self.quote_age_seconds(now),
            worst_leg_spread_pct=self.worst_leg_spread_pct,
            rationale=rationale,
        )


def _strike_key(strike: float) -> int:
    """Strikes as integer thousandths, so float equality never decides a lookup."""
    return round(strike * 1000)


DEFAULT_TARGET_DELTA = 0.17
"""Short-strike delta the strategy aims at. Roughly a five-in-six chance of
expiring worthless, which is the shape of edge selling premium is built on."""

DELTA_BUCKET = 0.02
"""Granularity at which two short deltas count as the same risk.

Finer than this and the tiebreak below never gets to decide anything, because
no two candidates are ever quite equal; coarser and genuinely different strikes
get treated as interchangeable. Two cents of delta is also about the width of
the quote noise the greeks arrive with.
"""


def rank(target_delta: float = DEFAULT_TARGET_DELTA):
    """Ordering for the menu the model is shown. Best first.

    The obvious key is reward-to-risk, and it is wrong here for a reason that
    only appears when the ranking is measured against a real chain:
    **reward-to-risk rises monotonically with the short delta.** More delta is
    more premium is more credit against the same width. Ranking by it therefore
    sorts candidates by exactly the quantity the delta band exists to limit, and
    whatever sits at the band's upper edge wins every time.

    On the SPY chain of 2026-08-28 the target delta was 0.17 with a tolerance of
    0.08, and the top twenty-five by reward-to-risk had a median short delta of
    **0.241** - twenty of the twenty-five clustered at 0.22 and 0.24. The target
    was decorative. Worse, the Proposer's own instructions say the edge is win
    rate rather than size of win, and it was being handed a menu ranked by the
    opposite of that.

    Ranking by ``edge`` was tried next and rejected. It fixes the direction but
    is not stable: it favours low delta hard enough that the menu's median moved
    to 0.127 with credits of eight dollars, and where it landed depended on the
    reward-to-risk floor rather than on anything about the market - 0.198 at a
    floor of 0.10, 0.127 at 0.05. A ranking that moves when an unrelated
    threshold moves is not measuring what it claims to.

    So the ranking says what the strategy says: get near the target delta, and
    among candidates that are equally near it, take the one that pays best.
    Measured on the same chain, the menu's median delta became 0.179 at a floor
    of 0.05 and 0.198 at 0.10 - close to target under both, which is the
    property the other two orderings lacked.
    """

    def key(candidate: SpreadCandidate | IronCondorCandidate) -> tuple[float, float, float]:
        distance = round(abs(candidate.short_delta - target_delta) / DELTA_BUCKET)
        return (distance, -candidate.reward_to_risk, candidate.max_loss_per_contract)

    return key


def build_credit_spreads(
    quotes: Iterable[ContractQuote],
    *,
    underlying: str,
    right: Right = Right.PUT,
    target_delta: float = DEFAULT_TARGET_DELTA,
    delta_tolerance: float = 0.08,
    widths: Sequence[float] = DEFAULT_WIDTHS,
    max_spread_pct: float = 10.0,
    min_reward_to_risk: float = 0.10,
    max_execution_drag: float = 0.30,
    max_days_to_expiry: int | None = 7,
    last_expiry: date | None = None,
    today: date | None = None,
) -> list[SpreadCandidate]:
    """Enumerate every tradeable credit spread that fits the strategy.

    The short strike is picked by delta rather than by implied volatility rank:
    the chain snapshot carries greeks but no IV field, so delta is the only
    probability-like measure actually available. At a short delta near 0.17 the
    contract expires worthless roughly five times in six, which is the shape of
    edge this strategy is built on.

    Returns candidates nearest the target delta first, best-paid among
    equals. Not by reward-to-risk: see ``rank`` for why that ordering quietly
    overrides the delta target it is supposed to work within.
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
        # A hard last date as well as a rolling window: the competition ends on
        # a fixed day, and nothing should still be open when the account is read.
        if last_expiry is not None and expiry > last_expiry:
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
            # A spread that hands a third of its premium to the bid-ask
            # on the way in is paying a certain cost for an uncertain
            # edge. That is decidable from the snapshot, so it is
            # decided here rather than left to the model.
            if candidate.execution_drag > max_execution_drag:
                continue

            candidates.append(candidate)

    return sorted(candidates, key=rank(target_delta))


@dataclass(frozen=True)
class IronCondorCandidate(Expectancy):
    """A put credit spread and a call credit spread on the same expiry.

    Worth building because only one side can finish in the money. The two
    spreads collect two credits but the structure risks roughly one width, so
    the reward-to-risk of the pair beats either side alone - measured against
    the live SPY chain on 2026-08-28, roughly threefold.

    That is not free money, and it should not be described as such. A condor
    with both shorts near 0.23 delta finishes fully profitable maybe 54% of the
    time, against roughly 77% for the put spread alone: the better ratio is
    paid for with a lower win rate, because options are priced by people who
    can also do this arithmetic. What the structure genuinely buys is capital
    efficiency - the same risk budget carries far more credit - and the
    50%-profit exit rule collects on that long before expiry decides anything.

    The worst case is not assumed to be "the wider wing minus the credit". It
    is measured by the same payoff engine the hard gate uses, so the number
    quoted here and the number the gate enforces agree by construction rather
    than by two implementations happening to match.
    """

    underlying: str
    expiry: date
    put_side: SpreadCandidate
    call_side: SpreadCandidate

    @property
    def legs(self) -> tuple[OptionLeg, ...]:
        return self.put_side.legs + self.call_side.legs

    @property
    def quotes(self) -> tuple[ContractQuote, ...]:
        return self.put_side.quotes + self.call_side.quotes

    @property
    def structure(self) -> str:
        return "iron condor"

    @property
    def credit_per_contract(self) -> float:
        return self.put_side.credit_per_contract + self.call_side.credit_per_contract

    @property
    def mid_credit_per_contract(self) -> float:
        return self.put_side.mid_credit_per_contract + self.call_side.mid_credit_per_contract

    @property
    def max_loss_per_contract(self) -> float:
        loss = worst_case_loss(self.legs, 1, self.credit_per_contract)
        # Both wings are long-protected, so the structure is always bounded.
        # If that ever stops being true the caller should hear about it rather
        # than receive a plausible number.
        if loss is None:  # pragma: no cover - unreachable for a well-formed condor
            raise ValueError(f"iron condor {self.key} has unbounded risk")
        return loss

    @property
    def reward_to_risk(self) -> float:
        risk = self.max_loss_per_contract
        return self.credit_per_contract / risk if risk > 0 else 0.0

    @property
    def short_delta(self) -> float:
        """The riskier of the two short strikes - the one likely to be tested."""
        return max(self.put_side.short_delta, self.call_side.short_delta)

    @property
    def tested_probability(self) -> float:
        """Chance that either wing finishes in the money.

        The two outcomes are mutually exclusive at expiry - the underlying
        cannot close below the put strike and above the call strike - so the
        probabilities add instead of compounding. This is the number that makes
        a condor a different bet from the put spread inside it: two credits,
        but also two ways to be wrong.
        """
        return min(1.0, self.put_side.short_delta + self.call_side.short_delta)

    @property
    def worst_leg_spread_pct(self) -> float:
        return max(quote.spread_pct_of_mid for quote in self.quotes)

    def quote_age_seconds(self, now: datetime | None = None) -> float:
        return max(quote.age_seconds(now) for quote in self.quotes)

    @property
    def key(self) -> str:
        return f"{self.put_side.key}+{self.call_side.key}"

    def describe(self) -> str:
        return (
            f"{self.key} | iron condor | exp {self.expiry} | "
            f"put {self.put_side.short_leg.strike:g}/{self.put_side.long_leg.strike:g} "
            f"call {self.call_side.short_leg.strike:g}/{self.call_side.long_leg.strike:g} | "
            f"credit ${self.credit_per_contract:.0f} | "
            f"risk ${self.max_loss_per_contract:.0f} | "
            f"r/r {self.reward_to_risk:.2f} | widest short delta {self.short_delta:.3f} | "
            f"needs {self.break_even_win_rate:.0%} wins, chain implies "
            f"{self.implied_win_rate:.0%} (edge {self.edge:+.1%}) | "
            f"worst leg spread {self.worst_leg_spread_pct:.1f}%"
        )

    def to_proposed_trade(
        self,
        contracts: int,
        rationale: str = "",
        now: datetime | None = None,
    ) -> ProposedTrade:
        return ProposedTrade(
            underlying=self.underlying,
            legs=self.legs,
            contracts=contracts,
            net_cash=self.credit_per_contract * contracts,
            quote_age_seconds=self.quote_age_seconds(now),
            worst_leg_spread_pct=self.worst_leg_spread_pct,
            rationale=rationale,
        )


def build_iron_condors(
    put_spreads: Sequence[SpreadCandidate],
    call_spreads: Sequence[SpreadCandidate],
    *,
    min_reward_to_risk: float = 0.20,
    max_execution_drag: float = 0.30,
    max_per_expiry: int = 3,
) -> list[IronCondorCandidate]:
    """Pair put and call spreads of the same expiry into iron condors.

    The reward-to-risk floor defaults higher than for a single vertical,
    because a condor that does not clearly beat its own put side is not worth
    the extra two legs of execution risk and commission.

    Only the best few per expiry are returned. Pairing every put with every
    call produces hundreds of near-identical structures, which buries the
    model in noise without adding a single distinct choice.
    """
    by_expiry: dict[date, list[IronCondorCandidate]] = {}

    for put_side in put_spreads:
        if put_side.right is not Right.PUT:
            continue
        for call_side in call_spreads:
            if call_side.right is not Right.CALL:
                continue
            if call_side.expiry != put_side.expiry:
                continue
            if call_side.underlying != put_side.underlying:
                continue
            # A short call strike at or below the short put strike is an
            # inverted condor: the wings overlap and both sides can lose.
            if call_side.short_leg.strike <= put_side.short_leg.strike:
                continue

            condor = IronCondorCandidate(
                underlying=put_side.underlying,
                expiry=put_side.expiry,
                put_side=put_side,
                call_side=call_side,
            )
            if condor.reward_to_risk < min_reward_to_risk:
                continue
            if condor.execution_drag > max_execution_drag:
                continue
            by_expiry.setdefault(condor.expiry, []).append(condor)

    out: list[IronCondorCandidate] = []
    for expiry in sorted(by_expiry):
        ranked = sorted(by_expiry[expiry], key=rank())
        out.extend(ranked[:max_per_expiry])

    return sorted(out, key=rank())


def size_position(
    candidate: SpreadCandidate | IronCondorCandidate,
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
