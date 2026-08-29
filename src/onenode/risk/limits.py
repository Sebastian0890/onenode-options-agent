"""The risk budget, in one place.

These numbers are the agent's constitution. They live in code rather than in a
prompt on purpose: a limit that sits in a prompt is a suggestion the model can
argue with, and over a week of unattended running it eventually will.

Sizing rationale (competition context: 4 trading days on a $100k paper
account). A 1.5% cap per trade with a 6% portfolio ceiling means the worst
conceivable day - every open position losing its maximum simultaneously - costs
6% of the account. The -3% daily stop halts trading long before that. The
account cannot be destroyed by a single bad session, which matters more than
squeezing out extra return: a blown-up account also destroys the demo, the
write-up, and three of the four judging criteria.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    # --- Position sizing -------------------------------------------------
    max_risk_per_trade_pct: float = 1.5
    """Worst-case loss of a single new trade, as a percent of equity."""

    max_portfolio_risk_pct: float = 6.0
    """Combined worst case of every open position plus the new one."""

    max_open_positions: int = 5

    max_contracts_per_order: int = 25
    """Blunt backstop against a misplaced decimal point in sizing."""

    # --- Loss limits -----------------------------------------------------
    daily_stop_pct: float = -3.0
    """Day P&L at or below this halts all new trades until the next session."""

    # --- Timing ----------------------------------------------------------
    min_minutes_to_close: float = 30.0
    """No new positions inside this window; liquidity thins and fills worsen."""

    min_days_to_expiry: int = 0
    max_days_to_expiry: int = 7
    max_distinct_expiries: int = 1
    """Single-expiry structures only. Calendars need vega handling we do not do."""

    # --- Instrument selection --------------------------------------------
    allowed_underlyings: frozenset[str] = frozenset({"SPY", "QQQ", "IWM"})
    """Liquid index ETFs only: no earnings gaps, no single-name headline risk."""

    # --- Quote quality ----------------------------------------------------
    # The free Alpaca data plan serves an indicative options feed rather than
    # full OPRA, so the quote the agent reasons about is not guaranteed to be
    # the quote it trades against. These two gates refuse the cases where that
    # gap is most likely to bite.
    max_quote_age_seconds: float = 300.0
    max_spread_pct_of_mid: float = 10.0

    # --- Trade quality ----------------------------------------------------
    min_reward_to_risk: float = 0.10
    """Reject premium so thin that one loss undoes many wins."""

    max_execution_drag: float = 0.30
    """Most of a structure's theoretical credit that may be handed to the
    bid-ask spread on the way in.

    This replaced a filter that looked more principled and was not. Comparing
    the credit against the win rate implied by delta appears to test whether a
    trade is worth taking, but options are priced so that the risk-neutral
    expected value is about zero before costs - so that comparison measures the
    bid-ask spread and calls it edge. On the calibration chain it scored a
    perfectly ordinary SPY spread at -5.0%, which is almost exactly what
    crossing the spread cost. A gate on it would have refused every trade for
    the whole week and looked disciplined doing it.

    Execution cost is the part a snapshot can settle honestly: it is paid on
    every fill, in full, whatever happens next. Thirty percent is a wide floor
    on purpose - it is meant to catch the illiquid strike whose spread eats the
    trade, not to second-guess ordinary ones.

    The expectancy figures themselves are still computed and still shown to the
    Proposer and the journal, because the *ordering* they give between two
    candidates is real even where their zero point is not."""

    def risk_budget_per_trade(self, equity: float) -> float:
        return equity * self.max_risk_per_trade_pct / 100.0

    def risk_budget_portfolio(self, equity: float) -> float:
        return equity * self.max_portfolio_risk_pct / 100.0


DEFAULT_LIMITS = RiskLimits()
