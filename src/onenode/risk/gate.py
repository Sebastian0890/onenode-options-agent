"""The hard gate: the last thing between a proposal and a live order.

Every trade this agent places passes through :func:`evaluate`. No language
model is involved here and none can reach past it - the LLM layers propose and
veto, but only this function can approve. That asymmetry is the whole design:
the models are allowed to be creative, and are not allowed to be dangerous.

The gate collects every violation rather than returning on the first one, so a
rejection explains itself completely in the trade journal instead of one
problem at a time.
"""

from __future__ import annotations

from datetime import date

from .limits import DEFAULT_LIMITS, RiskLimits
from .models import (
    AccountSnapshot,
    GateDecision,
    MarketClock,
    ProposedTrade,
)
from .payoff import max_gain, worst_case_loss


def evaluate(
    trade: ProposedTrade,
    account: AccountSnapshot,
    clock: MarketClock,
    today: date,
    limits: RiskLimits = DEFAULT_LIMITS,
) -> GateDecision:
    """Decide whether ``trade`` may be sent to the broker.

    Pure function: same inputs, same verdict, no network, no clock of its own.
    That is what makes the risk policy testable rather than merely asserted.
    """
    violations: list[str] = []

    # --- Session state ---------------------------------------------------
    if not clock.is_open:
        violations.append("market is closed")
    elif clock.minutes_to_close < limits.min_minutes_to_close:
        violations.append(
            f"too close to the bell: {clock.minutes_to_close:.0f}min left, "
            f"minimum is {limits.min_minutes_to_close:.0f}min"
        )

    # --- Daily stop ------------------------------------------------------
    # Checked against realised day P&L, so a session that has already gone
    # badly cannot be argued back open by a confident-sounding proposal.
    if account.day_pnl_pct <= limits.daily_stop_pct:
        violations.append(
            f"daily stop hit: day P&L {account.day_pnl_pct:+.2f}% "
            f"at or past limit {limits.daily_stop_pct:+.2f}%"
        )

    # --- Instrument ------------------------------------------------------
    if trade.underlying.upper() not in limits.allowed_underlyings:
        allowed = ", ".join(sorted(limits.allowed_underlyings))
        violations.append(f"underlying {trade.underlying!r} not in allowlist ({allowed})")

    # --- Structure -------------------------------------------------------
    if len(trade.expiries) > limits.max_distinct_expiries:
        violations.append(
            f"{len(trade.expiries)} distinct expiries, "
            f"maximum is {limits.max_distinct_expiries}"
        )

    for expiry in trade.expiries:
        dte = (expiry - today).days
        if dte < limits.min_days_to_expiry:
            violations.append(f"expiry {expiry} is in the past ({dte} DTE)")
        elif dte > limits.max_days_to_expiry:
            violations.append(
                f"expiry {expiry} is {dte} DTE, maximum is {limits.max_days_to_expiry}"
            )

    if trade.contracts > limits.max_contracts_per_order:
        violations.append(
            f"{trade.contracts} contracts exceeds the per-order cap "
            f"of {limits.max_contracts_per_order}"
        )

    # --- Quote quality ---------------------------------------------------
    if trade.quote_age_seconds > limits.max_quote_age_seconds:
        violations.append(
            f"stale quote: {trade.quote_age_seconds:.0f}s old, "
            f"maximum is {limits.max_quote_age_seconds:.0f}s"
        )

    if trade.worst_leg_spread_pct > limits.max_spread_pct_of_mid:
        violations.append(
            f"wide market: worst leg spread is {trade.worst_leg_spread_pct:.1f}% of mid, "
            f"maximum is {limits.max_spread_pct_of_mid:.1f}%"
        )

    # --- Risk ------------------------------------------------------------
    # Computed from the legs, never read from the proposal. A model cannot
    # understate the risk of its own idea to get it past this point.
    loss = worst_case_loss(trade.legs, trade.contracts, trade.net_cash)

    if loss is None:
        violations.append("undefined risk: loss is unbounded, position is not permitted")
        return GateDecision.block(violations, worst_case_loss=None)

    budget_trade = limits.risk_budget_per_trade(account.equity)
    if loss > budget_trade:
        violations.append(
            f"trade risk ${loss:,.2f} exceeds per-trade budget ${budget_trade:,.2f} "
            f"({limits.max_risk_per_trade_pct:.2f}% of ${account.equity:,.2f})"
        )

    budget_portfolio = limits.risk_budget_portfolio(account.equity)
    committed = account.committed_risk + loss
    if committed > budget_portfolio:
        violations.append(
            f"portfolio risk ${committed:,.2f} (${account.committed_risk:,.2f} open "
            f"+ ${loss:,.2f} new) exceeds ceiling ${budget_portfolio:,.2f}"
        )

    if account.open_positions >= limits.max_open_positions:
        violations.append(
            f"{account.open_positions} positions already open, "
            f"maximum is {limits.max_open_positions}"
        )

    # --- Trade quality ---------------------------------------------------
    gain = max_gain(trade.legs, trade.contracts, trade.net_cash)
    if gain is not None and loss > 0:
        reward_to_risk = gain / loss
        if reward_to_risk < limits.min_reward_to_risk:
            violations.append(
                f"reward-to-risk {reward_to_risk:.3f} below floor "
                f"{limits.min_reward_to_risk:.3f} (max gain ${gain:,.2f} "
                f"against ${loss:,.2f} risk)"
            )

    if violations:
        return GateDecision.block(violations, worst_case_loss=loss)
    return GateDecision.allow(loss)
