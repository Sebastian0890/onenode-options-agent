"""When to close a position that is already open.

Opening trades is the part that feels like the strategy. Closing them is where
the money actually is, and it needs no model at all - the four rules below are
arithmetic on the position and the calendar, so they run identically whether
the market is calm, the API is slow, or nobody is watching.

The expiry rule is the one that matters most. A short spread carried into its
final hour stops being a probability bet and becomes a coin flip with
assignment risk attached, and the agent runs unattended during the trading day.
It closes rather than finds out.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .portfolio import PositionGroup


@dataclass(frozen=True)
class ExitRules:
    profit_target: float = 0.50
    """Close once this fraction of the maximum profit is banked.

    Half the credit is typically earned in well under half the time. Holding
    for the rest means risking the whole width to collect what is left.
    """

    stop_multiple: float = 2.0
    """Close when the loss reaches this multiple of the credit received."""

    close_on_expiry_day_minutes: float = 90.0
    """Flatten anything expiring today once the session is inside this window."""

    max_days_held_past_expiry: int = 0
    """Anything at or past expiry is closed on sight."""


DEFAULT_EXIT_RULES = ExitRules()


@dataclass(frozen=True)
class ExitDecision:
    should_close: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.should_close


HOLD = ExitDecision(should_close=False)


def evaluate_exit(
    group: PositionGroup,
    *,
    today: date,
    minutes_to_close: float,
    rules: ExitRules = DEFAULT_EXIT_RULES,
) -> ExitDecision:
    """Decide whether an open structure should be closed now.

    Pure function of the position and the clock. Rules are checked in order of
    urgency: expiry first, because a position that must be closed today is not
    made safe by being profitable.
    """
    dte = group.days_to_expiry(today)

    if dte < 0:
        return ExitDecision(True, f"expired {abs(dte)} day(s) ago and still open")

    if dte <= rules.max_days_held_past_expiry and minutes_to_close <= (
        rules.close_on_expiry_day_minutes
    ):
        return ExitDecision(
            True,
            f"expires today with {minutes_to_close:.0f}min left; closing rather than "
            "carrying assignment risk into the bell",
        )

    credit = group.credit_received
    if credit <= 0:
        # Not a credit structure - the profit-capture rules do not apply, and
        # nothing here knows what its target should be. Leave it alone and say so.
        return HOLD

    capture = group.profit_capture
    if capture >= rules.profit_target:
        return ExitDecision(
            True,
            f"captured {100 * capture:.0f}% of max profit "
            f"(${group.unrealized_pl:,.2f} of ${credit:,.2f})",
        )

    if group.unrealized_pl <= -rules.stop_multiple * credit:
        return ExitDecision(
            True,
            f"loss ${abs(group.unrealized_pl):,.2f} reached "
            f"{rules.stop_multiple:g}x the ${credit:,.2f} credit",
        )

    return HOLD


def positions_to_close(
    groups: list[PositionGroup],
    *,
    today: date,
    minutes_to_close: float,
    rules: ExitRules = DEFAULT_EXIT_RULES,
) -> list[tuple[PositionGroup, str]]:
    """Every open structure that meets an exit rule, with the reason."""
    out = []
    for group in groups:
        decision = evaluate_exit(group, today=today, minutes_to_close=minutes_to_close, rules=rules)
        if decision.should_close:
            out.append((group, decision.reason))
    return out
