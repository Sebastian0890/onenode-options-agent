"""The Risk Officer: an independent reviewer whose only power is to say no.

Two things make this more than theatre. It runs on a different model lineage
from the Proposer, so it does not inherit the same blind spots. And it never
sees the Proposer's rationale - only the trade and the account - so it forms its
own view instead of grading an argument it has already been persuaded by.

Its prompt is adversarial by construction: find the reason not to, and refuse
when uncertain. That asymmetry is deliberate. A reviewer that splits the
difference approves everything eventually.

Which model answers is decided in ``llm.py``. When the only model left belongs
to the same family that proposed the trade, the review still happens - an
isolated context is worth something - but the verdict is stamped ``degraded``,
because a second opinion from the same lineage is not the thing this layer
claims to provide.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from ..risk.models import ProposedTrade, Side
from ..strategy import IronCondorCandidate, SpreadCandidate
from . import llm

ROLE_ENV = "ONENODE_REVIEWER_PROVIDER"

SYSTEM = """You are the risk officer for an autonomous options trading agent \
on a $100,000 paper account. A trade has been proposed. Your job is to find the \
reason it should not be placed.

You are not asked whether the trade is reasonable. You are asked what is wrong \
with it. Look for: the short strike sitting where the underlying could \
plausibly reach before expiry, credit too thin for the risk taken, an account \
already carrying more exposure than a new position should be added to, timing \
too near expiry for the position to be managed.

If you are uncertain, reject. A window missed costs nothing; a bad position \
held over a weekend costs real money. There will be another candidate in \
fifteen minutes.

Approve only when you genuinely cannot find a substantive objection.

Answer with JSON only, no prose and no code fences:
{"approve": true or false, "reason": "one or two sentences"}"""


class RiskVerdict(BaseModel):
    """The reviewer's answer, plus the receipt for who produced it."""

    approve: bool
    reason: str
    reviewer: str = Field(default="", description="Backend that produced this verdict.")
    degraded: bool = Field(
        default=False,
        description="True when the reviewer shares a model family with the proposer.",
    )


def nearest_short_distance(candidate: SpreadCandidate | IronCondorCandidate, spot: float) -> float:
    """Distance from spot to the closest short strike.

    For a condor this is the wing that gets tested first, which is the number
    that actually matters when judging whether the structure is too tight.
    """
    shorts = [leg.strike for leg in candidate.legs if leg.side is Side.SELL]
    return min((abs(spot - strike) for strike in shorts), default=0.0)


def build_prompt(
    candidate: SpreadCandidate | IronCondorCandidate,
    trade: ProposedTrade,
    *,
    spot: float,
    equity: float,
    day_pnl_pct: float,
    open_positions: int,
    committed_risk: float,
    minutes_to_close: float,
    worst_case_loss: float,
) -> str:
    """Describe the trade and the account - and nothing about why it was chosen."""
    distance = nearest_short_distance(candidate, spot)
    pct = 100 * distance / spot if spot else 0.0
    return f"""Proposed trade
  {candidate.structure} on {candidate.underlying}
  {candidate.describe()}
  expiry {candidate.expiry}
  contracts: {trade.contracts}
  credit collected: ${trade.net_cash:,.2f}
  worst case loss: ${worst_case_loss:,.2f}
  nearest short strike delta: {candidate.short_delta:.3f}
  distance from spot to nearest short strike: {distance:,.2f} ({pct:.2f}%)

Market
  {candidate.underlying} spot: {spot:,.2f}
  minutes to close: {minutes_to_close:.0f}

Account
  equity: ${equity:,.2f}
  day P&L: {day_pnl_pct:+.2f}%
  open structures: {open_positions}
  risk already committed: ${committed_risk:,.2f}

Should this trade be placed?"""


def review_trade(
    candidate: SpreadCandidate | IronCondorCandidate,
    trade: ProposedTrade,
    *,
    spot: float,
    equity: float,
    day_pnl_pct: float,
    open_positions: int,
    committed_risk: float,
    minutes_to_close: float,
    worst_case_loss: float,
    proposer_family: str = "",
    timeout: float = 40.0,
    ask=llm.ask,
) -> RiskVerdict:
    """Get an independent verdict on a proposed trade.

    Any failure of the reviewer is a veto, not a pass. If the second opinion
    cannot be obtained, the trade does not happen - an unreviewed trade is
    exactly what this layer exists to prevent. The same goes for a reply that
    arrives but cannot be read: an unparseable verdict is not an approval.
    """
    prompt = build_prompt(
        candidate,
        trade,
        spot=spot,
        equity=equity,
        day_pnl_pct=day_pnl_pct,
        open_positions=open_positions,
        committed_risk=committed_risk,
        minutes_to_close=minutes_to_close,
        worst_case_loss=worst_case_loss,
    )

    try:
        reply = ask(
            system=SYSTEM,
            user=prompt,
            role_env=ROLE_ENV,
            exclude_families=(proposer_family,) if proposer_family else (),
            max_tokens=512,
            timeout=timeout,
        )
    except llm.LLMUnavailable as exc:
        return RiskVerdict(
            approve=False, reason=f"No reviewer available: {exc}", reviewer="unavailable"
        )

    try:
        verdict = RiskVerdict.model_validate({**reply.payload, "reviewer": reply.label})
    except ValidationError as exc:
        return RiskVerdict(
            approve=False,
            reason=(
                f"Reviewer reply could not be read ({exc.error_count()} errors); "
                f"treating as a veto."
            ),
            reviewer=reply.label,
        )

    verdict.degraded = bool(proposer_family) and reply.family == proposer_family
    return verdict
