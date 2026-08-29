"""Tests for the two model-driven roles.

Neither role is trusted, and these tests are about what happens when a model
misbehaves rather than when it cooperates. Every route through here has the same
destination: a model that answers badly must produce "no trade", never a trade
on a guess. A test that only checks the happy path would pass on a version of
this code that trades on garbage.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from onenode.agents import llm
from onenode.agents.proposer import propose_trade
from onenode.agents.risk_officer import review_trade
from onenode.broker.cli import ContractQuote
from onenode.risk.models import Right
from onenode.strategy import build_credit_spreads

TODAY = date(2026, 8, 28)
QUOTED_AT = datetime(2026, 8, 28, 18, 42, tzinfo=UTC)


def quote(strike: int, bid: float, ask: float, delta: float) -> ContractQuote:
    return ContractQuote(
        symbol=f"SPY260831P{strike * 1000:08d}",
        bid=bid,
        ask=ask,
        quoted_at=QUOTED_AT,
        delta=delta,
    )


PUT_CHAIN = [
    quote(766, 1.60, 1.64, -0.24),
    quote(764, 1.20, 1.24, -0.17),
    quote(763, 1.02, 1.06, -0.15),
    quote(759, 0.56, 0.60, -0.09),
]


@pytest.fixture
def candidates():
    built = build_credit_spreads(PUT_CHAIN, underlying="SPY", right=Right.PUT, today=TODAY)
    assert built, "fixture chain must produce at least one candidate"
    return built


MARKET = {
    "underlying": "SPY",
    "spot": 769.0,
    "day_pnl_pct": 0.1,
    "equity": 100_000.0,
    "open_positions": 0,
    "committed_risk": 0.0,
    "minutes_to_close": 180.0,
}


def replying(payload, *, model="gemini-2.5-flash", provider="gemini"):
    """An ``ask`` that returns one canned payload, ignoring the prompt."""

    def _ask(**kwargs):
        return llm.Reply(payload=payload, provider=provider, model=model)

    return _ask


def failing(message="every provider is down"):
    def _ask(**kwargs):
        raise llm.LLMUnavailable(message)

    return _ask


class TestProposer:
    def test_records_which_model_actually_chose(self, candidates):
        chosen = candidates[0].key
        decision = propose_trade(
            candidates,
            **MARKET,
            ask=replying(
                {"action": "trade", "candidate_key": chosen, "rationale": "delta is comfortable"}
            ),
        )
        assert decision.action == "trade"
        assert decision.proposer == "gemini:gemini-2.5-flash"
        assert decision.family == "gemini"

    def test_a_key_that_is_not_on_the_menu_stands_aside(self, candidates):
        """A near-miss is still a miss. Correcting it would mean guessing."""
        decision = propose_trade(
            candidates,
            **MARKET,
            ask=replying(
                {"action": "trade", "candidate_key": "SPY999999P00001000", "rationale": "this one"}
            ),
        )
        assert decision.action == "stand_aside"
        assert "was not on the menu" in decision.rationale

    def test_a_reply_of_the_wrong_shape_stands_aside(self, candidates):
        decision = propose_trade(candidates, **MARKET, ask=replying({"verdict": "buy everything"}))
        assert decision.action == "stand_aside"
        assert decision.proposer == "gemini:gemini-2.5-flash"

    def test_an_empty_menu_never_reaches_a_model(self):
        """Nothing to choose from is answered by arithmetic, not by tokens."""
        decision = propose_trade(
            [], **MARKET, ask=lambda **kw: pytest.fail("a model was called with no candidates")
        )
        assert decision.action == "stand_aside"

    def test_an_unreachable_model_propagates(self, candidates):
        """The run loop turns this into a journalled failure; swallowing it here
        would hide an outage behind an ordinary-looking pass."""
        with pytest.raises(llm.LLMUnavailable):
            propose_trade(candidates, **MARKET, ask=failing())


REVIEW = {
    "spot": 769.0,
    "equity": 100_000.0,
    "day_pnl_pct": 0.1,
    "open_positions": 0,
    "committed_risk": 0.0,
    "minutes_to_close": 180.0,
    "worst_case_loss": 380.0,
}


class TestRiskOfficer:
    def test_approves_when_the_reviewer_finds_nothing_wrong(self, candidates):
        candidate = candidates[0]
        verdict = review_trade(
            candidate,
            candidate.to_proposed_trade(1),
            **REVIEW,
            proposer_family="gemini",
            ask=replying(
                {"approve": True, "reason": "strike is far enough out"},
                provider="groq",
                model="openai/gpt-oss-120b",
            ),
        )
        assert verdict.approve
        assert verdict.reviewer == "groq:openai/gpt-oss-120b"
        assert not verdict.degraded

    def test_a_reviewer_of_the_proposer_s_own_family_is_marked_degraded(self, candidates):
        """It still reviews - an isolated context is worth something - but the
        claim of an independent second opinion is not made."""
        candidate = candidates[0]
        verdict = review_trade(
            candidate,
            candidate.to_proposed_trade(1),
            **REVIEW,
            proposer_family="gemini",
            ask=replying({"approve": True, "reason": "looks fine"}),
        )
        assert verdict.approve
        assert verdict.degraded

    def test_an_unreachable_reviewer_is_a_veto(self, candidates):
        candidate = candidates[0]
        verdict = review_trade(
            candidate, candidate.to_proposed_trade(1), **REVIEW, ask=failing("all hosts refused")
        )
        assert not verdict.approve
        assert "all hosts refused" in verdict.reason
        assert verdict.reviewer == "unavailable"

    def test_an_unreadable_verdict_is_a_veto(self, candidates):
        """Absence of a readable "no" is not a "yes"."""
        candidate = candidates[0]
        verdict = review_trade(
            candidate,
            candidate.to_proposed_trade(1),
            **REVIEW,
            ask=replying({"thoughts": "I am not sure what you want"}),
        )
        assert not verdict.approve
        assert "treating as a veto" in verdict.reason

    def test_the_reviewer_is_never_shown_why_the_trade_was_chosen(self, candidates):
        """Independence of context: the reviewer forms its own view rather than
        grading an argument it has already been given."""
        candidate = candidates[0]
        trade = candidate.to_proposed_trade(1, rationale="I love this trade, it is perfect")
        seen: dict = {}

        def _ask(**kwargs):
            seen.update(kwargs)
            return llm.Reply(payload={"approve": False, "reason": "no"}, provider="groq", model="x")

        review_trade(candidate, trade, **REVIEW, ask=_ask)
        assert "I love this trade" not in seen["user"]

    def test_the_proposer_s_family_is_the_one_excluded(self, candidates):
        candidate = candidates[0]
        seen: dict = {}

        def _ask(**kwargs):
            seen.update(kwargs)
            return llm.Reply(payload={"approve": False, "reason": "no"}, provider="groq", model="x")

        review_trade(
            candidate, candidate.to_proposed_trade(1), **REVIEW, proposer_family="claude", ask=_ask
        )
        assert seen["exclude_families"] == ("claude",)
