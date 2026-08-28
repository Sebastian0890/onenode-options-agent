"""The two model layers: one that proposes, one that objects.

Neither can place a trade. The Proposer picks from a menu the code built; the
Risk Officer can only say no. Approval belongs solely to the hard gate, which
runs no model at all.

The Risk Officer deliberately runs on a different model family from the
Proposer, and never sees the Proposer's reasoning. A reviewer that inherits the
argument it is meant to check is not a reviewer.
"""

from .proposer import ProposerDecision, propose_trade
from .risk_officer import RiskVerdict, review_trade

__all__ = ["ProposerDecision", "RiskVerdict", "propose_trade", "review_trade"]
