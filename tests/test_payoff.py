"""Tests for the expiry-payoff engine.

Each case pins a number that can be checked by hand, because this module is
what the gate trusts instead of trusting the model.
"""

from __future__ import annotations

from datetime import date

import pytest

from onenode.risk.models import OptionLeg, Right, Side
from onenode.risk.payoff import max_gain, profit_at, slope_at_infinity, worst_case_loss

EXPIRY = date(2026, 9, 4)


def leg(strike: int, right: str, side: Side, ratio: int = 1) -> OptionLeg:
    symbol = f"SPY260904{right}{strike * 1000:08d}"
    return OptionLeg.from_symbol(symbol, side=side, ratio_qty=ratio)


class TestSymbolParsing:
    def test_parses_occ_symbol(self):
        parsed = OptionLeg.from_symbol("SPY260904P00450000", side=Side.SELL)
        assert parsed.right is Right.PUT
        assert parsed.strike == pytest.approx(450.0)
        assert parsed.expiry == EXPIRY
        assert parsed.signed_ratio == -1

    def test_parses_fractional_strike(self):
        parsed = OptionLeg.from_symbol("IWM260904C00212500", side=Side.BUY)
        assert parsed.strike == pytest.approx(212.5)
        assert parsed.signed_ratio == 1

    @pytest.mark.parametrize(
        "symbol",
        [
            "SPY260904X00450000",  # not a call or put
            "SPY2609040045000",  # too short
            "spy260904p00450000extra",  # trailing junk
            "260904P00450000",  # no root
        ],
    )
    def test_rejects_malformed_symbols(self, symbol):
        with pytest.raises(ValueError):
            OptionLeg.from_symbol(symbol, side=Side.BUY)

    def test_rejects_nonpositive_ratio(self):
        with pytest.raises(ValueError):
            OptionLeg.from_symbol("SPY260904P00450000", side=Side.BUY, ratio_qty=0)


class TestBullPutSpread:
    """Sell the 450 put, buy the 445 put, collect $1.00 per share."""

    legs = (leg(450, "P", Side.SELL), leg(445, "P", Side.BUY))
    net_cash = 100.0  # one contract

    def test_worst_case_is_width_minus_credit(self):
        assert worst_case_loss(self.legs, 1, self.net_cash) == pytest.approx(400.0)

    def test_max_gain_is_the_credit(self):
        assert max_gain(self.legs, 1, self.net_cash) == pytest.approx(100.0)

    def test_risk_scales_with_contracts(self):
        assert worst_case_loss(self.legs, 5, 500.0) == pytest.approx(2000.0)

    def test_profitable_above_the_short_strike(self):
        assert profit_at(self.legs, 1, self.net_cash, 460.0) == pytest.approx(100.0)

    def test_full_loss_below_the_long_strike(self):
        assert profit_at(self.legs, 1, self.net_cash, 440.0) == pytest.approx(-400.0)

    def test_breaks_even_at_short_strike_minus_credit(self):
        assert profit_at(self.legs, 1, self.net_cash, 449.0) == pytest.approx(0.0)


class TestIronCondor:
    """Short 445/440 put spread and short 460/465 call spread for $2.00."""

    legs = (
        leg(445, "P", Side.SELL),
        leg(440, "P", Side.BUY),
        leg(460, "C", Side.SELL),
        leg(465, "C", Side.BUY),
    )
    net_cash = 200.0

    def test_risk_is_bounded_on_both_wings(self):
        assert slope_at_infinity(self.legs) == 0

    def test_worst_case_is_one_wing_minus_credit(self):
        assert worst_case_loss(self.legs, 1, self.net_cash) == pytest.approx(300.0)

    def test_max_gain_is_the_credit(self):
        assert max_gain(self.legs, 1, self.net_cash) == pytest.approx(200.0)

    def test_keeps_full_credit_between_the_short_strikes(self):
        assert profit_at(self.legs, 1, self.net_cash, 452.0) == pytest.approx(200.0)

    @pytest.mark.parametrize("price", [0.0, 300.0, 440.0, 465.0, 900.0])
    def test_never_loses_more_than_the_worst_case(self, price):
        assert profit_at(self.legs, 1, self.net_cash, price) >= -300.0 - 1e-9


class TestUndefinedRisk:
    def test_naked_short_call_is_unbounded(self):
        legs = (leg(460, "C", Side.SELL),)
        assert slope_at_infinity(legs) == -1
        assert worst_case_loss(legs, 1, 200.0) is None

    def test_ratio_spread_short_more_calls_than_long_is_unbounded(self):
        legs = (leg(455, "C", Side.BUY), leg(460, "C", Side.SELL, ratio=2))
        assert worst_case_loss(legs, 1, 50.0) is None

    def test_naked_short_put_is_bounded_but_enormous(self):
        legs = (leg(450, "P", Side.SELL),)
        # Worst case is the underlying going to zero: 450 * 100 less the credit.
        assert worst_case_loss(legs, 1, 100.0) == pytest.approx(44_900.0)


class TestLongPremium:
    def test_long_call_risk_is_the_debit_paid(self):
        legs = (leg(450, "C", Side.BUY),)
        assert worst_case_loss(legs, 1, -300.0) == pytest.approx(300.0)

    def test_long_call_upside_is_unbounded(self):
        legs = (leg(450, "C", Side.BUY),)
        assert max_gain(legs, 1, -300.0) is None


class TestEdgeCases:
    def test_empty_structure_is_an_error(self):
        with pytest.raises(ValueError):
            worst_case_loss((), 1, 0.0)

    def test_riskless_structure_reports_zero_not_negative(self):
        # Buy and sell the same contract: no exposure, and the credit is kept.
        legs = (leg(450, "P", Side.SELL), leg(450, "P", Side.BUY))
        assert worst_case_loss(legs, 1, 50.0) == pytest.approx(0.0)
