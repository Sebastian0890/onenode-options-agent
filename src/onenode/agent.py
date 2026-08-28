"""One run of the agent: wake up, look around, act at most once, exit.

The whole thing is a single pass with no internal loop. That is a deliberate
fit to how it is deployed - a scheduled job every fifteen minutes - and it
makes each run independently reproducible from the journal: there is no
in-memory state carried between runs, because there is no between.

Order of operations is chosen so the cheap and safe things happen first:

1. Preflight. A misconfigured account fails here, not at order submission.
2. Market closed - stop. Nothing below this line is meaningful.
3. Exits. Always allowed, even when the daily stop has halted new trades.
   Closing a position is how you stop losing money, so it is never gated.
4. Daily stop. Checked before any model is called, because tokens spent on a
   proposal that cannot be placed are tokens wasted.
5. Propose, review, gate, execute - at most one new position per run.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import date

from .agents.proposer import find_candidate, propose_trade
from .agents.risk_officer import review_trade
from .broker.cli import AlpacaCLI, AlpacaCLIError
from .execution import close_position, closing_limit_price, submit_spread
from .journal import Journal, write_markdown
from .management import DEFAULT_EXIT_RULES, ExitRules, positions_to_close
from .portfolio import committed_risk, group_option_positions
from .risk.gate import evaluate
from .risk.limits import DEFAULT_LIMITS, RiskLimits
from .risk.models import AccountSnapshot, Right
from .strategy import build_credit_spreads, build_iron_condors, size_position

DEFAULT_UNDERLYINGS = ("SPY", "QQQ", "IWM")


@dataclass
class RunResult:
    """What happened, in a form the caller can assert on and print."""

    status: str
    detail: str = ""
    orders_placed: int = 0
    positions_closed: int = 0
    events: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 1 if self.status == "error" else 0


def _snapshot(account: dict, groups: list) -> AccountSnapshot:
    risk = committed_risk(groups)
    return AccountSnapshot(
        equity=float(account["equity"]),
        last_equity=float(account["last_equity"]),
        open_positions=len(groups),
        # An unmeasurable position is treated as infinite exposure, which stops
        # new trades cold. That is the correct response to not knowing.
        committed_risk=risk if risk is not None else float("inf"),
    )


def run_once(
    cli: AlpacaCLI | None = None,
    journal: Journal | None = None,
    *,
    limits: RiskLimits = DEFAULT_LIMITS,
    exit_rules: ExitRules = DEFAULT_EXIT_RULES,
    underlyings: tuple[str, ...] = DEFAULT_UNDERLYINGS,
    today: date | None = None,
    dry_run: bool = False,
) -> RunResult:
    """Execute one full pass. Never raises for ordinary conditions."""
    run_id = uuid.uuid4().hex[:12]
    cli = cli or AlpacaCLI()
    journal = journal or Journal(run_id=run_id)
    journal.run_id = run_id
    today = today or date.today()
    result = RunResult(status="ok")

    def note(event: str, **data):
        journal.record(event, **data)
        result.events.append(event)

    note("run_started", dry_run=dry_run, underlyings=list(underlyings))

    # --- 1. Preflight ----------------------------------------------------
    try:
        problems = cli.preflight()
    except AlpacaCLIError as exc:
        note("error", reason=f"preflight could not run: {exc}")
        return RunResult(status="error", detail=str(exc), events=result.events)

    if problems:
        note("preflight_failed", violations=problems)
        return RunResult(status="error", detail="; ".join(problems), events=result.events)

    # --- 2. Session ------------------------------------------------------
    clock = cli.clock()
    if not clock.is_open:
        note("market_closed")
        return RunResult(status="market_closed", events=result.events)

    account = cli.account()
    groups = group_option_positions(cli.positions())
    snapshot = _snapshot(account, groups)

    note(
        "account_state",
        equity=round(snapshot.equity, 2),
        day_pnl_pct=round(snapshot.day_pnl_pct, 3),
        open_structures=len(groups),
        committed_risk=round(snapshot.committed_risk, 2)
        if snapshot.committed_risk != float("inf")
        else "unbounded",
        minutes_to_close=round(clock.minutes_to_close),
    )

    # --- 3. Exits, always allowed ----------------------------------------
    for group, reason in positions_to_close(
        groups, today=today, minutes_to_close=clock.minutes_to_close, rules=exit_rules
    ):
        price = closing_limit_price(group)
        try:
            order = close_position(cli, group, price, dry_run=dry_run)
            result.positions_closed += 1
            note(
                "position_closed",
                candidate=group.key,
                contracts=group.contracts,
                reason=reason,
                order_id=order.get("id", "dry-run"),
            )
        except AlpacaCLIError as exc:
            note("close_failed", candidate=group.key, reason=f"{reason} | {exc}")

    # --- 4. Daily stop, before any model is called -----------------------
    if snapshot.day_pnl_pct <= limits.daily_stop_pct:
        note(
            "halted",
            reason=(
                f"daily stop: {snapshot.day_pnl_pct:+.2f}% at or past "
                f"{limits.daily_stop_pct:+.2f}%. No new positions this session."
            ),
        )
        write_markdown(journal.entries())
        return RunResult(
            status="halted",
            detail="daily stop",
            positions_closed=result.positions_closed,
            events=result.events,
        )

    if clock.minutes_to_close < limits.min_minutes_to_close:
        note("no_new_positions", reason=f"{clock.minutes_to_close:.0f}min to close")
        write_markdown(journal.entries())
        return RunResult(
            status="too_late",
            positions_closed=result.positions_closed,
            events=result.events,
        )

    if len(groups) >= limits.max_open_positions:
        note("no_new_positions", reason=f"{len(groups)} structures already open")
        write_markdown(journal.entries())
        return RunResult(
            status="at_capacity",
            positions_closed=result.positions_closed,
            events=result.events,
        )

    # --- 5. Build the menu -----------------------------------------------
    candidates: list = []
    spots: dict[str, float] = {}
    for underlying in underlyings:
        if underlying not in limits.allowed_underlyings:
            continue
        try:
            spot = cli.latest_price(underlying)
            puts = cli.option_chain(
                underlying,
                option_type="put",
                strike_gte=spot * 0.94,
                strike_lte=spot * 1.00,
                limit=1000,
            )
            calls = cli.option_chain(
                underlying,
                option_type="call",
                strike_gte=spot * 1.00,
                strike_lte=spot * 1.06,
                limit=1000,
            )
        except AlpacaCLIError as exc:
            note("data_failed", candidate=underlying, reason=str(exc))
            continue

        spots[underlying] = spot
        shared = {
            "underlying": underlying,
            "max_days_to_expiry": limits.max_days_to_expiry,
            "max_spread_pct": limits.max_spread_pct_of_mid,
            "min_reward_to_risk": limits.min_reward_to_risk,
            "today": today,
        }
        put_spreads = build_credit_spreads(list(puts.values()), right=Right.PUT, **shared)
        call_spreads = build_credit_spreads(list(calls.values()), right=Right.CALL, **shared)

        # Condors first in the list, and they will usually stay near the top:
        # two credits against roughly one width beats either side alone.
        candidates += build_iron_condors(put_spreads, call_spreads)
        candidates += put_spreads
        candidates += call_spreads

    candidates.sort(key=lambda c: -c.reward_to_risk)
    note(
        "candidates_found",
        contracts=len(candidates),
        reason=f"{sum(1 for c in candidates if c.structure == 'iron condor')} condors",
    )

    if not candidates:
        write_markdown(journal.entries())
        return RunResult(
            status="no_candidates",
            positions_closed=result.positions_closed,
            events=result.events,
        )

    # --- 6. Propose -------------------------------------------------------
    primary = candidates[0].underlying
    try:
        decision = propose_trade(
            candidates,
            underlying=primary,
            spot=spots.get(primary, 0.0),
            day_pnl_pct=snapshot.day_pnl_pct,
            equity=snapshot.equity,
            open_positions=len(groups),
            committed_risk=snapshot.committed_risk,
            minutes_to_close=clock.minutes_to_close,
        )
    except Exception as exc:  # noqa: BLE001 - a broken proposer must not trade
        note("proposer_failed", reason=str(exc))
        write_markdown(journal.entries())
        return RunResult(
            status="proposer_failed",
            detail=str(exc),
            positions_closed=result.positions_closed,
            events=result.events,
        )

    note("proposal", candidate=decision.candidate_key, rationale=decision.rationale)

    if decision.action != "trade":
        write_markdown(journal.entries())
        return RunResult(
            status="stood_aside",
            detail=decision.rationale,
            positions_closed=result.positions_closed,
            events=result.events,
        )

    candidate = find_candidate(candidates, decision.candidate_key)
    if candidate is None:
        note("proposal_unusable", reason=f"{decision.candidate_key!r} is not a live candidate")
        write_markdown(journal.entries())
        return RunResult(
            status="stood_aside",
            positions_closed=result.positions_closed,
            events=result.events,
        )

    # --- 7. Size, then review, then gate ----------------------------------
    contracts = size_position(
        candidate,
        snapshot.equity,
        limits.max_risk_per_trade_pct,
        limits.max_contracts_per_order,
    )
    if contracts == 0:
        note("unsizeable", candidate=candidate.key, reason="one contract exceeds the budget")
        write_markdown(journal.entries())
        return RunResult(
            status="unsizeable",
            positions_closed=result.positions_closed,
            events=result.events,
        )

    trade = candidate.to_proposed_trade(contracts, rationale=decision.rationale)

    verdict = review_trade(
        candidate,
        trade,
        spot=spots.get(candidate.underlying, 0.0),
        equity=snapshot.equity,
        day_pnl_pct=snapshot.day_pnl_pct,
        open_positions=len(groups),
        committed_risk=snapshot.committed_risk,
        minutes_to_close=clock.minutes_to_close,
        worst_case_loss=candidate.max_loss_per_contract * contracts,
    )

    if not verdict.approve:
        note(
            "risk_officer_veto",
            candidate=candidate.key,
            contracts=contracts,
            reason=verdict.reason,
            verdict=verdict.reviewer,
        )
        write_markdown(journal.entries())
        return RunResult(
            status="vetoed",
            detail=verdict.reason,
            positions_closed=result.positions_closed,
            events=result.events,
        )

    gate = evaluate(trade, snapshot, clock, today=today, limits=limits)
    if not gate.approved:
        note(
            "gate_blocked",
            candidate=candidate.key,
            contracts=contracts,
            violations=list(gate.violations),
        )
        write_markdown(journal.entries())
        return RunResult(
            status="blocked",
            detail="; ".join(gate.violations),
            positions_closed=result.positions_closed,
            events=result.events,
        )

    note(
        "gate_approved",
        candidate=candidate.key,
        contracts=contracts,
        credit=round(trade.net_cash, 2),
        risk=round(gate.worst_case_loss or 0.0, 2),
        verdict=verdict.reviewer,
    )

    # --- 8. Execute -------------------------------------------------------
    try:
        order = submit_spread(
            cli,
            candidate,
            contracts,
            dry_run=dry_run,
            client_order_id=f"onenode-{run_id}",
        )
    except AlpacaCLIError as exc:
        note("order_failed", candidate=candidate.key, reason=str(exc))
        write_markdown(journal.entries())
        return RunResult(
            status="order_failed",
            detail=str(exc),
            positions_closed=result.positions_closed,
            events=result.events,
        )

    result.orders_placed = 1
    note(
        "order_placed",
        candidate=candidate.key,
        contracts=contracts,
        credit=round(trade.net_cash, 2),
        risk=round(gate.worst_case_loss or 0.0, 2),
        order_id=order.get("id", "dry-run"),
        rationale=decision.rationale,
    )

    write_markdown(journal.entries())
    return RunResult(
        status="traded",
        detail=candidate.key,
        orders_placed=1,
        positions_closed=result.positions_closed,
        events=result.events,
    )


def main() -> int:
    dry_run = os.environ.get("ONENODE_DRY_RUN", "").lower() in {"1", "true", "yes"}
    result = run_once(dry_run=dry_run)
    print(f"{result.status}: {result.detail}" if result.detail else result.status)
    print(f"orders placed: {result.orders_placed}, positions closed: {result.positions_closed}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
