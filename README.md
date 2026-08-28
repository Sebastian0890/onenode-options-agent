# OneNode Options Agent

An autonomous, defined-risk options trading agent built on Alpaca's Trading API,
MCP server and CLI. Built for the
[Alpaca AI Trading Agents Hackathon](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon),
28 August – 4 September 2026.

**Paper trading only. No real capital is at risk at any point.**

---

## The idea

Most trading agents are a language model with an order endpoint attached. That
works right up until the model has a bad idea at 3pm on a Tuesday while nobody
is watching.

This agent separates the three things that get conflated in that design:

| Layer | Runs on | Can it place a trade? |
|---|---|---|
| **Proposer** | Claude | No. It can only suggest. |
| **Risk Officer** | An independent open-weights model | No. It can only veto. |
| **Hard Gate** | Plain Python, no model | **Yes — and only it can.** |

The Proposer reads the market and argues for a trade. The Risk Officer sees the
proposal without the Proposer's reasoning and tries to knock it down. Neither
can reach past the Hard Gate, which is an ordinary pure function with no network
access and no notion of persuasion.

The models are allowed to be creative. They are not allowed to be dangerous.

## The Hard Gate

The gate is the part worth reading first: [`src/onenode/risk/`](src/onenode/risk/).

It computes the worst case of a proposal **from the option legs themselves**
rather than believing the number the model attached to it. A model that
understates the risk of its own idea cannot talk its way past arithmetic.

Profit at expiry is piecewise linear in the underlying price with kinks only at
the strikes, so sampling zero and every strike finds the true minimum. The slope
as price grows without bound reveals unbounded risk — and a position whose loss
cannot be stated is rejected outright rather than sized down.

Current limits, all enforced in code rather than in a prompt:

| Limit | Value |
|---|---|
| Worst case per trade | 1.5% of equity |
| Worst case across all open positions | 6% of equity |
| Daily stop | −3% halts new trades for the session |
| Undefined-risk positions | Never permitted |
| Instruments | SPY, QQQ, IWM only |
| Expiry window | 0–7 DTE, single expiry |
| No new positions | Final 30 minutes of the session |
| Quote age / spread | 300s / 10% of mid |

The last row exists because Alpaca's free data plan serves an *indicative*
options feed rather than full OPRA. The quote the agent reasons about is not
guaranteed to be the quote it trades against, so the agent refuses the cases
where that gap is most likely to bite instead of pretending the data is better
than it is.

## Alpaca surfaces used

**The CLI drives the autonomous loop.** The agent wakes on a schedule, looks around, acts
at most once and exits — the long-running-agent-session case Alpaca built the CLI for — and
every action it takes is a command a human can paste into a terminal and reproduce. Orders
go as a single `order_class: mleg` limit order, verified against the broker's own `--dry-run`.

**The MCP server handles interactive analysis.** [`.mcp.json`](.mcp.json) wires up
`uvx alpaca-mcp-server` for conversational inspection of the account and the chain alongside
a run. It is deliberately not in the scheduled job, where it would add latency without adding
anything the CLI does not already give.

```bash
export ALPACA_API_KEY=PK... ALPACA_SECRET_KEY=...
uvx alpaca-mcp-server
```

## Running the tests

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
pytest
```

The risk layer has no dependencies, no network calls and no model calls, so the
tests run offline in under a second.

## The write-up

[`ONEPAGER.md`](ONEPAGER.md) — the AI logic, the risk gates, and the Alpaca implementation
on one page.

## Status

Work in progress during the hackathon week. See the commit history — the trade
journal is committed back to this repository on every agent run, so the full
decision record is public and timestamped.

## Disclosures

This project trades in Alpaca's paper environment. Paper trading is a
simulation: it does not involve actual securities transactions or real funds,
and hypothetical results do not represent actual trading or guarantee future
results.

Nothing here is investment advice or a recommendation to buy, sell or hold any
security. Options trading is not suitable for all investors; see
[Characteristics and Risks of Standardized Options](https://www.theocc.com/company-information/documents-and-archives/options-disclosure-document).

## License

MIT — see [LICENSE](LICENSE).
