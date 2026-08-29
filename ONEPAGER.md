# OneNode Options Agent — one-page write-up

An autonomous options trading agent for Alpaca paper trading. Two language models argue
about each trade and neither can place one; approval belongs to a layer with no model in
it at all.

**Repo:** github.com/Sebastian0890/onenode-options-agent · **Paper account:** `9a62e116-038c-474e-92b6-cbbcc40f462c`
**Paper trading only. No real capital at any point.**

---

## AI logic

Most trading agents are a language model with an order endpoint attached. That works until
the model has a bad idea at 3pm on a Tuesday while nobody is watching. This one separates
three things that design conflates:

| Layer | Runs on | Can place a trade? |
|---|---|---|
| Candidate builder | Plain Python | — |
| **Proposer** | Whichever model is configured | No. It can only suggest. |
| **Risk Officer** | A model from a different family | No. It can only veto. |
| **Hard Gate** | Plain Python, no model | **Yes — and only it can.** |

**The model is never asked to name a contract.** A language model asked for an option symbol
will eventually name one that does not exist, or quote a price nobody is offering. So the code
enumerates every credit spread and iron condor that is really in the chain, really quoted and
really tradeable, and the Proposer's answer is one key from that list. A key that is not on the
list is treated as standing aside, not as something to correct — if the model cannot pick from
twenty-five strings, guessing at its intent is not an improvement.

Standing aside is a first-class answer and the prompt says so. An agent that must trade every
time it is asked will trade badly.

**The Risk Officer runs on a different model family and never sees the Proposer's reasoning.**
It is prompted to refute rather than to assess, and to refuse when uncertain. A reviewer that
inherits the argument it is meant to check is not a reviewer, and one that splits the difference
approves everything eventually. Any failure to reach it counts as a veto: an unreviewed trade is
exactly what the layer exists to prevent.

**No model is named in the trading code.** Model identifiers rot faster than a hackathon lasts —
while this was being built, GitHub Models entered its retirement brownout, Groq shut down the two
Llama models most published code has hardcoded, and Cerebras closed the free tier this project
would otherwise have used. So a provider states its preferences as substrings and the concrete
model is resolved against whatever the host's live catalogue reports, which turns a retirement
into one wasted request instead of one lost trading day.

Independence is tracked by **model lineage rather than by vendor**, because two hosts serving the
same open checkpoint are one opinion sold twice. When only one lineage is reachable the review
still happens — an isolated context is worth something — but the verdict is stamped `degraded`
and the journal says so. The agent never claims a second opinion it did not get.

## Risk gates

The gate is a pure function — same inputs, same verdict, no network, no clock of its own — which
is what makes the risk policy testable rather than asserted. It collects every violation rather
than returning on the first, so a rejection explains itself completely in the journal.

**It computes the worst case from the option legs itself** rather than trusting the figure attached
to the proposal. Profit at expiry is piecewise linear in the underlying price with kinks only at
the strikes, so sampling zero and every strike finds the true minimum; the slope as price grows
without bound reveals unbounded risk. A model that understates the risk of its own idea cannot
talk its way past arithmetic, and a position whose loss cannot be stated is rejected outright
rather than sized down.

| Limit | Value |
|---|---|
| Worst case per trade | 1.5% of equity |
| Worst case across all open positions | 6% of equity |
| Daily stop | −3% halts new trades for the session |
| Undefined-risk positions | Never permitted |
| Instruments | SPY, QQQ, IWM |
| Expiry window | 0–7 DTE, single expiry |
| No new positions | Final 30 minutes of the session |
| Quote age / spread | 300s / 10% of mid |

These live in code, not in a prompt. A limit in a prompt is a suggestion the model can argue with,
and over a week of unattended running it eventually will.

Two details that matter more than they look. **Open positions are regrouped into structures before
being measured** — summing the legs of a spread separately makes a harmless put spread look like
unbounded risk, and an agent that overstates its own exposure quietly stops trading mid-week. And
what bounds the next trade is *remaining* risk, not risk at entry: premium already collected is
spent. **Exits are never gated** by the daily stop, because closing a position is how you stop
losing money.

Position sizing is derived from the risk budget, never proposed by a model. Exits need no model
either: four arithmetic rules, checked expiry-first, because a short spread carried into its final
hour is a coin flip with assignment attached and being green does not make it safe.

## Alpaca infrastructure

**Trading API via the Alpaca CLI** for the autonomous loop. The agent wakes on a schedule, looks
around, acts at most once and exits — the long-running-agent-session case the CLI was built for —
and every action it takes is a command a human can paste into a terminal and reproduce. Orders go
as a single `order_class: mleg` limit order with `position_intent` explicit on every leg; legging
in separately would leave the position briefly naked, the exact state the gate forbids. Verified
against the broker's own `--dry-run`.

**MCP server** (`.mcp.json`, `uvx alpaca-mcp-server`) for interactive analysis of the account and
chain alongside the run — the heavier, conversational path, kept out of the scheduled job where it
would only add latency.

**Market clock** from `/v2/clock` rather than a calendar of our own, so holidays and early closes
need no maintenance. **Preflight** asserts an active account, no blocks, and options level 3 at
startup, so a misconfigured account fails on the first run rather than at order submission.

**On the data**: strike selection runs on delta because the chain snapshot carries greeks but no
implied-volatility field — verified against the live API, not assumed from documentation. The free
plan's options feed is indicative rather than full OPRA, so the quote the agent reasons about is
not guaranteed to be the quote it trades against; the age and spread gates refuse the cases where
that gap is most likely to bite instead of pretending the data is better than it is.

**GitHub Actions** runs the agent every 15 minutes across the session, refuses any key not starting
with `PK`, and commits the trade journal back to the repository — so the decision record is public
and timestamped by something other than us. The journal records the runs where nothing happened
too: an agent that only logs its trades looks decisive in hindsight.

---

*Paper trading is a simulation involving no actual securities transactions and no real funds;
hypothetical results do not represent actual trading. Nothing here is investment advice.*
