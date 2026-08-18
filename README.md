# futmarket

**Learning how the EA FC Ultimate Team market behaves, and trading it.**

![tests](https://github.com/vinceobt/futMarketSignal/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A self-improving machine-learning system for **EA FC Ultimate Team**. It learns how
cards behave — how they react to promos, reward drops, day of week, and the
release-crash cycle — and tells you **what to buy, at what price, and when to sell**.

It never trades for you and never touches your EA account. It reads public market
data, learns from it, and makes recommendations.

> For the measured market findings, the strategy rationale, and the traps that
> produced convincing-but-wrong numbers along the way, see
> [docs/ENGINEERING-NOTES.md](docs/ENGINEERING-NOTES.md).

## What it does

- **Collects** the whole FC 26 market's prices, full price history since launch,
  real completed-sale prices, the promo/SBC/TOTW/news calendar, and social buzz.
- **Learns** card behaviour and cohort relationships (how groups of similar cards
  move together), with strict no-leakage discipline and walk-forward validation.
- **Recommends** liquid cards to buy — with a buy band, sell target, stop, **how
  long to hold**, the reasons, and the fut.gg link — and **grades its own past
  calls** so you can see whether it actually works.
- **Runs 24/7** on a schedule, refreshing a live dashboard and pinging Discord.

**Liquidity is rule #1:** it only recommends cards that genuinely sell fast, because
a correct call you can't get out of makes no money.

**Costs come first.** EA's 5% tax plus sell slippage means a trade must gain
**+7.4%** just to break even, and the median tradeable card doesn't move at all
over a fortnight. So the model predicts how far a card will beat *the market*,
and a card is only recommended when that is expected to clear the round trip.
When nothing does, it says **buy nothing** — which is usually the right answer.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.11+** | |
| Data | **pandas**, **NumPy** | one row per (card, day); the feature matrix runs to millions of rows, so memory layout matters |
| ML | **scikit-learn** — `HistGradientBoostingClassifier` / `Regressor` | histogram gradient boosting handles the missing values and mixed categoricals natively, and needs no system OpenMP, so it installs clean without Homebrew |
| Model artifacts | **joblib** | versioned per horizon, with the gate's measured payoff profile stored alongside |
| Storage | **SQLite** (WAL, `busy_timeout`) | ~2.85M price snapshots in one file; WAL lets the 2-hourly collector write while training reads |
| HTTP | **httpx** | every collector; ~2s spacing with exponential backoff |
| Browser | **patchright** (Playwright fork) | only where plain HTTP can't reach — discovering the rotating CDN price-index URL, and the X/Twitter session |
| Web | **FastAPI** + **Uvicorn** | server-rendered read-only dashboard, no frontend build step |
| Config | **YAML** + `.env` | runtime knobs in `config.yaml`; credentials never in tracked files |
| Tooling | **uv**, **pytest**, **GitHub Actions** | 279 tests, none of which touch the network |
| Deployment | **launchd** (macOS) · **systemd** units + timer (Linux) | the 24/7 loop; see [deploy/](deploy/) |

No frontend framework and no ORM: the dashboard is generated HTML and the queries
are hand-written SQL, because the schema is small, the read patterns are known,
and both stayed easier to reason about than the abstraction would have been.

## Results so far

The system grades every one of its own past calls and reports **alpha** — how the
trade did against the same-month, same-liquidity-tier market — not just raw
return. Live record as of 2026-08-05:

| Strategy | Graded trades | Return on capital | Alpha vs market | Win rate |
|---|---|---|---|---|
| `release_v1` — buy a promo card 4–6 days into its release crash | 7 | **+8.6%** | **+15.5%** | 86% |
| `relval_v1` — buy a deep dip on its 30-day floor | 137 | **−11.0%** | **−4.9%** | 23% |
| `weekend_v1` — buy a high-swing card into the weekend trough | *backtest only* | +4.2% | +16.7pp | 59% |

`relval_v1` is losing money and is shown here for the same reason it is still in
the codebase: **the scoreboard is only worth anything if it reports the losers.**
Finding that out required first fixing a grader that was measuring itself — stops
placed at −15% were realizing −34.9%, and two thirds of trades had no benchmark
recorded at all. That story is in the
[engineering notes](docs/ENGINEERING-NOTES.md#10-status-as-of-2026-08-05).

### Why most of this market is untradeable

EA's 5% sell tax plus slippage means a round trip must gain **+7.4% just to break
even** — and the median tradeable card doesn't move at all over a fortnight. So
the median trade, before any skill is involved, is **−6.9%**. The previous
strategy bought a shallow dip whose entire bounce was ~4.7%; it was
mathematically incapable of making money, however good the model got. Every gate
since has had to clear the round trip before it was allowed to trade.

## Setup

```bash
uv sync --extra web --extra ml --extra dev      # or: make install
```

Copy `.env.example` to `.env` and fill in any keys you want (Discord webhook,
YouTube/Reddit). Everything else lives in `config.yaml`.

## Commands

```bash
futmarket picks                 # what to buy right now, at what price, and why
futmarket evaluate --gate all   # would this rule have made money? month by month
futmarket scorecard             # how past picks have actually done
futmarket dashboard --port 8899 # live web dashboard (open /ml)

# data in
futmarket build-registry        # crawl the full card universe
futmarket collect-bulk          # whole-market price snapshot
futmarket backfill-history      # deep daily history since release
futmarket build-calendar        # promos / SBCs / EA news
futmarket sale-stats            # what cards really sold for
futmarket social [--x]          # Reddit / YouTube / X buzz

# learn
futmarket score-liquidity       # A/B/C tradeability tiers
futmarket build-dataset         # assemble the ML feature matrix
futmarket train                 # train + walk-forward validate the models
```

Run via the venv: `.venv/bin/futmarket …` (or `make picks`, `make dashboard`, …).

## Running 24/7 (macOS)

`make autonomous-install` installs a LaunchAgent that every 2 hours refreshes
prices, records picks, grades the scorecard, refreshes the dashboard, and posts a
Discord summary. See [scripts/ml_daily.sh](scripts/ml_daily.sh) and
[deploy/](deploy/) (including `deploy/cloud/` for an always-on server).

**Reach the dashboard from your phone** (same wifi): the dashboard LaunchAgent
([deploy/com.futmarket.dashboard.plist](deploy/com.futmarket.dashboard.plist))
binds the LAN, so open `http://<your-mac-ip>:8899/`. Viewing and consulting are
open; the action buttons need the access key (`?key=…`) from any device but your
Mac.

## Layout

```
src/futmarket/
  collectors/   raw data in (prices, history, calendar, sales, social)
  services/     orchestration (registry, backfill, liquidity, scorecard, notify…)
  ml/           the brain (dataset, cohorts, labels, evaluate, train, picks, dashboard)
  cli.py db.py config.py webapp.py …
tests/          pytest suite (no network)
docs/           engineering notes + data-source reference
deploy/         LaunchAgents (macOS) and systemd units (Linux) for 24/7 running
```

## Testing

```bash
make test        # or: .venv/bin/python -m pytest -q
```

Tests never hit the network.
