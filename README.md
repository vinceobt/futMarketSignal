# futmarket

A self-improving machine-learning system for **EA FC Ultimate Team**. It learns how
cards behave — how they react to promos, reward drops, day of week, and the
release-crash cycle — and tells you **what to buy, at what price, and when to sell**.

It never trades for you and never touches your EA account. It reads public market
data, learns from it, and makes recommendations.

> For the full project brief, market findings, and hard-won traps, see
> [docs/ENGINEERING-NOTES.md](docs/ENGINEERING-NOTES.md).

## What it does

- **Collects** the whole FC 26 market's prices, full price history since launch,
  real completed-sale prices, the promo/SBC/TOTW/news calendar, and social buzz.
- **Learns** card behaviour and cohort relationships (how groups of similar cards
  move together), with strict no-leakage discipline and walk-forward validation.
- **Recommends** liquid cards to buy — with a buy band, sell target, stop, the
  reasons, and the fut.gg link — and **grades its own past calls** so you can see
  whether it actually works.
- **Runs 24/7** on a schedule, refreshing a live dashboard and pinging Discord.

**Liquidity is rule #1:** it only recommends cards that genuinely sell fast, because
a correct call you can't get out of makes no money.

## Setup

```bash
uv sync --extra web --extra ml --extra dev      # or: make install
```

Copy `.env.example` to `.env` and fill in any keys you want (Discord webhook,
YouTube/Reddit). Everything else lives in `config.yaml`.

## Commands

```bash
futmarket picks                 # what to buy right now, at what price, and why
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

## Layout

```
src/futmarket/
  collectors/   raw data in (prices, history, calendar, sales, social)
  services/     orchestration (registry, backfill, liquidity, scorecard, notify…)
  ml/           the brain (dataset, cohorts, labels, train, picks, dashboard)
  cli.py db.py config.py webapp.py …
tests/          pytest suite (no network)
```

## Testing

```bash
make test        # or: .venv/bin/python -m pytest -q
```

Tests never hit the network.
