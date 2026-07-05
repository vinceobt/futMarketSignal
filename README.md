# fc-market-analytics

**FUT Market Desk** — market intelligence for EA FC Ultimate Team player prices.
It tracks a personal watchlist as clean time-series data, engineers features
that reflect *why* prices move, backtests trading rules before trusting them,
and surfaces **BUY / SELL / HOLD signals with a plain-language rationale** —
through a CLI, alerts, and an interactive local web dashboard.

---

## Scope boundary — read this first

This is a **decision-support tool only**. It tells you when a card looks cheap
or expensive; **you** decide what to do. 

## How it works

The system is a linear pipeline. Each stage reads the previous stage's output
from a single SQLite database and never reaches around it:

```
                         ┌──────────────────┐
                         │  Event calendar  │  (you log TOTW/SBC/PROMO dates)
                         └────────┬─────────┘
                                  │
[ Collector ] → [ Price store ] → [ Feature engine ] → [ Signal engine ] → [ Alerts ]
  one adapter     append-only       momentum, z-score,   BUY/SELL/HOLD       console /
  per source      snapshots         market index, event  + confidence        Discord
                      │             distance, weekend     + reason             │
                      │                     │                  │               │
                      └─────────────────────┴──────────────────┴───────────────┘
                                            │
                                   [ Backtest harness ]  ← gates which rules are trusted
                                            │
                                   [ Web dashboard ]     ← read-only view of all of the above
```

**The data flow, stage by stage:**

1. **Collector** (`scheduler.py` + `collectors/`) polls a price *source* on a
   polite schedule and writes one row per (player, source, minute) into
   `price_snapshots`. Sources are pluggable adapters behind a 15-line contract
   (`collectors/base.py`), so the site/API being read is swappable and the rest
   of the system doesn't care where a price came from.

2. **Price store** (`db.py`) is append-only SQLite. History is never
   overwritten; a `UNIQUE(player_id, source, minute)` index makes re-runs and
   restarts idempotent.

3. **Feature engine** (`features.py`) turns raw prices into signal-ready
   features per snapshot: short/medium/long momentum (`pct_change_1h/24h/7d`),
   a trailing-24h `rolling_mean`/`rolling_std` and the **z-score** (how many
   standard deviations the current price sits from its recent norm),
   `days_to_next_event` from the calendar, and a Thu–Sun `is_weekend_window`
   flag (Weekend League drives demand).

4. **Signal engine** (`signals.py`) applies one interpretable rule to the
   latest features: price well below its norm → **BUY** (unless a supply-flooding
   event is imminent, which suppresses it); well above → **SELL**; otherwise
   **HOLD**. Every signal carries a confidence and a reason built from the real
   numbers, e.g. *"price 80,000 is 1.8σ below its 24h average (92,000) and no
   crashing event is pending — undervalued, buy."*

5. **Backtest harness** (`backtest.py`) is the trust gate. It replays the
   **exact same rule** over a player's history — net of EA's 5% sell tax — and
   compares it to naive baselines (buy-and-hold, random timing). A rule is only
   worth acting on once it beats every baseline. Because the backtester and the
   live engine call the same `evaluate()`, there is no drift between what you
   test and what fires.

6. **Alerts** (`alerts.py`) push signals to the console or a Discord webhook, in
   a daily **digest** or a **realtime** mode (strong signals only).

7. **Dashboard** (`webapp.py` + `dashboard.py` + `web/`) is a read-only FastAPI
   app rendering all of the above. It reuses the same pipeline functions — no
   duplicated logic — and makes no external calls.

### Data model (SQLite, `data/market.db`)

| Table | Purpose |
|---|---|
| `players` | watchlist metadata (name, rating, position, version, platform) |
| `price_snapshots` | **append-only** raw price history; the source of truth |
| `market_events` | TOTW / SBC / PROMO / PATCH dates you log; drives event features |
| `features` | derived per-snapshot features; **rebuildable** from snapshots anytime |
| `signals` | generated BUY/SELL/HOLD decisions with confidence + reason |

---

## Data source status


Until a real source is flowing, the signals and backtests run on whatever is in
the store (mock or hand-logged), and **the signal rule remains unproven** — see
the last line of "Build phases".

---

## Quickstart

```bash
uv venv --python 3.12 .venv          # or python3.11+ -m venv .venv
make install                         # installs the app + web + dev extras
make dashboard                       # launch the dashboard → http://127.0.0.1:8000
```

Then open **http://127.0.0.1:8000**, paste fut.gg player URLs into *Tracked
player links*, and hit **Scrape all**. That's the whole loop from the browser.

`make` uses the project's virtualenv directly, so you never need to activate it.
Run `make` (or `make help`) to see all targets: `dashboard`, `collect`,
`backtest`, `signals`, `players`, `build-features`, `test`.

Requires Python 3.11+. Pandas powers the feature engine; FastAPI/uvicorn power
the dashboard (the `web` extra); patchright (headless Chromium) powers the
fut.gg scraper.

### Running commands without `make`

The `futmarket` console script only exists inside the venv. Any of these work:

```bash
source .venv/bin/activate && futmarket dashboard   # activate, then use the name
.venv/bin/futmarket dashboard                        # call it via the venv
.venv/bin/python -m futmarket dashboard              # module form (no PATH needed)
uv run futmarket dashboard                            # via uv
```

Set `FUTMARKET_CONFIG=/path/to/config.yaml` to avoid passing `--config` each time.

### Access & security

The dashboard has buttons that run code on the host (scrape, delete, run jobs),
so it is **key-protected**. On launch it prints an access key:

```
  FUT Market Desk  →  http://127.0.0.1:8000
  Access key: 7f3a9c…      (generated for this run)
```

Open the URL, paste that key once, and you're in (a cookie session keeps you
logged in; **Lock** in the header logs out). Set a stable key with
`export FUTMARKET_KEY=your-secret` so it survives restarts.

Hardening in place: localhost-only by default (binding a public host needs
`--allow-remote` **and** a fixed `FUTMARKET_KEY`), cross-site/CSRF requests are
rejected, writes are rate-limited, and a strict self-only CSP + security headers
are sent. It's a personal-use local tool — don't expose it publicly without
understanding that anyone with the key can drive it.

---

## The daily workflow

Everything below is also drivable from the **dashboard** — the execution
buttons, the scraper Start/Stop panel, the *Momentum* scanner, and the *Tracked
player links* manager map one-to-one onto these commands. The CLI is the same
core, handy for scripting and cron.

**Momentum scanner (discovery).** The watchlist only knows the players *you*
added; the momentum scanner shows the market's biggest special-card movers so it
can tell *you* what to look at. In the dashboard, hit **Refresh** on the
*Momentum — market movers* card and **Track** any mover — it's added to your
watchlist and auto-scraped, and the normal history/z-score/backtest pipeline
takes over. From the CLI:

```bash
futmarket momentum --refresh            # scan fut.gg, show top movers
futmarket momentum --limit 20           # show cached movers
```

It's a **discovery aid**, not a signal: momentum shows *what's* moving, not
whether it's cheap — confirm a mover with its own history and backtest before
trusting it. It's a heavy market-wide fetch, so refresh sparingly.


```bash
# 1. Pull the full market price history for every watchlisted card.
#    fut.gg serves each card's history back to release, so one pass already
#    gives you enough data to backtest — no waiting to accumulate samples.
make collect                                # or: futmarket collect-once
futmarket run                               # keep polling on config's schedule

# 2. Log market events so the engine knows what's coming
futmarket log-event --type SBC --start 2026-07-08 \
  --player 203376-virgil-van-dijk-26-117643888

# 3. Recompute features from the latest prices
make build-features

# 4. Check whether the rule is even trustworthy on your data
make backtest                               # prints PROMOTE / NOT-PROVEN per player

# 5. Read the signals
make signals                                # store BUY/SELL/HOLD + reasons
futmarket alert --digest                    # or push a summary to console/Discord

# 6. Or just look at (and drive) everything at once
make dashboard                              # http://127.0.0.1:8000
```

Player IDs are derived from the fut.gg URL (`futmarket players` lists them).

To explore the UI before you have real data:
`python scripts/seed_demo.py` fills the store with deterministic **synthetic**
history (clearly not a data source — it invents prices).

---

## Command reference

| Command | What it does |
|---|---|
| `futmarket players` | watchlist with snapshot counts and last-seen time |
| `futmarket log-price <id> <coins>` | record a price manually (source=`manual`) |
| `futmarket history <id> [--limit N]` | snapshot table with Δ% between rows |
| `futmarket collect-once` | one polite collection pass (non-manual source) |
| `futmarket run` | scheduled polling loop (Ctrl-C to stop) |
| `futmarket log-event --type --start [--end --player --notes]` | add a market event |
| `futmarket build-features [--source]` | compute + persist the `features` table |
| `futmarket features <id> [--source]` | show the latest feature set for a player |
| `futmarket backtest [id] [--source --tax]` | rule vs. baselines + promote verdict |
| `futmarket signals [--source]` | evaluate + store current signals |
| `futmarket alert [--digest\|--realtime] [--source]` | deliver signals |
| `futmarket dashboard [--host --port --source]` | launch the web UI |

`--source` defaults to whichever source has the most stored snapshots.

---

## Dashboard

`futmarket dashboard` launches a local FastAPI app (`webapp.py`) serving a
single-page UI (`src/futmarket/web/`) that reads the same SQLite store through
`dashboard.build_payload()` — no separate logic, no external requests. It shows
KPI tiles, a watchlist with per-player signal pills, an interactive price chart
(crosshair tooltip, 24h-average overlay, weekend shading, event markers), a
signal banner with confidence meter and reason, and the backtest verdict per
player. Light/dark aware. **Read-only, local, single-user** (see scope
boundary) — it renders what the pipeline stored and never collects or trades.

---

## Rebound strategy & alerts

Beyond the z-score signal, the app has a **rebound advisor**: it hunts cards that
**reliably bounce off a floor** over ~1 month, alerts you to **BUY** when one is
near its floor now, tracks a **paper position**, and alerts you to **SELL** once
it's up a fixed profit target (net of the 5% tax). Alerts go to **Discord**.

```bash
futmarket advise --dry-run          # show what it WOULD buy/sell — writes nothing, sends nothing
futmarket advise                    # for real: open/close paper positions + send Discord alerts
futmarket positions                 # list open/closed paper positions
futmarket backtest --strategy rebound   # score the strategy vs baselines on YOUR data
```

Set `alerts.destination: discord` and paste a channel **webhook URL** in
`config.yaml`. Tune the pattern under `strategy:` (`window_days`, `min_bounces`,
`buy_zone_pct`, `target_pct`, `stop_pct`, `momentum_screen_limit`). When you run
autonomously (below), `advise` runs after every collection pass, so alerts fire on
their own. In the dashboard, a **Positions** card shows open/closed positions and
a **Run advisor** button.

**How it decides.** The launch price is irrelevant — only the *recent trading
range* matters. Over the trailing `window_days` it finds the support floor,
counts *completed rebounds* (dip to the floor → recover ≥ `target_pct`), and
checks the **sequence of rebound troughs**: a card that keeps bouncing off ~the
same floor is a stable rebounder; one making progressively *lower lows* is a
falling knife and is rejected. Crucially, a card that launched high, **crashed,
and then settled into a range** is a valid buy — the crash is ignored because it
never dips into the buy-zone, so it's never counted as a trough. BUY = reliable
rebounder **and** price in the buy-zone now.

**Selling — two modes (`strategy.sell_mode`):**
- `target` (default): sell at a **fixed profit** — `target_pct` (default **25%**)
  net of the 5% tax. Predictable, always exits.
- `resistance`: **ride the swing to the ceiling** — sell when price returns near
  the range's resistance (the `resistance_pctile` percentile of the card's actual
  rebound *peaks*, crash-robust). Captures the whole move (on a 1.2k↔3k card that's
  ~+120% vs ~+25% for a fixed target) but may miss an exit if a cycle falls short
  of the ceiling.

Both modes honor the optional stop below the floor.

**Honesty caveats.**
- **Paper only** — it alerts *you* and tracks a virtual position; it never trades
  or touches EA (see scope boundary).
- **Backtest it first.** On a watchlist of promo/TOTS cards that launched high and
  crashed (not rebounders), it correctly makes **zero trades** — it only fires on
  genuinely range-bound cards. Prove it on your data with `backtest --strategy
  rebound` before trusting the alerts, and feed it cards that actually oscillate.

---

## Discovery scan (find new range-bound cards)

`futmarket scan-momentum` refreshes fut.gg's momentum movers, screens the ones you
don't track for the rebound pattern, and **auto-adds only the ones in their
buy-zone right now** (actionable immediately — reliable rebounders sitting above
their floor are noted and picked up on a later scan when they dip). `--dry-run`
reports without adding. It writes to `data/scan.log` and posts a Discord notice
when it adds cards.

Discovery moves on the scale of days, so it runs **every 6 hours**, not every
cycle. It's installed automatically alongside the collector by
`make autonomous-install` (below) as a second LaunchAgent — it needs no wake of
its own; it rides the collector's wakes. Change the cadence with
`--scan-interval-min N`, or skip it with `--no-scan`.

## Run it autonomously (macOS)

Collect on a schedule with no terminal and no dashboard open — **3× per hour
(every 20 min), waking the Mac from sleep** so it keeps going 24/7:

```bash
make autonomous-install       # every 20 min, wakes the Mac (default)
make autonomous-status        # is it loaded? next wake? recent log
make autonomous-log           # follow the collection log
make autonomous-uninstall     # stop and remove it
```

Options: `bash scripts/install_autonomous.sh --interval-min 30` for a different
cadence, or `--no-wake` for **awake-only** (no sudo, no wake-ups; collects only
while the Mac is awake).

**How it works.** A launchd LaunchAgent runs `collect-once` every 20 min. Because
a LaunchAgent can't run while the Mac sleeps, each run also arms the *next* wake
(`pmset schedule wake`) so the Mac wakes itself, scrapes for ~30s, and sleeps
again — a self-perpetuating chain that re-arms on any wake/login if broken.

**One-time sudo.** Wake mode installs a **narrowly scoped** passwordless sudoers
rule at `/etc/sudoers.d/futmarket-pmset` that permits *only* `pmset schedule
wake`/`cancelall` — nothing else. `--no-wake` skips it entirely.

**Honest caveats:**
- **Powered off = no runs;** the chain re-arms at the next login/wake.
- Waking every 20 min prevents deep sleep — a small power/thermal cost; on
  **battery** macOS may defer wakes (Low Power Mode suppresses them). Best on AC.
- Use the LaunchAgent **or** the dashboard's continuous mode, not both (harmless —
  writes are idempotent — just redundant scraping).
- If your `poll_minutes × skip_guard_fraction` is ≥ the interval, some runs skip
  as "fresh" (the installer warns and tells you how to fix it). Data granularity
  is unaffected — fut.gg backfills each card's full history on every hit.

---

## Configuration

Everything lives in `config.yaml` — nothing operational is hardcoded.

**Collection & politeness**

| Key | Meaning |
|---|---|
| `source` | active adapter: `manual`, `mock` (or a permitted HTTP adapter you add) |
| `platform` | `console` (PS+Xbox share one market) or `pc` |
| `poll_minutes` | minutes between passes; **floor of 30** enforced |
| `inter_player_delay_seconds` | `[min, max]` jittered delay between players |
| `skip_guard_fraction` | skip a player fresher than `poll_minutes × this` |
| `max_consecutive_failures`, `cooldown_minutes` | circuit breaker |
| `user_agent` | honest identifying UA (put a real contact in it) |
| `max_watchlist_size` | hard cap (default 50), enforced at load |
| `database_path`, `log_path` | storage locations |

**`signals:`** — the rule thresholds, shared by the backtester and live engine.
Tune here, re-backtest, then trust.

| Key | Meaning |
|---|---|
| `buy_z` | BUY when z-score ≤ this (default −1.5) |
| `sell_z` | SELL when z-score ≥ this (default 1.5) |
| `event_guard_days` | suppress BUY if a crashing event is within this many days |
| `momentum_guard_pct` | **opt-in** momentum confirmation (default 0 = off) |
| `tax_rate` | EA sell tax used in backtest P&L (default 0.05) |

**Momentum confirmation (`momentum_guard_pct`).** The base rule is pure
mean-reversion (z-score): it answers *"is the price cheap or dear?"* but not
*"is it still moving that way?"* — so it can catch a falling knife (a card that
looks cheap every hour on the way down) or sell a winner too early. Setting
`momentum_guard_pct > 0` adds a directional filter using the 24h price change:
a BUY is held while the price is still falling faster than the threshold, and a
SELL is held while it's still climbing faster. It uses `pct_change_24h` — the
same field the backtester replays — so the live and backtested rules never
diverge.

**It ships off (0) on purpose.** In our own backtest it *reduced* returns on
coarse daily history (during a steady crash, "below the 24h mean" and "24h change
negative" fire together, so it over-blocks the dip-buys that recovered). Turn it
on only after it beats the plain rule in `futmarket backtest` on your own
accumulated (ideally hourly) data — tune, backtest, then trust.

**`alerts:`**

| Key | Meaning |
|---|---|
| `destination` | `console` (default) or `discord` |
| `webhook_url` | Discord incoming webhook; sending publishes externally, so it stays `null`/console until set deliberately |
| `min_confidence` | realtime mode only pushes signals at/above this |

**`watchlist:`** — a plain list of **fut.gg player URLs**, used as a **one-time
seed**. Open a card on [fut.gg](https://www.fut.gg), copy the URL, and paste it in:

```yaml
watchlist:
  - https://www.fut.gg/players/239085-erling-haaland/26-184788461/
  - https://www.fut.gg/players/231747-kylian-mbappe/26-231747/
```

The app derives a **stable `player_id`** from the URL path (it keys all stored
history, so the URL for a card shouldn't change once data accumulates) and a
display `name`, and fills in `rating`/`version` best-effort by reading the page.
Cap is 50 (`max_watchlist_size`).

**After first run the watchlist lives in the database, not this file** — so
add/remove is a safe transactional operation instead of a file rewrite. Manage
it from the dashboard's *Tracked player links* panel, or from the CLI:

```bash
futmarket watchlist list                 # show tracked players
futmarket watchlist add <fut.gg URL>     # track one
futmarket watchlist remove <player_id>   # stop tracking (id from `list`)
futmarket watchlist import urls.txt      # bulk-add (one URL per line)
```

Editing the `watchlist:` block in `config.yaml` after the first run has no effect
(the DB is authoritative); use `watchlist import` to bulk-add from a file instead.

---

## Guarantees

- **Append-only history** — snapshots are never overwritten.
- **Idempotent collection** — the unique index + freshness skip-guard mean
  re-runs and restarts can't duplicate rows.
- **Graceful degradation** — one player failing never stops a pass; a source
  failing repeatedly trips a breaker and cools down instead of hammering.
- **No test/production drift** — the backtester replays the same `evaluate()`
  the live engine fires.
- **Structured logs** — every collect/skip/failure/signal in `data/collector.log`.

---

## Build phases

- **Phase 0 (done)** — collector pipeline, storage, CLI, tests.
- **Phase 1 (done)** — feature engine (`features.py`): momentum, rolling
  mean/std/z-score, market index + normalized price, event distance, weekend
  flag; derived `features` table.
- **Phase 2 (done)** — backtesting harness (`backtest.py`): long-only sim net of
  the 5% sell tax; hit rate, avg return, total return, max drawdown vs.
  buy-and-hold and random baselines; `beats_baselines` gates promotion.
- **Phase 3 (done)** — signal engine (`signals.py`) with confidence + reasons,
  and alert delivery (`alerts.py`) to console/Discord in digest/realtime mode.
  ML is intentionally deferred: it earns a place only after beating the
  baselines in this same harness.
- **Dashboard (done)** — interactive local web UI over the whole pipeline.

Each phase is gated on validating the previous one against real data. The
signal rule ships **tuned but unproven on real prices** — its thresholds must
clear `futmarket backtest` on genuine accumulated history before any alert
should be acted on.

---

## Project layout

```
config.yaml                 all runtime knobs + the watchlist
src/futmarket/
  config.py                 load + validate config
  db.py                     schema, snapshots, features, signals, events
  collectors/               source adapters (base contract, mock, ...)
  scheduler.py              collection passes + polling loop + circuit breaker
  features.py               Phase 1 feature engine
  backtest.py               Phase 2 backtest harness + baselines
  signals.py                Phase 3 rule engine (BUY/SELL/HOLD + reason)
  alerts.py                 console / Discord delivery
  dashboard.py              builds the JSON payload for the UI
  webapp.py                 FastAPI app
  log.py                    structured logging setup
  web/                      index.html, app.css, app.js (the dashboard UI)
  cli.py                    the `futmarket` command
scripts/
  seed_demo.py              synthetic demo data for exploring the UI
  build_artifact.py         bundles a self-contained dashboard preview
tests/                      pytest suite (feature math, P&L, rule, idempotency)
data/                       market.db + collector.log (created at runtime)
```
