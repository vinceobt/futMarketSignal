# fc-market-analytics — project brief

Read this first. It exists so a new session can be useful in five minutes instead
of rediscovering everything (and re-making the same mistakes).

---

## 1. What we are building

**A self-improving machine-learning system that learns how EA FC Ultimate Team
cards behave, and tells the owner what to buy and sell.**

The owner's words: *"a mini almighty trading guru"* — a model that trains on
history, keeps training forever, and makes independent decisions from what it
sees on fut.gg, EA's site, and social media.

**End goal:** by the EA FC 27 launch (~Sept 2027) the model is at expert level
and still improving. Training never stops; it carries forward into each new game.

### What the owner explicitly wants
- **Extensive training on historical data.** Feed it everything.
- **Learn behaviour, not formulas** — how cards react to news, promos, reward
  drops, day of week, time of day.
- **Independent decisions.** It should say *"buy this card, at this price, now"*.
- **Plain output.** No statistics lectures. Real, actionable calls.

### What the owner explicitly does NOT want
- **Comparisons against the old rules engine.** It never made money — that's why
  this project exists. Don't spend effort benchmarking against it.
- **Mathematical jargon** in explanations. Say what it means for trading.

---

## 2. Current state (2026-07-20)

Working, end to end. `futmarket picks` produces real recommendations.

| Layer | State |
|---|---|
| Card registry | 9,998 cards (8,231 tradeable) |
| Price history | **2.5M snapshots**, 99.3% of tradeable cards, full season since 2025-09-08 |
| Event calendar | 247 events — promos, SBCs, EA announcements |
| Features | card behaviour + cohorts + lifecycle + weekly rhythm + release curve |
| Models | scikit-learn HistGradientBoosting — forecast + direction heads |
| Output | `futmarket picks` — buy band, sell target, stop, reasons, card URL |

**Honest status:** the model has a real but modest edge that is roughly **equal
to trading costs** (~11% round trip). Profitable if you buy patiently near the
listed price; unprofitable if you chase at market. It is not yet proven to make
money live.

---

## 3. Market knowledge we measured (this is the valuable part)

All of these came from the data, not from assumptions. They drive the features.

### The weekly supply cycle
Rewards flood the market with cards; supply then dries up.

```
Mon   +0.75%   ▲          Weekend League rewards, recovery begins
Tue   +1.37%   ▲▲
Wed   +2.18%   ▲▲▲        ← weekly PEAK
Thu   -3.11%   ▼▼▼        Rivals rewards land
Fri   -9.22%   ▼▼▼▼▼      promo drops + Weekend League starts
Sat  -14.46%   ▼▼▼▼▼▼▼
Sun  -19.98%   ▼▼▼▼▼▼▼▼   ← weekly TROUGH
```
Full-season day averages: **Saturday −1.86%** (worst), **Monday +1.31%** (best).
Trade shape: **buy into the Sunday trough, sell into the Wednesday peak.**

### The specific dump windows (UTC, July hourly data)
| Window | Move | Cause |
|---|---|---|
| Thu 15:00 | **−3.99%** | Rivals rewards → packs opened |
| Sat 18:00–19:00 | −1.5% | Weekend League grind |
| Fri 18:00 | −1.40% | promo drop + WL kickoff |
| Mon 00:00–04:00 | **+0.8 to +1.0%** | supply dries up |

### The release curve
A promo card, indexed to its release price of 100:
```
day 0  100.0     day 4   67.4     day 9   66.5  ← bottom (-33%)
day 1   84.4     day 7   67.6     day 13  74.4  ← recovery (+12%)
```
**Promos drop on Friday** (1,846 of them this season).

### Execution reality
- The going rate is **~1.04×** the cheapest listing — you do **not** need to snipe.
- Liquid cards trade **300–1,000 sales/hour**; thin ones 1–2.
- Round-trip cost ≈ **11%** (4% buying over listing + 2% selling under + 5% EA tax).
- Breakeven is at about a **3.3%** buy premium — patience decides profitability.

---

## 4. Traps — every one of these was a real bug

**Do not re-introduce these.** Most were caught by the owner checking live prices,
not by tests.

1. **Never quote a price from completed sales.** They describe the last several
   hours; on a moving card they lag badly (quoted 173k while the card listed at
   241k). **Anchor the buy band to the live listing** + ~5%. Sales are for
   *liquidity only*.
2. **Card names repeat across versions.** There are seven "João Félix" cards from
   700 to 607,000 coins. **Always print the fut.gg URL.**
3. **Never exclude newly released cards.** A 20-day history minimum once hid all
   117 July releases — the exact release-crash pattern the model is meant to trade.
4. **Never exclude cheap cards wholesale.** A 5,000-coin floor cut 69% of the
   market including the entire SBC fodder tier. Fodder is a volume game.
5. **Don't trust compounded backtest returns.** The all-in accounting compounds
   every trade and produced a *2,606,886%* "result". Judge on median card,
   per-trade return, and share of cards profitable.
6. **Percentage returns on cheap cards are noise.** "+31%" on a 200-coin card is
   63 coins and unfillable.
7. **Below the 30-day low is a falling knife, not a discount.** Gate it out.
8. **Backfill oldest-first.** Backfilling only currently-liquid cards created
   survivorship bias that made random trading look profitable (+3.6%/week).
9. **SQLite lock contention.** A long fetch will lock out a training run at its
   final insert. Commit per item, `busy_timeout=60000`, don't run two heavy
   writers at once.
10. **Tests must never hit the network.** A defaulted-on EA news fetch silently
    leaked into two tests.

---

## 5. Architecture

```
collectors/          raw data in (all plain httpx unless noted)
  card_list_source     full card registry
  bulk_price_source    whole market's prices in one pass (decoded CDN files, patchright)
  history_source       per-card daily history + real completed sales
  sbc_source           SBC windows
  ea_news_source       EA announcements
  social_sources       Reddit + YouTube buzz
  x_source             X/Twitter leaker feed (saved session)
services/            orchestration
  registry, backfill, bulk_collect, sales, liquidity, calendar
  scorecard    grades past picks -> the honest track record
  social       matches chatter to cards (prominence-resolved surnames)
  notify       posts a run summary to Discord
ml/                  the brain
  dataset      one row per (card, day): card + cohort + lifecycle + behaviour
  cohorts      groups (rating/position/league/nation/version/band) + relative strength
  lifecycle    season position, days to/from promos
  labels       forward returns + triple-barrier (tax-adjusted)
  validation   walk-forward splits WITH EMBARGO (never random k-fold)
  train        two heads, gated against a dumb baseline
  picks        the user-facing recommendations
  insights     cached market rhythms the dashboard reads
  dashboard    the live /ml web page (server-rendered, read-only)
top-level: cli, config, db, timeseries, secrets, security, alerts, webapp

The old rules-based engine (strategy/backtest/features/signals/advisor/scanner
and their collectors) was removed — this is an ML-only codebase now.
```

**Leakage discipline:** every rolling stat is shifted a day; a test asserts that
truncating the future leaves past features unchanged. Keep it that way.

---

## 6. Commands

```bash
futmarket picks --min-sales-per-hour 5     # ← the product. what to buy, and why
futmarket scorecard                        # how past picks have actually done
futmarket train                            # retrain (~15 min on full data)
futmarket build-dataset                    # inspect the feature matrix
futmarket sale-stats --limit 500           # refresh real sold prices
futmarket build-registry                   # refresh the card universe
futmarket collect-bulk                     # whole-market price snapshot (4 seconds)
futmarket backfill-history --order oldest   # deep history, oldest cards first
futmarket score-liquidity                  # recompute tradeability
futmarket build-calendar                   # promos/SBCs/EA news
futmarket social [--x]                     # collect Reddit/YouTube/X buzz
futmarket insights                         # refresh the dashboard's market rhythms
futmarket notify                           # post a run summary to Discord
futmarket dashboard --port 8899            # the live web dashboard (open /ml)
```

The 24/7 loop (`scripts/ml_daily.sh`, run by the `com.futmarket.ml` LaunchAgent)
chains: collect-bulk → picks --save → scorecard → insights → notify.

Environment: `uv`-managed venv, run via `.venv/bin/python -m futmarket.cli …`.
There is no `pip` in the venv — use `uv pip install`.

---

## 7. Data sources (all plain HTTP unless noted)

| What | Endpoint |
|---|---|
| Card registry | `www.fut.gg/api/fut/players/v2/26/?page=N` |
| Whole-market prices | `r2.fut.gg/26/player-prices-index…json` + `…-ps5-dyn….json` (browser needed to discover the rotating URL) |
| Per-card history + sales | POST `…/api/fut/price-access/sign/` then GET the signed URL |
| SBCs | `www.fut.gg/api/fut/sbc/26/?page=N` |
| EA news | `www.ea.com/games/ea-sports-fc/fc-26/news?page=N` (parse `__NEXT_DATA__`) |

Rate limits are real: use ~1.5–2s between per-card calls with exponential backoff.

---

## 8. What is NOT built yet

1. **News reactions** — the model knows *when* EA announced things but was never
   taught how cards *moved because of it*. Twitter/creator leaks aren't connected
   at all. The owner asked for this repeatedly; it's the biggest gap.
2. **Hour-of-day features** — the dump windows above are real, but only July has
   hourly history. The collector banks it now, so this sharpens weekly.
3. **Live paper-trading track record** — `picks --save` records calls; nothing
   scores them yet. This is the only real proof the system works.
4. **Sale-price coverage** — only a few thousand of 8,231 cards have real sale
   data.

---

## 9. How to work on this

- **Ship things the owner can look at.** Long measurement detours frustrate; a
  working `picks` output is worth more than another metric.
- **Show real output early and expect it to be wrong.** The owner spots bad picks
  in seconds by checking live prices. That feedback loop has been the single most
  effective debugging tool in this project.
- **Explain in trading terms**, not statistics.
- **Be honest about negative results.** The edge-equals-costs finding matters more
  than any optimistic number.
