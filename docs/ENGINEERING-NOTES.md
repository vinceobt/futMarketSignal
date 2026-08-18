# Engineering notes

Everything below was measured, not assumed. It records what I learned building
this system: how the EA FC transfer market actually behaves, which trades survive
costs, which ideas failed, and the bugs that produced convincing but wrong
numbers along the way.

If you only read one section, read §3 (what I measured) and §4 (the traps).

---

## 1. What this is

**A self-improving machine-learning system that learns how EA FC Ultimate Team
cards behave, and says what to buy and sell.**

The goal I set: a model that trains on history, keeps training forever, and
makes independent decisions from what it sees on fut.gg, EA's site, and social
media — not a formula I hand-tuned.

**End goal:** by the EA FC 27 launch (~Sept 2027) the model is at expert level
and still improving. Training never stops; it carries forward into each new game.

### Design goals
- **Extensive training on historical data.** Feed it everything.
- **Learn behaviour, not formulas** — how cards react to news, promos, reward
  drops, day of week, time of day.
- **Independent decisions.** It should say *"buy this card, at this price, now"*.
- **Plain output.** No statistics lectures. Real, actionable calls.

### Non-goals
- **Benchmarking against the old rules engine.** It never made money — that's
  why this project exists. Beating it would prove nothing.
- **Statistical jargon in the output.** Every number the system prints has to
  say what it means for a trade.

---

## 2. Current state (2026-07-25)

Working, end to end, with a **command-center dashboard** you run everything from.

| Layer | State |
|---|---|
| Card registry | 9,998 cards (8,231 tradeable) |
| Price history | **2.85M snapshots**, full season since 2025-09-08 |
| Event calendar | 247 events; promos classified by **type** (Icon/Hero/TOTS/…) |
| Features | card behaviour + cohorts + lifecycle + weekly rhythm + release curve + social buzz + major-promo distance + **market drift / excess return** |
| Models | scikit-learn HistGradientBoosting — **excess** + **clears-cost** heads, one per horizon (3/5/7/10/14/21d) |
| Strategies | **`release_v1`** (promo card 4–6 days into its release crash, held ~3 weeks), **`relval_v1`** (deep dip, z ≤ −1.5 on its 30-day floor, held ~2 weeks) and **`weekend_v1`** (a tier-A card that swings ≥30% weekly, bought Sat/Sun, sold 3–4 days later). Near-disjoint; judged separately. |
| Prices | listing index (`futgg`) **+ real completed sales banked as their own series (`futgg_sold`)** |
| Output | `futmarket picks` · `futmarket advise` (consult) · **`futmarket evaluate`** (the honest scoreboard) · dashboard `/ml` |

### The pivotal findings (why the strategy is what it is)

**1. The round trip is the whole game.** 5% EA tax + 2% sell slippage means a
trade must gain **+7.4% gross to break even**. And **the median tradeable card
does not move at all over a fortnight** — so the median *trade*, before any skill
is involved, is exactly **−6.9%**. Half the market is inert; the costs are the
entire obstacle. Any rule whose expected move is smaller than 7.4% loses, however
good the model is. (Some months the market genuinely falls too: −56% net in
2025-09, −19% in 2026-02.)

**2. The old strategy was mathematically incapable of working.** `dip_v1` bought
a shallow dip (z ≤ −0.5) and held 5 days. That bounce is ~4.7% gross — below the
break-even. Measured month by month on tradeable cards it nets **−2.85%**.

**3. But the dip signal is real.** A *deep* dip (z ≤ −1.5, within 3% of the
30-day floor) held **14 days** nets **+4.14%** and beats the market in **10 of 11
months**; z ≤ −2.0 nets **+9.53%** with a 65% win rate. The edge grows with
holding period because the bounce needs time to clear the costs.

**4. Alpha, not absolute return, is the honest measure.** In a market where half
of all cards are inert, "−3% in a month when everything did −9%" is a good call.
Every backtest and the live scorecard now report the gate's median minus the
same-month, same-tier universe median.

**5. Illiquid cards fake an edge.** Tier C returns +17.3% on the dip trade, tier A
just +1.6%. That inversion is measurement noise — the thinner the card, the more
its "price" is one stale listing. Headlines are restricted to tier A/B.

**6. The release crash is the better trade, and it isn't where the folklore says.**
Measured on liquid special cards (base golds excluded — they have no release
event), net of all costs, month by month:

| Buy at age | held 7d | held 14d | held 21d |
|---|---|---|---|
| 0–1 days (the debut) | −11.3% | −6.9% | −3.3% |
| **4–6 days** | −0.7% | **+7.0%** | **+11.5%** |
| 7–9 days (the "day-nine bottom") | −2.0% | +1.6% | +3.7% |

At 21 days the 4–6 window returns **+11.5% median, +25pp alpha, 67% win rate,
positive in 10 of 10 months** — and unlike the dip trade, **tier A is the *best*
tier** (+35.6%, 79% win, vs +10.4% on tier B). When the *most* tradeable cards do
best, the effect is real; the reverse pattern is what an artifact looks like.

Two corrections to long-held beliefs here: the bottom to buy is **day 4–6, not
day 9**, and **never buy the debut** — age 0–1 loses money and lags the market at
every horizon.

**7. We can predict *whether* a dip pays, not *how much*.** Measured walk-forward
on the traded slice:

| Head | Asks | Result |
|---|---|---|
| `clears` | will this trade beat the market and cover costs? | **52–59% vs a 35–42% base rate** — real skill |
| `excess` | by how much will it beat the market? | **−12% to −19% skill** — worse than assuming it doesn't |

So the strategy uses the classifier for the per-card call and the gate's own
**measured** win/loss history for the size of the payoff. Nothing is built on the
magnitude head; it is still trained and reported so we notice if it ever starts
working. This is why `picks._choose_horizon` takes probabilities and a payoff
profile rather than a predicted return.

**8. The weekly cycle is only tradeable on the cards that actually swing.**
Prices sag into the weekend as rewards flood the market and recover midweek. The
direction is the strongest day pattern there is — every one of the six best
buy/sell day pairs is a weekend buy — but on the average card the best pair
(Sunday→Thursday) is **+4.60% gross, −2.62% net**. Buying "the weekend" as a
blanket rule loses money for exactly the reason `dip_v1` did.

What makes it a trade is that the swing is a property of the **individual card**
and it persists week to week. Bucketing each weekend trade by that card's *own*
trailing swing (prior weeks only):

| Card's past swing | next trade, net |
|---|---|
| 0–5% | −3.18% |
| 10–15% | +1.18% |
| 20–25% | +6.89% |
| 30–35% | **+19.57%** |

So the rule is **"buy the cards that breathe, in the weekend"** — most cards are
inert, some swing 20%+ every single week. Two things settle the shape of it:

- **The sell day is 3–4 days out, and holding longer gives it all back.** On tier
  A at a swing floor of 30: hold 2d −2.9%, **3d +2.4%, 4d +4.2%**, 5d −0.1%,
  6d −6.9%, 7d −14.1%. By day 7 the card is simply back in the next weekend dump.
  Measured, not assumed — the horizon list carries 2/3/4/5/6 so every midweek
  exit could be graded, and the model picks between 3, 4 and 5 per card.
- **The swing floor has to be high, and tier A is the tell.** At swing ≥10 the
  blend is −1.9% net and tier A is *worse* than tier B — the artifact signature.
  Raising the floor inverts it: at ≥30, tier A nets **+4.2% with +16.7pp alpha
  and a 59% win rate over 1,059 trades**, while tier B nets −4.7%. Trading tier A
  alone is the difference between a trade and a mirage.

Note the trap this walks past: alpha alone would have shipped the ≥10 version,
which beats the market in 11 of 11 months and still **loses coins**. You cannot
spend alpha — the absolute number has to clear zero too.

The trained model then confirmed the sell day independently. Out of sample, on
the gated slice, what the trade actually pays:

| hold | 2d | 3d | **4d** | 5d | 6d | 7d | 14d |
|---|---|---|---|---|---|---|---|
| pays off | 47% | 58% | **61%** | 52% | 41% | 32% | 27% |
| blind EV | +0.6% | +4.9% | **+6.2%** | +2.1% | −7.1% | −14.1% | −19.3% |

Both the win rate *and* the size of the loss get worse past 5 days (the median
loss goes from −14.7% at 4d to −24.8% at 6d): hold into the next weekend and you
are simply in the next dump. And `weekend_v1` is the only gate where the model
has skill at **every** horizon (1.10x–1.38x over the base rate).

**Which is more than can be said for the release gate.** The same training run
puts `release` at **0.842x at 14 days and 0.978x at 21** — i.e. *worse than
picking blind*, at exactly the two horizons `release_v1` trades. Its measured
payoff is still strong, so the trade may work while the model's per-card call on
it does not; but nothing should be added to that strategy until this is
understood.

**Honest status:** `relval_v1` and `weekend_v1` both backtest positive and are
**not yet proven live**. Backtests flatter; the coin-weighted scorecard, read as
**alpha**, is the real judge. See §10.

The previous live record (`dip_v1`, 53 stops / 6 targets) measures a **bug, not a
strategy** — see trap #11. Those rows are retagged `dip_v1_broken` and never
appear in a headline.

---

## 3. What I measured (this is the valuable part)

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

**Do not re-introduce these.** Most were caught by checking live prices by hand,
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
11. **Barriers must be levels on the series that grades them.** `dip_v1` derived
    its stop from `entry` (= `buy_high`, ~5% over market) but the scorer compared
    it against the raw market price. The stop therefore landed **0.76% *above***
    the live price on average, and **53 of 59** trades stopped out — most within
    hours. A test now asserts `stop_price < market_price` at any buy premium.
12. **Never grade on a different price series than you train on.** Labels used
    daily prices; scoring walked every 2-hourly tick. The median card-day ranges
    **14.3%**, so the tick version stops out on sampling jitter. Both ends now
    read the same robust daily series (`db.daily_prices`).
13. **Validate on the population you actually trade.** The old direction head
    showed 2.4x lift over 771k rows and went **0-for-26** on its highest-confidence
    live picks — it was never measured on the narrow gated slice it was asked
    about. Training now reports a `gated_precision` beside every global metric.
14. **Take the day's median snapshot, not its last.** Using the last print gave
    day-to-day return std of **193%**; the median gives **41%**. More than half
    this market's apparent volatility was an artifact of collector timing.
15. **One open position per card.** The loop re-derives the shortlist every two
    hours. Without a dedupe, 92 "trades" were really 37 cards, some simultaneously
    `open` and `stop`. `db.has_open_pick` guards both `generate` and `save`.
16. **A model that beats a weak baseline has proved nothing.** The forecaster beat
    "assume no change" by **0.42%** — i.e. not at all — and shipped anyway. The
    baseline is now "assume this card moves with the market".
17. **The release trade dies silently if new cards aren't collected within a day.**
    Cards released 19–24 Jul got their first price snapshot on the 25th, so every
    one of those day-4-to-6 windows was missed and `picks` simply returned
    nothing — looking like "no opportunities" rather than a broken pipeline. The
    loop now runs `build-registry --max-pages 4` **every cycle** (page 1 of
    fut.gg's list is the newest cards, so it costs 4 requests), and
    `MIN_HISTORY_DAYS` is 2 to leave slack for a day of collection lag. If
    `release_v1` goes quiet for more than a week, check this first:
    `select max(release_date) from card_meta` against the first snapshot date for
    those cards.
18. **`scripts/ml_daily.sh` is a template, not what runs.** The LaunchAgent runs
    the rendered copy at `data/.ml_daily.rendered.sh`. Editing the template alone
    changes nothing — re-render it (`make autonomous-install`, or the `sed` block
    in `scripts/install_ml.sh` if you don't want to bounce the agent).
19. **A capped collector must order by staleness, not by rank.** `sale-stats`
    ordered by liquidity, so `--limit N` re-fetched the same N cards every cycle
    and coverage never grew past the head of the list. Now: never-fetched first,
    then least-recently-refreshed. Any future capped sweep needs the same shape.
20. **Don't bulk-sweep fut.gg.** A continuous 8k-card run at 1.6s stalled at ~600
    cards behind escalating backoff (10s → 320s) and banked nothing further for
    40 minutes. Rate limits are real (§7). Collect in slices from the 2-hourly
    loop instead — 250 cards at 2s covers the universe in three days and then
    keeps it fresh.
21. **A barrier trade exits at its barrier, not at the price that broke it.** The
    scorer booked whichever daily price it happened to observe on the day a
    barrier was crossed. Live, stops correctly placed at −15% were realizing
    **−34.9%** (tail to −68%, prints as far as 57% *below* the stop), while
    targets booked up to 54% *above* the target. It inflated wins and losses at
    once and made the whole `relval_v1` record unreadable. Book the level; charge
    genuine gap risk through `sell_slippage_pct`, where it can be measured.
22. **The benchmark window must end when the trade did, not at its horizon.**
    `_benchmark_pct` measured the market from entry to *horizon*, so a trade that
    stopped out on day 2 of a 10-day horizon asked what the market did over a
    window ending eight days in the future, got nothing back, and stored no
    benchmark. All 89 `target`/`stop` rows had NULL `benchmark_pct` while the 46
    `expired` ones — the ones that by construction went nowhere — had one. Alpha,
    the headline number, was being computed from the third of the record least
    able to show any.
23. **Barriers must also be bounded by the horizon.** The scorer walked every
    price after the pick, not just those inside its window. Combined with the
    loop going dark for 48 hours (it has), a pick whose deadline passed unscored
    got graded on whatever the card did *after* it should have been sold.

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
  labels       excess return + clears-cost, at every horizon (+ triple-barrier)
  evaluate     THE HONEST SCOREBOARD: costs, alpha vs market, by month and tier
  validation   walk-forward splits WITH EMBARGO (never random k-fold)
  train        two heads x 5 horizons, judged on the slice we actually trade
  picks        the user-facing recommendations (incl. how long to hold)
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
futmarket picks --min-sales-per-hour 5     # ← the product. both strategies, ranked
futmarket picks --strategy release_v1      # just the promo release-crash trade
futmarket evaluate --gate all              # ← would this rule have made money?
futmarket scorecard                        # how past picks have actually done
futmarket regrade --dry-run                # re-score closed picks after a grading change
futmarket train                            # retrain (~30 min: 2 heads x 5 horizons)
futmarket build-dataset                    # inspect the feature matrix
futmarket sale-stats --tiers ABC           # real sold prices; banks the 'futgg_sold' series
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
   at all. This is the biggest remaining gap.
2. **Hour-of-day features** — the dump windows above are real, but only July has
   hourly history. The collector banks it now, so this sharpens weekly.
3. **Live proof of `relval_v1`** — the strategy backtests positive and beats the
   market in 10 of 11 months, but has no live record yet. This is the only real
   proof the system works.
4. **A proven-cleaner price signal.** Real completed sales are now banked as
   their own series (`futgg_sold`, ~5 points/card/fetch, collected every cycle),
   but **the noise claim is not yet demonstrated**: over the first 321
   overlapping hours the sold series had lower variance (std 60% vs 66%) and a
   *higher* median hourly move (10.0% vs 7.7%). Re-measure after a few weeks —
   with 3+ days of history `--source futgg_sold` feeds the whole existing
   pipeline (features, labels, evaluate, train) with no further work.
   Also measured: real trades clear only **~0.8% above** the cheapest listing at
   hourly grain, which is why `buy_premium_pct` stays near zero.
5. **More event-driven trades.** The release curve is now traded (`release_v1`,
   §2 finding #6) and is the strongest edge measured so far. The same treatment
   has *not* been applied to the other events already in the calendar: SBC
   fodder spikes, TOTW day, reward drops. Each is a candidate gate — add it to
   `evaluate.GATES` and run `futmarket evaluate` before writing any strategy code.

---

## 9. Working principles

- **Ship something you can look at.** Long measurement detours stall progress; a
  working `picks` output is worth more than another metric.
- **Show real output early and expect it to be wrong.** Checking a pick against
  the live market takes seconds and exposes bad ones immediately. That loop has
  been the single most effective debugging tool in this project.
- **Explain in trading terms**, not statistics.
- **Be honest about negative results.** The edge-equals-costs finding matters more
  than any optimistic number.

---

## 10. Status as of 2026-08-05

Two things happened on 2026-08-05, and the second is why the first matters.

**The live record was measuring the grader, not the strategies.** `relval_v1` had
135 graded trades reading −14.8% on capital. Three defects, all now fixed and
tested (traps #21–#23): trades were booked at whatever price broke a barrier
rather than at the barrier, so stops set at −15% realized −34.9%; the benchmark
window ran to the horizon rather than to the actual exit, so **89 of 135 rows had
no benchmark at all** and alpha was computed from the 46 that went nowhere; and
barrier walks weren't bounded by the horizon, so a pick left unscored during a
loop outage got graded on what happened afterwards.

All 142 closed rows were then re-scored under the fixed rules (`futmarket
regrade`; the barriers and full price history are still in the DB, so this is
exact, not an estimate). 96 changed, 4 changed verdict, and 96 gained the
benchmark they never had. **The honest live record:**

| Strategy | Graded | On capital | Alpha | Win |
|---|---|---|---|---|
| `release_v1` | 7 | **+8.6%** | **+15.5%** | 86% |
| `relval_v1` | 137 | **−11.0%** | **−4.9%** | 23% |

So the grading bug was real but it was not the whole story: correctly graded,
`relval_v1` is still clearly losing (57 stops against 32 targets), and its alpha
is negative on a sample big enough to believe. Per the decision tree below that
means **the gate is overfit to the season** — it is the next thing to re-cut, and
`futmarket evaluate --gate relval_v1` including 2026-07 and 2026-08 is where to
start. `release_v1` is positive on both counts but 7 trades is not yet a record.

**A third strategy was added — `weekend_v1`, the weekly supply cycle** (finding
#8). Buy a tier-A card that swings ≥30% weekly on a Saturday or Sunday, sell 3–4
days later: +4.2% net, +16.7pp alpha, 59% win over 1,059 trades. The sell day was
derived rather than assumed — horizons 2/3/4/5/6 exist precisely so every midweek
exit could be graded, and the curve peaks hard at 3–4 days and is deeply negative
by 7.

**The next move is still not building. It is reading the alpha number** — now
that the grader can produce an honest one.

### What was wrong in the 2026-07-25 rebuild (all measured, all fixed then)
| Defect | Evidence |
|---|---|
| Stops landed **above** the market price | avg −0.76% "room"; 53 stops / 6 targets |
| Model confidence was **inverted** live | high-conf 0-for-26 (−15.0%); low-conf 5-of-17 (+6.6%) |
| Labels on daily prices, scoring on 2-hourly ticks | median card-day ranges 14.3% |
| Daily price took the **last** snapshot | return std 193% vs 41% for the median |
| Forecaster had no skill and was never stored | 0.42% over "assume no change"; `predictions` empty |
| Track record double-counted | 92 picks over 37 cards, some `open` and `stop` at once |
| Horizon too short to clear costs | 5-day bounce ≈4.7% gross vs a 7.4% round trip |

### What was built
- **`ml/evaluate.py` — the honest scoreboard.** Costs applied once, calendar-matched
  forward returns, results **by month and by liquidity tier**, headline is
  **median alpha vs the same-month universe**. `futmarket evaluate --gate all`.
  Nothing ships unless it beats the market in ≥8 of 11 months.
- **`relval_v1`** — deep dip (z ≤ −1.5, floor ≤ 3%), stop **outside** the noise
  (≥15%), barriers anchored to the price series that grades them, and a
  **model-chosen holding period** per trade (best expected net **per day held**).
  If nothing clears the round trip, it says **buy nothing** — the old code had no
  way to give that answer.
- **Market-relative model** — two heads (`excess`, `clears`) at 5 horizons,
  validated on the **gated slice**, not the whole market. Only `clears` earns its
  place (see finding #6); expected value pairs its probability with the gate's
  measured payoff profile, stored in the artifact by `evaluate.payoff_profile`.
- **Robust daily price** — the day's median with an outlier guard, used by the
  dataset *and* the scorecard so labels and grading can never disagree again.
- **Alpha in the scorecard, dashboard and Discord**, plus one open position per card.

### What to check in ~3 weeks (the whole point)
1. `futmarket scorecard` / dashboard Track record — now printing **every** live
   strategy, not just the first. **Read `alpha_vs_market_pct` first**, then
   `return_on_capital_pct`. Trades accumulate slowly: the gates are selective,
   and `weekend_v1` can only open on two days a week.
2. If alpha is **solidly positive** → the edge is real; consider loosening the
   gate to trade more (`entry_z_max` toward −1.0, or the weekend swing floor
   toward 20), and re-run `futmarket evaluate` to confirm it still clears costs.
3. If alpha is **around zero** → the signal is real but execution is eating it.
   Check whether the cards are actually fillable at the quoted band; raise
   `buy_premium_pct` in config.yaml to the truth and re-evaluate.
4. If alpha is **negative** → the gate is overfit to the season. Re-run
   `futmarket evaluate --gate all` including the newest month and look at which
   months carry it (2026-06 is exceptional; 2025-10 is negative for every gate).

**Check the absolute return too, not only alpha.** The weekend gate at a swing
floor of 10 beats the market in 11 of 11 months and still loses coins. Alpha says
the model knows something; only the absolute number says you made money.

### Open items / next moves (roughly by value)
- **Prove it live** (above) — nothing to build, just read the number.
- **Better price signal.** Real completed sales cover only ~2,000 of 8,231 cards.
  The lowest-BIN snapshot is the noisiest input in the system and the biggest
  remaining accuracy win; `sale-stats` coverage is the cheapest path to it.
- **Re-grade or retire the pre-2026-08-05 record.** The 142 `relval_v1` /
  `release_v1` rows graded under the broken barrier convention are still in
  `pick_log` and still counted. Either re-run them through the fixed
  `score_pick` (the barriers and the price series are both still there, so this
  is exact) or retag them the way `dip_v1_broken` was. Leaving them is the worst
  of the three — the headline currently blends honest and dishonest rows.
- **Event-driven moves.** Mean reversion pays ~5-10% over a fortnight. A promo
  release crash is −33% over four days. The release curve and the weekly cycle
  are now both traded; **SBC fodder spikes, TOTW day and reward drops are not.**
  Each is a candidate gate — add it to `evaluate.GATES` and run
  `futmarket evaluate` before writing any strategy code.
- **`generate_all` rebuilds the feature matrix once per strategy.** Three
  strategies means three ~3-minute builds every 2-hourly cycle. Build once and
  pass the frame in.
- **Make the magnitude head work, or drop it.** `excess` currently has negative
  skill (finding #6). Either better inputs fix it (real sale prices, event
  features) or it should go — carrying a head nothing consumes is a liability,
  because the next person will assume it works.
- **News *content*** beyond promo type is still unbuilt — the largest research bet.
- **Retrain cadence:** the 2-hourly cycle does NOT retrain. Retrain manually
  (`futmarket train`, ~30 min now) after meaningful new data, or add a weekly one.
- **Watch out:** bump `STRATEGY_VERSION` in `ml/picks.py` whenever the trade logic
  changes, so the record stays honest. `dip_v1_broken` and `legacy` are excluded
  from every headline but kept in the DB.

### Running state
Mac must stay **awake + plugged in** (LaunchAgents: `awake`, `dashboard`, `ml`).
The `ml` cycle every 2h: collect-bulk → picks --save → sell-alerts → scorecard →
notify. Dashboard: `http://localhost:8899/` (or the LAN IP from a phone).
