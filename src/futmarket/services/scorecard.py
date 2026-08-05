"""The track record: what actually happened to every recommendation.

Until this existed we could only ever say whether a pick *looked* reasonable on
the day. That is not evidence. This closes the loop: every pick is recorded with
the price you'd have paid and the barriers it was judged against, then scored
once the market has had its say.

Two things this gets right that the previous version did not, and both of them
were costing real money:

**It reads the same price series the model was trained on.** Scoring used to walk
every raw snapshot -- every two-hourly print of the cheapest live listing. The
labels were built on daily prices. So a trade was graded on "did any tick touch
the stop", while the model had been taught "did the *day* close through it". The
median card-day ranges 14.3%, so the tick version stops out on sampling jitter
alone: 53 of 59 graded dip picks were stopped, most of them within hours. Both
ends now read ``db.daily_prices``.

**It records what the market did over the same window.** The median tradeable
card doesn't move at all over a fortnight, so after the ~7.4% round trip the
median *trade* is -6.9% before any skill is involved. A pick that returned -3% in
a month when everything returned -9% was a good call. Absolute return alone
cannot say that, so every closed pick now carries a benchmark and the headline
reports **alpha** beside it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from .. import db

logger = logging.getLogger(__name__)

TARGET, STOP, EXPIRED = "target", "stop", "expired"

# Below this a "win" is a handful of coins on a card nobody buys — a percentage
# move that's pure noise and unfillable. Judge the track record only on cards you
# could actually trade, so one lucky penny-card can't fake a good week.
MIN_TRADEABLE_PRICE = 1000

# The strategies currently being traded. They are judged **separately** — a deep
# dip and a promo release crash are different bets with different payoffs, and
# blending them would hide one working while the other doesn't. Everything else
# (the old rules engine, and the dip picks graded under the broken stop
# convention) is reported separately again — see db._migrate.
CURRENT_STRATEGIES = ("release_v1", "relval_v1", "weekend_v1")
CURRENT_STRATEGY = CURRENT_STRATEGIES[0]


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def score_pick(pick, daily: list[tuple[str, int, int]], *, tax_rate: float = 0.05,
               sell_slippage_pct: float = 0.0,
               now: datetime | None = None) -> tuple[str, int, float, str] | None:
    """(status, exit_price, realized_pct, exit_day) for one pick, else None if open.

    ``daily`` is the robust one-price-per-day series from ``db.daily_prices``.
    Whichever barrier a *daily* price closes through first wins; if neither is
    touched by the horizon, the position is marked out at the last observed
    price. The realized return is net of EA's tax AND a sell-under-listing
    slippage, so it reflects what you'd actually clear (entry already models the
    buy premium via ``buy_high``).

    **A barrier trade exits at the barrier, not at the price that broke it.** The
    old version booked whichever daily price it happened to observe on the day a
    barrier was crossed, which is not a trade anyone could have made in either
    direction: you have a sell listed *at* the target, so it fills there and not
    54% above it; and you are watching the position every two hours, so you come
    out near the stop rather than 57% below it. Both of those are real numbers
    from the live record -- the convention inflated wins and losses at once and
    made ``relval_v1`` read -34.9% on stops whose barrier sat at -15%. Genuine gap
    risk is charged through ``sell_slippage_pct``, where it can be measured,
    rather than smuggled into the exit price where it cannot.
    """
    now = now or datetime.now(timezone.utc)
    picked = _parse(pick["picked_at"])
    horizon = int(pick["chosen_horizon_days"] or pick["horizon_days"])
    deadline = picked + timedelta(days=horizon)
    entry = int(pick["entry_price"])
    if entry <= 0:
        return None

    # net proceeds per coin of sale price: after tax, then after selling slightly
    # under the going rate to actually get filled.
    sale_net = (1 - tax_rate) * (1 - sell_slippage_pct / 100.0)

    def realized(price):
        return (price * sale_net / entry - 1) * 100

    # Only prices *within the horizon* can decide the trade. Scoring runs on a
    # loop that has missed whole days before now, and without this bound a pick
    # whose deadline passed unscored gets graded on prices from after it expired.
    picked_day = picked.strftime("%Y-%m-%d")
    deadline_day = deadline.strftime("%Y-%m-%d")
    after = [(d, p) for d, p, _ in daily
             if picked_day < d <= deadline_day and p > 0]
    for day, price in after:
        if price >= pick["target_price"]:
            return TARGET, int(pick["target_price"]), realized(pick["target_price"]), day
        if price <= pick["stop_price"]:
            return STOP, int(pick["stop_price"]), realized(pick["stop_price"]), day

    if now < deadline:
        return None                      # still running, no verdict yet
    if not after:
        return None                      # no prices seen since the pick
    last_day, last_price = after[-1]
    return EXPIRED, last_price, realized(last_price), last_day


def market_return_pct(conn, start_day: str, end_day: str, *, source: str = "futgg",
                      min_price: int = MIN_TRADEABLE_PRICE) -> float | None:
    """Gross % move of the median tradeable card between two days.

    The comparison that makes a track record mean something. Deliberately the
    *median*, not the mean: a handful of cards that tripled would otherwise set
    a benchmark no real portfolio ever earned.
    """
    from statistics import median

    rows = conn.execute(
        """SELECT s.player_id,
                  AVG(CASE WHEN substr(s.timestamp,1,10)=? THEN s.price END) AS p0,
                  AVG(CASE WHEN substr(s.timestamp,1,10)=? THEN s.price END) AS p1
           FROM price_snapshots s
           WHERE s.source=? AND s.price>0
             AND substr(s.timestamp,1,10) IN (?, ?)
           GROUP BY s.player_id""",
        (start_day, end_day, source, start_day, end_day)).fetchall()

    moves = [(r["p1"] / r["p0"] - 1.0) * 100.0 for r in rows
             if r["p0"] and r["p1"] and r["p0"] >= min_price]
    return median(moves) if moves else None


def _benchmark_pct(conn, pick, exit_day: str, *, source: str, tax_rate: float,
                   sell_slippage_pct: float, cache: dict) -> float | None:
    """``market_return_pct`` over this pick's exact window, charged the same costs.

    The window ends on the day the trade actually came out, **not** on its
    horizon. Using the horizon looked harmless and silently destroyed most of the
    alpha record: a trade that stopped out on day 2 of a 10-day horizon asked the
    market what it did over a window ending eight days in the future, got None
    back, and stored no benchmark at all. Live, that left every ``target`` and
    ``stop`` row without a benchmark and only the ``expired`` ones -- the ones
    that ran their full horizon -- with one, so alpha was being read off the
    third of the record that had, by construction, gone nowhere.

    Cached per window: a scoring run resolves many picks made in the same cycle,
    and the market only needs measuring once per (start, end) pair.
    """
    picked = _parse(pick["picked_at"])
    window = (picked.strftime("%Y-%m-%d"), exit_day)
    if window not in cache:
        cache[window] = market_return_pct(conn, *window, source=source)
    gross = cache[window]
    if gross is None:
        return None
    sale_net = (1 - tax_rate) * (1 - sell_slippage_pct / 100.0)
    return ((1 + gross / 100.0) * sale_net - 1) * 100.0


def score_open_picks(conn, *, title: str = "fc26", source: str = "futgg",
                     tax_rate: float = 0.05, sell_slippage_pct: float = 0.0,
                     now: datetime | None = None) -> dict:
    """Resolve every pick the market has answered. Returns counts by outcome."""
    now = now or datetime.now(timezone.utc)
    res = {"checked": 0, TARGET: 0, STOP: 0, EXPIRED: 0, "still_open": 0}
    bench_cache: dict = {}
    for pick in db.open_picks(conn, title=title):
        res["checked"] += 1
        daily = db.daily_prices(conn, pick["player_id"], source)
        outcome = score_pick(pick, daily, tax_rate=tax_rate,
                             sell_slippage_pct=sell_slippage_pct, now=now)
        if outcome is None:
            res["still_open"] += 1
            continue
        status, exit_price, realized, exit_day = outcome
        bench = _benchmark_pct(conn, pick, exit_day, source=source, tax_rate=tax_rate,
                               sell_slippage_pct=sell_slippage_pct,
                               cache=bench_cache)
        db.close_pick(conn, pick["id"], status=status, exit_price=exit_price,
                      realized_pct=realized, benchmark_pct=bench, at=now)
        res[status] += 1
    conn.commit()
    logger.info("scorecard: %s", res)
    return res


def regrade_closed_picks(conn, *, title: str = "fc26", source: str = "futgg",
                         strategies: tuple[str, ...] | None = None,
                         tax_rate: float = 0.05, sell_slippage_pct: float = 0.0,
                         dry_run: bool = False,
                         now: datetime | None = None) -> dict:
    """Re-score already-closed picks against the current grading rules.

    Grading conventions have changed twice now, and both times the stored record
    silently became a mix of numbers produced by different rules. That is worse
    than either convention on its own: the headline blends them and means
    nothing. This re-runs the closed rows through ``score_pick`` as it stands
    today, which is exact rather than approximate -- the barriers, the entry and
    the whole price series are all still in the database, so nothing has to be
    estimated.

    A pick may legitimately change status: a stop that was only touched *after*
    the horizon is an expiry, not a stop. Rows the market can no longer answer
    (no prices in their window) are left exactly as they are and counted in
    ``skipped``.
    """
    now = now or datetime.now(timezone.utc)
    sql = ("SELECT * FROM pick_log WHERE title=? "
           "AND status IN ('target','stop','expired')")
    params: list = [title]
    if strategies:
        sql += f" AND strategy IN ({','.join('?' * len(strategies))})"
        params.extend(strategies)

    res = {"checked": 0, "changed": 0, "skipped": 0, "status_changed": 0,
           "benchmark_added": 0}
    bench_cache: dict = {}
    for pick in conn.execute(sql, params).fetchall():
        res["checked"] += 1
        daily = db.daily_prices(conn, pick["player_id"], source)
        outcome = score_pick(pick, daily, tax_rate=tax_rate,
                             sell_slippage_pct=sell_slippage_pct, now=now)
        if outcome is None:
            res["skipped"] += 1
            continue
        status, exit_price, realized, exit_day = outcome
        bench = _benchmark_pct(conn, pick, exit_day, source=source,
                              tax_rate=tax_rate,
                              sell_slippage_pct=sell_slippage_pct,
                              cache=bench_cache)
        if status != pick["status"]:
            res["status_changed"] += 1
        if bench is not None and pick["benchmark_pct"] is None:
            res["benchmark_added"] += 1
        if (status != pick["status"] or exit_price != pick["exit_price"]
                or bench != pick["benchmark_pct"]):
            res["changed"] += 1
        if not dry_run:
            db.close_pick(conn, pick["id"], status=status, exit_price=exit_price,
                          realized_pct=realized, benchmark_pct=bench,
                          at=_parse(pick["scored_at"]) if pick["scored_at"] else now)
    if not dry_run:
        conn.commit()
    logger.info("regrade%s: %s", " (dry run)" if dry_run else "", res)
    return res


def summaries(conn, *, title: str = "fc26", strategies=CURRENT_STRATEGIES,
              **kwargs) -> dict[str, dict]:
    """One honest record per live strategy, never blended."""
    return {s: summary(conn, title=title, strategy=s, **kwargs) for s in strategies}


def summary(conn, *, title: str = "fc26", min_price: int = MIN_TRADEABLE_PRICE,
            strategy: str | None = CURRENT_STRATEGY) -> dict:
    """The honest headline: how the recommendations have actually done.

    Judged in *coins*, not percentages, and only on cards you could actually
    trade (>= ``min_price``). Two numbers matter:

    ``return_on_capital_pct`` -- if you'd bought one of every pick, how did the
    whole pile do. Weights each result by the coins at stake, so a +285%
    penny-card can't outvote a real trade.

    ``alpha_vs_market_pct`` -- the same figure minus what the market did over
    those same windows. In a market whose median card is inert, this is the one
    that says whether the system knows anything.

    ``strategy`` filters to one strategy (default: the current one), so it is
    never blended with 'legacy' or with 'dip_v1_broken', whose stops were graded
    against the wrong price series.
    """
    sql = ("SELECT status, realized_pct, entry_price, benchmark_pct "
           "FROM pick_log WHERE title=?")
    params = [title]
    if strategy is not None:
        sql += " AND strategy=?"
        params.append(strategy)
    rows = conn.execute(sql, params).fetchall()
    total = len(rows)
    closed = [r for r in rows if r["status"] != "open"]
    base = {"total": total, "open": total - len(closed), "closed": len(closed),
            "strategy": strategy or "all"}

    # Only cards we could actually fill, and that the market has answered.
    graded = [r for r in closed
              if r["realized_pct"] is not None
              and r["entry_price"] and r["entry_price"] >= min_price]
    base["graded"] = len(graded)
    if not graded:
        return base

    n = len(graded)
    wins = [r for r in graded if r["status"] == TARGET]
    rets = sorted(r["realized_pct"] for r in graded)
    # net coins made/lost buying one of each card
    coins = [r["entry_price"] * r["realized_pct"] / 100.0 for r in graded]
    capital = sum(r["entry_price"] for r in graded)
    median = rets[n // 2] if n % 2 else (rets[n // 2 - 1] + rets[n // 2]) / 2

    benched = [r for r in graded if r["benchmark_pct"] is not None]
    alpha = None
    if benched:
        bench_coins = sum(r["entry_price"] * r["benchmark_pct"] / 100.0 for r in benched)
        bench_capital = sum(r["entry_price"] for r in benched)
        own_coins = sum(r["entry_price"] * r["realized_pct"] / 100.0 for r in benched)
        if bench_capital:
            alpha = round((own_coins - bench_coins) / bench_capital * 100, 2)

    return {
        **base,
        "hit_target": len(wins),
        "hit_stop": sum(1 for r in graded if r["status"] == STOP),
        "expired": sum(1 for r in graded if r["status"] == EXPIRED),
        "win_rate": round(len(wins) / n, 4),
        "profitable_share": round(sum(1 for r in graded if r["realized_pct"] > 0) / n, 4),
        "coins_pnl": round(sum(coins)),
        "avg_coins_per_trade": round(sum(coins) / n),
        "return_on_capital_pct": round(sum(coins) / capital * 100, 2) if capital else 0.0,
        "alpha_vs_market_pct": alpha,
        "benchmarked": len(benched),
        "median_return_pct": round(median, 2),
    }
