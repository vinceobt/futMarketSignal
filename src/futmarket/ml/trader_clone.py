"""Learn to make calls like the Discord pros.

The traders' buy calls are teaching examples: at the moment a pro said "buy this
card", the card's market looked a certain way (price, momentum, how far off its
recent floor, day of week...). This trains a model to recognise that setup, so it
can make its OWN buy calls on today's market -- imitating the traders' judgement.

  positives = (card, day) a trader actually called a BUY
  negatives = (card, day) with no trader call -> "not a buy"
  model     = HistGradientBoosting classifier -> P(a pro would buy this now)

Then we score every live card and surface the model's own buy list.
"""

from __future__ import annotations

import math
import sqlite3
import statistics
from datetime import timedelta

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from futmarket.ml.discord_scorecard import (build_index, resolve, series, _near,
                                            _entry, _to_utc, ROOT)

BUY_PREMIUM = 1.04   # you pay ~4% over the listing
SELL_TAX = 0.95      # EA takes 5% on the sale

FEAT_NAMES = ["log_price", "ret_1d", "ret_3d", "ret_7d", "range_pos_14",
              "drawdown_14", "vol_14", "rating", "day_of_week"]


def feats(ser, asof, rating):
    """Market-state features for a card as of `asof`, from its price series."""
    hist = [(t, p) for t, p in ser if t <= asof and p > 0]
    if len(hist) < 5:
        return None
    price = hist[-1][1]

    def ret(days):
        cut = asof - timedelta(days=days)
        past = [p for t, p in hist if t <= cut]
        return price / past[-1] - 1 if past else 0.0

    p14 = [p for t, p in hist if t >= asof - timedelta(days=14)]
    mn, mx = min(p14), max(p14)
    rng = (price - mn) / (mx - mn) if mx > mn else 0.5
    dd = price / mx - 1 if mx > 0 else 0.0
    rets = [p14[i] / p14[i - 1] - 1 for i in range(1, len(p14)) if p14[i - 1] > 0]
    vol = statistics.pstdev(rets) if len(rets) > 1 else 0.0
    return [math.log(max(price, 1)), ret(1), ret(3), ret(7), rng, dd, vol,
            rating or 0, asof.weekday()]


def build_dataset(m, d, idx, neg_per_pos=2):
    """Positives from trader buy calls; negatives from random no-call (card, day)."""
    # positives
    pos_keys = set()          # (pid, date) that ARE trader buys -> keep out of negatives
    X, y, dates = [], [], []
    rating_of = dict(m.execute("SELECT player_id, rating FROM card_meta"))
    for row in d.execute("SELECT card, version, price, price_kind, timestamp "
                         "FROM discord_calls WHERE action='buy'"):
        call = dict(zip(["card", "version", "price", "price_kind", "timestamp"], row))
        pid = resolve(m, idx, call)
        if not pid:
            continue
        when = _to_utc(call["timestamp"])
        pos_keys.add((pid, when.date()))
        ser = series(m, pid, when - timedelta(days=20), when)
        f = feats(ser, when, rating_of.get(pid))
        if f:
            X.append(f); y.append(1); dates.append(when)

    n_pos = len(y)
    # negatives: the SAME cards pros trade, on OTHER days when nobody called a buy.
    # This forces the model to learn *timing*, not just which cards are popular.
    pool = sorted({pid for pid, _ in pos_keys})
    all_days = sorted({dt.date() for dt in dates})
    rng = np.random.default_rng(7)
    tries = 0
    while sum(v == 0 for v in y) < n_pos * neg_per_pos and tries < n_pos * neg_per_pos * 40:
        tries += 1
        pid = pool[rng.integers(len(pool))]
        day = all_days[rng.integers(len(all_days))]
        if (pid, day) in pos_keys:
            continue
        when = _to_utc(f"{day}T18:00:00Z")
        ser = series(m, pid, when - timedelta(days=20), when)
        f = feats(ser, when, rating_of.get(pid))
        if f:
            X.append(f); y.append(0); dates.append(when)
    return np.array(X), np.array(y), dates


def train_and_pick(top=20):
    m = sqlite3.connect(ROOT / "data" / "market.db")
    d = sqlite3.connect(ROOT / "data" / "discord.db")
    idx = build_index(m)

    X, y, dates = build_dataset(m, d, idx)
    print(f"training rows: {len(y)}  ({int(sum(y))} trader buys, {int(len(y)-sum(y))} non-calls)")

    # time split: learn on earlier calls, test on later ones (no leakage)
    order = np.argsort([dt.timestamp() for dt in dates])
    X, y = X[order], y[order]
    cut = int(len(y) * 0.75)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         max_depth=4, random_state=0)
    clf.fit(X[:cut], y[:cut])
    auc = roc_auc_score(y[cut:], clf.predict_proba(X[cut:])[:, 1])
    print(f"held-out AUC (can it tell a pro's buy setup from a random card?): {auc:.3f}")
    print("  0.5 = coin-flip, 1.0 = perfect.  feature weights learned from the pros.\n")

    # retrain on everything, then make the model's OWN calls on today's market
    clf.fit(X, y)
    latest = _to_utc(m.execute("SELECT MAX(timestamp) FROM price_snapshots").fetchone()[0])
    rating_of = dict(m.execute("SELECT player_id, rating FROM card_meta"))
    rows = []
    active = [r[0] for r in m.execute(
        "SELECT DISTINCT player_id FROM price_snapshots WHERE source='futgg' "
        "AND timestamp >= ?", ((latest - timedelta(days=3)).isoformat(),))]
    for pid in active:
        ser = series(m, pid, latest - timedelta(days=20), latest)
        f = feats(ser, latest, rating_of.get(pid))
        if f:
            rows.append((clf.predict_proba([f])[0, 1], pid, ser[-1][1]))
    rows.sort(reverse=True)
    print(f"=== the model's own BUY calls right now (top {top}), learned from the pros ===")
    print(f"{'conf':>5}  {'price':>8}  card")
    for prob, pid, price in rows[:top]:
        name, url = m.execute("SELECT name, url FROM card_meta WHERE player_id=?",
                              (pid,)).fetchone()
        print(f"{prob*100:>4.0f}%  {price:>8,}  {name}  {url or ''}")
    return clf, rows


def feats2(ser, asof, rating, ev_days):
    """Richer market-state features, incl. distance to the nearest promo event."""
    base = feats(ser, asof, rating)
    if base is None:
        return None
    hist = [(t, p) for t, p in ser if t <= asof and p > 0]
    price = hist[-1][1]
    lo30 = min((p for t, p in hist if t >= asof - timedelta(days=30)), default=price)
    dist_lo = price / lo30 - 1 if lo30 > 0 else 0.0            # >0 = above its floor
    r14 = price / next((p for t, p in reversed(hist)
                        if t <= asof - timedelta(days=14)), price) - 1
    fut = [ (e - asof.date()).days for e in ev_days if e >= asof.date() ]
    past = [ (asof.date() - e).days for e in ev_days if e <= asof.date() ]
    to_next = min(fut) if fut else 30
    since_last = min(past) if past else 30
    return base + [dist_lo, r14, to_next, since_last]


def realized_net(m, pid, when, call, hold=3):
    """Honest round-trip: fill on the buy, sell `hold` days later, all fees in."""
    ser = series(m, pid, when - timedelta(days=2), when + timedelta(days=hold + 2))
    entry, ft = _entry(ser, when, call)
    if not entry:
        return None
    ex = _near(ser, ft + timedelta(days=hold))
    if not ex:
        return None
    return (ex * SELL_TAX) / (entry * BUY_PREMIUM) - 1


def train_profit(top=20, hold=3, min_price=5000, max_price=80000):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = sqlite3.connect(ROOT / "data" / "market.db")
    d = sqlite3.connect(ROOT / "data" / "discord.db")
    idx = build_index(m)
    rating_of = dict(m.execute("SELECT player_id, rating FROM card_meta"))
    ev_days = sorted({_to_utc(f"{r[0]}T00:00:00Z").date()
                      for r in m.execute("SELECT start_date FROM market_events")})
    print(f"(fillable mid-tier only: {min_price:,}-{max_price:,} coins)")

    X, yv, dates = [], [], []
    for row in d.execute("SELECT card, version, price, price_kind, timestamp "
                         "FROM discord_calls WHERE action='buy'"):
        call = dict(zip(["card", "version", "price", "price_kind", "timestamp"], row))
        pid = resolve(m, idx, call)
        if not pid:
            continue
        when = _to_utc(call["timestamp"])
        ser = series(m, pid, when - timedelta(days=20), when + timedelta(days=hold + 2))
        px = _near([r for r in ser if r[0] <= when], when)
        if not px or not (min_price <= px <= max_price):   # fillable tier only
            continue
        f = feats2(ser, when, rating_of.get(pid), ev_days)
        r = realized_net(m, pid, when, call, hold)
        if f and r is not None:
            X.append(f); yv.append(r); dates.append(when)
    X, yv = np.array(X), np.array(yv)
    order = np.argsort([dt.timestamp() for dt in dates])
    X, yv = X[order], yv[order]
    cut = int(len(yv) * 0.75)
    print(f"trader buy setups with a known outcome: {len(yv)} | "
          f"baseline: on average these returned {yv.mean()*100:+.1f}% net over {hold}d")

    reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                        max_depth=4, random_state=0)
    reg.fit(X[:cut], yv[:cut])
    pred = reg.predict(X[cut:])
    act = yv[cut:]
    # THE honest test: take the calls the model liked most on held-out data,
    # and see what they ACTUALLY returned.
    for frac, label in [(0.10, "top 10%"), (0.25, "top 25%")]:
        k = max(1, int(len(pred) * frac))
        picked = act[np.argsort(pred)[::-1][:k]]
        print(f"  held-out {label} the model picked -> actually returned "
              f"{picked.mean()*100:+.1f}% net (vs {act.mean()*100:+.1f}% for all)")

    reg.fit(X, yv)
    latest = _to_utc(m.execute("SELECT MAX(timestamp) FROM price_snapshots").fetchone()[0])
    active = [r[0] for r in m.execute(
        "SELECT DISTINCT player_id FROM price_snapshots WHERE source='futgg' "
        "AND timestamp >= ?", ((latest - timedelta(days=3)).isoformat(),))]
    picks = []
    for pid in active:
        ser = series(m, pid, latest - timedelta(days=20), latest)
        price = _near([r for r in ser if r[0] >= latest - timedelta(days=1)], latest)
        if not price or not (min_price <= price <= max_price):   # fillable tier only
            continue
        f = feats2(ser, latest, rating_of.get(pid), ev_days)
        if f:
            picks.append((reg.predict([f])[0], pid, price))
    picks.sort(reverse=True)
    print(f"\n=== model's most PROFITABLE-looking buys right now (predicted {hold}d net) ===")
    print(f"{'pred':>6}  {'price':>8}  card")
    for pr, pid, price in picks[:top]:
        name, url = m.execute("SELECT name, url FROM card_meta WHERE player_id=?",
                              (pid,)).fetchone()
        print(f"{pr*100:>+5.1f}%  {price or 0:>8,}  {name}  {url or ''}")


def realized_weekly(m, pid, when, call):
    """Buy on the dip, SELL INTO THE NEXT WEDNESDAY PEAK (<=7d), all fees in."""
    ser = series(m, pid, when - timedelta(days=2), when + timedelta(days=9))
    entry, ft = _entry(ser, when, call)
    if not entry:
        return None, None
    # next Wednesday (weekday 2) on/after the fill, within 7 days
    exit_dt = next((ft + timedelta(days=k) for k in range(1, 8)
                    if (ft + timedelta(days=k)).weekday() == 2), ft + timedelta(days=5))
    ex = _near(ser, exit_dt)
    if not ex:
        return None, None
    return (ex * SELL_TAX) / (entry * BUY_PREMIUM) - 1, entry


def _tier(p):
    return ("cheap  (<5k)" if p < 5000 else "mid    (5-40k)" if p < 40000
            else "premium(40-150k)" if p < 150000 else "elite  (150k+)")


def advise(per_tier=8):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = sqlite3.connect(ROOT / "data" / "market.db")
    d = sqlite3.connect(ROOT / "data" / "discord.db")
    idx = build_index(m)
    rating_of = dict(m.execute("SELECT player_id, rating FROM card_meta"))
    liq = {r[0]: r[1] for r in m.execute(
        "SELECT player_id, updates_per_day FROM liquidity WHERE title='fc26'")}
    ev_days = sorted({_to_utc(f"{r[0]}T00:00:00Z").date()
                      for r in m.execute("SELECT start_date FROM market_events")})

    X, yv, ent, dates = [], [], [], []
    for row in d.execute("SELECT card, version, price, price_kind, timestamp "
                         "FROM discord_calls WHERE action='buy'"):
        call = dict(zip(["card", "version", "price", "price_kind", "timestamp"], row))
        pid = resolve(m, idx, call)
        if not pid:
            continue
        when = _to_utc(call["timestamp"])
        f = feats2(series(m, pid, when - timedelta(days=20), when), when,
                   rating_of.get(pid), ev_days)
        r, entry = realized_weekly(m, pid, when, call)
        if f and r is not None:
            X.append(f); yv.append(r); ent.append(entry); dates.append(when)
    X, yv, ent = np.array(X), np.clip(yv, -0.5, 0.5), np.array(ent)  # winsorize noise
    order = np.argsort([dt.timestamp() for dt in dates])
    X, yv, ent = X[order], yv[order], ent[order]
    cut = int(len(yv) * 0.75)
    reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                        max_depth=4, random_state=0)
    reg.fit(X[:cut], yv[:cut])
    pred, act, pent = reg.predict(X[cut:]), yv[cut:], ent[cut:]

    print(f"trained on {len(yv)} trader buys (all tiers), sell-into-Wednesday exit\n")
    print("=== honest held-out: what the model's TOP picks actually returned, by tier ===")
    print(f"{'tier':18} {'setups':>7} {'avg':>7} {'model top 20%':>14}")
    for name, lo, hi in [("cheap  (<5k)", 0, 5000), ("mid    (5-40k)", 5000, 40000),
                         ("premium(40-150k)", 40000, 150000), ("elite  (150k+)", 150000, 9e9)]:
        mask = (pent >= lo) & (pent < hi)
        if mask.sum() < 8:
            print(f"{name:18} {int(mask.sum()):>7}  (too few to judge)")
            continue
        a, p = act[mask], pred[mask]
        k = max(1, int(len(p) * 0.20))
        top = a[np.argsort(p)[::-1][:k]]
        print(f"{name:18} {int(mask.sum()):>7} {a.mean()*100:>+6.1f}% {top.mean()*100:>+13.1f}%")

    reg.fit(X, yv)
    latest = _to_utc(m.execute("SELECT MAX(timestamp) FROM price_snapshots").fetchone()[0])
    active = [r[0] for r in m.execute(
        "SELECT DISTINCT player_id FROM price_snapshots WHERE source='futgg' "
        "AND timestamp >= ?", ((latest - timedelta(days=3)).isoformat(),))]
    tiers: dict[str, list] = {}
    for pid in active:
        if (liq.get(pid) or 0) < 3:                 # skip illiquid/unfillable cards
            continue
        ser = series(m, pid, latest - timedelta(days=20), latest)
        price = _near([r for r in ser if r[0] >= latest - timedelta(days=1)], latest)
        if not price:
            continue
        f = feats2(ser, latest, rating_of.get(pid), ev_days)
        if f:
            pr = reg.predict([f])[0]
            tiers.setdefault(_tier(price), []).append((pr, pr * price, pid, price))

    print("\n=== tips right now, every tier (buy the dip, sell into Wednesday) ===")
    for name in ["cheap  (<5k)", "mid    (5-40k)", "premium(40-150k)", "elite  (150k+)"]:
        rows = sorted(tiers.get(name, []), reverse=True)[:per_tier]
        print(f"\n-- {name} --   pred%  coin_profit   price   upd/day  card")
        for pr, coins, pid, price in rows:
            nm = m.execute("SELECT name FROM card_meta WHERE player_id=?", (pid,)).fetchone()[0]
            print(f"   {pr*100:>+5.1f}%  {coins:>10,.0f}  {price:>8,}  {liq.get(pid,0):>6.0f}   {nm}")


if __name__ == "__main__":
    advise()
