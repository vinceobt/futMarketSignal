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

from futmarket.ml.discord_scorecard import build_index, resolve, series, _to_utc, ROOT

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


if __name__ == "__main__":
    train_and_pick()
