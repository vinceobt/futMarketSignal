"""The output: which cards to buy right now, at what price, how long to hold, and why.

Everything else in this package is machinery. This is the part a person uses.

The trade, from measured edge: buy a cheap/mid card on a **deep** dip — well
below its own normal price and right on its 30-day floor — and give it long
enough to come back. Measured across the season on tradeable cards only, month by
month (``ml.evaluate``):

    gate z<=-0.5, floor<=5%, held  5 days   median net **-2.85%**   (what shipped)
    gate z<=-1.5, floor<=3%, held 14 days   median net **+4.14%**, alpha +20pp
    gate z<=-2.0, floor<=2%, held 14 days   median net **+9.53%**, alpha +18pp

Three things changed from the version that lost money, and each one was worth
more than any amount of model tuning:

**Hold longer.** The round trip costs 7.4% before you pay a coin over the
listing. A five-day dip bounce is ~4.7% gross. The trade could not have worked at
that horizon however good the prediction was. The horizon is now chosen per card
from what the model expects, not fixed.

**Buy a real dislocation.** ``z <= -0.5`` is barely a dip. Demanding a genuine
one cuts the number of trades by two thirds and turns the median positive.

**Stop out of the noise, not into it.** The old stop sat 5% below a *marked-up*
entry and was compared against the raw price series, so it landed on average
0.76% ABOVE the live price -- 90% of trades were stopped before they began. Stops
are now levels on the same series that grades them, and never inside the card's
own daily jitter.

The buy price is still a *band* taken from real completed sales, not the lowest
listing. The lowest listing is usually a mispriced snipe nobody can catch; the
band is what the card genuinely changes hands for.
"""

from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import db
from ..services import sales as sales_service
from . import dataset, evaluate

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 20

# The trades this system knows how to make. Each names a gate in ``ml.evaluate``
# -- deliberately the *same* definition the backtest measures, so what we trade
# and what we claim to have measured can never drift apart.
#
# They are near-disjoint (2.3% overlap) and pay differently, so each carries its
# own strategy tag and the scorecard judges them separately. Measured month by
# month on tradeable cards, net of all costs:
#
#   relval_v1  deep dip, ~14d   median +4.1%, alpha +20pp, 10/11 months
#   release_v1 promo crash, ~21d  median +11.5%, alpha +25pp, 10/10 months
#
# The release trade is the stronger of the two and its liquidity profile is the
# right way round: tier A returns +35.6% at a 79% win rate, versus +10.4% on
# tier B. When the *most* tradeable cards do best, the effect is real -- the
# reverse pattern is what a measurement artifact looks like.
STRATEGIES: dict[str, dict] = {
    "relval_v1": {
        "gate": "relval_v1",
        "summary": "a deep dip, bought on the floor",
        "horizons": (7, 10, 14),
    },
    "release_v1": {
        "gate": "release",
        "summary": "a promo card four to six days into its release crash",
        "horizons": (14, 21),
        # This trade deliberately buys a card that is still making new lows, so
        # the dip strategy's "never below the 30-day floor" rule (trap #7) must
        # not apply: for a card six days old, every day is a new low. The
        # protection here is the time stop and the card's age, not the floor.
        "allow_below_floor": True,
    },
    "weekend_v1": {
        "gate": "weekend_v1",
        "summary": "a card that breathes hard on the weekly cycle, bought in the weekend dump",
        # The sell day, expressed as a holding period. Measured on tier A at a
        # swing floor of 30, the curve peaks sharply and then dies:
        #
        #     hold 2d  -2.9% net    hold 5d  -0.1% net
        #     hold 3d  +2.4% net    hold 6d  -6.9% net
        #     hold 4d  +4.2% net    hold 7d -14.1% net
        #
        # From a Saturday or Sunday buy that is a **Tuesday-to-Thursday** exit,
        # and holding into the next weekend gives all of it back -- by day 7 the
        # card is simply back in the next dump. Which of the three a given card
        # gets is the model's call, not a constant.
        "horizons": (3, 4, 5),
        # Tier A only -- see evaluate.GATE_TIERS, which is where that restriction
        # lives so the backtest, the payoff profile and this shortlist cannot
        # disagree about who is being traded. It is the whole difference between
        # a trade and a mirage: at this gate tier B nets -4.7% while tier A nets
        # +4.2%, and blended the tier-B mass drags the headline under water. A
        # thin card's "price" is one stale listing, so its weekly "swing" is
        # partly the collector's sampling rather than the market's.
        # By construction this trade buys a card at a weekly low, so the 30-day
        # floor guard (trap #7) would reject exactly what it exists to buy. Left
        # off so the live gate stays identical to the one measured -- the
        # backtest above does not apply it either, and a live filter the
        # measurement never saw is precisely how the two drift apart.
        "allow_below_floor": True,
        # Rank horizons by TOTAL expected net, not by return per day held. Per-day
        # is right when coins freed early can go into another trade, which is why
        # it is the default -- but this trade only exists on a Saturday or Sunday.
        # Exiting Tuesday instead of Thursday frees coins that have nothing to do
        # until the next weekend, so charging the trade for those two days would
        # pick the worse exit for a benefit that does not exist.
        "rank_by": "total",
    },
}
DEFAULT_STRATEGY = "release_v1"
# Kept for callers and tests that just want "the current default".
STRATEGY_VERSION = DEFAULT_STRATEGY
MIN_PRICE = 1_000            # ignore near-discard noise
MAX_PRICE = 40_000           # ignore efficiently-priced cards (icons) that lose after tax
# The entry gate. Measured: this is where the edge is, not in barrier tuning.
# Derived from the gate, never restated. These are only used to phrase reasons
# and the consult's verdict; the gate in ``ml.evaluate`` is the single definition
# of what we trade, and a second copy of the numbers here would be free to drift
# away from the thing the backtest actually measured.
ENTRY_Z_MAX = evaluate.GATES["relval_v1"]["z_score"][1]
ENTRY_FLOOR_MAX_PCT = evaluate.GATES["relval_v1"]["dist_to_floor_pct"][1]
RELEASE_AGE_MIN, RELEASE_AGE_MAX = evaluate.GATES["release"]["days_since_card_release"]
# Stops. The old 5% floor was inside the noise -- the median card-day ranges
# 14.3%, and even on the smoothed daily series the median move is 6.6%. A stop
# that tight is a coin-flip on jitter, and it truncates exactly the recovery the
# trade exists to capture.
STOP_BUFFER_PCT = 3.0
STOP_MIN_PCT = 15.0
STOP_MAX_PCT = 30.0
# Reward:risk compares the upside against the *worst case*, and the stop is now
# deliberately wide, so demanding reward >= max-loss would need a 30% ceiling and
# reject nearly every card. It stays only to throw out genuinely lopsided setups;
# whether a trade is worth taking is decided by expected value below, which is
# the honest test. (The old engine used this as its main gate, at 1.0.)
MIN_REWARD_RISK = 0.5
FALLBACK_TARGET_PCT = 25.0   # when a card's resistance isn't known yet
# Above this, a card's 30-day high is history rather than a target worth naming
# in the case for buying it. The best measured trade in the system pays ~35%.
MAX_QUOTABLE_RESISTANCE_PCT = 100.0
# What the model must expect before a card is worth the coins and the wait.
# Silence is a valid answer; the old version had no way to give one.
MIN_EXPECTED_NET_PCT = 3.0
MIN_CLEARS_PROB = evaluate.MIN_CLEARS_PROB


@dataclass
class Pick:
    player_id: str
    name: str
    rating: int | None
    version: str
    confidence: float
    price_now: int
    buy_low: int | None
    buy_high: int | None
    sell_target: int
    stop: int
    reward_risk: float
    liquidity_tier: str | None
    sales_per_hour: float | None
    hold_days: int = 14              # how long the model wants to give this trade
    expected_net_pct: float = 0.0    # what it expects to clear, after all costs
    expected_alpha_pct: float = 0.0  # ...over a random card from the same shortlist
    strategy: str = DEFAULT_STRATEGY  # which trade this is, so the record stays honest
    url: str | None = None           # exact card -- names repeat across versions
    reasons: list[str] = field(default_factory=list)


def _barriers(market_price: int, entry: int, ceil_pct, floor_pct, *,
              tax_rate: float, sell_slippage_pct: float = 0.0,
              stop_buffer_pct: float = STOP_BUFFER_PCT,
              stop_min_pct: float = STOP_MIN_PCT, stop_max_pct: float = STOP_MAX_PCT,
              payoff_target_pct: float | None = None) -> tuple[int, int, float]:
    """(sell_target, stop, reward:risk) as levels on the series that grades them.

    **This is the bug that broke the previous strategy.** Barriers used to be
    derived from ``entry`` -- the marked-up price you'd actually pay, ~5% over
    the market -- while the scorer compared them against the raw price series. A
    "5% stop" therefore sat 0.76% *above* the live price on average, and 53 of 59
    graded trades stopped out, most within hours. Nothing about the model was
    ever tested.

    So the levels are anchored to ``market_price``: the same number the scorecard
    reads. ``entry`` still decides the *return* -- you did pay the premium -- but
    it must never decide *where the exits sit*. Reward and risk are both net of
    tax and sell slippage, so the ratio is the real one.

    ``payoff_target_pct`` caps the target at what this gate has actually paid on
    a winning trade. Without it the target is the card's 30-day high, and on a
    card that has just crashed that is nonsense: a real pick quoted a sell 141%
    up with a reward:risk of 5.1, which is not a fortnight's trade, it is the
    price the card used to be. Resistance still caps from the other side -- there
    is no sense targeting *through* a level the card keeps failing at.
    """
    resistance_pct = (float(ceil_pct) if pd.notna(ceil_pct) and ceil_pct > 0
                      else FALLBACK_TARGET_PCT)
    target_pct = (min(resistance_pct, payoff_target_pct)
                  if payoff_target_pct is not None else resistance_pct)
    stop_dist = (float(floor_pct) + stop_buffer_pct) if pd.notna(floor_pct) else stop_min_pct
    stop_dist = min(max(stop_dist, stop_min_pct), stop_max_pct)
    target = int(round(market_price * (1 + target_pct / 100.0)))
    stop = int(round(market_price * (1 - stop_dist / 100.0)))

    sale_net = (1 - tax_rate) * (1 - sell_slippage_pct / 100.0)
    net_reward = target * sale_net / entry - 1.0
    net_risk = 1.0 - stop * sale_net / entry
    rr = net_reward / net_risk if net_risk > 0 else 0.0
    return target, stop, round(rr, 2)


def _choose_horizon(clears: dict[int, float], payoffs: dict[int, dict], *,
                    min_expected_net_pct: float = MIN_EXPECTED_NET_PCT,
                    min_clears_prob: float = MIN_CLEARS_PROB,
                    rank_by: str = "per_day"
                    ) -> tuple[int, float, float, float] | None:
    """How long to hold this card — or None if no horizon is worth trading.

    Expected value, assembled from two things we have evidence for:

    * ``clears[h]`` — the model's probability that a position opened here beats
      the market and covers its costs. Measured on the traded slice, this is
      **52-59% against a 35-42% base rate**: real, repeatable skill.
    * ``payoffs[h]`` — how big the win and the loss have actually been for this
      gate at this horizon, from the season's history.

    Deliberately *not* used: the regression head's predicted magnitude. It scores
    worse than assuming a card simply moves with the market (-12% to -19% skill),
    so building expected value on it would be building on nothing. The model is
    asked only the question it can answer.

    Among horizons clearing the bar we take the best return **per day held** —
    coins tied up in one card are coins not working in another.

    ``rank_by="total"`` drops that division, for a trade whose entry only exists
    on certain days. The weekly-cycle trade can only be opened on a Saturday or
    Sunday, so coming out on Tuesday rather than Thursday frees coins that have
    nothing to do until the next weekend. Charging those two days against the
    trade would pick the worse exit to buy an option that cannot be exercised.

    Returning None is a real answer and the old code could never give it: if
    nothing clears the 7.4% round trip, the honest recommendation is to buy
    nothing.
    """
    best = None
    for h, payoff in payoffs.items():
        probability = clears.get(h)
        if probability is None or probability < min_clears_prob:
            continue
        expected = (probability * payoff["win_net"]
                    + (1.0 - probability) * payoff["loss_net"])
        if expected < min_expected_net_pct:
            continue
        score = expected if rank_by == "total" else expected / h
        if best is None or score > best[0]:
            best = (score, h, expected, probability, payoff)
    if best is None:
        return None
    _, h, expected, probability, payoff = best
    # The edge over buying a random card from the same gate: what the model's
    # selection adds, in the same units as the expectation itself.
    baseline = (payoff["base_rate"] * payoff["win_net"]
                + (1.0 - payoff["base_rate"]) * payoff["loss_net"])
    return h, expected, expected - baseline, probability


# --- turning numbers into a human case ------------------------------------

def _reasons(row: pd.Series) -> list[str]:
    """The specific, checkable reasons this card looks interesting today."""
    out: list[str] = []

    # The core of the trade: a real dislocation, sitting on its own floor.
    z = row.get("z_score")
    if pd.notna(z) and z <= ENTRY_Z_MAX:
        out.append(f"deep on the dip — {abs(z):.1f} sigma below its own normal price")
    elif pd.notna(z) and z <= -0.5:
        out.append(f"on a shallow dip — only {abs(z):.1f} sigma below normal")
    floor = row.get("dist_to_floor_pct")
    if pd.notna(floor):
        if floor < 0:
            out.append(f"WARNING: {abs(floor):.0f}% BELOW its 30-day low — still falling")
        elif floor <= ENTRY_FLOOR_MAX_PCT:
            out.append(f"right at its floor ({floor:.0f}% above the 30-day low)")
    ceil = row.get("dist_to_ceiling_pct")
    # Only when the level is a plausible destination for this trade. On a card
    # that has crashed, its 30-day high is the price it *used to be* -- quoting
    # "resistance is 1830% up" as a reason to buy is how a real pick came to
    # advertise a sell 141% away (see _barriers). The target itself is already
    # capped at the gate's measured winning payoff; the reason must not promise
    # more than the target quotes.
    if pd.notna(ceil) and 8 <= ceil <= MAX_QUOTABLE_RESISTANCE_PCT:
        out.append(f"room to bounce — resistance is {ceil:.0f}% up")
    # How much of the recent move was the card's own rather than the whole market.
    own = row.get("excess_ret_7d")
    if pd.notna(own) and own <= -5:
        out.append(f"down {abs(own):.0f}% more than the market this week — "
                   f"that part is its own, and it's the part that comes back")

    dow = row.get("day_of_week")
    if pd.notna(dow):
        if dow in (6, 0):
            out.append("Sunday/Monday trough — supply dries up from here")
        elif dow == 3:
            out.append("Thursday rewards dump — supply flooding in")
        elif dow == 4:
            out.append("Friday promo + Weekend League — heavy supply")

    # How hard this specific card breathes on the weekly cycle. The whole weekend
    # trade turns on this being large: on an inert card the same weekend buy is a
    # 4.6% move against a 7.5% round trip, which is a slow way to lose money.
    swing = row.get("weekend_swing_med")
    if pd.notna(swing) and swing >= 20:
        weeks = row.get("weekend_swing_n")
        seen = f" over {int(weeks)} weeks" if pd.notna(weeks) else ""
        out.append(f"swings {swing:.0f}% every week{seen} — this one actually "
                   f"moves enough to pay for the tax")
    bounce = row.get("dist_to_week_trough_pct")
    if pd.notna(bounce) and pd.notna(swing) and swing >= 20:
        if bounce <= 2:
            out.append("still at this week's low — the dump hasn't turned yet")
        elif bounce >= 10:
            out.append(f"already {bounce:.0f}% off this week's low — "
                       f"you're buying after part of the move")

    age = row.get("days_since_card_release")
    if pd.notna(age):
        # Measured, and not where the folklore says: day 4-6 held three weeks
        # returns +11.5% net at +25pp of alpha, better than day 7-9 at every
        # horizon, while buying the debut loses money.
        if RELEASE_AGE_MIN <= age <= RELEASE_AGE_MAX:
            out.append(f"day {int(age)} of its release crash — the measured "
                       f"bottom, and the best window of the whole curve")
        elif 7 <= age <= 12:
            out.append(f"day {int(age)} after release — past the best window, "
                       f"but still recovering")
        elif age <= 2:
            out.append(f"only {int(age)}d old — release crash still running")

    ann = row.get("days_since_last_announce")
    if pd.notna(ann) and ann <= 1:
        out.append(f"EA announced a promo {int(ann)}d ago — the usual dip")
    elif pd.notna(ann) and 4 <= ann <= 7:
        out.append(f"{int(ann)}d after an EA announcement — the usual rebound")

    promo = row.get("days_to_next_promo")
    if pd.notna(promo) and promo <= 3:
        out.append(f"promo in {int(promo)}d — expect supply")
    # (live-SBC count is omitted: 50-60 are always running, so it appears on every
    #  card and discriminates nothing.)

    coh = row.get("cohort_ret_7d")
    rel = row.get("rel_strength_7d")
    if pd.notna(coh) and 3 <= abs(coh) <= 60:
        direction = "rising" if coh > 0 else "falling"
        out.append(f"its group is {direction} ({coh:+.0f}% this week)")
    # Beyond ~50% the relative figure is arithmetic noise off a tiny price, not
    # a real dislocation, so it isn't worth quoting as a reason.
    if pd.notna(rel) and -50 <= rel <= -5:
        out.append(f"lagging its group by {abs(rel):.0f}% — room to catch up")

    return out or ["model pattern match (no single standout reason)"]


def _fill_missing_bands(conn, player_ids: list[str], *, title: str = "fc26",
                        delay: float = 1.0, max_age_minutes: float = 60.0) -> int:
    """Fetch real sale data for shortlisted cards that lack it OR have gone stale.

    A pick without a band can only quote the lowest listing -- the very number we
    decided not to trust. A pick with a *stale* band is worse: it quotes a
    confident price that has since moved. Prices here can run 30%+ in a few
    hours, so anything older than an hour is refetched. The shortlist is short,
    so doing it on demand is cheap.
    """
    import time
    from datetime import datetime, timezone
    from ..collectors import history_source
    from ..collectors.base import SourceError
    from ..services import sales as sales_service

    now = datetime.now(timezone.utc)
    fetched = 0
    for pid in player_ids:
        existing = db.sale_stats_get(conn, pid)
        if existing is not None:
            try:
                age_min = (now - datetime.strptime(
                    existing["computed_at"], "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=timezone.utc)).total_seconds() / 60.0
            except (TypeError, ValueError):
                age_min = float("inf")
            if age_min <= max_age_minutes:
                continue
        meta = db.card_meta_get(conn, pid)
        if meta is None or meta["definition_id"] is None:
            continue
        game = (meta["title"] or "fc26").replace("fc", "") or "26"
        try:
            detail = history_source.fetch_card_detail(int(meta["definition_id"]), game)
        except SourceError as e:
            logger.warning("could not fetch sales for %s: %s", pid, e)
            continue
        stats = sales_service.summarise_sales(detail["sales"], detail.get("current"))
        if stats:
            db.upsert_sale_stats(conn, player_id=pid, title=title, **stats)
            conn.commit()
            fetched += 1
        if delay:
            time.sleep(delay)
    return fetched


def _load_latest_model(conn, *, kind: str, title: str = "fc26"):
    import joblib
    run = db.latest_model_run(conn, kind=kind, title=title)
    if run is None or not run["artifact_path"]:
        return None, None
    try:
        artifact = joblib.load(run["artifact_path"])
    except OSError as e:
        logger.warning("could not load model artifact: %s", e)
        return None, None
    return artifact, run


def _score_horizons(latest: pd.DataFrame, excess_art: dict, clears_art: dict,
                    horizons) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Clears-cost probability at every horizon, and the excess estimate.

    Only ``clears`` drives decisions. The excess numbers are carried for display
    and for tracking whether that head ever starts working — today it scores
    worse than assuming a card moves with the market.
    """
    def _cols(artifact, h):
        """Exactly the columns this model was fitted on — a feature that was
        constant when it trained was dropped, and passing it back would not match
        the shape the estimator expects."""
        return (artifact.get("feature_sets") or {}).get(h) or artifact["features"]

    excess, clears = {}, {}
    for h in horizons:
        model = excess_art["models"].get(h) if excess_art else None
        if model is not None:
            excess[h] = model.predict(latest[_cols(excess_art, h)])
        prob_model = clears_art["models"].get(h) if clears_art else None
        if prob_model is not None:
            clears[h] = prob_model.predict_proba(latest[_cols(clears_art, h)])[:, 1]
    return excess, clears


def generate(conn, *, source: str = "futgg", title: str = "fc26",
             strategy: str = DEFAULT_STRATEGY,
             tax_rate: float = 0.05, sell_slippage_pct: float = 2.0,
             buy_premium_pct: float = 0.0,
             limit: int = DEFAULT_LIMIT, min_price: int = MIN_PRICE,
             max_price: int = MAX_PRICE,
             stop_buffer_pct: float = STOP_BUFFER_PCT, stop_min_pct: float = STOP_MIN_PCT,
             stop_max_pct: float = STOP_MAX_PCT, min_reward_risk: float = MIN_REWARD_RISK,
             min_expected_net_pct: float = MIN_EXPECTED_NET_PCT,
             min_clears_prob: float = MIN_CLEARS_PROB,
             fetch_missing_bands: bool = True, min_sales_per_hour: float = 0.0,
             liquid_tiers: tuple[str, ...] = ("A", "B"),
             skip_held: bool = True) -> list[Pick]:
    """Today's buy candidates for one strategy, worth more than they cost to trade."""
    spec = STRATEGIES.get(strategy)
    if spec is None:
        raise ValueError(f"unknown strategy {strategy!r}; "
                         f"expected one of {sorted(STRATEGIES)}")
    # A trade whose gate names a buy day does not exist on the other days, and
    # saying so beats returning an empty list that reads like a broken pipeline
    # (trap #17). Checked before the feature matrix is built, which is minutes.
    day_bounds = evaluate.GATES.get(spec["gate"], {}).get("day_of_week")
    if day_bounds is not None:
        today = datetime.now(timezone.utc).weekday()
        lo, hi = day_bounds
        if (lo is not None and today < lo) or (hi is not None and today > hi):
            days = "/".join(calendar.day_name[int(d)]
                            for d in range(int(lo), int(hi) + 1))
            logger.info("%s only trades on %s — nothing to do today", strategy, days)
            return []

    clears_art, run = _load_latest_model(conn, kind="clears", title=title)
    excess_art, _ = _load_latest_model(conn, kind="excess", title=title)
    if clears_art is None:
        raise RuntimeError("no trained model — run `futmarket train` first")
    payoffs = (clears_art.get("payoffs") or {}).get(spec["gate"])
    if not payoffs:
        raise RuntimeError(
            f"no measured payoffs for gate {spec['gate']!r} — run `futmarket train`")

    frame = dataset.build_dataset(conn, source=source, title=title)
    if frame.empty:
        return []

    # latest row per card only: we're deciding about today, not replaying history
    latest = (frame.sort_values("date").groupby("player_id", observed=True)
              .tail(1).copy())
    horizons = tuple(h for h in spec["horizons"] if h in payoffs)
    if not horizons:
        return []

    if "liq_tier" in latest.columns:
        # A strategy may narrow the universe further than the caller asked. The
        # weekly-cycle trade is tier A only: its edge inverts on tier B, where a
        # card's "swing" is partly the collector sampling a stale listing.
        latest = latest[latest["liq_tier"].isin(
            evaluate.gate_tiers(spec["gate"], liquid_tiers))]
    # The tradeable universe: the edge lives in cheap/mid cards; expensive icons
    # are priced efficiently and lose after tax.
    latest = latest[(latest["price"] >= min_price) & (latest["price"] <= max_price)]
    # The entry gate — this IS the edge, and it is the *same* definition
    # `ml.evaluate` measured, so what we trade and what we claim can't drift.
    latest = latest[evaluate.gate_mask(latest, spec["gate"])]
    if not spec.get("allow_below_floor") and "dist_to_floor_pct" in latest.columns:
        # Below its own 30-day low is a falling knife, not a discount (trap #7).
        # The release trade is the deliberate exception: a six-day-old card makes
        # a new low every day, and its protection is the time stop instead.
        latest = latest[latest["dist_to_floor_pct"] >= 0]
    if skip_held:
        # One position per card. The loop re-derives this shortlist every two
        # hours; without this a single opportunity became a dozen "trades".
        held = [db.has_open_pick(conn, pid, title=title)
                for pid in latest["player_id"].astype(str)]
        latest = latest[~pd.Series(held, index=latest.index)]
    if latest.empty:
        return []

    # Own the frame outright before adding columns: the filters above leave a
    # view, and assigning into one is a SettingWithCopy trap.
    latest = latest.copy()
    for artifact in (excess_art, clears_art):
        if artifact:
            for c in [c for c in artifact["features"] if c not in latest.columns]:
                latest[c] = np.nan
    excess, clears = _score_horizons(latest, excess_art, clears_art, horizons)
    if not clears:
        return []

    # Decide, per card, how long to hold and whether it is worth holding at all.
    plans, keep_index = [], []
    for position, index in enumerate(latest.index):
        plan = _choose_horizon(
            {h: float(v[position]) for h, v in clears.items()}, payoffs,
            min_expected_net_pct=min_expected_net_pct,
            min_clears_prob=min_clears_prob,
            rank_by=spec.get("rank_by", "per_day"))
        if plan is not None:
            plans.append(plan)
            keep_index.append(index)
    if not keep_index:
        logger.info("no card clears the %.1f%% round trip today — buy nothing",
                    evaluate.round_trip_cost_pct(
                        tax_rate=tax_rate, sell_slippage_pct=sell_slippage_pct,
                        buy_premium_pct=buy_premium_pct))
        return []

    candidates = latest.loc[keep_index].copy()
    candidates["hold_days"] = [p[0] for p in plans]
    candidates["expected_net_pct"] = [p[1] for p in plans]
    # What the model's selection adds over buying a random card from the same
    # gate -- not a prediction of the card's move, which we cannot make.
    candidates["expected_alpha_pct"] = [p[2] for p in plans]
    candidates["confidence"] = [p[3] for p in plans]
    # Rank by what the trade is expected to clear, not by raw model confidence:
    # a 70%-likely 3% gain is worth less than a 50%-likely 20% one.
    candidates = candidates.sort_values("expected_net_pct",
                                        ascending=False).head(limit * 4)

    if fetch_missing_bands:
        # Fetch a wider shortlist than we'll show, so the sales filter below has
        # real rates to judge and isn't just dropping cards for missing data.
        _fill_missing_bands(conn, candidates["player_id"].tolist()[:limit * 3],
                            title=title)

    if min_sales_per_hour > 0:
        # Rule #1 in its truest form: confidence is worthless if you can't get out.
        keep = []
        for pid in candidates["player_id"]:
            stats = db.sale_stats_get(conn, pid)
            keep.append(stats is not None and (stats["sales_per_hour"] or 0)
                        >= min_sales_per_hour)
        candidates = candidates[pd.Series(keep, index=candidates.index)]

    picks: list[Pick] = []
    for row in candidates.itertuples(index=False):
        r = pd.Series(row._asdict())
        price = int(r["price"])
        stats = db.sale_stats_get(conn, r["player_id"])
        meta = db.card_meta_get(conn, r["player_id"])
        # The listed price refreshed moments ago is the only figure that is
        # genuinely current; sales lag it on anything that's moving.
        listed = stats["listed_price"] if stats is not None else None
        band = sales_service.buy_band(
            {"listed_price": listed, "sold_median": stats["sold_median"]}
            if stats is not None else None,
            listed_price=listed or price)
        # Two different prices, and conflating them is what broke the last
        # strategy. `entry` is what you PAY (band top, over the listing).
        # `price` is the market series the scorecard will grade against, so it
        # is what the barriers must be anchored to.
        entry = band[1] if band else price
        # What a winning trade at the chosen horizon has actually paid, converted
        # from a net return back to the gross price you must sell at.
        sale_net = (1 - tax_rate) * (1 - sell_slippage_pct / 100.0)
        win_net = payoffs[int(r["hold_days"])]["win_net"]
        payoff_target_pct = ((1 + win_net / 100.0) / sale_net - 1.0) * 100.0
        target, stop, rr = _barriers(
            price, entry, r.get("dist_to_ceiling_pct"), r.get("dist_to_floor_pct"),
            tax_rate=tax_rate, sell_slippage_pct=sell_slippage_pct,
            stop_buffer_pct=stop_buffer_pct,
            stop_min_pct=stop_min_pct, stop_max_pct=stop_max_pct,
            payoff_target_pct=payoff_target_pct)
        if rr < min_reward_risk:
            continue                     # upside not worth the downside — skip
        hold_days = int(r["hold_days"])
        expected_net = float(r["expected_net_pct"])
        reasons = _reasons(r)
        reasons.append(
            f"model gives it {float(r['confidence']):.0%} to pay off in "
            f"{hold_days} days — worth ~{expected_net:+.0f}% after tax and fees, "
            f"{float(r['expected_alpha_pct']):+.0f}% better than a random card "
            f"off the same shortlist")
        reasons.append(f"risking {round((1 - stop/entry) * 100)}% to make "
                       f"{round((target * (1 - tax_rate) / entry - 1) * 100)}% "
                       f"(reward:risk {rr:g})")
        picks.append(Pick(
            player_id=r["player_id"],
            name=str(r.get("name") or r["player_id"]),
            rating=int(r["rating"]) if pd.notna(r.get("rating")) else None,
            version=str(r.get("version") or ""),
            confidence=float(r["confidence"]),
            price_now=price,
            buy_low=band[0] if band else None,
            buy_high=band[1] if band else None,
            sell_target=target,
            stop=stop,
            reward_risk=rr,
            liquidity_tier=r.get("liq_tier"),
            sales_per_hour=(float(stats["sales_per_hour"])
                            if stats is not None and stats["sales_per_hour"] else None),
            hold_days=hold_days,
            expected_net_pct=expected_net,
            expected_alpha_pct=float(r["expected_alpha_pct"]),
            url=(meta["url"] if meta is not None else None),
            reasons=reasons,
        ))
        if len(picks) >= limit:
            break
    logger.info("generated %d %s picks from run %s", len(picks), strategy,
                run["run_id"] if run is not None else "?")
    for p in picks:
        p.strategy = strategy
    return picks


def generate_all(conn, *, strategies=tuple(STRATEGIES), **kwargs) -> list[Pick]:
    """Every strategy's candidates, best expected return first.

    The trades are near-disjoint -- a deep dip and a promo release crash select
    almost entirely different cards (2.3% overlap) -- so running them together
    widens the opportunity set rather than duplicating it. Where two strategies
    do land on the same card, ``db.has_open_pick`` still allows only one open
    position, so the record cannot double-count it (trap #15).
    """
    found: list[Pick] = []
    for name in strategies:
        try:
            found.extend(generate(conn, strategy=name, **kwargs))
        except RuntimeError as e:
            logger.warning("strategy %s unavailable: %s", name, e)
    return sorted(found, key=lambda p: p.expected_net_pct, reverse=True)


def save(conn, picks: list[Pick], *, title: str = "fc26",
         horizon_days: int | None = None) -> int:
    """Record today's picks so the market can grade them later.

    Stores the price you'd realistically have paid, the market price the barriers
    were anchored to, and how long the model wants to hold — so scoring never
    depends on re-deriving any of it afterwards. A card already held is skipped:
    the loop re-derives this list every two hours and one opportunity is one
    trade, not twelve.
    """
    from datetime import datetime, timezone
    run = db.latest_model_run(conn, kind="excess", title=title)
    run_id = run["run_id"] if run is not None else None
    now = datetime.now(timezone.utc)
    saved = 0
    for p in picks:
        if db.has_open_pick(conn, p.player_id, title=title):
            continue
        entry = p.buy_high or p.price_now
        if db.insert_pick(conn, player_id=p.player_id, title=title, at=now,
                          run_id=run_id, confidence=p.confidence,
                          entry_price=entry, buy_low=p.buy_low,
                          buy_high=p.buy_high, target_price=p.sell_target,
                          stop_price=p.stop,
                          horizon_days=horizon_days or p.hold_days,
                          chosen_horizon_days=p.hold_days,
                          market_price=p.price_now,
                          sales_per_hour=p.sales_per_hour,
                          reasons="; ".join(p.reasons),
                          strategy=p.strategy):
            saved += 1
    conn.commit()
    return saved
