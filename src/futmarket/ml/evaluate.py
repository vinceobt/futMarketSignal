"""The honest scoreboard — the one number that decides whether an idea ships.

Every optimistic figure this project ever produced came from skipping this step.
The +4.4% backtest that justified the last strategy was inflated two ways, and
both are corrected here:

* **Illiquid cards flatter everything.** Measured over the season, tier C cards
  return +17.3% on the dip trade and tier A only +1.6%. That inversion is the
  signature of measurement noise, not edge: the thinner the card, the more its
  "price" is one stale listing that can print any number. So headline results are
  restricted to tradeable cards (tier A/B) — the ones you could actually fill.

* **The market gives you nothing to start with.** The median 14-day *net* return
  of the entire liquid universe is negative in all eleven months on record. Most
  of that is not prices falling — it is the round trip: the median tradeable card
  doesn't move at all over a fortnight, so a flat market reads as exactly
  -6.9% once tax and slippage are paid. (Some months the market really does
  drop: -56% in 2025-09, -19% in 2026-02.) Either way, a strategy returning -3%
  in a month where everything returned -9% is genuinely good, and one returning
  +2% where everything returned +20% is bad. So the headline is **alpha** — the
  gate's median net return minus the same-month, same-tier universe median —
  reported month by month, never pooled.

Everything is a **median**. Means here are driven by a fat right tail: a handful
of cards that tripled make a losing rule look profitable, and you cannot spend a
mean you only touch one trade in fifty.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import validation

logger = logging.getLogger(__name__)

# Holding periods the strategy is allowed to choose between. The dip signal's
# edge grows with holding period -- it does not clear the round-trip cost at 5
# days -- so short horizons are kept only for comparison.
#
# 2, 4 and 6 exist for the weekly-cycle trade, where the holding period *is* the
# sell day: bought on a Sunday, these land on Tuesday, Thursday and Saturday, and
# without them most of the midweek exits could not be graded at all. The sell day
# is meant to be an output of the model rather than an assumption baked into the
# horizon list, which it silently would be if only 3 and 5 were available.
HORIZONS = (2, 3, 4, 5, 6, 7, 10, 14)

# Cards you could actually sell. Rule #1: everything else is a paper gain.
TRADEABLE_TIERS = ("A", "B")

# A month needs at least this many trades before its median means anything.
MIN_MONTH_ROWS = 25

# The clears-cost probability at which a card becomes a pick. Shared so that the
# precision reported at training time is measured at the threshold the strategy
# will actually use -- reporting at 0.5 while trading at 0.45 would grade a
# different model than the one that ships.
MIN_CLEARS_PROB = 0.45

# The entry gates worth comparing. Each is a plain dict of column -> (lo, hi)
# bounds, applied inclusively; None means unbounded on that side.
GATES: dict[str, dict[str, tuple[float | None, float | None]]] = {
    # What the system shipped: barely a dip, and measured to lose after costs.
    "dip_v1": {"z_score": (None, -0.5), "dist_to_floor_pct": (0.0, 5.0)},
    # A real dislocation. Median net +11.4% at 14 days vs -2.5% for dip_v1 at 5.
    "relval_v1": {"z_score": (None, -1.5), "dist_to_floor_pct": (0.0, 3.0)},
    # Deeper still: better per trade, but only ~2k opportunities a season.
    "deep": {"z_score": (None, -2.0), "dist_to_floor_pct": (0.0, 2.0)},
    # A different trade entirely: the promo release curve. A special card crashes
    # after release, and the money is in the slide, not the debut -- buying at
    # age 0-1 loses 7-16% and lags the market by 4-9pp, while age 4-6 held two to
    # three weeks gains 4-9% at +7 to +10pp of alpha. Note this is NOT the
    # "day nine bottom" folklore: day 4-6 measured better than day 7-9 at every
    # horizon. Base golds are excluded -- they have no release event.
    "release": {"is_special": (1.0, 1.0), "days_since_card_release": (4.0, 6.0)},
    # The weekly supply cycle. Rewards flood the market Friday to Sunday and
    # supply dries up midweek, so the trade is to buy into the weekend and sell
    # into the recovery.
    #
    # Note what this gate does NOT say: anything about when to sell. It fixes the
    # buy day and the kind of card, and the holding period is left to the model,
    # because the sell day is the open question. (For the record, the best pair on
    # the median card measured Sunday->Thursday -- a day later than the Wednesday
    # peak in the weekly table -- but that is an average across every card, and
    # cheap fodder need not peak when premium cards do.)
    #
    # The swing floor is what makes it a trade at all, and it has to be high.
    # Buying every card in the weekend is +4.6% gross and **-2.6% net**. Measured
    # across the threshold at a 4-day hold, on tier A:
    #
    #     swing >= 10   blended -1.9% net   tier A -3.9%  <- A worse than B
    #     swing >= 20      +1.6% net (A)   +10.3pp alpha   53% win  n=2,426
    #     swing >= 30      +4.2% net (A)   +16.7pp alpha   59% win  n=1,059
    #     swing >= 40      +4.7% net (A)   +12.3pp alpha   58% win  n=  493
    #
    # 30 is where the trade earns its costs with a sample still worth trusting.
    # Two things move together as the floor rises and both matter: tier A goes
    # from *worse* than tier B to decisively better (the artifact signature
    # inverting into the real one, as on the release trade), and the absolute
    # return crosses zero. Alpha alone would have shipped the 10% version, which
    # loses money in a falling market -- you cannot spend alpha.
    "weekend_v1": {"day_of_week": (5.0, 6.0),          # Saturday and Sunday
                   "weekend_swing_med": (30.0, None)},
}

# Gates that trade a narrower universe than "anything tradeable". Kept beside the
# gates themselves so the backtest, the payoff profile and the live shortlist all
# read one definition -- a payoff measured on A+B while only A is traded would
# price every weekend trade off the losing half of the population.
GATE_TIERS: dict[str, tuple[str, ...]] = {
    "weekend_v1": ("A",),
}


def gate_tiers(gate: str | dict | None,
               default: tuple[str, ...] | None = TRADEABLE_TIERS
               ) -> tuple[str, ...] | None:
    """Which liquidity tiers this gate is traded on."""
    return GATE_TIERS.get(gate, default) if isinstance(gate, str) else default


# --- costs ------------------------------------------------------------------

def net_return_pct(gross_pct, *, tax_rate: float = 0.05,
                   sell_slippage_pct: float = 2.0,
                   buy_premium_pct: float = 0.0):
    """What a gross price move is actually worth once the round trip is paid.

    Three costs, applied where they really land: you buy a little *over* the
    cheapest listing to get filled, EA takes its cut of the sale, and you list a
    little *under* the going rate so the sale happens at all. Together these are
    ~7% before any buy premium, which is why a 4.7% five-day bounce is not a
    trade no matter how confidently a model predicts it.
    """
    gross = np.asarray(gross_pct, dtype="float64")
    entry = 1.0 + buy_premium_pct / 100.0
    proceeds = (1.0 + gross / 100.0) * (1.0 - tax_rate) * (1.0 - sell_slippage_pct / 100.0)
    return (proceeds / entry - 1.0) * 100.0


def round_trip_cost_pct(*, tax_rate: float = 0.05, sell_slippage_pct: float = 2.0,
                        buy_premium_pct: float = 0.0) -> float:
    """The gross move a trade must make just to break even, in percent.

    Solves ``net_return_pct(g) == 0`` for g. With the default 5% tax and 2%
    slippage that is **+7.5%** before you pay a coin over the listing -- the
    single most important number in this project, and the reason a 5-day dip
    trade cannot work.
    """
    entry = 1.0 + buy_premium_pct / 100.0
    proceeds_per_coin = (1.0 - tax_rate) * (1.0 - sell_slippage_pct / 100.0)
    return (entry / proceeds_per_coin - 1.0) * 100.0


# --- forward returns, on the calendar rather than on row position ------------

def _as_datetime(dates: pd.Series) -> pd.Series:
    """`date` is stored as a category of 'YYYY-MM-DD' to keep 2M rows cheap."""
    raw = dates.astype(str) if isinstance(dates.dtype, pd.CategoricalDtype) else dates
    return pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")


def add_forward_returns(frame: pd.DataFrame, *, horizons=HORIZONS,
                        price_col: str = "price") -> pd.DataFrame:
    """Attach ``fwd_return_{h}d`` for each horizon, matched on the **calendar**.

    Deliberately not ``groupby.shift(-h)``: that counts *rows*, so a card whose
    collection gapped for three days would have its "5-day" return measured over
    eight. Joining on an explicit target date makes a missing day produce NaN --
    an honest gap -- instead of a silently mislabelled trade.
    """
    out = frame.copy()
    dt = _as_datetime(out["date"])
    out["_dt"] = dt
    lookup = out[["player_id", "_dt", price_col]].rename(
        columns={"_dt": "_target", price_col: "_future"})
    # player_id is categorical; a merge key must have identical categories on
    # both sides, and plain strings are simpler than reconciling them.
    lookup["player_id"] = lookup["player_id"].astype(str)
    # One price per card-day, or the merge would fan out and misalign the result
    # against the left frame.
    lookup = lookup.drop_duplicates(subset=["player_id", "_target"], keep="last")
    left_ids = out["player_id"].astype(str)

    for h in horizons:
        probe = pd.DataFrame({"player_id": left_ids,
                              "_target": dt + pd.Timedelta(days=h)})
        merged = probe.merge(lookup, on=["player_id", "_target"], how="left")
        future = merged["_future"].to_numpy(dtype="float64")
        base = out[price_col].to_numpy(dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"fwd_return_{h}d"] = np.where(
                base > 0, (future / base - 1.0) * 100.0, np.nan).astype("float32")
    return out.drop(columns=["_dt"])


def add_benchmark_returns(frame: pd.DataFrame, *, horizons=HORIZONS,
                          by: tuple[str, ...] = ("date",)) -> pd.DataFrame:
    """Attach ``bench_return_{h}d``: what the market did over the same window.

    This is the number a pick has to beat. Without it, every call in a market
    that deflates ~7%/week looks like a bad one, and the model has no way to tell
    "this card fell less than everything else" from "this card fell".
    """
    out = frame.copy()
    for h in horizons:
        col = f"fwd_return_{h}d"
        if col not in out.columns:
            continue
        bench = out.groupby(list(by), observed=True)[col].transform("median")
        out[f"bench_return_{h}d"] = bench.astype("float32")
        out[f"excess_return_{h}d"] = (out[col] - bench).astype("float32")
    return out


# --- gates ------------------------------------------------------------------

def gate_mask(frame: pd.DataFrame, gate: str | dict | None) -> pd.Series:
    """Rows a gate would have bought. Unknown columns are ignored, not fatal."""
    mask = pd.Series(True, index=frame.index)
    if gate is None:
        return mask
    spec = GATES[gate] if isinstance(gate, str) else gate
    for column, (lo, hi) in spec.items():
        if column not in frame.columns:
            logger.warning("gate column %s missing — not applied", column)
            continue
        values = frame[column]
        mask &= values.notna()
        if lo is not None:
            mask &= values >= lo
        if hi is not None:
            mask &= values <= hi
    return mask


# --- the scoreboard ---------------------------------------------------------

def _describe(net: pd.Series) -> dict:
    return {
        "n": int(len(net)),
        "median_net_pct": round(float(net.median()), 2),
        "mean_net_pct": round(float(net.mean()), 2),
        "win_rate": round(float((net > 0).mean()), 4),
    }


def backtest(frame: pd.DataFrame, *, horizon: int, gate: str | dict | None = None,
             tiers: tuple[str, ...] | None = TRADEABLE_TIERS,
             min_price: int = 1_000, max_price: int = 40_000,
             tax_rate: float = 0.05, sell_slippage_pct: float = 2.0,
             buy_premium_pct: float = 0.0,
             min_month_rows: int = MIN_MONTH_ROWS) -> dict:
    """How a gate would really have done — month by month, against the market.

    The universe (everything tradeable in the price range) is the benchmark; the
    gate is the subset actually bought. ``median_alpha_pp`` is the headline, and
    ``months_positive`` is the honesty check: an edge that only shows up in one
    exceptional month is not an edge.
    """
    col = f"fwd_return_{horizon}d"
    if col not in frame.columns:
        frame = add_forward_returns(frame, horizons=(horizon,))

    universe = frame.dropna(subset=[col])
    if tiers is not None and "liq_tier" in universe.columns:
        universe = universe[universe["liq_tier"].isin(tiers)]
    universe = universe[universe["price"].between(min_price, max_price)]
    if universe.empty:
        return {"horizon": horizon, "gate": gate, "n": 0,
                "note": "no rows in the tradeable universe"}

    universe = universe.copy()
    universe["month"] = _as_datetime(universe["date"]).dt.strftime("%Y-%m")
    costs = dict(tax_rate=tax_rate, sell_slippage_pct=sell_slippage_pct,
                 buy_premium_pct=buy_premium_pct)
    universe["net"] = net_return_pct(universe[col], **costs)

    picked = universe[gate_mask(universe, gate)]
    if picked.empty:
        return {"horizon": horizon, "gate": gate, "n": 0,
                "note": "gate selected no rows"}

    by_month, alphas = [], []
    bench_by_month = universe.groupby("month", observed=True)["net"].median()
    for month, block in picked.groupby("month", observed=True):
        if len(block) < min_month_rows:
            continue
        gate_median = float(block["net"].median())
        bench = float(bench_by_month.get(month, np.nan))
        alpha = gate_median - bench
        alphas.append(alpha)
        by_month.append({"month": month, "n": int(len(block)),
                         "gate_net_pct": round(gate_median, 2),
                         "universe_net_pct": round(bench, 2),
                         "alpha_pp": round(alpha, 2)})

    by_tier = []
    if "liq_tier" in picked.columns:
        for tier, block in picked.groupby("liq_tier", observed=True):
            if len(block) < min_month_rows:
                continue
            by_tier.append({"tier": str(tier), **_describe(block["net"])})

    result = {
        "horizon": horizon,
        "gate": gate if isinstance(gate, str) else "custom",
        "tiers": list(tiers) if tiers else "all",
        "round_trip_cost_pct": round(round_trip_cost_pct(**costs), 2),
        **_describe(picked["net"]),
        "universe_median_net_pct": round(float(universe["net"].median()), 2),
        "months": len(by_month),
        "months_positive": sum(1 for a in alphas if a > 0),
        "median_alpha_pp": round(float(np.median(alphas)), 2) if alphas else None,
        "by_month": by_month,
        "by_tier": by_tier,
    }
    logger.info("backtest gate=%s h=%dd: alpha %s pp, %d/%d months positive",
                result["gate"], horizon, result["median_alpha_pp"],
                result["months_positive"], result["months"])
    return result


def payoff_profile(frame: pd.DataFrame, *, gate: str | dict, horizons=HORIZONS,
                   tiers: tuple[str, ...] | None = TRADEABLE_TIERS,
                   min_price: int = 1_000, max_price: int = 40_000,
                   tax_rate: float = 0.05, sell_slippage_pct: float = 2.0,
                   buy_premium_pct: float = 0.0,
                   out_of_sample: bool = True, n_splits: int = 4
                   ) -> dict[int, dict]:
    """What a gated trade actually pays when it works, and costs when it doesn't.

    This exists because of a measured failure. The regression head that was meant
    to predict *how far* a card beats the market scores **worse than assuming it
    doesn't** (-12% to -19% skill, ~0% on the traded slice). Its magnitudes are
    not usable, so expected value must not be built on them.

    What the classifier *can* do is say which gated cards will pay -- 52-59%
    precision against a 35-42% base rate. So we take the per-card probability
    from the model, and the size of the win and the loss from the gate's own
    measured history. Both halves are then things we have evidence for.

    **Measured out of sample by default**, and that is not a formality: pooling
    the whole season puts the base rate at 14 days at 58%, while walk-forward
    says 40%. Using the in-sample figure would inflate every expected value by
    around 7 percentage points and quietly reintroduce exactly the optimism this
    module exists to remove. Set ``out_of_sample=False`` only to describe history,
    never to make a decision with.

    Returns per horizon: ``win_net`` (median net % when the trade cleared costs
    and beat the market), ``loss_net`` (median net % when it didn't),
    ``base_rate`` (how often it cleared), and ``in_sample_base_rate`` for
    comparison.
    """
    # Only derive what isn't already there: a caller that has computed forward
    # returns once (training does) must not pay for them again, and must not
    # have them silently recomputed underneath it.
    missing = tuple(h for h in horizons if f"fwd_return_{h}d" not in frame.columns)
    if missing:
        frame = add_forward_returns(frame, horizons=missing)
    missing = tuple(h for h in horizons if f"excess_return_{h}d" not in frame.columns)
    if missing:
        frame = add_benchmark_returns(frame, horizons=missing)
    costs = dict(tax_rate=tax_rate, sell_slippage_pct=sell_slippage_pct,
                 buy_premium_pct=buy_premium_pct)

    universe = frame
    if tiers is not None and "liq_tier" in universe.columns:
        universe = universe[universe["liq_tier"].isin(tiers)]
    universe = universe[universe["price"].between(min_price, max_price)]
    picked = universe[gate_mask(universe, gate)]

    out: dict[int, dict] = {}
    for h in horizons:
        col = f"fwd_return_{h}d"
        if col not in picked.columns:
            continue
        block = picked.dropna(subset=[col, f"excess_return_{h}d"])
        if len(block) < MIN_MONTH_ROWS:
            continue
        net = pd.Series(net_return_pct(block[col], **costs), index=block.index)
        cleared = (net > 0) & (block[f"excess_return_{h}d"] > 0)
        in_sample_rate = float(cleared.mean())

        scored = block
        if out_of_sample:
            # Only rows a model trained on the past would have had to face.
            tested = [test for _, test in validation.walk_forward_splits(
                block["date"], n_splits=n_splits, embargo_days=h)]
            if not tested:
                continue
            scored = block.iloc[np.concatenate(tested)]
            net = net.loc[scored.index]
            cleared = cleared.loc[scored.index]

        if len(scored) < MIN_MONTH_ROWS or not cleared.any() or cleared.all():
            continue
        out[h] = {
            "win_net": round(float(net[cleared].median()), 3),
            "loss_net": round(float(net[~cleared].median()), 3),
            "base_rate": round(float(cleared.mean()), 4),
            "in_sample_base_rate": round(in_sample_rate, 4),
            "n": int(len(scored)),
            "out_of_sample": bool(out_of_sample),
        }
    return out


def sweep(frame: pd.DataFrame, *, gates=("dip_v1", "relval_v1", "deep"),
          horizons=HORIZONS, **kwargs) -> list[dict]:
    """Every gate at every horizon — the table that picks the strategy."""
    frame = add_forward_returns(frame, horizons=horizons)
    out = []
    for gate in gates:
        for h in horizons:
            out.append(backtest(frame, horizon=h, gate=gate, **kwargs))
    return out


def format_report(result: dict) -> str:
    """The scoreboard as plain text, for the CLI."""
    if not result.get("n"):
        return f"  {result.get('note', 'no data')}"
    lines = [
        f"  gate {result['gate']}  hold {result['horizon']}d  "
        f"tiers {result['tiers']}  ({result['n']:,} trades)",
        f"  needs +{result['round_trip_cost_pct']:.1f}% gross just to break even",
        f"  median net {result['median_net_pct']:+.2f}%   "
        f"market {result['universe_median_net_pct']:+.2f}%   "
        f"ALPHA {result['median_alpha_pp']:+.2f}pp   "
        f"win {result['win_rate'] * 100:.0f}%",
        f"  months beating the market: {result['months_positive']}/{result['months']}",
    ]
    if result["by_month"]:
        lines.append("    month     n   gate     market   alpha")
        for m in result["by_month"]:
            lines.append(f"    {m['month']} {m['n']:5d} {m['gate_net_pct']:+7.2f}% "
                         f"{m['universe_net_pct']:+8.2f}% {m['alpha_pp']:+7.2f}pp")
    if result["by_tier"]:
        lines.append("    by liquidity tier (C flattering A/B is a noise warning):")
        for t in result["by_tier"]:
            lines.append(f"      tier {t['tier']}  n={t['n']:6d}  "
                         f"median {t['median_net_pct']:+.2f}%  "
                         f"win {t['win_rate'] * 100:.0f}%")
    return "\n".join(lines)
