"""The live dashboard page, rendered from the database on each request.

Split by cost, not by convenience:
  live    picks, scorecard, coverage, model runs -- single-row lookups, always
          current, re-read on every page load
  cached  the market rhythms, which sweep millions of rows (see insights.py).
          Their computed-at time is printed on the page, so a stale chart
          announces itself instead of quietly misleading.
"""

from __future__ import annotations

import html as _html
from datetime import datetime, timezone

from .. import db
from ..services import scorecard
from . import insights

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _esc(v) -> str:
    return _html.escape(str(v)) if v is not None else ""


def _fmt(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


# --------------------------------------------------------------------- charts

def _weekly_svg(weekly) -> str:
    if not weekly:
        return '<p class="empty">No price history yet.</p>'
    mx = max(abs(x["ret"]) for x in weekly) or 1
    # Fixed rows so labels never collide: value labels sit in reserved bands
    # above (positive) / below (negative) the bars; day labels get their own row.
    bw, gap = 54, 14
    top_lab, mid, maxbar, neg_lab, day_row, h = 16, 74, 44, 134, 158, 168
    width = 7 * (bw + gap) - gap
    parts = [f'<line class="axis" x1="0" y1="{mid}" x2="{width}" y2="{mid}"/>']
    for i, x in enumerate(weekly):
        v = x["ret"]
        bar = abs(v) / mx * maxbar
        y = mid - bar if v >= 0 else mid
        col = "var(--up)" if v >= 0 else "var(--down)"
        xx = i * (bw + gap)
        lab_y = top_lab if v >= 0 else neg_lab   # aligned bands, clear of bars + days
        parts.append(
            f'<g class="bar"><rect x="{xx}" y="{y:.1f}" width="{bw}" '
            f'height="{max(bar,1.5):.1f}" rx="4" fill="{col}">'
            f'<title>{x["day"]}: {v:+.2f}% over {x["n"]:,} moves</title></rect>'
            f'<text class="v" x="{xx+bw/2}" y="{lab_y}">{v:+.2f}%</text>'
            f'<text class="d" x="{xx+bw/2}" y="{day_row}">{x["day"]}</text></g>')
    return (f'<svg viewBox="0 0 {width} {h}" role="img" '
            f'aria-label="Average price move by weekday">{"".join(parts)}</svg>')


def _heat_svg(hourly) -> str:
    if not hourly:
        return ('<p class="empty">No intraday data yet — it accumulates as the '
                'collector runs.</p>')
    grid = {(h["day"], h["hour"]): h["ret"] for h in hourly}
    mx = max(abs(v) for v in grid.values()) or 1
    cell, lbl = 22, 40
    parts = []
    for r, day in enumerate(DAYS):
        for hr in range(24):
            v = grid.get((day, hr))
            x, y = lbl + hr * cell, r * cell
            if v is None:
                parts.append(f'<rect class="empty-cell" x="{x}" y="{y}" '
                             f'width="{cell-2}" height="{cell-2}" rx="3"/>')
            else:
                t = min(abs(v) / mx, 1)
                col = "var(--up)" if v >= 0 else "var(--down)"
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" rx="3" '
                    f'fill="{col}" fill-opacity="{0.12+0.88*t:.2f}">'
                    f'<title>{day} {hr:02d}:00 UTC — {v:+.2f}%</title></rect>')
        parts.append(f'<text class="d" x="{lbl-8}" y="{r*cell+15}" '
                     f'text-anchor="end">{day}</text>')
    for hr in range(0, 24, 3):
        parts.append(f'<text class="tick" x="{lbl+hr*cell+cell/2-1}" '
                     f'y="{7*cell+14}" text-anchor="middle">{hr:02d}</text>')
    return (f'<svg viewBox="0 0 {lbl+24*cell} {7*cell+22}" role="img" '
            f'aria-label="Average price move by weekday and hour UTC">'
            f'{"".join(parts)}</svg>')


def _release_svg(curve) -> str:
    if len(curve) < 3:
        return '<p class="empty">Not enough post-release history yet.</p>'
    cw, ch, pl, pr, pt, pb = 640, 200, 44, 16, 16, 28
    xs = [p["day"] for p in curve]
    ys = [p["index"] for p in curve]
    ymin, ymax = min(ys) - 4, max(ys) + 4
    span = (xs[-1] - xs[0]) or 1

    def px(i):
        return pl + (xs[i] - xs[0]) / span * (cw - pl - pr)

    def py(v):
        return pt + (ymax - v) / (ymax - ymin) * (ch - pt - pb)

    pts = " ".join(f"{px(i):.1f},{py(ys[i]):.1f}" for i in range(len(curve)))
    area = (f"M{px(0):.1f},{py(ys[0]):.1f} "
            + " ".join(f"L{px(i):.1f},{py(ys[i]):.1f}" for i in range(1, len(curve)))
            + f" L{px(len(curve)-1):.1f},{ch-pb} L{px(0):.1f},{ch-pb} Z")
    low = ys.index(min(ys))
    grid = "".join(
        f'<line class="grid" x1="{pl}" y1="{py(v):.1f}" x2="{cw-pr}" y2="{py(v):.1f}"/>'
        f'<text class="tick" x="{pl-8}" y="{py(v)+4:.1f}" text-anchor="end">{v}</text>'
        for v in (100, 85, 70) if ymin < v < ymax)
    xlabs = "".join(f'<text class="tick" x="{px(i):.1f}" y="{ch-8}" '
                    f'text-anchor="middle">{xs[i]}</text>'
                    for i in range(0, len(curve), 2))
    return f'''<svg viewBox="0 0 {cw} {ch}" role="img" aria-label="Promo card price after release, indexed to 100">
<defs><linearGradient id="rg" x1="0" y1="0" x2="0" y2="1">
<stop offset="0%" stop-color="var(--down)" stop-opacity=".28"/>
<stop offset="100%" stop-color="var(--down)" stop-opacity="0"/></linearGradient></defs>
{grid}<path d="{area}" fill="url(#rg)"/>
<polyline points="{pts}" fill="none" stroke="var(--down)" stroke-width="2" stroke-linejoin="round"/>
<circle cx="{px(low):.1f}" cy="{py(ys[low]):.1f}" r="5" fill="var(--down)" stroke="var(--surface)" stroke-width="2"/>
<text class="note" x="{px(low)+10:.1f}" y="{py(ys[low])+4:.1f}">day {xs[low]} · {ys[low]:.0f} — the bottom</text>
<circle cx="{px(len(curve)-1):.1f}" cy="{py(ys[-1]):.1f}" r="4" fill="var(--up)" stroke="var(--surface)" stroke-width="2"/>
{xlabs}<text class="tick" x="{pl}" y="12">index (release = 100)</text></svg>'''


# ---------------------------------------------------------------------- picks

def _promo_reactions_html(reactions) -> str:
    if not reactions:
        return ('<p class="empty">Not enough promo history yet — this fills in as '
                'the calendar and prices accumulate.</p>')
    rows = ""
    for r in reactions:
        col = "up" if r["avg_move"] >= 0 else "down"
        rows += (f'<tr><td>{_esc(r["type"])}</td>'
                 f'<td class="num {col}">{r["avg_move"]:+.2f}%</td>'
                 f'<td class="num">{r["n"]}</td></tr>')
    return (f'<table class="record"><thead><tr><th>Promo type</th>'
            f'<th>Avg daily move</th><th>Days seen</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def _pick_card(row) -> str:
    entry = row["entry_price"] or 0
    band = (f"{_fmt(row['buy_low'])} – {_fmt(row['buy_high'])}"
            if row["buy_low"] else f"~{_fmt(entry)}")
    conf = round((row["confidence"] or 0) * 100)
    # How long this trade was given. The mean-reversion edge only clears the
    # round trip over a week or two, so the hold is part of the call.
    hold = (f"~{row['chosen_horizon_days']}d" if row["chosen_horizon_days"]
            else f"~{row['horizon_days']}d")
    sph = f"{row['sales_per_hour']:.0f}/hr" if row["sales_per_hour"] else "—"
    upside = round((row["target_price"] / entry - 1) * 100) if entry else 0
    reasons = "".join(f"<li>{_esc(r)}</li>"
                      for r in (row["reasons"] or "").split("; ") if r)
    link = (f'<a class="lnk" href="{_esc(row["url"])}" target="_blank" '
            f'rel="noopener">open on fut.gg ↗</a>' if row["url"] else "")
    status = row["status"]
    if status == "open":
        chip = '<span class="chip">running</span>'
    elif status == "target":
        chip = f'<span class="chip win">hit target {row["realized_pct"]:+.0f}%</span>'
    elif status == "stop":
        chip = f'<span class="chip loss">stopped {row["realized_pct"]:+.0f}%</span>'
    else:
        chip = f'<span class="chip">expired {row["realized_pct"]:+.0f}%</span>'
    return f'''<article class="pick">
  <header><div><h3>{_esc(row["name"] or row["player_id"])}</h3>
    <p class="meta">{_esc(row["rating"] or "—")} · {_esc(row["version"] or "")}</p></div>
    <div class="conf"><span class="num">{conf}<small>%</small></span>
      <span class="lab">confidence</span></div></header>
  <dl class="levels">
    <div><dt>Buy</dt><dd class="buy">{band}</dd></div>
    <div><dt>Sell</dt><dd class="sell">{_fmt(row["target_price"])}<span class="pct">+{upside}%</span></dd></div>
    <div><dt>Stop</dt><dd class="stop">{_fmt(row["stop_price"])}</dd></div>
    <div><dt>Hold</dt><dd>{hold}</dd></div>
    <div><dt>Sells</dt><dd>{sph}</dd></div>
  </dl>
  <ul class="why">{reasons}</ul>
  <div class="foot">{chip}{link}</div></article>'''


# ---------------------------------------------------- fragments (fetched by JS)

_VERDICT_CLASS = {"BUY": "buy", "WATCH": "watch", "WAIT": "wait", "AVOID": "avoid"}


def _card_tag(row) -> str:
    rating = row.get("rating")
    has = rating is not None and rating == rating          # not None, not NaN
    return (f"{int(rating)} {row.get('version') or ''}".strip() if has
            else (row.get("version") or ""))


def render_card_reads(items) -> str:
    """Consult results: one block per matched card version, with its verdict."""
    if not items:
        return '<p class="empty">No card matched — try the fut.gg spelling.</p>'
    out = []
    for row, read in items:
        v = read["verdict"]
        vc = _VERDICT_CLASS.get(v, "wait")
        levels = ""
        if v == "BUY":
            levels = (
                '<dl class="levels">'
                f'<div><dt>Buy</dt><dd class="buy">~{_fmt(read["price"])}</dd></div>'
                f'<div><dt>Sell</dt><dd class="sell">{_fmt(read["target"])}</dd></div>'
                f'<div><dt>Stop</dt><dd class="stop">{_fmt(read["stop"])}</dd></div>'
                f'<div><dt>R:R</dt><dd>{read["reward_risk"]:g}</dd></div></dl>')
        reasons = "".join(f"<li>{_esc(r)}</li>" for r in read["reasons"])
        link = (f'<a class="lnk" href="{_esc(row["url"])}" target="_blank" '
                f'rel="noopener">fut.gg ↗</a>' if row.get("url") else "")
        out.append(
            f'<article class="read {vc}"><header>'
            f'<div><h3>{_esc(row.get("name"))}</h3>'
            f'<p class="meta">{_esc(_card_tag(row))} · now {_fmt(read["price"])}</p></div>'
            f'<span class="verdict {vc}">{v}</span></header>'
            f'<p class="headline">{_esc(read["headline"])}</p>{levels}'
            f'<ul class="why">{reasons}</ul>'
            f'<div class="foot">{link}</div></article>')
    return "".join(out)


def render_group_read(read) -> str:
    """Consult results for a whole cohort: the group's state + best buys."""
    if read.get("error"):
        return f'<p class="empty">{_esc(read["error"])}</p>'
    move = (f'{read["group_move_7d"]:+.0f}% this week'
            if read.get("group_move_7d") is not None else "flat / unknown")
    opps = read.get("opportunities") or []
    if opps:
        rows = "".join(
            f'<li><span class="nm">{_esc(o["name"])} '
            f'<em>{_esc(o.get("version") or "")}</em></span>'
            f'<b>{_fmt(o["price"])}</b>'
            f'<span class="c">{round((o["confidence"] or 0) * 100)}%</span>'
            + (f'<a class="lnk" href="{_esc(o["url"])}" target="_blank" '
               f'rel="noopener">↗</a>' if o.get("url") else "")
            + '</li>' for o in opps)
        opp_html = f'<ul class="opps">{rows}</ul>'
    else:
        opp_html = ('<p class="empty">None on a buyable dip right now — sit tight.</p>')
    return (f'<div class="group"><p class="glede"><b>{_esc(read["dim"])} '
            f'{_esc(read["value"])}</b> — {read["n"]} cards · group {move} · '
            f'{read["share_on_dip"]:.0f}% on a dip now</p>'
            f'<h4>Best buys in the group</h4>{opp_html}</div>')


_SUPERSEDED = {
    "legacy": "old +25%/−8% engine",
    "dip_v1_broken": "dip strategy, graded with stops that sat above the market price",
}

_STRATEGY_LABELS = {
    "release_v1": "Release crash — promo cards 4–6 days old, held ~3 weeks",
    "relval_v1": "Deep dip — oversold cards on their floor, held ~2 weeks",
}


def _legacy_line(*summaries) -> str:
    """Superseded strategies, shown muted so nothing is hidden.

    dip_v1 belongs here rather than in the headline: its stops were derived from
    a marked-up entry and compared against the raw price series, so they landed
    on average 0.76% *above* the live price. 53 of 59 trades stopped out before
    they began — that record measures a bug, not a strategy.
    """
    out = ""
    for s in summaries:
        if not s or not s.get("graded"):
            continue
        label = _SUPERSEDED.get(s.get("strategy", ""), s.get("strategy", "superseded"))
        out += (f'<p class="legacy-note">{_esc(label)} (excluded above): '
                f'{s["graded"]} graded · {s["return_on_capital_pct"]:+.1f}% '
                f'on capital · {s["win_rate"]*100:.0f}% win</p>')
    return out


def render_scorecard_full(sc, graded_rows, *, superseded=()) -> str:
    """The honest track record for ONE strategy: coin-weighted stats + recent
    graded trades. Superseded strategies shown muted beneath if passed."""
    if not sc.get("graded"):
        return (f'<p class="empty">{sc.get("total", 0)} picks recorded, '
                f'{sc.get("closed", 0)} resolved — none graded yet. The record '
                f'fills in as the picks mature.</p>'
                f'{_legacy_line(*superseded)}')

    def stat(k, n, s):
        return f'<div class="stat"><div class="k">{k}</div><div class="n">{n}</div><div class="s">{s}</div></div>'

    # Alpha leads. The median tradeable card doesn't move at all over a
    # fortnight, so after the ~7.4% round trip a flat market already reads as
    # -6.9%: absolute return alone cannot tell a good call in a bad month from a
    # bad call. Return on capital stays beside it — it is what you actually keep.
    alpha = sc.get("alpha_vs_market_pct")
    alpha_stat = (stat("Alpha vs market", f'{alpha:+.1f}%',
                       f'{sc.get("benchmarked", 0)} benchmarked')
                  if alpha is not None else
                  stat("Alpha vs market", "—", "no benchmarked trades yet"))
    stats = (alpha_stat
             + stat("Return on capital", f'{sc["return_on_capital_pct"]:+.1f}%',
                    "what you actually keep")
             + stat("Win rate", f'{sc["win_rate"]*100:.0f}%', f'{sc["graded"]} graded')
             + stat("Total P&amp;L", f'{sc["coins_pnl"]:+,}',
                    f'{sc["avg_coins_per_trade"]:+,}/trade'))

    rows = ""
    for g in graded_rows:
        st = g["status"]
        chip = "win" if st == "target" else "loss" if st == "stop" else ""
        pct = g["realized_pct"]
        pcol = "up" if (pct or 0) >= 0 else "down"
        rows += (f'<tr><td>{_esc(g["name"] or g["player_id"])}</td>'
                 f'<td class="num">{_fmt(g["entry_price"])}</td>'
                 f'<td><span class="chip {chip}">{_esc(st)}</span></td>'
                 f'<td class="num {pcol}">{pct:+.0f}%</td></tr>')
    table = (f'<table class="record"><thead><tr><th>Card</th><th>Entry</th>'
             f'<th>Outcome</th><th>Net</th></tr></thead><tbody>{rows}</tbody></table>'
             if rows else "")
    return f'<div class="stats4">{stats}</div>{table}{_legacy_line(*superseded)}'


CONTROLS = [
    ("collect-bulk", "Refresh prices", "grab the whole market's latest prices"),
    ("picks", "Run picks", "recompute today's buy list"),
    ("scorecard", "Grade record", "score any picks the market has answered"),
    ("insights", "Refresh rhythms", "recompute the weekly/hourly charts"),
    ("advise-refresh", "Refresh consult", "rescore every card for the consult"),
    ("trader-tips", "Trader tips", "recompute all-tier buy tips from the pros (~1 min)"),
    ("train", "Retrain model", "retrain on all data (~10 min)"),
]


def render_controls() -> str:
    btns = "".join(
        f'<button class="ctl" data-run="{a}" title="{_esc(desc)}">{_esc(label)}</button>'
        for a, label, desc in CONTROLS)
    return (f'<div class="controls-row">{btns}</div>'
            f'<p id="ctl-status" class="ctl-status">Idle. Actions run one at a '
            f'time; the page updates when they finish.</p>')


def holding_fragment(conn, *, source: str = "futgg", title: str = "fc26") -> str:
    """Open positions from the current strategy: where each sits vs its target,
    and a SELL flag when it's there. Derived live from pick_log + latest prices."""
    from ..services.notify import NEAR_TARGET
    picks = [p for p in db.open_picks(conn, title=title)
             if p["strategy"] in scorecard.CURRENT_STRATEGIES]
    if not picks:
        return ('<p class="empty">No open positions from the current strategy. '
                'Picks you save appear here until they hit target or stop.</p>')
    rows = ""
    for p in picks:
        meta = db.card_meta_get(conn, p["player_id"])
        name = _esc(meta["name"] if meta else p["player_id"])
        price = db.latest_price(conn, p["player_id"], source)
        entry, target, stop = p["entry_price"], p["target_price"], p["stop_price"]
        if price is None:
            flag, fc, nowcol, to_tgt = "—", "", "", "—"
        else:
            net = (price * 0.95 / entry - 1) * 100
            nowcol = f'<span class="{"up" if net >= 0 else "down"}">{net:+.0f}%</span>'
            to_tgt = f"{(target / price - 1) * 100:+.0f}%"
            if price >= target * NEAR_TARGET:
                flag, fc = "SELL", "win"
            elif price <= stop:
                flag, fc = "CUT", "loss"
            else:
                flag, fc = "hold", ""
        rows += (f'<tr><td>{name}</td><td class="num">{_fmt(entry)}</td>'
                 f'<td class="num">{_fmt(price)}</td><td class="num">{nowcol}</td>'
                 f'<td class="num">{to_tgt}</td>'
                 f'<td><span class="chip {fc}">{flag}</span></td></tr>')
    return (f'<table class="record"><thead><tr><th>Card</th><th>Bought</th>'
            f'<th>Now</th><th>P&amp;L</th><th>To target</th><th></th>'
            f'</tr></thead><tbody>{rows}</tbody></table>')


def picks_fragment(conn, *, title: str = "fc26", limit: int = 12) -> str:
    """The picks grid HTML — used by the initial render and the JS refresh."""
    picks = conn.execute(
        """SELECT p.*, m.name, m.rating, m.version, m.url FROM pick_log p
           LEFT JOIN card_meta m ON m.player_id = p.player_id
           WHERE p.title = ? ORDER BY p.picked_at DESC, p.confidence DESC
           LIMIT ?""", (title, limit)).fetchall()
    return ("".join(_pick_card(r) for r in picks) if picks else
            '<p class="empty">No picks recorded yet — run the picks control.</p>')


def record_fragment(conn, *, title: str = "fc26") -> str:
    """The track record — **one block per live strategy**, never blended.

    A deep dip and a promo release crash are different bets with different
    payoffs; a single combined number would let one carry the other and hide
    which is actually working. Superseded strategies stay muted beneath so
    nothing is hidden.
    """
    superseded = [scorecard.summary(conn, title=title, strategy=s)
                  for s in _SUPERSEDED]
    blocks = []
    for name in scorecard.CURRENT_STRATEGIES:
        sc = scorecard.summary(conn, title=title, strategy=name)
        graded = conn.execute(
            """SELECT p.player_id, p.entry_price, p.status, p.realized_pct, m.name
               FROM pick_log p LEFT JOIN card_meta m ON m.player_id = p.player_id
               WHERE p.title = ? AND p.strategy = ?
                 AND p.status IN ('target','stop','expired') AND p.entry_price >= 1000
               ORDER BY p.scored_at DESC LIMIT 8""", (title, name)).fetchall()
        blocks.append(f'<h4 class="strat">{_esc(_STRATEGY_LABELS.get(name, name))}</h4>'
                      + render_scorecard_full(sc, graded))
    # Superseded records are shown once, after every live strategy.
    return "".join(blocks) + _legacy_line(*superseded)


# --------------------------------------------------------------------- render

_TIP_TIERS = {"cheap  (<5k)": ("Cheap", "under 5k"),
              "mid    (5-40k)": ("Mid", "5k – 40k"),
              "premium(40-150k)": ("Premium", "40k – 150k"),
              "elite  (150k+)": ("Elite", "150k+")}


def render_trader_tips(tips) -> str:
    """The trader-tips panel HTML, read from the cached advisor result."""
    if not tips:
        return ('<p class="empty">Not computed yet — hit the "Trader tips" control '
                'below (takes ~1 min).</p>')

    chips = ""
    for h in tips.get("held_out", []):
        if h.get("top") is None:
            continue
        lab = _TIP_TIERS.get(h["tier"], (h["tier"], ""))[0]
        cls = "up" if h["top"] >= 0 else "down"
        chips += f'<span class="ho {cls}">{lab} {h["top"]*100:+.0f}%</span>'

    head = (f'<p class="lede">Learned from {tips.get("trained_on", 0):,} of the pros’ '
            f'buy calls — buy the dip, sell into the Wednesday peak. The edge is thin and '
            f'the pros mostly win on fast execution, so treat this as a <b>watchlist to '
            f'paper-trade</b>, not a guarantee.</p>'
            f'<div class="ho-row"><span class="ho-lbl">held-out top picks</span>{chips or "—"}'
            f'<span class="ho-age">· as of {_esc(tips.get("computed_at", ""))}</span></div>')

    cards = ""
    for name, (lab, rng) in _TIP_TIERS.items():
        rows = tips.get("tiers", {}).get(name, [])
        if not rows:
            continue
        body = ""
        for t in rows:
            nm = (f'<a href="{_esc(t["url"])}" target="_blank" rel="noopener" class="tnm">'
                  f'{_esc(t["name"])}</a>' if t.get("url")
                  else f'<span class="tnm">{_esc(t["name"])}</span>')
            pcls = "up" if t["pred"] >= 0 else "down"
            sign = "+" if t["coins"] >= 0 else ""
            body += (f'<tr><td class="t-card">{nm}'
                     f'<span class="t-liq">{t["upd"]:.0f} sales/day</span></td>'
                     f'<td class="t-num t-pred {pcls}">{t["pred"]*100:+.1f}%</td>'
                     f'<td class="t-num t-coin">{sign}{_fmt(t["coins"])}</td>'
                     f'<td class="t-num t-price">{_fmt(t["price"])}</td></tr>')
        cards += (f'<div class="tier-card"><div class="tier-head">'
                  f'<span class="tier-tag">{lab}</span><span class="tier-rng">{rng}</span></div>'
                  f'<table class="tips"><thead><tr><th class="t-card">card</th>'
                  f'<th class="t-num">gain</th><th class="t-num">profit/card</th>'
                  f'<th class="t-num">price</th></tr></thead><tbody>{body}</tbody></table></div>')
    return head + f'<div class="tips-grid">{cards}</div>'


def trader_tips_fragment(conn) -> str:
    """The trader-tips panel HTML, read from cache — used by render and the JS refresh."""
    import json
    raw = db.meta_get(conn, "trader_tips")
    return render_trader_tips(json.loads(raw) if raw else None)


def render(conn, *, source: str = "futgg", title: str = "fc26",
           limit: int = 12) -> str:
    """Build the page. Live queries for everything cheap; cached rhythms."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    q1 = lambda sql, args=(): conn.execute(sql, args).fetchone()[0]  # noqa: E731

    cards = q1("SELECT COUNT(*) FROM card_meta WHERE tradeable=1")
    snaps = q1("SELECT COUNT(*) FROM price_snapshots WHERE source=?", (source,))
    deep = q1("SELECT COUNT(*) FROM (SELECT player_id FROM price_snapshots "
              "WHERE source=? GROUP BY player_id HAVING COUNT(*)>=10)", (source,))
    sale_rows = q1("SELECT COUNT(*) FROM sale_stats")
    events = q1("SELECT COUNT(*) FROM market_events")
    last_price = q1("SELECT MAX(timestamp) FROM price_snapshots WHERE source=?",
                    (source,)) or "—"
    first_price = q1("SELECT MIN(timestamp) FROM price_snapshots WHERE source=?",
                     (source,)) or "—"
    tiers = {t: q1("SELECT COUNT(*) FROM liquidity WHERE tier=?", (t,))
             for t in "ABC"}

    from .picks import STRATEGY_VERSION
    sc = scorecard.summary(conn, title=title, strategy=STRATEGY_VERSION)
    run = db.latest_model_run(conn, kind="clears", title=title)
    metrics = {}
    if run and run["metrics_json"]:
        import json
        try:
            metrics = json.loads(run["metrics_json"])
        except json.JSONDecodeError:
            metrics = {}

    picks_html = picks_fragment(conn, title=title, limit=limit)
    holding_html = holding_fragment(conn, source=source, title=title)
    record_html = record_fragment(conn, title=title)
    controls_html = render_controls()

    trader_tips_html = trader_tips_fragment(conn)

    stats = insights.load(conn) or {}
    rhythm_age = stats.get("computed_at", "not computed yet")

    # freshness of the price feed, stated plainly
    try:
        age_min = (datetime.now(timezone.utc) - datetime.strptime(
            last_price, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        ).total_seconds() / 60
        fresh = (f'<span class="chip win">prices {age_min:.0f} min old</span>'
                 if age_min < 180 else
                 f'<span class="chip loss">prices {age_min/60:.1f} h old</span>')
    except (TypeError, ValueError):
        fresh = '<span class="chip loss">no price data</span>'

    graded = sc.get("graded", 0)
    if graded:
        record = (f'{sc["win_rate"]*100:.0f}%',
                  f'{graded} graded · {sc["return_on_capital_pct"]:+.1f}% on capital')
    else:
        record = ("—", f'{sc.get("open", 0)} running, none graded yet')

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FUT Market Desk — live</title>
<style>{_CSS}</style></head><body>
<div class="wrap">
<header class="top">
  <div><span class="rule"></span><h1>FUT Market Desk</h1>
    <p>Consult the model, see what it wants to buy, and how it's really doing</p></div>
  <div class="stamp">{fresh}<p class="mono">{now}</p></div>
</header>

<nav class="tabs">
  <a href="#consult">Consult</a><a href="#picks">Picks</a>
  <a href="#holding">Holding</a><a href="#record">Track record</a>
  <a href="#rhythms">Rhythms</a><a href="#controls">Controls</a>
</nav>

<section id="consult"><h2>Consult the model</h2>
  <p class="lede">Ask about any card or group — no filters, an honest read on each.
    Type a name (try <b>Mbappe</b>) or tap a group.</p>
  <input id="consult-q" class="search" type="search" autocomplete="off"
    placeholder="Ask about any card…  e.g. Mbappe, Haaland, Saka" aria-label="Consult a card">
  <div class="chips">
    <span class="chip-lbl">Groups:</span>
    <button class="gchip" data-group="rating:83">83s</button>
    <button class="gchip" data-group="rating:84">84s</button>
    <button class="gchip" data-group="rating:85">85s</button>
    <button class="gchip" data-group="rating:86">86s</button>
    <button class="gchip" data-group="rating:87">87s</button>
    <button class="gchip" data-group="league:Premier League">Premier League</button>
    <button class="gchip" data-group="league:LaLiga EA SPORTS">LaLiga</button>
    <button class="gchip" data-group="position:ST">ST</button>
    <button class="gchip" data-group="position:GK">GK</button>
  </div>
  <div id="consult-out" class="consult-out"></div></section>

<div class="tiles">
  <div class="tile"><div class="k">Open picks</div><div class="n">{sc.get('open',0)}</div>
    <div class="s">{record[1]}</div></div>
  <div class="tile"><div class="k">Win rate</div><div class="n">{record[0]}</div>
    <div class="s">of graded picks</div></div>
  <div class="tile"><div class="k">Cards tracked</div><div class="n">{_fmt(cards)}</div>
    <div class="s">{_fmt(deep)} with full history</div></div>
  <div class="tile"><div class="k">Price records</div><div class="n">{snaps/1e6:.2f}M</div>
    <div class="s">{first_price[:10]} → {last_price[:10]}</div></div>
  <div class="tile"><div class="k">Best-pick accuracy</div>
    <div class="n">{metrics.get('precision_at_top_decile',0)*100:.0f}%</div>
    <div class="s">vs {metrics.get('base_rate',0)*100:.0f}% at random</div></div>
</div>

<section id="picks"><h2>Picks</h2>
  <p class="lede">Cheap/mid cards on the dip — oversold, near their own floor, sized to
    each card. Buy prices are ranges anchored to the live listing. Only cards that
    genuinely trade are shown.</p>
  <div id="picks-out" class="picks">{picks_html}</div></section>

<section id="trader-tips"><h2>Trader tips (all tiers)</h2>
  <p class="lede">Buy tips across every price tier — cheap to elite — learned from the
    Discord pros' calls. Ranked by predicted return, coin profit, and how fillable the
    card is. The pros' edge is mostly fast execution, so treat this as a watchlist.</p>
  <div id="trader-tips-out">{trader_tips_html}</div></section>

<section id="holding"><h2>Holding</h2>
  <p class="lede">Open positions from the current strategy — where each sits versus its
    target right now. <b>SELL</b> means it's reached target; you get a Discord ping too.</p>
  <div id="holding-out">{holding_html}</div></section>

<section id="record"><h2>Track record</h2>
  <p class="lede">The honest, coin-weighted scoreboard — judged only on tradeable cards,
    net of tax. Return on capital is the number that matters.</p>
  <div id="record-out">{record_html}</div></section>

<section id="rhythms"><h2>The weekly supply cycle</h2>
  <div class="charts">
    <div class="card"><h3>Average price move by day</h3>
      <p class="sub">Rewards flood the market with cards, then supply dries up. Prices climb
        Monday to Wednesday and bleed from Thursday through the weekend. Bars sit above or
        below the line, so direction reads without relying on colour.</p>
      <div class="plot">{_weekly_svg(stats.get('weekly', []))}</div></div>
    <div class="card"><h3>When cards get dumped — by hour (UTC)</h3>
      <p class="sub">Teal is rising, red is falling, darker is stronger. The bruises line up
        with reward drops: Thursday afternoon, Friday evening, Saturday evening. Monday
        morning is the recovery. Hover any cell for its number.</p>
      <div class="plot">{_heat_svg(stats.get('hourly', []))}</div></div>
  </div></section>

<section><h2>What a new promo card does</h2>
  <div class="card"><h3>Price after release, indexed to 100</h3>
    <p class="sub">Promos land on Friday. The card falls hard over its first few days,
      bottoms out around day nine, then recovers. This is why the model is told how old a
      card is, not just what it costs.</p>
    <div class="plot">{_release_svg(stats.get('release_curve', []))}</div></div>
  <div class="card"><h3>How each promo type moves the market</h3>
    <p class="sub">The model knows <em>when</em> EA drops a promo; this is what each
      <em>kind</em> does — the average daily move over the days around every event of
      that type. Icons and Heroes reprice the market differently than a routine SBC.</p>
    {_promo_reactions_html(stats.get('promo_reactions', []))}</div></section>

<section><h2>Data &amp; model health</h2>
  <div class="two">
    <div class="card"><h3>Coverage</h3><ul class="kv">
      <li><span>Cards that can be sold</span><b>{_fmt(cards)}</b></li>
      <li><span>With full price history</span><b>{_fmt(deep)}</b></li>
      <li><span>With real sold-price data</span><b>{_fmt(sale_rows)}</b></li>
      <li><span>Sells fast (tier A)</span><b>{_fmt(tiers['A'])}</b></li>
      <li><span>Moderate (tier B)</span><b>{_fmt(tiers['B'])}</b></li>
      <li><span>Slow (tier C)</span><b>{_fmt(tiers['C'])}</b></li></ul></div>
    <div class="card"><h3>Model</h3><ul class="kv">
      <li><span>Best-pick accuracy</span><b>{metrics.get('precision_at_top_decile',0)*100:.1f}%</b></li>
      <li><span>Random baseline</span><b>{metrics.get('base_rate',0)*100:.1f}%</b></li>
      <li><span>Lift over random</span><b>{metrics.get('lift_vs_base_rate','—')}×</b></li>
      <li><span>Trained on</span><b>{_fmt(run['n_samples']) if run else '—'} rows</b></li>
      <li><span>Events in calendar</span><b>{_fmt(events)}</b></li>
      <li><span>Rhythms computed</span><b>{_esc(rhythm_age)}</b></li></ul></div>
  </div>
  <p class="callout"><b>Read this before trading on it.</b> The model picks better than
    chance, but its edge is roughly the same size as the cost of trading — about 11% round
    trip once you count buying above the cheapest listing, selling under it, and EA's 5%
    tax. Patience is the deciding factor: bid near the listed price and it pays; chase at
    market and it doesn't.{
      ' No picks have been graded yet, so there is no track record to lean on.'
      if not graded else ''}</p>
</section>

<section id="controls"><h2>Controls</h2>
  <p class="lede">Run the system from here. Actions run one at a time in the background;
    picks and the track record refresh automatically when a job finishes.</p>
  {controls_html}</section>

<footer><span>fc-market-analytics — picks &amp; record refresh live</span>
  <span class="mono">{snaps:,} price records</span></footer>
</div><script src="/app.js"></script></body></html>"""


_CSS = """
:root{--ground:#FAF8F4;--surface:#FFFFFF;--raised:#F2EFE8;--ink:#16201D;--ink-2:#4C5A55;
  --ink-3:#7C8A84;--line:#E2DED4;--accent:#A8762A;--up:#0F8770;--down:#B54A41;
  --shadow:0 1px 2px rgba(20,30,26,.06);}
@media (prefers-color-scheme:dark){:root{--ground:#0D1513;--surface:#141F1C;--raised:#1B2825;
  --ink:#E9EFEB;--ink-2:#A4B3AD;--ink-3:#71827C;--line:#243330;--accent:#D4A03C;
  --up:#12A088;--down:#C4574E;--shadow:none;}}
:root[data-theme="dark"]{--ground:#0D1513;--surface:#141F1C;--raised:#1B2825;--ink:#E9EFEB;
  --ink-2:#A4B3AD;--ink-3:#71827C;--line:#243330;--accent:#D4A03C;--up:#12A088;
  --down:#C4574E;--shadow:none;}
:root[data-theme="light"]{--ground:#FAF8F4;--surface:#FFFFFF;--raised:#F2EFE8;--ink:#16201D;
  --ink-2:#4C5A55;--ink-3:#7C8A84;--line:#E2DED4;--accent:#A8762A;--up:#0F8770;
  --down:#B54A41;--shadow:0 1px 2px rgba(20,30,26,.06);}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:400 15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 72px}
h1,h2,h3{font-family:"Avenir Next Condensed","Helvetica Neue",system-ui,sans-serif;
  letter-spacing:.01em;text-wrap:balance;margin:0}
h1{font-size:30px;font-weight:600}
h2{font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.13em;color:var(--ink-3)}
.mono,.num,dd,.tick,.v,b{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-variant-numeric:tabular-nums}
header.top{display:flex;justify-content:space-between;align-items:baseline;gap:16px;flex-wrap:wrap;
  border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:26px}
header.top p{margin:4px 0 0;color:var(--ink-3);font-size:13px}
.stamp{text-align:right}
.rule{display:inline-block;width:26px;height:3px;background:var(--accent);border-radius:2px;margin-bottom:10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:38px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;box-shadow:var(--shadow)}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--ink-3)}
.tile .n{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-variant-numeric:tabular-nums;
  font-size:26px;font-weight:600;margin-top:5px;letter-spacing:-.02em}
.tile .s{font-size:12px;color:var(--ink-3);margin-top:2px}
section{margin-bottom:40px}section>h2{margin-bottom:14px}
.lede{color:var(--ink-2);font-size:14px;margin:0 0 16px;max-width:62ch}
.picks{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:14px}
.pick{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px;
  box-shadow:var(--shadow);border-left:3px solid var(--accent)}
.pick header{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:12px}
.pick h3{font-size:17px;font-weight:600}
.pick .meta{margin:2px 0 0;font-size:12px;color:var(--ink-3)}
.conf{text-align:right;flex:none}
.conf .num{font-size:21px;font-weight:600;color:var(--accent);display:block;line-height:1}
.conf .num small{font-size:12px;font-weight:500}
.conf .lab{font-size:10px;text-transform:uppercase;letter-spacing:.09em;color:var(--ink-3)}
.levels{display:grid;grid-template-columns:1fr 1fr;gap:9px 14px;margin:0 0 12px;padding:11px 0;
  border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.levels div{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.levels dt{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink-3)}
.levels dd{margin:0;font-size:13px;font-weight:600}
dd.buy{color:var(--ink)}dd.sell{color:var(--up)}dd.stop{color:var(--down)}
.pct{font-size:11px;color:var(--ink-3);margin-left:5px;font-weight:400}
.why{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:5px}
.why li{font-size:12.5px;color:var(--ink-2);padding-left:13px;position:relative;line-height:1.4}
.why li::before{content:"";position:absolute;left:0;top:7px;width:5px;height:5px;border-radius:50%;
  background:var(--accent);opacity:.5}
.foot{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-top:11px;flex-wrap:wrap}
.chip{font-size:11px;padding:3px 9px;border-radius:99px;border:1px solid var(--line);
  background:var(--raised);color:var(--ink-2)}
.chip.win{color:var(--up);border-color:color-mix(in srgb,var(--up) 35%,transparent)}
.chip.loss{color:var(--down);border-color:color-mix(in srgb,var(--down) 35%,transparent)}
.lnk{font-size:12px;color:var(--accent);text-decoration:none;border-bottom:1px solid currentColor;padding-bottom:1px}
.lnk:hover,.lnk:focus-visible{opacity:.75}
.charts{display:grid;gap:14px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:var(--shadow)}
.card h3{font-size:15px;font-weight:600;margin-bottom:3px}
.card p.sub{margin:0 0 14px;font-size:13px;color:var(--ink-2);max-width:62ch}
.plot{overflow-x:auto}
svg{display:block;width:100%;height:auto;min-width:420px}
.axis{stroke:var(--ink-3);stroke-width:1;opacity:.45}
.grid{stroke:var(--line);stroke-width:1}
text{fill:var(--ink-3);font-size:10px;font-family:ui-monospace,"SF Mono",Menlo,monospace}
text.v{fill:var(--ink-2);font-size:10.5px;text-anchor:middle;font-weight:600}
text.d{fill:var(--ink-2);font-size:11px;text-anchor:middle}
text.note{fill:var(--ink-2);font-size:11px;font-family:system-ui,sans-serif}
rect.empty-cell{fill:var(--raised)}
.bar rect{transition:opacity .15s}.bar:hover rect{opacity:.78}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:760px){.two{grid-template-columns:1fr}}
.kv{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px}
.kv li{display:flex;justify-content:space-between;align-items:baseline;gap:12px;font-size:13px;
  padding-bottom:8px;border-bottom:1px solid var(--line)}
.kv li:last-child{border-bottom:0;padding-bottom:0}
.kv span{color:var(--ink-2)}.kv b{font-weight:600}
.callout{background:var(--raised);border:1px solid var(--line);border-left:3px solid var(--down);
  border-radius:10px;padding:14px 16px;font-size:13.5px;color:var(--ink-2);line-height:1.5;margin-top:14px}
.callout b{color:var(--ink)}
.empty{color:var(--ink-3);font-size:13px;margin:0}
code{font-family:ui-monospace,Menlo,monospace;font-size:12px;background:var(--raised);
  padding:1px 5px;border-radius:4px}
footer{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);font-size:12px;
  color:var(--ink-3);display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap}
a:focus-visible{outline:2px solid var(--accent);outline-offset:3px;border-radius:3px}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}

/* nav */
nav.tabs{position:sticky;top:0;z-index:5;display:flex;gap:6px;flex-wrap:wrap;
  background:color-mix(in srgb,var(--ground) 88%,transparent);backdrop-filter:blur(8px);
  padding:10px 0;margin:-4px 0 26px;border-bottom:1px solid var(--line)}
nav.tabs a{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;
  color:var(--ink-2);text-decoration:none;padding:6px 12px;border-radius:99px;border:1px solid transparent}
nav.tabs a:hover,nav.tabs a:focus-visible{background:var(--surface);border-color:var(--line);color:var(--ink)}
/* consult */
.search{width:100%;font:400 16px/1.4 system-ui,sans-serif;color:var(--ink);
  background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:14px 16px;box-shadow:var(--shadow);margin-bottom:12px}
.search:focus{outline:none;border-color:var(--accent)}
.chips{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-bottom:16px}
.chip-lbl{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--ink-3);margin-right:2px}
.gchip{font:500 12.5px system-ui,sans-serif;color:var(--ink-2);background:var(--surface);
  border:1px solid var(--line);border-radius:99px;padding:6px 13px;cursor:pointer;transition:.12s}
.gchip:hover,.gchip:focus-visible{border-color:var(--accent);color:var(--ink)}
.consult-out{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
.consult-out:empty{display:none}
.read{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px;
  box-shadow:var(--shadow);border-left:3px solid var(--ink-3)}
.read.buy{border-left-color:var(--up)}.read.avoid{border-left-color:var(--down)}
.read.watch,.read.wait{border-left-color:var(--accent)}
.read header{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:9px}
.read h3{font-size:16px;font-weight:600}.read .meta{margin:2px 0 0;font-size:12px;color:var(--ink-3)}
.verdict{font:600 11px system-ui,sans-serif;text-transform:uppercase;letter-spacing:.07em;
  padding:4px 10px;border-radius:99px;border:1px solid var(--line);white-space:nowrap;flex:none}
.verdict.buy{color:var(--up);border-color:color-mix(in srgb,var(--up) 40%,transparent);
  background:color-mix(in srgb,var(--up) 10%,transparent)}
.verdict.avoid{color:var(--down);border-color:color-mix(in srgb,var(--down) 40%,transparent);
  background:color-mix(in srgb,var(--down) 10%,transparent)}
.verdict.watch,.verdict.wait{color:var(--accent);border-color:color-mix(in srgb,var(--accent) 40%,transparent);
  background:color-mix(in srgb,var(--accent) 10%,transparent)}
.read .headline{margin:0 0 10px;font-size:13.5px;color:var(--ink-2);line-height:1.45}
.read .levels{margin:0 0 10px}
.group{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px;
  box-shadow:var(--shadow);grid-column:1/-1}
.glede{margin:0 0 12px;font-size:14px;color:var(--ink-2)}.glede b{color:var(--ink)}
.group h4{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--ink-3)}
.opps{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:6px}
.opps li{display:flex;align-items:baseline;gap:10px;font-size:13.5px;padding-bottom:6px;
  border-bottom:1px solid var(--line)}
.opps li:last-child{border-bottom:0}.opps .nm{flex:1;color:var(--ink)}.opps .nm em{color:var(--ink-3);font-style:normal;font-size:12px}
.opps b{font-weight:600}.opps .c{color:var(--accent);font-size:12px;font-weight:600}
/* track record */
.stats4{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:16px}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;box-shadow:var(--shadow)}
.stat .k{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--ink-3)}
.stat .n{font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums;font-size:24px;font-weight:600;margin-top:5px}
.stat .s{font-size:12px;color:var(--ink-3);margin-top:2px}
table.record{width:100%;border-collapse:collapse;font-size:13px}
table.record th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.08em;
  color:var(--ink-3);font-weight:600;padding:6px 10px;border-bottom:1px solid var(--line)}
table.record td{padding:7px 10px;border-bottom:1px solid var(--line)}
table.record td.num{font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums;text-align:right}
td.up{color:var(--up)}td.down{color:var(--down)}
.legacy-note{margin:14px 0 0;font-size:12px;color:var(--ink-3);font-style:italic}
/* controls */
.controls-row{display:flex;flex-wrap:wrap;gap:9px}
button.ctl{font:600 13px system-ui,sans-serif;color:var(--ink);background:var(--surface);
  border:1px solid var(--line);border-radius:10px;padding:10px 15px;cursor:pointer;
  box-shadow:var(--shadow);transition:.12s}
button.ctl:hover:not(:disabled),button.ctl:focus-visible{border-color:var(--accent);color:var(--accent)}
button.ctl:disabled{opacity:.5;cursor:progress}
.ctl-status{margin:14px 0 0;font-size:13px;color:var(--ink-2);
  background:var(--raised);border:1px solid var(--line);border-radius:8px;padding:10px 13px}
/* trader tips */
.ho-row{display:flex;flex-wrap:wrap;align-items:center;gap:7px;margin:-4px 0 18px}
.ho-lbl{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--ink-3)}
.ho{font-size:12px;padding:2px 9px;border-radius:99px;border:1px solid var(--line);
  font-variant-numeric:tabular-nums}
.ho.up{color:var(--up);border-color:color-mix(in srgb,var(--up) 32%,transparent)}
.ho.down{color:var(--down);border-color:color-mix(in srgb,var(--down) 32%,transparent)}
.ho-age{font-size:12px;color:var(--ink-3)}
.tips-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.tier-card{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  padding:14px 16px 6px;box-shadow:var(--shadow)}
.tier-head{display:flex;align-items:baseline;gap:9px;margin-bottom:2px}
.tier-tag{font-family:"Avenir Next Condensed","Helvetica Neue",system-ui,sans-serif;
  font-size:15px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--accent)}
.tier-rng{font-size:11px;color:var(--ink-3);font-variant-numeric:tabular-nums}
table.tips{width:100%;border-collapse:collapse}
table.tips th{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-3);
  font-weight:600;padding:4px 0 7px;border-bottom:1px solid var(--line)}
table.tips th.t-card{text-align:left}
table.tips td{padding:8px 0;border-bottom:1px solid var(--line);font-size:13px;vertical-align:baseline}
table.tips tbody tr:last-child td{border-bottom:none}
.tips td.t-card{padding-right:10px}
.tips .tnm{color:var(--ink);text-decoration:none;font-weight:500;
  border-bottom:1px solid transparent;line-height:1.3}
a.tnm:hover,a.tnm:focus-visible{color:var(--accent);border-bottom-color:currentColor}
.tips .t-liq{display:block;font-size:11px;color:var(--ink-3);font-variant-numeric:tabular-nums;margin-top:1px}
.tips .t-num{text-align:right;white-space:nowrap;padding-left:12px;
  font-family:ui-monospace,"SF Mono",Menlo,monospace;font-variant-numeric:tabular-nums}
.tips td.t-pred{font-weight:600}
.tips td.t-pred.up{color:var(--up)}.tips td.t-pred.down{color:var(--down)}
.tips td.t-coin{color:var(--ink);font-weight:500}
.tips td.t-price{color:var(--ink-3)}
"""


APP_JS = """'use strict';
const $ = s => document.querySelector(s);
async function frag(url, sel){
  const t = $(sel); if(!t) return;
  t.innerHTML = '<p class="empty">…</p>';
  try { const r = await fetch(url); const j = await r.json(); t.innerHTML = j.html || ''; }
  catch(e){ t.innerHTML = '<p class="empty">could not load — try again</p>'; }
}
// consult: search-as-you-type (debounced) + Enter
const box = $('#consult-q');
if (box){
  let tmr;
  const go = () => { const q = box.value.trim();
    if(!q){ $('#consult-out').innerHTML=''; return; }
    frag('/api/advise?q=' + encodeURIComponent(q), '#consult-out'); };
  box.addEventListener('input', () => { clearTimeout(tmr); tmr = setTimeout(go, 350); });
  box.addEventListener('keydown', e => { if(e.key==='Enter'){ clearTimeout(tmr); go(); } });
}
document.querySelectorAll('[data-group]').forEach(el => el.addEventListener('click', () => {
  const i = el.dataset.group.indexOf(':');
  const dim = el.dataset.group.slice(0,i), val = el.dataset.group.slice(i+1);
  frag('/api/advise/group?dim=' + encodeURIComponent(dim) + '&value=' + encodeURIComponent(val), '#consult-out');
}));
// controls
function refreshLive(){ frag('/api/scorecard', '#record-out'); frag('/api/picks', '#picks-out'); frag('/api/holding', '#holding-out'); frag('/api/trader-tips', '#trader-tips-out'); }
async function poll(id, action, btn){
  const s = $('#ctl-status');
  try {
    const r = await fetch('/api/jobs/' + id); const j = await r.json();
    s.textContent = action + ': ' + j.status + (j.detail ? (' — ' + j.detail) : '');
    if (j.status === 'running' || j.status === 'queued'){ setTimeout(() => poll(id, action, btn), 1500); }
    else { btn.disabled = false; if (j.status === 'done') refreshLive(); }
  } catch(e){ btn.disabled = false; }
}
document.querySelectorAll('[data-run]').forEach(btn => btn.addEventListener('click', async () => {
  const action = btn.dataset.run, s = $('#ctl-status');
  btn.disabled = true; s.textContent = 'Starting ' + action + '…';
  try {
    const r = await fetch('/api/run/' + action, {method:'POST', headers:{'X-Requested-With':'fetch'}});
    const j = await r.json();
    if (j.error){ s.textContent = j.error; btn.disabled = false; return; }
    poll(j.job_id, action, btn);
  } catch(e){ s.textContent = 'failed to start'; btn.disabled = false; }
}));
// keep picks + record fresh without a full-page reload
setInterval(refreshLive, 120000);
"""
