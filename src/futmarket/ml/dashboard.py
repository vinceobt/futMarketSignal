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
    bw, gap, h, mid = 54, 14, 120, 60
    width = 7 * (bw + gap) - gap
    parts = [f'<line class="axis" x1="0" y1="{mid}" x2="{width}" y2="{mid}"/>']
    for i, x in enumerate(weekly):
        v = x["ret"]
        bar = abs(v) / mx * 52
        y = mid - bar if v >= 0 else mid
        col = "var(--up)" if v >= 0 else "var(--down)"
        xx = i * (bw + gap)
        lab_y = y - 8 if v >= 0 else y + bar + 16
        parts.append(
            f'<g class="bar"><rect x="{xx}" y="{y:.1f}" width="{bw}" '
            f'height="{max(bar,1.5):.1f}" rx="4" fill="{col}">'
            f'<title>{x["day"]}: {v:+.2f}% over {x["n"]:,} moves</title></rect>'
            f'<text class="v" x="{xx+bw/2}" y="{lab_y:.1f}">{v:+.2f}%</text>'
            f'<text class="d" x="{xx+bw/2}" y="{h-4}">{x["day"]}</text></g>')
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

def _pick_card(row) -> str:
    entry = row["entry_price"] or 0
    band = (f"{_fmt(row['buy_low'])} – {_fmt(row['buy_high'])}"
            if row["buy_low"] else f"~{_fmt(entry)}")
    conf = round((row["confidence"] or 0) * 100)
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
    <div><dt>Sells</dt><dd>{sph}</dd></div>
  </dl>
  <ul class="why">{reasons}</ul>
  <div class="foot">{chip}{link}</div></article>'''


# --------------------------------------------------------------------- render

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

    sc = scorecard.summary(conn, title=title)
    run = db.latest_model_run(conn, kind="direction", title=title)
    metrics = {}
    if run and run["metrics_json"]:
        import json
        try:
            metrics = json.loads(run["metrics_json"])
        except json.JSONDecodeError:
            metrics = {}

    picks = conn.execute(
        """SELECT p.*, m.name, m.rating, m.version, m.url FROM pick_log p
           LEFT JOIN card_meta m ON m.player_id = p.player_id
           WHERE p.title = ? ORDER BY p.picked_at DESC, p.confidence DESC
           LIMIT ?""", (title, limit)).fetchall()
    picks_html = ("".join(_pick_card(r) for r in picks) if picks else
                  '<p class="empty">No picks recorded yet — run '
                  '<code>futmarket picks --save</code>.</p>')

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
<meta http-equiv="refresh" content="120">
<title>FUT Market Desk — live</title>
<style>{_CSS}</style></head><body>
<div class="wrap">
<header class="top">
  <div><span class="rule"></span><h1>FUT Market Desk</h1>
    <p>What the model wants to buy, and the market rhythms behind it</p></div>
  <div class="stamp">{fresh}<p class="mono">{now}</p></div>
</header>

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

<section><h2>Picks</h2>
  <p class="lede">Ranked by the model's confidence. Buy prices are ranges anchored to the
    live listing — you pay a little over the cheapest to actually get filled. Only cards
    that genuinely trade are shown.</p>
  <div class="picks">{picks_html}</div></section>

<section><h2>The weekly supply cycle</h2>
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
    <div class="plot">{_release_svg(stats.get('release_curve', []))}</div></div></section>

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

<footer><span>fc-market-analytics — page refreshes every 2 minutes</span>
  <span class="mono">{snaps:,} price records</span></footer>
</div></body></html>"""


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
"""
