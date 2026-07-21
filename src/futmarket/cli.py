"""futmarket CLI — the ML trading system.

Data in:
  futmarket build-registry          crawl fut.gg's full card list
  futmarket collect-bulk            snapshot the whole market's prices
  futmarket backfill-history        deep daily history since release
  futmarket build-calendar          promos/TOTW/SBCs/EA news
  futmarket sale-stats              what cards really sold for
  futmarket social [--x]            Reddit/YouTube/X buzz

Learn + decide:
  futmarket score-liquidity         A/B/C tradeability tiers (rule #1)
  futmarket build-dataset           assemble the ML feature matrix
  futmarket train                   train + walk-forward validate the models
  futmarket picks                   what to buy, at what price, and why
  futmarket scorecard               how past picks have actually done
  futmarket insights                refresh the dashboard's market rhythms
  futmarket notify                  post a run summary to Discord
  futmarket dashboard               the live web dashboard
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import db
from .config import Config, ConfigError, load_config
from .log import setup_logging

_LOOPBACK = {"127.0.0.1", "::1", "localhost", ""}


def _load(args) -> Config:
    try:
        return load_config(args.config)
    except ConfigError as exc:
        sys.exit(f"config error: {exc}")


def _resolve_source(conn, requested: str | None, config=None) -> str:
    """Which stored price series to build features from. Explicit --source wins;
    otherwise pick the source holding the most snapshots (the one you actually
    have data for). On a fresh/empty DB fall back to the configured collection
    source so discovery can bootstrap (e.g. right after a clean-slate wipe)."""
    if requested:
        return requested
    row = conn.execute(
        "SELECT source, COUNT(*) AS n FROM price_snapshots "
        "GROUP BY source ORDER BY n DESC LIMIT 1").fetchone()
    if row is not None:
        return row["source"]
    if config is not None:
        return config.source
    sys.exit("no price snapshots stored yet — collect some first")


def cmd_build_registry(args) -> None:
    from .services import registry
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)

    def progress(seen, tradeable):
        print(f"  ...{seen} cards ({tradeable} tradeable)")

    print(f"crawling fut.gg card list (game {args.game}"
          f"{', max %d pages' % args.max_pages if args.max_pages else ''})...")
    res = registry.refresh_registry(conn, game=args.game, max_pages=args.max_pages,
                                    delay=args.delay, progress=progress)
    total = db.card_count(conn)
    print(f"registry: {res['seen']} cards this crawl "
          f"({res['tradeable']} tradeable); {total} total in card_meta")


def cmd_collect_bulk(args) -> None:
    from .services import bulk_collect
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    if db.card_count(conn) == 0:
        sys.exit("card_meta is empty — run `futmarket build-registry` first")
    print(f"fetching whole-market prices ({config.platform})...")
    res = bulk_collect.collect_bulk(conn, platform=config.platform)
    print(f"bulk collect: fetched {res['fetched']:,} prices, "
          f"matched {res['matched']:,} tracked cards, "
          f"inserted {res['inserted']:,} new snapshots "
          f"({res['unknown']:,} priced ids not in registry)")


def cmd_x_login(args) -> None:
    """One-off: capture an X session so the reader can use it."""
    from .collectors import x_source
    from .collectors.base import SourceError

    if args.from_cookies:
        # The reliable path: X blocks logins inside an automated browser, so we
        # reuse the session from a tab you're already signed into instead.
        print("Reusing a session you're already logged into — no new window needed.\n")
        print("In your logged-in X tab:")
        print("  1. Open DevTools  (Cmd+Option+I)")
        print("  2. Application tab -> Cookies -> https://x.com")
        print("  3. Copy these two values:\n")
        try:
            auth_token = input("  auth_token: ").strip()
            ct0 = input("  ct0 (Enter to skip): ").strip()
            path = x_source.build_session_from_cookies(
                auth_token, ct0, args.session_file)
        except SourceError as e:
            sys.exit(str(e))
        print(f"\nSaved to {path} (gitignored).")
        print("Test it:  futmarket social --x --no-reddit --no-youtube")
        return

    print("Use a BURNER X account — automated reading breaks X's terms, and a")
    print("suspension should land on a throwaway, not your real account.\n")
    print("If the login window gets blocked, use:  futmarket x-login --from-cookies\n")
    try:
        path = x_source.save_session(args.session_file, timeout_s=args.timeout)
    except SourceError as e:
        sys.exit(str(e))
    print(f"\nSaved to {path} (gitignored).")
    print("Now add the handles to config.yaml under twitter.creators, then run")
    print("`futmarket social --x` to read them.")


def cmd_social(args) -> None:
    from .services import social
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    if args.show:
        rows = social.buzz_table(conn, limit=args.show)
        if not rows:
            print("no buzz recorded yet — run `futmarket social`")
            return
        print(f"{'player':<26}{'mentions':>9}{'lean':>9}{'cards':>7}  where")
        print("-" * 62)
        for r in rows:
            lean = r["sentiment"] or 0
            mood = "bullish" if lean > 0.15 else "bearish" if lean < -0.15 else "mixed"
            print(f"{(r['name'] or '?')[:24]:<26}{r['mentions']:>9}{mood:>9}"
                  f"{r['cards']:>7}  {r['platforms'] or ''}")
        return
    try:
        res = social.collect(conn, reddit=not args.no_reddit,
                             youtube=not args.no_youtube, x=args.x)
    except Exception as exc:
        sys.exit(str(exc))
    print(f"social: {res['posts']} posts fetched, {res['matched']} mentioned a "
          f"known player, {res['cards']} card signals stored "
          f"({res['names_indexed']} names indexed)")


def cmd_insights(args) -> None:
    from .ml import insights
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    source = _resolve_source(conn, args.source, config)
    stats = insights.refresh(conn, source=source)
    print(f"insights refreshed at {stats['computed_at']}: "
          f"{len(stats['weekly'])} weekday points, {len(stats['hourly'])} hourly cells, "
          f"{len(stats['release_curve'])} release-curve days")


def cmd_notify(args) -> None:
    """Post a short run summary to Discord so the owner can keep account."""
    from . import secrets
    from .services import notify
    config = _load(args)
    setup_logging(config.log_path)
    webhook = secrets.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        print("no DISCORD_WEBHOOK_URL set (add it to .env) — skipping notification")
        return
    conn = db.connect(config.database_path)
    if notify.send_run_summary(conn, webhook):
        print("sent Discord notification")
    else:
        print("could not send Discord notification (see log)")


def cmd_scorecard(args) -> None:
    from .services import scorecard
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    source = _resolve_source(conn, args.source, config)

    res = scorecard.score_open_picks(conn, source=source, tax_rate=config.tax_rate)
    print(f"scored {res['checked']} open pick(s): {res['target']} hit target, "
          f"{res['stop']} stopped, {res['expired']} expired, "
          f"{res['still_open']} still running\n")

    s = scorecard.summary(conn)
    if not s.get("graded"):
        print(f"{s.get('total', 0)} pick(s) recorded, {s.get('closed', 0)} resolved, "
              f"none tradeable enough to judge yet.")
        print("Run `futmarket picks --save` daily, and `collect-bulk` regularly "
              "so there are prices to score against.")
        return
    pnl = s["coins_pnl"]
    print("TRACK RECORD  (real, tradeable cards only — net of tax)")
    print(f"  picks recorded    {s['total']}  ({s['open']} still open)")
    print(f"  graded            {s['graded']}   target {s['hit_target']} · "
          f"stop {s['hit_stop']} · flat {s['expired']}")
    print(f"  win rate          {s['win_rate']:.0%}")
    print(f"  return on capital {s['return_on_capital_pct']:+.1f}%   "
          f"← the number that matters (weights by coins at stake)")
    print(f"  total P&L         {pnl:+,} coins   "
          f"({s['avg_coins_per_trade']:+,}/trade)")
    print(f"  median trade      {s['median_return_pct']:+.1f}%")

    if args.list:
        print("\nrecent picks:")
        for r in db.picks_log(conn, limit=args.list):
            meta = db.card_meta_get(conn, r["player_id"])
            name = (meta["name"] if meta else r["player_id"])[:22]
            got = f"{r['realized_pct']:+.1f}%" if r["realized_pct"] is not None else "--"
            print(f"  {r['picked_at'][:10]}  {name:<22} {r['status']:<8} "
                  f"entry {r['entry_price']:>9,}  {got:>8}")


def cmd_picks(args) -> None:
    from .ml import picks as picks_mod
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    source = _resolve_source(conn, args.source, config)
    try:
        found = picks_mod.generate(
            conn, source=source, tax_rate=config.tax_rate,
            min_confidence=args.min_confidence, limit=args.limit,
            min_price=args.min_price, max_price=config.max_price,
            entry_z_max=config.entry_z_max,
            entry_floor_max_pct=config.entry_floor_max_pct,
            stop_buffer_pct=config.stop_buffer_pct, stop_min_pct=config.stop_min_pct,
            stop_max_pct=config.stop_max_pct, min_reward_risk=config.min_reward_risk,
            min_sales_per_hour=args.min_sales_per_hour)
    except RuntimeError as e:
        sys.exit(str(e))

    if not found:
        print("no cards clear the confidence bar today — sit out")
        return

    print(f"\n{len(found)} candidate(s), most confident first\n")
    for i, p in enumerate(found, 1):
        band = (f"{p.buy_low:,}-{p.buy_high:,}" if p.buy_low
                else f"~{p.price_now:,} (no sale data yet)")
        rating = f"{p.rating}" if p.rating else "--"
        print(f"{i:>2}. {p.name}  ({rating} {p.version[:28]})")
        print(f"    BUY   {band}          confidence {p.confidence:.0%}")
        print(f"    SELL  ~{p.sell_target:,}      stop {p.stop:,}   "
              f"(reward:risk {p.reward_risk:g})")
        liq = f"tier {p.liquidity_tier}" if p.liquidity_tier else "?"
        if p.sales_per_hour:
            liq += f", {p.sales_per_hour:.0f} sales/hr"
        print(f"    {liq}")
        for r in p.reasons:
            print(f"      - {r}")
        if p.url:
            print(f"    {p.url}")
        print()

    if args.save:
        n = picks_mod.save(conn, found)
        print(f"recorded {n} picks for later scoring")


def _print_card_read(row, read) -> None:
    import pandas as pd
    tag = f"{int(row['rating'])} {row.get('version') or ''}".strip() if pd.notna(
        row.get("rating")) else (row.get("version") or "")
    print(f"■ {row['name']}  ({tag})   now {read['price']:,}")
    print(f"  {read['verdict']}: {read['headline']}")
    if read["verdict"] == "BUY":
        print(f"     sell ~{read['target']:,}   stop {read['stop']:,}   "
              f"(reward:risk {read['reward_risk']:g})")
    for r in read["reasons"]:
        print(f"       - {r}")
    if row.get("url"):
        print(f"     {row['url']}")
    print()


def cmd_advise(args) -> None:
    """Consult the model about a specific card or a whole group."""
    import pandas as pd
    from .ml import advise
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    source = _resolve_source(conn, args.source, config)
    try:
        frame = advise.get_scored(conn, source=source, rebuild=args.refresh)
    except RuntimeError as e:
        sys.exit(str(e))
    if frame.empty:
        sys.exit("no data yet — run `futmarket backfill-history` and `train` first")

    # group mode: --rating / --league / --position / --nation
    for dim, val in (("rating", args.rating), ("league", args.league),
                     ("position", args.position), ("nation", args.nation)):
        if val is not None:
            r = advise.cohort_read(frame, dim=dim, value=val, tax_rate=config.tax_rate)
            if "error" in r:
                sys.exit(r["error"])
            move = (f"{r['group_move_7d']:+.0f}% this week"
                    if r["group_move_7d"] is not None else "flat/unknown")
            print(f"\n{dim} {val}: {r['n']} cards · group {move} · "
                  f"{r['share_on_dip']:.0f}% are on a dip right now")
            if r["opportunities"]:
                print("\nbest buys in the group right now:")
                for o in r["opportunities"]:
                    print(f"  • {o['name']} ({o.get('version') or ''}) "
                          f"@ {o['price']:,} — {o['confidence']:.0%} confident"
                          + (f"  {o['url']}" if o.get("url") else ""))
            else:
                print("\nnone of them are on a buyable dip right now — sit tight.")
            return

    if not args.query:
        sys.exit("name a card (e.g. `futmarket advise mbappe`) or a group "
                 "(e.g. `futmarket advise --rating 84`)")
    matches = advise.find_cards(frame, args.query, version=args.version)
    if matches.empty:
        sys.exit(f"no card matching {args.query!r} — try the fut.gg spelling")
    print(f"\nAsked about {args.query!r} — {len(matches)} card(s):\n")
    for _, row in matches.iterrows():
        _print_card_read(row, advise.card_read(row, tax_rate=config.tax_rate))


def cmd_sale_stats(args) -> None:
    from .services import sales
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    if args.show:
        rows = db.sale_stats_list(conn, limit=args.show)
        if not rows:
            sys.exit("no sale stats yet — run `futmarket sale-stats` first")
        print(f"{'card':<26}{'buy band':>20}{'median':>10}{'vs listed':>11}{'sales/hr':>10}")
        print("-" * 77)
        for r in rows:
            meta = db.card_meta_get(conn, r["player_id"])
            name = (meta["name"] if meta else r["player_id"])[:24]
            band = f"{r['sold_p25']:,}-{r['sold_median']:,}"
            print(f"{name:<26}{band:>20}{r['sold_median']:>10,}"
                  f"{r['sold_vs_listed'] or 0:>10.2f}x{r['sales_per_hour']:>10.1f}")
        return

    def progress(res):
        print(f"  ...{res['cards']} cards ({res['stored']} stored, {res['failed']} failed)")

    print(f"fetching real completed-sale prices"
          f"{' (limit %d)' % args.limit if args.limit else ''}...")
    res = sales.refresh_sale_stats(conn, limit=args.limit, delay=args.delay,
                                   progress=progress)
    print(f"sale stats: {res['cards']} cards fetched, {res['stored']} stored, "
          f"{res['thin']} too few sales, {res['failed']} failed")


def cmd_train(args) -> None:
    from .ml import train as train_mod
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    source = _resolve_source(conn, args.source, config)
    horizon = args.horizon or config.horizon_days
    print(f"training on source={source!r}, horizon={horizon}d "
          f"(walk-forward, {args.splits} folds)...")
    res = train_mod.train(conn, source=source, horizon=horizon,
                          stop_buffer_pct=config.stop_buffer_pct,
                          stop_min_pct=config.stop_min_pct,
                          stop_max_pct=config.stop_max_pct, n_splits=args.splits)
    if "error" in res:
        sys.exit(res["error"])

    print(f"\ndata: {res['rows']:,} rows, {res['cards']:,} cards\n")

    f = res["forecast"]
    print("FORECASTER (price move + confidence band)")
    if f.get("folds"):
        verdict = "BEATS baseline" if f["beat_baseline"] else "does NOT beat baseline"
        print(f"  folds              {f['folds']}")
        print(f"  MAE                {f['mae']}%  (baseline {f['baseline_mae']}%)")
        print(f"  skill vs baseline  {f['skill_vs_baseline_pct']}%   -> {verdict}")
        print(f"  band coverage      {f['band_coverage_p10_p90']:.1%} (target ~80%)")
    else:
        print(f"  {f.get('note', 'not evaluated')}")

    d = res["direction"]
    print("\nDIRECTION (will a trade hit target before stop?)")
    if d.get("folds"):
        verdict = "BEATS base rate" if d["beat_baseline"] else "does NOT beat base rate"
        print(f"  folds              {d['folds']}")
        print(f"  avg precision      {d['avg_precision']}  (base rate {d['base_rate']})")
        print(f"  lift vs base rate  {d['lift_vs_base_rate']}x   -> {verdict}")
        print(f"  precision @top 10% {d['precision_at_top_decile']:.1%}")
    else:
        print(f"  {d.get('note', 'not evaluated')}")

    if res["runs"]:
        print("\nregistered runs:")
        for kind, info in res["runs"].items():
            print(f"  {kind:<9} run_id={info['run_id']}  {info['artifact']}")


def cmd_build_dataset(args) -> None:
    from .ml import dataset
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    source = _resolve_source(conn, args.source, config)
    print(f"assembling the feature matrix from source={source!r}...")
    frame = dataset.build_dataset(conn, source=source)
    if frame.empty:
        sys.exit("no data — run `futmarket backfill-history` first")

    present = [c for c in dataset.FEATURE_COLUMNS if c in frame.columns]
    print(f"dataset: {len(frame):,} rows x {frame.shape[1]} cols, "
          f"{frame['player_id'].nunique():,} cards, "
          f"{frame['date'].min()} .. {frame['date'].max()}")
    print(f"features: {len(present)}")
    coverage = frame[present].notna().mean().sort_values()
    print("\nleast-populated features (coverage):")
    for name, frac in coverage.head(6).items():
        print(f"  {name:<24} {frac:6.1%}")
    if args.out:
        frame.to_parquet(args.out) if args.out.endswith(".parquet") else \
            frame.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


def cmd_build_calendar(args) -> None:
    from .services import calendar
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    if db.card_count(conn) == 0:
        sys.exit("card_meta is empty — run `futmarket build-registry` first")
    parts = ["promos/TOTW from card releases"]
    if not args.no_sbc:
        parts.append("SBCs from fut.gg")
    if not args.no_news:
        parts.append("announcements from EA news")
    print(f"building the game calendar ({', '.join(parts)})...")
    res = calendar.build_calendar(conn, include_sbc=not args.no_sbc,
                                  include_news=not args.no_news)
    print(f"calendar: launch {res['launch']}, {res['promo']} promos, "
          f"{res['totw']} TOTW drops, {res['sbc']} SBCs, {res['news']} EA articles")


def cmd_events(args) -> None:
    config = _load(args)
    conn = db.connect(config.database_path)
    rows = db.events_list(conn, event_type=args.type, limit=args.limit)
    if not rows:
        print("no events — run `futmarket build-calendar`")
        return
    for r in rows:
        window = r["start_date"] + (f" -> {r['end_date']}" if r["end_date"] else "")
        print(f"{r['event_type']:<6} {window:<26} {(r['notes'] or '')[:52]}")
    print(f"\n{len(rows)} event(s)")


def cmd_backfill_history(args) -> None:
    from .services import backfill
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    if db.card_count(conn) == 0:
        sys.exit("card_meta is empty — run `futmarket build-registry` first")
    tiers = tuple(t.strip().upper() for t in args.tiers.split(",")) if args.tiers else None

    def progress(res):
        print(f"  ...{res['cards']} cards, {res['inserted']:,} snapshots "
              f"({res['failed']} failed)")

    scope = f"tiers {','.join(tiers)}" if tiers else (
        "oldest-first" if args.order == "oldest" else "liquid-first")
    print(f"backfilling daily history ({scope}"
          f"{', limit %d' % args.limit if args.limit else ''})...")
    res = backfill.backfill_history(conn, tiers=tiers, limit=args.limit,
                                    delay=args.delay, order=args.order,
                                    max_consecutive_failures=args.max_failures)
    print(f"backfill: {res['cards']} cards, {res['inserted']:,} new snapshots "
          f"from {res['points']:,} points ({res['failed']} failed, {res['skipped']} skipped)")


def cmd_score_liquidity(args) -> None:
    from .services import liquidity
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    if db.card_count(conn) == 0:
        sys.exit("card_meta is empty — run `futmarket build-registry` first")
    source = args.source  # None = across all sources
    res = liquidity.refresh_liquidity(conn, source=source, window_days=args.window_days)
    print(f"liquidity scored: A={res['A']} B={res['B']} C={res['C']}  "
          f"(measured={res['measured']}, provisional={res['provisional']})")


def cmd_dashboard(args) -> None:
    from .webapp import create_app
    config = _load(args)  # validate config early
    setup_logging(config.log_path)
    try:
        import uvicorn
    except ImportError:
        sys.exit("dashboard needs extra deps — install with:  pip install -e '.[web]'")

    # Refuse to expose the dashboard on the network unless asked to, so it can't
    # be bound publicly by accident.
    if args.host not in _LOOPBACK and not args.allow_remote:
        sys.exit(
            f"refusing to bind {args.host} (reachable from the network).\n"
            "To expose it deliberately: set a fixed key (export FUTMARKET_KEY=…) and\n"
            "re-run with --allow-remote."
        )

    # Fail loudly if the port is already taken (usually a dashboard you forgot to
    # stop) — otherwise you'd silently keep hitting the old, stale server.
    import socket
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((args.host if args.host not in ("", "localhost") else "127.0.0.1",
                    args.port))
    except OSError:
        sys.exit(
            f"port {args.port} is already in use — most likely a dashboard you "
            f"didn't fully stop.\n"
            f"  find it:  lsof -iTCP:{args.port} -sTCP:LISTEN -n -P\n"
            f"  stop it:  kill $(lsof -tiTCP:{args.port} -sTCP:LISTEN)\n"
            f"  or run on another port:  futmarket dashboard --port {args.port + 1}"
        )
    finally:
        probe.close()

    app = create_app(args.config, args.source)
    key = app.state.access_key
    print(f"\n  FUT Market Desk  →  http://{args.host}:{args.port}/ml")
    print(f"  Access key: {key}")
    if app.state.key_generated:
        print("  (generated for this run — set FUTMARKET_KEY to keep it stable)")
    if args.host not in _LOOPBACK:
        print("  ⚠ bound to the network — anyone who reaches this URL needs the key above")
    print("  (Ctrl-C to stop)\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="futmarket", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",
                        default=os.environ.get("FUTMARKET_CONFIG", str(Path("config.yaml"))),
                        help="path to config.yaml (default: $FUTMARKET_CONFIG or ./config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-registry",
                       help="crawl fut.gg's full card list into the card_meta registry")
    p.add_argument("--game", default="26", help="fut.gg game id (default: 26 = EA FC 26)")
    p.add_argument("--max-pages", type=int, default=None,
                   help="cap pages crawled (default: all; ~30 cards/page)")
    p.add_argument("--delay", type=float, default=0.5,
                   help="seconds between page fetches (politeness)")
    p.set_defaults(func=cmd_build_registry)

    sub.add_parser("collect-bulk",
                   help="snapshot the whole market's prices in one pass (fut.gg bulk CDN)"
                   ).set_defaults(func=cmd_collect_bulk)

    p = sub.add_parser("picks", help="what to buy right now, at what price, and why")
    p.add_argument("--source", default=None)
    p.add_argument("--limit", type=int, default=20, help="how many candidates to show")
    p.add_argument("--min-confidence", type=float, default=0.30,
                   help="ignore cards the model is less sure about than this")
    p.add_argument("--min-price", type=int, default=1000,
                   help="ignore near-discard cards (their %% moves are noise "
                        "and the coins aren't worth trading)")
    p.add_argument("--min-sales-per-hour", type=float, default=3.0,
                   help="only cards that genuinely sell this often -- confidence "
                        "is worthless if you cannot get out. 0 disables the check, "
                        "which lets in dead cards the model likes purely because "
                        "%% moves are large on cheap prices")
    p.add_argument("--save", action="store_true",
                   help="record these picks so they can be scored later")
    p.set_defaults(func=cmd_picks)

    p = sub.add_parser("advise",
                       help="consult the model about a card (e.g. `advise mbappe`) "
                            "or a group (e.g. `advise --rating 84`)")
    p.add_argument("query", nargs="?", default=None, help="card name to ask about")
    p.add_argument("--version", default=None,
                   help="narrow to a card version, e.g. 'gold', 'TOTW'")
    p.add_argument("--rating", default=None, help="ask about a rating band, e.g. 84")
    p.add_argument("--league", default=None, help="ask about a league")
    p.add_argument("--position", default=None, help="ask about a position, e.g. ST")
    p.add_argument("--nation", default=None, help="ask about a nation")
    p.add_argument("--source", default=None)
    p.add_argument("--refresh", action="store_true",
                   help="rebuild the scored cache instead of using it")
    p.set_defaults(func=cmd_advise)

    p = sub.add_parser("x-login",
                       help="capture an X session (burner account) for the reader")
    p.add_argument("--session-file", default=".x_session.json")
    p.add_argument("--from-cookies", action="store_true",
                   help="paste auth_token/ct0 from a tab you're already logged into "
                        "(use this if the login window is blocked)")
    p.add_argument("--timeout", type=int, default=300,
                   help="seconds to wait for you to finish logging in")
    p.set_defaults(func=cmd_x_login)

    p = sub.add_parser("social",
                       help="collect Reddit/YouTube chatter and match it to cards")
    p.add_argument("--no-reddit", action="store_true")
    p.add_argument("--no-youtube", action="store_true")
    p.add_argument("--x", action="store_true",
                   help="also read the X creator list (needs `futmarket x-login`)")
    p.add_argument("--show", type=int, default=None, metavar="N",
                   help="show the N most-talked-about cards instead of collecting")
    p.set_defaults(func=cmd_social)

    p = sub.add_parser("insights",
                       help="recompute the cached market rhythms the dashboard shows")
    p.add_argument("--source", default=None)
    p.set_defaults(func=cmd_insights)

    p = sub.add_parser("scorecard",
                       help="score past picks and show the real track record")
    p.add_argument("--source", default=None)
    p.add_argument("--list", type=int, default=0, metavar="N",
                   help="also list the N most recent picks")
    p.set_defaults(func=cmd_scorecard)

    p = sub.add_parser("notify",
                       help="post a short run summary to Discord (needs DISCORD_WEBHOOK_URL)")
    p.set_defaults(func=cmd_notify)

    p = sub.add_parser("sale-stats",
                       help="fetch what cards REALLY sold for (true price band + trade rate)")
    p.add_argument("--limit", type=int, default=None, help="cap cards fetched")
    p.add_argument("--delay", type=float, default=1.5, help="seconds between cards")
    p.add_argument("--show", type=int, default=None, metavar="N",
                   help="just display the top N stored bands instead of fetching")
    p.set_defaults(func=cmd_sale_stats)

    p = sub.add_parser("train",
                       help="train + walk-forward validate the forecast and direction models")
    p.add_argument("--source", default=None)
    p.add_argument("--horizon", type=int, default=None,
                   help="label horizon in days (default: config horizon_days)")
    p.add_argument("--splits", type=int, default=4, help="walk-forward folds")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("build-dataset",
                       help="assemble the ML feature matrix (card + cohort + lifecycle)")
    p.add_argument("--source", default=None)
    p.add_argument("--out", default=None,
                   help="write the matrix to a .csv or .parquet file")
    p.set_defaults(func=cmd_build_dataset)

    p = sub.add_parser("build-calendar",
                       help="derive the promo/TOTW/SBC calendar (lifecycle backbone)")
    p.add_argument("--no-sbc", action="store_true", help="skip the fut.gg SBC feed")
    p.add_argument("--no-news", action="store_true", help="skip the EA news feed")
    p.set_defaults(func=cmd_build_calendar)

    p = sub.add_parser("events", help="list the market event calendar")
    p.add_argument("--type", default=None, help="PROMO | TOTW | SBC")
    p.add_argument("--limit", type=int, default=40)
    p.set_defaults(func=cmd_events)

    p = sub.add_parser("backfill-history",
                       help="backfill daily price history since release, liquid-first")
    p.add_argument("--tiers", default=None,
                   help="comma-separated liquidity tiers to include, e.g. A,B (default: all)")
    p.add_argument("--limit", type=int, default=None, help="cap cards backfilled")
    p.add_argument("--delay", type=float, default=1.0,
                   help="seconds between cards (politeness)")
    p.add_argument("--max-failures", type=int, default=5,
                   help="stop after this many consecutive failures (raise it for "
                        "long bulk runs that may hit extended rate-limit cooldowns)")
    p.add_argument("--order", choices=["liquidity", "oldest"], default="liquidity",
                   help="'oldest' prioritises earliest-released cards (longest "
                        "histories) — use it to de-skew a recency-heavy training set")
    p.set_defaults(func=cmd_backfill_history)

    p = sub.add_parser("score-liquidity",
                       help="score card_meta into A/B/C tradeability tiers (rule #1)")
    p.add_argument("--source", default=None,
                   help="price source for activity (default: across all sources)")
    p.add_argument("--window-days", type=int, default=14,
                   help="trailing window for the activity measure")
    p.set_defaults(func=cmd_score_liquidity)

    p = sub.add_parser("dashboard", aliases=["serve"],
                       help="launch the live web dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--source", default=None)
    p.add_argument("--allow-remote", action="store_true",
                   help="permit binding a non-localhost host (needs a fixed FUTMARKET_KEY)")
    p.set_defaults(func=cmd_dashboard)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
