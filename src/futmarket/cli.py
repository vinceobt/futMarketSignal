"""futmarket CLI.

  futmarket collect-once            one polite pass over the watchlist
  futmarket run                     polling loop (Ctrl-C to stop)
  futmarket log-price ID PRICE      manually record a price you saw in-app
  futmarket history ID [--limit N]  print a player's snapshot history
  futmarket players                 list tracked players
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import db
from .collectors import get_source
from .collectors.base import SourceError
from .config import Config, ConfigError, load_config
from .log import setup_logging
from .services import watch


def _load(args) -> Config:
    try:
        return load_config(args.config)
    except ConfigError as exc:
        sys.exit(f"config error: {exc}")


def _watchlist_entry(conn, config: Config, player_id: str):
    entries = watch.effective_entries(conn, config)
    for entry in entries:
        if entry.player_id == player_id:
            return entry
    known = ", ".join(e.player_id for e in entries)
    sys.exit(f"unknown player_id {player_id!r}. Watchlist: {known}")


def cmd_collect_once(args) -> None:
    config = _load(args)
    setup_logging(config.log_path)
    from .scheduler import run_pass
    try:
        source = get_source(config.source, config)
    except SourceError as exc:
        sys.exit(str(exc))
    conn = db.connect(config.database_path)
    result = run_pass(config, conn, source)
    print(f"collected={len(result.collected)} skipped={len(result.skipped_fresh)} "
          f"failed={len(result.failed)} breaker_tripped={result.aborted_by_breaker}")


def cmd_run(args) -> None:
    config = _load(args)
    setup_logging(config.log_path)
    from .scheduler import run_forever
    try:
        run_forever(config)
    except (KeyboardInterrupt, SystemExit):
        print("\nstopped.")


def cmd_log_price(args) -> None:
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    entry = _watchlist_entry(conn, config, args.player_id)
    if args.price <= 0:
        sys.exit("price must be a positive coin amount")
    db.upsert_player(conn, player_id=entry.player_id, name=entry.name,
                     rating=entry.rating, position=entry.position,
                     version=entry.version, platform=config.platform)
    inserted = db.insert_snapshot(conn, player_id=entry.player_id, price=args.price,
                                  source="manual", at=datetime.now(timezone.utc))
    conn.commit()
    if inserted:
        print(f"logged {entry.name} ({entry.player_id}) @ {args.price:,} coins")
    else:
        print("already have a manual snapshot for this minute — not duplicated")


def cmd_history(args) -> None:
    config = _load(args)
    conn = db.connect(config.database_path)
    rows = db.history(conn, args.player_id, args.limit)
    if not rows:
        print(f"no snapshots for {args.player_id!r} yet")
        return
    print(f"{'timestamp (UTC)':<22} {'price':>10} {'Δ%':>8}  source")
    prev_price = None
    for row in reversed(rows):  # oldest first so Δ% reads naturally
        delta = ""
        if prev_price:
            delta = f"{(row['price'] - prev_price) / prev_price * 100:+.1f}%"
        print(f"{row['timestamp']:<22} {row['price']:>10,} {delta:>8}  {row['source']}")
        prev_price = row["price"]


def cmd_players(args) -> None:
    config = _load(args)
    conn = db.connect(config.database_path)
    for entry in watch.effective_entries(conn, config):
        row = conn.execute(
            "SELECT COUNT(*) AS n, MAX(timestamp) AS last FROM price_snapshots "
            "WHERE player_id=?", (entry.player_id,)).fetchone()
        # rating/version are enriched into the players table at collection time;
        # the URL-only watchlist entry doesn't carry them.
        meta = conn.execute(
            "SELECT name, rating, version FROM players WHERE player_id=?",
            (entry.player_id,)).fetchone()
        name = (meta["name"] if meta and meta["name"] else entry.name)
        rating = (meta["rating"] if meta and meta["rating"] is not None else "-")
        version = (meta["version"] if meta and meta["version"] else entry.version)
        print(f"{entry.player_id:<40} {name:<22} {str(rating):<4} "
              f"{version:<22} snapshots={row['n']:<4} last={row['last'] or '-'}")


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
    sys.exit("no price snapshots stored yet — log or collect some first")


def _fmt(v, spec=".2f"):
    return "-" if v is None else format(v, spec)


def cmd_build_features(args) -> None:
    from . import features
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    source = _resolve_source(conn, args.source, config)
    written = features.build_and_store(conn, config, source)
    print(f"built {written} feature rows from source={source!r}")


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

    print(f"backfilling daily history"
          f"{' (tiers %s)' % ','.join(tiers) if tiers else ' (liquid-first)'}"
          f"{', limit %d' % args.limit if args.limit else ''}...")
    res = backfill.backfill_history(conn, tiers=tiers, limit=args.limit, delay=args.delay)
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


def cmd_features(args) -> None:
    from . import features
    config = _load(args)
    conn = db.connect(config.database_path)
    entry = _watchlist_entry(conn, config, args.player_id)
    source = _resolve_source(conn, args.source, config)
    table = features.compute_feature_table(conn, entry.player_id, source)
    if not table:
        print(f"no {source!r} snapshots for {entry.player_id!r} yet")
        return
    f = table[-1]  # latest
    print(f"{entry.name} ({entry.player_id})  source={source}  as of {f.timestamp}")
    print(f"  price              {f.price:>12,}")
    print(f"  pct_change_1h      {_fmt(f.pct_change_1h):>12}%")
    print(f"  pct_change_24h     {_fmt(f.pct_change_24h):>12}%")
    print(f"  pct_change_7d      {_fmt(f.pct_change_7d):>12}%")
    print(f"  rolling_mean_24h   {_fmt(f.rolling_mean_24h, ',.0f'):>12}")
    print(f"  rolling_std_24h    {_fmt(f.rolling_std_24h, ',.0f'):>12}")
    print(f"  z_score            {_fmt(f.z_score):>12}")
    ev = f"{f.days_to_next_event}d ({f.next_event_type})" if f.days_to_next_event is not None else "-"
    print(f"  days_to_next_event {ev:>12}")
    print(f"  is_weekend_window  {'yes' if f.is_weekend_window else 'no':>12}")


def cmd_watchlist(args) -> None:
    config = _load(args)
    conn = db.connect(config.database_path)
    action = args.action
    if action == "list":
        entries = watch.effective_entries(conn, config)
        if not entries:
            print("watchlist is empty — add a fut.gg URL with:  futmarket watchlist add <url>")
            return
        for e in entries:
            print(f"{e.player_id:<40} {e.name:<22} {e.url}")
        print(f"\n{len(entries)} player(s), cap {config.max_watchlist_size}")
    elif action == "add":
        try:
            added = watch.add(conn, config, args.url)
        except ConfigError as exc:
            sys.exit(f"cannot add: {exc}")
        print(f"added {added['name']} ({added['player_id']})")
    elif action == "remove":
        try:
            watch.remove(conn, config, args.player_id)
        except ConfigError as exc:
            sys.exit(str(exc))
        print(f"removed {args.player_id}")
    elif action == "import":
        try:
            text = Path(args.file).read_text()
        except OSError as exc:
            sys.exit(f"cannot read {args.file}: {exc}")
        res = watch.import_urls(conn, config, text)
        print(f"added {len(res['added'])}, skipped {len(res['skipped'])} (already tracked), "
              f"errors {len(res['errors'])}")
        for e in res["errors"]:
            print(f"  ! {e['url']}: {e['error']}")


def cmd_momentum(args) -> None:
    config = _load(args)
    conn = db.connect(config.database_path)
    if args.refresh:
        from .collectors import momentum_source
        setup_logging(config.log_path)
        limit = args.limit or config.momentum_limit
        print(f"fetching fut.gg momentum (top {limit})…")
        rows = momentum_source.fetch_momentum(limit=limit)
        db.momentum_replace(conn, rows)
        db.meta_set(conn, "momentum_updated_at",
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        conn.commit()
    cached = db.momentum_list(conn)
    if not cached:
        print("no momentum data yet — run:  futmarket momentum --refresh")
        return
    updated = db.meta_get(conn, "momentum_updated_at")
    print(f"top movers (as of {updated}):")
    for r in cached[: (args.limit or len(cached))]:
        rtg = r["rating"] if r["rating"] is not None else "-"
        print(f"  #{r['rank']:<2} mom={r['momentum']:5.1f}  {r['name'][:24]:24} "
              f"{str(rtg):>3} {r['position']:<3} {r['price']:>9,}  {r['player_id']}")


def cmd_advise(args) -> None:
    from .services import advisor
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    source = _resolve_source(conn, args.source, config)
    res = advisor.run(conn, config, source, dry_run=args.dry_run)
    tag = "would " if args.dry_run else ""
    for o in res["opened"]:
        print(f"{tag}BUY  {o['msg']}")
    for c in res["closed"]:
        print(f"{tag}SELL {c['msg']}")
    print(f"\nscreened {res['screened']} players · "
          f"{tag}opened {len(res['opened'])} · {tag}closed {len(res['closed'])} · "
          f"holding {len(res['holding'])}")
    if not res["opened"] and not res["closed"]:
        print("no BUY/SELL triggers this pass.")


def cmd_scan_momentum(args) -> None:
    from .services import scanner
    from . import alerts as alertmod
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    source = _resolve_source(conn, args.source, config)
    alerter = None if args.dry_run else alertmod.get_alerter(config)
    print(f"scanning fut.gg momentum for new range-bound cards "
          f"(up to {config.strategy_momentum_screen_limit})…")
    res = scanner.scan(conn, config, source, add=not args.dry_run, alerter=alerter)
    if res["added"]:
        print(f"now watching {len(res['added'])} for a dip to their floor: {', '.join(res['added'])}")
    waiting = [n for n in res["reliable"] if n not in res["added"]]
    if waiting:
        note = "dry-run" if args.dry_run else "not added — watchlist full or add failed"
        print(f"reliable but untracked ({note}): {', '.join(waiting)}")
    print(f"\nscanned {res['scanned']} untracked movers · "
          f"reliable {len(res['reliable'])} · added {len(res['added'])}")


def cmd_positions(args) -> None:
    config = _load(args)
    conn = db.connect(config.database_path)
    rows = db.positions_list(conn, status=args.status)
    if not rows:
        print("no positions yet.")
        return
    for r in rows:
        if r["status"] == "open":
            print(f"OPEN   {r['player_id']:<40} entry {r['entry_price']:>9,}  "
                  f"target {r['target_price']:>9,}  since {r['entry_ts']}")
        else:
            print(f"CLOSED {r['player_id']:<40} entry {r['entry_price']:>9,}  "
                  f"exit {r['exit_price'] or 0:>9,}  {r['realized_pct']:+.1f}%  {r['reason']}")


def cmd_log_event(args) -> None:
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    if args.player and args.player not in {e.player_id for e in watch.effective_entries(conn, config)}:
        sys.exit(f"unknown player_id {args.player!r}")
    event_id = db.insert_event(conn, event_type=args.type.upper(),
                               start_date=args.start, end_date=args.end,
                               player_id=args.player, notes=args.notes)
    conn.commit()
    scope = args.player or "market-wide"
    print(f"logged event #{event_id}: {args.type.upper()} {scope} starting {args.start}")


def cmd_backtest(args) -> None:
    from . import backtest, features
    from .strategy import StrategyParams
    config = _load(args)
    conn = db.connect(config.database_path)
    source = _resolve_source(conn, args.source, config)
    from dataclasses import replace
    tax = args.tax if args.tax is not None else config.tax_rate
    strat_params = StrategyParams.from_config(config)
    if args.tax is not None:
        strat_params = replace(strat_params, tax_rate=tax)

    targets = ([_watchlist_entry(conn, config, args.player_id)] if args.player_id
               else list(watch.effective_entries(conn, config)))
    any_output = False
    for entry in targets:
        table = features.compute_feature_table(conn, entry.player_id, source)
        if len(table) < 2:
            continue
        any_output = True
        series = features.to_series(db.snapshots(conn, entry.player_id, source))
        cmp = backtest.evaluate_rebound(table, series, strat_params)
        print(f"\n{entry.name} ({entry.player_id})  source={source}  "
              f"snapshots={len(table)}  tax={tax:.0%}")
        header = f"  {'strategy':<16}{'trades':>7}{'hit%':>7}{'avg/trade':>11}{'total':>9}{'maxDD':>8}"
        print(header)
        for r in [cmp.candidate, *cmp.baselines]:
            print(f"  {r.label:<16}{r.n_trades:>7}{r.hit_rate*100:>6.0f}%"
                  f"{r.avg_trade_return*100:>10.1f}%{r.total_return*100:>8.1f}%"
                  f"{r.max_drawdown*100:>7.1f}%")
        if cmp.candidate.n_trades == 0:
            verdict = "NO TRADES — strategy found no qualifying setups (stayed flat)"
        elif cmp.promote(config.backtest_max_dd_multiple):
            verdict = "PROMOTE — beats every baseline (return AND drawdown)"
        elif cmp.candidate.total_return <= 0:
            verdict = ("DO NOT PROMOTE — lost money overall "
                       "(not trading this card beats every strategy)")
        elif cmp.promote(float("inf")):
            verdict = ("DO NOT PROMOTE — beats returns but its drawdown is worse "
                       "than the baselines'")
        else:
            verdict = "DO NOT PROMOTE — fails to beat a baseline"
        print(f"  -> {verdict}")
    if not any_output:
        print(f"not enough history in source={source!r} to backtest "
              f"(need >=2 snapshots per player)")


def cmd_signals(args) -> None:
    from .services import analytics
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    source = _resolve_source(conn, args.source, config)
    res = analytics.run_signals(conn, config, source)
    if not res["signals"]:
        print(f"no price history available in source={source!r}")
        return
    for s in res["signals"]:
        print(f"[{s['type']:<4}] {s['name']:<22} {s['confidence']:>5.0%}  {s['reason']}")
    print(f"\n{res['buys']} BUY · {res['sells']} SELL · {res['holds']} HOLD · "
          f"{res['skips']} SKIP")


def cmd_alert(args) -> None:
    from datetime import datetime, timezone
    from . import alerts as alertmod
    from .services import analytics
    from .signals import BUY, SELL
    config = _load(args)
    setup_logging(config.log_path)
    conn = db.connect(config.database_path)
    source = _resolve_source(conn, args.source, config)
    try:
        alerter = alertmod.get_alerter(config)
    except ValueError as exc:
        sys.exit(str(exc))

    built = []
    for entry in watch.effective_entries(conn, config):
        decision = analytics.evaluate_player(conn, config, source, entry.player_id)
        if decision is None:
            continue
        built.append(alertmod.Alert(entry.player_id, entry.name, decision.action,
                                    decision.confidence, decision.detail))

    if args.digest:
        date_str = datetime.now(timezone.utc).date().isoformat()
        alerter.send(alertmod.format_digest(built, date_str))
        return
    # realtime: only strong, actionable signals
    strong = [a for a in built if a.signal_type in (BUY, SELL)
              and a.confidence >= config.min_confidence_alert]
    if not strong:
        print(f"no signals at/above confidence {config.min_confidence_alert:.0%}")
        return
    for a in strong:
        alerter.send(alertmod.format_realtime(a))


_LOOPBACK = {"127.0.0.1", "::1", "localhost", ""}


def cmd_dashboard(args) -> None:
    from .webapp import create_app
    config = _load(args)  # validate config early
    setup_logging(config.log_path)
    try:
        import uvicorn
    except ImportError:
        sys.exit("dashboard needs extra deps — install with:  pip install -e '.[web]'")

    # Refuse to expose the control surface on the network unless asked to, so it
    # can't be bound publicly by accident.
    if args.host not in _LOOPBACK and not args.allow_remote:
        sys.exit(
            f"refusing to bind {args.host} (reachable from the network).\n"
            "The dashboard can trigger scrapes/deletes, so it stays local by default.\n"
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
    print(f"\n  FUT Market Desk  →  http://{args.host}:{args.port}")
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

    sub.add_parser("collect-once").set_defaults(func=cmd_collect_once)
    sub.add_parser("run").set_defaults(func=cmd_run)

    p = sub.add_parser("log-price")
    p.add_argument("player_id")
    p.add_argument("price", type=int)
    p.set_defaults(func=cmd_log_price)

    p = sub.add_parser("history")
    p.add_argument("player_id")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_history)

    sub.add_parser("players").set_defaults(func=cmd_players)

    p = sub.add_parser("momentum", help="show fut.gg market movers (--refresh to fetch)")
    p.add_argument("--limit", type=int, default=None, help="how many movers to show")
    p.add_argument("--refresh", action="store_true", help="fetch fresh data from fut.gg")
    p.set_defaults(func=cmd_momentum)

    p = sub.add_parser("watchlist", help="manage tracked player links (DB-backed)")
    wsub = p.add_subparsers(dest="action", required=True)
    wsub.add_parser("list", help="show tracked players")
    wa = wsub.add_parser("add", help="track a fut.gg player URL")
    wa.add_argument("url")
    wr = wsub.add_parser("remove", help="stop tracking a player by id")
    wr.add_argument("player_id")
    wi = wsub.add_parser("import", help="bulk-add URLs from a file (one per line)")
    wi.add_argument("file")
    p.set_defaults(func=cmd_watchlist)

    p = sub.add_parser("features", help="show the latest computed features for a player")
    p.add_argument("player_id")
    p.add_argument("--source", default=None, help="price source to use (default: most-populated)")
    p.set_defaults(func=cmd_features)

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
    p.set_defaults(func=cmd_backfill_history)

    p = sub.add_parser("score-liquidity",
                       help="score card_meta into A/B/C tradeability tiers (rule #1)")
    p.add_argument("--source", default=None,
                   help="price source for activity (default: across all sources)")
    p.add_argument("--window-days", type=int, default=14,
                   help="trailing window for the activity measure")
    p.set_defaults(func=cmd_score_liquidity)

    p = sub.add_parser("build-features", help="compute + persist the features table")
    p.add_argument("--source", default=None, help="price source to use (default: most-populated)")
    p.set_defaults(func=cmd_build_features)

    p = sub.add_parser("log-event", help="record a market-moving event (TOTW/SBC/PROMO/...)")
    p.add_argument("--type", required=True, help="TOTW | SBC | PROMO | PATCH | OTHER")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--player", default=None, help="player_id, or omit for market-wide")
    p.add_argument("--notes", default=None)
    p.set_defaults(func=cmd_log_event)

    p = sub.add_parser("backtest", help="replay the decision engine vs naive baselines")
    p.add_argument("player_id", nargs="?", default=None, help="one player, or omit for all")
    p.add_argument("--source", default=None)
    p.add_argument("--tax", type=float, default=None, help="sell tax (default: config tax_rate)")
    p.add_argument("--strategy", choices=["rebound"], default="rebound",
                   help="kept for muscle memory; the unified engine is the only strategy")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("advise", help="run the rebound advisor: analyze, manage positions, alert")
    p.add_argument("--source", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="print intended BUY/SELL without writing positions or sending alerts")
    p.set_defaults(func=cmd_advise)

    p = sub.add_parser("positions", help="list paper positions opened by the advisor")
    p.add_argument("--status", choices=["open", "closed"], default=None)
    p.set_defaults(func=cmd_positions)

    p = sub.add_parser("scan-momentum", help="discover + track new range-bound cards from fut.gg momentum")
    p.add_argument("--source", default=None)
    p.add_argument("--dry-run", action="store_true", help="report finds without adding them")
    p.set_defaults(func=cmd_scan_momentum)

    p = sub.add_parser("signals", help="evaluate + store current BUY/SELL/HOLD signals")
    p.add_argument("--source", default=None)
    p.set_defaults(func=cmd_signals)

    p = sub.add_parser("alert", help="deliver signals (console or Discord)")
    p.add_argument("--source", default=None)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--digest", action="store_true", help="one summary of all players")
    mode.add_argument("--realtime", action="store_true", help="only strong BUY/SELL (default)")
    p.set_defaults(func=cmd_alert)

    p = sub.add_parser("dashboard", aliases=["serve"],
                       help="launch the interactive web dashboard (API + worker)")
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
