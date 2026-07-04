"""futmarket CLI.

  futmarket collect-once            one polite pass over the watchlist
  futmarket run                     polling loop (Ctrl-C to stop)
  futmarket log-price ID PRICE      manually record a price you saw in-app
  futmarket history ID [--limit N]  print a player's snapshot history
  futmarket players                 list tracked players
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import db
from .collectors import get_source
from .collectors.base import SourceError
from .config import Config, ConfigError, load_config
from .log import setup_logging


def _load(args) -> Config:
    try:
        return load_config(args.config)
    except ConfigError as exc:
        sys.exit(f"config error: {exc}")


def _watchlist_entry(config: Config, player_id: str):
    for entry in config.watchlist:
        if entry.player_id == player_id:
            return entry
    known = ", ".join(e.player_id for e in config.watchlist)
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
    entry = _watchlist_entry(config, args.player_id)
    if args.price <= 0:
        sys.exit("price must be a positive coin amount")
    conn = db.connect(config.database_path)
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
    for entry in config.watchlist:
        row = conn.execute(
            "SELECT COUNT(*) AS n, MAX(timestamp) AS last FROM price_snapshots "
            "WHERE player_id=?", (entry.player_id,)).fetchone()
        print(f"{entry.player_id:<20} {entry.name:<22} {entry.rating} {entry.position:<4} "
              f"{entry.version:<20} snapshots={row['n']:<4} last={row['last'] or '-'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="futmarket", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path("config.yaml")),
                        help="path to config.yaml (default: ./config.yaml)")
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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
