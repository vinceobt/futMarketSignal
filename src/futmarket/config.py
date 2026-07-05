"""Load and validate config.yaml. All runtime knobs come from here, not code."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_SOURCES = {"mock", "manual", "turnstile_mock"}
VALID_PLATFORMS = {"console", "pc"}
MIN_POLL_MINUTES = 30

# Matches a fut.gg player URL and captures the base player slug and the optional
# card segment, e.g.
#   https://www.fut.gg/players/239085-erling-haaland/26-184788461/
#     -> ("239085-erling-haaland", "26-184788461")
#   https://www.fut.gg/players/231747-kylian-mbappe/
#     -> ("231747-kylian-mbappe", None)
_PLAYER_URL_RE = re.compile(
    r"fut\.gg/players/(\d+-[a-z0-9-]+?)(?:/(\d+-\d+))?/?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WatchlistEntry:
    player_id: str
    name: str
    url: str
    rating: int | None = None
    position: str = ""
    version: str = ""


def parse_player_url(url: str) -> tuple[str, str]:
    """Derive a stable (player_id, display_name) from a fut.gg player URL.

    player_id is the flattened path slug (unique per card version); the display
    name is title-cased from the name portion of the base slug. rating/position/
    version are not derivable offline and are enriched later at fetch time.
    """
    m = _PLAYER_URL_RE.search(url.strip())
    if not m:
        raise ConfigError(f"not a fut.gg player URL: {url!r}")
    base_slug, card_seg = m.group(1), m.group(2)
    player_id = f"{base_slug}-{card_seg}" if card_seg else base_slug
    # base_slug is "{baseId}-{name-slug}"; drop the leading numeric id for the name.
    name_slug = base_slug.split("-", 1)[1] if "-" in base_slug else base_slug
    name = name_slug.replace("-", " ").title()
    return player_id, name


@dataclass(frozen=True)
class Config:
    source: str
    platform: str
    poll_minutes: int
    inter_player_delay_seconds: tuple[float, float]
    skip_guard_fraction: float
    max_consecutive_failures: int
    cooldown_minutes: int
    user_agent: str
    max_watchlist_size: int
    database_path: Path
    log_path: Path
    watchlist: tuple[WatchlistEntry, ...] = field(default_factory=tuple)
    # Signal rule thresholds (shared by the backtester and the live engine).
    buy_z: float = -1.5
    sell_z: float = 1.5
    event_guard_days: int = 2
    momentum_guard_pct: float = 0.0  # block a BUY still falling / hold a SELL still rising (24h %); 0 = off (opt-in, backtest first)
    tax_rate: float = 0.05  # EA transfer-market tax on sales
    # Alerts
    min_confidence_alert: float = 0.6  # realtime pushes only at/above this
    alert_destination: str = "console"  # console | discord
    webhook_url: str | None = None
    # Momentum scanner: how many top movers to pull from fut.gg per refresh.
    momentum_limit: int = 30
    # Rebound strategy (the autonomous advisor).
    strategy_window_days: int = 30
    strategy_min_bounces: int = 3
    strategy_buy_zone_pct: float = 3.0
    strategy_sell_mode: str = "target"
    strategy_target_pct: float = 25.0
    strategy_resistance_pctile: float = 80.0
    strategy_stop_pct: float = 8.0
    strategy_floor_drift_pct: float = 10.0
    strategy_momentum_screen_limit: int = 8  # movers to scrape+screen per cycle; 0 = watchlist only


class ConfigError(ValueError):
    pass


VALID_SELL_MODES = {"target", "resistance"}


def _valid_sell_mode(v) -> str:
    mode = str(v)
    if mode not in VALID_SELL_MODES:
        raise ConfigError(f"strategy.sell_mode must be one of {sorted(VALID_SELL_MODES)}, got {mode!r}")
    return mode


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    signals_cfg = raw.get("signals") or {}
    alerts_cfg = raw.get("alerts") or {}
    strategy_cfg = raw.get("strategy") or {}

    source = raw.get("source", "manual")
    if source not in VALID_SOURCES:
        raise ConfigError(f"source must be one of {sorted(VALID_SOURCES)}, got {source!r}")

    platform = raw.get("platform", "console")
    if platform not in VALID_PLATFORMS:
        raise ConfigError(f"platform must be one of {sorted(VALID_PLATFORMS)}, got {platform!r}")

    poll_minutes = int(raw.get("poll_minutes", 45))
    if poll_minutes < MIN_POLL_MINUTES:
        raise ConfigError(f"poll_minutes must be >= {MIN_POLL_MINUTES} (politeness floor)")

    delay = raw.get("inter_player_delay_seconds", [3, 6])
    if not (isinstance(delay, list) and len(delay) == 2 and delay[0] <= delay[1]):
        raise ConfigError("inter_player_delay_seconds must be [min, max]")

    max_size = int(raw.get("max_watchlist_size", 50))
    # Be forgiving about how the URLs are pasted: a proper YAML list is ideal,
    # but a bare block of whitespace/newline-separated URLs (missing the "- "
    # markers) is folded by YAML into one string — split it back into URLs.
    raw_watch = raw.get("watchlist") or []
    tokens: list[str] = []
    if isinstance(raw_watch, str):
        tokens = raw_watch.split()
    else:
        for item in raw_watch:
            if not isinstance(item, str):
                raise ConfigError(
                    f"watchlist entries must be fut.gg player URLs (strings), got {item!r}"
                )
            tokens.extend(item.split())

    entries = []
    seen_ids: set[str] = set()
    for url in tokens:
        player_id, name = parse_player_url(url)
        if player_id in seen_ids:
            raise ConfigError(f"duplicate player in watchlist: {player_id}")
        seen_ids.add(player_id)
        entries.append(WatchlistEntry(player_id=player_id, name=name, url=url))
    if len(entries) > max_size:
        raise ConfigError(f"watchlist has {len(entries)} players, cap is {max_size}")

    base = path.parent
    return Config(
        source=source,
        platform=platform,
        poll_minutes=poll_minutes,
        inter_player_delay_seconds=(float(delay[0]), float(delay[1])),
        skip_guard_fraction=float(raw.get("skip_guard_fraction", 0.5)),
        max_consecutive_failures=int(raw.get("max_consecutive_failures", 3)),
        cooldown_minutes=int(raw.get("cooldown_minutes", 60)),
        user_agent=str(raw.get("user_agent", "fc-market-analytics/0.1")),
        max_watchlist_size=max_size,
        database_path=base / raw.get("database_path", "data/market.db"),
        log_path=base / raw.get("log_path", "data/collector.log"),
        watchlist=tuple(entries),
        buy_z=float(signals_cfg.get("buy_z", -1.5)),
        sell_z=float(signals_cfg.get("sell_z", 1.5)),
        event_guard_days=int(signals_cfg.get("event_guard_days", 2)),
        momentum_guard_pct=float(signals_cfg.get("momentum_guard_pct", 0.0)),
        tax_rate=float(signals_cfg.get("tax_rate", 0.05)),
        min_confidence_alert=float(alerts_cfg.get("min_confidence", 0.6)),
        alert_destination=str(alerts_cfg.get("destination", "console")),
        webhook_url=alerts_cfg.get("webhook_url") or None,
        momentum_limit=int(raw.get("momentum_limit", 30)),
        strategy_window_days=int(strategy_cfg.get("window_days", 30)),
        strategy_min_bounces=int(strategy_cfg.get("min_bounces", 3)),
        strategy_buy_zone_pct=float(strategy_cfg.get("buy_zone_pct", 3.0)),
        strategy_sell_mode=_valid_sell_mode(strategy_cfg.get("sell_mode", "target")),
        strategy_target_pct=float(strategy_cfg.get("target_pct", 25.0)),
        strategy_resistance_pctile=float(strategy_cfg.get("resistance_pctile", 80.0)),
        strategy_stop_pct=float(strategy_cfg.get("stop_pct", 8.0)),
        strategy_floor_drift_pct=float(strategy_cfg.get("floor_drift_pct", 10.0)),
        strategy_momentum_screen_limit=int(strategy_cfg.get("momentum_screen_limit", 8)),
    )
