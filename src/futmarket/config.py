"""Load and validate config.yaml. All runtime knobs come from here, not code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_SOURCES = {"mock", "manual", "turnstile_mock"}
VALID_PLATFORMS = {"console", "pc"}
MIN_POLL_MINUTES = 30


@dataclass(frozen=True)
class WatchlistEntry:
    player_id: str
    name: str
    rating: int
    position: str
    version: str


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


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}

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
    entries = []
    seen_ids: set[str] = set()
    for item in raw.get("watchlist", []):
        entry = WatchlistEntry(
            player_id=str(item["player_id"]),
            name=str(item["name"]),
            rating=int(item["rating"]),
            position=str(item["position"]),
            version=str(item.get("version", "Base Gold")),
        )
        if entry.player_id in seen_ids:
            raise ConfigError(f"duplicate player_id in watchlist: {entry.player_id}")
        seen_ids.add(entry.player_id)
        entries.append(entry)
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
    )
