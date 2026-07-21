"""Load and validate config.yaml. All runtime knobs come from here, not code."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

VALID_SOURCES = {"futgg"}
VALID_PLATFORMS = {"console", "pc"}

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


class ConfigError(ValueError):
    pass


def parse_player_url(url: str) -> tuple[str, str]:
    """Derive a stable (player_id, display_name) from a fut.gg player URL.

    player_id is the flattened path slug (unique per card version); the display
    name is title-cased from the name portion of the base slug.
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
    source: str                 # which price series to learn from (fut.gg bulk)
    platform: str               # console | pc
    database_path: Path
    log_path: Path
    tax_rate: float             # EA transfer-market sell tax
    target_pct: float           # profit target for the triple-barrier labels
    stop_pct: float             # stop-loss for the triple-barrier labels


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}

    source = raw.get("source", "futgg")
    if source not in VALID_SOURCES:
        raise ConfigError(f"source must be one of {sorted(VALID_SOURCES)}, got {source!r}")

    platform = raw.get("platform", "console")
    if platform not in VALID_PLATFORMS:
        raise ConfigError(f"platform must be one of {sorted(VALID_PLATFORMS)}, got {platform!r}")

    base = path.parent
    return Config(
        source=source,
        platform=platform,
        database_path=base / raw.get("database_path", "data/market.db"),
        log_path=base / raw.get("log_path", "data/collector.log"),
        tax_rate=float(raw.get("tax_rate", 0.05)),
        target_pct=float(raw.get("target_pct", 25.0)),
        stop_pct=float(raw.get("stop_pct", 8.0)),
    )
