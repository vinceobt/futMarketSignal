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
    sell_slippage_pct: float    # selling under the going rate to actually get filled
    buy_premium_pct: float      # paying over the cheapest listing to actually get filled
    horizon_days: int           # default holding period; the model picks per trade
    # The trade, from measured edge: buy a cheap/mid card on a DEEP dip and give
    # it long enough to come back. Measured month by month on tradeable cards:
    # the old shallow gate at 5 days nets -2.85%; this one at 14 days nets +4.14%
    # and beats the market in 10 of 11 months.
    min_price: int              # ignore near-discard noise below this
    max_price: int              # ignore efficiently-priced cards above this
    # NOTE: the entry gates themselves are NOT config. They live in
    # ``ml.evaluate.GATES``, which is also what the backtest measures, so what we
    # trade and what we claim to have measured cannot drift apart. A gate change
    # is a strategy change: it has to clear `futmarket evaluate` in 8+ of 11
    # months before it ships, which is not something a YAML edit should do
    # silently.
    # Per-card stop: just below the card's support, with room for noise and a cap.
    # The floor must stay outside the card's own daily jitter -- a 5% stop was
    # inside it, and 90% of trades were stopped out before they began.
    stop_buffer_pct: float
    stop_min_pct: float
    stop_max_pct: float
    min_reward_risk: float      # skip trades whose upside isn't worth the downside
    min_expected_net_pct: float  # don't trade unless the model expects this, after costs


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
        sell_slippage_pct=float(raw.get("sell_slippage_pct", 2.0)),
        buy_premium_pct=float(raw.get("buy_premium_pct", 0.0)),
        horizon_days=int(raw.get("horizon_days", 14)),
        min_price=int(raw.get("min_price", 1000)),
        max_price=int(raw.get("max_price", 40000)),
        stop_buffer_pct=float(raw.get("stop_buffer_pct", 3.0)),
        stop_min_pct=float(raw.get("stop_min_pct", 15.0)),
        stop_max_pct=float(raw.get("stop_max_pct", 30.0)),
        min_reward_risk=float(raw.get("min_reward_risk", 0.5)),
        min_expected_net_pct=float(raw.get("min_expected_net_pct", 3.0)),
    )
