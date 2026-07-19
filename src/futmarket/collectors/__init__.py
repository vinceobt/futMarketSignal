from .base import PriceQuote, PriceSource, SourceError
from .futnext_source import FutNextSource
from .mock import MockSource
from .turnstile_source import TurnstileMockSource


def get_source(name: str, config) -> PriceSource:
    """Resolve the configured automated source. 'manual' has no poller — prices
    enter via `futmarket log-price`, so it is rejected here on purpose."""
    if name == "mock":
        return MockSource()
    if name == "turnstile_mock":
        return TurnstileMockSource()
    if name == "futnext":
        return FutNextSource()
    if name == "manual":
        raise SourceError(
            "source is 'manual': there is nothing to poll. "
            "Log prices with `futmarket log-price <player_id> <price>` "
            "or switch `source:` in config.yaml."
        )
    raise SourceError(f"unknown source: {name!r}")
