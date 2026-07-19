"""Discovery scan: a reliable rebounder gets tracked as soon as it's found —
even when it sits above its buy-zone — so the 20-min advisor loop (not the
6-hourly scan) is what catches the actual dip to the floor."""

import math
from datetime import datetime, timedelta, timezone

from futmarket.collectors.base import PriceQuote
from futmarket.collectors.momentum_source import MomentumRow
from futmarket.services import scanner, watch


def _rebounder_history(now, days=30, per_day=4, cycles=4.25,
                       lo=40_000, hi=55_000):
    """A clean sawtooth bouncing lo→hi several times, ending mid-range (above
    the buy-zone): reliable per the strategy, but not a BUY right now."""
    n = days * per_day
    start = now - timedelta(days=days)
    pts = []
    for i in range(n):
        frac = 0.5 - 0.5 * math.cos(i / (n / cycles) * 2 * math.pi)
        pts.append((start + timedelta(hours=i * 24 / per_day),
                    int(lo + (hi - lo) * frac)))
    return tuple(pts)


class _FakeCardSource:
    name = "turnstile_mock"

    def __init__(self, now):
        self.now = now

    def fetch_price(self, player, platform):
        return PriceQuote(player_id=player.player_id, price=47_500,
                          source=self.name, fetched_at=self.now,
                          history=_rebounder_history(self.now))


class _SpyAlerter:
    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)


def test_reliable_mover_is_tracked_even_above_buy_zone(config, conn, monkeypatch):
    now = datetime.now(timezone.utc)
    mover = MomentumRow(player_id="9-test-card-26-9", name="Test Card",
                        url="https://www.fut.gg/players/9-test-card/26-9/",
                        price=47_500, momentum=12.0, rating=88,
                        position="ST", rarity="Team of the Season")
    monkeypatch.setattr(scanner.momentum_source, "fetch_momentum",
                        lambda limit: [mover])
    monkeypatch.setattr(scanner, "TurnstileMockSource",
                        lambda: _FakeCardSource(now))

    alerter = _SpyAlerter()
    res = scanner.scan(conn, config, "turnstile_mock",
                       add=True, alerter=alerter, sleep=lambda s: None)

    assert res["reliable"] == ["Test Card"]
    assert res["added"] == ["Test Card"]        # tracked NOW, not "on a dip"
    tracked = {e.player_id for e in watch.effective_entries(conn, config)}
    assert "9-test-card-26-9" in tracked
    assert len(alerter.sent) == 1 and "watching" in alerter.sent[0]


def test_dry_run_adds_nothing(config, conn, monkeypatch):
    now = datetime.now(timezone.utc)
    mover = MomentumRow(player_id="9-test-card-26-9", name="Test Card",
                        url="https://www.fut.gg/players/9-test-card/26-9/",
                        price=47_500, momentum=12.0, rating=88,
                        position="ST", rarity="TOTS")
    monkeypatch.setattr(scanner.momentum_source, "fetch_momentum",
                        lambda limit: [mover])
    monkeypatch.setattr(scanner, "TurnstileMockSource",
                        lambda: _FakeCardSource(now))

    res = scanner.scan(conn, config, "turnstile_mock",
                       add=False, alerter=None, sleep=lambda s: None)

    assert res["reliable"] == ["Test Card"]
    assert res["added"] == []
    tracked = {e.player_id for e in watch.effective_entries(conn, config)}
    assert "9-test-card-26-9" not in tracked
