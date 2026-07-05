"""Alert delivery. Console by default; Discord via an incoming webhook.

Two modes, per the brief:
  - realtime: push individual strong signals as they appear
  - digest:   one daily summary of the current signal picture

Telegram would be a third Alerter with the same 3-method shape (a bot-token
POST to sendMessage) — not wired until someone wants it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger("futmarket.alerts")


@dataclass(frozen=True)
class Alert:
    player_id: str
    name: str
    signal_type: str
    confidence: float
    reason: str


def format_realtime(a: Alert) -> str:
    return (f"[{a.signal_type}] {a.name} — {a.reason} "
            f"(confidence {a.confidence:.0%})")


def _coins(n) -> str:
    n = float(n)
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{round(n / 1e3)}K"
    return str(round(n))


def format_trade_alert(kind: str, name: str, price: int, *, view=None,
                       target: int | None = None, realized_pct: float | None = None,
                       reason: str = "") -> str:
    """One-line BUY/SELL message for the rebound advisor (Discord/console)."""
    if kind == "BUY":
        floor = f"~{_coins(view.floor)}" if view and view.floor else "its floor"
        tgt = f" → target {_coins(target)}" if target else ""
        return (f"🟢 BUY {name} @ {_coins(price)} — bounced off {floor} "
                f"{getattr(view, 'bounces', '?')}× in-range{tgt}")
    # SELL
    pnl = f" ({realized_pct:+.1f}% net)" if realized_pct is not None else ""
    return f"🔴 SELL {name} @ {_coins(price)}{pnl} — {reason}"


def format_digest(alerts: list[Alert], date_str: str) -> str:
    if not alerts:
        return f"FUT digest {date_str}: nothing actionable — all HOLD."
    lines = [f"FUT digest {date_str}:"]
    for a in sorted(alerts, key=lambda x: (-x.confidence, x.name)):
        lines.append(f"  • [{a.signal_type}] {a.name} ({a.confidence:.0%}): {a.reason}")
    return "\n".join(lines)


class Alerter(Protocol):
    def send(self, text: str) -> None: ...


class ConsoleAlerter:
    name = "console"

    def send(self, text: str) -> None:
        print(text)
        log.info("alert delivered via console")


class DiscordAlerter:
    """Posts to a Discord incoming-webhook URL. Sending publishes externally,
    so this is only used when the user has explicitly configured a webhook."""
    name = "discord"

    def __init__(self, webhook_url: str):
        if not webhook_url:
            raise ValueError("alert_destination is 'discord' but no webhook_url is set")
        self.webhook_url = webhook_url

    def send(self, text: str) -> None:
        import httpx
        # Discord hard-caps message content at 2000 chars.
        resp = httpx.post(self.webhook_url, json={"content": text[:1990]}, timeout=10)
        resp.raise_for_status()
        log.info("alert delivered via discord status=%s", resp.status_code)


def get_alerter(config) -> Alerter:
    if config.alert_destination == "discord":
        return DiscordAlerter(config.webhook_url)
    return ConsoleAlerter()
