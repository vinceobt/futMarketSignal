"""Alert delivery: posting a message to a Discord incoming webhook.

Kept small — the run summary in `services/notify.py` composes the text and this
just delivers it. Sending publishes externally, so it only runs when the user has
configured a webhook.
"""

from __future__ import annotations

import logging

log = logging.getLogger("futmarket.alerts")


def _coins(n) -> str:
    n = float(n)
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{round(n / 1e3)}K"
    return str(round(n))


def _card_tag(rating: int | None, version: str | None) -> str:
    """Card-identity suffix so you never buy the wrong version — e.g.
    " · 94 Team of the Season". Empty when neither field is known."""
    bits = [str(rating)] if rating else []
    if version:
        bits.append(version)
    return f" · {' '.join(bits)}" if bits else ""


class DiscordAlerter:
    """Posts to a Discord incoming-webhook URL. Sending publishes externally, so
    this is only used when the user has explicitly configured a webhook."""
    name = "discord"

    def __init__(self, webhook_url: str):
        if not webhook_url:
            raise ValueError("DiscordAlerter needs a webhook_url")
        self.webhook_url = webhook_url

    def send(self, text: str) -> None:
        import httpx
        # Discord hard-caps message content at 2000 chars.
        resp = httpx.post(self.webhook_url, json={"content": text[:1990]}, timeout=10)
        resp.raise_for_status()
        log.info("alert delivered via discord status=%s", resp.status_code)
