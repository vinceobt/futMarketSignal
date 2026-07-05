import re
import json
import logging
from datetime import datetime, timezone
from patchright.sync_api import sync_playwright

from ..config import WatchlistEntry
from .base import PriceQuote
from .base import SourceError

logger = logging.getLogger(__name__)

# Card segment of a fut.gg URL, e.g. ".../26-184788461/" -> "26-184788461".
_CARD_SEG_RE = re.compile(r"/(\d+-\d+)/?$")
# Rating before "OVR", e.g. "... Glory Hunters 97 OVR" -> 97.
_RATING_RE = re.compile(r"\b(\d{1,3})\s+OVR\b", re.IGNORECASE)
# Open Graph card descriptor on a fut.gg card page, e.g.
#   "Aurélien Tchouaméni Team of the Season 95 OVR CDM card in EA FC 26"
_OG_ALT_RE = re.compile(r'<meta property="og:image:alt" content="([^"]+)"')
_OG_DESC_RE = re.compile(r'<meta property="og:description" content="([^"]+)"')


class TurnstileMockSource:
    """
    A source that uses patchright to bypass Cloudflare Turnstile explicitly
    and fetch real data from fut.gg.
    """
    name = "turnstile_mock"

    def resolve_futgg_url(self, player) -> str:
        # The user pastes the full fut.gg URL into the watchlist; use it directly.
        return player.url

    def _rating_and_version(self, label: str, player_name: str) -> dict:
        """From a card label like "Aurélien Tchouaméni Team of the Season 95 OVR",
        pull the rating and the version (the label with the player-name words and
        the trailing rating removed). Word-count stripping is accent-safe."""
        meta: dict = {}
        rating_m = _RATING_RE.search(label)
        if not rating_m:
            return meta
        meta["rating"] = int(rating_m.group(1))
        head = label[: rating_m.start()].strip()
        name_words = len(player_name.split()) if player_name else 0
        version = " ".join(head.split()[name_words:]).strip()
        meta["version"] = version or None
        return meta

    def _extract_metadata(self, html: str, target_url: str, player_name: str) -> dict:
        """Best-effort rating/version for the *specific* card in the pasted URL.

        Primary source is the page's Open Graph descriptor (present on every card
        page and specific to that card). Falls back to an ld+json ItemList on
        page templates that carry one. Any failure returns {} — enrichment is a
        nice-to-have, never load-bearing.
        """
        try:
            # Primary: Open Graph. e.g. og:image:alt =
            #   "Aurélien Tchouaméni Team of the Season 95 OVR CDM card in EA FC 26"
            og = _OG_ALT_RE.search(html) or _OG_DESC_RE.search(html)
            if og:
                meta = self._rating_and_version(og.group(1), player_name)
                if meta:
                    return meta

            # Fallback: ld+json ItemList keyed by the card id in the URL.
            card_m = _CARD_SEG_RE.search(target_url)
            card_seg = card_m.group(1) if card_m else None
            for block in re.findall(
                r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>',
                html, re.DOTALL,
            ):
                data = json.loads(block)
                if data.get("@type") != "ItemList":
                    continue
                for el in data.get("itemListElement", []):
                    item = el.get("item", {})
                    if card_seg and card_seg not in item.get("url", ""):
                        continue
                    return self._rating_and_version(item.get("name", ""), player_name)
        except Exception as e:
            logger.debug(f"metadata extract error: {e}")
        return {}

    @staticmethod
    def _parse_history(raw) -> tuple[tuple[datetime, int], ...]:
        """fut.gg's data['history'] is a list of {date, price}; turn it into
        sorted (datetime, int) points. The card's full market series lives here
        (daily since release, hourly for the recent day)."""
        points: list[tuple[datetime, int]] = []
        for row in raw or []:
            try:
                d = row["date"].replace("Z", "+00:00")
                dt = datetime.fromisoformat(d).astimezone(timezone.utc)
                price = int(row["price"])
                if price > 0:
                    points.append((dt, price))
            except (KeyError, TypeError, ValueError):
                continue
        points.sort(key=lambda p: p[0])
        return tuple(points)

    def fetch_price(self, player: WatchlistEntry, platform: str) -> PriceQuote:
        now = datetime.now(timezone.utc)
        target_url = self.resolve_futgg_url(player)

        extracted_price = None
        history: tuple[tuple[datetime, int], ...] = ()
        html = ""

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            def handle_response(response):
                nonlocal extracted_price, history
                if 'price' in response.url and 'api' in response.url:
                    logger.info(f"Checking URL: {response.url}")
                    try:
                        data = response.json()
                        if 'data' in data:
                            content = data['data']
                            if isinstance(content, list):
                                for item in content:
                                    if item.get('price'):
                                        extracted_price = item['price']
                                        logger.info(f"Intercepted fallback base price: {extracted_price} from {response.url}")
                            elif isinstance(content, dict):
                                if 'currentPrice' in content and isinstance(content['currentPrice'], dict):
                                    extracted_price = content['currentPrice'].get('price')
                                    logger.info(f"Intercepted EXACT specific price: {extracted_price} from {response.url}")
                                if content.get('history'):
                                    history = self._parse_history(content['history'])
                                    logger.info(f"Intercepted {len(history)} history points from {response.url}")
                    except Exception as e:
                        logger.debug(f"JSON extract error: {e}")

            page.on("response", handle_response)

            try:
                # Navigating directly simulates a user, solving/avoiding CF behind the scenes via Patchright
                page.goto(target_url, wait_until="load")
                page.wait_for_timeout(5000) # allow API calls to finish
                html = page.content()
            except Exception as e:
                logger.error(f"Navigation error: {e}")
            finally:
                browser.close()

        if not extracted_price:
            raise SourceError(
                f"could not extract price for {player.name} from {target_url}"
            )

        meta = self._extract_metadata(html, target_url, player.name)
        return PriceQuote(
            player_id=player.player_id,
            price=extracted_price,
            source=self.name,
            fetched_at=now,
            rating=meta.get("rating"),
            version=meta.get("version"),
            history=history,
        )
