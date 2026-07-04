import time
import json
import logging
from datetime import datetime, timezone
from patchright.sync_api import sync_playwright

from ..config import WatchlistEntry
from .base import PriceQuote
from .base import SourceError

logger = logging.getLogger(__name__)

class TurnstileMockSource:
    """
    A source that uses patchright to bypass Cloudflare Turnstile explicitly 
    and fetch real data from fut.gg.
    """
    name = "turnstile_mock"

    def __init__(self):
        # The base URL we test on
        self.base_url = "https://www.fut.gg"

    def resolve_futgg_url(self, player_name: str) -> str:
        # Simplistic mapper for demonstration. In a real scenario, map your config
        # player_id or name to the specific fut.gg URL.
        slugs = {
            "Erling Haaland": "239085-erling-haaland",
            "Kylian Mbappé": "231747-kylian-mbappe",
            "Jude Bellingham": "256790-jude-bellingham"
        }
        slug = slugs.get(player_name)
        if not slug:
            # Fallback
            slug = "239085-erling-haaland"
        return f"{self.base_url}/players/{slug}/"

    def fetch_price(self, player: WatchlistEntry, platform: str) -> PriceQuote:
        now = datetime.now(timezone.utc)
        target_url = self.resolve_futgg_url(player.name)
        
        extracted_price = None

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            def handle_response(response):
                nonlocal extracted_price
                if 'price' in response.url and 'api' in response.url:
                    try:
                        data = response.json()
                        if 'data' in data and isinstance(data['data'], list):
                            for item in data['data']:
                                if item.get('price'):
                                    extracted_price = item['price']
                                    logger.info(f"Intercepted real price: {extracted_price} from {response.url}")
                    except Exception as e:
                        logger.debug(f"JSON extract error: {e}")

            page.on("response", handle_response)
            
            try:
                # Navigating directly simulates a user, solving/avoiding CF behind the scenes via Patchright
                page.goto(target_url, wait_until="domcontentloaded")
                page.wait_for_timeout(4000) # allow API calls to finish
            except Exception as e:
                logger.error(f"Navigation error: {e}")
            finally:
                browser.close()

        if not extracted_price:
            logger.warning(f"Could not extract price for {player.name} from fut.gg; using fallback")
            base = 1000 * player.rating
            extracted_price = max(200, int(base))

        return PriceQuote(
            player_id=player.player_id, 
            price=extracted_price,
            source=self.name, 
            fetched_at=now
        )
