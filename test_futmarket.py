import logging
logging.basicConfig(level=logging.DEBUG)

from futmarket.config import load_config
from futmarket.collectors import get_source

config = load_config('config.yaml')
source = get_source(config.source, config)
for player in config.watchlist[:1]:
    print(f"Fetching price for {player.name}...")
    quote = source.fetch_price(player, config.platform)
    print("FINAL QUOTE:", quote)
